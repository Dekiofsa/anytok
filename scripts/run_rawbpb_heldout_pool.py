#!/usr/bin/env python3
"""Monitor-only observer for the raw-BPB held-out full-pool eval on Tillicum (42 checkpoints x 1,000 tokenizers).

Polls the cluster over the user's existing SSH control connection by streaming
scripts/heldout_pool_status.py to a cluster Python on stdin, and writes
<local-state>/status.json (atomic) and <local-state>/events.jsonl for the dashboard
(plans/run-dashboard.json, id rawbpb-heldout-pool, adapter progress_json).

It never submits, cancels or changes anything on the cluster and never starts an SSH
master connection (BatchMode, existing ControlPath only).

Exit codes: 0 every checkpoint complete, 2 a setup step, the smoke test or a scoring task
failed, 3 the observer lost the cluster. Python 3.9 compatible, standard library only.
"""
import argparse
import glob
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from heldout_pool_status import PHASES
from run_regret_continuation import (OPTS, SSHError, StatusError, atomic_write_json, dq, is_active,
                                     is_fail, is_pending, is_running, parse_elapsed, ssh_run, utc_now)

PLATFORM = 'Tillicum · H200'
PROFILE = 'stf'
CHECKPOINTS = 42
TOKENIZERS = 1000
TOTAL = CHECKPOINTS * TOKENIZERS
SMOKE_CHECKPOINT = 'raw-bpb-020000'
EVENT_MAX_AGE = 600.0
AMBIGUOUS_POLLS = 2
STATUS_SCRIPT = Path(__file__).resolve().parent / 'heldout_pool_status.py'


def smoke_passed(receipt):
    return (isinstance(receipt, dict) and receipt.get('smaller_vocab_bitwise_equal') is True
            and 'H200' in str(receipt.get('gpu')))


def evaluate(status):
    """Map one remote status object to {stage, detail, error, completed, exit, ambiguous}.

    Pure function of the status. 'exit' is 0 when every checkpoint is complete, 2 for a
    failure, otherwise None. 'ambiguous' marks a state reported as a failure only if it
    persists for AMBIGUOUS_POLLS polls.
    """
    out = dict(completed=0, exit=None, ambiguous=False)

    def result(stage, detail=None, error=None):
        out['stage'] = stage
        if detail:
            out['detail'] = detail
        if error:
            out['error'] = error
        out['exit'] = 0 if stage == 'complete' else (2 if stage == 'failed' else None)
        return out

    if not status.get('stage_exists'):
        return result('preparing', 'the stage folder does not exist yet')
    checkpoints = status.get('checkpoints') or {}
    out['completed'] = sum(min(int(c.get('scored') or 0), TOKENIZERS) for c in checkpoints.values())
    done = sum(1 for c in checkpoints.values() if c.get('complete'))
    if len(checkpoints) == CHECKPOINTS and done == CHECKPOINTS:
        out['completed'] = TOTAL
        return result('complete', 'all %d checkpoints scored on all %d tokenizers' % (CHECKPOINTS, TOKENIZERS))

    setup = status.get('setup') or {}
    for phase in PHASES:
        log = setup.get(phase) or {}
        if log.get('exit') not in (None, 0):
            return result('failed', error='setup phase %s stopped at "%s" (exit %s)'
                          % (phase, log.get('step'), log.get('exit')))
    failures = status.get('worker_failures') or []
    if failures:
        last = failures[-1]
        return result('failed', error='%s failed%s: %s' % (
            last.get('worker'), ' on %s' % last['checkpoint'] if last.get('checkpoint') else '', last.get('error')))
    jobs = status.get('jobs') or []
    smoke_jobs = [j for j in jobs if '_' not in str(j.get('id'))]
    tasks = [j for j in jobs if '_' in str(j.get('id'))]
    for job in smoke_jobs + tasks:
        if is_fail(job.get('state')):
            return result('failed', error='Slurm job %s %s (exit %s, elapsed %s)' % (
                job['id'], job.get('state'), job.get('exit_code'), job.get('elapsed')))

    if tasks:
        running = [j for j in tasks if is_running(j.get('state'))]
        pending = [j for j in tasks if is_pending(j.get('state'))]
        text = '%d of %d checkpoints complete; %d running, %d queued' % (done, CHECKPOINTS, len(running), len(pending))
        if running:
            return result('scoring', text)
        if pending:
            return result('queued', text)
        out['ambiguous'] = True
        return result('scoring', text, error='every scoring task ended but %d checkpoints are incomplete'
                      % (CHECKPOINTS - done))

    markers = status.get('markers') or {}
    receipt = status.get('smoke')
    inputs = setup.get('inputs') or {}
    running = ['%s: %s' % (phase, setup[phase].get('step') or 'starting') for phase in PHASES[1:]
               if setup.get(phase) and setup[phase].get('exit') is None]
    if any(is_active(j.get('state')) for j in smoke_jobs):
        return result('smoke', 'GPU smoke test %s' % ', '.join('%s %s' % (j['id'], j.get('state')) for j in smoke_jobs))
    if smoke_jobs and not smoke_passed(receipt):
        if receipt is not None:
            return result('failed', error='the smoke receipt does not show a bitwise-equal H200 check: %s' % receipt)
        out['ambiguous'] = True
        return result('smoke', 'smoke job ended without a receipt; confirming',
                      error='smoke job %s ended without hardware-check.json' % smoke_jobs[-1]['id'])
    if markers.get('preflight-passed'):
        return result('ready', 'preflight passed for every checkpoint; scoring not submitted yet')
    if running:
        return result('preparing', '; '.join(running))
    if smoke_passed(receipt):
        return result('smoke_passed', 'smoke test passed on %s' % receipt.get('gpu'))
    if markers.get('smoke-checkpoint-staged'):
        return result('preparing', 'phase 1 done; the smoke test has not been submitted')
    return result('preparing', 'phase 1: %s' % (inputs.get('step') or 'starting'))


def build_workers(status):
    labels = list((status.get('checkpoints') or {}).keys())
    checkpoints = status.get('checkpoints') or {}
    workers = []
    for job in status.get('jobs') or []:
        slurm = job.get('state') or 'UNKNOWN'
        job_id = str(job['id'])
        entry = dict(id=job_id, profile=PROFILE, platform=PLATFORM, slurm_state=slurm)
        if '_' in job_id:
            index = job_id.split('_', 1)[1]
            label = labels[int(index)] if index.isdigit() and int(index) < len(labels) else None
            entry['task'] = 'score %s' % (label or 'array task %s' % index)
            if label and is_running(slurm):
                entry['state'] = 'Tokenizer %d/%d' % (checkpoints[label].get('scored') or 0, TOKENIZERS)
        else:
            entry['task'] = 'GPU smoke test (%s)' % SMOKE_CHECKPOINT
        if 'state' not in entry:
            reason = job.get('reason') or ''
            entry['state'] = '%s%s' % (slurm, ' (%s)' % reason if reason and is_pending(slurm) else '')
        entry['elapsed_seconds'] = parse_elapsed(job.get('elapsed') or '')
        if job.get('exit_code'):
            entry['exit_code'] = job['exit_code']
        workers.append(entry)
    return workers


class Observer:
    def __init__(self, args):
        self.a = args
        self.local = Path(args.local_state)
        self.local.mkdir(parents=True, exist_ok=True)
        self.status_path = self.local / 'status.json'
        self.events_path = self.local / 'events.jsonl'
        self.ssh_failures = 0
        self.status_errors = 0
        self.ambiguous_polls = 0
        self.last_status = {}
        self.last_event_sig = None
        self.last_event_time = 0.0
        self.started_at = utc_now()
        try:
            prev = json.loads(self.status_path.read_text())
            if isinstance(prev.get('observer_started_at'), str):
                self.started_at = prev['observer_started_at']
        except (OSError, ValueError):
            pass

    def fetch_status(self):
        cmd = '%s -I - --stage %s' % (dq(self.a.python), dq(self.a.remote_stage))
        rc, out, err = ssh_run(self.a.host, cmd, stdin_text=Path(self.a.status_script).read_text())
        if rc == 255:
            raise SSHError('ssh failure: %s' % (err or out).strip()[:300])
        if rc != 0:
            raise StatusError('status command rc=%s: %s' % (rc, (err or out).strip()[:300]))
        lines = [line for line in out.splitlines() if line.strip()]
        if not lines:
            raise StatusError('status command printed nothing')
        try:
            data = json.loads(lines[-1])
        except ValueError as exc:
            raise StatusError('status command output is not JSON: %s' % exc)
        if not isinstance(data, dict):
            raise StatusError('status command output is not an object')
        return data

    def emit(self, record, now):
        record['at_utc'] = utc_now()
        record.setdefault('total', TOTAL)
        record.setdefault('unit', 'tokenizer scores')
        record.setdefault('platform', PLATFORM)
        record.setdefault('profile', PROFILE)
        record['observer_started_at'] = self.started_at
        try:
            record['elapsed_seconds'] = max(0, int((datetime.now(timezone.utc)
                                                    - datetime.fromisoformat(self.started_at)).total_seconds()))
        except ValueError:
            pass
        self.last_status = record
        atomic_write_json(self.status_path, record)
        sig = (record.get('stage'), record.get('detail'), record.get('state'),
               tuple((w['id'], w.get('slurm_state')) for w in record.get('workers', [])))
        if sig != self.last_event_sig or now - self.last_event_time >= EVENT_MAX_AGE:
            event = {'at_utc': record['at_utc'], 'stage': record.get('stage'),
                     'completed': record.get('completed'), 'total': record['total']}
            for key in ('state', 'error', 'detail'):
                if record.get(key) is not None:
                    event[key] = record[key]
            event['worker_states'] = {w['id']: w.get('slurm_state') for w in record.get('workers', [])}
            with open(str(self.events_path), 'a') as handle:
                handle.write(json.dumps(event) + '\n')
                handle.flush()
            self.last_event_sig = sig
            self.last_event_time = now
        line = {'at_utc': record['at_utc'], 'stage': record.get('stage'), 'completed': record.get('completed'),
                'detail': record.get('detail'), 'ssh_failures': self.ssh_failures}
        if record.get('error'):
            line['error'] = record['error']
        print(json.dumps(line), flush=True)

    def stop(self, why, now):
        """Record that the observer itself stopped (the Slurm jobs are not affected)."""
        record = dict(self.last_status)
        record.update(stage='stopped', state='stopped', error=why)
        record.setdefault('completed', 0)
        self.emit(record, now)

    def poll(self, now):
        """One poll. Returns an exit code, or None to continue."""
        try:
            status = self.fetch_status()
        except (SSHError, StatusError) as exc:
            if isinstance(exc, SSHError):
                self.ssh_failures += 1
            else:
                self.status_errors += 1
            record = dict(self.last_status)
            record.update(ssh_failures=self.ssh_failures, status_errors=self.status_errors, last_error=str(exc))
            if max(self.ssh_failures, self.status_errors) >= self.a.max_ssh_failures:
                self.last_status = record
                self.stop('observer lost the cluster after %d failed polls; the Slurm jobs are not affected: %s'
                          % (max(self.ssh_failures, self.status_errors), exc), now)
                return 3
            record.setdefault('stage', 'preparing')
            record.setdefault('completed', 0)
            self.emit(record, now)
            return None
        self.ssh_failures = 0
        self.status_errors = 0
        verdict = evaluate(status)
        record = {'completed': verdict['completed'], 'stage': verdict['stage'], 'workers': build_workers(status),
                  'remote_logs': ['%s/logs/setup-%s.log' % (self.a.remote_stage, phase) for phase in PHASES]
                                 + ['%s/raw-bpb/logs/' % self.a.remote_stage],
                  'ssh_failures': 0, 'markers': status.get('markers'), 'smoke': status.get('smoke'),
                  'preflight': status.get('preflight'),
                  'checkpoints_complete': sum(1 for c in (status.get('checkpoints') or {}).values() if c.get('complete'))}
        for key in ('detail', 'error'):
            if verdict.get(key):
                record[key] = verdict[key]
        if verdict['ambiguous'] and verdict['exit'] is None:
            self.ambiguous_polls += 1
            if self.ambiguous_polls < AMBIGUOUS_POLLS:
                record.pop('error', None)
                self.emit(record, now)
                return None
            record['stage'] = 'failed'
            verdict['exit'] = 2
        else:
            self.ambiguous_polls = 0
        if record['stage'] == 'failed':
            record['state'] = 'failed'
        self.emit(record, now)
        return verdict['exit']


def parse_args(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--host', required=True, help='NETID@tillicum.hyak.uw.edu')
    parser.add_argument('--control-path', default='~/.ssh/cm-tillicum-*',
                        help='SSH ControlPath socket; the default glob must match exactly one socket')
    parser.add_argument('--remote-stage', default='/gpfs/projects/stf/$USER/anytok-heldout',
                        help='remote stage folder ($USER expands on the cluster)')
    parser.add_argument('--python', default='/usr/bin/python3', help='remote python executable')
    parser.add_argument('--status-script', default=str(STATUS_SCRIPT))
    parser.add_argument('--local-state', default=str(root / 'modal_results' / 'rawbpb_heldout_pool'))
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--max-ssh-failures', type=int, default=5)
    args = parser.parse_args(argv)
    args.control_path = os.path.expanduser(args.control_path)
    if any(character in args.control_path for character in '*?['):
        sockets = [path for path in glob.glob(args.control_path) if is_socket(path)]
        if len(sockets) != 1:
            parser.error(f'--control-path glob {args.control_path} matched {len(sockets)} sockets; pass the exact path')
        args.control_path = sockets[0]
    return args


def is_socket(path):
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False


def main(argv=None):
    args = parse_args(argv)
    OPTS['control_path'] = args.control_path
    observer = Observer(args)
    try:
        while True:
            code = observer.poll(time.time())
            if code is not None:
                return code
            time.sleep(args.interval)
    except KeyboardInterrupt:
        observer.stop('observer interrupted; the Slurm jobs are not affected', time.time())
        return 130


if __name__ == '__main__':
    sys.exit(main())
