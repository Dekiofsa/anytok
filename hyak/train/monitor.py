"""Dashboard status of a training stage, derived from the trainer's own ``<run>/events.jsonl``.

    python -m hyak.train.monitor --run {uniform,rl} [--smoke] [--once] [--interval 15] [--parent-pid PID]
    python -m hyak.train.monitor --run {uniform,rl} [--smoke] --finalize --exit-code N [--log PATH]

Runs as a background process of the training sbatch script (not a thread of the trainer).  It
replaces the Modal controllers' log parsing (``scripts/run_anytok_pool_20k.py:55-74``,
``scripts/run_anytok_pool_rl_20k.py:70-85``): every ``--interval`` seconds (15, as there) it reads
new trainer lines, keeps the last record per ``(stage, step)`` (a resumed segment replays steps, so
duplicates are harmless) and writes ``<monitor>/status.json`` through
``hyak.common.status.StatusWriter`` with

* ``step``: step of the newest ``train`` record in file order (the trainer's current position),
* ``stage``: ``starting`` / ``initial_eval`` / ``training`` / ``segment_paused`` / ``auditing`` /
  ``complete`` (``paused_at`` at step 20,000 becomes ``complete`` once ``final_audit.json`` is valid),
* ``seconds_per_step``: median ``seconds`` of the last 100 train records; ``eta_seconds`` =
  ``(20000 - step) * seconds_per_step`` once 20 train records exist,
* ``elapsed_seconds``: running time summed over all jobs of the stage,
* one worker row (Slurm job id, ``Step N / 20,000`` or ``RL step N / 20,000``), the last
  ``peak_gib``, the last sanity evaluation and, for RL, the controller's policy fields
  (``train_bpb``, ``policy_entropy``, ``policy_effective_tokenizers``, ``policy_max_probability``) and the
  paired metrics of the straight-BPB-reward trainer (D8: ``policy_reward_bpb``, ``paired_canonical_bpb``,
  ``paired_gap_bpb``, ``paired_valid_examples``, ``paired_maximum_positions``).

The status object is appended to ``<monitor>/events.jsonl`` only when it changed, plus one
timestamped event per trainer milestone (``initial_eval``, ``eval``, ``checkpoint``, ``paused_at``),
reported exactly once across monitor processes and jobs (the byte offset and inode already reported
are kept in the status as ``trainer_log``).  For RL each milestone event also carries ``paired``: those
paired metrics of the newest train record before it (``train_step`` names its step).  Non-finite numbers in trainer rows (the trainers write
NaN with ``json.dumps``) become null.  In follow mode a refresh error is logged and retried at the
next interval; the monitor never exits because of it (``--once`` exits 1 with a traceback).  When a
run starts from scratch, ``run_segment`` moves the stage's earlier monitor files to
``previous-<utc>/`` and the monitor's elapsed time restarts with that job.  The smoke stage ends
``complete`` once ``run_segment --smoke`` has written ``smoke_verified.json`` into the smoke directory.
For the RL run it refreshes ``policy-monitor.json`` after each new 500-step checkpoint by running
``python -m hyak.tools.policy_monitor --once`` (output appended to ``policy-monitor.log``).

``--finalize --exit-code N`` refreshes once, then records the job's end through
``hyak.common.status.finalize``: exit 0 keeps ``complete`` or becomes ``paused``; 130/137/143 give
``stopped``; any other code gives ``failed`` with the tail of ``--log``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

from hyak.common import slurm
from hyak.common import status as status_module
from hyak.common.config import load_config
from hyak.common.io import read_json, tail_lines
from hyak.common.paths import (FINAL_STEP, RUN_RL, RUN_UNIFORM, STAGE_SMOKE_TRAIN, STAGE_TRAIN_RL,
                               STAGE_TRAIN_UNIFORM, Paths, validate_run)
from hyak.common.status import StatusWriter
from hyak.train import retention

STAGES = {RUN_UNIFORM: STAGE_TRAIN_UNIFORM, RUN_RL: STAGE_TRAIN_RL}
#: The controllers' window and threshold (run_anytok_pool_20k.py:68-70).
MEDIAN_WINDOW = 100
ETA_MIN_RECORDS = 20
DEFAULT_INTERVAL = 15.0
#: Fields of the last RL train record copied into the status (run_anytok_pool_rl_20k.py:79-82).
RL_FIELDS = ('train_bpb', 'policy_entropy', 'policy_effective_tokenizers', 'policy_max_probability',
             'policy_unique_tokenizers')
#: Paired metrics of the straight-BPB-reward RL trainer (D8), passed to the status and to every RL milestone event.
PAIRED_FIELDS = ('train_bpb', 'policy_reward_bpb', 'paired_canonical_bpb', 'paired_gap_bpb', 'paired_valid_examples',
                 'paired_maximum_positions')
MILESTONES = ('initial_eval', 'eval', 'checkpoint', 'paused_at', 'complete', 'settings_change')
SMOKE_FINAL_STEP = 3
#: Written by ``run_segment --smoke`` into the smoke directory after smoke_verified.
SMOKE_VERIFIED = 'smoke_verified.json'



class TrainerLog:
    """Incremental reader of a trainer ``events.jsonl`` (no timestamps; one JSON object per line).

    Keeps the last record per ``(stage, step)``; only small per-step facts of ``train`` records are
    retained (``seconds``), plus the newest train record, evaluations and the last event.
    """

    def __init__(self, path, reported=None):
        self.path = Path(path)
        #: ``{"inode", "offset"}`` persisted by an earlier monitor process (status extra ``trainer_log``):
        #: rows that end at or before ``offset`` of the file with that inode were reported already.
        self._restored = reported if isinstance(reported, dict) else None
        self.reported_offset = 0
        self._reset()

    def _reset(self):
        self.offset = 0
        self.identity = None
        self.partial = b''
        self.seconds = {}          # train step -> seconds
        self.last_train = None     # newest train record in file order
        self.evals = {}            # step -> results
        self.initial_eval = None
        self.last = None           # last record of any stage, in file order
        self.records = 0
        self.fresh = []            # rows of the last poll that no monitor process has reported yet
        self.reported_offset = 0   # a replaced or truncated file is reported from its start

    @property
    def consumed(self) -> int:
        """Byte offset up to which complete lines have been parsed."""
        return self.offset - len(self.partial)

    def checkpoint(self):
        """What to persist so that the next monitor process does not report the same rows again."""
        if self.identity is None:
            return None
        return {'inode': self.identity[1], 'offset': max(self.reported_offset, self.consumed)}

    def poll(self) -> list:
        """Read appended lines; returns the new records (in file order)."""
        self.fresh = []
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self.identity is not None:
                self._reset()
            return []
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            self._reset()
        if self.identity is None and self._restored is not None:
            # first sight of the file in this process: trust the persisted offset only for the same inode
            # (st_dev is not compared: it may differ between nodes of a cluster file system) and only
            # while the file is at least that long
            offset, inode = self._restored.get('offset'), self._restored.get('inode')
            if isinstance(offset, int) and inode == stat.st_ino and 0 <= offset <= stat.st_size:
                self.reported_offset = offset
            self._restored = None
        self.identity = identity
        if stat.st_size == self.offset:
            return []
        with open(self.path, 'rb') as stream:
            stream.seek(self.offset)
            data = stream.read()
        end = self.offset - len(self.partial)      # file offset where the next complete line starts
        self.offset += len(data)
        data = self.partial + data
        lines = data.split(b'\n')
        self.partial = lines.pop()
        new = []
        for raw in lines:
            end += len(raw) + 1
            try:
                row = json.loads(raw.decode('utf-8', errors='replace'))
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            self._add(row)
            new.append(row)
            if end > self.reported_offset:
                self.fresh.append(row)
        self.reported_offset = max(self.reported_offset, end)
        return new

    def _add(self, row):
        self.records += 1
        self.last = row
        stage, step = row.get('stage'), row.get('step')
        if stage == 'train' and isinstance(step, int):
            seconds = row.get('seconds')
            self.seconds[step] = seconds if isinstance(seconds, (int, float)) and math.isfinite(seconds) else None
            self.last_train = row
        elif stage == 'eval' and isinstance(step, int):
            self.evals[step] = row.get('results')
        elif stage == 'initial_eval':
            self.initial_eval = row.get('results')

    @property
    def step(self) -> int:
        return self.last_train['step'] if self.last_train else 0

    def seconds_per_step(self):
        """Median ``seconds`` of the last 100 distinct train steps up to the current step."""
        current = self.step
        values = [self.seconds[s] for s in sorted(s for s in self.seconds if s <= current)[-MEDIAN_WINDOW:]]
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    def train_records(self) -> int:
        return len(self.seconds)


def _finite(value):
    """``(copy, replaced)``: a JSON value with every non-finite float replaced by None (the pinned trainers
    write NaN/Infinity with ``json.dumps``; status and event files must stay strict JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None, True
    if isinstance(value, dict):
        out, replaced = {}, False
        for key, item in value.items():
            out[key], hit = _finite(item)
            replaced = replaced or hit
        return out, replaced
    if isinstance(value, (list, tuple)):
        out, replaced = [], False
        for item in value:
            clean, hit = _finite(item)
            out.append(clean)
            replaced = replaced or hit
        return out, replaced
    return value, False


def _summarise_results(results):
    if not isinstance(results, list):
        return None
    return [{k: r.get(k) for k in ('generated_bpb', 'canonical_bpb', 'examples', 'bytes', 'truncated_forms')}
            for r in results if isinstance(r, dict)]


class Monitor:
    """Builds and writes the status of one training stage (see the module docstring)."""

    def __init__(self, run: str, paths: Paths, cfg: dict, *, smoke: bool = False, refresh_policy=None,
                 clock=time.time):
        self.run = validate_run(run)
        self.paths = paths
        self.smoke = smoke
        self.stage = STAGE_SMOKE_TRAIN if smoke else STAGES[run]
        self.run_dir = paths.smoke_dir(run) if smoke else paths.run_dir(run)
        self.final_step = SMOKE_FINAL_STEP if smoke else FINAL_STEP
        self.monitor_dir = paths.monitor_dir(self.stage)
        self.writer = StatusWriter(self.stage, self.final_step, 'steps', 'step', cfg=cfg, monitor_dir=self.monitor_dir)
        self.clock = clock
        self.refresh_policy = refresh_policy
        self.policy_steps = set()
        self.last_key = None
        previous = self._read_status()
        self.log = TrainerLog(self.run_dir / 'events.jsonl', reported=previous.get('trainer_log'))
        self._status_seen = False
        value = previous.get('elapsed_seconds')
        value = float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
        if previous.get('elapsed_job') == self.writer.job and slurm.job_start_time() is not None:
            # Same job, clock from $SLURM_JOB_START_TIME: this process counts the whole job itself.
            self.elapsed_before = float(previous.get('elapsed_before_job_seconds') or 0)
        else:
            # Another job, or a later monitor process of this job (07_train's final --once refresh) whose
            # clock starts now: keep what was counted so far.
            self.elapsed_before = value

    def _read_status(self) -> dict:
        try:
            data = read_json(self.monitor_dir / 'status.json', {})
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _complete(self) -> bool:
        if self.smoke:
            # run_segment --smoke writes this marker after smoke_verified (the smoke directory is wiped first)
            return (self.run_dir / SMOKE_VERIFIED).is_file()
        try:
            info = retention.inspect_run_dir(self.run_dir)
        except retention.CheckpointError:
            return False
        newest = info['newest_full_step']
        return newest is not None and newest >= FINAL_STEP and retention.audit_valid(self.run_dir, self.run)

    def _stage(self, complete: bool) -> str:
        if complete:
            return 'complete'
        last = self.log.last
        if last is None:
            return 'starting'
        stage = last.get('stage')
        if stage == 'initial_eval':
            return 'initial_eval'
        if stage in ('paused_at', 'complete'):
            return 'auditing' if (not self.smoke and last.get('step') == self.final_step) else 'segment_paused'
        return 'training'

    def status_fields(self) -> dict:
        """Everything of the next status write (without writing)."""
        complete = self._complete()
        step = self.log.step
        if complete:
            step = max(step, self.final_step)
        per_step = self.log.seconds_per_step()
        eta = None
        if complete:
            eta = 0
        elif per_step is not None and self.log.train_records() >= ETA_MIN_RECORDS:
            eta = round(max(0, self.final_step - step) * per_step, 1)
        elapsed = round(self.elapsed_before + max(0.0, self.clock() - self.writer.started_at), 1)
        stage = self._stage(complete)
        last = self.log.last_train or {}
        extra = {'run': self.run, 'final_step': self.final_step, 'seconds_per_step': per_step,
                 'peak_gib': last.get('peak_gib'), 'tokenizer_vocab_size': last.get('tokenizer_vocab_size'),
                 'train_records': self.log.train_records(),
                 'trainer_last_event': (self.log.last or {}).get('stage'),
                 'elapsed_before_job_seconds': self.elapsed_before, 'elapsed_job': self.writer.job,
                 'trainer_log': self.log.checkpoint()}
        if self.log.evals:
            last_eval = max(self.log.evals)
            extra['last_eval'] = {'step': last_eval, 'results': _summarise_results(self.log.evals[last_eval])}
        if self.log.initial_eval is not None:
            extra['initial_eval'] = _summarise_results(self.log.initial_eval)
        if self.run == RUN_RL:
            extra.update({key: last.get(key) for key in RL_FIELDS + PAIRED_FIELDS})
        extra, self.non_finite = _finite(extra)
        label = 'RL step' if self.run == RUN_RL else 'Step'
        task = f'{label} {step:,} / {self.final_step:,}'
        worker_state = {'training': 'Training', 'initial_eval': 'Evaluating', 'starting': 'Starting',
                        'segment_paused': 'Between segments', 'auditing': 'Auditing checkpoints'}.get(stage, 'Active')
        return {'step': step, 'stage': stage, 'state': 'complete' if complete else 'running',
                'eta_seconds': eta, 'elapsed_seconds': elapsed, 'task': task, 'worker_state': worker_state,
                'extra': extra}

    def refresh(self, force: bool = False) -> dict:
        """Read new trainer lines, emit milestone events and rewrite status.json.

        The status object is appended to ``events.jsonl`` only when step, stage, state, the last
        evaluation or the trainer's last event changed (or ``force``).
        """
        if self._status_seen and not self.writer.status_path.exists():
            # run_segment moved this stage's status to previous-<utc>/ (a run started from scratch): the new
            # attempt's elapsed time starts with this job
            self.elapsed_before = 0.0
            self.last_key = None
            self.writer.event({'event': 'monitor_attempt_reset', 'reason': 'status.json was rotated away'})
        latest_train = self.log.last_train
        new = self.log.poll()
        fresh = {id(row) for row in self.log.fresh}
        for row in new:
            if row.get('stage') == 'train' and isinstance(row.get('step'), int):
                latest_train = row
            if id(row) not in fresh or row.get('stage') not in MILESTONES:
                continue
            fields = {k: v for k, v in row.items() if k != 'results'}
            if 'results' in row:
                fields['results'] = _summarise_results(row['results'])
            if self.run == RUN_RL and latest_train is not None:
                fields['paired'] = {'train_step': latest_train.get('step'),
                                    **{key: latest_train.get(key) for key in PAIRED_FIELDS}}
            fields, replaced = _finite(fields)
            if replaced:
                fields['non_finite'] = True
            try:
                self.writer.event({'event': f'trainer_{row.get("stage")}', **fields})
            except ValueError as exc:   # one odd row must not cost the rest of the poll
                self.writer.event({'event': f'trainer_{row.get("stage")}', 'step': fields.get('step')
                                   if isinstance(fields.get('step'), int) else None, 'unserialisable': str(exc)[-300:]})
        self._maybe_refresh_policy(new, force)
        fields = self.status_fields()
        key = (fields['step'], fields['stage'], fields['state'], fields['extra'].get('last_eval', {}).get('step'),
               (self.log.last or {}).get('stage'))
        changed = key != self.last_key
        workers = [] if fields['state'] == 'complete' else [
            self.writer.worker_row(task=fields['task'], state=fields['worker_state'], eta_seconds=fields['eta_seconds'],
                                   peak_gib=fields['extra']['peak_gib'],
                                   seconds_per_step=fields['extra']['seconds_per_step'])]
        result = self.writer.update(fields['step'], stage=fields['stage'], state=fields['state'],
                                    eta_seconds=fields['eta_seconds'], elapsed_seconds=fields['elapsed_seconds'],
                                    workers=workers, append=changed or force, **fields['extra'])
        self.last_key = key
        self._status_seen = True
        return result

    def _maybe_refresh_policy(self, new, force):
        if self.run != RUN_RL or self.smoke or self.refresh_policy is None:
            return
        due = {row['step'] for row in new if row.get('stage') == 'checkpoint' and isinstance(row.get('step'), int)
               and row['step'] > 0 and row['step'] % retention.CHECKPOINT_INTERVAL == 0}
        if (due - self.policy_steps) or (force and not self.policy_steps):
            try:
                self.refresh_policy()
            except Exception as exc:  # noqa: BLE001 -- the policy panel must never stop status updates
                self.writer.event({'event': 'policy_monitor_failed', 'error': str(exc)[-500:]})
            self.policy_steps |= due or {0}


def policy_refresher(paths: Paths, config=None, python=None, timeout=900):
    """``python -m hyak.tools.policy_monitor --once`` with output appended to ``<monitor>/policy-monitor.log``."""
    log = paths.monitor_dir(STAGE_TRAIN_RL) / 'policy-monitor.log'

    def refresh():
        command = [python or sys.executable, '-m', 'hyak.tools.policy_monitor', '--once']
        if config is not None:
            command += ['--config', str(config)]
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'ab') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        if result.returncode != 0:
            # 2 = configuration/input error, 1 = crash (traceback in the log); the caller records
            # policy_monitor_failed and keeps writing the run's status
            last = next((line for line in reversed(tail_lines(log, 20)) if line.strip()), '')
            raise RuntimeError(f'python -m hyak.tools.policy_monitor --once exited {result.returncode}: '
                               f'{last.strip()[-300:] or "(no output)"} (see {log})')

    return refresh


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _SafeRefresh:
    """Follow mode: a refresh error is logged (stderr + one ``monitor_refresh_failed`` event per new message)
    and retried at the next interval; the monitor never exits because of it."""

    def __init__(self, monitor):
        self.monitor = monitor
        self.last_message = None

    def __call__(self, force=False):
        try:
            result = self.monitor.refresh(force=force)
        except Exception as exc:  # noqa: BLE001 -- live status must survive transient storage errors and odd rows
            message = f'{type(exc).__name__}: {exc}'[-500:]
            print(f'[hyak monitor] WARNING: refresh failed (retrying every interval): {message}',
                  file=sys.stderr, flush=True)
            if message != self.last_message:
                self.last_message = message
                try:
                    self.monitor.writer.event({'event': 'monitor_refresh_failed', 'error': message})
                except Exception:  # noqa: BLE001
                    pass
            return None
        if self.last_message is not None:
            self.last_message = None
            try:
                self.monitor.writer.event({'event': 'monitor_refresh_recovered'})
            except Exception:  # noqa: BLE001
                pass
        return result


def follow(monitor: Monitor, *, interval: float = DEFAULT_INTERVAL, once: bool = False, parent_pid=None) -> None:
    """Refresh until SIGTERM/SIGINT (or the parent process exits); a last refresh on the way out.

    ``once``: one refresh whose error propagates (the CLI then exits 1).  Follow mode never exits on a
    refresh error (:class:`_SafeRefresh`).
    """
    if once:
        monitor.refresh(force=True)
        return
    stop = threading.Event()
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    refresh = _SafeRefresh(monitor)
    try:
        refresh(force=True)
        while not stop.wait(interval):
            refresh()
            if parent_pid is not None and not _alive(parent_pid):
                break
        refresh(force=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def finalize(monitor: Monitor, exit_code: int, log_path=None) -> dict:
    """One last refresh, then ``hyak.common.status.finalize`` for this job."""
    try:
        monitor.refresh(force=True)
    except Exception as exc:  # noqa: BLE001 -- the job's end must be recorded anyway
        print(f'[hyak monitor] WARNING: final refresh failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
    return status_module.finalize(monitor.stage, exit_code, log_path=log_path, monitor_dir=monitor.monitor_dir,
                                  platform=monitor.writer.platform)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.train.monitor', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, choices=(RUN_UNIFORM, RUN_RL))
    parser.add_argument('--smoke', action='store_true', help='monitor $ANYTOK_RUNS/smoke/<run> (stage smoke-train)')
    parser.add_argument('--once', action='store_true', help='refresh once and exit')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL)
    parser.add_argument('--parent-pid', type=int, help='exit when this process is gone')
    parser.add_argument('--finalize', action='store_true', help='record the end of this job')
    parser.add_argument('--exit-code', type=int, help='with --finalize: the training exit status')
    parser.add_argument('--log', type=Path, help='with --finalize: log whose tail becomes the error text')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    if args.finalize and args.exit_code is None:
        parser.error('--finalize needs --exit-code')
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
        refresher = None if args.smoke or args.run != RUN_RL else policy_refresher(paths, args.config)
        monitor = Monitor(args.run, paths, cfg, smoke=args.smoke, refresh_policy=refresher)
        if args.finalize:
            result = finalize(monitor, args.exit_code, args.log)
            print(json.dumps({'stage': result.get('stage'), 'state': result.get('state'), 'step': result.get('step')}))
            return 0
        follow(monitor, interval=args.interval, once=args.once, parent_pid=args.parent_pid)
    except Exception as exc:  # noqa: BLE001 -- unexpected: full traceback (the job only warns on it)
        traceback.print_exc(file=sys.stderr)
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
