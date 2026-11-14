"""Mark the status of dead jobs as stopped (a login-node command; read-only ``squeue``).

    python3 -m hyak.tools.reconcile [--stage STAGE ...] [--dry-run]

A job that is killed hard (the time limit or ``scancel`` followed by SIGKILL, a node failure) never runs
its exit trap, so its stage keeps saying ``running`` with an active worker row and a frozen ETA until
another job of the stage starts.  This command looks at every stage whose ``status.json`` is ``running``
or ``waiting``, collects the Slurm job ids of its worker rows (``status.json``, ``workers/*.json`` and the
last writer's ``job_id``), asks ``squeue`` and rewrites the status as ``stopped`` (no workers, an error
text naming the jobs and the last write) **only when every one of them is definitely finished**.  It
leaves a stage alone when

* any of its jobs is alive or its liveness is unknown (squeue failing is never read as "finished");
* a job of the user named ``anytok-<stage>`` is pending, running or in another unfinished state (it is
  about to correct the status itself, or has just started and not written its status yet);
* the status file changed while it was being examined, or is not running any more.

Nothing else is touched: progress and every extra field are carried over, and the command never
submits, cancels or signals a job.  It is not part of the dashboard.  Run it by hand, or from
``watch -n 300``, on a node where ``squeue`` works.  Python >= 3.9, standard library only.

Output: one JSON line per examined stage.  Exit status: 0 done (including nothing to do), 2 usage or
environment error (``squeue`` cannot be asked while a stage is marked running).
"""
from __future__ import annotations

import argparse
import json
import sys

from hyak.common import slurm
from hyak.common.io import read_json, utc_now
from hyak.common.paths import STAGES, default_monitor_dir, validate_stage
from hyak.common.status import RUNNING_STATES, StatusWriter, _carry, _meta, _platform


def _job_ids(status, directory):
    """Slurm job ids named by a status, its worker rows and the worker files of its stage."""
    jobs = set()
    job_id = str(status.get('job_id') or '')
    if job_id:
        jobs.add(job_id)
    for row in status.get('workers') or []:
        if isinstance(row, dict):
            jobs.add(str(row.get('job_id') or str(row.get('id', '')).split('.')[0]))
    try:
        files = sorted((directory / 'workers').glob('*.json'))
    except OSError:
        files = []
    for path in files:
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get('job_id'):
            jobs.add(str(data['job_id']))
    return sorted(job for job in jobs if job and job != 'None' and not job.startswith('local-'))


def _alive(job, table):
    """True / False / None for one job: the listing first, then a single-job query."""
    if table is not None and job in table:
        verdict = slurm.state_alive(table[job][0])
        if verdict is not None:
            return verdict
    return slurm.is_job_alive(job)


def reconcile(stage, *, dry_run=False, table=None, directory=None):
    """Examine one stage.  Returns a dict with ``stage`` and ``action``, or None when it is not running.

    ``table`` is ``slurm.job_table()`` (None: squeue cannot be asked).
    """
    stage = validate_stage(stage)
    directory = directory or default_monitor_dir(stage)
    path = directory / 'status.json'
    try:
        status = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(status, dict) or status.get('state') not in RUNNING_STATES:
        return None
    jobs = _job_ids(status, directory)
    if table is None:
        return {'stage': stage, 'action': 'left alone', 'reason': 'squeue could not be asked', 'jobs': jobs}
    if not jobs:
        return {'stage': stage, 'action': 'left alone', 'reason': 'the status names no Slurm job', 'jobs': jobs}
    verdicts = {job: _alive(job, table) for job in jobs}
    if any(verdict is not False for verdict in verdicts.values()):
        return {'stage': stage, 'action': 'left alone', 'reason': 'a job is alive or its state is unknown',
                'jobs': verdicts}
    name = f'anytok-{stage}'
    pending = sorted(job for job, (state, job_name) in table.items()
                     if job_name == name and slurm.state_alive(state) is not False)
    if pending:
        return {'stage': stage, 'action': 'left alone', 'reason': f'job(s) {", ".join(pending)} named {name} '
                'will correct the status themselves', 'jobs': verdicts}
    field, total, unit = _meta(stage, status)
    reason = (f'job(s) {", ".join(jobs)} ended without recording their end (time limit, scancel or node '
              f'failure); last status write {status.get("at_utc")}; reconciled {utc_now()}')
    if not dry_run:
        latest = read_json(path, {})  # the status must not have moved since it was judged
        # the whole object, not only at_utc: two writes can fall into the same millisecond
        if not isinstance(latest, dict) or latest != status or latest.get('state') not in RUNNING_STATES:
            return {'stage': stage, 'action': 'left alone', 'reason': 'the status changed meanwhile', 'jobs': verdicts}
        writer = StatusWriter(stage, total, unit, field, _platform(stage, status, None), directory,
                              job='reconcile')
        for job in jobs:
            writer.remove_job_rows(job)
        writer.update(status.get(field, 0) or 0, stage='stopped', state='stopped', workers=[], error=reason,
                      elapsed_seconds=status.get('elapsed_seconds'), event='reconciled',
                      **_carry(status, field))
    return {'stage': stage, 'action': 'would mark stopped' if dry_run else 'marked stopped', 'jobs': verdicts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python3 -m hyak.tools.reconcile', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', action='append', choices=STAGES, metavar='STAGE',
                        help='stage to examine (repeatable; default: every stage)')
    parser.add_argument('--dry-run', action='store_true', help='say what would be marked; write nothing')
    args = parser.parse_args(argv)
    stages = args.stage or list(STAGES)
    # squeue is asked once, and only when a stage is running (a quiet cluster costs no call).
    running = []
    for stage in stages:
        try:
            status = read_json(default_monitor_dir(stage) / 'status.json')
        except (OSError, ValueError):
            continue
        if isinstance(status, dict) and status.get('state') in RUNNING_STATES:
            running.append(stage)
    if not running:
        print(json.dumps({'action': 'nothing to do', 'reason': 'no stage is running or waiting'}))
        return 0
    table = slurm.job_table(max_age=0)
    if table is None:
        print('ERROR: squeue could not be asked, so no status was changed. Remedy: run this on the login node '
              'where squeue works (module load slurm if needed).', file=sys.stderr)
        return 2
    for stage in running:
        result = reconcile(stage, dry_run=args.dry_run, table=table)
        if result:
            print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
