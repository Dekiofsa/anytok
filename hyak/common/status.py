"""Dashboard status for Slurm jobs: ``status.json`` (atomic) + ``events.jsonl`` (timestamped lines).

The files live in ``<repo>/modal_results/hyak_tillicum/<stage>/`` and are read by
the dashboard's ``progress_json`` adapter (``monitoring/state.py:341-393``):

* the progress numerator under the registry's ``completed_field`` (``completed``,
  or ``step`` for training), ``total`` (used when the registry has none), ``stage``
  (badge text; ``complete`` matters for entries with ``completion_stage``),
  ``state`` (``stopped`` / ``failed`` are shown as terminal; monitoring/state.py:369),
  ``at_utc``, ``eta_seconds``, ``elapsed_seconds`` and ``workers`` rows with ``id``,
  ``profile``, ``platform``, ``task``, ``state``, ``started_at``, ``elapsed_seconds``,
  ``eta_seconds`` (monitoring/state.py:372-380).  A worker's ``log`` (absolute path of
  its Slurm output) is shown when the worker is selected (scripts/run_dashboard.py:36).

Several jobs may work on one stage at the same time (pool, eval).  Each
StatusWriter keeps its own rows in ``<stage>/workers/<job>__<host>__<pid>.json``
and every status write merges the rows of all jobs that are still alive
(one cached ``squeue`` listing of the user's jobs per process and minute, see
``hyak.common.slurm.job_table``; a job the listing does not name gets one single-job query; same-host
pid check for local runs; otherwise files fresher than ``stale_seconds``; an empty or failing squeue never
means "finished").  So every job writes the same complete worker list.  The row that ``start`` leaves
behind is dropped as soon as its process is gone.

Non-finite floats are never an error for the writers: ``null`` is written with the raw value as a string
next to it (``loss_raw: "nan"``).

Command line, for sbatch scripts (any Python >= 3.9)::

    python -m hyak.common.status start    --stage S [--task TEXT]
    python -m hyak.common.status event    --stage S --name NAME [--set key=value ...]
    python -m hyak.common.status finalize --stage S --exit-code N [--log PATH] [--message TEXT]
    python -m hyak.common.status show     --stage S

``finalize`` records the end of this job: exit 0 keeps a terminal state written
by the stage (complete/paused/stopped/failed) or, when no other job of the stage
is alive, turns a still-running state into ``paused``; 130/137/143 (killed by
INT/KILL/TERM) give ``stopped``; any other non-zero code gives ``failed`` with the
log tail as ``error`` (long lines keep their first 150 and their last 450 characters; a failure the stage
wrote itself for this job keeps its label and is prepended to the message).  Exit 0 with ``--message``
stores the message as ``note`` (job.sh passes its JOB_NOTE for 'not now' jobs); the next job's ``start``
drops it.  A ``complete`` state survives ``start`` and exit 0.  Exit status of the command line: 0, or 2 for a usage or environment error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

from hyak.common import slurm
from hyak.common.io import (append_jsonl, atomic_write_json, finite_json, json_text, read_json, tail_lines,
                            utc_now)
from hyak.common.paths import STAGE_PROGRESS, default_monitor_dir, validate_stage

RUNNING_STATES = ('running', 'waiting')
TERMINAL_STATES = ('complete', 'paused', 'stopped', 'failed')
#: Values a stage may write as ``state``.
STATES = RUNNING_STATES + TERMINAL_STATES
#: States the dashboard renders as terminal (monitoring/state.py:369).
DASHBOARD_TERMINAL_STATES = ('stopped', 'failed', 'superseded', 'budget_stop')
#: Exit codes of a child killed by a signal (128 + signal number) that mean "stopped", not "failed".
SIGNAL_EXIT_CODES = {130: 'SIGINT', 137: 'SIGKILL', 143: 'SIGTERM'}
#: Default age after which another job's worker file is ignored when its liveness cannot be checked.
DEFAULT_STALE_SECONDS = 6 * 3600
#: How long one liveness answer for another job is reused (the job listing behind it has the same lifetime).
LIVENESS_SECONDS = 60
#: finalize keeps a log line whole up to this many characters, else its first and its last part.
LINE_KEEP = 600
LINE_HEAD = 150

_RESERVED = frozenset({'at_utc', 'stage', 'state', 'unit', 'completed_field', 'hyak_stage', 'platform',
                       'eta_seconds', 'elapsed_seconds', 'workers', 'error', 'job_id', 'host'})
_FIELD_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_FILE_SAFE = re.compile(r'[^A-Za-z0-9_.-]')


def _number(value, name, allow_none=True):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number{" or None" if allow_none else ""}, got {value!r}')
    return value


def _measure(value, name):
    """A number or None for eta/elapsed values.  Non-finite floats pass (they are written as null plus a
    ``<name>_raw`` string, see :func:`hyak.common.io.finite_json`); other types raise ValueError."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number or None, got {value!r}')
    return value


def _clip(line: str) -> str:
    """A log line for the error text: whole when short, else its first part and its END (remedies come last)."""
    if len(line) <= LINE_KEEP:
        return line
    return line[:LINE_HEAD] + ' ... ' + line[-(LINE_KEEP - LINE_HEAD):]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class StatusWriter:
    """Writes one stage's ``status.json`` + ``events.jsonl`` for the ``progress_json`` adapter.

    Args:
        stage: pipeline stage name (``hyak.common.paths.STAGES``), e.g. ``'pool'``.
        total: progress denominator written as ``total`` (None while unknown).
        unit: label after the total, e.g. ``'tokenizers'``.
        completed_field: key of the numerator (must match the registry entry:
            ``'step'`` for the training stages, ``'completed'`` otherwise; see
            ``hyak.common.paths.STAGE_PROGRESS``).
        platform: worker/platform text; default ``stage_value(cfg, 'HYAK_PLATFORM', stage)``
            from ``cfg`` or ``load_config()``.
        monitor_dir: default ``<repo>/modal_results/hyak_tillicum/<stage>``
            (``$HYAK_MONITOR_ROOT`` in tests).
        cfg: config dict used only to resolve ``platform``.
        job: worker id; default ``$SLURM_JOB_ID`` or ``local-<pid>``.
        strict: when False (default), OSError while writing is reported on stderr and
            counted in ``write_errors`` instead of raised, so a transient storage error
            never kills a long job.  Invalid arguments always raise ValueError.
        stale_seconds: age after which another job's rows are dropped when its
            liveness cannot be determined.
        started_at: job start (UNIX time); default ``$SLURM_JOB_START_TIME`` or now.
    """

    def __init__(self, stage, total, unit, completed_field='completed', platform=None, monitor_dir=None, *,
                 cfg=None, job=None, strict=False, stale_seconds=DEFAULT_STALE_SECONDS, started_at=None):
        self.stage = validate_stage(stage)
        self.total = _number(total, 'total')
        if not isinstance(unit, str) or not unit:
            raise ValueError(f'unit must be a non-empty string, got {unit!r}')
        self.unit = unit
        if not _FIELD_RE.fullmatch(completed_field or '') or completed_field in _RESERVED or completed_field == 'total':
            raise ValueError(f'completed_field must be a plain, non-reserved key, got {completed_field!r}')
        self.completed_field = completed_field
        if platform is None:
            from hyak.common.config import load_config, stage_value
            platform = stage_value(cfg if cfg is not None else load_config(), 'HYAK_PLATFORM', stage)
        self.platform = platform
        self.monitor_dir = Path(monitor_dir) if monitor_dir is not None else default_monitor_dir(stage)
        self.status_path = self.monitor_dir / 'status.json'
        self.events_path = self.monitor_dir / 'events.jsonl'
        self.workers_dir = self.monitor_dir / 'workers'
        self.job = str(job) if job is not None else slurm.worker_id()
        self.host = slurm.node_name()
        self.pid = os.getpid()
        self.profile = slurm.profile_label()
        start = started_at if started_at is not None else slurm.job_start_time()
        self.started_at = float(start) if start is not None else time.time()
        self.strict = strict
        self.stale_seconds = stale_seconds
        self.write_errors = 0
        self.last = None
        self._alive = {}
        self._lock = threading.RLock()
        key = _FILE_SAFE.sub('_', f'{self.job}__{self.host}__{self.pid}')
        self.worker_file = self.workers_dir / f'{key}.json'

    # --- rows ----------------------------------------------------------------------
    def worker_row(self, task=None, state=None, *, suffix=None, eta_seconds=None, elapsed_seconds=None,
                   **extra) -> dict:
        """One dashboard worker row for this job.

        ``id`` is the job id (``<job>.<suffix>`` for several rows per job), ``profile``
        is ``<cluster>/<partition>`` or ``local``, ``platform`` this writer's
        platform, ``started_at`` the job start, ``elapsed_seconds`` time since it,
        ``log`` the job's Slurm output file when it exists.  An ``index=`` extra is stored as ``pool_index``.
        """
        row = {
            'id': self.job if suffix is None else f'{self.job}.{suffix}',
            'profile': self.profile,
            'platform': self.platform,
            'task': task if task is not None else self.stage,
            'state': state if state is not None else 'Active',
            'started_at': _iso(self.started_at),
            'elapsed_seconds': _measure(elapsed_seconds, 'elapsed_seconds') if elapsed_seconds is not None
            else round(max(0.0, time.time() - self.started_at), 1),
            'eta_seconds': _measure(eta_seconds, 'eta_seconds'),
            'host': self.host,
            'job_id': self.job,
        }
        log = self.monitor_dir / f'slurm-{self.job}.out'
        if log.is_file():
            row['log'] = str(log)
        if 'index' in extra:
            # The unchanged dashboard UI switches to its tokenizer-pool layout ('#NNNN . algorithm', vocabulary
            # column, no task text) for any row that has an ``index`` key (monitoring/web/app.js).  Rows
            # here carry a free-text task instead, so the key is renamed.
            extra['pool_index'] = extra.pop('index')
        row.update(extra)
        return row

    # --- writes ----------------------------------------------------------------------
    def update(self, completed, stage=None, state=None, eta_seconds=None, elapsed_seconds=None, workers=None,
               error=None, *, append=True, **extra) -> dict:
        """Write ``status.json`` atomically and (``append=True``) the same object to ``events.jsonl``.

        Args:
            completed: progress numerator (finite number >= 0), written under ``completed_field``.
            stage: badge text (default: ``complete``/``paused``/... for terminal states, else ``running``).
                With ``state='complete'`` it must be ``'complete'`` or None.
            state: one of ``STATES`` (default ``running``).
            eta_seconds, elapsed_seconds: finite numbers or None.
            workers: ALL rows of this process (from :meth:`worker_row`; each call replaces
                the previous set, so a process with several worker loops passes every
                loop's row each time); default one row ``worker_row(task=stage)`` while
                running and none in a terminal state.  Rows of other live jobs of the
                stage are merged in.  Thread-safe within one process.
            error: text shown in events/status (``None`` when healthy).
            extra: further JSON fields (``total=`` also updates the denominator).

        Non-finite floats anywhere except ``completed`` and ``total`` never raise: they are written as
        ``null`` with the raw value as a string next to it (``loss`` -> ``loss: null, loss_raw: "nan"``).
        The line appended to ``events.jsonl`` is the status without its ``workers`` rows (those grow with
        the number of jobs and are only useful in ``status.json``).

        Returns the written status dict.  Raises ValueError for invalid values (non-finite ``completed`` or
        ``total``, unknown state, reserved keys); OSError only when ``strict``.
        """
        completed = _number(completed, 'completed', allow_none=False)
        if completed < 0:
            raise ValueError(f'completed must be >= 0, got {completed}')
        state = 'running' if state is None else state
        if state not in STATES:
            raise ValueError(f'state must be one of {STATES}, got {state!r}')
        if stage is None:
            stage = state if state in TERMINAL_STATES else 'running'
        if state == 'complete' and stage != 'complete':
            raise ValueError("state 'complete' requires stage 'complete' (the dashboard keys completion on it)")
        if not isinstance(stage, str) or not stage:
            raise ValueError(f'stage must be a non-empty string, got {stage!r}')
        total = _number(extra.pop('total'), 'total') if 'total' in extra else self.total
        clash = sorted(set(extra) & (_RESERVED | {self.completed_field}))
        if clash:
            raise ValueError(f'reserved status keys passed as extra fields: {clash}')
        if workers is None:
            workers = [] if state in TERMINAL_STATES else [self.worker_row(task=stage)]
        for row in workers:
            if not isinstance(row, dict) or 'id' not in row:
                raise ValueError(f'worker rows must be dicts with an id, got {row!r}')
        own = [dict(row) for row in workers]
        status = {
            'at_utc': utc_now(),
            'stage': stage,
            'state': state,
            self.completed_field: completed,
            'total': total,
            'unit': self.unit,
            'completed_field': self.completed_field,
            'hyak_stage': self.stage,
            'platform': self.platform,
            'eta_seconds': _measure(eta_seconds, 'eta_seconds'),
            'elapsed_seconds': _measure(elapsed_seconds, 'elapsed_seconds'),
            'workers': [],
            'error': error,
            'job_id': self.job,
            'host': self.host,
        }
        status.update(extra)
        status = finite_json(status)
        own = finite_json(own)
        try:
            json_text(status)  # fail on unserialisable values before touching any file
            json_text(own)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'status for stage {self.stage} is not valid JSON: {exc}') from exc
        self.total = total
        with self._lock:
            try:
                self._store_own_rows(own)
            except OSError as exc:
                self._io_failure(exc)
            status['workers'] = self._merged_rows(own)
            try:
                atomic_write_json(self.status_path, status)
            except OSError as exc:
                self._io_failure(exc)
            if append:
                try:
                    append_jsonl(self.events_path, {k: v for k, v in status.items() if k != 'workers'})
                except OSError as exc:
                    self._io_failure(exc)
            self.last = status
        return status

    def event(self, obj: dict) -> dict:
        """Append ``obj`` to ``events.jsonl`` (adds ``at_utc``, ``hyak_stage``, ``job_id``, ``host`` if absent)."""
        if not isinstance(obj, dict):
            raise ValueError(f'event must be a dict, got {obj!r}')
        row = finite_json({'at_utc': utc_now(), 'hyak_stage': self.stage, 'job_id': self.job,
                           'host': self.host, **obj})
        try:
            json_text(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'event for stage {self.stage} is not valid JSON: {exc}') from exc
        try:
            append_jsonl(self.events_path, row)
        except OSError as exc:
            self._io_failure(exc)
        return row

    def remove_job_rows(self, job=None) -> int:
        """Delete the worker files of ``job`` (default: this writer's job, any pid). Returns how many."""
        job = str(job) if job is not None else self.job
        removed = 0
        for path, data in self._worker_files():
            if str(data.get('job_id')) == job:
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
        return removed

    def other_rows(self) -> list:
        """Rows of other live jobs of this stage (see the module docstring for liveness)."""
        return self._merged_rows([])

    # --- internals -------------------------------------------------------------------
    def _io_failure(self, exc):
        self.write_errors += 1
        if self.strict:
            raise exc
        print(f'[hyak status] WARNING: could not write status for stage {self.stage} '
              f'in {self.monitor_dir}: {exc}', file=sys.stderr, flush=True)

    def _store_own_rows(self, rows):
        if rows:
            atomic_write_json(self.worker_file, {'job_id': self.job, 'host': self.host, 'pid': self.pid,
                                                 'updated_at': utc_now(), 'rows': rows})
        else:
            try:
                self.worker_file.unlink()
            except FileNotFoundError:
                pass

    def _worker_files(self):
        try:
            paths = sorted(self.workers_dir.glob('*.json'))
        except OSError:
            return []
        files = []
        for path in paths:
            try:
                data = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                files.append((path, data))
        return files

    def _job_alive(self, job):
        """Whether another job is alive: True, False (kept for ever) or None (unknown).

        The answer comes from the per-process listing of the user's jobs (``slurm.job_states``: one squeue
        call per minute for every job of the stage together).  A job the listing does not name is asked
        about individually, which is rare: it is finished (and then cached as False for ever) or belongs
        to another user.  An unusable listing never counts as "finished".
        """
        now = time.time()
        cached = self._alive.get(job)
        if cached is not None and (cached[0] is False or now - cached[1] < LIVENESS_SECONDS):
            return cached[0]
        alive = None
        states = slurm.job_states()
        if states is not None and job in states:
            alive = slurm.state_alive(states[job])
        if alive is None:
            alive = slurm.is_job_alive(job)
        if alive is not None:
            self._alive[job] = (alive, now)
        return alive

    def _merged_rows(self, own):
        rows = {row['id']: row for row in own}
        for path, data in self._worker_files():
            if path == self.worker_file:
                continue
            job, host, pid = str(data.get('job_id', '')), data.get('host'), data.get('pid')
            alive = None
            if job == self.job and host == self.host and isinstance(pid, int) and pid != self.pid:
                # Another process of this very job (``status start`` leaves its 'Starting' row behind):
                # once it is gone its row is stale, whatever the state the stage writes now.
                if not _pid_alive(pid):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
            if job and not job.startswith('local-'):
                alive = self._job_alive(job)
            elif host == self.host and isinstance(pid, int):
                alive = _pid_alive(pid)
            if alive is False:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if alive is None:
                try:
                    age = time.time() - path.stat().st_mtime
                except OSError:
                    continue
                if age > self.stale_seconds:
                    continue
            for row in data.get('rows', []):
                if isinstance(row, dict) and 'id' in row and row['id'] not in rows:
                    rows[row['id']] = row
        return [rows[key] for key in sorted(rows, key=str)]


def _iso(timestamp: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _meta(stage, status):
    defaults = STAGE_PROGRESS[stage]
    field = status.get('completed_field') or defaults['completed_field']
    total = status['total'] if 'total' in status else defaults['total']
    unit = status.get('unit') or defaults['unit']
    return field, total, unit


def _platform(stage, status, platform):
    if platform:
        return platform
    if status.get('platform'):
        return status['platform']
    try:
        from hyak.common.config import load_config, stage_value
        return stage_value(load_config(), 'HYAK_PLATFORM', stage)
    except Exception:  # finalisation must work even when the config is what broke the job
        return 'Hyak'


def _carry(status, field):
    return {k: v for k, v in status.items() if k not in _RESERVED and k not in (field, 'total', 'exit_code', 'event')}


def _read_status(path):
    try:
        data = read_json(path, {})
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def start(stage, *, task=None, monitor_dir=None, platform=None, job=None) -> dict:
    """Mark this job as started: keep progress and meta from an existing status, add this job's row.

    A ``complete`` state stays complete (only the row is added); anything else becomes ``running``.
    """
    stage = validate_stage(stage)
    monitor_dir = Path(monitor_dir) if monitor_dir is not None else default_monitor_dir(stage)
    status = _read_status(monitor_dir / 'status.json')
    field, total, unit = _meta(stage, status)
    writer = StatusWriter(stage, total, unit, field, _platform(stage, status, platform), monitor_dir, job=job)
    complete = status.get('state') == 'complete'
    row = writer.worker_row(task=task or ('starting' if not complete else 'already complete'), state='Starting')
    return writer.update(status.get(field, 0) or 0, stage='complete' if complete else 'starting',
                         state='complete' if complete else 'running', workers=[row],
                         eta_seconds=None if complete else status.get('eta_seconds'),
                         elapsed_seconds=status.get('elapsed_seconds'), event='job_start',
                         **{k: v for k, v in _carry(status, field).items() if k != 'note'})   # the earlier job's note


def finalize(stage, exit_code, *, log_path=None, message=None, monitor_dir=None, platform=None, job=None) -> dict:
    """Record the end of this job in the stage status (semantics in the module docstring). Returns the status."""
    stage = validate_stage(stage)
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code < 0:
        raise ValueError(f'exit_code must be a non-negative integer, got {exit_code!r}')
    monitor_dir = Path(monitor_dir) if monitor_dir is not None else default_monitor_dir(stage)
    status = _read_status(monitor_dir / 'status.json')
    field, total, unit = _meta(stage, status)
    writer = StatusWriter(stage, total, unit, field, _platform(stage, status, platform), monitor_dir, job=job)
    writer.remove_job_rows()
    others = writer.other_rows()
    previous = status.get('state')
    error = status.get('error')
    if exit_code == 0:
        if previous in TERMINAL_STATES:
            state, label = previous, status.get('stage') or previous
        elif others:
            state, label = previous or 'running', status.get('stage') or 'running'
        else:
            state, label = 'paused', 'paused'
        if state == 'complete':
            label = 'complete'
    else:
        state = 'stopped' if exit_code in SIGNAL_EXIT_CODES else 'failed'
        label = state
        reason = message or (f'job {writer.job} was terminated by {SIGNAL_EXIT_CODES[exit_code]}'
                             if exit_code in SIGNAL_EXIT_CODES else f'job {writer.job} exited with code {exit_code}')
        lines = tail_lines(log_path, 20) if log_path else []
        if lines:
            reason += ' | log tail: ' + ' / '.join(_clip(line.strip()) for line in lines if line.strip())
        if previous == 'failed' and state == 'failed' and str(status.get('job_id')) == writer.job:
            # This job's own stage already wrote a specific failure (label and one-line verdict): keep both.
            label = status.get('stage') or label
            if error:
                reason = f'{error} | {reason}'
        from hyak.common.config import redact_text
        error = redact_text(reason)
    terminal = state in TERMINAL_STATES
    extra = _carry(status, field)
    if exit_code == 0 and message:
        # e.g. job.sh's JOB_NOTE for a 'not now' job (exit 75 of the tool, job exit 0): shown, not an error
        from hyak.common.config import redact_text
        extra['note'] = redact_text(str(message))
    return writer.update(status.get(field, 0) or 0, stage=label, state=state, workers=[],
                         eta_seconds=None if terminal else status.get('eta_seconds'),
                         elapsed_seconds=status.get('elapsed_seconds'), error=error,
                         exit_code=exit_code, event='job_end', **extra)


def _parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.common.status', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p_start = sub.add_parser('start', help='mark this job as started')
    p_event = sub.add_parser('event', help='append one event line')
    p_final = sub.add_parser('finalize', help='record the end of this job')
    p_show = sub.add_parser('show', help='print status.json')
    for p in (p_start, p_event, p_final, p_show):
        p.add_argument('--stage', required=True)
        p.add_argument('--monitor-dir', type=Path, help='default: <repo>/modal_results/hyak_tillicum/<stage>')
    p_start.add_argument('--task')
    p_event.add_argument('--name', required=True)
    p_event.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                         help='extra field; VALUE is parsed as JSON when possible')
    p_final.add_argument('--exit-code', type=int, required=True)
    p_final.add_argument('--log', type=Path, help='log whose tail becomes the error text on failure')
    p_final.add_argument('--message')
    args = parser.parse_args(argv)
    try:
        if args.command == 'start':
            result = start(args.stage, task=args.task, monitor_dir=args.monitor_dir)
        elif args.command == 'event':
            fields = {}
            for item in args.set:
                key, sep, value = item.partition('=')
                if not sep or not _FIELD_RE.fullmatch(key):
                    raise ValueError(f'--set expects KEY=VALUE, got {item!r}')
                fields[key] = _parse_value(value)
            stage = validate_stage(args.stage)
            directory = args.monitor_dir or default_monitor_dir(stage)
            status = _read_status(Path(directory) / 'status.json')
            field, total, unit = _meta(stage, status)
            writer = StatusWriter(stage, total, unit, field, _platform(stage, status, None), directory)
            result = writer.event({'event': args.name, **fields})
        elif args.command == 'finalize':
            result = finalize(args.stage, args.exit_code, log_path=args.log, message=args.message,
                              monitor_dir=args.monitor_dir)
        else:
            stage = validate_stage(args.stage)
            result = _read_status(Path(args.monitor_dir or default_monitor_dir(stage)) / 'status.json')
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}. Remedy: check the arguments and that the monitor directory is writable '
              f'(exit 2 = usage or environment error).', file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
