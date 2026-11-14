"""Slurm queries for jobs and wrappers.  Every function here returns a value; none raises into callers
(except the pure parsers, which raise ValueError on malformed input).

Semantics confirmed on slurm.schedmd.com (sbatch and squeue manual pages, 2026-11):

* ``SLURM_JOB_END_TIME``: "The UNIX timestamp for a job's projected end time";
  ``SLURM_JOB_START_TIME``: the job's start time (sbatch OUTPUT ENVIRONMENT VARIABLES).
* ``squeue -o %L``: "Time left for the job to execute in days-hours:minutes:seconds";
  "NOT_SET" if not yet established, "UNLIMITED" for no limit.
* ``--time`` formats: "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours",
  "days-hours:minutes", "days-hours:minutes:seconds"; zero means no limit.
* ``--signal``: sig_time must be an integer between 0 and 65535; the signal may be sent
  up to 60 seconds earlier than specified; ``B:`` signals only the batch shell.
* ``--dependency=singleton``: "only one job by that name and owned by that user can be
  running or suspended at any point in time".
"""
from __future__ import annotations

import math
import os
import re
import socket
import subprocess
import threading
import time

#: squeue %T states of a job that has not finished (it may still hold or regain resources).
ALIVE_STATES = frozenset({
    'PENDING', 'RUNNING', 'SUSPENDED', 'COMPLETING', 'CONFIGURING', 'RESIZING', 'REQUEUED',
    'REQUEUE_HOLD', 'REQUEUE_FED', 'SIGNALING', 'STAGE_OUT', 'STOPPED', 'RESV_DEL_HOLD',
    'SPECIAL_EXIT',
})
#: squeue %T states of a finished job.
DEAD_STATES = frozenset({
    'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY',
    'PREEMPTED', 'REVOKED', 'TIMEOUT',
})

#: Largest --signal sig_time Slurm accepts.
MAX_SIGNAL_SECONDS = 65535

_SLURM_JOB_RE = re.compile(r'\d+(?:[_+]\d+)?')
_TIME_DHMS = re.compile(r'(\d+)-(\d+)(?::(\d+)(?::(\d+))?)?')
_TIME_HMS = re.compile(r'(\d+):(\d+):(\d+)')
_TIME_MS = re.compile(r'(\d+):(\d+)')
_TIME_M = re.compile(r'(\d+)')


def job_id():
    """``$SLURM_JOB_ID`` or None outside a Slurm job."""
    value = os.environ.get('SLURM_JOB_ID', '').strip()
    return value or None


def worker_id() -> str:
    """Stable id of this process's job for status rows: the Slurm job id, else ``local-<pid>``."""
    return job_id() or f'local-{os.getpid()}'


def node_name() -> str:
    """``$SLURMD_NODENAME`` or the host name."""
    value = os.environ.get('SLURMD_NODENAME', '').strip()
    if value:
        return value
    try:
        return socket.gethostname() or 'unknown-host'
    except OSError:
        return 'unknown-host'


def cluster_name():
    """``$SLURM_CLUSTER_NAME`` or None."""
    return os.environ.get('SLURM_CLUSTER_NAME', '').strip() or None


def partition():
    """``$SLURM_JOB_PARTITION`` or None."""
    return os.environ.get('SLURM_JOB_PARTITION', '').strip() or None


def profile_label() -> str:
    """Dashboard ``profile`` of this job: ``<cluster>/<partition>`` (either part may be missing), else ``local``."""
    parts = [p for p in (cluster_name(), partition()) if p]
    return '/'.join(parts) if parts else 'local'


def job_start_time():
    """``$SLURM_JOB_START_TIME`` as a UNIX timestamp (float), or None."""
    return _epoch(os.environ.get('SLURM_JOB_START_TIME'))


def job_end_time():
    """``$SLURM_JOB_END_TIME`` as a UNIX timestamp (float), or None (unset, empty, zero or malformed)."""
    return _epoch(os.environ.get('SLURM_JOB_END_TIME'))


def _epoch(value):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def parse_time_limit(text) -> float:
    """Seconds of a Slurm time string.

    Accepts the ``--time`` formats ``M``, ``M:S``, ``H:M:S``, ``D-H``, ``D-H:M``,
    ``D-H:M:S`` (so squeue's ``MM:SS`` and ``[D-]HH:MM:SS`` too) and
    ``UNLIMITED`` / ``INFINITE`` (returns ``math.inf``).  Returns an int for finite
    values.  Raises ValueError for anything else (including ``NOT_SET``).
    Mirrors ``hyak_parse_time`` in hyak/slurm/lib.sh.
    """
    value = str(text).strip()
    if value.upper() in ('UNLIMITED', 'INFINITE'):
        return math.inf
    match = _TIME_DHMS.fullmatch(value)
    if match:
        days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    match = _TIME_HMS.fullmatch(value)
    if match:
        hours, minutes, seconds = (int(g) for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    match = _TIME_MS.fullmatch(value)
    if match:
        minutes, seconds = (int(g) for g in match.groups())
        return minutes * 60 + seconds
    match = _TIME_M.fullmatch(value)
    if match:
        return int(match.group(1)) * 60
    raise ValueError(f'Not a Slurm time limit: {text!r} (expected e.g. 24:00:00, 1-00:00:00, 90)')


def _squeue_environment():
    """The environment for squeue: the caller's without ``SQUEUE_*``.

    squeue reads filters from the environment (SQUEUE_STATES, SQUEUE_PARTITION, SQUEUE_NAMES,
    SQUEUE_ACCOUNT, SQUEUE_QOS, SQUEUE_USERS, ...; squeue manual, "ENVIRONMENT VARIABLES").  A running
    job hidden by such a filter prints nothing with exit status 0, which must never be mistaken for a
    finished job, and sbatch --export=ALL hands the submitting shell's variables to every job.
    ``SLURM_CLUSTERS`` is kept: it selects the cluster the jobs were submitted to.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith('SQUEUE_')}


def _run(command, timeout=20.0):
    """Run a read-only command (squeue); return (returncode, stdout, stderr) or None if it cannot run or times out."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors='replace',
                                timeout=timeout, check=False, env=_squeue_environment())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return result.returncode, result.stdout or '', result.stderr or ''


def squeue_time_left(job=None, timeout=20.0):
    """Seconds left for ``job`` (default: this job) from ``squeue -h -j <id> -o %L``.

    Returns ``math.inf`` for UNLIMITED, None if unknown (no job, squeue missing,
    timeout, NOT_SET, unparseable output).
    """
    job = job or job_id()
    if not job or not _SLURM_JOB_RE.fullmatch(str(job)):
        return None
    result = _run(['squeue', '-h', '-j', str(job), '-o', '%L'], timeout=timeout)
    if result is None or result[0] != 0:
        return None
    lines = [line.strip() for line in result[1].splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    try:
        return float(parse_time_limit(lines[0]))
    except ValueError:
        return None


def seconds_remaining(default=None, *, now=None):
    """Seconds until this job's time limit.

    Order: ``$SLURM_JOB_END_TIME`` (minus now, floored at 0), then
    ``squeue -h -j $SLURM_JOB_ID -o %L``, then ``default``.  May return
    ``math.inf`` (UNLIMITED).  Never raises.
    """
    end = job_end_time()
    if end is not None:
        current = time.time() if now is None else now
        return max(0.0, end - current)
    left = squeue_time_left()
    if left is not None:
        return left
    return default


def state_alive(state):
    """True for a squeue ``%T`` state of an unfinished job, False for a finished one, None for anything else."""
    state = str(state).strip().upper()
    if state in ALIVE_STATES:
        return True
    if state in DEAD_STATES:
        return False
    return None


def is_job_alive(job_id, timeout=20.0):
    """Whether a Slurm job has not finished: True, False, or None when it cannot be determined.

    Uses ``squeue -h -t all -j <id> -o %T`` with the caller's ``SQUEUE_*`` variables removed.  ``-t all``
    makes squeue print finished jobs that the controller still remembers, so a listed job is True for
    pending / running / completing-like states and False for finished states.  Only "Invalid job id
    specified" (a job the controller has forgotten) means False without a state.  Empty output with exit
    status 0 is **unknown** (None): it is also what a filter hiding a running job produces.  Non-Slurm ids
    (e.g. ``local-123``), a missing ``squeue`` command, timeouts and unknown states give None.  Never raises.
    """
    job = str(job_id).strip() if job_id is not None else ''
    if not _SLURM_JOB_RE.fullmatch(job):
        return None
    result = _run(['squeue', '-h', '-t', 'all', '-j', job, '-o', '%T'], timeout=timeout)
    if result is None:
        return None
    code, out, err = result
    if code != 0:
        return False if 'invalid job id' in (out + err).lower() else None
    states = {line.strip().upper() for line in out.splitlines() if line.strip()}
    if not states:
        return None
    if states & ALIVE_STATES:
        return True
    if states <= DEAD_STATES:
        return False
    return None


#: A listing of the user's jobs is reused for this long (one squeue call per process and minute).
LISTING_SECONDS = 60.0
_clock = time.monotonic
_listing_lock = threading.Lock()
_listing = None  # (taken_at, table or None)


def reset_job_listing():
    """Forget the cached job listing (tests)."""
    global _listing
    with _listing_lock:
        _listing = None


def job_table(*, max_age=LISTING_SECONDS, timeout=20.0):
    """``{job id: (STATE, name)}`` of the calling user's jobs, or None when squeue cannot be asked.

    One ``squeue -h -t all -u <uid> -o '%i|%T|%j'`` call, cached per process for ``max_age`` seconds (a
    failed call is cached too, so a broken squeue is not hammered).  ``-t all`` includes jobs that finished
    recently.  A job that is absent from the table is **not** known to be finished: it may belong to another
    user, be an array task listed under another id, or be gone from the controller; ask
    :func:`is_job_alive` about it.  Never raises.
    """
    global _listing
    with _listing_lock:
        now = _clock()
        if _listing is not None and 0 <= now - _listing[0] < max_age:
            return _listing[1]
        result = _run(['squeue', '-h', '-t', 'all', '-u', str(os.getuid()), '-o', '%i|%T|%j'], timeout=timeout)
        table = None
        if result is not None and result[0] == 0:
            table = {}
            for line in result[1].splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split('|', 2)
                if len(parts) != 3 or not parts[0] or not parts[1]:
                    table = None  # not the format asked for: an incomplete listing is no listing
                    break
                table[parts[0]] = (parts[1].strip().upper(), parts[2])
        _listing = (now, table)
        return table


def job_states(*, max_age=LISTING_SECONDS, timeout=20.0):
    """``{job id: STATE}`` view of :func:`job_table` (None when squeue cannot be asked)."""
    table = job_table(max_age=max_age, timeout=timeout)
    return None if table is None else {job: state for job, (state, _name) in table.items()}


def signal_seconds(margin_minutes: int, time_limit_seconds) -> int:
    """``--signal=B:TERM@<seconds>`` value: min(margin, a quarter of the time limit, 65535); 0 = no signal.

    Mirrors the arithmetic in hyak/slurm/submit.sh.
    """
    if margin_minutes <= 0:
        return 0
    value = margin_minutes * 60
    if math.isfinite(time_limit_seconds):
        value = min(value, int(time_limit_seconds) // 4)
    return max(0, min(value, MAX_SIGNAL_SECONDS))


def sbatch_resources(cfg: dict, stage: str, debug: bool = False) -> dict:
    """Resolved sbatch resources of a stage, exactly as hyak/slurm/submit.sh computes them.

    Precedence for QoS and time: ``debug=True`` (submit.sh --debug) -> debug values;
    else a non-empty per-stage override ``HYAK_QOS_<STAGE>`` / ``HYAK_TIME_<STAGE>``;
    else the debug values for DEBUG_STAGES (check-env, probe); else the globals.
    Other flags: non-empty per-stage override, else the global.  A value of
    ``none`` (or empty) for partition, qos, account, gres, cpus, mem means the
    flag is omitted (None here).  Returns keys partition, qos, account, gres,
    cpus, mem, time, time_seconds, signal_seconds, margin_minutes, job_name,
    singleton, debug, sbatch_file, run.
    """
    from hyak.common.config import stage_value, OMITTABLE_FLAGS
    from hyak.common.paths import (DEBUG_STAGES, SBATCH_FILES, SINGLETON_STAGES, STAGE_RUN,
                                   stage_suffix)
    suffix = stage_suffix(stage)
    use_debug_values = bool(debug) or (stage in DEBUG_STAGES)

    def pick(flag, debug_key=None):
        override = cfg.get(f'HYAK_{flag}_{suffix}', '')
        if debug and debug_key:
            value = cfg[debug_key]
        elif override != '':
            value = override
        elif use_debug_values and debug_key:
            value = cfg[debug_key]
        else:
            value = cfg[f'HYAK_{flag}']
        if flag in OMITTABLE_FLAGS and value.strip().lower() in ('', 'none'):
            return None
        return value

    time_text = pick('TIME', 'HYAK_TIME_DEBUG')
    seconds = parse_time_limit(time_text)
    margin = int(stage_value(cfg, 'HYAK_SEGMENT_MARGIN_MINUTES', stage))
    return {
        'partition': pick('PARTITION'),
        'qos': pick('QOS', 'HYAK_QOS_DEBUG'),
        'account': pick('ACCOUNT'),
        'gres': pick('GRES'),
        'cpus': pick('CPUS'),
        'mem': pick('MEM'),
        'time': time_text,
        'time_seconds': seconds,
        'margin_minutes': margin,
        'signal_seconds': signal_seconds(margin, seconds),
        'job_name': f'anytok-{stage}',
        'singleton': stage in SINGLETON_STAGES,
        'debug': use_debug_values,
        'sbatch_file': SBATCH_FILES[stage],
        'run': STAGE_RUN.get(stage),
    }
