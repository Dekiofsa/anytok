"""Generate the 1,000 pool tokenizers on Slurm (DESIGN.md D9 and 5.F; port of
``train_one`` in scripts/modal_generate_tokenizer_pool.py:65-102).

    python -m hyak.pool.worker [--workers 2] [--threads 4] [--indices 413 86 ... | pilots | 0-99]
                               [--pilots-first | --no-pilots-first] [--pilots-only]
                               [--min-remaining-minutes 150] [--max-attempts 3] [--retry-failed]
                               [--tokenizer-timeout-hours 6] [--claim-timeout-hours 6] [--report]
                               [--families bpe/no-regex ...]

Every job runs ``--workers`` loops.  Each loop picks the next index without a
verified manifest (the 12 pilot indices first), claims it with an atomic
``mkdir`` of ``input/pool/claims/NNNN`` (``owner.json`` with job id, host, pid,
start time; a heartbeat file every minute), and runs the unchanged worker
``scripts/tokenizer_pool.py --train-config C --recipe input/pool-recipe.json
--destination D`` as a child process with ``RAYON_NUM_THREADS=4`` and
``TOKENIZERS_PARALLELISM=true`` (scripts/modal_generate_tokenizer_pool.py:47,86-88).
The child writes into a staging directory inside the claim; the result is
verified, gets a ``provenance`` block (scripts/modal_generate_tokenizer_pool.py:89-91,
192-197, plus this port's identity fields), is written atomically, and the
staging directory is renamed to ``input/pool/tokenizers/NNNN``.

Rules:

* A completed manifest always wins: an index whose manifest verifies is never
  retrained, and a finished child never replaces a verified tokenizer.
* An existing manifest that does not verify (other configuration, provenance or
  file hash) blocks its index: nothing is retrained or deleted
  (scripts/modal_generate_tokenizer_pool.py:71-77 raised in that case).
* A claim may be taken over only when its owner job is provably gone
  (``squeue`` no longer lists it / a local owner pid has exited), when it was made by an
  earlier incarnation of this same job (a requeued job keeps its job id: another
  ``SLURM_RESTART_COUNT``, or another process that is not alive here), when the claim
  directory has no owner file after 10 minutes, or when its heartbeat is older
  than ``--claim-timeout-hours`` and the owner's job is not known to be alive.
* A failure writes ``tokenizers/NNNN/failure.json`` (original fields ``config``,
  ``error``, ``at_utc`` plus ``attempts`` and ``history``) and releases the claim;
  after ``--max-attempts`` failures the index is left for the user (never another
  configuration).  An owner that died while training (its claim had a staging
  directory when it was taken over: the job was killed with its tokenizer) counts as a
  failed attempt too.  A child stopped because the job is ending is "interrupted" and
  does not count as a failure.
* A brought-in pool (``full-results.json`` present, or ``tokenizer.json`` without this
  worker's ``manifest.json``) is never touched: such an index is "blocked", and a pool
  job on a pool that already has an index refuses to start (exit 2).  An existing
  ``tokenizers/NNNN`` is moved aside only when it holds nothing but ``failure.json``.
* No new tokenizer is started when less than ``--min-remaining-minutes`` of job
  time remain, or when the slowest finished member of its algorithm/pretokenizer
  family (x1.25) would not fit.  On SIGTERM (Slurm's warning signal) no new
  tokenizer starts and running ones continue until 2 minutes before the end; a
  second SIGTERM or a SIGINT stops them at once.

Exit status: 0 when this job finished normally (also when the pool is complete although
an attempt failed in this job); 3 when an index failed in this job or the pool has blocked
or exhausted indices (the pool cannot complete without the user); 2 for invalid inputs or
environment (message with a remedy); 1 only for an unexpected crash (traceback).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

from hyak.common import slurm
from hyak.common.config import ConfigError, int_value, load_config, redact_text
from hyak.common.io import append_jsonl, atomic_write_json, atomic_write_text, fsync_dir, read_json, tail_lines, utc_now
from hyak.common.paths import STAGE_POOL, Paths
from hyak.common.status import StatusWriter
from hyak.pool import (
    ALGORITHMS, PATCH_FILES, PATCH_SHA256, PLAN_REL, PRETOKENIZERS, RAYON_THREADS, TOKENIZERS_VERSION, UNIGRAM_PATCHES, UPSTREAM_REVISION,
    UPSTREAM_TAG, WORKER_REL, WORKER_SHA256, PoolError, family, file_sha256, index_name, load_plan,
    manifest_problems, manifest_text, parse_indices, recipe_identity, recipe_problems, tokenizer_json_problems)

OWNER_FILE = 'owner.json'
HEARTBEAT_FILE = 'heartbeat.json'
OWNERLESS_GRACE_SECONDS = 600
BREAK_LOCK_STALE_SECONDS = 300
DEFAULT_KILL_MARGIN_SECONDS = 120
#: Slack added to a family's slowest observed run when deciding whether it fits in the job.
FAMILY_FIT_FACTOR = 1.25
EXIT_OK, EXIT_INPUT, EXIT_ATTENTION = 0, 2, 3


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _restart_count() -> int:
    """``$SLURM_RESTART_COUNT`` (0 outside Slurm and before the first requeue)."""
    try:
        return int(os.environ.get('SLURM_RESTART_COUNT') or 0)
    except ValueError:
        return 0


def _log(event: str, **fields) -> None:
    print(json.dumps({'at_utc': utc_now(), 'event': event, **fields}, ensure_ascii=False), flush=True)


# --- claims -------------------------------------------------------------------------------------
@dataclass
class Claim:
    index: int
    path: Path
    token: str
    owner: dict


class ClaimStore:
    """Atomic per-index claims under ``root`` (``input/pool/claims``).

    ``claims/NNNN`` is created with ``os.mkdir`` (fails if it exists, also across
    nodes of a POSIX file system).  The owner then writes ``owner.json`` and
    refreshes ``heartbeat.json``.  Stale claims are broken by one process at a time
    (``claims/NNNN.break`` lock directory): it re-reads the claim under the lock,
    renames it away and the index becomes claimable again.
    """

    def __init__(self, root, *, timeout_seconds=6 * 3600, ownerless_grace=OWNERLESS_GRACE_SECONDS,
                 alive=None, clock=time.time, job=None, host=None, pid=None, on_event=None, restart_count=None):
        self.root = Path(root)
        self.timeout = float(timeout_seconds)
        self.ownerless_grace = float(ownerless_grace)
        self._alive = alive or slurm.is_job_alive
        self.clock = clock
        self.job = slurm.job_id() if job is None else job
        self.host = host or slurm.node_name()
        self.pid = pid or os.getpid()
        self.restart_count = _restart_count() if restart_count is None else int(restart_count)
        self.on_event = on_event or (lambda name, **fields: None)
        self._alive_cache = {}

    def path(self, index: int) -> Path:
        return self.root / index_name(index)

    def _job_alive(self, job):
        """``is_job_alive`` with a 60 s cache (a definite False is kept), to spare the Slurm controller."""
        now = self.clock()
        cached = self._alive_cache.get(str(job))
        if cached is not None and (cached[0] is False or now - cached[1] < 60):
            return cached[0]
        alive = self._alive(job)
        if alive is not None:
            self._alive_cache[str(job)] = (alive, now)
        return alive

    # inspection ------------------------------------------------------------------------
    def inspect(self, index: int):
        """None when no claim exists, else ``{"owner", "token", "dir_age", "heartbeat_age"}``."""
        path = self.path(index)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        now = self.clock()
        owner = None
        try:
            owner = read_json(path / OWNER_FILE)
        except (OSError, ValueError):
            owner = None
        if not isinstance(owner, dict) or not owner.get('token'):
            return {'owner': None, 'token': None, 'dir_age': now - stat.st_mtime, 'heartbeat_age': None}
        last = owner.get('started_unix') if isinstance(owner.get('started_unix'), (int, float)) else stat.st_mtime
        try:
            beat = read_json(path / HEARTBEAT_FILE)
            if isinstance(beat, dict) and beat.get('token') == owner['token'] and isinstance(beat.get('at_unix'), (int, float)):
                last = max(last, beat['at_unix'])
        except (OSError, ValueError):
            pass
        return {'owner': owner, 'token': owner['token'], 'dir_age': now - stat.st_mtime, 'heartbeat_age': now - last}

    def stale_reason(self, info):
        """Why a claim may be taken over, or None when it must be left alone."""
        if info is None:
            return None
        owner = info['owner']
        if owner is None:
            if info['dir_age'] > self.ownerless_grace:
                return f'claim directory without owner file for {info["dir_age"]:.0f} s'
            return None
        job = owner.get('job_id')
        if job:
            alive = self._job_alive(job)
            if alive is False:
                return f'owner job {job} is no longer known to squeue'
            if alive is True:
                if str(job) == str(self.job):
                    # A requeued job keeps its job id: the claims of its earlier incarnation are dead.
                    count = owner.get('restart_count', 0)
                    if count != self.restart_count:
                        return (f'owner is incarnation {count} of this job (requeued; this is incarnation '
                                f'{self.restart_count})')
                    if owner.get('pid') != self.pid and (owner.get('host') != self.host
                                                         or not _pid_alive(owner.get('pid'))):
                        return f'owner process {owner.get("pid")} on {owner.get("host")} of this job has exited'
                return None  # a claim of another live job is never taken over
        elif owner.get('host') == self.host and isinstance(owner.get('pid'), int):
            if owner.get('pid') == self.pid:
                return None
            return None if _pid_alive(owner['pid']) else f'owner process {owner["pid"]} on {self.host} has exited'
        age = info['heartbeat_age']
        if age is not None and age > self.timeout:
            return f'no heartbeat for {age / 3600:.2f} h (claim timeout {self.timeout / 3600:.2f} h)'
        return None

    # claiming --------------------------------------------------------------------------------
    def try_claim(self, index: int, slot: int = 0):
        """Claim ``index``; returns a Claim, or None when another live owner holds it."""
        path = self.path(index)
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            try:
                os.mkdir(path)
            except FileExistsError:
                info = self.inspect(index)
                if info is None:
                    continue
                reason = self.stale_reason(info)
                if reason is None:
                    return None
                if not self._break(index, info, reason):
                    return None
                continue
            token = uuid.uuid4().hex
            now = self.clock()
            owner = {'token': token, 'index': index, 'job_id': self.job, 'worker_id': self.job or f'local-{self.pid}',
                     'host': self.host, 'pid': self.pid, 'slot': slot, 'started_unix': now,
                     'started_at': utc_now(), 'cluster': slurm.cluster_name(), 'restart_count': self.restart_count}
            try:
                atomic_write_json(path / OWNER_FILE, owner)
                atomic_write_json(path / HEARTBEAT_FILE, {'token': token, 'at_unix': now})
            except OSError:
                shutil.rmtree(path, ignore_errors=True)
                raise
            return Claim(index, path, token, owner)
        return None

    def _break(self, index, judged, reason) -> bool:
        name = index_name(index)
        lock = self.root / f'{name}.break'
        try:
            os.mkdir(lock)
        except FileExistsError:
            try:
                if self.clock() - lock.stat().st_mtime > BREAK_LOCK_STALE_SECONDS:
                    os.rmdir(lock)
            except OSError:
                pass
            return False
        try:
            again = self.inspect(index)
            if again is None:
                return True
            if again['token'] != judged['token'] or self.stale_reason(again) is None:
                return False
            grave = self.root / f'.stale-{name}-{uuid.uuid4().hex}'
            try:
                os.rename(self.path(index), grave)
            except FileNotFoundError:
                return True
            moved = None
            try:
                moved = (read_json(grave / OWNER_FILE) or {}).get('token')
            except (OSError, ValueError, AttributeError):
                moved = None
            if moved != judged['token']:
                # A fresh claim replaced the stale one between the check and the rename: put it back.
                if not self.path(index).exists():
                    try:
                        os.rename(grave, self.path(index))
                    except OSError:
                        pass
                return False
            had_staging = (grave / 'staging').is_dir()
            self.on_event('claim_takeover', index=index, reason=reason, previous_owner=judged['owner'],
                          had_staging=had_staging)
            shutil.rmtree(grave, ignore_errors=True)
            return True
        finally:
            try:
                os.rmdir(lock)
            except OSError:
                pass

    def held(self, claim: Claim) -> bool:
        try:
            owner = read_json(claim.path / OWNER_FILE)
        except (OSError, ValueError):
            return False
        return isinstance(owner, dict) and owner.get('token') == claim.token

    def heartbeat(self, claim: Claim) -> None:
        try:
            atomic_write_json(claim.path / HEARTBEAT_FILE, {'token': claim.token, 'at_unix': self.clock()})
        except OSError as exc:
            print(f'[hyak pool] WARNING: heartbeat of claim {claim.index} failed: {exc}', file=sys.stderr, flush=True)

    def release(self, claim: Claim) -> bool:
        """Remove the claim if this owner still holds it."""
        if not self.held(claim):
            return False
        grave = self.root / f'.released-{index_name(claim.index)}-{claim.token}'
        try:
            os.rename(claim.path, grave)
        except OSError:
            return False
        shutil.rmtree(grave, ignore_errors=True)
        return True

    def cleanup(self, older_than=3600) -> None:
        """Remove leftovers of interrupted takeovers/releases (``.stale-*``, ``.released-*``)."""
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith(('.stale-', '.released-')):
                try:
                    if self.clock() - entry.stat().st_mtime > older_than:
                        shutil.rmtree(entry, ignore_errors=True)
                except OSError:
                    pass


# --- pool contents --------------------------------------------------------------------------------
class PoolState:
    """Classification of every index: complete / missing / failed / blocked (cached by file stat)."""

    def __init__(self, plan, tokenizers_dir, identity):
        self.plan = plan
        self.dir = Path(tokenizers_dir)
        self.identity = identity
        self._cache = {}
        self.summaries = {}

    def classify(self, index: int):
        """``(state, detail)`` with state in complete / missing / failed / blocked."""
        folder = self.dir / index_name(index)
        manifest_path, tokenizer_path, failure_path = folder / 'manifest.json', folder / 'tokenizer.json', folder / 'failure.json'
        try:
            m_stat = manifest_path.stat()
        except FileNotFoundError:
            m_stat = None
        if m_stat is not None:
            try:
                t_stat = tokenizer_path.stat()
                key = (m_stat.st_mtime_ns, m_stat.st_size, t_stat.st_mtime_ns, t_stat.st_size)
            except FileNotFoundError:
                key = (m_stat.st_mtime_ns, m_stat.st_size, None, None)
            cached = self._cache.get(index)
            if cached and cached[0] == key:
                return cached[1]
            try:
                manifest = read_json(manifest_path)
            except (OSError, ValueError) as exc:
                result = ('blocked', f'unreadable manifest: {exc}')
            else:
                problems = manifest_problems(manifest, self.plan['configs'][index], self.plan, tokenizer_path,
                                             check_hash=True, identity=self.identity)
                if problems:
                    result = ('blocked', '; '.join(problems))
                else:
                    result = ('complete', None)
                    self.summaries[index] = {k: manifest.get(k) for k in
                                             ('train_seconds', 'elapsed_seconds', 'cpu_seconds', 'peak_rss_gib')}
            self._cache[index] = (key, result)
            return result
        self._cache.pop(index, None)
        self.summaries.pop(index, None)
        try:
            others = sorted(set(os.listdir(folder)) - {'failure.json'})
        except (FileNotFoundError, NotADirectoryError):
            others = []
        if others:
            # This worker publishes tokenizer.json and manifest.json together (one rename) and otherwise writes only
            # failure.json: anything else (e.g. a tokenizer.json of a brought-in original pool) is never retrained,
            # replaced or moved.
            what = ('tokenizer.json without manifest.json' if 'tokenizer.json' in others else f'unexpected files {others}')
            return ('blocked', f'{what} in {folder} (not written by this worker; a brought-in pool must not be used '
                               f'with pool jobs)')
        if failure_path.exists():
            try:
                failure = read_json(failure_path)
                attempts = int(failure.get('attempts', 1))
            except (OSError, ValueError, TypeError, AttributeError):
                attempts = 1
            return ('failed', attempts)
        return ('missing', None)

    def family_stats(self):
        """Per algorithm/pretokenizer: planned, complete, timing summaries of completed members."""
        stats = {}
        for config in self.plan['configs']:
            item = stats.setdefault(family(config), {'planned': 0, 'complete': 0, '_e': [], '_t': [], '_c': [], '_r': []})
            item['planned'] += 1
            summary = self.summaries.get(config['index'])
            if summary:
                item['complete'] += 1
                for key, field in (('_e', 'elapsed_seconds'), ('_t', 'train_seconds'), ('_c', 'cpu_seconds'),
                                   ('_r', 'peak_rss_gib')):
                    value = summary.get(field)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        item[key].append(float(value))
        result = {}
        for name, item in sorted(stats.items()):
            row = {'planned': item['planned'], 'complete': item['complete']}
            if item['_e']:
                row.update(median_elapsed_seconds=round(statistics.median(item['_e']), 1),
                           max_elapsed_seconds=round(max(item['_e']), 1))
            if item['_t']:
                row['median_train_seconds'] = round(statistics.median(item['_t']), 1)
            if item['_c']:
                row['median_cpu_seconds'] = round(statistics.median(item['_c']), 1)
            if item['_r']:
                row['max_peak_rss_gib'] = round(max(item['_r']), 3)
            result[name] = row
        return result


# --- the worker -------------------------------------------------------------------------------------
class PoolWorker:
    def __init__(self, args, cfg, paths):
        self.args = args
        self.cfg = cfg
        self.paths = paths
        self.repo = paths.repo
        self.lock = threading.RLock()
        self.status_lock = threading.Lock()    # status.json writes (refresh_status), outside self.lock
        self._status_seq = 0                   # snapshot number (taken under self.lock)
        self._status_written_seq = 0           # newest snapshot written
        self._last_status = {}
        self._last_progress = None             # (completed, total, failed, blocked, exhausted, pilots) of the last 'progress' event
        self.stopping = threading.Event()      # no new tokenizers (first SIGTERM)
        self.hard_stop = threading.Event()     # stop running children now (second SIGTERM / SIGINT)
        self.failed_here = []
        self.local_failures = {}
        self.slot_rows = {}
        self.started = time.time()
        self._end_time = None
        self._end_checked = 0.0
        self._setup()

    # setup ----------------------------------------------------------------------------------------
    def _setup(self):
        args, paths = self.args, self.paths
        plan_path = Path(args.plan) if args.plan else self.repo / PLAN_REL
        self.plan, self.plan_info = load_plan(plan_path, allow_test_plan=args.allow_test_plan)
        self.production = self.plan_info['production']
        if not self.production:
            if not args.pool_dir:
                raise PoolError('A test plan needs an explicit --pool-dir (it must never write into the production pool)')
            if os.path.normpath(args.pool_dir) == os.path.normpath(str(paths.pool_dir)):
                raise PoolError(f'--pool-dir {args.pool_dir} is the production pool; use another directory for a test plan')
        elif args.worker_script or args.no_wheel_check:
            raise PoolError('--worker-script and --no-wheel-check are only allowed with --allow-test-plan')
        self.pool_dir = Path(args.pool_dir) if args.pool_dir else paths.pool_dir
        self.tokenizers_dir = self.pool_dir / 'tokenizers'
        self.claims_dir = self.pool_dir / 'claims'
        self.monitor_dir = Path(args.monitor_dir) if args.monitor_dir else paths.monitor_dir(STAGE_POOL)
        self.log_dir = self.monitor_dir / 'tokenizer-logs'
        self.count = len(self.plan['configs'])

        # Worker script: the unchanged scripts/tokenizer_pool.py (pinned hash) unless a test override.
        self.worker_script = Path(args.worker_script) if args.worker_script else self.repo / WORKER_REL
        if not self.worker_script.is_file():
            raise PoolError(f'{self.worker_script} not found')
        self.worker_sha256 = file_sha256(self.worker_script)
        if not args.worker_script and self.worker_sha256 != WORKER_SHA256:
            raise PoolError(f'{self.worker_script} has SHA-256 {self.worker_sha256}, expected {WORKER_SHA256}. '
                            f'Restore it: git checkout -- {WORKER_REL}')

        self.threads = args.threads if args.threads is not None else int_value(self.cfg, 'HYAK_POOL_THREADS')
        self.workers = args.workers if args.workers is not None else int_value(self.cfg, 'HYAK_POOL_WORKERS')
        if self.workers < 1:
            raise PoolError('--workers must be >= 1')
        if self.threads != RAYON_THREADS and not args.allow_test_plan:
            raise PoolError(f'The original pool used RAYON_NUM_THREADS={RAYON_THREADS} (Unigram results may depend on '
                            f'it); --threads {self.threads} is not allowed for the production pool')
        cpus = os.environ.get('SLURM_CPUS_PER_TASK', '')
        if cpus.isdigit() and self.workers * self.threads > int(cpus):
            print(f'[hyak pool] WARNING: {self.workers} workers x {self.threads} threads exceed the {cpus} CPUs '
                  'of this job; tokenizers will compete for cores', file=sys.stderr, flush=True)

        # Generation recipe (input/pool-recipe.json): equal to the plan's recipe except source_path.
        self.recipe_path = Path(args.recipe) if args.recipe else paths.pool_recipe
        try:
            raw = self.recipe_path.read_bytes()
            recipe = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise PoolError(f'Cannot read the generation recipe {self.recipe_path} ({exc}). Remedy: run the '
                            'stage-inputs step (python -m hyak.corpus.stage_inputs)') from exc
        problems = recipe_problems(recipe, self.plan['recipe'])
        if problems:
            raise PoolError(f'{self.recipe_path} does not match the plan recipe: ' + '; '.join(problems))
        # Identity of the recipe without this cluster's absolute paths: each rewritable path is reduced to the
        # trailing components recipe_problems pins, so the same corpus under another path keeps the identity
        # (the corpus fingerprint below covers the file content).
        self.recipe_sha256 = recipe_identity(recipe)
        corpus = {}
        missing = []
        for bucket in recipe['buckets']:
            folder = Path(bucket['source_path'])
            train = folder / 'train.jsonl.gz'
            try:
                size = train.stat().st_size
            except OSError:
                missing.append(str(train))
                continue
            train_sha = None
            try:
                bucket_manifest = read_json(folder / 'manifest.json')
                train_sha = ((bucket_manifest or {}).get('files') or {}).get('train.jsonl.gz')
            except (OSError, ValueError, AttributeError):
                train_sha = None
            corpus[bucket['name']] = {'train_bytes': size, 'train_sha256': train_sha}
        if missing:
            raise PoolError('Corpus files missing: ' + ', '.join(missing[:5])
                            + '. Remedy: finish the corpus build (python -m hyak.corpus.build) and stage-inputs')

        # The installed patched wheel (wheels/provenance.json written by hyak/pool/build_wheel.sh).
        self.wheel = None
        self.identity_note = None
        if not args.no_wheel_check:
            try:
                self.wheel = check_installed_wheel(Path(args.wheel_provenance) if args.wheel_provenance
                                                   else paths.wheel_provenance, self.worker_script)
            except PoolError as exc:
                if not args.report:
                    raise
                # --report only reads: without the wheel the provenance identity cannot be compared.
                self.identity_note = f'provenance identity not checked: {exc}'
        self.identity = {
            'plan_sha256': self.plan['plan_sha256'],
            'plan_file_sha256': self.plan_info['file_sha256'],
            'production_plan': self.production,
            'worker_sha256': self.worker_sha256,
            'rayon_threads': self.threads,
            'runtime_wheel_sha256': self.wheel['sha256'] if self.wheel else None,
            'upstream_revision': self.wheel['revision'] if self.wheel else None,
            'patches': self.wheel['patches'] if self.wheel else None,
            'recipe_sha256': self.recipe_sha256,
            'corpus': corpus,
        }
        self.state = PoolState(self.plan, self.tokenizers_dir, None if self.identity_note else self.identity)
        self.claims = ClaimStore(self.claims_dir, timeout_seconds=args.claim_timeout_hours * 3600,
                                 on_event=self._event)
        self.status = StatusWriter(STAGE_POOL, self.count, 'tokenizers', 'completed', monitor_dir=self.monitor_dir,
                                   cfg=self.cfg)
        self.wrapper_sha256 = file_sha256(Path(__file__))
        pilots = list(self.plan['pilot_indices'])
        if args.indices:
            order = parse_indices(args.indices, self.count, pilots)
        elif args.pilots_only:
            order = pilots
        else:
            order = list(range(self.count))
        if args.pilots_first and not args.indices:
            order = [i for i in pilots if i in order] + [i for i in order if i not in pilots]
        if args.pilots_only:
            order = [i for i in order if i in pilots]
        if args.families:
            known = {f'{a}/{p}' for a in ALGORITHMS for p in PRETOKENIZERS}
            unknown = sorted(set(args.families) - known)
            if unknown:
                raise PoolError(f'Unknown families {unknown}; expected algorithm/pretokenizer, e.g. bpe/no-regex')
            order = [i for i in order if family(self.plan['configs'][i]) in set(args.families)]
        self.order = order
        self.pilots = pilots

    # events / status ------------------------------------------------------------------------------
    def _event(self, name, **fields):
        if name == 'claim_takeover' and fields.get('had_staging'):
            self._record_owner_died(fields['index'], fields.get('previous_owner') or {}, fields.get('reason'))
        _log(name, **fields)
        try:
            self.status.event({'event': name, **fields})
        except ValueError as exc:
            print(f'[hyak pool] WARNING: event {name} not recorded: {exc}', file=sys.stderr, flush=True)

    def _scan(self):
        counts = {'complete': 0, 'missing': 0, 'failed': 0, 'blocked': 0}
        failed, blocked, exhausted = [], [], []
        for index in range(self.count):
            state, detail = self.state.classify(index)
            counts[state] += 1
            if state == 'failed':
                failed.append(index)
                if detail >= self.args.max_attempts:
                    exhausted.append(index)
            elif state == 'blocked':
                blocked.append(index)
        return counts, failed, blocked, exhausted

    def refresh_status(self, final=False):
        """Rewrite status.json (never appended to events.jsonl); a 'progress' event only when the counts change.

        ``self.lock`` is held only for the snapshot (scan, family statistics, slot rows, ETA sum); reading
        other jobs' rows and writing happen outside it under ``self.status_lock``, so the slot threads
        are not blocked by status I/O on a slow file system.  A snapshot older than the last one written
        is not written (two threads can refresh at once); its caller gets the newest written status.
        """
        with self.lock:
            counts, failed, blocked, exhausted = self._scan()
            families = self.state.family_stats()
            completed = counts['complete']
            pilots_done = sum(self.state.classify(i)[0] == 'complete' for i in self.pilots)
            rows = [] if final else [dict(row) for _, row in sorted(self.slot_rows.items())]
            eta_total = self._eta_total(families)
            self._status_seq += 1
            seq = self._status_seq
        extra = dict(families=families, failed_indices=failed, blocked_indices=blocked,
                     exhausted_indices=exhausted, pilots_complete=pilots_done, pilots_total=len(self.pilots),
                     plan_sha256=self.plan['plan_sha256'], production_plan=self.production,
                     max_attempts=self.args.max_attempts)
        with self.status_lock:
            if seq < self._status_written_seq:
                return self._last_status
            self._status_written_seq = seq
            progress = (completed, self.count, tuple(failed), tuple(blocked), tuple(exhausted), pilots_done)
            if progress != self._last_progress:
                self._last_progress = progress
                self._event('progress', completed=completed, total=self.count, failed_indices=failed,
                            blocked_indices=blocked, exhausted_indices=exhausted, pilots_complete=pilots_done)
            if completed == self.count:
                self._last_status = self.status.update(completed, stage='complete', state='complete', workers=[],
                                                       append=False, **extra)
                return self._last_status
            others = self.status.other_rows()
            active = max(1, len(rows) + len(others))
            eta = None if eta_total is None else round(eta_total / active, 0)
            stage = 'pilots' if pilots_done < len(self.pilots) else 'generating'
            error = None
            if blocked or exhausted:
                error = (f'{len(blocked)} blocked and {len(exhausted)} exhausted indices need attention '
                         f'(see python -m hyak.pool.worker --report)')
            state = 'running'
            if final and not others:
                state, stage = 'paused', 'paused'
            self._last_status = self.status.update(completed, stage=stage, state=state, workers=rows, eta_seconds=eta,
                                                   elapsed_seconds=round(time.time() - self.started, 1), error=error,
                                                   append=False, **extra)
            return self._last_status

    def _eta_total(self, families):
        """Summed median duration of the tokenizers still to build (None without any timing); call under the lock."""
        overall = [row['median_elapsed_seconds'] for row in families.values() if 'median_elapsed_seconds' in row]
        if not overall:
            return None
        fallback = statistics.median(overall)
        total = 0.0
        for config in self.plan['configs']:
            if self.state.summaries.get(config['index']) is None:
                row = families.get(family(config), {})
                total += row.get('median_elapsed_seconds', fallback)
        return total

    def _row(self, slot, task, state, eta=None, **extra):
        row = self.status.worker_row(task=task, state=state, suffix=f's{slot}', eta_seconds=eta, **extra)
        with self.lock:
            self.slot_rows[slot] = row

    # time ---------------------------------------------------------------------------------------------
    def remaining(self) -> float:
        """Seconds of job time left (inf outside Slurm); re-queried every 10 minutes."""
        now = time.time()
        if self._end_time is None or now - self._end_checked > 600:
            left = slurm.seconds_remaining()
            self._end_time = math.inf if left is None or not math.isfinite(left) else now + left
            self._end_checked = now
        return self._end_time - now

    def _family_estimate(self, config):
        with self.lock:
            values = [s['elapsed_seconds'] for i, s in list(self.state.summaries.items())
                      if family(self.plan['configs'][i]) == family(config)
                      and isinstance(s.get('elapsed_seconds'), (int, float))]
        return max(values) * FAMILY_FIT_FACTOR if values else None

    # scheduling ------------------------------------------------------------------------------------------
    def claim_next(self, slot):
        min_remaining = self.args.min_remaining_minutes * 60
        with self.lock:
            remaining = self.remaining()
            if remaining < min_remaining:
                return None, f'{remaining / 60:.0f} min of job time left (< {self.args.min_remaining_minutes} min)'
            deferred = 0
            for index in self.order:
                state, detail = self.state.classify(index)
                if state in ('complete', 'blocked'):
                    continue
                if state == 'failed':
                    # At most one attempt per index and job (a later job may retry, up to --max-attempts).
                    exhausted = detail >= self.args.max_attempts
                    if (exhausted and not self.args.retry_failed) or self.local_failures.get(index, 0) >= 1:
                        continue
                estimate = self._family_estimate(self.plan['configs'][index])
                if estimate is not None and remaining - self.args.kill_margin_seconds < estimate:
                    deferred += 1
                    continue
                if remaining <= self.args.kill_margin_seconds:
                    deferred += 1  # it would be stopped at once: no start-and-kill loop near the end of the job
                    continue
                claim = self.claims.try_claim(index, slot)
                if claim is not None:
                    return claim, None
            reason = 'no claimable index left'
            if deferred:
                reason += f' ({deferred} deferred: their family needs more time than this job has left)'
            return None, reason

    def run_slot(self, slot):
        try:
            while not self.stopping.is_set():
                claim, reason = self.claim_next(slot)
                if claim is None:
                    self._event('slot_idle', slot=slot, reason=reason)
                    break
                try:
                    self.process(slot, claim)
                finally:
                    released = self.claims.release(claim)
                    if not released:
                        self._event('claim_not_released', index=claim.index, slot=slot,
                                    reason='claim no longer held by this worker')
                    self.refresh_status()
        except Exception as exc:  # a crashed loop must be visible and must not hang the job
            self._event('slot_crashed', slot=slot, error=f'{type(exc).__name__}: {exc}')
            with self.lock:
                self.failed_here.append(('slot', slot))
        finally:
            with self.lock:
                self.slot_rows.pop(slot, None)

    # one tokenizer ------------------------------------------------------------------------------------------
    def process(self, slot, claim):
        index = claim.index
        config = self.plan['configs'][index]
        name = index_name(index)
        with self.lock:
            state, detail = self.state.classify(index)
        if state == 'complete':
            self._event('already_complete', index=index, slot=slot)
            return
        if state == 'blocked':
            self._event('index_blocked', index=index, reason=detail)
            return
        previous_failures = detail if state == 'failed' else 0
        if previous_failures >= self.args.max_attempts and not self.args.retry_failed:
            # e.g. its owner jobs died with it (counted at the takeover): left for the user
            self._event('index_exhausted', index=index, attempts=previous_failures,
                        max_attempts=self.args.max_attempts)
            return
        attempt = previous_failures + 1
        staging = claim.path / 'staging'
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        out = staging / 'out'
        config_file = staging / 'config.json'
        # scripts/modal_generate_tokenizer_pool.py:84 wrote json.dumps(config) the same way.
        config_file.write_text(json.dumps(config))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f'{name}.{self.claims.job or "local-" + str(os.getpid())}.s{slot}.a{attempt}.log'
        command = [sys.executable, str(self.worker_script), '--train-config', str(config_file),
                   '--recipe', str(self.recipe_path), '--destination', str(out)]
        env = dict(os.environ)
        env.update(RAYON_NUM_THREADS=str(self.threads), TOKENIZERS_PARALLELISM='true',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        label = f'{name} {family(config)} {config["vocab_size"] // 1000}k'
        estimate = self._family_estimate(config)
        started_at, started = utc_now(), time.time()
        self._row(slot, f'{label}: training', 'Training', eta=estimate / FAMILY_FIT_FACTOR if estimate else None,
                  index=index, attempt=attempt)
        self.refresh_status()
        self._event('tokenizer_start', index=index, slot=slot, family=family(config), vocab_size=config['vocab_size'],
                    attempt=attempt, log=str(log_path))
        outcome, returncode, why = self._run_child(command, env, log_path, claim, slot, label, started, estimate)
        finished_at = utc_now()
        elapsed = time.time() - started
        if outcome == 'interrupted':
            self._event('tokenizer_interrupted', index=index, slot=slot, reason=why, elapsed_seconds=round(elapsed, 1))
            return
        if outcome == 'ok':
            try:
                manifest = self._verified_output(out, config)
            except PoolError as exc:
                outcome, why = 'failed', str(exc)
        if outcome != 'ok':
            self._record_failure(index, config, attempt, why, returncode, log_path)
            return
        manifest['provenance'] = self._provenance(slot, attempt, previous_failures, started_at, finished_at)
        atomic_write_text(out / 'manifest.json', manifest_text(manifest))
        self._publish(claim, out, slot, manifest, elapsed)

    def _run_child(self, command, env, log_path, claim, slot, label, started, estimate):
        """Run the worker child; returns (outcome, returncode, reason) with outcome ok/failed/interrupted."""
        timeout = self.args.tokenizer_timeout_hours * 3600
        with open(log_path, 'ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(self.repo),
                                       start_new_session=True)
        last_beat = time.time()
        stop_reason = None
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            now = time.time()
            if now - last_beat >= self.args.heartbeat_seconds:
                self.claims.heartbeat(claim)
                phase = _child_phase(log_path)
                eta = max(0.0, estimate / FAMILY_FIT_FACTOR - (now - started)) if estimate else None
                self._row(slot, f'{label}: {phase}', 'Training', eta=eta, index=claim.index)
                last_beat = now
            if stop_reason is None:
                if self.hard_stop.is_set():
                    stop_reason = 'stop requested (second SIGTERM or SIGINT)'
                elif self.remaining() < self.args.kill_margin_seconds:
                    stop_reason = f'job ends in {self.remaining():.0f} s'
                elif now - started > timeout:
                    stop_reason = 'timeout'
                if stop_reason is not None:
                    _terminate(process)
                    returncode = process.returncode
                    break
            time.sleep(self.args.poll_seconds)
        if stop_reason == 'timeout':
            return 'failed', returncode, (f'timed out after {self.args.tokenizer_timeout_hours:g} h (the Modal pool used a '
                                          '6 h limit); if the pilots show longer runs, raise --tokenizer-timeout-hours')
        if stop_reason is not None:
            return 'interrupted', returncode, stop_reason
        if returncode == 0:
            return 'ok', 0, None
        if returncode < 0 and (self.stopping.is_set() or self.remaining() < 2 * self.args.kill_margin_seconds):
            return 'interrupted', returncode, f'killed by signal {-returncode} while the job was ending'
        tail = ' / '.join(line.strip()[:300] for line in tail_lines(log_path, 8) if line.strip())
        cause = f'killed by signal {-returncode}' if returncode < 0 else f'exit status {returncode}'
        if returncode == -signal.SIGKILL:
            cause += (' (SIGKILL: usually the job\'s memory limit; check sacct -j <job> -o JobID,MaxRSS,State, then '
                      'give this family its own jobs with --families ... --workers 1 or raise HYAK_MEM_POOL)')
        return 'failed', returncode, redact_text(f'tokenizer_pool.py {cause}: {tail}', self.cfg)

    def _verified_output(self, out, config):
        manifest_path, tokenizer_path = out / 'manifest.json', out / 'tokenizer.json'
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise PoolError(f'child wrote no readable manifest: {exc}') from exc
        problems = manifest_problems(manifest, config, self.plan, tokenizer_path, check_hash=True,
                                     require_provenance=False)
        if 'provenance' in manifest:
            problems.append('child manifest unexpectedly has a provenance block')
        extra = sorted(p.name for p in out.iterdir() if p.name not in ('manifest.json', 'tokenizer.json'))
        if extra:
            problems.append(f'unexpected files {extra}')
        if not problems:
            problems = tokenizer_json_problems(tokenizer_path.read_bytes(), config, config['vocab_size'])
        if problems:
            raise PoolError('output failed verification: ' + '; '.join(problems))
        return manifest

    def _provenance(self, slot, attempt, previous_failures, started_at, finished_at):
        # Original keys (scripts/modal_generate_tokenizer_pool.py:192-197): plan_sha256, worker_sha256,
        # rayon_threads, runtime_wheel_sha256, unigram_patches.  wheel_sha256 (the historical base wheel
        # of the plan) is not repeated: that wheel was not used here.
        provenance = dict(self.identity)
        provenance.update({
            'unigram_patches': list(UNIGRAM_PATCHES),
            'patch_order': list(PATCH_FILES),
            'upstream_tag': UPSTREAM_TAG,
            'runtime_wheel_name': self.wheel['name'] if self.wheel else None,
            'tokenizers_parallelism': 'true',
            'tokenizers_version': self.wheel['version'] if self.wheel else None,
            'python_version': platform.python_version(),
            'host': slurm.node_name(),
            'slurm_job_id': slurm.job_id(),
            'slurm_cluster': slurm.cluster_name(),
            'slurm_partition': slurm.partition(),
            'worker_slot': slot,
            'attempt': attempt,
            'previous_failed_attempts': previous_failures,
            'started_at': started_at,
            'finished_at': finished_at,
            'wrapper': 'hyak.pool.worker',
            'wrapper_sha256': self.wrapper_sha256,
        })
        return provenance

    def _publish(self, claim, out, slot, manifest, elapsed):
        index = claim.index
        final = self.tokenizers_dir / index_name(index)
        with self.lock:
            if not self.claims.held(claim):
                self._event('claim_lost', index=index, slot=slot, detail='result discarded; another worker owns the index')
                return
            if final.exists():
                state, detail = self.state.classify(index)
                if state == 'complete':
                    self._event('already_complete', index=index, slot=slot, detail='result discarded (completed manifest wins)')
                    return
                if (final / 'manifest.json').exists():
                    self._event('index_blocked', index=index, reason=f'existing manifest does not verify: {detail}')
                    return
                leftovers = sorted(p.name for p in final.iterdir()) if final.is_dir() else ['(not a directory)']
                if any(name != 'failure.json' for name in leftovers):
                    # Only a failure record of this worker may be moved aside; anything else is never touched.
                    self._event('index_blocked', index=index, reason=f'{final} holds {leftovers}; result discarded')
                    return
                os.rename(final, claim.path / 'previous')
            self.tokenizers_dir.mkdir(parents=True, exist_ok=True)
            os.rename(out, final)
            fsync_dir(self.tokenizers_dir)
            state, detail = self.state.classify(index)
            if state != 'complete':
                raise PoolError(f'published tokenizer {index} does not verify: {detail}')
        self._event('tokenizer_complete', index=index, slot=slot, family=family(manifest['config']),
                    train_seconds=round(manifest['train_seconds'], 1), elapsed_seconds=round(elapsed, 1),
                    cpu_seconds=round(manifest['cpu_seconds'], 1), peak_rss_gib=round(manifest['peak_rss_gib'], 3),
                    tokenizer_bytes=manifest['tokenizer_bytes'], tokenizer_sha256=manifest['tokenizer_sha256'])

    def _record_failure(self, index, config, attempt, error, returncode, log_path):
        folder = self.tokenizers_dir / index_name(index)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'failure.json'
        try:
            previous = read_json(path, {})
        except (OSError, ValueError):
            previous = {}
        history = list(previous.get('history', [])) if isinstance(previous, dict) else []
        entry = {'at_utc': utc_now(), 'attempt': attempt, 'error': error, 'returncode': returncode,
                 'job_id': slurm.job_id(), 'host': slurm.node_name(), 'log': str(log_path)}
        history.append(entry)
        # Original failure.json fields (scripts/modal_generate_tokenizer_pool.py:98-100) + attempts/history.
        record = {'config': config, 'error': error, 'at_utc': entry['at_utc'], 'attempts': attempt, 'history': history}
        with self.lock:
            atomic_write_text(path, json.dumps(record, indent=2) + '\n')
            self.failed_here.append(index)
            self.local_failures[index] = self.local_failures.get(index, 0) + 1
        self._event('tokenizer_failed', index=index, attempt=attempt, max_attempts=self.args.max_attempts,
                    error=error, log=str(log_path))

    def _record_owner_died(self, index, owner, reason):
        """A claim with a staging directory was taken over: its owner died while training (counts as an attempt)."""
        folder = self.tokenizers_dir / index_name(index)
        path = folder / 'failure.json'
        try:
            with self.lock:
                if (folder / 'manifest.json').exists() or (folder / 'tokenizer.json').exists():
                    return
                try:
                    previous = read_json(path, {})
                except (OSError, ValueError):
                    previous = {}
                previous = previous if isinstance(previous, dict) else {}
                attempts = int(previous.get('attempts', 0) or 0) + 1
                error = (f'owner job {owner.get("job_id")} on {owner.get("host")} (pid {owner.get("pid")}) ended while '
                         f'training this tokenizer ({reason}); the job was probably killed with it (memory limit?)')
                entry = {'at_utc': utc_now(), 'attempt': attempts, 'kind': 'owner_died', 'error': error,
                         'returncode': None, 'job_id': owner.get('job_id'), 'host': owner.get('host'), 'log': None}
                record = {'config': self.plan['configs'][index], 'error': error, 'at_utc': entry['at_utc'],
                          'attempts': attempts, 'history': list(previous.get('history', [])) + [entry]}
                folder.mkdir(parents=True, exist_ok=True)
                atomic_write_text(path, json.dumps(record, indent=2) + '\n')
            _log('tokenizer_owner_died', index=index, attempts=attempts, max_attempts=self.args.max_attempts,
                 error=error)
            self.status.event({'event': 'tokenizer_owner_died', 'index': index, 'attempts': attempts,
                               'max_attempts': self.args.max_attempts, 'error': error})
        except (OSError, ValueError) as exc:
            print(f'[hyak pool] WARNING: cannot record the dead owner of index {index}: {exc}', file=sys.stderr, flush=True)

    # main ----------------------------------------------------------------------------------------------
    def report(self):
        counts, failed, blocked, exhausted = self._scan()
        claims = []
        for index in range(self.count):
            info = self.claims.inspect(index)
            if info is not None:
                claims.append({'index': index, 'owner': (info['owner'] or {}).get('worker_id'),
                               'host': (info['owner'] or {}).get('host'),
                               'heartbeat_age_seconds': None if info['heartbeat_age'] is None else round(info['heartbeat_age']),
                               'stale': self.claims.stale_reason(info)})
        details = {i: self.state.classify(i)[1] for i in blocked}
        return {'pool_dir': str(self.pool_dir), 'plan_sha256': self.plan['plan_sha256'], 'production_plan': self.production,
                'counts': counts, 'total': self.count, 'claims': claims, 'failed_indices': failed,
                'exhausted_indices': exhausted, 'blocked': details, 'families': self.state.family_stats(),
                'identity': self.identity, 'identity_checked': self.identity_note is None,
                'identity_note': self.identity_note}

    def run(self) -> int:
        index_file = self.pool_dir / 'full-results.json'
        if index_file.exists():
            counts, _, _, _ = self._scan()
            if counts['complete'] != self.count:
                raise PoolError(f'{index_file} exists but only {counts["complete"]} of {self.count} tokenizers carry '
                                f'this worker\'s verified manifest: this is a brought-in pool (hyak/README.md section '
                                f'11) or an index written by another pool. Pool jobs must not run on it; nothing was '
                                f'changed. Remedy: use the pool as it is (collect/scorecard/training), or move '
                                f'{self.pool_dir} away to generate a new pool')
        self.claims.cleanup()
        self.tokenizers_dir.mkdir(parents=True, exist_ok=True)
        self._event('pool_job_start', workers=self.workers, threads=self.threads, plan_sha256=self.plan['plan_sha256'],
                    production_plan=self.production, order_head=self.order[:12], wheel=self.identity['runtime_wheel_sha256'])
        self.refresh_status()
        threads = [threading.Thread(target=self.run_slot, args=(slot,), name=f'pool-slot-{slot}', daemon=True)
                   for slot in range(self.workers)]
        for thread in threads:
            thread.start()
            time.sleep(self.args.stagger_seconds)
        last = time.time()
        while any(t.is_alive() for t in threads):
            for thread in threads:
                thread.join(timeout=1.0)
            if time.time() - last >= self.args.status_seconds:
                self.refresh_status()
                last = time.time()
        status = self.refresh_status(final=True)
        counts, _, blocked, exhausted = self._scan()
        self._event('pool_job_end', completed=status.get('completed'), failed_here=len(self.failed_here),
                    blocked=len(blocked), exhausted=len(exhausted))
        if counts['complete'] == self.count:
            return EXIT_OK  # complete pool: an attempt that failed here was retried successfully elsewhere
        if self.failed_here or blocked or exhausted:
            return EXIT_ATTENTION
        return EXIT_OK

    def install_signal_handlers(self):
        def on_term(signum, _frame):
            if self.stopping.is_set():
                self.hard_stop.set()
                _log('signal', signal=signum, action='stopping running tokenizers now')
            else:
                self.stopping.set()
                _log('signal', signal=signum, action='no new tokenizers; running ones continue until '
                                                     f'{self.args.kill_margin_seconds} s before the job ends')

        def on_int(signum, _frame):
            self.stopping.set()
            self.hard_stop.set()
            _log('signal', signal=signum, action='stopping now')

        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_int)


def _terminate(process, grace=30.0):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        process.wait()


def _child_phase(log_path) -> str:
    """Last stage printed by scripts/tokenizer_pool.py (training / corpus_consumed / complete)."""
    for line in reversed(tail_lines(log_path, 20)):
        try:
            stage = json.loads(line).get('stage')
        except (ValueError, AttributeError):
            continue
        if stage == 'corpus_consumed':
            return 'corpus read, learning vocabulary'
        if stage:
            return stage
    return 'starting'


def check_installed_wheel(provenance_path: Path, worker_script: Path) -> dict:
    """Prove the patched wheel of ``wheels/provenance.json`` is what this interpreter (and the child) imports.

    Checks: the wheel file hash, the pinned upstream revision and patch hashes, tokenizers 0.22.2,
    the patched BpeTrainer state (scripts/tokenizer_pool.py:156), the installed compiled extension is
    byte-identical to the wheel's, and the child (scripts/ first on sys.path) imports the same package.
    """
    remedy = f'Remedy: bash hyak/pool/build_wheel.sh install (then rerun)'
    try:
        prov = read_json(provenance_path)
    except (OSError, ValueError) as exc:
        raise PoolError(f'Cannot read {provenance_path} ({exc}). Remedy: build the wheel first '
                        '(bash hyak/pool/build_wheel.sh fetch, then build and install)') from exc
    wheel_path = provenance_path.parent / prov['wheel']['name']
    if not wheel_path.is_file() or file_sha256(wheel_path) != prov['wheel']['sha256']:
        raise PoolError(f'{wheel_path} is missing or differs from {provenance_path}. {remedy}')
    if prov.get('upstream', {}).get('revision') != UPSTREAM_REVISION or prov.get('patches') != {
            name: PATCH_SHA256[name] for name in PATCH_FILES}:
        raise PoolError(f'{provenance_path} records another upstream revision or patch set')
    try:
        import tokenizers
        from tokenizers import trainers
    except ImportError as exc:
        raise PoolError(f'tokenizers is not importable in {sys.executable}. {remedy}') from exc
    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise PoolError(f'tokenizers {tokenizers.__version__} is installed, expected {TOKENIZERS_VERSION}. {remedy}')
    state = json.loads(trainers.BpeTrainer(noise_per_million=5, noise_seed=7, random_from_all=True).__getstate__())['BpeTrainer']
    if (state.get('noise_per_million'), state.get('noise_seed'), state.get('random_from_all')) != (5, 7, True):
        raise PoolError(f'the patched noise trainer is not installed in {sys.executable}. {remedy}')
    package = Path(tokenizers.__file__).resolve().parent
    extension = package / Path(prov['wheel']['extension_member']).name
    if not extension.is_file() or file_sha256(extension) != prov['wheel']['extension_sha256']:
        raise PoolError(f'the installed extension {extension} is not the one in {wheel_path.name}. {remedy}')
    probe = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, sys.argv[1]); import tokenizers; '
                            'print(tokenizers.__file__)', str(worker_script.parent)],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0 or Path(probe.stdout.strip()).resolve().parent != package:
        raise PoolError(f'the worker child would import tokenizers from {probe.stdout.strip() or probe.stderr.strip()}, '
                        f'not {package}')
    return {'sha256': prov['wheel']['sha256'], 'name': prov['wheel']['name'], 'revision': UPSTREAM_REVISION,
            'patches': dict(prov['patches']), 'version': tokenizers.__version__,
            'extension_sha256': prov['wheel']['extension_sha256']}


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m hyak.pool.worker', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, help='parallel tokenizers in this job (default HYAK_POOL_WORKERS)')
    parser.add_argument('--threads', type=int, help='RAYON_NUM_THREADS per tokenizer (default HYAK_POOL_THREADS; must be 4)')
    parser.add_argument('--indices', nargs='+', help='only these indices, in this order (ints, a-b ranges, "pilots", '
                        '"binding" = the seven tokenizers at the native-length cap of the pool panel, '
                        '"pilots+binding")')
    parser.add_argument('--pilots-first', dest='pilots_first', action='store_true', default=True,
                        help='claim the 12 pilot indices before all others (default)')
    parser.add_argument('--no-pilots-first', dest='pilots_first', action='store_false')
    parser.add_argument('--pilots-only', action='store_true', help='only the 12 pilot indices')
    parser.add_argument('--families', nargs='+', metavar='ALGORITHM/PRETOKENIZER',
                        help='only these families, e.g. unigram/no-regex (to give a memory-heavy family '
                             'its own jobs with --workers 1)')
    parser.add_argument('--min-remaining-minutes', type=float, default=150.0,
                        help='do not start a tokenizer with less job time left (default 150)')
    parser.add_argument('--max-attempts', type=int, default=3, help='failed attempts after which an index is left alone')
    parser.add_argument('--retry-failed', action='store_true', help='retry exhausted indices once in this job')
    parser.add_argument('--tokenizer-timeout-hours', type=float, default=6.0,
                        help='kill a tokenizer after this long (Modal used 6 h; counts as a failure)')
    parser.add_argument('--claim-timeout-hours', type=float, default=6.0,
                        help='take over a claim without heartbeat for this long when its job state is unknown')
    parser.add_argument('--kill-margin-seconds', type=float, default=DEFAULT_KILL_MARGIN_SECONDS,
                        help='stop running tokenizers this long before the job ends')
    parser.add_argument('--report', action='store_true', help='print the pool state as JSON and exit (no claims)')
    parser.add_argument('--config', help='config file (default $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--plan', help=f'pool plan (default {PLAN_REL}; anything else needs --allow-test-plan)')
    parser.add_argument('--allow-test-plan', action='store_true', help='tests/local smoke runs only')
    parser.add_argument('--pool-dir', help='pool directory (default input/pool; required for a test plan)')
    parser.add_argument('--recipe', help='generation recipe (default input/pool-recipe.json)')
    parser.add_argument('--wheel-provenance', help='default $ANYTOK_ROOT/wheels/provenance.json')
    parser.add_argument('--monitor-dir', help='status directory (default <repo>/modal_results/hyak_tillicum/pool)')
    parser.add_argument('--worker-script', help=argparse.SUPPRESS)
    parser.add_argument('--no-wheel-check', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--poll-seconds', type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument('--heartbeat-seconds', type=float, default=60.0, help=argparse.SUPPRESS)
    parser.add_argument('--status-seconds', type=float, default=60.0, help=argparse.SUPPRESS)
    parser.add_argument('--stagger-seconds', type=float, default=1.0, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.max_attempts < 1:
            raise PoolError('--max-attempts must be >= 1')
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
        if args.tokenizer_timeout_hours <= 0 or args.claim_timeout_hours <= 0:
            raise PoolError('--tokenizer-timeout-hours and --claim-timeout-hours must be positive')
        if args.workers is not None and args.workers < 1:
            raise PoolError('--workers must be >= 1')
        worker = PoolWorker(args, cfg, paths)
        if args.report:
            print(json.dumps(worker.report(), indent=2, sort_keys=True))
            return EXIT_OK
        worker.install_signal_handlers()
        return worker.run()
    except (PoolError, ConfigError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return EXIT_INPUT


if __name__ == '__main__':
    sys.exit(main())
