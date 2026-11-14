"""One training segment of the uniform or RL 20k run: the Slurm port of the Modal launchers' ``train_segment``.

    python -m hyak.train.run_segment --run {uniform,rl} [--max-seconds S] [--config FILE]
    python -m hyak.train.run_segment --run {uniform,rl} --should-start [--max-seconds S]
    python -m hyak.train.run_segment --run {uniform,rl} --smoke [--max-seconds S]

(Line numbers of ``scripts/*.py`` below refer to commit c2b9c40, those that name the joint regret
launcher or its receipt to 3de64a5, where the launcher grew by a few lines.)

One process runs the pinned trainer once, in-process and in the main thread, exactly like
``scripts/modal_anytok_pool_20k.py:256-354`` and ``scripts/modal_anytok_pool_rl_20k.py:184-270``:

1. verify the staged code tree (``code/<run>``), the inputs (``input/manifest.json``; large files
   hashed once per run and cached by size+mtime; the trainer re-hashes the corpus itself; a
   test-scale corpus is refused) and ``input/model`` (exactly the three pinned SmolLM2 files, hashed
   in full at every segment against ``hyak.model.fetch_model.expected_files()``);
2. decide the mode from the run directory: fresh, resume, already complete, or dirty (no full
   restart: cleared when that is safe, otherwise refused), repair what a hard kill left behind,
   verify the newest full restart and move later saves to ``interrupted/<utc>/``
   (``hyak.train.retention.preflight``; a newest restart without manifest whose directory holds exactly
   a full restart's files is adopted, anything else stops with the remedy);
3. RL only: ``hyak.train.pairing.ensure`` (freezes or verifies the uniform files, D6);
4. import ``scripts.train_joint_smollm_input`` from the staged tree (``TrainerAdapter``), install the
   chunked hypernetwork forward when ``HYAK_HN_CHUNK_ROWS > 0`` (D2), wrap ``save_checkpoint`` with the
   run's retention hook, verify ``<run>/hyak_segment.json`` (chunk setting, code, initial
   hypernetwork, input paths must not change during a run);
5. arm a SIGTERM timer ``min(--max-seconds, seconds_remaining - HYAK_SEGMENT_MARGIN_MINUTES - 5 min)``
   (D14; 5 minutes before Slurm's backup ``--signal=B:TERM@margin``; nothing starts when less than
   20 minutes + margin remain); the timer waits while an RL weights-only save runs.  Set ``sys.argv``
   and call ``trainer.main()``; ``segment_end`` reports how the trainer was stopped and how long the
   stop took (``stop_source``, ``stop_seconds``);
6. post-checks (newest restart verified, ``state.pt`` step, RL policy counters, progress, the
   trainer's last event; an RL pause at a weights-only grid step above the newest restart is a
   warning; RL: the paired-protocol receipt of :func:`validate_rl_protocol`, a failure is exit 4) and,
   at update 20,000, the retention step of the last save (idempotent) and the grid audit writing
   ``final_audit.json``.

The RL run is the straight-BPB-reward counterpart of Qingyun's joint regret RL20k (D8, the user's
decision of 2026-11-14): its staged trainer is ``hyak/variants/anytok_pool_straight_rl_20k`` (the regret
trainer with ``bandit.update(pool_index, train_bpb)``), launched with the regret launcher's explicit
policy flags (entropy 0, learning rate 0.001, baseline decay 0.99).  Its corrected paired cuts change
the sanity examples, so its ``initial_eval.json`` is not expected to equal the uniform run's (which
keeps the original cuts): only the sanity tokenizer specs are compared.

``hyak_segment.json`` also records the model hashes (compared on resume and with the uniform run),
the software environment (Python, torch, CUDA, transformers, the hash of
``$ANYTOK_ROOT/env/train.freeze.txt``; a change between segments is a loud warning, not a refusal),
and accepted chunk-setting changes.  A run that starts from scratch moves an earlier attempt's
monitor files (other jobs' train logs, policy snapshots, status, events) to ``<monitor>/previous-<utc>/``.
The run's wrapper lock is waited for up to 10 minutes (then exit 75).

The uniform run starts with a one-update bootstrap segment (``--max-updates-this-run 1`` in place
of ``--stop-at-update 20000``, D6); after a successful uniform segment the uniform files are frozen
for the RL run (``pairing.ensure(role='uniform')``).

Exit status (project-wide contract):

* 0  a clean pause, completion, an already complete run, too little time left to start, a stop signal
  before the trainer started, an interrupted final audit;
* 1  never deliberate: an unexpected exception (inside the pinned trainer, torch, transformers, the
  Rust helper or this wrapper); the full traceback is on stderr (the Slurm log), the last stderr line
  is ``ERROR: <type>: <message>``;
* 2  usage, configuration, input or environment error (missing staged code or inputs, a test-scale
  corpus, a trainer argument check) -- the message names the remedy;
* 3  a protocol stop that needs the user's decision (a checkpoint that cannot be adopted, a run
  directory that can neither resume nor start, a pairing problem, settings that changed mid-run);
* 4  a check failed (staged code or inputs differ from their pins or manifests, a checkpoint failed
  verification, the trainer ended without the expected checkpoint or event);
* 75 not now: the RL run waits for the uniform bootstrap (``input/pairing.json``), or the run's lock
  was still held after 10 minutes (a predecessor job being killed, a repair command).

``--should-start`` prints its answer as JSON and exits 0 when a segment should start now and 75 when
not (complete, too little time, or RL before the uniform bootstrap); errors as above.

``--accept-chunk-change REASON`` accepts, once and on the record, a changed chunked-forward setting
(``HYAK_HN_CHUNK_ROWS`` or ``hyak/memory/chunked_hypernet.py``) in the middle of a run (for example
after an out-of-memory failure late in a run): the change and the reason are written to the events
and to ``hyak_segment.json``; for the RL run the difference from the uniform run is accepted too.
Without it such a change is refused (exit 3).

Output: JSON lines on stdout (the trainer's own lines plus this wrapper's ``verified_inputs``,
``verified_restart``, ``committed_checkpoint``, ``compacted_checkpoints``, ``segment_start``,
``segment_end`` ...; redirect stdout to ``<monitor>/<job>-train.log`` for the learning panel) and the
same wrapper events, timestamped, in ``<monitor>/events.jsonl``.  Human-readable problems go to stderr.

``--smoke`` (in ``$ANYTOK_RUNS/smoke/<run>``, wiped first) runs the launchers' pre-flight sequence as
two child processes: RL 2 updates, then a resume to 3 (counters must equal 3,
``scripts/run_anytok_pool_rl_20k.py:159-170``, and the paired-protocol receipt of each phase and of
step 3 must pass); uniform the bootstrap, then a resume to 3.  Same ``--save-every`` / ``--eval-every``
as production; no ``--protected-steps`` (they may not exceed the stop).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import faulthandler
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from hyak.code import pinned
from hyak.common import slurm
from hyak.common.config import ConfigError, int_value, load_config, repo_root, stage_value
from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now
from hyak.common.paths import (FINAL_STEP, RUN_RL, RUN_UNIFORM, STAGE_SMOKE_TRAIN, STAGE_TRAIN_RL,
                               STAGE_TRAIN_UNIFORM, Paths, validate_run)
from hyak.common.status import StatusWriter
from hyak.train import pairing, retention

# --- pinned facts (DESIGN non-negotiable 2; notes/modal-launchers.md 3.1) -------------------------------
#: The uniform trainer is the original one; the RL trainer is the straight-BPB-reward variant of Qingyun's
#: joint regret RL20k trainer (hyak/variants/anytok_pool_straight_rl_20k, D8).
TRAINER_SHA256 = {RUN_UNIFORM: 'c6406d263fb9cbc8638492d8c77327ce249e215eb74dc43e10335bc95a11ed52',
                  RUN_RL: '8dd3b8570dd6545123ab1aee5d338a8957d72ece26e48a759196d2619f43ff51'}
#: Repository source of each run's staged trainer.
TRAINER_SOURCE = {RUN_UNIFORM: 'runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py',
                  RUN_RL: 'hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py'}
POLICY_SHA256 = '6dbfcbf8c6552ef98b56de08c09ae59bd35a2d7d1a75e7e295cfcf5346a90b8c'
#: hf_hypernet/paired_reward.py of runtime/anytok_pool_regret_rl_20k (3de64a5), imported by the RL trainer.
PAIRED_REWARD_SHA256 = '406cfc58bf1c46cef0dc6785bbb313ca5797c0d35bf6cd786b3115a92e5523c1'
PAIRED_REWARD_SOURCE = 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py'
RECIPE_SHA256 = 'b3bc193333aa13d3a513f1c8978805835f837190cd0e5991742c55e9a0752cfe'
PLAN_FILE_SHA256 = '4bbd89c6d40993a036ec33072a8300acf9f50a222e8b258c9ec2e0b24fe43b8e'
#: D1: size of the random-init hypernetwork's model.safetensors (the original warmup's size).
HYPERNET_INIT_BYTES = 1_376_746_800
SO_REL = 'rust_utils/target/release/librust_utils.so'
TRAINER_REL = retention.TRAINER_REL
POLICY_REL = retention.POLICY_REL
PAIRED_REWARD_REL = 'hf_hypernet/paired_reward.py'
POOL_SIZE = 1000
CORPUS_BUCKETS = 23

#: D14: never start a segment with less than this (plus the margin) left in the job.
MIN_START_SECONDS = 20 * 60
#: The launchers' lower bound for max_seconds (modal_anytok_pool_20k.py:260).
MIN_MAX_SECONDS = 60
#: Inputs at least this large are hashed once per run and cached by (path, size, mtime, inode).
LARGE_FILE_BYTES = 16 * 1024 * 1024
#: D6 bootstrap length.
BOOTSTRAP_UPDATES = 1
#: Original trainer arguments (modal_anytok_pool_20k.py:330-335, modal_anytok_pool_rl_20k.py:254-256).
MICROBATCH = '32'
SAVE_EVERY = '500'
EVAL_EVERY = '500'
SEED = '42'
PROTECTED_STEPS = [str(step) for step in range(500, FINAL_STEP + 1, 500)]
#: RL policy flags passed explicitly, as the joint regret launcher does (scripts/modal_anytok_pool_rl_20k.py:262-264
#: with POLICY of scripts/anytok_pool_rl_config.py and ANYTOK_RL_EXPERIMENT=regret: entropy 0).
POLICY_ENTROPY = '0'
POLICY_LR = '0.001'
POLICY_BASELINE_DECAY = '0.99'
#: The RL receipt after every segment and smoke (the regret launcher's checks, modal_anytok_pool_rl_20k.py:280-289,
#: plus the straight reward of D8).
RL_POLICY_REWARD = 'paired_pre_model_update_transferred_bpb'
RL_POLICY_SETTINGS = {'entropy_weight': 0.0, 'learning_rate': 0.001, 'baseline_decay': 0.99, 'seed': 43}
PAIRED_VALID_EXAMPLES = 128
PAIRED_MAX_POSITIONS = 128
PAIRED_GAP_TOLERANCE = 1e-10
POLICY_REWARD_TOLERANCE = 1e-12
#: Smoke: final step and the step after the first child (uniform: the one-update bootstrap).
SMOKE_FINAL_STEP = 3
SMOKE_FIRST_STEP = {RUN_UNIFORM: 1, RUN_RL: 2}
EXPECTED_GPU = 'H200'
STAGES = {RUN_UNIFORM: STAGE_TRAIN_UNIFORM, RUN_RL: STAGE_TRAIN_RL}
SEGMENT_RECORD_VERSION = 1
#: Chunked-forward settings that identify the computation (hyak.memory.chunked_hypernet.settings()
#: says to compare these on resume).
CHUNK_IDENTITY_KEYS = ('chunk_rows', 'use_checkpoint', 'module_sha256', 'algorithm_version')
#: hyak_segment.json fields that must not change while a run is in progress.
COMPARED_RECORD_FIELDS = ('run', 'chunk_identity', 'code_tree_sha256', 'init_sha256', 'fixed_paths',
                          'model_sha256')
#: Recorded per segment; a change between segments is a loud warning, not a refusal.
ENVIRONMENT_FIELDS = ('python', 'torch', 'cuda', 'transformers', 'env_freeze_sha256')
#: Environment variables recorded in the segment_start event (numerics, allocator, offline mode).
RECORDED_ENV_VARS = ('PYTORCH_CUDA_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'PYTHONHASHSEED',
                     'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'PYTHONOPTIMIZE')
#: The in-process stop timer fires this much earlier than the margin, i.e. before Slurm's backup
#: ``--signal=B:TERM@<margin>`` (which Slurm may send up to 60 s early).
STOP_TIMER_LEAD_SECONDS = 300
#: The stop timer waits at most this long for a running RL weights-only save before it sends SIGTERM.
STOP_DEFER_MAX_SECONDS = 900
#: A segment waits this long for the run's wrapper lock (a predecessor job still being killed, a
#: repair command), polling every LOCK_POLL_SECONDS; then it exits 75 (not now).
LOCK_WAIT_SECONDS = 600
LOCK_POLL_SECONDS = 10.0
#: Marker the smoke writes into its directory after smoke_verified (the monitor then reports complete).
SMOKE_VERIFIED = 'smoke_verified.json'

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PROTOCOL = 3
EXIT_CHECK = 4
EXIT_NOT_NOW = 75
#: Outcomes of run_segment that the command line reports with exit 75.
NOT_NOW_OUTCOMES = ('uniform_bootstrap_pending', 'run_locked')


class SegmentError(RuntimeError):
    """A segment cannot start or did not end correctly; the message says why and what to do.

    ``exit_code`` follows the project contract: 2 usage/input/environment, 3 a protocol stop that
    needs the user's decision, 4 a failed check (the default).
    """

    def __init__(self, message, *, exit_code: int = EXIT_CHECK):
        super().__init__(message)
        self.exit_code = exit_code


# --- trainer adapter ----------------------------------------------------------------------------------------
class TrainerAdapter:
    """Everything that touches the pinned trainer (tests substitute a fake trainer file or subclass).

    ``load()`` mirrors ``sys.path.insert(0, str(CODE)); import scripts.train_joint_smollm_input as
    trainer`` (modal_anytok_pool_20k.py:267-268; modal_anytok_pool_rl_20k.py:190-191) and checks that
    the module, its ``REPO`` and ``hf_hypernet`` all come from the staged tree.
    """

    MODULE = 'scripts.train_joint_smollm_input'

    def __init__(self, code_dir):
        self.code_dir = Path(code_dir)
        self.module = None

    def load(self):
        if self.MODULE in sys.modules:
            raise SegmentError('the trainer module is already imported in this process; one process runs '
                               'exactly one trainer invocation (start a new process)', exit_code=EXIT_USAGE)
        trainer_file = self.code_dir / TRAINER_REL
        sys.path.insert(0, str(self.code_dir))
        module = importlib.import_module(self.MODULE)
        if Path(module.__file__).resolve() != trainer_file.resolve():
            raise SegmentError(f'imported {module.__file__}, not the staged trainer {trainer_file}; another '
                               '"scripts" package shadows the staged tree on sys.path', exit_code=EXIT_USAGE)
        repo = getattr(module, 'REPO', None)
        if repo is None or Path(repo).resolve() != self.code_dir.resolve():
            raise SegmentError(f'trainer REPO is {repo}, expected the staged tree {self.code_dir}', exit_code=EXIT_USAGE)
        package = sys.modules.get('hf_hypernet')
        if package is not None and Path(package.__file__).resolve().parent != (self.code_dir / 'hf_hypernet').resolve():
            raise SegmentError(f'hf_hypernet was imported from {package.__file__}, not from {self.code_dir}',
                               exit_code=EXIT_USAGE)
        for name in ('main', 'save_checkpoint', 'ZettHypernet'):
            if not hasattr(module, name):
                raise SegmentError(f'the staged trainer has no {name}; is {trainer_file} the pinned file?')
        self.module = module
        return module

    def install_memory_patch(self, chunk_rows: int) -> dict:
        """D2: ``hyak.memory.chunked_hypernet.install(trainer.ZettHypernet, chunk_rows)`` when chunk_rows > 0."""
        if chunk_rows <= 0:
            return {'chunk_rows': 0, 'module_sha256': None, 'installed': False}
        try:
            chunked = importlib.import_module('hyak.memory.chunked_hypernet')
        except ImportError as exc:
            raise SegmentError(f'HYAK_HN_CHUNK_ROWS={chunk_rows} needs hyak.memory.chunked_hypernet, which cannot be '
                               f'imported ({exc}). Remedy: update the checkout; setting HYAK_HN_CHUNK_ROWS=0 is only '
                               'viable with about 200 GiB of GPU memory', exit_code=EXIT_USAGE) from exc
        chunked.install(self.module.ZettHypernet, chunk_rows)
        if not chunked.is_installed(self.module.ZettHypernet):
            raise SegmentError('hyak.memory.chunked_hypernet.install did not patch the trainer ZettHypernet')
        settings = dict(chunked.settings())
        if settings.get('chunk_rows') != chunk_rows or not settings.get('module_sha256'):
            raise SegmentError(f'unexpected chunked forward settings {settings} (wanted chunk_rows={chunk_rows})')
        settings['installed'] = True
        return settings

    def wrap_save(self, make_hook) -> None:
        """Replace the module-global ``save_checkpoint``; ``main()`` looks the name up at call time."""
        original = self.module.save_checkpoint
        self.module.save_checkpoint = make_hook(original)

    def run_main(self, argv) -> None:
        previous = sys.argv
        sys.argv = list(argv)
        try:
            self.module.main()
        finally:
            sys.argv = previous

    def gpu_info(self) -> dict:
        try:
            import torch
        except ImportError:
            return {'name': None, 'count': 0, 'torch': None, 'cuda': None}
        info = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'name': None, 'count': 0}
        if torch.cuda.is_available():
            info.update(name=torch.cuda.get_device_name(0), count=torch.cuda.device_count())
        return info

    def load_state(self, path):
        import torch
        return torch.load(path, map_location='cpu', weights_only=True)


# --- events ------------------------------------------------------------------------------------------------
class Emitter:
    """Wrapper events: one JSON line on stdout (like the launchers' prints) + a timestamped monitor event."""

    def __init__(self, stage: str, run: str, cfg: dict):
        self.run = run
        self.writer = StatusWriter(stage, None, 'steps', 'step', cfg=cfg)

    def __call__(self, name: str, **fields):
        fields = _jsonable(fields)
        print(json.dumps({'stage': name, 'run': self.run, **fields}, sort_keys=True), flush=True)
        try:
            self.writer.event({'event': name, 'run': self.run, **fields})
        except ValueError as exc:  # never lose the event because of one odd value
            self.writer.event({'event': name, 'run': self.run, 'unserialisable': str(exc)})

    def warn(self, message: str, **fields):
        print(f'[hyak run_segment] WARNING: {message}', file=sys.stderr, flush=True)
        self('warning', message=message, **fields)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ('inf' if value > 0 else '-inf')
    return value


# --- code and input verification --------------------------------------------------------------------------
def expected_code_files(run: str, repo=None) -> dict:
    """``{relative path: sha256}`` of the staged tree of ``run``, from the pinned files in this checkout.

    Layout of the Modal images (modal_anytok_pool_20k.py:43-48; the joint regret image of
    modal_anytok_pool_rl_20k.py:56-67): ``hf_hypernet/*`` of ``runtime/anytok_pool_20k`` (+ RL
    ``settings_policy.py`` of ``runtime/anytok_pool_settings_rl_20k`` and ``paired_reward.py`` of
    ``runtime/anytok_pool_regret_rl_20k``), the run's trainer at ``scripts/`` (RL: the straight-reward
    variant, :data:`TRAINER_SOURCE`), and the committed ``librust_utils.so``.
    """
    repo = Path(repo) if repo is not None else repo_root()
    base = repo / 'runtime/anytok_pool_20k'
    files = {f'hf_hypernet/{p.name}': sha256_file(p) for p in sorted((base / 'hf_hypernet').glob('*.py'))}
    trainer = repo / TRAINER_SOURCE[run]
    files[TRAINER_REL] = sha256_file(trainer)
    if files[TRAINER_REL] != TRAINER_SHA256[run]:
        raise SegmentError(f'{trainer} is not the pinned {run} trainer (SHA-256 {files[TRAINER_REL]}); restore it '
                           '(python -m hyak.code.pinned --check names the remedy)')
    if run == RUN_RL:
        files[POLICY_REL] = sha256_file(repo / 'runtime/anytok_pool_settings_rl_20k' / POLICY_REL)
        if files[POLICY_REL] != POLICY_SHA256:
            raise SegmentError('runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py is not the pinned policy')
        files[PAIRED_REWARD_REL] = sha256_file(repo / PAIRED_REWARD_SOURCE)
        if files[PAIRED_REWARD_REL] != PAIRED_REWARD_SHA256:
            raise SegmentError(f'{PAIRED_REWARD_SOURCE} is not the pinned paired-reward helper')
    files[SO_REL] = sha256_file(base / SO_REL)
    return files


def _manifest_records(node, prefix=()):
    """All ``(key path, record)`` pairs of a JSON tree whose record has a ``sha256``."""
    if isinstance(node, dict):
        if isinstance(node.get('sha256'), str):
            yield prefix, node
        for key, value in node.items():
            yield from _manifest_records(value, prefix + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _manifest_records(value, prefix + (str(index),))


def _rebuilt_library_allowed(code_manifest: Path, run: str, digest: str) -> bool:
    """Whether code_manifest.json records ``digest`` for this run's librust_utils.so (fallback rebuild)."""
    try:
        data = read_json(code_manifest)
    except (OSError, ValueError):
        return False
    for keys, record in _manifest_records(data):
        joined = '/'.join(keys)
        if record.get('sha256') == digest and joined.endswith('librust_utils.so') and run in keys + tuple(joined.split('/')):
            return True
    return False


def verify_code(code_dir, run: str, code_manifest, expected=None) -> dict:
    """Check that ``code_dir`` is exactly the staged union layout with the pinned bytes.

    Python sources must equal the pinned snapshot files; ``librust_utils.so`` must equal the committed
    binary or the rebuild that ``code_manifest.json`` records for this run.  ``__pycache__`` is
    ignored; symlinks and extra or missing files are errors.  Returns the hashes recorded in
    ``hyak_segment.json`` (``code_tree_sha256`` over every file of the tree).
    """
    code_dir = Path(code_dir)
    code_manifest = Path(code_manifest)
    if not code_dir.is_dir():
        raise SegmentError(f'staged code tree {code_dir} is missing. Remedy: python -m hyak.code.stage_code',
                           exit_code=EXIT_USAGE)
    if not code_manifest.is_file():
        raise SegmentError(f'{code_manifest} is missing. Remedy: python -m hyak.code.stage_code', exit_code=EXIT_USAGE)
    expected = expected_code_files(run) if expected is None else dict(expected)
    tree = {}
    for path in sorted(code_dir.rglob('*')):
        if '__pycache__' in path.relative_to(code_dir).parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise SegmentError(f'{path} is a symlink; the staged tree must hold real copies (the trainer resolves '
                               'its own path). Remedy: python -m hyak.code.stage_code --force')
        if path.is_file():
            tree[path.relative_to(code_dir).as_posix()] = sha256_file(path)
    missing = sorted(set(expected) - set(tree))
    extra = sorted(set(tree) - set(expected))
    if missing or extra:
        raise SegmentError(f'staged code tree {code_dir} differs from the pinned layout: missing {missing}, '
                           f'unexpected {extra}. Remedy: python -m hyak.code.stage_code --verify')
    changed = sorted(rel for rel in expected if rel != SO_REL and tree[rel] != expected[rel])
    if changed:
        raise SegmentError(f'staged files differ from the pinned snapshots: {changed}. The pinned trainers must stay '
                           'byte-identical. Remedy: python -m hyak.code.stage_code --verify')
    rebuilt = tree[SO_REL] != expected[SO_REL]
    if rebuilt and not _rebuilt_library_allowed(code_manifest, run, tree[SO_REL]):
        raise SegmentError(f'{code_dir / SO_REL} is neither the committed binary nor a rebuild recorded in '
                           f'{code_manifest}. Remedy: hyak/env/build_rust_utils.sh, then stage_code --verify')
    digest = hashlib.sha256(''.join(f'{rel}\0{sha}\n' for rel, sha in sorted(tree.items())).encode()).hexdigest()
    return {'code_tree_sha256': digest, 'code_manifest_sha256': sha256_file(code_manifest),
            'trainer_sha256': tree[TRAINER_REL], 'policy_sha256': tree.get(POLICY_REL),
            'hf_hypernet_sha256': {rel: sha for rel, sha in tree.items() if rel.startswith('hf_hypernet/')},
            'rust_utils_sha256': tree[SO_REL], 'rust_utils_rebuilt': rebuilt, 'files': len(tree)}


class HashCache:
    """SHA-256 of large files, cached per run by (real path, size, mtime_ns, inode, device)."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            data = read_json(self.path, {})
        except ValueError:
            data = {}
        self.entries = data.get('entries', {}) if isinstance(data, dict) else {}
        self.dirty = False

    def sha256(self, path) -> str:
        real = os.path.realpath(path)
        stat = os.stat(real)
        if stat.st_size < LARGE_FILE_BYTES:
            return sha256_file(real)
        key = [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev]
        cached = self.entries.get(real)
        if isinstance(cached, dict) and cached.get('key') == key:
            return cached['sha256']
        digest = sha256_file(real)
        after = os.stat(real)
        if [after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev] != key:
            raise SegmentError(f'{path} changed while it was being hashed; no job may write inputs during training')
        self.entries[real] = {'key': key, 'sha256': digest, 'at_utc': utc_now()}
        self.dirty = True
        return digest

    def save(self):
        if self.dirty:
            atomic_write_json(self.path, {'version': 1, 'entries': self.entries})
            self.dirty = False


def _need(path: Path, what: str, remedy: str):
    if not path.is_file():
        raise SegmentError(f'{what} {path} is missing. Remedy: {remedy}', exit_code=EXIT_USAGE)


def verify_model(model_dir, expected=None) -> dict:
    """``input/model`` holds exactly the pinned SmolLM2 files; returns ``{name: sha256}`` (hashed in full).

    The launcher hashed ``model/*`` at every segment (modal_anytok_pool_20k.py:94-104); here the pins
    are ``hyak.model.fetch_model.expected_files()`` (plans/zett-modal-20k-inputs.json).  ``expected``
    (``{name: {"bytes", "sha256"}}``) is a test parameter only.
    """
    from hyak.model import fetch_model
    remedy = 'python -m hyak.model.fetch_model (it reports and never overwrites wrong files)'
    if expected is None:
        try:
            expected = fetch_model.expected_files()
        except (fetch_model.FetchError, OSError, ValueError, KeyError) as exc:
            raise SegmentError(f'cannot read the pinned model hashes: {exc}', exit_code=EXIT_USAGE) from exc
    model_dir = Path(model_dir)
    problems = fetch_model.verify_model_dir(model_dir, expected, check_hash=False)
    if problems:
        missing = any(text.endswith(': missing') or text.endswith('directory missing') for text in problems)
        raise SegmentError(f'{model_dir} is not the pinned model: ' + '; '.join(problems) + f'. Remedy: {remedy}',
                           exit_code=EXIT_USAGE if missing else EXIT_CHECK)
    digests = {}
    for name in sorted(expected):
        digests[name] = sha256_file(model_dir / name)
        if digests[name] != expected[name]['sha256']:
            raise SegmentError(f'{model_dir / name} has SHA-256 {digests[name]}, the pin is {expected[name]["sha256"]}. '
                               f'Both runs must start from the pinned SmolLM2 files. Remedy: {remedy}')
    return digests


def verify_inputs(paths: Paths, run: str, cache: HashCache, *, init_bytes: int = HYPERNET_INIT_BYTES,
                  plan_file_sha256: str = PLAN_FILE_SHA256, model_expected=None,
                  allow_test_scale: bool = False) -> dict:
    """The trainer's inputs exist, are the pinned recipe/plan/model and match ``input/manifest.json``.

    Port of the launchers' required-file check and ``verify_input_files``
    (modal_anytok_pool_20k.py:94-104, 270-276) plus the RL pool check
    (modal_anytok_pool_rl_20k.py:219-220: a complete 1,000-tokenizer pool).  The model directory is
    verified against the pins (:func:`verify_model`).  A manifest whose
    ``provenance.corpus.test_scale`` is true is refused.  Each model file is hashed once: the digests
    of :func:`verify_model` serve the ``model/*`` entries of the manifest.  Returns ``{"file_count",
    "bytes", "manifest_sha256", "init_sha256", "model_sha256", "corpus_test_scale", "init_recipe",
    "init_recipe_warning"}`` (see :func:`read_init_recipe`).  ``init_bytes``,
    ``plan_file_sha256``, ``model_expected`` and ``allow_test_scale`` are test parameters only
    (miniature worlds); production uses the pins and never passes them.
    """
    stage = 'python -m hyak.corpus.stage_inputs (see hyak/README.md)'
    _need(paths.input_manifest, 'input manifest', stage)
    _need(paths.recipe, 'recipe', stage)
    _need(paths.input_corpus_manifest, 'corpus manifest', stage)
    model_sha = verify_model(paths.model_dir, model_expected)
    for name in ('config.json', 'model.safetensors'):
        _need(paths.hypernet_init / name, 'initial hypernetwork file', 'python -m hyak.model.init_hypernet')
    _need(paths.pool_plan, 'pool plan', 'python -m hyak.pool.collect')
    _need(paths.pool_results, 'pool index', 'python -m hyak.pool.collect')
    if sha256_file(paths.recipe) != RECIPE_SHA256:
        raise SegmentError(f'{paths.recipe} is not data/balanced-full-v3.json (SHA-256 differs). Remedy: {stage}')
    if sha256_file(paths.pool_plan) != plan_file_sha256:
        raise SegmentError(f'{paths.pool_plan} is not plans/tokenizer-pool-1000-v1.json. Remedy: python -m hyak.pool.collect')
    recipe = read_json(paths.recipe)
    names = [bucket['name'] for bucket in recipe['buckets']]
    if len(names) != CORPUS_BUCKETS:
        raise SegmentError(f'{paths.recipe} has {len(names)} buckets, expected {CORPUS_BUCKETS}')
    corpus = read_json(paths.input_corpus_manifest)
    if not isinstance(corpus, dict) or sorted(corpus) != sorted(names):
        raise SegmentError(f'{paths.input_corpus_manifest} must have exactly the 23 bucket keys of the recipe '
                           '(the trainer checks len == 23). Remedy: ' + stage)
    for name in names:
        train = paths.input_corpus_train(name)
        entry = corpus[name] if isinstance(corpus[name], dict) else {}
        if not train.is_file() or train.stat().st_size != entry.get('compressed_bytes'):
            raise SegmentError(f'{train} is missing or not the size recorded in the corpus manifest. Remedy: {stage}')
        _need(paths.input_eval_file(name), 'held-out file', stage)
    plan_sha = read_json(paths.pool_plan)['plan_sha256']
    index = read_json(paths.pool_results)
    if (not isinstance(index, dict) or index.get('completed') != POOL_SIZE
            or len(index.get('tokenizers') or []) != POOL_SIZE or index.get('plan_sha256') != plan_sha):
        raise SegmentError(f'{paths.pool_results} is not the complete {POOL_SIZE}-tokenizer pool of the plan. '
                           'Remedy: finish the pool jobs, then python -m hyak.pool.collect', exit_code=EXIT_USAGE)
    init_model = paths.hypernet_init / 'model.safetensors'
    if init_model.stat().st_size != init_bytes:
        raise SegmentError(f'{init_model} has {init_model.stat().st_size} bytes, expected {init_bytes} (D1: the '
                           'random-init hypernetwork before byte-fallback expansion). Remedy: python -m hyak.model.init_hypernet')
    manifest = read_json(paths.input_manifest)
    provenance = manifest.get('provenance') if isinstance(manifest, dict) else None
    corpus_info = provenance.get('corpus') if isinstance(provenance, dict) else None
    test_scale = bool(isinstance(corpus_info, dict) and corpus_info.get('test_scale'))
    if test_scale and not allow_test_scale:
        raise SegmentError(f'{paths.input_manifest} marks a test-scale corpus (provenance.corpus.test_scale): the '
                           'runs must train on the production corpus. Remedy: build the production corpus '
                           '(python -m hyak.corpus.build) and stage it with python -m hyak.corpus.stage_inputs',
                           exit_code=EXIT_USAGE)
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise SegmentError(f'{paths.input_manifest} must be {{"files": {{"<path under input/>": {{"bytes", "sha256"}}}}}}')
    # verify_model has just hashed every model file in full; a production manifest lists them again
    # (model/config.json, model/model.safetensors, model/tokenizer.json), so reuse those digests
    # instead of reading the 3.4 GB of weights a second time
    model_digests = {os.path.realpath(paths.model_dir / name): digest for name, digest in model_sha.items()}
    total = 0
    for name, expected in sorted(files.items()):
        path = paths.input / name
        if not path.is_file() or path.stat().st_size != expected.get('bytes'):
            raise SegmentError(f'Missing or wrong-sized input: {name}. Remedy: {stage}')
        digest = model_digests.get(os.path.realpath(path))
        if digest is None:
            digest = cache.sha256(path)
        if digest != expected.get('sha256'):
            raise SegmentError(f'Checksum mismatch: {name} differs from {paths.input_manifest}. Inputs must not change '
                               'after staging (both runs and their resumes depend on them)')
        total += expected['bytes']
    init_sha = {name: cache.sha256(paths.hypernet_init / name) for name in ('config.json', 'model.safetensors')}
    cache.save()
    init_recipe, init_recipe_warning = read_init_recipe(paths.hypernet_init_manifest, init_sha)
    return {'file_count': len(files), 'bytes': total, 'manifest_sha256': sha256_file(paths.input_manifest),
            'init_sha256': init_sha, 'model_sha256': model_sha, 'corpus_test_scale': test_scale,
            'init_recipe': init_recipe, 'init_recipe_warning': init_recipe_warning}


#: Keys of ``init_manifest.json['recipe']`` copied into the segment record as ``init_recipe``.
INIT_RECIPE_KEYS = ('option', 'seed', 'prior_bias_std', 'calibration', 'rescalers_from_fixture')


def read_init_recipe(manifest_path, init_sha: dict):
    """The recipe of the initial hypernetwork from its ``init_manifest.json``; never raises.

    Returns ``(recipe, None)`` when the manifest's ``outputs`` record the same SHA-256 for
    ``config.json`` and ``model.safetensors`` as ``init_sha`` (just computed), otherwise
    ``(None, warning text)``.  Information only: the initial weights themselves are compared through
    ``init_sha256``; the recipe says how they were built (option (a)/(b)/(c), seed, prior bias).
    """
    manifest_path = Path(manifest_path)
    try:
        if not manifest_path.is_file():
            return None, (f'{manifest_path} is missing, so the recipe of the initial hypernetwork (option, seed, '
                          'prior bias) is not recorded for this run; hyak.model.init_hypernet writes it')
        manifest = read_json(manifest_path)
        outputs = manifest.get('outputs') if isinstance(manifest, dict) else None
        recipe = manifest.get('recipe') if isinstance(manifest, dict) else None
        if not isinstance(outputs, dict) or not isinstance(recipe, dict):
            return None, f'{manifest_path} has no outputs/recipe; the init recipe is not recorded for this run'
        differing = [name for name in ('config.json', 'model.safetensors')
                     if not isinstance(outputs.get(name), dict) or outputs[name].get('sha256') != init_sha.get(name)]
        if differing:
            return None, (f'{manifest_path} does not describe the files next to it ({", ".join(differing)}: '
                          'SHA-256 differs); the init recipe is not recorded for this run')
        return {key: recipe.get(key) for key in INIT_RECIPE_KEYS}, None
    except Exception as exc:  # noqa: BLE001 - information only, must never stop a segment
        return None, f'{manifest_path} could not be read ({type(exc).__name__}: {exc}); the init recipe is not recorded'


# --- argv ----------------------------------------------------------------------------------------------------
def uniform_argv(paths: Paths, code_dir, output, *, resume: bool, stop_at=None, max_updates=None,
                 protected: bool = True) -> list:
    """Uniform trainer argv (modal_anytok_pool_20k.py:321-337).

    D6 bootstrap: ``max_updates=1`` puts ``--max-updates-this-run 1`` where ``--stop-at-update`` was
    (the trainer forbids both; the protected-step check is skipped without ``--stop-at-update``).
    """
    if (stop_at is None) == (max_updates is None):
        raise ValueError('exactly one of stop_at and max_updates')
    argv = [str(Path(code_dir) / TRAINER_REL),
            '--recipe', str(paths.recipe),
            '--data-dir', str(paths.input_corpus),
            '--data-manifest', str(paths.input_corpus_manifest),
            '--eval-dir', str(paths.input_eval),
            '--output', str(output),
            '--warmup-checkpoint', str(paths.hypernet_init),
            '--model', str(paths.model_dir),
            '--tokenizer-pool', str(paths.pool_dir),
            '--microbatch-size', MICROBATCH, '--save-every', SAVE_EVERY]
    if protected:
        argv += ['--protected-steps', *PROTECTED_STEPS]
    argv += ['--eval-every', EVAL_EVERY]
    if max_updates is not None:
        argv += ['--max-updates-this-run', str(max_updates)]
    else:
        argv += ['--stop-at-update', str(stop_at)]
    argv += ['--seed', SEED]
    if resume:
        argv.append('--resume')
    return argv


def rl_argv(paths: Paths, code_dir, output, *, resume: bool, stop_at: int, protected: bool = True) -> list:
    """RL trainer argv (modal_anytok_pool_rl_20k.py:255-268 of the joint regret launcher): the policy flags
    ``--policy-entropy 0 --policy-lr 0.001 --policy-baseline-decay 0.99`` are passed explicitly, as there."""
    if paths.pool_dir.parent / 'uniform_run_config.json' != paths.uniform_run_config:
        raise SegmentError('the RL trainer reads <tokenizer-pool>/../uniform_run_config.json, which is not '
                           f'{paths.uniform_run_config}', exit_code=EXIT_USAGE)
    argv = [str(Path(code_dir) / TRAINER_REL),
            '--recipe', str(paths.recipe), '--data-dir', str(paths.input_corpus),
            '--data-manifest', str(paths.input_corpus_manifest), '--eval-dir', str(paths.input_eval),
            '--output', str(output), '--warmup-checkpoint', str(paths.hypernet_init),
            '--model', str(paths.model_dir), '--tokenizer-pool', str(paths.pool_dir),
            '--eval-tokenizers', str(paths.uniform_heldout),
            '--microbatch-size', MICROBATCH, '--save-every', SAVE_EVERY, '--eval-every', EVAL_EVERY,
            '--policy-entropy', POLICY_ENTROPY, '--policy-lr', POLICY_LR,
            '--policy-baseline-decay', POLICY_BASELINE_DECAY,
            '--stop-at-update', str(stop_at), '--seed', SEED]
    if protected:
        argv += ['--protected-steps', *PROTECTED_STEPS]
    if resume:
        argv += ['--resume']
    return argv


def fixed_paths(paths: Paths, run: str, output) -> dict:
    """Paths the trainer compares (as strings or resolved) on resume; recorded in hyak_segment.json."""
    rows = {'recipe': str(paths.recipe), 'data_dir': str(paths.input_corpus),
            'data_manifest': str(paths.input_corpus_manifest), 'eval_dir': str(paths.input_eval),
            'output': str(output), 'warmup_checkpoint': str(paths.hypernet_init),
            'warmup_checkpoint_resolved': str(paths.hypernet_init.resolve()),
            'model': str(paths.model_dir), 'tokenizer_pool': str(paths.pool_dir)}
    if run == RUN_RL:
        rows['eval_tokenizers'] = str(paths.uniform_heldout)
    return rows


# --- timing ----------------------------------------------------------------------------------------------------
def stop_budget(remaining, margin_seconds: float, max_seconds=None, minimum: float = MIN_START_SECONDS,
                lead: float = 0.0):
    """Seconds until the in-process SIGTERM (D14), or None when a segment must not start.

    ``remaining`` is the job's time left (None or inf when unknown/unlimited).  With a finite
    ``remaining``: None when ``remaining < margin + minimum``, else ``remaining - margin - lead``
    (``lead``: the timer fires that much before Slurm's ``--signal=B:TERM@margin``).  The result is
    capped by ``max_seconds``; ``math.inf`` means no timer (no limit known, none given).
    """
    budget = math.inf
    if remaining is not None and math.isfinite(remaining):
        if remaining < margin_seconds + minimum:
            return None
        budget = max(0.0, remaining - margin_seconds - lead)
    if max_seconds is not None:
        budget = min(budget, float(max_seconds))
    return budget


class StopTimer:
    """The in-process stop timer: SIGTERM to this process after ``seconds`` (D14), save-aware.

    When it fires while an RL weights-only grid save runs (``save_activity.weights_only``), it waits
    (polling every ``poll`` seconds, at most ``max_defer`` seconds) until that save is over: a stop
    signal inside such a save would end the run in ``paused_at`` without a full restart (RL trainer
    652-660; 654-662 in the D8 variant).  ``fired_at`` (``time.monotonic()``) and ``deferred_seconds`` are recorded for the
    segment_end event.
    """

    def __init__(self, seconds: float, save_activity=None, *, poll: float = 1.0, max_defer: float = STOP_DEFER_MAX_SECONDS,
                 send=None):
        self.seconds = max(0.0, float(seconds))
        self.save_activity = save_activity
        self.poll = poll
        self.max_defer = max_defer
        self.send = send or (lambda: os.kill(os.getpid(), signal.SIGTERM))
        self.fired_at = None
        self.deferred_seconds = 0.0
        self.deferred_step = None
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name='hyak-stop-timer', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def cancel(self):
        self._cancel.set()

    def _run(self):
        if self._cancel.wait(self.seconds):
            return
        began = time.monotonic()
        activity = self.save_activity
        while activity is not None and activity.weights_only and time.monotonic() - began < self.max_defer:
            self.deferred_step = activity.step
            if self._cancel.wait(self.poll):
                return
        self.deferred_seconds = round(time.monotonic() - began, 1)
        if self._cancel.is_set():
            return
        self.fired_at = time.monotonic()
        self.send()


class EarlySignals:
    """SIGTERM/SIGINT before the trainer installs its own handlers (trainer 357-365) only set a flag.

    A signal that arrives after the trainer was started but before its handlers exist is re-sent
    one second later, when the trainer's ``request_stop`` is in place.
    """

    def __init__(self):
        self.received = None
        self.received_at = None
        self.trainer_started = False
        self._previous = {}
        self._resend = None

    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handler)
        return self

    def rearm(self):
        """After ``trainer.main()`` returned: its handlers would swallow a stop; record it here instead."""
        self.cancel()
        self.trainer_started = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handler)

    def _handler(self, signum, _frame):
        self.received = signum
        if self.received_at is None:
            self.received_at = time.monotonic()
        print(json.dumps({'stage': 'stop_requested_by_wrapper', 'signal': signum}), flush=True)
        if self.trainer_started and self._resend is None:
            self._resend = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
            self._resend.daemon = True
            self._resend.start()

    def cancel(self):
        if self._resend is not None:
            self._resend.cancel()

    def restore(self):
        self.cancel()
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}


# --- run directory -------------------------------------------------------------------------------------------
class TrainerDirLock:
    """Hold ``<run>/.run.lock`` (the trainer's own flock, trainer 352-356) while the wrapper repairs the run."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / '.run.lock'
        self.handle = None

    def __enter__(self):
        if self.path.is_file():
            self.handle = open(self.path, 'a+')
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.handle.close()
                self.handle = None
                raise SegmentError(f'{self.path} is locked: a trainer process is still running in this run '
                                   'directory (not through this wrapper?). Stop it first',
                                   exit_code=EXIT_PROTOCOL) from exc
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        return False


def clear_dirty(run_dir: Path, archive_root: Path, label: str) -> dict:
    """Empty a run directory that holds no checkpoint (a first start that died before its first save).

    ``.writing-*`` staging directories are deleted; every other entry (``run_config.json``,
    ``heldout_tokenizers.json``, ``initial_eval.json``, ``events.jsonl``, ``.run.lock``,
    ``hyak_segment.json``) is moved to ``archive_root/<run>-<label>/`` for diagnosis.  The trainer
    refuses a non-empty directory without ``--resume`` and ``--resume`` without a restart
    (trainer 346-350, 400-403), so this is the only way forward.
    """
    target = archive_root / f'{run_dir.name}-{label}'
    moved, deleted = [], []
    for path in sorted(run_dir.iterdir()):
        if path.name.startswith('.writing-'):
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
            deleted.append(path.name)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))
            moved.append(path.name)
    run_dir.rmdir()
    return {'moved': moved, 'deleted': deleted, 'archive': str(target) if moved else None}


def _last_trainer_event(path: Path):
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


@dataclass
class Plan:
    """What one segment will run."""

    mode: str              # fresh | resume
    start_step: int | None
    target_step: int       # the step at which the trainer stops by itself
    argv: list
    bootstrap: bool


def plan_segment(run, paths, code_dir, run_dir, start_step, smoke_phase=None) -> Plan:
    """The trainer argv for this segment (see uniform_argv / rl_argv)."""
    fresh = start_step is None
    if smoke_phase is None:
        if run == RUN_UNIFORM and fresh:
            return Plan('fresh', None, BOOTSTRAP_UPDATES,
                        uniform_argv(paths, code_dir, run_dir, resume=False, max_updates=BOOTSTRAP_UPDATES),
                        True)
        if run == RUN_UNIFORM:
            argv = uniform_argv(paths, code_dir, run_dir, resume=True, stop_at=FINAL_STEP)
        else:
            argv = rl_argv(paths, code_dir, run_dir, resume=not fresh, stop_at=FINAL_STEP)
        return Plan('fresh' if fresh else 'resume', start_step, FINAL_STEP, argv, False)
    if smoke_phase == 'first':
        if not fresh:
            raise SegmentError(f'smoke phase "first" expects an empty {run_dir}')
        if run == RUN_UNIFORM:
            argv = uniform_argv(paths, code_dir, run_dir, resume=False, max_updates=BOOTSTRAP_UPDATES, protected=False)
        else:
            argv = rl_argv(paths, code_dir, run_dir, resume=False, stop_at=SMOKE_FIRST_STEP[RUN_RL], protected=False)
        return Plan('fresh', None, SMOKE_FIRST_STEP[run], argv, run == RUN_UNIFORM)
    if smoke_phase == 'second':
        if start_step != SMOKE_FIRST_STEP[run]:
            raise SegmentError(f'smoke phase "second" expects a full restart at step {SMOKE_FIRST_STEP[run]} '
                               f'in {run_dir}, found {start_step}')
        if run == RUN_UNIFORM:
            argv = uniform_argv(paths, code_dir, run_dir, resume=True, stop_at=SMOKE_FINAL_STEP, protected=False)
        else:
            argv = rl_argv(paths, code_dir, run_dir, resume=True, stop_at=SMOKE_FINAL_STEP, protected=False)
        return Plan('resume', start_step, SMOKE_FINAL_STEP, argv, False)
    raise ValueError(f'unknown smoke phase {smoke_phase!r}')


def compare_records(stored: dict, current: dict) -> list:
    """Differences between the stored hyak_segment.json and this segment's settings."""
    return [f'{key}: run started with {stored.get(key)!r}, now {current.get(key)!r}'
            for key in COMPARED_RECORD_FIELDS if stored.get(key) != current.get(key)]


def _check_state(run, state, step):
    if int(state['step']) != step:
        raise SegmentError(f'state.pt of step-{step:06d} says step {state["step"]}')
    if run == RUN_RL:
        bandit = state['bandit']
        # modal_anytok_pool_rl_20k.py:268-269
        if bandit['updates'] != step or int(bandit['counts'].sum()) != step:
            raise SegmentError(f'Policy did not resume with the model: policy updates {bandit["updates"]}, '
                               f'counts {int(bandit["counts"].sum())}, model step {step}')


def _paused_on_weights_only_save(run, run_dir: Path, newest_step: int, last):
    """RL only: step N if the trainer's last event is ``paused_at N`` above the newest full restart and
    ``step-N`` is a weights-only grid checkpoint, else None.

    The RL trainer evaluates ``full_restart=(... or stop_requested)`` before it calls
    ``save_checkpoint`` (RL trainer 652-656, 654-658 in the D8 variant); a stop signal that arrives during a weights-only grid
    save (Slurm's backup signal, a manual stop) therefore ends in ``paused_at N`` without a restart at
    N.  Nothing else produces this state.
    """
    if run != RUN_RL or not isinstance(last, dict) or last.get('stage') != 'paused_at':
        return None
    step = last.get('step')
    if isinstance(step, bool) or not isinstance(step, int) or step <= newest_step:
        return None
    if step % retention.CHECKPOINT_INTERVAL or step % retention.RESUME_INTERVAL == 0:
        return None
    folder = run_dir / retention.step_name(step)
    if (folder / 'state.pt').exists():
        return None
    try:
        manifest = retention.read_manifest(folder)
    except retention.CheckpointError:
        return None
    if manifest.get('kind') != retention.WEIGHTS_ONLY or manifest.get('step') != step:
        return None
    return step


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def validate_rl_protocol(run_dir: Path, state: dict) -> dict:
    """The RL receipt of a newest full restart; raises SegmentError (exit 4) on any failure.

    The joint regret launcher's checks (scripts/modal_anytok_pool_rl_20k.py:280-289): ``run_config.json``
    policy entropy weight 0; the state's metrics ``paired_valid_examples == 128``,
    ``paired_maximum_positions <= 128`` and ``|paired_gap_bpb - (train_bpb - paired_canonical_bpb)| <= 1e-10``.
    Plus the straight reward of D8: ``policy_reward`` is ``paired_pre_model_update_transferred_bpb`` and
    ``|policy_reward_bpb - train_bpb| <= 1e-12``, and the other explicit policy settings (learning rate 0.001,
    baseline decay 0.99, seed 43).  At step 0 (the RL trainer's initial checkpoint, before any update) the
    metrics checks do not apply; ``metrics_checked`` is then false.
    """
    config_path = Path(run_dir) / 'run_config.json'
    try:
        config = read_json(config_path)
    except (OSError, ValueError) as exc:
        raise SegmentError(f'paired-protocol check: cannot read {config_path}: {exc}') from exc
    problems = []
    policy = config.get('policy') if isinstance(config, dict) else None
    policy = policy if isinstance(policy, dict) else {}
    for key, wanted in RL_POLICY_SETTINGS.items():
        if policy.get(key) != wanted:
            problems.append(f'run_config.json policy {key} is {policy.get(key)!r}, expected {wanted!r}')
    if config.get('policy_reward') != RL_POLICY_REWARD:
        problems.append(f'run_config.json policy_reward is {config.get("policy_reward")!r}, expected '
                        f'{RL_POLICY_REWARD!r} (the straight-BPB-reward trainer, D8)')
    step = int(state.get('step', -1))
    metrics = state.get('metrics') if isinstance(state.get('metrics'), dict) else {}
    receipt = {'step': step, 'policy_reward': config.get('policy_reward'),
               'entropy_weight': policy.get('entropy_weight'), 'metrics_checked': step > 0}
    if step > 0:
        values = {key: metrics.get(key) for key in ('train_bpb', 'paired_canonical_bpb', 'paired_gap_bpb',
                                                    'policy_reward_bpb', 'paired_valid_examples',
                                                    'paired_maximum_positions')}
        missing = sorted(key for key, value in values.items() if _number(value) is None)
        if missing:
            problems.append(f'the metrics of step {step} lack finite {missing}')
        else:
            if values['paired_valid_examples'] != PAIRED_VALID_EXAMPLES:
                problems.append(f'paired_valid_examples {values["paired_valid_examples"]}, expected {PAIRED_VALID_EXAMPLES}')
            if values['paired_maximum_positions'] > PAIRED_MAX_POSITIONS:
                problems.append(f'paired_maximum_positions {values["paired_maximum_positions"]} > {PAIRED_MAX_POSITIONS}')
            residual = abs(values['paired_gap_bpb'] - (values['train_bpb'] - values['paired_canonical_bpb']))
            if residual > PAIRED_GAP_TOLERANCE:
                problems.append(f'paired_gap_bpb differs from train_bpb - paired_canonical_bpb by {residual!r}')
            difference = abs(values['policy_reward_bpb'] - values['train_bpb'])
            if difference > POLICY_REWARD_TOLERANCE:
                problems.append(f'policy_reward_bpb {values["policy_reward_bpb"]!r} is not train_bpb '
                                f'{values["train_bpb"]!r} (difference {difference!r}): the policy did not receive the '
                                'straight transferred BPB')
            receipt.update(values)
    if problems:
        raise SegmentError(f'paired-protocol check of {run_dir} (step {step}) failed: ' + '; '.join(problems)
                           + '. The RL run must be the straight-BPB-reward counterpart of the joint regret run (D8); '
                           'inspect the trainer and its staged code before resuming')
    return receipt


def post_checks(run, run_dir: Path, adapter, *, start_step, emit) -> int:
    """Newest restart verified; state step; RL counters; progress; the trainer's last event (paused_at).

    One tolerated exception (RL): a pause at a weights-only grid checkpoint above the newest full
    restart (see :func:`_paused_on_weights_only_save`) is a warning; the segment counts as paused at
    the newest full restart and the next segment replays from there.
    """
    full = retention.full_restarts(run_dir)
    if not full:
        raise SegmentError(f'the trainer returned but {run_dir} holds no full restart')
    newest_step, newest = full[-1]
    checked = retention.verify_checkpoint(run_dir, newest_step, run, retention.FULL)
    state = adapter.load_state(newest / 'state.pt')
    _check_state(run, state, newest_step)
    last = _last_trainer_event(run_dir / 'events.jsonl')
    late = _paused_on_weights_only_save(run, run_dir, newest_step, last)
    if late is not None:
        emit.warn(f'a stop signal arrived while the weights-only checkpoint of step {late} was being saved, so the '
                  f'trainer paused at {late} without a full restart (RL trainer 654-662). Nothing is damaged: the '
                  f'next segment resumes from step {newest_step} and replays {late - newest_step} updates',
                  paused_step=late, restart_step=newest_step)
    else:
        if start_step is not None and newest_step <= start_step:
            raise SegmentError(f'Training segment made no checkpoint progress (still at step {newest_step})')
        if not last or last.get('stage') not in ('paused_at', 'complete') or last.get('step') != newest_step:
            raise SegmentError(f'the trainer\'s last event is {last}, expected paused_at at step {newest_step}')
    emit('verified_restart', step=newest_step, kind=checked['kind'], bytes=checked['bytes'], files=checked['files'])
    if run == RUN_RL:
        emit('paired_protocol_verified', **validate_rl_protocol(run_dir, state))
    return newest_step


# --- the segment ---------------------------------------------------------------------------------------------
def _dist_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 -- a broken distribution must not stop training
        return None


def environment_snapshot(paths: Paths, gpu: dict) -> dict:
    """Software environment of this segment: Python, torch, CUDA, transformers and the freeze file hash."""
    freeze = paths.env_freeze('train')
    try:
        freeze_sha = sha256_file(freeze) if freeze.is_file() else None
    except OSError:
        freeze_sha = None
    return {'python': sys.version.split()[0], 'torch': gpu.get('torch'), 'cuda': gpu.get('cuda'),
            'transformers': _dist_version('transformers'), 'env_freeze_sha256': freeze_sha,
            'env_freeze_path': str(freeze)}


def _segment_record(run, paths, cfg, run_dir, code, inputs, chunk, gpu, environment=None) -> dict:
    environment = environment if environment is not None else environment_snapshot(paths, gpu)
    return {'version': SEGMENT_RECORD_VERSION, 'run': run, 'created_at_utc': utc_now(),
            'created_by_job': slurm.worker_id(), 'host': slurm.node_name(),
            'chunk_rows': chunk['chunk_rows'], 'chunk_module_sha256': chunk.get('module_sha256'),
            'chunk_identity': {key: chunk.get(key) for key in CHUNK_IDENTITY_KEYS},
            'chunk_settings': chunk, 'code_tree_sha256': code['code_tree_sha256'],
            'code_manifest_sha256': code['code_manifest_sha256'], 'trainer_sha256': code['trainer_sha256'],
            'policy_sha256': code['policy_sha256'], 'hf_hypernet_sha256': code['hf_hypernet_sha256'],
            'trainer_variant': trainer_variant(run, code['trainer_sha256']),
            'rust_utils_sha256': code['rust_utils_sha256'], 'init_sha256': inputs['init_sha256'],
            'init_recipe': inputs.get('init_recipe'),           # information only, not in COMPARED_RECORD_FIELDS
            'model_sha256': inputs.get('model_sha256'), 'corpus_test_scale': inputs.get('corpus_test_scale', False),
            'input_manifest_sha256': inputs['manifest_sha256'], 'fixed_paths': fixed_paths(paths, run, run_dir),
            'platform': stage_value(cfg, 'HYAK_PLATFORM', STAGES[run]), 'gpu_name': gpu.get('name'),
            'gpu_count': gpu.get('count'), 'torch': gpu.get('torch'), 'cuda': gpu.get('cuda'),
            'transformers': environment.get('transformers'), 'env_freeze_sha256': environment.get('env_freeze_sha256'),
            'python': sys.version.split()[0], 'environment': environment}


def trainer_variant(run: str, trainer_sha256: str) -> dict:
    """Which trainer this run executes (recorded in hyak_segment.json; information, covered by code_tree_sha256)."""
    if run == RUN_RL:
        record = dict(pinned.RL_TRAINER_VARIANT)
    else:
        record = {'name': 'original_uniform', 'path': pinned.UNIFORM_TRAINER, 'sha256': pinned.UNIFORM_TRAINER_SHA256}
    record['staged_sha256'] = trainer_sha256
    return record


def _outcome(name, **fields):
    return {'outcome': name, **fields}


def _previous_environment(stored: dict) -> dict:
    history = stored.get('environment_history')
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        return history[-1]
    if isinstance(stored.get('environment'), dict):
        return stored['environment']
    return {key: stored.get(key) for key in ENVIRONMENT_FIELDS}


def _check_environment(stored: dict, environment: dict, record_path: Path, emit, start_step) -> None:
    """A changed software environment between segments: loud warning + event, recorded (not a refusal)."""
    previous = _previous_environment(stored)
    changes = {key: {'before': previous.get(key), 'now': environment.get(key)} for key in ENVIRONMENT_FIELDS
               if previous.get(key) != environment.get(key)}
    if not changes:
        return
    text = ', '.join(f'{key} {value["before"]!r} -> {value["now"]!r}' for key, value in changes.items())
    emit.warn(f'THE SOFTWARE ENVIRONMENT CHANGED since the previous segment of this run ({text}). The run continues, '
              'but its segments no longer share one environment; if this was not intended, restore the environment '
              '(hyak/env/setup_train_env.sh with the original pins) before the next job', changes=changes)
    emit('environment_changed', step=start_step, changes=changes)
    entry = {**environment, 'at_utc': utc_now(), 'job': slurm.worker_id(), 'step': start_step}
    stored = dict(stored)
    stored['environment_history'] = list(stored.get('environment_history') or []) + [entry]
    atomic_write_json(record_path, stored)


def _rotate_previous_attempt(paths: Paths, stage: str, emit) -> None:
    """A run that starts from scratch: move an earlier attempt's monitor files to ``previous-<utc>/``.

    Moved (only when another job's ``*-train.log`` or ``policy-monitor.json`` shows an earlier attempt):
    other jobs' ``*-train.log`` and ``slurm-*.out``, ``policy-monitor.json``, ``policy-monitor.log``,
    ``events.jsonl`` and ``status.json``.  This job's own logs stay (they are open and being written);
    the dashboard reads only the top level of the stage directory.
    """
    monitor = paths.monitor_dir(stage)
    if not monitor.is_dir():
        return
    job = slurm.job_id()
    mine = {f'{job}-train.log', f'slurm-{job}.out'} if job else set()
    names = sorted(p.name for p in monitor.iterdir() if p.is_file())
    logs = [name for name in names if (name.endswith('-train.log') or (name.startswith('slurm-') and name.endswith('.out')))
            and name not in mine]
    trained = any(name.endswith('-train.log') and _trainer_started_in(monitor / name) for name in logs)
    if not trained and 'policy-monitor.json' not in names:
        return
    moving = logs + [name for name in ('policy-monitor.json', 'policy-monitor.log', 'events.jsonl', 'status.json')
                     if name in names]
    target = monitor / f'previous-{retention.interrupted_label()}'
    target.mkdir(parents=True, exist_ok=False)
    for name in moving:
        shutil.move(str(monitor / name), str(target / name))
    emit('previous_attempt_rotated', moved=moving, to=str(target))


def _trainer_started_in(log: Path) -> bool:
    """Whether a train log shows a trainer start (its initial_eval line): evidence of an earlier attempt."""
    try:
        with open(log, 'rb') as stream:
            head = stream.read(4 * 1024 * 1024)
    except OSError:
        return False
    return b'"stage": "initial_eval"' in head


def _retire_final_audit(run_dir: Path, emit) -> None:
    """A run that trains is not complete: an existing final_audit.json (e.g. after a quarantine of step
    20,000 and a replay) would describe files that are replaced, so it is renamed before the trainer starts."""
    audit = run_dir / retention.FINAL_AUDIT
    if audit.is_file():
        target = run_dir / f'final_audit.stale-{retention.interrupted_label()}.json'
        os.replace(audit, target)
        emit('final_audit_retired', moved_to=str(target))


def run_segment(run: str, *, smoke_phase=None, max_seconds=None, config=None, adapter_factory=TrainerAdapter,
                expected_code=None, init_bytes: int = HYPERNET_INIT_BYTES,
                plan_file_sha256: str = PLAN_FILE_SHA256, accept_chunk_change=None, model_expected=None,
                allow_test_scale: bool = False, lock_wait_seconds: float = LOCK_WAIT_SECONDS) -> dict:
    """Run one segment; returns ``{"outcome": ..., ...}``.  Raises on every failure (see the exit statuses).

    Outcomes: ``paused``, ``bootstrap_done``, ``complete``, ``already_complete``,
    ``insufficient_time``, ``stopped_before_start``, ``audit_interrupted``, ``smoke_phase_done``, and
    the 'not now' outcomes ``uniform_bootstrap_pending`` and ``run_locked`` (exit 75).
    ``accept_chunk_change`` (the reason text of ``--accept-chunk-change``) accepts a changed chunk
    setting on the record.  ``adapter_factory``, ``expected_code``, ``init_bytes``, ``plan_file_sha256``,
    ``model_expected``, ``allow_test_scale`` and ``lock_wait_seconds`` are test parameters (no command-line flag).
    """
    run = validate_run(run)
    if max_seconds is not None and max_seconds < MIN_MAX_SECONDS:
        raise ValueError(f'--max-seconds must be at least {MIN_MAX_SECONDS}')
    if accept_chunk_change is not None and not str(accept_chunk_change).strip():
        raise ValueError('--accept-chunk-change needs a reason text')
    cfg = load_config(config)
    paths = Paths.from_config(cfg)
    smoke = smoke_phase is not None
    stage = STAGE_SMOKE_TRAIN if smoke else STAGES[run]
    run_dir = paths.smoke_dir(run) if smoke else paths.run_dir(run)
    code_dir = paths.code_dir(run)
    emit = Emitter(stage, run, cfg)
    signals = EarlySignals().install()
    try:
        lock = retention.WrapperLock(
            retention.wrapper_lock_path(paths, run, smoke), wait_seconds=lock_wait_seconds,
            poll_seconds=min(LOCK_POLL_SECONDS, max(lock_wait_seconds, 0.05)),
            on_wait=lambda holder: emit('waiting_for_run_lock', holder=holder, wait_seconds=lock_wait_seconds),
            should_stop=lambda: signals.received is not None)
        try:
            lock.__enter__()
        except retention.LockHeld as exc:
            if signals.received is not None:
                emit('stopped_before_start', signal=signals.received, step=None)
                return _outcome('stopped_before_start', step=None)
            emit.warn(f'not starting: {exc}', waited_seconds=round(lock.waited, 1))
            emit('run_locked', waited_seconds=round(lock.waited, 1), error=str(exc))
            return _outcome('run_locked', step=None, waited_seconds=round(lock.waited, 1))
        try:
            if lock.waited:
                emit('run_lock_acquired', waited_seconds=round(lock.waited, 1))
            return _run_locked(run, cfg, paths, smoke_phase, run_dir, code_dir, emit, signals, max_seconds,
                               adapter_factory, expected_code, init_bytes, stage, plan_file_sha256,
                               accept_chunk_change=accept_chunk_change, model_expected=model_expected,
                               allow_test_scale=allow_test_scale)
        finally:
            lock.__exit__(None, None, None)
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt,)):
            emit('segment_failed', error=f'{type(exc).__name__}: {exc}'[-2000:],
                 traceback=traceback.format_exc()[-6000:])
        raise
    finally:
        signals.restore()


def _accept_chunk_change(stored, record, record_path, reason, start_step, emit) -> dict:
    change = {'at_utc': utc_now(), 'job': slurm.worker_id(), 'step': start_step,
              'from': stored.get('chunk_identity'), 'to': record.get('chunk_identity'), 'reason': str(reason)}
    updated = dict(stored)
    for key in ('chunk_rows', 'chunk_module_sha256', 'chunk_identity', 'chunk_settings'):
        updated[key] = record.get(key)
    updated['chunk_changes'] = list(stored.get('chunk_changes') or []) + [change]
    atomic_write_json(record_path, updated)
    emit.warn(f'the chunked-forward setting of this run changed at step {start_step} and was ACCEPTED with '
              f'--accept-chunk-change ({reason!r}): {change["from"]} -> {change["to"]}', change=change)
    emit('chunk_change_accepted', **change)
    return updated


def _run_locked(run, cfg, paths, smoke_phase, run_dir, code_dir, emit, signals, max_seconds, adapter_factory,
                expected_code, init_bytes, stage, plan_file_sha256=PLAN_FILE_SHA256, *, accept_chunk_change=None,
                model_expected=None, allow_test_scale=False):
    smoke = smoke_phase is not None
    code = verify_code(code_dir, run, paths.code_manifest, expected_code)
    emit('verified_code', code_tree_sha256=code['code_tree_sha256'], files=code['files'],
         rust_utils_rebuilt=code['rust_utils_rebuilt'])
    cache = HashCache(paths.runs / '.hyak-cache' / f'input-sha256-{run}.json')
    inputs = verify_inputs(paths, run, cache, init_bytes=init_bytes, plan_file_sha256=plan_file_sha256,
                           model_expected=model_expected, allow_test_scale=allow_test_scale)
    emit('verified_inputs', file_count=inputs['file_count'], bytes=inputs['bytes'],
         manifest_sha256=inputs['manifest_sha256'], model_sha256=inputs['model_sha256'],
         corpus_test_scale=inputs['corpus_test_scale'])
    if inputs['corpus_test_scale']:
        emit.warn('input/manifest.json marks a TEST-SCALE corpus (allowed only by the explicit test parameter)')
    if inputs.get('init_recipe_warning'):
        emit.warn(inputs['init_recipe_warning'], init_recipe=None)
    pair = None
    if run == RUN_RL:
        try:
            pair = pairing.ensure(paths, role=RUN_RL, emit=emit)
        except pairing.UniformNotReady as exc:
            emit('uniform_bootstrap_pending', reason=str(exc))
            return _outcome('uniform_bootstrap_pending', step=None, reason=str(exc))
    elif not smoke:
        pairing.verify_uniform_unchanged(paths)

    with TrainerDirLock(run_dir):
        info = retention.inspect_run_dir(run_dir)
        if info['exists'] and info['entries'] and info['newest_full_step'] is None:
            _handle_dirty(run, paths, run_dir, info, smoke, emit)
        elif _unverified_first_save(run, paths, info, smoke):
            result = clear_dirty(run_dir, paths.runs / 'cleared', retention.interrupted_label())
            emit('cleared_run_directory', reason='the first save of the run has no manifest (killed between the '
                 'trainer\'s save and the hook); at most one update is lost', **result)
        if run == RUN_UNIFORM and not smoke and retention.inspect_run_dir(run_dir)['newest_full_step'] is None \
                and pairing.is_frozen(paths):
            raise SegmentError(
                f'The uniform/RL pair is frozen ({paths.pairing}), but the uniform run {run_dir} has no full restart. '
                'Restarting the uniform run from scratch would draw new sanity tokenizers and break its pairing '
                'with the RL run. Remedy: restore the uniform run directory from a backup; to begin a new '
                'experiment instead, move both run directories, input/pairing.json and input/uniform_*.json away '
                'together (and the monitor directories, see the run guide)', exit_code=EXIT_PROTOCOL)
        pre = retention.preflight(run_dir, run, adopt_code_dir=code_dir, remedy_suffix=' --smoke' if smoke else '')
    for key, name in (('removed_writing', 'removed_stale_staging'), ('finished_compactions', 'finished_compaction'),
                      ('finished_deletions', 'finished_deletion'), ('removed_manifest_tmp', 'removed_stale_manifest_tmp')):
        if getattr(pre, key):
            emit(name, entries=getattr(pre, key))
    if pre.adopted is not None:
        emit.warn(f'the newest full restart step {pre.adopted} had no manifest (a kill between the trainer\'s rename '
                  'and the manifest hook); it holds exactly the files of a full restart and was adopted '
                  '(manifest written from its files)', step=pre.adopted)
        emit('adopted_restart', step=pre.adopted, bytes=pre.newest_full_bytes)
    if pre.moved_interrupted:
        emit('moved_interrupted', steps=pre.moved_interrupted, to=pre.interrupted_dir,
             restart_step=pre.newest_full_step)
    start_step = pre.newest_full_step
    record_path = run_dir / retention.SEGMENT_RECORD
    if start_step is not None:
        emit('verified_restart', step=start_step, kind=retention.FULL, bytes=pre.newest_full_bytes)
        if not record_path.is_file():
            raise SegmentError(f'{record_path} is missing although {run_dir} has a full restart, so the chunk setting, '
                               'code and initialisation of this run cannot be proven. Restore it from a backup',
                               exit_code=EXIT_PROTOCOL)
        stored = read_json(record_path)
    else:
        stored = None

    if not smoke and start_step is not None and start_step >= FINAL_STEP:
        return _finish_complete(run, paths, run_dir, adapter_factory(code_dir), emit, start_step, already=True,
                                signals=signals)

    plan = plan_segment(run, paths, code_dir, run_dir, start_step, smoke_phase)
    margin = int_value(cfg, 'HYAK_SEGMENT_MARGIN_MINUTES', stage) * 60
    if stop_budget(slurm.seconds_remaining(), margin, max_seconds) is None:
        emit('insufficient_time', seconds_remaining=slurm.seconds_remaining(), margin_seconds=margin,
             minimum_seconds=MIN_START_SECONDS, step=start_step)
        return _outcome('insufficient_time', step=start_step)
    if signals.received is not None:
        emit('stopped_before_start', signal=signals.received, step=start_step)
        return _outcome('stopped_before_start', step=start_step)

    adapter = adapter_factory(code_dir)
    adapter.load()
    chunk = adapter.install_memory_patch(int_value(cfg, 'HYAK_HN_CHUNK_ROWS'))
    gpu = adapter.gpu_info()
    if not gpu.get('name') or EXPECTED_GPU not in gpu['name']:
        emit.warn(f'GPU is {gpu.get("name")!r}, not an {EXPECTED_GPU}; the run is calibrated for the H200 '
                  '(memory patch chunk size, time limits)', gpu=gpu)
    environment = environment_snapshot(paths, gpu)
    record = _segment_record(run, paths, cfg, run_dir, code, inputs, chunk, gpu, environment)
    if stored is not None:
        differing = [key for key in COMPARED_RECORD_FIELDS if stored.get(key) != record.get(key)]
        chunk_only = differing == ['chunk_identity']
        if differing and not (chunk_only and accept_chunk_change):
            problems = compare_records(stored, record)
            hint = (' If the chunk setting must change mid-run (for example after an out-of-memory failure), accept '
                    'it on the record with --accept-chunk-change "<reason>"' if 'chunk_identity' in differing else '')
            raise SegmentError(f'{record_path} does not match this segment: ' + '; '.join(problems)
                               + '. A run must keep its code, initial hypernetwork, model files, chunk setting '
                               '(HYAK_HN_CHUNK_ROWS, hyak/memory/chunked_hypernet.py) and storage paths until step '
                               '20,000. Remedy: restore the original setting.' + hint, exit_code=EXIT_PROTOCOL)
        if differing:
            stored = _accept_chunk_change(stored, record, record_path, accept_chunk_change, start_step, emit)
        if stored.get('gpu_count') not in (None, gpu.get('count')):
            emit.warn(f'{gpu.get("count")} visible GPUs, the run started with {stored.get("gpu_count")} '
                      '(the trainer restores one CUDA RNG state per device)')
        _check_environment(stored, environment, record_path, emit, start_step)
    if pair is not None:
        accepted_before = [entry.get('difference') for entry in ((stored or {}).get('accepted_pairing_differences') or [])
                           if isinstance(entry, dict)]
        ignore = ('chunk_identity',) if (accept_chunk_change or accepted_before) else ()
        ignored = pairing.check_rl_settings(pair, record, ignore=ignore)
        new = [text for text in ignored if text not in accepted_before]
        if new and not accept_chunk_change:
            raise pairing.PairingError('The RL run does not match the uniform run it is compared with: ' + '; '.join(new)
                                       + '. Remedy: use the uniform run\'s HYAK_HN_CHUNK_ROWS, or accept the difference '
                                       'on the record with --accept-chunk-change "<reason>"')
        if new:
            entries = [{'difference': text, 'reason': str(accept_chunk_change), 'at_utc': utc_now(),
                        'job': slurm.worker_id(), 'step': start_step} for text in new]
            emit.warn(f'the RL run\'s chunk setting differs from the uniform run\'s and was ACCEPTED with '
                      f'--accept-chunk-change ({accept_chunk_change!r}): {"; ".join(new)}')
            emit('chunk_change_accepted', against='uniform', differences=new, reason=str(accept_chunk_change))
            if stored is not None:
                stored = dict(stored)
                stored['accepted_pairing_differences'] = list(stored.get('accepted_pairing_differences') or []) + entries
                atomic_write_json(record_path, stored)
            else:
                record['accepted_pairing_differences'] = entries
    if stored is None and record_path.exists():
        raise SegmentError(f'{run_dir} is not empty before a fresh start', exit_code=EXIT_PROTOCOL)

    def write_record_once(_step):
        if not record_path.exists():
            atomic_write_json(record_path, record)

    save_activity = retention.SaveActivity()
    adapter.wrap_save(lambda original: retention.save_hook(run, original, run_dir, code_dir, emit=emit,
                                                           before_manifest=write_record_once,
                                                           save_activity=save_activity))
    budget = stop_budget(slurm.seconds_remaining(), margin, max_seconds, lead=STOP_TIMER_LEAD_SECONDS)
    if budget is None:
        emit('insufficient_time', seconds_remaining=slurm.seconds_remaining(), margin_seconds=margin, step=start_step)
        return _outcome('insufficient_time', step=start_step)
    timer = None
    if math.isfinite(budget):
        timer = StopTimer(budget, save_activity)
    elif not smoke or slurm.job_id() is not None:
        emit.warn('the job time limit is unknown (no SLURM_JOB_END_TIME, squeue failed) and no --max-seconds was '
                  'given: no in-process stop timer; only Slurm --signal=B:TERM can stop this segment gracefully')
    signals.trainer_started = True
    if signals.received is not None:
        # A stop arrived while the trainer was being imported and prepared (seconds with the real trainer):
        # its own handlers are not installed yet, so it would never see that signal.
        signals.cancel()
        signals.trainer_started = False
        emit('stopped_before_start', signal=signals.received, step=start_step)
        return _outcome('stopped_before_start', step=start_step)
    if not smoke:
        if start_step is None:
            try:
                _rotate_previous_attempt(paths, stage, emit)
            except OSError as exc:     # display files only: never a reason not to train
                emit.warn(f'could not move the earlier attempt\'s monitor files aside: {exc}')
        _retire_final_audit(run_dir, emit)
    emit('segment_start', mode=plan.mode, bootstrap=plan.bootstrap, start_step=start_step,
         target_step=plan.target_step, stop_timer_seconds=budget if math.isfinite(budget) else None,
         stop_timer_lead_seconds=STOP_TIMER_LEAD_SECONDS if timer is not None and max_seconds is None else 0,
         margin_seconds=margin, chunk_rows=chunk['chunk_rows'], gpu=gpu.get('name'), argv=plan.argv,
         environment={key: environment.get(key) for key in ENVIRONMENT_FIELDS},
         env={name: os.environ.get(name) for name in RECORDED_ENV_VARS})
    started = time.monotonic()
    if timer is not None:
        timer.start()
    try:
        adapter.run_main(plan.argv)
    except SystemExit as exc:
        raise SegmentError(f'the trainer exited with status {exc.code} (an argument check failed; see stderr)',
                           exit_code=EXIT_USAGE) from exc
    finally:
        returned = time.monotonic()
        if timer is not None:
            timer.cancel()
        signals.rearm()
    stop = _stop_measurement(timer, signals, returned)
    newest = post_checks(run, run_dir, adapter, start_step=start_step, emit=emit)
    if newest > plan.target_step:
        raise SegmentError(f'the trainer saved step {newest}, beyond this segment\'s stop at {plan.target_step}')
    elapsed = round(time.monotonic() - started, 1)
    if run == RUN_UNIFORM and not smoke:
        try:
            pairing.ensure(paths, role=RUN_UNIFORM, emit=emit)
        except pairing.RLDirectoryNotEmpty as exc:
            emit.warn(f'the uniform files were not frozen: {exc}')
    if run == RUN_RL and plan.mode == 'fresh':
        _compare_initial_eval(paths, run_dir, emit)
    if not smoke and newest >= FINAL_STEP:
        result = _finish_complete(run, paths, run_dir, adapter, emit, newest, already=False, signals=signals)
        result['elapsed_seconds'] = elapsed
        return result
    outcome = 'smoke_phase_done' if smoke else ('bootstrap_done' if plan.bootstrap else 'paused')
    if stop['stop_source'] is None and newest < plan.target_step:
        stop['stop_source'] = 'external'          # Slurm's --signal=B:TERM or a manual stop, forwarded by the shell
    emit('segment_end', outcome=outcome, step=newest, elapsed_seconds=elapsed, **stop)
    result = _outcome(outcome, step=newest, elapsed_seconds=elapsed, start_step=start_step)
    if smoke:
        result['stop_timer_seconds'] = budget if math.isfinite(budget) else None
    return result


def _stop_measurement(timer, signals, returned: float) -> dict:
    """How the trainer was stopped and how long the stop took (signal to the trainer's return)."""
    if timer is not None and timer.fired_at is not None:
        return {'stop_source': 'timer', 'stop_seconds': round(returned - timer.fired_at, 1),
                'stop_deferred_seconds': timer.deferred_seconds, 'stop_deferred_step': timer.deferred_step}
    if signals.received_at is not None:
        return {'stop_source': 'signal', 'stop_seconds': round(returned - signals.received_at, 1)}
    return {'stop_source': None, 'stop_seconds': None}


def _unverified_first_save(run, paths, info, smoke) -> bool:
    """True when the only checkpoint is the run's first save (RL step 0, uniform bootstrap step 1) without a
    manifest, nothing was moved aside and (uniform) nothing is frozen yet: restarting loses at most one update."""
    first = 0 if run == RUN_RL else 1
    return (info['exists'] and info['steps'] == [{'step': first, 'full': True, 'manifest': False}]
            and retention.INTERRUPTED not in info['entries']
            and (run == RUN_RL or smoke or not pairing.is_frozen(paths)))


def _handle_dirty(run, paths, run_dir, info, smoke, emit):
    steps = info['steps']
    if steps or retention.INTERRUPTED in info['entries']:
        raise SegmentError(
            f'{run_dir} has checkpoint directories {[row["step"] for row in steps]} (or interrupted/) but no full '
            'restart, so it can neither resume nor start fresh, and clearing it could destroy progress. Inspect it '
            f'with: python -m hyak.train.retention summary --run {run}{" --smoke" if smoke else ""}',
            exit_code=EXIT_PROTOCOL)
    if run == RUN_UNIFORM and not smoke and pairing.is_frozen(paths):
        raise SegmentError(
            f'{run_dir} has no full restart but the uniform/RL pair is frozen ({paths.pairing}); clearing it and '
            'starting the uniform run again would draw new sanity tokenizers and break the pairing with the RL '
            'run. Remedy: restore the uniform run directory from a backup; to begin a new experiment instead, '
            'move both run directories, input/pairing.json and input/uniform_*.json away together (and the '
            'monitor directories, see the run guide)', exit_code=EXIT_PROTOCOL)
    # RL: always safe (it writes a full step-0 restart first); uniform: only before the pair is frozen.
    result = clear_dirty(run_dir, paths.runs / 'cleared', retention.interrupted_label())
    emit('cleared_run_directory', reason='no full restart (a first start died before its first save)', **result)


#: Why the RL run's step-0 evaluation is not compared with the uniform run's (D8).
INITIAL_EVAL_NOTE = ('narrowed (D8): the RL trainer uses the corrected paired cuts for its sanity examples and the '
                     'uniform trainer the original cuts, so initial_eval.json is not expected to be equal; only the '
                     'sanity tokenizer specs (heldout_tokenizers.json) must equal the frozen uniform ones')


def compare_sanity_panel(paths, rl_dir) -> dict:
    """The narrowed step-0 pairing check of the RL run (D8).

    ``initial_eval_equal`` (information only) and ``sanity_tokenizers_equal`` (the RL run's
    ``heldout_tokenizers.json`` against the frozen ``input/uniform_heldout_tokenizers.json``; None when a
    file cannot be read).
    """
    result = {'check': INITIAL_EVAL_NOTE}
    try:
        result['initial_eval_equal'] = (read_json(paths.run_dir(RUN_UNIFORM) / 'initial_eval.json')
                                        == read_json(Path(rl_dir) / 'initial_eval.json'))
    except (OSError, ValueError):
        result['initial_eval_equal'] = None
    try:
        result['sanity_tokenizers_equal'] = (read_json(paths.uniform_heldout)
                                             == read_json(Path(rl_dir) / 'heldout_tokenizers.json'))
    except (OSError, ValueError):
        result['sanity_tokenizers_equal'] = None
    return result


def _compare_initial_eval(paths, rl_dir, emit):
    result = compare_sanity_panel(paths, rl_dir)
    emit('initial_eval_pairing', **result)
    if result['sanity_tokenizers_equal'] is not True:
        emit.warn(f'{Path(rl_dir) / "heldout_tokenizers.json"} is not the frozen {paths.uniform_heldout} '
                  f'(result {result["sanity_tokenizers_equal"]}): the two runs do not share their sanity tokenizers')


def _finish_complete(run, paths, run_dir, adapter, emit, step, *, already, signals=None):
    newest = run_dir / retention.step_name(step)
    _check_state(run, adapter.load_state(newest / 'state.pt'), step)
    if step != FINAL_STEP:
        raise SegmentError(f'newest full restart is step {step}, beyond {FINAL_STEP}')
    if retention.audit_valid(run_dir, run):
        receipt = read_json(run_dir / retention.FINAL_AUDIT)
    else:
        # the retention step of the last save, idempotent (a kill inside the last hook, or an adopt of step 20,000)
        finished = retention.finish_retention(run_dir, run)
        if any(finished.values()):
            emit('finished_retention', step=step, **finished)
        stop = (lambda: signals.received is not None) if signals is not None else None
        try:
            receipt = retention.audit_final(run_dir, run, paths.input_manifest, should_stop=stop)
        except retention.AuditInterrupted as exc:
            emit('final_audit_interrupted', step=step, reason=str(exc))
            return _outcome('audit_interrupted', step=step)
        emit('final_audit', **receipt)
    outcome = 'already_complete' if already else 'complete'
    emit('segment_end', outcome=outcome, step=step)
    emit('run_complete', step=step, final_audit=receipt)
    return _outcome(outcome, step=step, final_audit=receipt)


# --- query for the sbatch loop ----------------------------------------------------------------------------------
def should_start(run: str, *, max_seconds=None, config=None) -> dict:
    """Read-only: whether the sbatch loop should start another segment now.

    ``{"start": bool, "reason": "complete" | "insufficient_time" | "uniform_bootstrap_pending" |
    "audit_pending" | "fresh" | "resume", ...}`` (the command line exits 0 for start, 75 for not).
    """
    run = validate_run(run)
    cfg = load_config(config)
    paths = Paths.from_config(cfg)
    run_dir = paths.run_dir(run)
    info = retention.inspect_run_dir(run_dir)
    newest = info['newest_full_step']
    base = {'run': run, 'newest_full_step': newest}
    if newest is not None and newest >= FINAL_STEP:
        if retention.audit_valid(run_dir, run):
            return {**base, 'start': False, 'reason': 'complete'}
        return {**base, 'start': True, 'reason': 'audit_pending'}
    if run == RUN_RL and not pairing.uniform_ready(paths):
        return {**base, 'start': False, 'reason': 'uniform_bootstrap_pending',
                'remedy': 'the uniform bootstrap has not frozen input/pairing.json yet; wait for the pairing_frozen '
                          'event of train-uniform, then resubmit train-rl'}
    margin = int_value(cfg, 'HYAK_SEGMENT_MARGIN_MINUTES', STAGES[run]) * 60
    remaining = slurm.seconds_remaining()
    if stop_budget(remaining, margin, max_seconds) is None:
        return {**base, 'start': False, 'reason': 'insufficient_time', 'seconds_remaining': remaining,
                'margin_seconds': margin, 'minimum_seconds': MIN_START_SECONDS}
    return {**base, 'start': True, 'reason': 'resume' if newest is not None else 'fresh',
            'seconds_remaining': remaining if remaining is None or math.isfinite(remaining) else 'inf'}


# --- smoke ---------------------------------------------------------------------------------------------------------
def run_smoke(run: str, *, max_seconds=None, config=None, child_command=None, adapter_factory=TrainerAdapter,
              lock_wait_seconds: float = LOCK_WAIT_SECONDS) -> dict:
    """Pre-flight sequence in ``$ANYTOK_RUNS/smoke/<run>``: two child processes, then the launchers' checks.

    ``child_command`` (default ``[sys.executable, '-m', 'hyak.train.run_segment']``) starts each child.
    """
    run = validate_run(run)
    cfg = load_config(config)
    paths = Paths.from_config(cfg)
    smoke_dir = paths.smoke_dir(run)
    if smoke_dir.parent != paths.smoke_root or paths.smoke_root.parent != paths.runs:
        raise SegmentError(f'refusing to wipe {smoke_dir}: not inside {paths.smoke_root}', exit_code=EXIT_USAGE)
    emit = Emitter(STAGE_SMOKE_TRAIN, run, cfg)
    try:
        lock = retention.WrapperLock(retention.wrapper_lock_path(paths, run, True), wait_seconds=lock_wait_seconds,
                                     poll_seconds=min(LOCK_POLL_SECONDS, max(lock_wait_seconds, 0.05)),
                                     on_wait=lambda holder: emit('waiting_for_run_lock', holder=holder))
        with lock:
            if smoke_dir.exists():
                with TrainerDirLock(smoke_dir):
                    pass
                shutil.rmtree(smoke_dir)
            emit('smoke_start', directory=str(smoke_dir))
    except retention.LockHeld as exc:
        emit.warn(f'not starting the smoke: {exc}')
        emit('run_locked', error=str(exc))
        return _outcome('run_locked', step=None)
    if run == RUN_RL:
        try:
            pairing.ensure(paths, role=RUN_RL, emit=emit)
        except pairing.UniformNotReady as exc:
            emit('uniform_bootstrap_pending', reason=str(exc))
            return _outcome('uniform_bootstrap_pending', step=None, reason=str(exc))
    child_command = list(child_command or [sys.executable, '-m', 'hyak.train.run_segment'])
    results = []
    child = {'process': None}

    def forward(signum, _frame):
        if child['process'] is not None and child['process'].poll() is None:
            child['process'].send_signal(signal.SIGTERM)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for phase in ('first', 'second'):
            result_file = paths.runs / '.hyak-cache' / f'smoke-{run}-{phase}-{os.getpid()}.json'
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.unlink(missing_ok=True)
            command = child_command + ['--run', run, '--smoke-phase', phase, '--result-file', str(result_file)]
            if max_seconds is not None:
                command += ['--max-seconds', str(max_seconds)]
            if config is not None:
                command += ['--config', str(config)]
            child['process'] = subprocess.Popen(command)
            code = child['process'].wait()
            child['process'] = None
            if code != 0:
                raise SegmentError(f'smoke phase {phase} exited with status {code} (its reason is above in this log)',
                                   exit_code=code if code in (EXIT_FAILED, EXIT_USAGE, EXIT_PROTOCOL, EXIT_CHECK) else EXIT_CHECK)
            result = read_json(result_file)
            result_file.unlink(missing_ok=True)
            if result.get('outcome') != 'smoke_phase_done':
                raise SegmentError(f'smoke phase {phase} did not train: {result}')
            results.append(result)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    checks = verify_smoke(run, paths, smoke_dir, adapter_factory(paths.code_dir(run)), emit)
    checks['stop_timer_seconds'] = [result.get('stop_timer_seconds') for result in results]
    if slurm.job_id() is not None and any(value is None for value in checks['stop_timer_seconds']):
        emit.warn('the smoke phases ran without an in-process stop timer (job time limit unknown: no '
                  'SLURM_JOB_END_TIME and squeue failed); training jobs would then rely on Slurm --signal=B:TERM alone')
    emit('smoke_verified', **checks)
    # read by hyak.train.monitor --smoke: the smoke stage then ends "complete"
    atomic_write_json(smoke_dir / SMOKE_VERIFIED, {'run': run, 'at_utc': utc_now(), 'job': slurm.worker_id(), **checks})
    return {'outcome': 'smoke_verified', 'phases': results, **checks}


def verify_smoke(run, paths, smoke_dir: Path, adapter, emit) -> dict:
    """The launchers' smoke expectations (modal_anytok_pool_rl_20k.py:264-270; run_anytok_pool_rl_20k.py:165-168)."""
    full = retention.full_restarts(smoke_dir)
    if not full or full[-1][0] != SMOKE_FINAL_STEP:
        raise SegmentError(f'smoke: newest full restart is {full[-1][0] if full else None}, expected {SMOKE_FINAL_STEP}')
    retention.verify_checkpoint(smoke_dir, SMOKE_FINAL_STEP, run, retention.FULL)
    state = adapter.load_state(full[-1][1] / 'state.pt')
    _check_state(run, state, SMOKE_FINAL_STEP)
    steps = [step for step, _ in retention.step_dirs(smoke_dir)]
    expected_steps = [1, SMOKE_FINAL_STEP] if run == RUN_UNIFORM else [SMOKE_FINAL_STEP]
    if steps != expected_steps:
        raise SegmentError(f'smoke: checkpoint directories {steps}, expected {expected_steps} '
                           '(retention hook or trainer pruning misbehaved)')
    train, paused, initial = set(), [], False
    for line in (smoke_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('stage') == 'train':
            train.add(row['step'])
        elif row.get('stage') == 'paused_at':
            paused.append(row['step'])
        elif row.get('stage') == 'initial_eval':
            initial = True
    if train != set(range(1, SMOKE_FINAL_STEP + 1)) or not initial:
        raise SegmentError(f'smoke: train events for steps {sorted(train)}, initial_eval {initial}')
    if paused != [SMOKE_FIRST_STEP[run], SMOKE_FINAL_STEP]:
        raise SegmentError(f'smoke: paused_at events {paused}, expected {[SMOKE_FIRST_STEP[run], SMOKE_FINAL_STEP]}')
    checks = {'step': SMOKE_FINAL_STEP, 'checkpoints': steps, 'train_steps': sorted(train)}
    if run == RUN_RL:
        checks['policy_updates'] = state['bandit']['updates']
        checks['paired_protocol'] = validate_rl_protocol(smoke_dir, state)
        pairing_check = compare_sanity_panel(paths, smoke_dir)
        # information only (D8): the corrected cuts make the RL sanity examples differ from the uniform run's
        checks['initial_eval_equals_uniform'] = pairing_check['initial_eval_equal']
        checks['sanity_tokenizers_equal_uniform'] = pairing_check['sanity_tokenizers_equal']
        checks['initial_eval_check'] = pairing_check['check']
        if pairing_check['sanity_tokenizers_equal'] is not True:
            emit.warn('the RL smoke heldout_tokenizers.json is not the frozen uniform pair '
                      f'({paths.uniform_heldout}): the runs would not share their sanity tokenizers')
    return checks


# --- command line ----------------------------------------------------------------------------------------------------
def exit_code_for(exc: BaseException) -> int:
    """Exit status of a deliberate refusal (see the module docstring); 1 for anything unexpected."""
    if isinstance(exc, ConfigError):
        return EXIT_USAGE
    if isinstance(exc, SegmentError):
        return exc.exit_code
    if isinstance(exc, retention.ManifestMissing):
        return EXIT_PROTOCOL
    if isinstance(exc, retention.CheckpointError):
        return EXIT_CHECK
    if isinstance(exc, pairing.PairingError):
        return EXIT_PROTOCOL
    return EXIT_FAILED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.train.run_segment', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, choices=(RUN_UNIFORM, RUN_RL))
    parser.add_argument('--max-seconds', type=int, help='stop the trainer after at most this many seconds (>= 60)')
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--accept-chunk-change', metavar='REASON',
                        help='accept, on the record, a changed HYAK_HN_CHUNK_ROWS or chunked_hypernet.py mid-run')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--should-start', action='store_true', help='exit 0 if a segment should start now, 75 if not')
    mode.add_argument('--smoke', action='store_true', help='pre-flight sequence in $ANYTOK_RUNS/smoke/<run>')
    mode.add_argument('--smoke-phase', choices=('first', 'second'), help=argparse.SUPPRESS)
    parser.add_argument('--result-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_seconds is not None and args.max_seconds < MIN_MAX_SECONDS:
        parser.error(f'--max-seconds must be at least {MIN_MAX_SECONDS}')
    if args.accept_chunk_change is not None and not args.accept_chunk_change.strip():
        parser.error('--accept-chunk-change needs a reason text')
    if args.accept_chunk_change is not None and (args.should_start or args.smoke):
        parser.error('--accept-chunk-change applies to training segments only')
    try:
        faulthandler.enable()                  # native crashes (Rust helper, CUDA) leave a Python stack in the log
    except (RuntimeError, ValueError, AttributeError, OSError):
        pass
    try:
        if args.should_start:
            result = should_start(args.run, max_seconds=args.max_seconds, config=args.config)
            print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
            return EXIT_OK if result['start'] else EXIT_NOT_NOW
        if args.smoke:
            result = run_smoke(args.run, max_seconds=args.max_seconds, config=args.config)
        else:
            kwargs = {'accept_chunk_change': args.accept_chunk_change} if args.accept_chunk_change is not None else {}
            result = run_segment(args.run, smoke_phase=args.smoke_phase, max_seconds=args.max_seconds,
                                 config=args.config, **kwargs)
        if args.result_file is not None:
            atomic_write_json(args.result_file, _jsonable(result))
    except Exception as exc:  # noqa: BLE001 -- every failure is reported with its exit status
        code = exit_code_for(exc)
        if code == EXIT_FAILED:
            # unexpected (trainer, torch, transformers, Rust helper, this wrapper): the Slurm log is the only
            # place a remote reader can look, so the whole traceback goes there
            traceback.print_exc(file=sys.stderr)
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        else:
            print(f'ERROR: {exc}' if isinstance(exc, ConfigError) else f'ERROR: {type(exc).__name__}: {exc}',
                  file=sys.stderr, flush=True)
        return code
    print(json.dumps({'stage': 'segment_result', **_jsonable(result)}, sort_keys=True), flush=True)
    return EXIT_NOT_NOW if result.get('outcome') in NOT_NOW_OUTCOMES else EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
