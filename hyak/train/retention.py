"""Checkpoint manifests, retention, pre-flight repair and the final grid audit of the two 20k runs.

Ported from the Modal launchers (the Modal volume commits are replaced by an fsync of every file of
the new checkpoint and of its directories, done before older restart state is compacted or deleted):

* uniform, ``scripts/modal_anytok_pool_20k.py``: ``verify_checkpoint_files`` (107-126),
  ``compact_grid_checkpoints`` (129-164), the save hook ``save_and_commit`` (298-319) and
  ``audit_final_grid`` (215-231);
* RL, ``scripts/modal_anytok_pool_rl_20k.py``: ``verify_checkpoint`` (142-155), the save hook
  ``save_and_commit`` (222-247), the replay preparation of ``train_segment`` (194-218) and
  ``audit_final`` (273-279).

The two trainers save differently and each hook is ported for its own trainer:

* the uniform trainer writes a full restart at every save (every 500, at the stop step and on a
  stop request) and prunes off-grid restarts itself (``prune_checkpoints``, uniform trainer
  188-206, protected steps 500..20000); the hook adds ``manifest.json`` and converts older grid
  checkpoints that are not multiples of 2,000 to weights-only, keeping the newest full restart;
* the RL trainer writes a full step-0 restart, weights-only checkpoints at multiples of 500 and
  full restarts at multiples of 2,000, at the stop step and on a stop request, and never prunes;
  after each new full restart the hook deletes step 0 and off-grid restarts and downgrades older
  non-milestone grid restarts to weights-only.

Result for a finished run (both): 40 grid checkpoints, full restarts at the 10 multiples of 2,000,
weights-only checkpoints at the other 30 (DESIGN D8).

Additions for a POSIX filesystem (the Modal volume never showed partial state), all stated in the
port report: stale ``.writing-*`` staging directories are removed before a segment; ``step-*``
directories newer than the newest full restart are moved to ``interrupted/<utc>/`` for the uniform
run too (the RL launcher did this; without it the uniform trainer raises ``FileExistsError`` when it
replays past a weights-only grid directory); a compaction or deletion that was interrupted half way
is completed; manifests are written atomically.

A newest full restart without ``manifest.json`` (a kill between the trainer's rename and the hook)
is adopted automatically by the pre-flight when its directory holds exactly the file set of a full
restart (the trainer's atomic rename is its completion signal); anything else still stops the
segment with the manual remedies below.

Command line (login node or job; takes the run's wrapper lock, so not while a segment runs)::

    python -m hyak.train.retention summary    --run R [--smoke]
    python -m hyak.train.retention verify     --run R [--smoke] [--step N]
    python -m hyak.train.retention audit      --run R [--smoke]
    python -m hyak.train.retention adopt      --run R [--smoke] --step N
    python -m hyak.train.retention quarantine --run R [--smoke] --step N [--force]

``verify``, ``audit`` and ``adopt`` hash whole checkpoints (23.5 GB per full restart, 493 GB for the
audit): run them in a job (``srun``/``salloc``), not on the login node.  ``quarantine`` refuses,
without ``--force``, a step below the newest full restart (the final audit needs every grid step) and
the only full restart of a run (it could then neither resume nor start).

Exit status: 0 success; 2 usage or configuration error; 4 a checkpoint check failed; 75 the run's
lock is held by a training segment or another repair command (retry later).  Unexpected errors print
a traceback and exit 1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

from hyak.common.io import atomic_write_text, file_record, fsync_dir, read_json, sha256_file, utc_now
from hyak.common.paths import FINAL_STEP, RUN_RL, RUN_UNIFORM, validate_run

#: Grid interval of saved checkpoints (modal_anytok_pool_20k.py:31).
CHECKPOINT_INTERVAL = 500
#: Interval of retained full restarts (modal_anytok_pool_20k.py:32; RL trainer: full_restart at step % 2000).
RESUME_INTERVAL = 2000
#: Weights-only file set of the uniform run (modal_anytok_pool_20k.py:33).
UNIFORM_WEIGHT_FILES = ('hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt')
#: Weights-only file set of the RL run (modal_anytok_pool_rl_20k.py:28).
RL_WEIGHT_FILES = ('hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt', 'policy.pt')
#: Files a full restart adds to the weights.
RESTART_FILES = ('optimizer.pt', 'state.pt')
FULL = 'full_restart_checkpoint'
WEIGHTS_ONLY = 'weights_only_checkpoint'
#: Trainer and policy inside a staged code tree (hashed into every manifest).
TRAINER_REL = 'scripts/train_joint_smollm_input.py'
POLICY_REL = 'hf_hypernet/settings_policy.py'
#: Run-directory files written by this port.
SEGMENT_RECORD = 'hyak_segment.json'
FINAL_AUDIT = 'final_audit.json'
INTERRUPTED = 'interrupted'

_STEP_RE = re.compile(r'step-(\d{6})')
#: Temporary file of an interrupted atomic manifest write (hyak.common.io.atomic_write_bytes).
_MANIFEST_TMP_RE = re.compile(r'\.manifest\.json\..*\.tmp')
#: Exit statuses of the command line (project-wide contract).
EXIT_USAGE = 2
EXIT_CHECK = 4
EXIT_LOCKED = 75


class CheckpointError(ValueError):
    """A checkpoint or run directory is not in the state the launchers' checks require."""


class ManifestMissing(CheckpointError):
    """The newest full restart has no manifest.json (crash between the trainer's save and the hook)."""


class AuditInterrupted(RuntimeError):
    """The final audit stopped early on request (a job's stop signal); it can simply be run again."""


def weight_files(run: str) -> frozenset:
    """Weights-only file set of a run (uniform 3 files, RL 4 with policy.pt)."""
    return frozenset(UNIFORM_WEIGHT_FILES if validate_run(run) == RUN_UNIFORM else RL_WEIGHT_FILES)


def full_files(run: str) -> frozenset:
    """File set of a full restart of a run."""
    return weight_files(run) | frozenset(RESTART_FILES)


def step_name(step: int) -> str:
    """``step-NNNNNN`` exactly as the trainers name checkpoint directories (``f"step-{step:06d}"``)."""
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 999_999:
        raise ValueError(f'step must be an integer in 0..999999, got {step!r}')
    return f'step-{step:06d}'


def grid_steps(final_step: int = FINAL_STEP) -> list:
    """The 40 grid steps 500, 1000, ..., 20000 (modal_anytok_pool_20k.py:218)."""
    return list(range(CHECKPOINT_INTERVAL, final_step + 1, CHECKPOINT_INTERVAL))


def step_dirs(run_dir) -> list:
    """``[(step, path)]`` of every ``step-*`` entry of ``run_dir``, sorted by step.

    The launchers and the RL hook parse every ``step-*`` name with ``int()`` (an unexpected name
    would crash the hook in the middle of a save), so anything that is not a ``step-NNNNNN``
    directory is rejected here, before a segment starts.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    rows = []
    for path in sorted(run_dir.glob('step-*')):
        match = _STEP_RE.fullmatch(path.name)
        if not match or not path.is_dir() or path.is_symlink():
            raise CheckpointError(
                f'Unexpected entry {path} in the run directory: only step-NNNNNN checkpoint directories '
                f'may be named step-*. Remedy: move it out of {run_dir}')
        rows.append((int(match.group(1)), path))
    return sorted(rows)


def full_restarts(run_dir) -> list:
    """``[(step, path)]`` of directories holding ``state.pt`` (what ``--resume`` picks from; trainer 431-435)."""
    return [(step, path) for step, path in step_dirs(run_dir) if (path / 'state.pt').is_file()]


def read_manifest(folder) -> dict:
    folder = Path(folder)
    path = folder / 'manifest.json'
    if not path.is_file():
        raise ManifestMissing(f'{path} is missing')
    try:
        manifest = read_json(path)
    except ValueError as exc:
        raise CheckpointError(str(exc)) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        raise CheckpointError(f'{path}: not a checkpoint manifest')
    return manifest


def _record_files(folder: Path) -> dict:
    """``{relative path: {bytes, sha256}}`` of every file of a checkpoint (manifest.json itself excluded)."""
    rows = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.relative_to(folder).as_posix() != 'manifest.json':
            rows[path.relative_to(folder).as_posix()] = file_record(path)
    return rows


def write_uniform_manifest(checkpoint, step: int, code_dir) -> dict:
    """Manifest of a uniform checkpoint (modal_anytok_pool_20k.py:300-308; written atomically here).

    Every uniform save is a full restart, so the kind is always ``full_restart_checkpoint``.
    Raises CheckpointError after writing if the file set is not the one
    ``verify_checkpoint_files`` accepts (it would refuse this checkpoint at the next resume).
    """
    checkpoint = Path(checkpoint)
    manifest = {'step': step, 'kind': FULL, 'files': _record_files(checkpoint),
                'trainer_sha256': sha256_file(Path(code_dir) / TRAINER_REL)}
    atomic_write_text(checkpoint / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    if set(manifest['files']) != full_files(RUN_UNIFORM):
        raise CheckpointError(f'{checkpoint}: unexpected checkpoint files {sorted(manifest["files"])}; '
                              f'expected {sorted(full_files(RUN_UNIFORM))}')
    return manifest


def write_rl_manifest(folder, step: int, code_dir) -> dict:
    """Manifest of an RL checkpoint (modal_anytok_pool_rl_20k.py:224-230, ``save_json`` format)."""
    folder = Path(folder)
    files = _record_files(folder)
    full = 'state.pt' in files
    manifest = {'step': step, 'kind': FULL if full else WEIGHTS_ONLY, 'files': files,
                'trainer_sha256': sha256_file(Path(code_dir) / TRAINER_REL),
                'policy_sha256': sha256_file(Path(code_dir) / POLICY_REL)}
    atomic_write_text(folder / 'manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    expected = full_files(RUN_RL) if full else weight_files(RUN_RL)
    if set(files) != expected:
        raise CheckpointError(f'{folder}: unexpected checkpoint files {sorted(files)}; expected {sorted(expected)}')
    return manifest


def verify_checkpoint(run_dir, step: int, run: str, expected_kind=None) -> dict:
    """Size + SHA-256 check of one checkpoint against its manifest.

    Port of ``verify_checkpoint_files`` (modal_anytok_pool_20k.py:107-126) and ``verify_checkpoint``
    (modal_anytok_pool_rl_20k.py:142-155); ``expected_kind=FULL`` is the RL ``require_restart``.
    Returns ``{"step", "kind", "files", "bytes", "manifest"}``.  Raises CheckpointError
    (ManifestMissing when there is no manifest).
    """
    checkpoint = Path(run_dir) / step_name(step)
    manifest = read_manifest(checkpoint)
    kind = manifest.get('kind')
    if (kind not in (FULL, WEIGHTS_ONLY) or manifest.get('step') != step
            or (expected_kind is not None and kind != expected_kind)):
        raise CheckpointError(f'Invalid checkpoint manifest: {checkpoint} (kind {kind!r}, step '
                              f'{manifest.get("step")!r}; expected {expected_kind or "any kind"} at step {step})')
    expected_names = full_files(run) if kind == FULL else weight_files(run)
    if set(manifest['files']) != expected_names:
        raise CheckpointError(f'Unexpected checkpoint files: {checkpoint} lists {sorted(manifest["files"])}')
    total = 0
    for name, expected in sorted(manifest['files'].items()):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != expected.get('bytes'):
            raise CheckpointError(f'Missing or wrong-sized checkpoint file: {path}')
        if sha256_file(path) != expected.get('sha256'):
            raise CheckpointError(f'Checkpoint checksum mismatch: {path}')
        total += expected['bytes']
    return {'step': step, 'kind': kind, 'files': len(manifest['files']), 'bytes': total, 'manifest': manifest}


def compact_grid_checkpoints(run_dir, keep_latest: bool = True) -> list:
    """Uniform retention after each save (modal_anytok_pool_20k.py:129-164, verbatim logic).

    Off-grid steps are left alone (deleted only with ``keep_latest=False``, the Modal
    ``compact-source`` action); multiples of 2,000 and the newest full restart stay full; every
    other grid checkpoint that is still full has its weight files re-hashed against its manifest,
    its manifest rewritten (atomically) to ``weights_only_checkpoint`` and its ``optimizer.pt`` and
    ``state.pt`` deleted.  Returns the changed steps.
    """
    checkpoints = [path for _, path in step_dirs(run_dir)]
    full = [path for path in checkpoints if (path / 'state.pt').is_file()]
    latest_full = full[-1] if full else None
    changed = []
    for checkpoint in checkpoints:
        step = int(checkpoint.name.removeprefix('step-'))
        if step % CHECKPOINT_INTERVAL:
            if not keep_latest:
                shutil.rmtree(checkpoint)
                changed.append(step)
            continue
        if step % RESUME_INTERVAL == 0 or (keep_latest and checkpoint == latest_full):
            continue
        manifest = read_manifest(checkpoint)
        if manifest['kind'] == WEIGHTS_ONLY:
            continue
        if manifest['kind'] != FULL:
            raise CheckpointError(f'Unknown checkpoint kind at {checkpoint}')
        for name in UNIFORM_WEIGHT_FILES:
            expected = manifest['files'][name]
            path = checkpoint / name
            if path.stat().st_size != expected['bytes'] or sha256_file(path) != expected['sha256']:
                raise CheckpointError(f'Weight checksum mismatch: {path}')
        manifest['kind'] = WEIGHTS_ONLY
        manifest['files'] = {name: manifest['files'][name] for name in UNIFORM_WEIGHT_FILES}
        atomic_write_text(checkpoint / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
        (checkpoint / 'optimizer.pt').unlink(missing_ok=True)
        (checkpoint / 'state.pt').unlink(missing_ok=True)
        changed.append(step)
    return changed


def prune_after_full_restart(run_dir, step: int) -> dict:
    """RL retention after a new full restart at ``step`` (modal_anytok_pool_rl_20k.py:232-245).

    Older directories with ``state.pt`` that are not multiples of 2,000: step 0 and off-grid
    steps are deleted; grid steps are downgraded to weights-only (manifest rewritten, then
    ``state.pt`` and ``optimizer.pt`` deleted).  Returns ``{"deleted": [...], "downgraded": [...]}``.
    """
    deleted, downgraded = [], []
    for old_step, old in step_dirs(run_dir):
        if old_step >= step or (old_step and old_step % RESUME_INTERVAL == 0) or not (old / 'state.pt').is_file():
            continue
        if old_step == 0 or old_step % CHECKPOINT_INTERVAL:
            shutil.rmtree(old)
            deleted.append(old_step)
        else:
            prior = read_manifest(old)
            prior.update(kind=WEIGHTS_ONLY, files={n: prior['files'][n] for n in RL_WEIGHT_FILES})
            atomic_write_text(old / 'manifest.json', json.dumps(prior, indent=2, ensure_ascii=False) + '\n')
            (old / 'state.pt').unlink()
            (old / 'optimizer.pt').unlink()
            downgraded.append(old_step)
    return {'deleted': deleted, 'downgraded': downgraded}


def _same_run(output, run_dir: Path) -> None:
    if os.path.normpath(str(output)) != os.path.normpath(str(run_dir)):
        raise CheckpointError(f'trainer saved into {output}, but the hook manages {run_dir}')


def fsync_checkpoint(folder) -> int:
    """Make a new checkpoint durable before older restart state is removed (replaces the Modal commits).

    ``os.fsync`` of every regular file under ``folder`` (errors propagate: the caller must then keep
    the older restart), then of every directory in it, of ``folder`` and of its parent (the
    trainer's rename).  Returns the number of files synced.
    """
    folder = Path(folder)
    count = 0
    directories = [folder]
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            count += 1
    for directory in reversed(directories):
        fsync_dir(directory)
    fsync_dir(folder.parent)
    return count


class SaveActivity:
    """Which save the trainer is in (read by run_segment's stop timer from another thread).

    ``weights_only`` is true while an RL weights-only grid save (trainer write, manifest hashing,
    fsync) runs: a stop signal then would end the run in ``paused_at`` without a full restart (RL
    trainer 652-660; 654-662 in the D8 variant), so the stop timer waits until it is over.
    """

    def __init__(self):
        self.step = None
        self.weights_only = False

    def begin(self, step, full_restart: bool):
        self.step = step
        self.weights_only = not full_restart

    def end(self):
        self.weights_only = False
        self.step = None


def uniform_save_hook(original_save, run_dir, code_dir, *, emit=None, before_manifest=None,
                      clock=time.monotonic, save_activity=None):
    """Wrap the uniform trainer's ``save_checkpoint`` (port of modal_anytok_pool_20k.py:296-319).

    ``original_save(output, step, lm, hn, optimizer, corpus, queue_batches, py_rng, noise_rng,
    metrics, protected_steps=())`` is called unchanged (it prunes off-grid restarts itself), then
    the manifest is written and the grid compacted.  ``before_manifest(step)`` runs before the
    manifest (used for the run's first ``hyak_segment.json``); ``emit(name, **fields)`` receives
    ``compacted_checkpoints`` and ``committed_checkpoint`` exactly as the launcher printed them.
    """
    run_dir, code_dir = Path(run_dir), Path(code_dir)
    start = clock()

    def save_and_record(output, step, *args, **kwargs):
        _same_run(output, run_dir)
        if save_activity is not None:
            save_activity.begin(step, True)
        try:
            original_save(output, step, *args, **kwargs)
            checkpoint = Path(output) / step_name(step)
            if before_manifest is not None:
                before_manifest(step)
            write_uniform_manifest(checkpoint, step, code_dir)
            fsync_checkpoint(checkpoint)
        finally:
            if save_activity is not None:
                save_activity.end()
        compacted = compact_grid_checkpoints(run_dir)
        if compacted and emit is not None:
            emit('compacted_checkpoints', steps=compacted)
        if emit is not None:
            emit('committed_checkpoint', step=step, full_restart=True,
                 elapsed_seconds=round(clock() - start, 2))

    save_and_record.hyak_wrapped = original_save
    return save_and_record


def rl_save_hook(original_save, run_dir, code_dir, *, emit=None, before_manifest=None, save_activity=None):
    """Wrap the RL trainer's ``save_checkpoint`` (port of modal_anytok_pool_rl_20k.py:221-247).

    ``original_save(output, step, lm, hn, optimizer, corpus, queue_batches, py_rng, noise_rng,
    metrics, protected_steps=(), bandit=None, full_restart=True)`` is called unchanged (it never
    prunes); the manifest kind follows the presence of ``state.pt``; after a full restart older
    restart state is pruned (``prune_after_full_restart``).  ``save_activity`` (a
    :class:`SaveActivity`) is told while a weights-only save runs.
    """
    run_dir, code_dir = Path(run_dir), Path(code_dir)

    def save_and_record(output, step, *args, **kwargs):
        _same_run(output, run_dir)
        if save_activity is not None:
            save_activity.begin(step, bool(kwargs.get('full_restart', True)))
        try:
            original_save(output, step, *args, **kwargs)
            folder = Path(output) / step_name(step)
            if before_manifest is not None:
                before_manifest(step)
            manifest = write_rl_manifest(folder, step, code_dir)
            fsync_checkpoint(folder)
        finally:
            if save_activity is not None:
                save_activity.end()
        full = manifest['kind'] == FULL
        if full:
            pruned = prune_after_full_restart(run_dir, step)
            if emit is not None and (pruned['deleted'] or pruned['downgraded']):
                emit('pruned_restarts', step=step, **pruned)
        if emit is not None:
            emit('committed_checkpoint', step=step, full_restart=full)

    save_and_record.hyak_wrapped = original_save
    return save_and_record


def save_hook(run: str, original_save, run_dir, code_dir, **kwargs):
    """The hook of ``run`` (``uniform_save_hook`` or ``rl_save_hook``)."""
    if validate_run(run) == RUN_UNIFORM:
        return uniform_save_hook(original_save, run_dir, code_dir, **kwargs)
    return rl_save_hook(original_save, run_dir, code_dir, **kwargs)


# --- pre-flight ------------------------------------------------------------------------------------
def interrupted_label() -> str:
    """``interrupted/<label>``: UTC time with microseconds (the RL launcher used ``time.time_ns()``)."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


@dataclass
class Preflight:
    """What :func:`preflight` found and changed."""

    newest_full_step: int | None = None
    newest_full_bytes: int | None = None
    removed_writing: list = field(default_factory=list)
    finished_compactions: list = field(default_factory=list)
    finished_deletions: list = field(default_factory=list)
    moved_interrupted: list = field(default_factory=list)
    interrupted_dir: str | None = None
    steps_without_restart: list = field(default_factory=list)
    removed_manifest_tmp: list = field(default_factory=list)
    adopted: int | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _plain_names(folder: Path) -> set:
    """Relative paths of the files of a checkpoint directory (no hashing; manifest.json excluded)."""
    return {path.relative_to(folder).as_posix() for path in folder.rglob('*')
            if path.is_file() and path.relative_to(folder).as_posix() != 'manifest.json'}


def remove_manifest_temporaries(folder: Path) -> list:
    """Delete ``.manifest.json.*.tmp`` left by a kill inside an atomic manifest write; returns their names."""
    removed = []
    for path in sorted(Path(folder).glob('.manifest.json.*.tmp')):
        if _MANIFEST_TMP_RE.fullmatch(path.name) and path.is_file() and not path.is_symlink():
            path.unlink()
            removed.append(path.name)
    return removed


def _no_restart_left(run: str, run_dir: Path, suffix: str) -> str:
    """Remedy text when the newest full restart is the only one (quarantine would leave none)."""
    if run == RUN_RL:
        return (f'it is the only full restart of the run, so quarantining it would leave a run that can neither '
                f'resume nor start. Remedy: move the RL run directory away (mv {run_dir} {run_dir}.broken) and '
                f'start the RL run again from scratch (it uses the frozen uniform pair); inspect it first with '
                f'python -m hyak.train.retention summary --run {run}{suffix}')
    return (f'it is the only full restart of the run, so quarantining it would leave a run that can neither resume '
            f'nor start, and the uniform run must not start from scratch once the pair is frozen. Remedy: restore '
            f'{run_dir} from a backup, or begin a new experiment by moving both run directories, '
            f'input/pairing.json and input/uniform_*.json away together (see the run guide); inspect it first with '
            f'python -m hyak.train.retention summary --run {run}{suffix}')


def preflight(run_dir, run: str, *, verify_newest: bool = True, label: str | None = None,
              adopt_code_dir=None, remedy_suffix: str = '') -> Preflight:
    """Repair what a hard kill can leave behind, verify the newest full restart, move later saves aside.

    Call only while holding the run's wrapper lock and the trainer's ``.run.lock``.  In order:

    1. reject unexpected ``step-*`` entries (:func:`step_dirs`);
    2. delete stale ``.writing-*`` staging directories (a killed save; trainer 216) and stale
       ``.manifest.json.*.tmp`` files in step directories (a kill inside an atomic manifest write);
    3. finish interrupted compactions/downgrades: a manifest that already says weights-only but
       ``state.pt`` / ``optimizer.pt`` still present (modal_anytok_pool_20k.py:156-162 and
       modal_anytok_pool_rl_20k.py:241-244 rewrite the manifest first, then delete);
    4. finish interrupted deletions of restarts older than the newest one: step 0 or an off-grid
       step without ``state.pt`` (every off-grid or step-0 save of either trainer is a full
       restart, so such a directory is the remainder of ``shutil.rmtree``);
    5. verify the newest full restart against its manifest (``verified_restart`` in both launchers;
       raises :class:`ManifestMissing` / :class:`CheckpointError`).  With ``adopt_code_dir``, a
       newest full restart without a manifest whose directory holds exactly the file set of a full
       restart (and a run with ``hyak_segment.json``) is adopted instead (``adopted``): the
       trainer renames a checkpoint into place only after writing all of it;
    6. move every ``step-*`` newer than it to ``interrupted/<label>/`` (modal_anytok_pool_rl_20k.py:211-218).

    With no full restart nothing is verified or moved; ``steps_without_restart`` lists the step
    directories found (the caller decides whether the directory may be cleared).  ``remedy_suffix``
    (``' --smoke'`` for smoke runs) is appended to the commands named in error messages.
    """
    run_dir = Path(run_dir)
    result = Preflight()
    if not run_dir.is_dir():
        return result
    steps = step_dirs(run_dir)
    for path in sorted(run_dir.glob('.writing-*')):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        result.removed_writing.append(path.name)
    for step, path in steps:
        result.removed_manifest_tmp += [f'{path.name}/{name}' for name in remove_manifest_temporaries(path)]
    for step, path in steps:
        manifest_path = path / 'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = read_manifest(path)
        if manifest.get('kind') == WEIGHTS_ONLY and any((path / name).exists() for name in RESTART_FILES):
            for name in RESTART_FILES:
                (path / name).unlink(missing_ok=True)
            result.finished_compactions.append(step)
    full = full_restarts(run_dir)
    if not full:
        result.steps_without_restart = [step for step, _ in step_dirs(run_dir)]
        return result
    newest_step, newest = full[-1]
    for step, path in step_dirs(run_dir):
        if step < newest_step and (step == 0 or step % CHECKPOINT_INTERVAL) and not (path / 'state.pt').is_file():
            shutil.rmtree(path)
            result.finished_deletions.append(step)
    result.newest_full_step = newest_step
    if verify_newest:
        suffix = remedy_suffix
        older = len(full) > 1
        quarantine_cmd = f'python -m hyak.train.retention quarantine --run {run}{suffix} --step {newest_step}'
        try:
            checked = verify_checkpoint(run_dir, newest_step, run, FULL)
        except ManifestMissing as exc:
            names = _plain_names(newest)
            complete = names == full_files(run)
            if complete and adopt_code_dir is not None and (run_dir / SEGMENT_RECORD).is_file():
                manifest = adopt(run_dir, run, newest_step, adopt_code_dir)
                result.adopted = newest_step
                result.newest_full_bytes = sum(entry['bytes'] for entry in manifest['files'].values())
                checked = None
            elif complete:
                raise ManifestMissing(
                    f'The newest full restart {newest} has no manifest.json: the job was killed between the '
                    f'trainer writing it and the manifest hook. Its files are complete (the trainer renames '
                    f'the directory atomically) but cannot be verified. Remedy, either: '
                    f'python -m hyak.train.retention adopt --run {run}{suffix} --step {newest_step}   (hash it as is)'
                    + (f', or {quarantine_cmd}   (resume from the previous restart)' if older else '')) from exc
            else:
                raise ManifestMissing(
                    f'The newest full restart {newest} has no manifest.json and holds only {sorted(names)}, not the '
                    f'file set of a full restart: it is the remainder of an interrupted deletion (the trainer '
                    f'prunes older restarts) or write, not a checkpoint. '
                    + (f'Remedy: {quarantine_cmd}   (resume from the previous restart)' if older
                       else 'It cannot be adopted, and ' + _no_restart_left(run, run_dir, suffix))) from exc
        except CheckpointError as exc:
            raise CheckpointError(
                f'The newest full restart {newest} failed verification: {exc}. Remedy: investigate the storage; '
                + (f'to resume from the previous restart run: {quarantine_cmd}' if older
                   else _no_restart_left(run, run_dir, suffix))) from exc
        if checked is not None:
            result.newest_full_bytes = checked['bytes']
    later = [(step, path) for step, path in step_dirs(run_dir) if step > newest_step]
    if later:
        label = label or interrupted_label()
        archive = run_dir / INTERRUPTED / label
        archive.mkdir(parents=True, exist_ok=False)
        for step, path in later:
            shutil.move(str(path), str(archive / path.name))
            result.moved_interrupted.append(step)
        result.interrupted_dir = str(archive)
    return result


def inspect_run_dir(run_dir) -> dict:
    """Read-only summary: ``exists``, ``entries``, ``steps`` [{step, full, manifest}], ``newest_full_step``,
    ``final_audit`` (parsed or None), ``segment_record`` (bool)."""
    run_dir = Path(run_dir)
    info = {'path': str(run_dir), 'exists': run_dir.is_dir(), 'entries': [], 'steps': [],
            'newest_full_step': None, 'final_audit': None, 'segment_record': False}
    if not info['exists']:
        return info
    info['entries'] = sorted(p.name for p in run_dir.iterdir())
    for step, path in step_dirs(run_dir):
        info['steps'].append({'step': step, 'full': (path / 'state.pt').is_file(),
                              'manifest': (path / 'manifest.json').is_file()})
    full = [row['step'] for row in info['steps'] if row['full']]
    info['newest_full_step'] = full[-1] if full else None
    audit = run_dir / FINAL_AUDIT
    if audit.is_file():
        try:
            info['final_audit'] = read_json(audit)
        except ValueError:
            info['final_audit'] = None
    info['segment_record'] = (run_dir / SEGMENT_RECORD).is_file()
    return info


def audit_valid(run_dir, run: str) -> bool:
    """True when ``final_audit.json`` exists and describes the complete 40-checkpoint grid at 20,000."""
    path = Path(run_dir) / FINAL_AUDIT
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get('final_step') != FINAL_STEP:
        return False
    key = 'checkpoint_count' if validate_run(run) == RUN_UNIFORM else 'checkpoints'
    if data.get(key) != len(grid_steps()):
        return False
    # a receipt that names the manifest of step 20,000 is stale once that checkpoint was replaced
    recorded = data.get('final_manifest_sha256')
    return recorded is None or recorded == _final_manifest_sha(run_dir)


def _final_manifest_sha(run_dir):
    path = Path(run_dir) / step_name(FINAL_STEP) / 'manifest.json'
    try:
        return sha256_file(path)
    except OSError:
        return None


def _audit_preconditions(run_dir: Path, run: str, steps, input_manifest) -> None:
    """Cheap checks before any hashing: every grid manifest exists with the kind the rule accepts.

    Raises one CheckpointError listing every missing step and every rejected kind (uniform: exact
    kinds; RL: a full restart at multiples of 2,000) and, for the uniform run, a missing input manifest.
    """
    if run == RUN_UNIFORM and (input_manifest is None or not Path(input_manifest).is_file()):
        raise CheckpointError(f'input manifest {input_manifest} is missing; the uniform audit records its hash')
    missing, wrong = [], []
    for step in steps:
        try:
            manifest = read_manifest(Path(run_dir) / step_name(step))
        except ManifestMissing:
            missing.append(step)
            continue
        if run == RUN_UNIFORM:
            expected = FULL if step % RESUME_INTERVAL == 0 else WEIGHTS_ONLY
        else:
            expected = FULL if step % RESUME_INTERVAL == 0 else None
        kind = manifest.get('kind')
        if manifest.get('step') != step or (expected is not None and kind != expected) or kind not in (FULL, WEIGHTS_ONLY):
            wrong.append(f'step-{step:06d} (kind {kind!r}, step {manifest.get("step")!r}; expected '
                         f'{expected or "any kind"} at step {step})')
    if missing or wrong:
        parts = []
        if missing:
            parts.append(f'{len(missing)} grid checkpoints have no manifest: {missing[:12]}{" ..." if len(missing) > 12 else ""}')
        if wrong:
            parts.append('wrong kind: ' + '; '.join(wrong[:12]))
        raise CheckpointError(f'{run_dir} is not a complete {FINAL_STEP}-update grid ({"; ".join(parts)}). '
                              'A grid step of the wrong kind right after the last save is completed by the next '
                              f'train-{run} job (it finishes the retention step of the last save, then audits)')


def finish_retention(run_dir, run: str) -> dict:
    """Idempotent retention pass for the newest full restart (the hook's last step, after a kill in it).

    Returns ``{}`` unless the newest directory with ``state.pt`` has a manifest; uniform:
    ``{"compacted": [...]}`` (:func:`compact_grid_checkpoints`); RL: ``{"deleted", "downgraded"}``
    (:func:`prune_after_full_restart`).
    """
    run_dir = Path(run_dir)
    full = full_restarts(run_dir)
    if not full or not (full[-1][1] / 'manifest.json').is_file():
        return {}
    if validate_run(run) == RUN_UNIFORM:
        return {'compacted': compact_grid_checkpoints(run_dir)}
    return prune_after_full_restart(run_dir, full[-1][0])


def audit_final(run_dir, run: str, input_manifest=None, *, should_stop=None) -> dict:
    """Final grid audit of a finished run, written to ``<run_dir>/final_audit.json``.

    uniform (modal_anytok_pool_20k.py:215-231): every grid step verified, full restart exactly at
    multiples of 2,000 and weights-only elsewhere; receipt ``checkpoint_count``,
    ``full_checkpoint_count``, ``final_step``, ``checkpoint_bytes``, ``input_manifest_sha256``.
    RL (modal_anytok_pool_rl_20k.py:273-279): every grid step verified, a full restart required at
    multiples of 2,000; receipt ``checkpoints``, ``final_step``, ``bytes``.  Both receipts here also
    carry ``audited_at_utc`` and ``extra_step_dirs`` (step directories outside the grid; the
    original runs had none).  ``should_stop()`` is checked before each checkpoint; when it returns
    True the audit raises :class:`AuditInterrupted` and writes nothing.
    """
    run_dir = Path(run_dir)
    run = validate_run(run)
    steps = grid_steps()
    _audit_preconditions(run_dir, run, steps, input_manifest)

    def check(step, expected):
        if should_stop is not None and should_stop():
            raise AuditInterrupted(f'stop requested before checking step {step}')
        return verify_checkpoint(run_dir, step, run, expected)

    extra = [step for step, _ in step_dirs(run_dir) if step not in set(steps)]
    if run == RUN_UNIFORM:
        total = 0
        full_count = 0
        for step in steps:
            expected = FULL if step % RESUME_INTERVAL == 0 else WEIGHTS_ONLY
            checked = check(step, expected)
            total += checked['bytes']
            full_count += checked['kind'] == FULL
        receipt = {'checkpoint_count': FINAL_STEP // CHECKPOINT_INTERVAL,
                   'full_checkpoint_count': FINAL_STEP // RESUME_INTERVAL,
                   'final_step': FINAL_STEP, 'checkpoint_bytes': total,
                   'input_manifest_sha256': sha256_file(input_manifest)}
        if full_count != receipt['full_checkpoint_count']:
            raise CheckpointError(f'{full_count} full restarts in the grid, expected {receipt["full_checkpoint_count"]}')
        receipt.update(audited_at_utc=utc_now(), extra_step_dirs=extra,
                       final_manifest_sha256=_final_manifest_sha(run_dir))
        atomic_write_text(run_dir / FINAL_AUDIT, json.dumps(receipt, indent=2) + '\n')
        return receipt
    manifests = [check(step, FULL if step % RESUME_INTERVAL == 0 else None) for step in steps]
    receipt = {'checkpoints': len(manifests), 'final_step': FINAL_STEP,
               'bytes': sum(f['bytes'] for m in manifests for f in m['manifest']['files'].values()),
               'full_checkpoint_count': sum(m['kind'] == FULL for m in manifests),
               'audited_at_utc': utc_now(), 'extra_step_dirs': extra,
               'final_manifest_sha256': _final_manifest_sha(run_dir)}
    atomic_write_text(run_dir / FINAL_AUDIT, json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    return receipt


# --- manual repairs (CLI) ----------------------------------------------------------------------------
def adopt(run_dir, run: str, step: int, code_dir) -> dict:
    """Write the missing manifest of a complete checkpoint (after a kill between save and hook).

    Requires the exact file set of a full restart (or, RL only, of a weights-only checkpoint) and
    ``hyak_segment.json`` in the run directory (it is written before the first manifest).
    """
    run_dir = Path(run_dir)
    folder = run_dir / step_name(step)
    if not folder.is_dir():
        raise CheckpointError(f'{folder} does not exist')
    if (folder / 'manifest.json').exists():
        raise CheckpointError(f'{folder}/manifest.json already exists; nothing to adopt')
    if not (run_dir / SEGMENT_RECORD).is_file():
        raise CheckpointError(f'{run_dir / SEGMENT_RECORD} is missing, so the run settings of this checkpoint '
                              'cannot be proven; quarantine it instead')
    remove_manifest_temporaries(folder)
    names = _plain_names(folder)
    allowed = [full_files(run)] + ([weight_files(run)] if run == RUN_RL else [])
    if names not in allowed:
        raise CheckpointError(f'{folder} holds {sorted(names)}, not a complete checkpoint; quarantine it instead')
    if run == RUN_UNIFORM:
        manifest = write_uniform_manifest(folder, step, code_dir)
    else:
        manifest = write_rl_manifest(folder, step, code_dir)
    fsync_checkpoint(folder)
    return manifest


def quarantine(run_dir, step: int, label: str | None = None, *, force: bool = False) -> Path:
    """Move ``step-NNNNNN`` to ``interrupted/manual-<utc>/`` (the next segment resumes from the previous restart).

    Without ``force`` it refuses a step below the newest full restart (the final audit needs every
    grid step, and the run would not fall back to it anyway) and the only full restart of a run.
    """
    run_dir = Path(run_dir)
    folder = run_dir / step_name(step)
    if not folder.is_dir():
        raise CheckpointError(f'{folder} does not exist')
    full = [s for s, _ in full_restarts(run_dir)]
    if not force and full and step < full[-1]:
        raise CheckpointError(
            f'step {step} is below the newest full restart (step {full[-1]}): the run does not resume from it, '
            'and moving a grid checkpoint away makes the final audit fail. Did you mean the newest restart? '
            'Pass --force only if this is really intended')
    if not force and full == [step]:
        raise CheckpointError(
            f'step {step} is the only full restart of {run_dir}: without it the run can neither resume nor '
            'start. Pass --force only if you will then restore or restart the run as the run guide says')
    archive = run_dir / INTERRUPTED / f'manual-{label or interrupted_label()}'
    archive.mkdir(parents=True, exist_ok=False)
    target = archive / folder.name
    shutil.move(str(folder), str(target))
    return target


def wrapper_lock_path(paths, run: str, smoke: bool = False) -> Path:
    """Lock file held by every process that changes a run directory (run_segment, this CLI)."""
    return paths.runs / '.locks' / f'{"smoke-" if smoke else ""}{validate_run(run)}.lock'


class LockHeld(RuntimeError):
    """The run's wrapper lock is held by another process (after the optional wait)."""


class WrapperLock:
    """Exclusive ``flock`` on the run's wrapper lock file; raises :class:`LockHeld` when held.

    ``wait_seconds`` > 0 retries every ``poll_seconds`` until then (a predecessor job that is still
    being killed, or a repair command that is hashing); ``on_wait(holder)`` is called once before the
    first wait with the holder record read from the file; ``should_stop()`` ends the wait early.
    """

    def __init__(self, path, *, wait_seconds: float = 0, poll_seconds: float = 5.0, on_wait=None, should_stop=None):
        self.path = Path(path)
        self.handle = None
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self.on_wait = on_wait
        self.should_stop = should_stop
        self.waited = 0.0

    def _holder(self):
        try:
            return json.loads(self.path.read_text().strip() or 'null')
        except (OSError, ValueError):
            return None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, 'a+')
        started = time.monotonic()
        told = False
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                self.waited = time.monotonic() - started
                stop = self.should_stop is not None and self.should_stop()
                if stop or self.waited >= self.wait_seconds:
                    holder = self._holder()
                    self.handle.close()
                    self.handle = None
                    raise LockHeld(f'{self.path} is held by another process ({holder}): a training segment or repair '
                                   'command is already working on this run' +
                                   (f' (waited {self.waited:.0f} s)' if self.wait_seconds else '') +
                                   '. Wait for it (squeue -u $USER) and retry') from exc
                if self.on_wait is not None and not told:
                    told = True
                    self.on_wait(self._holder())
                time.sleep(min(self.poll_seconds, max(0.0, self.wait_seconds - self.waited)) or 0.01)
        self.waited = time.monotonic() - started
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({'pid': os.getpid(), 'at_utc': utc_now(),
                                      'job': os.environ.get('SLURM_JOB_ID')}) + '\n')
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.train.retention', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('summary', 'verify', 'audit', 'adopt', 'quarantine'))
    parser.add_argument('--run', required=True, choices=(RUN_UNIFORM, RUN_RL))
    parser.add_argument('--smoke', action='store_true', help='act on $ANYTOK_RUNS/smoke/<run>')
    parser.add_argument('--step', type=int, help='checkpoint step (verify: default the newest full restart)')
    parser.add_argument('--force', action='store_true',
                        help='quarantine: also a step below the newest full restart or the only full restart')
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    args = parser.parse_args(argv)
    if args.command in ('adopt', 'quarantine') and args.step is None:
        parser.error(f'{args.command} needs --step')
    from hyak.common.config import ConfigError, load_config
    from hyak.common.paths import Paths
    try:
        paths = Paths.from_config(load_config(args.config))
        run_dir = paths.smoke_dir(args.run) if args.smoke else paths.run_dir(args.run)
        if args.command == 'summary':
            result = inspect_run_dir(run_dir)
        else:
            with WrapperLock(wrapper_lock_path(paths, args.run, args.smoke)):
                if args.command == 'verify':
                    step = args.step
                    if step is None:
                        full = full_restarts(run_dir)
                        if not full:
                            raise CheckpointError(f'{run_dir} has no full restart')
                        step = full[-1][0]
                    result = verify_checkpoint(run_dir, step, args.run)
                    result.pop('manifest')
                elif args.command == 'audit':
                    result = audit_final(run_dir, args.run, paths.input_manifest)
                elif args.command == 'adopt':
                    result = adopt(run_dir, args.run, args.step, paths.code_dir(args.run))
                else:
                    before = [s for s, _ in full_restarts(run_dir)]
                    target = quarantine(run_dir, args.step, force=args.force)
                    after = [s for s, _ in full_restarts(run_dir)]
                    resume = after[-1] if after else None
                    given_up = (before[-1] - resume) if before and resume is not None and before[-1] > resume else 0
                    folder = run_dir / step_name(args.step)
                    result = {'moved_to': str(target), 'resume_step': resume, 'updates_given_up': given_up,
                              'undo': f'mv {target} {folder}'}
                    print(f'quarantined step {args.step}: the run resumes from step {resume} '
                          f'({given_up} updates are trained again). Undo with: mv {target} {folder}', file=sys.stderr)
    except LockHeld as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return EXIT_LOCKED
    except CheckpointError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return EXIT_CHECK
    except (ConfigError, ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 -- unexpected: full traceback, exit 1
        traceback.print_exc(file=sys.stderr)
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
