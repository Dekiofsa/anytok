"""Pairing of the RL run with the uniform run (DESIGN D6).

The RL trainer needs two files that only the uniform trainer's first start produces
(``scripts/modal_anytok_pool_rl_20k.py:129-130`` copied them into the RL volume):

* ``<uniform run>/run_config.json``  -> ``input/uniform_run_config.json``  (the RL trainer reads
  ``<--tokenizer-pool>/../uniform_run_config.json`` and compares 21 settings with its own,
  RL trainer 414-421; 406-413 in the straight-reward variant the port runs, D8);
* ``<uniform run>/heldout_tokenizers.json`` -> ``input/uniform_heldout_tokenizers.json`` (the RL
  ``--eval-tokenizers``; its hash is ``fixed_eval_sha256`` in the RL ``run_config.json``).

The two sanity tokenizers come from an unseeded Rust RNG, so a uniform run restarted from scratch
gets a different pair and the two runs silently stop sharing a sanity panel.  Hence:

* :func:`ensure` freezes both files (atomic, hash-verified copies + ``input/pairing.json``) once the
  uniform run has a full restart, and on every later call verifies that the frozen copies and the
  uniform run's files are still byte-identical to what was frozen;
* the uniform wrapper refuses to start the uniform run from scratch once anything was frozen
  (:func:`is_frozen`);
* :func:`check_rl_settings` requires both runs to use the same staged ``hf_hypernet`` sources,
  initial hypernetwork, model files (``model_sha256``) and memory-patch setting (the trainer's own
  ``run_config.json`` hashes cover neither the hypernetwork code nor the chunked forward; gaps.md N6);
  a chunk-setting difference can be accepted on the record (``--accept-chunk-change``).

``pairing.json``::

    {"version": 1, "frozen_at_utc", "frozen_by_job", "uniform_run_dir",
     "uniform_full_restart_step_at_freeze", "model_sha256",
     "uniform_run_config": {"source", "copy", "bytes", "sha256"},
     "uniform_heldout_tokenizers": {"source", "copy", "bytes", "sha256"},
     "rl_trainer_variant": {"name", "path", "sha256", "base_path", "base_sha256", ...},
     "uniform_segment": <the uniform run's hyak_segment.json>}

``rl_trainer_variant`` names the RL trainer the uniform run is paired with (D8: the straight-BPB-reward
variant of the joint regret trainer, ``hyak.code.pinned.RL_TRAINER_VARIANT``); a record that names
another RL trainer is refused.  The RL trainer reads the uniform ``run_config.json`` only for its 21
paired settings; its sanity examples use the corrected paired cuts, so the two runs' ``initial_eval.json``
differ by design (``hyak.train.run_segment.compare_sanity_panel``).
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from hyak.code import pinned
from hyak.common.io import atomic_copy, atomic_write_json, file_record, read_json, sha256_file, utc_now
from hyak.common.paths import RUN_RL, RUN_UNIFORM
from hyak.train import retention

PAIRING_VERSION = 1
#: Settings of the two runs' hyak_segment.json that must be identical.
SHARED_SETTINGS = ('chunk_identity', 'init_sha256', 'model_sha256')
#: Staged files that only the RL code tree has (the policy and the paired-reward helper of the regret snapshot).
RL_ONLY_SOURCES = ('hf_hypernet/settings_policy.py', 'hf_hypernet/paired_reward.py')


class PairingError(RuntimeError):
    """The RL run cannot be (or is no longer) paired with the uniform run; the message says why and what to do."""


class RLDirectoryNotEmpty(PairingError):
    """The RL run directory has content although nothing was frozen."""


class UniformNotReady(PairingError):
    """Nothing is frozen yet and the uniform run has no verified full restart: the RL run must wait
    for the uniform bootstrap (not an error of either run; ``run_segment`` exits 75 'not now')."""


def _items(paths):
    uniform = paths.run_dir(RUN_UNIFORM)
    return (('uniform_run_config', uniform / 'run_config.json', paths.uniform_run_config),
            ('uniform_heldout_tokenizers', uniform / 'heldout_tokenizers.json', paths.uniform_heldout))


def is_frozen(paths) -> bool:
    """True once ``input/pairing.json`` or either frozen copy exists (the uniform run may then never start fresh)."""
    return paths.pairing.exists() or paths.uniform_run_config.exists() or paths.uniform_heldout.exists()


class _Lock:
    def __init__(self, paths):
        self.path = paths.input / '.pairing.lock'
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, 'a+')
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.handle.close()
        return False


def _has_content(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())


def load_record(paths) -> dict:
    """Parsed ``input/pairing.json`` (PairingError if it is malformed)."""
    try:
        record = read_json(paths.pairing)
    except ValueError as exc:
        raise PairingError(f'{paths.pairing} is not valid JSON ({exc}); restore it from a backup') from exc
    if (not isinstance(record, dict) or record.get('version') != PAIRING_VERSION
            or any(not isinstance(record.get(key), dict) for key, _, _ in _items(paths))
            or not isinstance(record.get('uniform_segment'), dict)):
        raise PairingError(f'{paths.pairing} is not a pairing record of version {PAIRING_VERSION}')
    return record


def verify(paths, record: dict) -> None:
    """Frozen copies and the uniform run's files must still equal the frozen hashes; the recorded RL trainer
    (when the record has one) must be the RL trainer of this checkout."""
    recorded = record.get('rl_trainer_variant')
    if isinstance(recorded, dict) and recorded.get('sha256') != pinned.RL_TRAINER_SHA256:
        raise PairingError(
            f'{paths.pairing} pairs the uniform run with the RL trainer {recorded.get("path")} (SHA-256 '
            f'{recorded.get("sha256")}), but this checkout runs {pinned.RL_TRAINER} ({pinned.RL_TRAINER_SHA256}). '
            'Remedy: restore the checkout the pair was frozen with')
    if record.get('uniform_run_dir') != str(paths.run_dir(RUN_UNIFORM)):
        raise PairingError(
            f'The pair was frozen from the uniform run at {record.get("uniform_run_dir")}, but the uniform run '
            f'is now configured at {paths.run_dir(RUN_UNIFORM)} (ANYTOK_RUNS changed?). Remedy: restore '
            'ANYTOK_RUNS in hyak/config.env; the run directories must keep one absolute path')
    for key, source, copy in _items(paths):
        expected = record[key].get('sha256')
        if not copy.is_file():
            raise PairingError(f'The frozen copy {copy} is missing (frozen {record.get("frozen_at_utc")}). '
                               f'Remedy: restore it from {source} only if that file still has SHA-256 {expected}')
        if sha256_file(copy) != expected:
            raise PairingError(f'The frozen copy {copy} changed after the freeze (expected SHA-256 {expected}). '
                               'The RL run depends on it byte for byte; restore the original file')
        if not source.is_file():
            raise PairingError(
                f'{source} is missing: the uniform run directory was wiped after the pair was frozen '
                f'({record.get("frozen_at_utc")}). The RL run would be compared with a uniform run that no '
                'longer exists. Remedy: restore the uniform run directory from a backup; do not restart the '
                'uniform run from scratch (its sanity tokenizers cannot be reproduced)')
        if sha256_file(source) != expected:
            raise PairingError(
                f'{source} differs from the frozen copy {copy}: the uniform run was restarted from scratch '
                'after the pair was frozen, so the RL run no longer shares its sanity tokenizers and settings. '
                'Remedy: restore the original uniform run directory')


def ensure(paths, *, role: str = RUN_RL, emit=None) -> dict:
    """Freeze the uniform files on first use, verify them on every later use; returns the pairing record.

    ``role='rl'`` (every RL start, including the RL smoke) and ``role='uniform'`` (after each
    successful uniform segment, so the pair is fixed as early as possible) behave the same, except
    that :class:`RLDirectoryNotEmpty` is meant to be reported as a warning by the uniform caller.
    Raises PairingError with a remedy for every refusal.
    """
    if role not in (RUN_RL, RUN_UNIFORM):
        raise ValueError(f'role must be rl or uniform, got {role!r}')
    with _Lock(paths):
        if paths.pairing.is_file():
            record = load_record(paths)
            verify(paths, record)
            return record
        uniform_dir = paths.run_dir(RUN_UNIFORM)
        rl_dir = paths.run_dir(RUN_RL)
        if _has_content(rl_dir):
            raise RLDirectoryNotEmpty(
                f'The RL run directory {rl_dir} is not empty, but {paths.pairing} does not exist: the RL run was '
                'started before the uniform pair was frozen (not through hyak.train.run_segment), so it may '
                'use other uniform files. Remedy: if the RL run has no progress worth keeping, move the '
                f'directory away (mv {rl_dir} {rl_dir}.unpaired) and start it again; otherwise restore '
                f'{paths.pairing}')
        full = retention.full_restarts(uniform_dir)
        if not full:
            raise UniformNotReady(
                f'The uniform run {uniform_dir} has no full restart yet, so its run_config.json and '
                'heldout_tokenizers.json are not final. Remedy: run the uniform bootstrap first '
                '(hyak/slurm/submit.sh train-uniform); the RL run can start after its first segment')
        newest_step, newest = full[-1]
        if not (newest / 'manifest.json').is_file():
            raise UniformNotReady(f'The newest uniform restart {newest} has no manifest yet; retry after the '
                               'uniform segment has finished writing it')
        segment_path = uniform_dir / retention.SEGMENT_RECORD
        if not segment_path.is_file():
            raise PairingError(f'{segment_path} is missing, so the uniform run settings cannot be recorded')
        try:
            segment = read_json(segment_path)
        except ValueError as exc:
            raise PairingError(str(exc)) from exc
        record = {'version': PAIRING_VERSION, 'frozen_at_utc': utc_now(),
                  'frozen_by_job': os.environ.get('SLURM_JOB_ID') or f'local-{os.getpid()}',
                  'uniform_run_dir': str(uniform_dir),
                  'uniform_full_restart_step_at_freeze': newest_step,
                  'model_sha256': segment.get('model_sha256'),
                  # how the shared initial hypernetwork was built (information only; None when its
                  # init_manifest.json was missing or did not match the files)
                  'init_recipe': segment.get('init_recipe'),
                  # the RL trainer this uniform run is paired with (D8)
                  'rl_trainer_variant': dict(pinned.RL_TRAINER_VARIANT),
                  'uniform_segment': segment}
        for key, source, copy in _items(paths):
            if not source.is_file():
                raise PairingError(f'{source} is missing although the uniform run has a full restart')
            wanted = file_record(source)
            if copy.exists():
                # A previous freeze died between the copies and pairing.json.
                if not copy.is_file() or file_record(copy) != wanted:
                    raise PairingError(
                        f'{copy} exists without {paths.pairing} and differs from {source}. Something else '
                        'wrote it; it cannot be trusted. Remedy: move it away only if no RL run has used it')
            else:
                copied = atomic_copy(source, copy)
                if copied != wanted:
                    raise PairingError(f'{source} changed while it was being copied; retry')
            record[key] = {'source': str(source), 'copy': str(copy), **wanted}
        atomic_write_json(paths.pairing, record)
        if emit is not None:
            emit('pairing_frozen', uniform_step=newest_step,
                 uniform_run_config_sha256=record['uniform_run_config']['sha256'],
                 uniform_heldout_sha256=record['uniform_heldout_tokenizers']['sha256'])
        return record


def verify_uniform_unchanged(paths) -> dict | None:
    """For the uniform wrapper: once frozen, the uniform files must still equal the frozen copies.

    Returns the record, or None when nothing is frozen yet.  A crash between the copies and
    ``pairing.json`` is completed by the next :func:`ensure`; here it only requires the copies that
    exist to equal the uniform files.
    """
    if paths.pairing.is_file():
        record = load_record(paths)
        verify(paths, record)
        return record
    for _, source, copy in _items(paths):
        if copy.exists() and (not source.is_file() or sha256_file(source) != sha256_file(copy)):
            raise PairingError(f'{copy} exists (an interrupted freeze) but differs from {source}; '
                               'the uniform run must not be restarted from scratch')
    return None


def uniform_ready(paths) -> bool:
    """Read-only: True when the pair is frozen or the uniform run has a full restart with a manifest
    (the next RL start can then freeze it); False while the RL run must wait for the uniform bootstrap."""
    if paths.pairing.is_file():
        return True
    try:
        full = retention.full_restarts(paths.run_dir(RUN_UNIFORM))
    except retention.CheckpointError:
        return True              # let the segment report the broken uniform directory
    return bool(full) and (full[-1][1] / 'manifest.json').is_file()


def check_rl_settings(record: dict, rl_segment: dict, *, ignore=()) -> list:
    """Both runs must share the hf_hypernet sources, the initial hypernetwork, the model files and the chunk
    setting.  Differences in the keys of ``ignore`` (an explicit, recorded override such as
    ``--accept-chunk-change``) are returned instead of raised."""
    uniform = record['uniform_segment']
    problems, ignored = [], []
    for key in SHARED_SETTINGS:
        if uniform.get(key) != rl_segment.get(key):
            text = f'{key}: uniform {uniform.get(key)!r}, rl {rl_segment.get(key)!r}'
            (ignored if key in ignore else problems).append(text)
    shared_u = {k: v for k, v in (uniform.get('hf_hypernet_sha256') or {}).items() if k not in RL_ONLY_SOURCES}
    shared_r = {k: v for k, v in (rl_segment.get('hf_hypernet_sha256') or {}).items() if k not in RL_ONLY_SOURCES}
    if not shared_u or shared_u != shared_r:
        differing = sorted(k for k in set(shared_u) | set(shared_r) if shared_u.get(k) != shared_r.get(k))
        problems.append(f'hf_hypernet sources differ: {differing or "uniform record has none"}')
    if problems:
        raise PairingError('The RL run does not match the uniform run it is compared with: ' + '; '.join(problems)
                           + '. Remedy: stage both code trees from the same commit (hyak.code.stage_code), use '
                           'the same input/hypernet_init, input/model and HYAK_HN_CHUNK_ROWS as the uniform run'
                           + ('. A deliberate chunk-setting difference can be accepted on the record with '
                              '--accept-chunk-change "<reason>"' if any(p.startswith('chunk_identity') for p in problems)
                              else ''))
    return ignored
