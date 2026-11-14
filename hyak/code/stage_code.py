"""Stage the two training code trees and write ``code/code_manifest.json`` (DESIGN.md 5.A).

    python -m hyak.code.stage_code [--force] [--rust-library committed|PATH [--rust-provenance JSON]]
    python -m hyak.code.stage_code --verify [--run uniform|rl]
    python -m hyak.code.stage_code --show

Builds ``$ANYTOK_ROOT/code/uniform`` and ``$ANYTOK_ROOT/code/rl`` exactly as the Modal training
images laid out ``/opt/zett`` (``hyak.code.pinned.tree_layout``; ``scripts/modal_anytok_pool_20k.py:37-50``,
``scripts/modal_anytok_pool_rl_20k.py:56-67`` with ``ANYTOK_RL_EXPERIMENT=regret``):

* uniform: ``hf_hypernet/`` (10 files) + ``scripts/train_joint_smollm_input.py`` of
  ``runtime/anytok_pool_20k`` + ``rust_utils/target/release/librust_utils.so`` (the original run);
* rl: the same ``hf_hypernet/`` + ``hf_hypernet/settings_policy.py`` of
  ``runtime/anytok_pool_settings_rl_20k`` + ``hf_hypernet/paired_reward.py`` of
  ``runtime/anytok_pool_regret_rl_20k`` + the same ``.so``, with the port's straight-BPB-reward variant
  of the regret trainer (``hyak/variants/anytok_pool_straight_rl_20k``) at
  ``scripts/train_joint_smollm_input.py``.

Every file is a real copy (never a symlink: the trainer resolves its own path,
``train_joint_smollm_input.py:22``), hash-verified against ``hyak.code.pinned`` before and after
copying; nothing else is staged (no ``__pycache__``).  Staged files are made read-only
(0444, directories 0555) so nobody edits a tree under a running experiment and Python cannot
add bytecode caches to it.

The manifest records bytes + SHA-256 of every staged file, the repository commit, the
Rust library's origin and the SHA-256 of ``hyak/memory/chunked_hypernet.py`` (``null`` when
that file does not exist at staging time).  Its ``identity`` part (file hashes, Rust library,
chunked-forward hash) is what resume pairing depends on, summarised as ``identity_sha256``;
per-tree ``tree_sha256`` values are also recorded.

Re-running is a no-op when the stage is identical (the manifest is not rewritten, even if
the repository commit changed).  Any difference to an existing stage is refused unless
``--force``, and ``--force`` is refused while ``$ANYTOK_RUNS/uniform`` or ``$ANYTOK_RUNS/rl``
is non-empty (a changed tree would break resume pairing).

``--rust-library PATH`` stages a rebuilt library (hyak/env/build_rust_utils.sh); without the
option an existing stage keeps the library it has, a new stage uses the committed one.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from hyak.code import pinned
from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now

MANIFEST_KIND = 'hyak_code_manifest'
MANIFEST_SCHEMA = 1
RUNS = ('uniform', 'rl')
#: Memory-saving hypernetwork forward (D2), hashed into the manifest when present.
CHUNKED_HYPERNET_REL = 'hyak/memory/chunked_hypernet.py'
#: Directories a staged tree contains (nothing else is allowed).
TREE_DIRS = ('hf_hypernet', 'rust_utils', 'rust_utils/target', 'rust_utils/target/release', 'scripts')
FILE_MODE = 0o444
DIR_MODE = 0o555
SOURCE_COMMITTED = 'committed'
SOURCE_REBUILT = 'rebuilt'


class StageError(RuntimeError):
    """The stage cannot be built or verified; the message says why and what to do."""


# --- small helpers ----------------------------------------------------------------------------
def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _record(path: Path) -> dict:
    size = path.stat().st_size
    digest = sha256_file(path)
    if path.stat().st_size != size:
        raise StageError(f'{path} changed while it was hashed; retry when nothing writes it')
    return {'bytes': size, 'sha256': digest}


def tree_sha256(files: dict) -> str:
    """Digest of a tree's ``{relative path: {"bytes", "sha256", ...}}`` (only bytes and sha256 count)."""
    return _digest({rel: {'bytes': v['bytes'], 'sha256': v['sha256']} for rel, v in sorted(files.items())})


def chunked_hypernet_record(repo) -> dict:
    """``{"path", "sha256", "bytes"}`` of ``hyak/memory/chunked_hypernet.py`` (``None`` values when absent)."""
    path = Path(repo) / CHUNKED_HYPERNET_REL
    if path.is_file():
        return {'path': CHUNKED_HYPERNET_REL, **_record(path)}
    return {'path': CHUNKED_HYPERNET_REL, 'sha256': None, 'bytes': None}


def _git(repo, *args):
    try:
        result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True,
                                timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()[-300:]
    return result.stdout, None


def repository_record(repo) -> dict:
    """Commit of the checkout and the changed paths under ``runtime/`` and ``hyak/`` (provenance only)."""
    record = {'path': str(repo), 'commit': None, 'dirty_paths': None, 'error': None}
    out, error = _git(repo, 'rev-parse', 'HEAD')
    if out is None:
        record['error'] = f'git rev-parse failed: {error}'
        return record
    record['commit'] = out.strip()
    out, error = _git(repo, 'status', '--porcelain=v1', '--untracked-files=all', '--', 'runtime', 'hyak')
    if out is None:
        record['error'] = f'git status failed: {error}'
    else:
        paths = sorted(line[3:] for line in out.splitlines() if len(line) > 3)
        record['dirty_paths'] = paths[:200]
        record['dirty_path_count'] = len(paths)
    return record


def identity(manifest: dict) -> dict:
    """The part of a manifest that resume pairing depends on (no timestamps, commits or paths)."""
    return {
        'trees': {run: {rel: {'bytes': v['bytes'], 'sha256': v['sha256']}
                        for rel, v in sorted(manifest['trees'][run]['files'].items())}
                  for run in sorted(manifest['trees'])},
        'rust_library': {k: manifest['rust_library'][k] for k in ('source', 'sha256', 'bytes')},
        'chunked_hypernet_sha256': manifest['chunked_hypernet']['sha256'],
    }


def identity_sha256(manifest: dict) -> str:
    """SHA-256 of :func:`identity` (canonical JSON)."""
    return _digest(identity(manifest))


def hf_hypernet_hashes(manifest: dict, run: str) -> dict:
    """``{"hf_hypernet/<file>": sha256}`` of one staged tree, for run pairing checks."""
    return {rel: v['sha256'] for rel, v in sorted(manifest['trees'][run]['files'].items())
            if rel.startswith('hf_hypernet/')}


def check_pairing(manifest: dict) -> list:
    """Problems if the two trees do not share the same hf_hypernet files and Rust library."""
    problems = []
    if not all(run in manifest.get('trees', {}) for run in RUNS):
        return ['manifest does not contain both trees']
    uniform, rl = hf_hypernet_hashes(manifest, 'uniform'), hf_hypernet_hashes(manifest, 'rl')
    for rel, sha in uniform.items():
        if rl.get(rel) != sha:
            problems.append(f'{rel} differs between the uniform and rl trees')
    extra = sorted(set(rl) - set(uniform) - set(pinned.RL_ONLY_RELS))
    if extra:
        problems.append(f'rl tree has unexpected hf_hypernet files: {extra}')
    files = {run: manifest['trees'][run]['files'] for run in RUNS}
    if files['uniform'].get(pinned.RUST_LIBRARY_REL) != files['rl'].get(pinned.RUST_LIBRARY_REL):
        problems.append('librust_utils.so differs between the uniform and rl trees')
    return problems


def scan_tree(tree: Path):
    """``({relative path: {"bytes", "sha256"}}, [problems])`` of a directory, never following links."""
    files, problems = {}, []
    for directory, dirnames, filenames in os.walk(tree, followlinks=False):
        base = Path(directory)
        for name in sorted(dirnames):
            path = base / name
            rel = path.relative_to(tree).as_posix()
            if path.is_symlink():
                problems.append(f'{rel}: symlinked directory (staged trees hold real files only)')
            elif rel not in TREE_DIRS:
                problems.append(f'{rel}/: unexpected directory')
        dirnames[:] = sorted(d for d in dirnames if not (base / d).is_symlink())
        for name in sorted(filenames):
            path = base / name
            rel = path.relative_to(tree).as_posix()
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                problems.append(f'{rel}: symlink (staged trees hold real files only)')
            elif not stat.S_ISREG(mode):
                problems.append(f'{rel}: not a regular file')
            else:
                files[rel] = _record(path)
    return files, problems


def _compare(expected: dict, actual: dict) -> list:
    problems = []
    for rel in sorted(set(expected) - set(actual)):
        problems.append(f'{rel}: missing')
    for rel in sorted(set(actual) - set(expected)):
        problems.append(f'{rel}: not part of the staged layout')
    for rel in sorted(set(expected) & set(actual)):
        e, a = expected[rel], actual[rel]
        if (e['bytes'], e['sha256']) != (a['bytes'], a['sha256']):
            problems.append(f'{rel}: {a["bytes"]} bytes sha256 {a["sha256"][:16]}..., '
                            f'expected {e["bytes"]} bytes sha256 {e["sha256"][:16]}...')
    return problems


def _set_modes(tree: Path, file_mode: int, dir_mode: int) -> None:
    """chmod every entry below ``tree`` (directories last when locking, first when unlocking)."""
    entries = []
    for directory, dirnames, filenames in os.walk(tree, followlinks=False):
        for name in filenames:
            entries.append((Path(directory) / name, file_mode))
        for name in dirnames:
            entries.append((Path(directory) / name, dir_mode))
    entries.append((tree, dir_mode))
    if dir_mode & stat.S_IWUSR:      # unlocking: parents first so children become reachable
        entries.sort(key=lambda item: len(item[0].parts))
    else:                            # locking: children first
        entries.sort(key=lambda item: -len(item[0].parts))
    for path, mode in entries:
        if not path.is_symlink():
            os.chmod(path, mode)


def _remove_tree(tree: Path) -> None:
    if not tree.exists() and not tree.is_symlink():
        return
    if tree.is_symlink() or not tree.is_dir():
        tree.unlink()
        return
    _set_modes(tree, 0o644, 0o755)
    shutil.rmtree(tree)


@contextmanager
def _stage_lock(code_root: Path):
    code_root.mkdir(parents=True, exist_ok=True)
    handle = open(code_root / '.stage.lock', 'a+')
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StageError(f'another stage_code process holds {code_root / ".stage.lock"}; '
                             'wait for it to finish') from exc
        except OSError as exc:  # e.g. ENOLCK on a filesystem without locks: warn, do not block
            print(f'[stage_code] WARNING: could not lock {code_root}: {exc}', file=sys.stderr)
        yield
    finally:
        handle.close()


def _non_empty(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return True


# --- desired stage --------------------------------------------------------------------------------
def _resolve_rust_library(paths, repo: Path, choice, provenance_path, existing):
    """Return (source path, record) of the library to stage."""
    committed = repo / pinned.RUST_LIBRARY
    if choice is None:
        if existing and existing.get('rust_library', {}).get('source') == SOURCE_REBUILT:
            # Keep the rebuilt library of the existing stage (read back from the uniform tree).
            kept = existing['rust_library']
            source = paths.code_dir('uniform') / pinned.RUST_LIBRARY_REL
            if not source.is_file():
                raise StageError(f'the existing stage uses a rebuilt librust_utils.so but {source} is missing; '
                                 'rerun hyak/env/build_rust_utils.sh install, or pass --rust-library committed --force')
            record = _record(source)
            if record['sha256'] != kept['sha256']:
                raise StageError(f'{source} no longer matches the rebuilt library recorded in the manifest; '
                                 'rerun hyak/env/build_rust_utils.sh install')
            return source, dict(kept)
        choice = SOURCE_COMMITTED
    if choice == SOURCE_COMMITTED:
        record = _record(committed)
        if record['sha256'] != pinned.RUST_LIBRARY_SHA256:
            raise StageError(f'{committed} does not match its pin; {pinned.remedy([pinned.RUST_LIBRARY + ":"])}')
        return committed, {'source': SOURCE_COMMITTED, **record, 'committed_sha256': pinned.RUST_LIBRARY_SHA256,
                           'origin': pinned.RUST_LIBRARY, 'provenance': None}
    source = Path(choice)
    if source.is_symlink() or not source.is_file():
        raise StageError(f'--rust-library {source} is not a regular file')
    record = _record(source)
    provenance = None
    if provenance_path:
        try:
            provenance = read_json(provenance_path)
        except (OSError, ValueError) as exc:
            raise StageError(f'cannot read --rust-provenance {provenance_path}: {exc}') from exc
        if isinstance(provenance, dict) and provenance.get('sha256') not in (None, record['sha256']):
            raise StageError(f'{provenance_path} describes sha256 {provenance.get("sha256")}, '
                             f'but {source} has {record["sha256"]}')
    kind = SOURCE_COMMITTED if record['sha256'] == pinned.RUST_LIBRARY_SHA256 else SOURCE_REBUILT
    return source, {'source': kind, **record, 'committed_sha256': pinned.RUST_LIBRARY_SHA256,
                    'origin': str(source.resolve()), 'provenance': provenance}


def _desired(repo: Path, rust_source: Path, rust_record: dict) -> dict:
    """``{run: {rel: {"bytes", "sha256", "source", "_path"}}}`` of the stage to build."""
    desired = {}
    for run in RUNS:
        files = {}
        for rel, source_rel in pinned.tree_layout(run).items():
            if rel == pinned.RUST_LIBRARY_REL:
                files[rel] = {'bytes': rust_record['bytes'], 'sha256': rust_record['sha256'],
                              'source': rust_record['origin'], '_path': rust_source}
            else:
                sha, size = pinned.STAGED_SOURCE_FILES[source_rel]
                files[rel] = {'bytes': size, 'sha256': sha, 'source': source_rel, '_path': repo / source_rel}
        desired[run] = files
    return desired


def _public(files: dict) -> dict:
    return {rel: {k: v for k, v in entry.items() if not k.startswith('_')} for rel, entry in sorted(files.items())}


def build_manifest(paths, repo: Path, desired: dict, rust_record: dict, chunked: dict) -> dict:
    """The manifest dict for a desired stage (``identity_sha256`` included)."""
    trees = {run: {'path': str(paths.code_dir(run)), 'files': _public(desired[run]),
                   'tree_sha256': tree_sha256(desired[run])} for run in RUNS}
    manifest = {
        'kind': MANIFEST_KIND,
        'schema': MANIFEST_SCHEMA,
        'created_at': utc_now(),
        'repository': repository_record(repo),
        'pinned_at_commit': pinned.PINNED_AT_COMMIT,
        'regret_pinned_at_commit': pinned.REGRET_PINNED_AT_COMMIT,
        'trees': trees,
        'rust_library': rust_record,
        'chunked_hypernet': chunked,
        'trainer_sha256': {'uniform': pinned.UNIFORM_TRAINER_SHA256, 'rl': pinned.RL_TRAINER_SHA256},
        'rl_trainer_variant': dict(pinned.RL_TRAINER_VARIANT),
        'settings_policy_sha256': pinned.SETTINGS_POLICY_SHA256,
        'paired_reward_sha256': pinned.PAIRED_REWARD_SHA256,
        'tool': {'stage_code_sha256': sha256_file(Path(__file__)),
                 'pinned_sha256': sha256_file(Path(pinned.__file__))},
    }
    manifest['identity_sha256'] = identity_sha256(manifest)
    return manifest


def load_manifest(paths) -> dict:
    """Read and structurally validate ``code/code_manifest.json``; raises StageError with a remedy."""
    path = paths.code_manifest
    try:
        manifest = read_json(path)
    except FileNotFoundError as exc:
        raise StageError(f'{path} not found. Remedy: python -m hyak.code.stage_code') from exc
    except (OSError, ValueError) as exc:
        raise StageError(f'cannot read {path}: {exc}') from exc
    if not isinstance(manifest, dict) or manifest.get('kind') != MANIFEST_KIND \
            or manifest.get('schema') != MANIFEST_SCHEMA:
        raise StageError(f'{path} is not a schema-{MANIFEST_SCHEMA} {MANIFEST_KIND}')
    try:
        recomputed = identity_sha256(manifest)
    except (KeyError, TypeError) as exc:
        raise StageError(f'{path} is incomplete ({exc!r})') from exc
    if recomputed != manifest.get('identity_sha256'):
        raise StageError(f'{path}: identity_sha256 does not match its content (edited by hand?); '
                         'restage with python -m hyak.code.stage_code --force before any run starts')
    return manifest


# --- staging ----------------------------------------------------------------------------------------
def _copy_tree(target_tmp: Path, files: dict) -> None:
    for rel, entry in sorted(files.items()):
        destination = target_tmp / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry['_path'], destination)   # real bytes; never a link
        copied = _record(destination)
        if (copied['bytes'], copied['sha256']) != (entry['bytes'], entry['sha256']):
            raise StageError(f'copy of {entry["_path"]} to {destination} is corrupt (hash mismatch); retry')


def _install_tree(paths, run: str, files: dict, log) -> None:
    code_root = paths.code_root
    target = paths.code_dir(run)
    tmp = code_root / f'.staging-{run}-{os.getpid()}'
    old = code_root / f'.old-{run}-{os.getpid()}'
    _remove_tree(tmp)
    _remove_tree(old)
    tmp.mkdir(parents=True)
    try:
        _copy_tree(tmp, files)
        actual, problems = scan_tree(tmp)
        problems += _compare({k: v for k, v in files.items()}, actual)
        if problems:
            raise StageError(f'staged copy of {run} is wrong: ' + '; '.join(problems))
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                _set_modes(target, 0o644, 0o755)
            os.replace(target, old)
        os.replace(tmp, target)
    except BaseException:
        _remove_tree(tmp)
        if old.exists() and not target.exists():
            os.replace(old, target)
            _set_modes(target, FILE_MODE, DIR_MODE)
        raise
    _set_modes(target, FILE_MODE, DIR_MODE)
    _remove_tree(old)
    log(f'staged {run}: {target} ({len(files)} files, tree_sha256 {tree_sha256(files)[:16]}...)')


def stage(paths, *, repo=None, rust_library=None, rust_provenance=None, force=False, log=print) -> dict:
    """Build or confirm the stage.  Returns ``{"changed", "trees_written", "manifest_written", "manifest"}``.

    ``rust_library``: None (keep the existing stage's library, else the committed one),
    ``'committed'``, or a path to a rebuilt ``librust_utils.so``.
    Raises StageError (with a remedy) on pin mismatches, refused overwrites and copy errors.
    """
    from hyak.common.config import repo_root
    repo = Path(repo) if repo is not None else repo_root()
    problems = pinned.check_files(repo, pinned.STAGED_SOURCE_FILES)
    if problems:
        raise StageError('pinned runtime or variant files differ from their pins: ' + '; '.join(problems)
                         + f'. Remedy: {pinned.remedy(problems)}')
    extra = pinned.unexpected_snapshot_files(repo)
    if extra:
        raise StageError(f'the runtime snapshots contain files that are not pinned: {extra}. Remedy: remove them '
                         f'(the snapshots must equal commit {pinned.REGRET_PINNED_AT_COMMIT[:12]})')
    with _stage_lock(paths.code_root):
        for leftover in sorted(paths.code_root.glob('.staging-*')) + sorted(paths.code_root.glob('.old-*')):
            log(f'removing {leftover} left by an interrupted staging run')
            _remove_tree(leftover)
        existing, unreadable = None, None
        if paths.code_manifest.exists() or paths.code_manifest.is_symlink():
            try:
                existing = load_manifest(paths)
            except StageError as exc:
                if not force:
                    raise StageError(f'{exc}. If no run has started, rerun with --force to restage') from exc
                unreadable = str(exc)
        rust_source, rust_record = _resolve_rust_library(paths, repo, rust_library, rust_provenance, existing)
        desired = _desired(repo, rust_source, rust_record)
        chunked = chunked_hypernet_record(repo)
        manifest = build_manifest(paths, repo, desired, rust_record, chunked)

        states, differences = {}, []
        for run in RUNS:
            target = paths.code_dir(run)
            if not target.exists() and not target.is_symlink():
                states[run] = 'missing'
                continue
            if target.is_symlink() or not target.is_dir():
                states[run] = 'different'
                differences.append(f'{target} is not a directory')
                continue
            actual, problems = scan_tree(target)
            problems += _compare(_public(desired[run]), actual)
            states[run] = 'different' if problems else 'same'
            differences += [f'{run}/{p}' for p in problems]
        identity_changed = existing is not None and identity(existing) != identity(manifest)
        if identity_changed:
            old, new = identity(existing), identity(manifest)
            if old['chunked_hypernet_sha256'] != new['chunked_hypernet_sha256']:
                differences.append(f'{CHUNKED_HYPERNET_REL}: sha256 {new["chunked_hypernet_sha256"]}, staged '
                                   f'with {old["chunked_hypernet_sha256"]}')
            if old['rust_library'] != new['rust_library']:
                differences.append(f'librust_utils.so: {new["rust_library"]}, staged with {old["rust_library"]}')
            if old['trees'] != new['trees'] and not differences:
                differences.append('staged file hashes differ from the manifest')
        if unreadable:
            differences.append(f'existing manifest unreadable: {unreadable}')
        needs_force = bool(unreadable) or identity_changed or any(state == 'different' for state in states.values())
        if needs_force:
            if not force:
                raise StageError('the existing stage differs from what would be staged: ' + '; '.join(differences)
                                 + '. A changed tree breaks resume pairing; if no run has started, rerun with --force')
            busy = [str(paths.run_dir(run)) for run in RUNS if _non_empty(paths.run_dir(run))]
            if busy:
                raise StageError('refusing --force: run directories already exist and are not empty: '
                                 + ', '.join(busy) + '. Restaging code under a started run breaks its resume '
                                 'pairing. Move those directories away first (or stage elsewhere).')
            log('WARNING: replacing the existing stage: ' + '; '.join(differences))

        trees_written = []
        for run in RUNS:
            if states[run] != 'same':
                _install_tree(paths, run, desired[run], log)
                trees_written.append(run)
        manifest_written = existing is None or identity_changed  # (an unreadable manifest has existing None)
        if manifest_written:
            atomic_write_json(paths.code_manifest, manifest)
            log(f'wrote {paths.code_manifest} (identity_sha256 {manifest["identity_sha256"]})')
        else:
            manifest = existing
            current = repository_record(repo).get('commit')
            recorded = (existing.get('repository') or {}).get('commit')
            if current != recorded:
                log(f'note: the repository is at {current}, the stage was recorded at {recorded}; '
                    'the staged files are identical, so the manifest is kept unchanged')
        if not trees_written and not manifest_written:
            log(f'stage is up to date (identity_sha256 {manifest["identity_sha256"]})')
        problems = verify_stage(paths, repo=repo)
        if problems:
            raise StageError('verification after staging failed: ' + '; '.join(problems))
        return {'changed': bool(trees_written or manifest_written), 'trees_written': trees_written,
                'manifest_written': manifest_written, 'manifest': manifest}


# --- verification --------------------------------------------------------------------------------
def verify_stage(paths, *, runs=RUNS, repo=None, check_chunked=True) -> list:
    """Problems of an existing stage (empty list = verified).

    Checks the manifest's integrity, every staged file's bytes and SHA-256, the exact file
    set (no extra files, no symlinks), each file against its pin (the library against the
    committed pin or the recorded rebuild), the pairing of the two trees, and (``check_chunked``)
    that ``hyak/memory/chunked_hypernet.py`` still has the hash recorded at staging time.
    """
    try:
        manifest = load_manifest(paths)
    except StageError as exc:
        return [str(exc)]
    problems = []
    rust = manifest['rust_library']
    for run in runs:
        if run not in manifest['trees']:
            problems.append(f'manifest has no {run} tree')
            continue
        entry = manifest['trees'][run]
        files = entry['files']
        if set(files) != set(pinned.tree_layout(run)):
            problems.append(f'{run}: manifest file list differs from the pinned layout')
        if tree_sha256(files) != entry.get('tree_sha256'):
            problems.append(f'{run}: tree_sha256 does not match the manifest file records')
        for rel, record in sorted(files.items()):
            source = pinned.tree_layout(run).get(rel)
            if rel == pinned.RUST_LIBRARY_REL:
                if (record['sha256'], record['bytes']) != (rust['sha256'], rust['bytes']):
                    problems.append(f'{run}/{rel}: manifest record differs from rust_library')
                if rust['source'] == SOURCE_COMMITTED and rust['sha256'] != pinned.RUST_LIBRARY_SHA256:
                    problems.append(f'{run}/{rel}: recorded as committed but not the pinned library')
            elif source in pinned.STAGED_SOURCE_FILES and \
                    (record['sha256'], record['bytes']) != pinned.STAGED_SOURCE_FILES[source]:
                problems.append(f'{run}/{rel}: recorded hash is not the pinned hash of {source}')
        tree = paths.code_dir(run)
        if tree.is_symlink() or not tree.is_dir():
            problems.append(f'{tree} is missing or not a directory. Remedy: python -m hyak.code.stage_code')
            continue
        actual, scan_problems = scan_tree(tree)
        problems += [f'{run}/{p}' for p in scan_problems + _compare(files, actual)]
    if set(runs) == set(RUNS):
        problems += check_pairing(manifest)
    if check_chunked:
        repo = Path(repo) if repo is not None else _default_repo()
        now = chunked_hypernet_record(repo)['sha256']
        recorded = manifest['chunked_hypernet']['sha256']
        if now != recorded:
            problems.append(f'{CHUNKED_HYPERNET_REL} in {repo} has sha256 {now}, the stage recorded {recorded}; '
                            'the memory-saving forward changed since staging (restage with --force only '
                            'if no run has started)')
    return problems


def _default_repo() -> Path:
    from hyak.common.config import repo_root
    return repo_root()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.code.stage_code', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true', help='verify the existing stage; change nothing')
    mode.add_argument('--show', action='store_true', help='print the manifest')
    parser.add_argument('--run', choices=RUNS, help='with --verify: only this tree')
    parser.add_argument('--force', action='store_true',
                        help='replace a differing stage (refused while a run directory is non-empty)')
    parser.add_argument('--rust-library', metavar='committed|PATH',
                        help='library to stage (default: keep the existing stage\'s, else the committed one)')
    parser.add_argument('--rust-provenance', type=Path, metavar='JSON',
                        help='build provenance of --rust-library (written by build_rust_utils.sh)')
    args = parser.parse_args(argv)
    from hyak.common.config import ConfigError, load_config
    from hyak.common.paths import Paths
    try:
        paths = Paths.from_config(load_config(args.config))
    except (ConfigError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    try:
        if args.show:
            print(json.dumps(load_manifest(paths), indent=2, sort_keys=True))
            return 0
        if args.verify:
            runs = (args.run,) if args.run else RUNS
            problems = verify_stage(paths, runs=runs)
            if problems:
                for line in problems:
                    print(f'PROBLEM {line}', file=sys.stderr)
                print(f'ERROR: staged code in {paths.code_root} failed verification ({len(problems)} problem(s)). '
                      'Remedy: python -m hyak.code.stage_code (add --force only if no run has started)',
                      file=sys.stderr)
                return 1
            manifest = load_manifest(paths)
            print(f'OK staged code verified: {", ".join(runs)} (identity_sha256 {manifest["identity_sha256"]})')
            return 0
        if args.rust_provenance and not args.rust_library:
            parser.error('--rust-provenance needs --rust-library')
        result = stage(paths, rust_library=args.rust_library, rust_provenance=args.rust_provenance,
                       force=args.force)
        manifest = result['manifest']
        print(json.dumps({'stage': 'code_staged', 'changed': result['changed'],
                          'trees_written': result['trees_written'], 'manifest': str(paths.code_manifest),
                          'identity_sha256': manifest['identity_sha256'],
                          'tree_sha256': {run: manifest['trees'][run]['tree_sha256'] for run in RUNS},
                          'rust_library': manifest['rust_library']['source'],
                          'chunked_hypernet_sha256': manifest['chunked_hypernet']['sha256']}, sort_keys=True))
        return 0
    except StageError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
