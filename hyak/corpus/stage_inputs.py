"""Stage the corpus into the trainer's input layout under ``$ANYTOK_ROOT/input/``.

Replaces the missing ``zett/.joint_corpus`` stager and the hand copy of the held-out files
(``scripts/modal_anytok_pool_20k.py:62-77`` staged ``recipe.json``, ``corpus/<bucket>/train.jsonl.gz``,
``corpus/manifest.json`` and ``eval/<bucket>.jsonl.gz``).  Writes::

    input/recipe.json                     byte copy of data/balanced-full-v3.json (SHA-256 b3bc1933...)
    input/corpus/<bucket>/train.jsonl.gz  relative symlink to corpus/<size>-v1/<bucket>/train.jsonl.gz,
                                          the pool chosen by the recipe's source_path
    input/corpus/manifest.json            exactly the 23 bucket keys; each {"compressed_bytes", "sha256",
                                          + provenance fields the trainer ignores}
    input/eval/<bucket>.jsonl.gz          verified copy of the pool's eval.jsonl.gz (exactly the 23 names)
    input/pool-recipe.json                the recipe with source_path / eval_path pointing at the pools
                                          on this cluster (for scripts/tokenizer_pool.py --recipe)
    input/manifest.json                   bytes + sha256 of every staged input (merged with other stages),
                                          including input/model/{config.json,model.safetensors,tokenizer.json}
                                          verified against the pin (hyak.model.fetch_model.expected_files)

Why linking the full pool file is equivalent to the original quota-sized joint files: the
trainer's ``BalancedTextStream`` (runtime/anytok_pool_20k/hf_hypernet/joint_training.py:76-98)
reads each bucket from its first line and never past ``train_target_bytes`` before rewinding, so
any file whose first ``train_target_bytes`` raw bytes are the same documents in the same order
gives the same text stream.  A file with *less* raw text than the quota would silently rewind
early, so staging refuses any bucket whose ``counts.train_bytes`` is below the recipe's
``train_target_bytes`` (``publish_recipes`` made the same check, scripts/modal_prepare_corpus.py:519-520).
The original staged files held exactly the quota (their rebuilt ``corpus/manifest.json`` hashes to the
pinned value, ``raw_bytes == quota_bytes`` in all 23 buckets; ``hyak.corpus.verify``), so the original
trainer also rewound at exactly the quota: the links give the same text stream, wrap-around included.

The corpus stop: staging refuses (exit 3) unless ``<corpus root>/scorecard.json`` (``python -m
hyak.corpus.verify``) is exact for exactly the bucket manifests being staged, or ``--accept-corpus-scorecard
"<reason>"`` is given (recorded under ``provenance.corpus.scorecard`` of ``input/manifest.json``).  A
test-scale stage (``--allow-test-scale``) skips this stop and the model registration.

The original staged trainer files (``--original-staged``): when ``input/corpus/<bucket>/train.jsonl.gz``
(23 regular files) and ``input/corpus/manifest.json`` are the original files, i.e. their sizes and SHA-256
equal ``plans/zett-modal-20k-inputs.json``, they are kept as they are and registered instead of links;
brought held-out files ``input/eval/<bucket>.jsonl.gz`` must equal their pins too (missing ones are copied
from the pool).  The corpus root is still needed (pool recipe, held-out copies, the corpus stop).

Durability: the links point into the corpus root, which must stay unchanged until both runs and their
evaluation are finished.  A corpus root under ``$ANYTOK_RUNS`` is refused; any root other than
``$ANYTOK_ROOT/corpus`` gets a loud warning.  ``provenance.corpus.gzip_mtime`` records the gzip header
time of both files of every bucket, so ``hyak.corpus.build --gzip-mtimes input/manifest.json`` (the
default for the production corpus root) regenerates a lost bucket byte for byte.

Safety: an existing ``input/recipe.json``, ``input/corpus/manifest.json``, staged train link or
held-out file is never changed while a run directory (``$ANYTOK_RUNS/uniform`` or ``/rl``) is
non-empty, because both trainers hash the recipe and the corpus manifest into ``run_config.json``
and refuse to resume after a change (resume pairing).  ``--force`` overrides this.  Test-scale
buckets (``hyak.corpus.build --test-scale``) are refused unless ``--allow-test-scale`` is given;
then the size assertion is relaxed and ``input/manifest.json`` records ``test_scale: true``.

Usage::

    python -m hyak.corpus.stage_inputs [--force] [--corpus-root DIR] [--input-dir DIR]
    python -m hyak.corpus.stage_inputs --check          # verify what is staged; no writes
    python -m hyak.corpus.stage_inputs --corpus-root <mini> --input-dir <dir> --allow-test-scale   # tests

Standard library only (any Python >= 3.9).  Exit 0 on success; 2 usage, configuration or environment
error (e.g. input/model not fetched); 3 the corpus stop (no exact scorecard; the user decides); 4 a
refused or inconsistent staging, or ``--check`` found a problem (the message says why and what to do);
1 only for an unexpected crash (traceback on stderr).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from hyak.common import slurm
from hyak.common.config import ConfigError, load_config, repo_root
from hyak.common.io import atomic_copy, atomic_write_json, atomic_write_text, file_record, read_json, sha256_file, utc_now
from hyak.common.paths import RUNS, STAGE_PROGRESS, STAGE_STAGE_INPUTS, Paths
from hyak.corpus import sources

EXIT_OK, EXIT_USAGE, EXIT_STOP, EXIT_CHECK = 0, 2, 3, 4
#: Sizes and SHA-256 of the files the original trainer read (corpus/<bucket>/train.jsonl.gz, corpus/manifest.json,
#: eval/<bucket>.jsonl.gz, model/*); its own SHA-256 is pinned in hyak.code.pinned.
INPUTS_PLAN = 'plans/zett-modal-20k-inputs.json'
#: Format of hyak.corpus.verify's scorecard.json that the corpus stop reads.
SCORECARD_FORMAT = 2

RECIPE_SOURCE = 'data/balanced-full-v3.json'
RECIPE_SHA256 = 'b3bc193333aa13d3a513f1c8978805835f837190cd0e5991742c55e9a0752cfe'
RECIPE_NAME = 'balanced-full-v3'
#: Bucket names the trainer's build_eval_set accepts (train_joint_smollm_input.py:92-97).
TRAINER_EVAL_BUCKETS = frozenset({
    "web-en", "web-es", "web-fr", "web-ru", "web-zh", "web-ja",
    "web-ar", "web-hi", "wiki-en", "wiki-es", "wiki-fr", "wiki-ru",
    "wiki-zh", "wiki-ja", "wiki-ar", "wiki-hi", "code", "papers-en",
    "math-en", "qa-en", "books-en", "medical-en", "legal-en",
})
#: ``kind`` of input/manifest.json (the original was ``zett_20k_inputs``).
INPUT_MANIFEST_KIND = 'anytok_hyak_inputs'
#: Fields of a corpus manifest entry besides the two the trainer reads (ignored by it).
CORPUS_ENTRY_FIELDS = ('compressed_bytes', 'sha256', 'source', 'train_bytes', 'train_docs', 'dataset', 'revision')
_LOCK_WAIT_SECONDS = 300.0


class StagingError(RuntimeError):
    """Staging refused or failed; the message says what to do."""


class MiniatureBucketError(StagingError):
    """A bucket built with ``hyak.corpus.build --test-scale`` (refused unless ``--allow-test-scale``)."""


class ScorecardStop(StagingError):
    """No exact corpus scorecard for the corpus being staged: a protocol stop that needs the user (exit 3)."""


class EnvironmentProblem(StagingError):
    """A prerequisite outside the corpus is missing or wrong (e.g. input/model not fetched; exit 2)."""


# --- layout (names derived from hyak.common.paths, optionally re-rooted for tests) ------------------
class InputLayout:
    """The ``input/`` files of :class:`hyak.common.paths.Paths`, optionally under another directory."""

    def __init__(self, paths: Paths, input_dir=None):
        self.paths = paths
        self.input = Path(os.path.normpath(str(input_dir))) if input_dir else paths.input

    def _map(self, path: Path) -> Path:
        return self.input / path.relative_to(self.paths.input)

    @property
    def recipe(self):
        return self._map(self.paths.recipe)

    @property
    def pool_recipe(self):
        return self._map(self.paths.pool_recipe)

    @property
    def corpus(self):
        return self._map(self.paths.input_corpus)

    @property
    def corpus_manifest(self):
        return self._map(self.paths.input_corpus_manifest)

    def corpus_train(self, bucket):
        return self._map(self.paths.input_corpus_train(bucket))

    @property
    def eval(self):
        return self._map(self.paths.input_eval)

    def eval_file(self, bucket):
        return self._map(self.paths.input_eval_file(bucket))

    @property
    def manifest(self):
        return self._map(self.paths.input_manifest)

    def relative(self, path: Path) -> str:
        return Path(path).relative_to(self.input).as_posix()


# --- the recipe --------------------------------------------------------------------------------------
def load_recipe(repo=None) -> tuple[bytes, dict]:
    """Bytes and parsed content of ``data/balanced-full-v3.json``, checked against everything the port relies on."""
    path = Path(repo or repo_root()) / RECIPE_SOURCE
    data = path.read_bytes()
    from hyak.common.io import sha256_bytes
    if sha256_bytes(data) != RECIPE_SHA256:
        raise StagingError(f'{path} has SHA-256 {sha256_bytes(data)}, expected {RECIPE_SHA256}: the pinned recipe '
                           'changed. Remedy: restore it with git checkout -- data/balanced-full-v3.json')
    recipe = json.loads(data)
    check_recipe(recipe)
    return data, recipe


def check_recipe(recipe: dict) -> None:
    """The recipe's names, order and targets must be what the port's plans produce."""
    if recipe.get('name') != RECIPE_NAME:
        raise StagingError(f'recipe name {recipe.get("name")!r} is not {RECIPE_NAME!r}')
    targets = sources.specialty_recipe_targets(sources.FULL_BYTES)
    names = [bucket['name'] for bucket in recipe['buckets']]
    if names != list(targets):
        raise StagingError(f'recipe bucket order {names} differs from specialty_recipe_targets')
    if [bucket['train_target_bytes'] for bucket in recipe['buckets']] != list(targets.values()):
        raise StagingError('recipe train_target_bytes differ from specialty_recipe_targets(10^10)')
    for bucket in recipe['buckets']:
        expected = f'{sources.ROOT}/{sources.recipe_pool(bucket["name"])}-v1/{bucket["name"]}'
        if bucket['source_path'] != expected or bucket['eval_path'] != f'{expected}/eval.jsonl.gz':
            raise StagingError(f'recipe paths of {bucket["name"]} are not under {expected}')
        if bucket['revision'] != sources.SOURCE_REVISIONS.get(bucket['dataset']):
            raise StagingError(f'recipe revision of {bucket["name"]} differs from SOURCE_REVISIONS')
    if set(names) != TRAINER_EVAL_BUCKETS:
        raise StagingError('recipe buckets differ from the trainer held-out bucket set')


def pool_bucket_dir(corpus_root: Path, bucket: dict) -> Path:
    """Pool directory of a recipe bucket: ``/corpus/<size>-v1/<bucket>`` -> ``<corpus_root>/<size>-v1/<bucket>``."""
    source = Path(bucket['source_path'])
    try:
        relative = source.relative_to(sources.ROOT)
    except ValueError as exc:
        raise StagingError(f'source_path {source} of {bucket["name"]} is not under {sources.ROOT}') from exc
    return Path(corpus_root) / relative


def pool_recipe(recipe: dict, corpus_root: Path) -> dict:
    """The recipe with ``source_path`` / ``eval_path`` rewritten to this cluster's pools (cf.
    scripts/modal_generate_fast_tokenizers.py:57-58, which rewrote source_path the same way)."""
    derived = json.loads(json.dumps(recipe))
    for bucket in derived['buckets']:
        directory = pool_bucket_dir(corpus_root, bucket)
        bucket['source_path'] = str(directory)
        bucket['eval_path'] = str(directory / 'eval.jsonl.gz')
    return derived


def recipe_text(recipe: dict) -> str:
    """Serialisation of the recipe files (``publish_recipes``: ``json.dumps(recipe, indent=2) + "\\n"``)."""
    return json.dumps(recipe, indent=2) + '\n'


# --- bucket checks -------------------------------------------------------------------------------------
def check_bucket(corpus_root: Path, bucket: dict, *, allow_test_scale=False) -> dict:
    """Check one pool bucket for staging; returns facts used by :func:`stage`.  Raises StagingError."""
    name = bucket['name']
    size, plan_spec = sources.bucket_specs()[name]
    directory = pool_bucket_dir(corpus_root, bucket)
    if directory.parent.name != f'{size}-v1':
        raise StagingError(f'{name}: recipe pool {directory.parent.name} differs from the plan pool {size}-v1')
    manifest_path = directory / 'manifest.json'
    try:
        manifest = read_json(manifest_path)
    except FileNotFoundError:
        raise StagingError(f'{name}: {manifest_path} does not exist; build the bucket first '
                           f'(python -m hyak.corpus.build --buckets {name})') from None
    except (OSError, ValueError) as exc:
        raise StagingError(f'{name}: unreadable {manifest_path}: {exc}') from exc
    marker = manifest.get('test_scale')
    spec = manifest.get('spec') or {}
    if marker is not None:
        if not allow_test_scale:
            raise MiniatureBucketError(f'{name}: {directory} is a TEST-SCALE bucket ({marker}); it must never be used for '
                                  f'training. Remedy: delete it and run the production build')
        expected = dict(plan_spec, train_target=marker.get('train_target'), eval_target=marker.get('eval_target'))
    else:
        expected = plan_spec
    if spec != expected:
        raise StagingError(f'{name}: manifest spec {spec} differs from the plan {expected}')
    if manifest.get('dataset') != bucket['dataset'] or manifest.get('revision') != bucket['revision']:
        raise StagingError(f'{name}: manifest dataset/revision {manifest.get("dataset")}@{manifest.get("revision")} '
                           f'differ from the recipe {bucket["dataset"]}@{bucket["revision"]}')
    counts = manifest.get('counts') or {}
    shortfall = bucket['train_target_bytes'] - int(counts.get('train_bytes', 0))
    if shortfall > 0 and marker is None:
        raise StagingError(f'{name}: only {counts.get("train_bytes")} raw training bytes, below the recipe quota '
                           f'{bucket["train_target_bytes"]}; the trainer would silently rewind early. '
                           f'Remedy: rebuild the bucket to the pool target')
    files = {}
    for split in ('train', 'eval'):
        path = directory / f'{split}.jsonl.gz'
        if not path.is_file():
            raise StagingError(f'{name}: {path} is missing')
        record = file_record(path)
        if record['sha256'] != (manifest.get('files') or {}).get(f'{split}.jsonl.gz'):
            raise StagingError(f'{name}: {path} has SHA-256 {record["sha256"]}, the bucket manifest says '
                               f'{(manifest.get("files") or {}).get(f"{split}.jsonl.gz")}. Remedy: delete {directory} and rebuild')
        files[split] = dict(record, path=path)
    return {'name': name, 'size': size, 'directory': directory, 'manifest': manifest, 'counts': counts,
            'test_scale': marker is not None, 'shortfall_bytes': max(0, shortfall), 'files': files,
            'manifest_sha256': sha256_file(manifest_path),
            'gzip_mtime': {f'{split}.jsonl.gz': gzip_mtime(files[split]['path']) for split in ('train', 'eval')}}


def corpus_manifest(facts: list, root: Path) -> dict:
    """``input/corpus/manifest.json`` content: exactly the 23 bucket keys (trainer verify_corpus contract)."""
    manifest = {}
    for fact in facts:
        train = fact['files']['train']
        try:
            source = Path(train['path']).relative_to(root).as_posix()
        except ValueError:
            source = str(train['path'])
        manifest[fact['name']] = {
            'compressed_bytes': train['bytes'], 'sha256': train['sha256'], 'source': source,
            'train_bytes': fact['counts']['train_bytes'], 'train_docs': fact['counts']['train_docs'],
            'dataset': fact['manifest']['dataset'], 'revision': fact['manifest']['revision'],
        }
    if len(manifest) != 23:
        raise StagingError(f'corpus manifest would have {len(manifest)} entries, the trainer needs 23')
    return manifest


def manifest_text(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + '\n'


# --- trainer contract ------------------------------------------------------------------------------------
def verify_corpus_contract(data_dir, manifest_path) -> None:
    """The pinned trainer's ``verify_corpus`` (runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py:242-255,
    identical at runtime/anytok_pool_settings_rl_20k/...:245-258 and in the port's RL trainer,
    hyak/variants/anytok_pool_straight_rl_20k/...:235-248), copied verbatim (the trainer cannot be
    imported without its CUDA/Rust dependencies).  Raises ValueError exactly where the trainer would."""
    import hashlib
    manifest = json.loads(Path(manifest_path).read_text())
    if len(manifest) != 23:
        raise ValueError("Expected checksums for all 23 corpus buckets")
    for name, item in manifest.items():
        path = Path(data_dir) / name / "train.jsonl.gz"
        if not path.is_file() or path.stat().st_size != item["compressed_bytes"]:
            raise ValueError(f"Missing or incomplete staged training file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Training file checksum mismatch: {path}")


def check_contract(layout: InputLayout, recipe: dict) -> list:
    """Problems with the staged files from the trainer's point of view ([] when consumable)."""
    problems = []
    try:
        verify_corpus_contract(layout.corpus, layout.corpus_manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f'trainer verify_corpus would fail: {exc}')
    try:
        keys = set(json.loads(layout.corpus_manifest.read_text()))
    except (OSError, ValueError):
        keys = set()
    names = [bucket['name'] for bucket in recipe['buckets']]
    if keys != set(names):
        problems.append(f'corpus manifest keys differ from the recipe buckets: extra {sorted(keys - set(names))}, '
                        f'missing {sorted(set(names) - keys)}')
    missing = [str(layout.corpus_train(n)) for n in names if not layout.corpus_train(n).is_file()]
    if missing:
        problems.append(f'BalancedTextStream would raise FileNotFoundError for {missing}')
    found = sorted(p.name for p in layout.eval.glob('*.jsonl.gz')
                   if p.stem.removesuffix('.jsonl') in TRAINER_EVAL_BUCKETS) if layout.eval.is_dir() else []
    if len(found) != 23:
        problems.append(f'build_eval_set needs 23 held-out files in {layout.eval}, found {len(found)}')
    return problems


# --- input/manifest.json (shared by several stages) --------------------------------------------------------
def _locked(lock_dir: Path):
    """Context manager: exclusive ``mkdir`` lock with stale-owner takeover (other stages update the same file)."""
    class _Lock:
        def __enter__(self):
            deadline = time.time() + _LOCK_WAIT_SECONDS
            lock_dir.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try:
                    os.mkdir(lock_dir)
                    atomic_write_json(lock_dir / 'owner.json', {'job_id': slurm.worker_id(), 'host': slurm.node_name(),
                                                                'pid': os.getpid(), 'at': utc_now()})
                    return self
                except FileExistsError:
                    try:
                        age = time.time() - lock_dir.stat().st_mtime
                    except OSError:
                        continue
                    if age > 600:  # a manifest update takes seconds; an older lock is stale
                        for child in lock_dir.glob('*'):
                            try:
                                child.unlink()
                            except OSError:
                                pass
                        try:
                            os.rmdir(lock_dir)
                        except OSError:
                            pass
                        continue
                    if time.time() > deadline:
                        raise StagingError(f'{lock_dir} is held by another process; remove it if no job is running')
                    time.sleep(1.0)

        def __exit__(self, *exc):
            try:
                (lock_dir / 'owner.json').unlink()
            except OSError:
                pass
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass
            return False
    return _Lock()


def update_input_manifest(manifest_path: Path, files: dict, *, provenance_key=None, provenance=None,
                          remove=()) -> dict:
    """Merge ``files`` (``{relative path: {"bytes", "sha256", ...}}``) into ``input/manifest.json``.

    Format: ``{"kind": "anytok_hyak_inputs", "files": {...}, "provenance": {<stage>: {...}}}``,
    written atomically with indent=2, sorted keys, under a lock directory next to it, so several
    stages (corpus staging, model fetch, init, pool collect) can each update their own entries.
    ``remove`` lists relative paths to drop.  Returns the written object.
    """
    manifest_path = Path(manifest_path)
    with _locked(manifest_path.with_name('.manifest.lock')):
        current = read_json(manifest_path, None) or {'kind': INPUT_MANIFEST_KIND, 'files': {}}
        if current.get('kind') != INPUT_MANIFEST_KIND or not isinstance(current.get('files'), dict):
            raise StagingError(f'{manifest_path} is not an {INPUT_MANIFEST_KIND} manifest; move it away and rerun')
        for key in remove:
            current['files'].pop(key, None)
        current['files'].update(files)
        if provenance_key is not None:
            current.setdefault('provenance', {})[provenance_key] = provenance
        atomic_write_json(manifest_path, current)
        return current


# --- staging -------------------------------------------------------------------------------------------------
def nonempty_run_dirs(paths: Paths) -> list:
    """Run directories that already contain anything (their run_config.json pins the recipe and manifest)."""
    found = []
    for run in RUNS:
        directory = paths.run_dir(run)
        try:
            if directory.is_dir() and any(directory.iterdir()):
                found.append(str(directory))
        except OSError:
            found.append(str(directory))
    return found


def trained_pool_tokenizers(paths: Paths) -> int:
    """Number of pool tokenizers already trained (their text came from the current pool files)."""
    try:
        return sum(1 for _ in paths.pool_tokenizers.glob('*/manifest.json'))
    except OSError:
        return 0


def _read(path: Path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def load_inputs_plan(repo=None) -> dict:
    """``files`` of plans/zett-modal-20k-inputs.json, checked against its pin (hyak.code.pinned)."""
    from hyak.code import pinned
    repo = Path(repo or repo_root())
    path = repo / INPUTS_PLAN
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EnvironmentProblem(f'cannot read {path}: {exc}') from exc
    from hyak.common.io import sha256_bytes
    expected = pinned.REFERENCE_FILES[INPUTS_PLAN][0]
    if sha256_bytes(data) != expected:
        raise EnvironmentProblem(f'{path} has SHA-256 {sha256_bytes(data)}, expected {expected}. Remedy: '
                                 f'git checkout -- {INPUTS_PLAN}')
    return json.loads(data)['files']


def gzip_mtime(path) -> int:
    """Header time of a gzip file (bytes 4-7, little-endian)."""
    with open(path, 'rb') as handle:
        return int.from_bytes(handle.read(8)[4:8], 'little')


def check_corpus_root(paths: Paths, corpus_root: Path, log=print) -> None:
    """Refuse a corpus root under $ANYTOK_RUNS; warn loudly about any root other than $ANYTOK_ROOT/corpus."""
    corpus_root = Path(os.path.normpath(str(corpus_root)))
    runs = Path(os.path.normpath(str(paths.runs)))
    if corpus_root == runs or runs in corpus_root.parents:
        raise StagingError(f'corpus root {corpus_root} is inside $ANYTOK_RUNS ({runs}); the staged links must point at '
                           f'storage that outlives both runs. Remedy: keep the corpus in {paths.corpus_root}')
    if corpus_root != Path(os.path.normpath(str(paths.corpus_root))):
        log(f'[hyak stage-inputs] WARNING: the staged links point outside $ANYTOK_ROOT/corpus, into {corpus_root}; '
            'that directory must stay unchanged (and must not be on purged scratch storage) until both runs and their '
            'evaluation are finished. A lost bucket can be rebuilt byte for byte with hyak.corpus.build '
            '--gzip-mtimes input/manifest.json')


def check_scorecard(corpus_root: Path, facts: list, *, accept_reason=None, log=print) -> dict:
    """The corpus stop: an exact ``scorecard.json`` of exactly these buckets, or an accepted reason.

    Returns the provenance record ``{"verdict", "exact", "at_utc", "sha256", "accepted_reason"}``; raises
    :class:`ScorecardStop` when the scorecard is missing, stale, of another corpus or not exact and no
    reason was given.
    """
    path = Path(corpus_root) / 'scorecard.json'
    problem, report = None, None
    try:
        report = read_json(path)
    except FileNotFoundError:
        problem = f'{path} does not exist'
    except (OSError, ValueError) as exc:
        problem = f'{path} is unreadable ({exc})'
    if report is not None:
        current = {fact['name']: fact['manifest_sha256'] for fact in facts}
        recorded = report.get('bucket_manifests') or {}
        if not isinstance(report, dict) or report.get('format') != SCORECARD_FORMAT:
            problem = f'{path} has another format (written by an older hyak.corpus.verify)'
        elif os.path.normpath(str(report.get('corpus_root', ''))) != os.path.normpath(str(corpus_root)):
            problem = f'{path} was written for {report.get("corpus_root")}, not {corpus_root}'
        elif any(recorded.get(name) != digest for name, digest in current.items()):
            changed = sorted(name for name, digest in current.items() if recorded.get(name) != digest)
            problem = f'{path} is older than the bucket manifests of {", ".join(changed)}'
        elif report.get('exact') is not True:
            problem = f'the corpus scorecard is not exact: {report.get("verdict")}'
    record = {'path': str(path), 'verdict': (report or {}).get('verdict') if isinstance(report, dict) else None,
              'exact': problem is None, 'at_utc': (report or {}).get('at_utc') if isinstance(report, dict) else None,
              'sha256': sha256_file(path) if path.is_file() else None, 'accepted_reason': None}
    if problem is None:
        return record
    if not accept_reason:
        raise ScorecardStop(f'{problem}. This is the corpus stop: do not stage the inputs or run the pool on a corpus '
                            'that is not the agreed one. Remedy: run python -m hyak.corpus.verify (the corpus job does) '
                            'and follow the reading rules in scorecard.txt and hyak/README.md section 10; continuing '
                            'with this corpus needs the user\'s approval, then pass --accept-corpus-scorecard "<reason>"')
    log(f'[hyak stage-inputs] WARNING: {problem}; continuing because --accept-corpus-scorecard was given: {accept_reason}')
    record['accepted_reason'] = accept_reason
    return record


def check_model(paths: Paths, layout, expected=None) -> dict:
    """``{"model/<name>": {"bytes", "sha256"}}`` of the three pinned model files, verified by SHA-256."""
    from hyak.model import fetch_model
    try:
        expected = fetch_model.expected_files() if expected is None else expected
    except fetch_model.FetchError as exc:
        raise EnvironmentProblem(str(exc)) from exc
    model_dir = layout.input / paths.model_dir.relative_to(paths.input)
    problems = fetch_model.verify_model_dir(model_dir, expected, check_hash=True)
    if problems:
        raise EnvironmentProblem('the model files do not match the pin: ' + '; '.join(problems)
                                 + '. Remedy: python -m hyak.model.fetch_model (or --from-dir <copy>), then rerun')
    return {layout.relative(model_dir / name): {'bytes': entry['bytes'], 'sha256': entry['sha256']}
            for name, entry in sorted(expected.items())}


def check_original_staged(layout, recipe: dict, plan_files: dict) -> dict:
    """Verify brought original trainer files against the pins; returns ``{relative path: record}``."""
    problems, records = [], {}
    wanted = [f'corpus/{bucket["name"]}/train.jsonl.gz' for bucket in recipe['buckets']] + ['corpus/manifest.json']
    for relative in wanted:
        path = layout.input / relative
        pin = plan_files.get(relative)
        if pin is None:
            problems.append(f'{INPUTS_PLAN} has no entry {relative}')
            continue
        if path.is_symlink() or not path.is_file():
            problems.append(f'{path} is not a regular file (bring the original file there)')
            continue
        record = file_record(path)
        if (record['bytes'], record['sha256']) != (int(pin['bytes']), pin['sha256']):
            problems.append(f'{path}: {record["bytes"]} bytes, SHA-256 {record["sha256"]}; the original has '
                            f'{pin["bytes"]} bytes, SHA-256 {pin["sha256"]}')
            continue
        records[relative] = {'bytes': record['bytes'], 'sha256': record['sha256'], 'origin': 'original'}
    if problems:
        raise StagingError('--original-staged: the brought files are not the original trainer inputs:\n  '
                           + '\n  '.join(problems))
    return records


def _links_to(link: Path, target: Path) -> bool:
    """``link`` is a symlink that resolves to the file ``target``."""
    try:
        return link.is_symlink() and os.path.samefile(link, target)
    except OSError:
        return False


def _make_link(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.link')
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    os.symlink(target, temporary)
    os.replace(temporary, path)


#: provenance.corpus keys that change on every call; kept from the previous stage when nothing else changed.
_VOLATILE_PROVENANCE = ('staged_at', 'job_id', 'host')


def stage(paths: Paths, *, corpus_root=None, input_dir=None, force=False, allow_test_scale=False,
          repo=None, log=print, writer=None, accept_corpus_scorecard=None, original_staged=False,
          model_expected=None) -> dict:
    """Stage everything listed in the module docstring.  Returns a report dict; raises StagingError.

    ``accept_corpus_scorecard``: reason to stage although the corpus scorecard is not exact (recorded).
    ``original_staged``: keep and register the brought original trainer files (module docstring).
    ``model_expected``: ``{name: {"bytes", "sha256"}}`` instead of the pin (tests only).
    """
    layout = InputLayout(paths, input_dir)
    corpus_root = Path(os.path.normpath(str(corpus_root))) if corpus_root else paths.corpus_root
    check_corpus_root(paths, corpus_root, log)
    recipe_bytes, recipe = load_recipe(repo)
    facts, problems, test_buckets = [], [], []
    for bucket in recipe['buckets']:
        try:
            facts.append(check_bucket(corpus_root, bucket, allow_test_scale=allow_test_scale))
        except MiniatureBucketError:
            test_buckets.append(bucket['name'])
        except StagingError as exc:
            problems.append(str(exc))
    if test_buckets:
        problems.insert(0, f'{len(test_buckets)} bucket(s) under {corpus_root} are TEST-SCALE miniatures '
                           f'({", ".join(test_buckets)}); they must never be used for training. Remedy: build the '
                           f'production corpus (python -m hyak.corpus.build); --allow-test-scale is for tests only')
    if problems:
        raise StagingError('cannot stage the corpus:\n  ' + '\n  '.join(problems))
    test_scale = any(fact['test_scale'] for fact in facts)
    if test_scale and not all(fact['test_scale'] for fact in facts):
        raise StagingError('mixed test-scale and production buckets; stage either a complete production corpus or '
                           'a complete test-scale corpus')
    if test_scale and original_staged:
        raise StagingError('--original-staged cannot be combined with a test-scale corpus')
    # --- the corpus stop and the model (production only) ------------------------------------------------------
    if test_scale:
        scorecard = {'skipped': 'test-scale corpus (--allow-test-scale)'}
        model_files = {}
    else:
        scorecard = check_scorecard(corpus_root, facts, accept_reason=accept_corpus_scorecard, log=log)
        model_files = check_model(paths, layout, model_expected)
    plan_files = load_inputs_plan(repo) if original_staged else {}
    originals = check_original_staged(layout, recipe, plan_files) if original_staged else {}
    manifest_obj = corpus_manifest(facts, paths.root if paths.root in corpus_root.parents else corpus_root)
    desired = {
        'recipe': recipe_bytes,
        'corpus_manifest': manifest_text(manifest_obj).encode('utf-8'),
        'pool_recipe': recipe_text(pool_recipe(recipe, corpus_root)).encode('utf-8'),
    }
    links = {fact['name']: os.path.relpath(fact['files']['train']['path'], layout.corpus_train(fact['name']).parent)
             for fact in facts}

    def eval_ok(fact, target) -> bool:
        """A staged held-out file is fine when it is the pool's file (or, --original-staged, the original)."""
        digest = file_record(target)['sha256']
        pin = plan_files.get(f'eval/{fact["name"]}.jsonl.gz')
        return digest == fact['files']['eval']['sha256'] or (pin is not None and digest == pin['sha256'])

    # --- what would change, and the resume-pairing guard -------------------------------------------
    changes = []
    if _read(layout.recipe) not in (None, desired['recipe']):
        changes.append(str(layout.recipe))
    if not original_staged and _read(layout.corpus_manifest) not in (None, desired['corpus_manifest']):
        changes.append(str(layout.corpus_manifest))
    for fact in facts:
        link = layout.corpus_train(fact['name'])
        if not original_staged and (link.exists() or link.is_symlink()):
            if not link.is_symlink():
                if not force:
                    raise StagingError(f'{link} is a regular file (an artifact you brought?); staging would replace it '
                                       f'with a link to {fact["files"]["train"]["path"]}. If these are the original '
                                       f'trainer files, pass --original-staged; else move it away or pass --force')
                changes.append(str(link))
            elif not _links_to(link, fact['files']['train']['path']):
                changes.append(str(link))
        target = layout.eval_file(fact['name'])
        if target.exists() and not eval_ok(fact, target):
            changes.append(str(target))
    runs = nonempty_run_dirs(paths)
    if changes and runs and not force:
        raise StagingError('staging would change ' + ', '.join(changes) + ' while run directories exist ('
                           + ', '.join(runs) + '). Both trainers hash the recipe and the corpus manifest into '
                           'run_config.json and refuse to resume after a change. Remedy: keep the current inputs, '
                           'or pass --force only if those runs will be restarted from scratch')
    trained = trained_pool_tokenizers(paths)
    if changes and trained and not force:
        raise StagingError(f'staging would change {", ".join(changes)} after {trained} pool tokenizer(s) were trained '
                           f'from the previous corpus ({paths.pool_tokenizers}). Remedy: keep the current inputs, or '
                           f'pass --force only if the tokenizer pool will be regenerated from the new corpus')
    if changes:
        log(f'[hyak stage-inputs] replacing: {", ".join(changes)}' + (' (--force)' if runs or trained else ''))
    # --- write ---------------------------------------------------------------------------------------------
    written = 0
    total = 2 * len(facts) + 3

    def tick(message):
        nonlocal written
        written += 1
        if writer is not None and (written % 8 == 0 or written == total):
            writer.update(written, stage='staging', total=total, task=message)

    layout.input.mkdir(parents=True, exist_ok=True)
    if _read(layout.recipe) != desired['recipe']:
        copied = atomic_copy(Path(repo or repo_root()) / RECIPE_SOURCE, layout.recipe)
        if copied['sha256'] != RECIPE_SHA256:
            raise StagingError(f'copy of the recipe has SHA-256 {copied["sha256"]}')
    tick('recipe')
    for fact in facts:
        link = layout.corpus_train(fact['name'])
        train_path = fact['files']['train']['path']
        if not original_staged:
            if not _links_to(link, train_path):
                _make_link(link, links[fact['name']])
                if not _links_to(link, train_path):
                    # a symlinked ancestor can defeat a lexical relative path: use the absolute target instead
                    _make_link(link, str(train_path))
                    if not _links_to(link, train_path):
                        raise StagingError(f'{link} does not resolve to {train_path}')
            links[fact['name']] = os.readlink(link)
        tick(f'link {fact["name"]}')
        target = layout.eval_file(fact['name'])
        if not target.exists() or not eval_ok(fact, target):
            copied = atomic_copy(fact['files']['eval']['path'], target)
            if copied['sha256'] != fact['files']['eval']['sha256']:
                raise StagingError(f'copy of {fact["files"]["eval"]["path"]} has SHA-256 {copied["sha256"]}, '
                                   f'expected {fact["files"]["eval"]["sha256"]}; the source changed while copying')
        tick(f'held-out {fact["name"]}')
    if not original_staged and _read(layout.corpus_manifest) != desired['corpus_manifest']:
        atomic_write_text(layout.corpus_manifest, desired['corpus_manifest'].decode('utf-8'))
    tick('corpus manifest')
    if _read(layout.pool_recipe) != desired['pool_recipe']:
        atomic_write_text(layout.pool_recipe, desired['pool_recipe'].decode('utf-8'))
    tick('pool recipe')
    extra_eval = sorted(p.name for p in layout.eval.glob('*.jsonl.gz')
                        if p.stem.removesuffix('.jsonl') not in TRAINER_EVAL_BUCKETS)
    if extra_eval:
        log(f'[hyak stage-inputs] WARNING: {layout.eval} also holds {", ".join(extra_eval)}; the trainer ignores '
            f'them, but the layout should contain exactly the 23 bucket files')
    # --- post-check from the trainer's point of view ----------------------------------------------------
    problems = check_contract(layout, recipe)
    if problems:
        raise StagingError('staged inputs fail the trainer contract: ' + '; '.join(problems))
    files = {layout.relative(layout.recipe): file_record(layout.recipe),
             layout.relative(layout.pool_recipe): file_record(layout.pool_recipe),
             layout.relative(layout.corpus_manifest): file_record(layout.corpus_manifest)}
    for fact in facts:
        relative = layout.relative(layout.corpus_train(fact['name']))
        if original_staged:
            files[relative] = originals[relative]
        else:
            files[relative] = {'bytes': fact['files']['train']['bytes'], 'sha256': fact['files']['train']['sha256'],
                               'link': links[fact['name']]}
        record = file_record(layout.eval_file(fact['name']))
        files[layout.relative(layout.eval_file(fact['name']))] = {'bytes': record['bytes'], 'sha256': record['sha256']}
    files.update(model_files)
    provenance = {
        'staged_at': utc_now(), 'corpus_root': str(corpus_root), 'test_scale': test_scale,
        'mode': 'original-staged' if original_staged else 'links',
        'recipe_sha256': RECIPE_SHA256,
        'corpus_manifest_sha256': files[layout.relative(layout.corpus_manifest)]['sha256'],
        'raw_train_bytes': {fact['name']: fact['counts']['train_bytes'] for fact in facts},
        'shortfall_bytes': {fact['name']: fact['shortfall_bytes'] for fact in facts if fact['shortfall_bytes']},
        'bucket_manifest_sha256': {fact['name']: fact['manifest_sha256'] for fact in facts},
        'gzip_mtime': {fact['name']: fact['gzip_mtime'] for fact in facts},
        'scorecard': scorecard,
        'job_id': slurm.job_id(), 'host': slurm.node_name(),
    }
    previous = read_json(layout.manifest, None) if layout.manifest.is_file() else None
    if isinstance(previous, dict):
        before = (previous.get('provenance') or {}).get('corpus') or {}
        same = ({k: v for k, v in before.items() if k not in _VOLATILE_PROVENANCE}
                == {k: v for k, v in provenance.items() if k not in _VOLATILE_PROVENANCE})
        if same and all((previous.get('files') or {}).get(k) == v for k, v in files.items()):
            provenance.update({k: before.get(k) for k in _VOLATILE_PROVENANCE})  # nothing changed: same bytes
    update_input_manifest(layout.manifest, files, provenance_key='corpus', provenance=provenance)
    report = {'input': str(layout.input), 'corpus_root': str(corpus_root), 'test_scale': test_scale,
              'mode': provenance['mode'], 'changed': changes, 'files': len(files), 'buckets': len(facts),
              'model_files': len(model_files), 'scorecard': scorecard,
              'corpus_manifest_sha256': provenance['corpus_manifest_sha256'],
              'shortfall_bytes': provenance['shortfall_bytes']}
    if test_scale:
        log('[hyak stage-inputs] WARNING: TEST-SCALE corpus staged (--allow-test-scale): buckets are far below the '
            'recipe quotas; never train on these inputs')
    return report


def check_staged(paths: Paths, *, corpus_root=None, input_dir=None, repo=None, model_expected=None) -> list:
    """``--check``: problems with what is staged now (no writes); [] when everything is consistent.

    Without ``corpus_root`` the corpus root recorded by the last stage is checked (so a stage with
    ``--corpus-root`` can be re-checked with a plain ``--check``).
    """
    layout = InputLayout(paths, input_dir)
    recorded = ((read_json(layout.manifest, {}) or {}).get('provenance', {}).get('corpus', {}) or {}).get('corpus_root')
    corpus_root = Path(os.path.normpath(str(corpus_root or recorded or paths.corpus_root)))
    problems = []
    try:
        recipe_bytes, recipe = load_recipe(repo)
        check_corpus_root(paths, corpus_root, log=lambda *a: None)
    except StagingError as exc:
        return [str(exc)]
    if _read(layout.recipe) != recipe_bytes:
        problems.append(f'{layout.recipe} is missing or differs from {RECIPE_SOURCE}')
    problems += check_contract(layout, recipe)
    try:
        staged = json.loads(layout.corpus_manifest.read_text())
    except (OSError, ValueError):
        staged = {}
    provenance = (read_json(layout.manifest, {}) or {}).get('provenance', {}).get('corpus', {})
    allow = bool(provenance.get('test_scale'))
    original = provenance.get('mode') == 'original-staged'
    plan_files = {}
    if original:
        try:
            plan_files = load_inputs_plan(repo)
            check_original_staged(layout, recipe, plan_files)
        except StagingError as exc:
            problems.append(str(exc))
    for bucket in recipe['buckets']:
        try:
            fact = check_bucket(corpus_root, bucket, allow_test_scale=allow)
        except StagingError as exc:
            problems.append(str(exc))
            continue
        entry = staged.get(bucket['name'], {})
        if not original and entry.get('sha256') != fact['files']['train']['sha256']:
            problems.append(f'{bucket["name"]}: staged train file differs from the pool file')
        target = layout.eval_file(bucket['name'])
        pin = plan_files.get(f'eval/{bucket["name"]}.jsonl.gz')
        digest = file_record(target)['sha256'] if target.is_file() else None
        if digest is None or (digest != fact['files']['eval']['sha256'] and (pin is None or digest != pin['sha256'])):
            problems.append(f'{bucket["name"]}: {target} is missing or differs from the pool eval.jsonl.gz')
    if _read(layout.pool_recipe) != recipe_text(pool_recipe(recipe, corpus_root)).encode('utf-8'):
        problems.append(f'{layout.pool_recipe} is missing or does not point at {corpus_root}')
    if provenance and not allow:
        try:
            check_model(paths, layout, model_expected)
        except StagingError as exc:
            problems.append(str(exc))
    return problems


def check_warnings(paths: Paths, *, corpus_root=None, input_dir=None) -> list:
    """``--check``: things that are consistent but must be seen (test-scale stage, corpus outside $ANYTOK_ROOT)."""
    layout = InputLayout(paths, input_dir)
    provenance = (read_json(layout.manifest, {}) or {}).get('provenance', {}).get('corpus', {})
    warnings = []
    if provenance.get('test_scale'):
        warnings.append(f'{layout.input} holds a TEST-SCALE corpus (staged with --allow-test-scale): never train on it')
    root = Path(os.path.normpath(str(corpus_root))) if corpus_root else Path(provenance.get('corpus_root') or paths.corpus_root)
    if os.path.normpath(str(root)) != os.path.normpath(str(paths.corpus_root)):
        warnings.append(f'the staged links point into {root}, outside $ANYTOK_ROOT/corpus; it must stay unchanged until '
                        'both runs and their evaluation are finished')
    scorecard = provenance.get('scorecard') or {}
    if scorecard.get('accepted_reason'):
        warnings.append(f'staged with a non-exact corpus scorecard, accepted because: {scorecard["accepted_reason"]}')
    return warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.corpus.stage_inputs', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-root', type=Path, help='default: $ANYTOK_ROOT/corpus')
    parser.add_argument('--input-dir', type=Path, help='default: $ANYTOK_ROOT/input (other values for tests)')
    parser.add_argument('--force', action='store_true',
                        help='replace changed inputs even though run directories exist (breaks their resume)')
    parser.add_argument('--allow-test-scale', action='store_true',
                        help='TEST ONLY: accept test-scale buckets and relax the quota assertion')
    parser.add_argument('--accept-corpus-scorecard', metavar='REASON',
                        help='stage although the corpus scorecard is not exact (needs the user\'s approval; the reason '
                             'is recorded in input/manifest.json)')
    parser.add_argument('--original-staged', action='store_true',
                        help='input/corpus holds the ORIGINAL staged trainer files (sizes and SHA-256 of '
                             f'{INPUTS_PLAN}): keep and register them instead of links')
    parser.add_argument('--check', action='store_true', help='verify the staged inputs; write nothing')
    parser.add_argument('--monitor-dir', type=Path, help='status directory (default: the stage-inputs monitor dir)')
    parser.add_argument('--config', type=Path, help='config file (default $HYAK_CONFIG or hyak/config.env)')
    args = parser.parse_args(argv)
    if args.accept_corpus_scorecard is not None and not args.accept_corpus_scorecard.strip():
        parser.error('--accept-corpus-scorecard needs a reason')
    try:
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
    except (ConfigError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return EXIT_USAGE
    if args.check:
        problems = check_staged(paths, corpus_root=args.corpus_root, input_dir=args.input_dir)
        for warning in check_warnings(paths, corpus_root=args.corpus_root, input_dir=args.input_dir):
            print(f'WARNING: {warning}')
        for problem in problems:
            print(f'PROBLEM: {problem}')
        print('OK: staged inputs are consistent' if not problems else f'{len(problems)} problem(s)')
        return EXIT_OK if not problems else EXIT_CHECK
    from hyak.common.status import StatusWriter
    meta = STAGE_PROGRESS[STAGE_STAGE_INPUTS]
    writer = StatusWriter(STAGE_STAGE_INPUTS, meta['total'], meta['unit'], meta['completed_field'], cfg=cfg,
                          monitor_dir=args.monitor_dir)
    writer.update(0, stage='checking corpus')
    try:
        report = stage(paths, corpus_root=args.corpus_root, input_dir=args.input_dir, force=args.force,
                       allow_test_scale=args.allow_test_scale, writer=writer,
                       accept_corpus_scorecard=args.accept_corpus_scorecard, original_staged=args.original_staged)
    except StagingError as exc:
        code, label = ((EXIT_STOP, 'corpus stop') if isinstance(exc, ScorecardStop) else
                       (EXIT_USAGE, 'environment') if isinstance(exc, EnvironmentProblem) else (EXIT_CHECK, 'refused'))
        print(f'ERROR: {exc}', file=sys.stderr)
        writer.update(0, stage=label, state='failed', error=str(exc)[:2000])
        return code
    except Exception as exc:  # noqa: BLE001 -- unexpected: full traceback, exit 1
        traceback.print_exc()
        writer.update(0, stage='crashed', state='failed', error=f'{type(exc).__name__}: {exc}'[:2000])
        return 1
    writer.update(report['files'], stage='complete', state='complete', total=report['files'], **{
        'report': report})
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
