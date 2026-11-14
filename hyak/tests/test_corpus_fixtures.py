"""Shared helpers for the corpus tests (no tests of its own besides self-checks).

* :func:`load_original` executes the Modal-free part of the *original*
  ``scripts/modal_prepare_corpus.py`` (every import except ``modal``, every constant, every
  undecorated function) so tests can run original and port side by side.
* :func:`extract_function` executes one function of a pinned file (e.g. the trainer's
  ``verify_corpus`` / ``build_eval_set``) without importing the file.
* Fake ``requests`` / ``datasets`` modules, synthetic record streams, frozen gzip time, and a
  production-like synthetic pool of all 23 buckets.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import random
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / 'scripts' / 'modal_prepare_corpus.py'
TRAINER = REPO / 'runtime' / 'anytok_pool_20k' / 'scripts' / 'train_joint_smollm_input.py'
RL_TRAINER = REPO / 'runtime' / 'anytok_pool_settings_rl_20k' / 'scripts' / 'train_joint_smollm_input.py'
#: The trainer the port's RL run executes (D8: straight-BPB-reward variant of the joint regret trainer).
VARIANT_RL_TRAINER = REPO / 'hyak' / 'variants' / 'anytok_pool_straight_rl_20k' / 'scripts' / 'train_joint_smollm_input.py'
JOINT_TRAINING = REPO / 'runtime' / 'anytok_pool_20k' / 'hf_hypernet' / 'joint_training.py'


def _is_modal(node) -> bool:
    return 'modal' in ast.unparse(node).split('(')[0] or 'modal.' in ast.unparse(node)


def load_original() -> dict:
    """Namespace of the original module without Modal (imports, constants, plain functions)."""
    tree = ast.parse(ORIGINAL.read_text(encoding='utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'modal' for alias in node.names):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.Assign) and not _is_modal(node.value):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and not node.decorator_list:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    namespace = {'__name__': 'original_modal_prepare_corpus'}
    exec(compile(module, str(ORIGINAL), 'exec'), namespace)
    return namespace


def function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(f'{name} not found in {path}')


def extract_function(path: Path, name: str, namespace: dict):
    """Compile and return function ``name`` of ``path`` with ``namespace`` as its globals."""
    node = function_node(path, name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


# --- fakes ---------------------------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, body=b''):
        self.status_code = status_code
        self.raw = io.BytesIO(body)
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeHTTPError(f'HTTP {self.status_code}')

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeHTTPError(Exception):
    pass


def fake_requests(script):
    """A ``requests`` stand-in.  ``script(url, call_number) -> bytes | int (status) | Exception``."""
    module = types.ModuleType('requests')

    class RequestException(Exception):
        pass

    class ConnectionError(RequestException):  # noqa: A001 (mirrors requests)
        pass

    class Timeout(RequestException):
        pass

    module.RequestException = RequestException
    module.ConnectionError = ConnectionError
    module.Timeout = Timeout
    module.calls = []

    def get(url, stream=False, timeout=None, headers=None):
        module.calls.append({'url': url, 'stream': stream, 'timeout': timeout, 'headers': headers})
        outcome = script(url, sum(1 for c in module.calls if c['url'] == url))
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, int):
            return FakeResponse(outcome)
        return FakeResponse(200, outcome)

    module.get = get
    return module


def fake_datasets(table):
    """A ``datasets`` stand-in.  ``table(args, kwargs) -> list of records`` for each load_dataset call."""
    module = types.ModuleType('datasets')
    module.calls = []

    def load_dataset(*args, **kwargs):
        module.calls.append((args, kwargs))
        return list(table(args, kwargs))

    module.load_dataset = load_dataset
    return module


def gz_lines(records) -> bytes:
    return gzip.compress(b''.join(json.dumps(r, ensure_ascii=False).encode('utf-8') + b'\n' for r in records))


def install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def freeze_gzip_time(monkeypatch, value=1_700_000_000.0):
    """Make gzip headers deterministic (gzip.GzipFile writes time.time() when mtime is None)."""
    monkeypatch.setattr(gzip.time, 'time', lambda: value)


# --- synthetic records --------------------------------------------------------------------------------
_WORDS = ('alpha', 'beta', 'gamma', 'δέλτα', 'эпсилон', '中文', 'العربية', 'हिन्दी', '日本語', 'café', '🙂',
          'line\nbreak', 'tab\tstop', 'quote"mark', 'back\\slash', 'nul\x00byte', 'cr\rreturn')


def synthetic_text(rng: random.Random, min_bytes=30, max_bytes=3000) -> str:
    target = rng.randint(min_bytes, max_bytes)
    words = []
    size = 0
    while size < target:
        word = rng.choice(_WORDS)
        words.append(word)
        size += len(word.encode('utf-8')) + 1
    return ' '.join(words)


def synthetic_records(kind: str, count: int, seed: int = 0, prefix: str = 'doc'):
    """Records shaped like each source kind (with empty/short/odd entries mixed in)."""
    rng = random.Random(seed)
    for i in range(count):
        text = synthetic_text(rng)
        roll = rng.random()
        if roll < 0.03:
            text = ''
        elif roll < 0.06:
            text = 'short'
        if kind == 'code':
            yield {'repo_name': f'owner{i % 7}/repo{i % 13}', 'path': f'src/{prefix}{i}.py', 'code': text,
                   'license': rng.choice(['mit', 'MIT', 'gpl-3.0', 'apache-2.0']),
                   'language': rng.choice(['Python', 'Ruby', 'C'])}
        elif kind == 'qa':
            yield {'id': f'{prefix}-{i}', 'metadata': {'provenance': f'stackexchange-dolma-0040.json.gz:{i}'},
                   'text': text}
        elif kind == 'books':
            yield {'id': f'gutenberg:{i // 3}:{i % 3}', 'split_key': f'gutenberg:{i // 3}', 'text': text or 'x' * 120}
        else:
            identifier = '' if roll > 0.98 else f'{prefix}-{i}'
            yield {'id': identifier, 'url': f'https://example.org/{i}', 'text': text}


def bucket_spec(kind='fineweb', **extra):
    spec = {'name': 'test-bucket', 'kind': kind, 'train_target': 6000, 'eval_target': 1500}
    spec.update(extra)
    return spec


# --- production-like synthetic pool -----------------------------------------------------------------------
def record_id(kind: str, bucket: str, side: str, i: int) -> str:
    """A record id in the shape the builder writes for ``kind`` (see sources.record_fields)."""
    if kind == 'code':
        return f'owner{i}/{bucket}-{side}-repo{i}/src/main.py'
    if kind == 'books':
        return f'gutenberg:{bucket}{side}{i}:0'
    return f'{bucket}-{side}-{i}'


def write_bucket(directory: Path, spec: dict, *, counts=None, test_scale=None, seed=0, train_docs=25, eval_docs=12):
    """Write ``train/eval.jsonl.gz`` + a manifest in the builder's format (ids obey the held-out rule).

    ``counts`` overrides the manifest counts (e.g. claim the original pool's ``train_bytes`` while
    the files are tiny) so staging can be tested without gigabytes; ``test_scale`` adds the marker.
    """
    from hyak.corpus import sources
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    repo, config, policy = sources.source_info(spec)
    hash_policy = policy not in sources.SEPARATE_EVAL_POLICIES
    rows = {'train': [], 'eval': []}
    i = 0
    while len(rows['train']) < train_docs or len(rows['eval']) < eval_docs:
        for side in ('train', 'eval'):
            identifier = record_id(spec['kind'], spec['name'], side, i)
            held = sources.held_out(sources.split_key_of(spec['kind'], identifier))
            if len(rows[side]) < (train_docs if side == 'train' else eval_docs) and (
                    not hash_policy or held == (side == 'eval')):
                rows[side].append({'id': identifier, 'text': synthetic_text(rng, 300, 3000)})
        i += 1
    for side in ('train', 'eval'):
        with gzip.open(directory / f'{side}.jsonl.gz', 'wb') as handle:
            for row in rows[side]:
                handle.write(json.dumps(row, ensure_ascii=False).encode('utf-8') + b'\n')
    real = {'train_bytes': sum(len(r['text'].encode()) for r in rows['train']), 'train_docs': train_docs,
            'eval_bytes': sum(len(r['text'].encode()) for r in rows['eval']), 'eval_docs': eval_docs,
            'seen_train_rows': i, 'seen_eval_rows': 0, 'skipped_docs': 0}
    manifest = {'spec': spec, 'dataset': repo, 'config': config, 'revision': sources.SOURCE_REVISIONS[repo],
                'split_policy': policy, 'counts': dict(real, **(counts or {})),
                'files': {n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
                          for n in ('train.jsonl.gz', 'eval.jsonl.gz')}}
    if test_scale is not None:
        manifest['test_scale'] = test_scale
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def write_pool(corpus_root: Path, *, test_scale=None, claim_original_counts=True):
    """All 23 buckets.  Production-like by default: manifests claim the recipe's train_available_bytes."""
    from hyak.corpus import sources
    recipe = json.loads((REPO / 'data' / 'balanced-full-v3.json').read_text())
    available = {b['name']: b['train_available_bytes'] for b in recipe['buckets']}
    manifests = {}
    for index, (name, (size, spec)) in enumerate(sources.bucket_specs().items()):
        if test_scale is not None:
            spec = dict(spec, train_target=test_scale['train_target'], eval_target=test_scale['eval_target'])
        counts = {'train_bytes': available[name]} if claim_original_counts and test_scale is None else None
        manifests[name] = write_bucket(Path(corpus_root) / f'{size}-v1' / name, spec, counts=counts,
                                       test_scale=test_scale, seed=index)
    return manifests


def test_original_namespace_has_the_ported_functions():
    namespace = load_original()
    for name in ('bucket_plan', 'prepare_bucket', 'records', 'held_out', 'SOURCE_REVISIONS', '_write_record'):
        assert callable(namespace[name]) or isinstance(namespace[name], dict)
    # the Modal-decorated functions and objects are not executed
    for name in ('modal', 'app', 'volume', 'image', 'prepare', 'verify_corpus', 'publish_recipes', 'main'):
        assert name not in namespace
