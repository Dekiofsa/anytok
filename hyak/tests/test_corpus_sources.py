"""The port in hyak/corpus/sources.py behaves exactly like scripts/modal_prepare_corpus.py.

Strategy: the original module is executed without Modal (test_corpus_fixtures.load_original) and
run side by side with the port on the same fake upstream data; outputs must be identical byte
for byte (gzip time frozen).  Unchanged functions must also be AST-identical to the original.
"""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import random
import zlib

import pytest

from hyak.corpus import sources
from hyak.tests.test_corpus_fixtures import (ORIGINAL, REPO, bucket_spec, fake_datasets, fake_requests,
                                             freeze_gzip_time, function_node, gz_lines, install, load_original,
                                             synthetic_records)

UNCHANGED = ('bucket_plan', 'extra_plan', 'specialty_plan', 'balanced_recipe_targets', 'specialty_recipe_targets',
             'plan_for_size', 'source_info', '_hf_records', '_code_records', '_book_records', 'records',
             'record_fields', 'held_out', '_write_record', '_sha256')
CONSTANTS = ('VOLUME_NAME', 'SOURCE_REVISIONS', 'NON_ENGLISH', 'WIKI_LANGUAGES', 'PILOT_BYTES', 'FULL_BYTES',
             'EXTRA_BYTES', 'EVAL_BYTES_PER_BUCKET', 'MAX_DOCUMENT_BYTES', 'PERMISSIVE_CODE_LICENSES',
             'CODE_LANGUAGES')


def _strip_docstring(node):
    node = copy.deepcopy(node)
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
        node.body = node.body[1:]
    return node


def _port_node(name):
    return function_node(REPO / 'hyak' / 'corpus' / 'sources.py', name)


def test_original_source_is_the_ported_version():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == sources.ORIGINAL_SOURCE_SHA256


@pytest.mark.parametrize('name', UNCHANGED)
def test_unchanged_functions_are_ast_identical(name):
    original = _strip_docstring(function_node(ORIGINAL, name))
    port = _strip_docstring(_port_node(name))
    assert ast.dump(original, include_attributes=False) == ast.dump(port, include_attributes=False)


class _RequestsToOpenUrl(ast.NodeTransformer):
    """The only allowed edit in the requests adapters: requests.get(url, stream=True, t) -> _open_url(url, t)."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node.func) == 'requests.get':
            keywords = [k for k in node.keywords if k.arg != 'stream']
            return ast.Call(func=ast.Name('_open_url', ast.Load()), args=node.args, keywords=keywords)
        return node


@pytest.mark.parametrize('name', ('_paper_records', '_common_pile_records', '_specialty_records'))
def test_requests_adapters_differ_only_in_the_request_call(name):
    original = _strip_docstring(function_node(ORIGINAL, name))
    original.body = [n for n in original.body if not (isinstance(n, ast.Import) and n.names[0].name == 'requests')]
    original = ast.fix_missing_locations(_RequestsToOpenUrl().visit(original))
    port = _strip_docstring(_port_node(name))
    assert ast.dump(original, include_attributes=False) == ast.dump(port, include_attributes=False)


def test_constants_equal_the_original():
    namespace = load_original()
    for name in CONSTANTS:
        assert getattr(sources, name) == namespace[name], name
    assert str(sources.ROOT) == '/corpus'


# --- plans and recipe targets ------------------------------------------------------------------------
def test_pool_plans_and_targets():
    full = sources.plan_for_size('full')
    assert [s['name'] for s in full][:2] == ['web-en', 'web-es'] and len(full) == 18
    assert sum(s['train_target'] for s in full) == 9_999_999_994
    assert [s['name'] for s in sources.plan_for_size('extra')] == ['math-en', 'qa-en', 'books-en']
    assert [s['name'] for s in sources.plan_for_size('specialty')] == ['medical-en', 'legal-en']
    total = sum(s['train_target'] for size in sources.POOL_SIZES for s in sources.plan_for_size(size))
    assert total == 13_599_999_994  # notes corpus.md 3.2: original pool targets, 13.6 GB
    assert all(s['eval_target'] == 5_000_000 for size in sources.POOL_SIZES for s in sources.plan_for_size(size))
    assert len(sources.bucket_specs()) == 23
    with pytest.raises(ValueError):
        sources.plan_for_size('huge')


def test_recipe_targets_regenerate_the_checked_in_recipe():
    recipe = json.loads((REPO / 'data' / 'balanced-full-v3.json').read_text())
    targets = sources.specialty_recipe_targets(sources.FULL_BYTES)
    assert list(targets) == [b['name'] for b in recipe['buckets']]
    assert list(targets.values()) == [b['train_target_bytes'] for b in recipe['buckets']]
    for bucket in recipe['buckets']:
        assert bucket['source_path'] == f'/corpus/{sources.recipe_pool(bucket["name"])}-v1/{bucket["name"]}'
        assert sources.SOURCE_REVISIONS[bucket['dataset']] == bucket['revision']
        size, spec = sources.bucket_specs()[bucket['name']]
        assert size == sources.recipe_pool(bucket['name'])
        assert spec['train_target'] <= bucket['train_available_bytes'] < spec['train_target'] + sources.MAX_DOCUMENT_BYTES
        assert sources.source_info(spec)[0] == bucket['dataset']


def test_publish_recipe_v3_reproduces_the_recipe_bytes(tmp_path):
    """With the original counts, the ported publish_recipes writes data/balanced-full-v3.json exactly."""
    checked_in = (REPO / 'data' / 'balanced-full-v3.json').read_bytes()
    recipe = json.loads(checked_in)
    for bucket in recipe['buckets']:
        size, spec = sources.bucket_specs()[bucket['name']]
        directory = tmp_path / f'{size}-v1' / bucket['name']
        directory.mkdir(parents=True)
        repo = sources.source_info(spec)[0]
        (directory / 'manifest.json').write_text(json.dumps({
            'spec': spec, 'dataset': repo, 'revision': sources.SOURCE_REVISIONS[repo],
            'counts': {'train_bytes': bucket['train_available_bytes']}}))
    regenerated, text = sources.publish_recipe_v3(tmp_path)
    assert text.encode('utf-8') == checked_in
    assert hashlib.sha256(text.encode()).hexdigest() == 'b3bc193333aa13d3a513f1c8978805835f837190cd0e5991742c55e9a0752cfe'
    manifest = tmp_path / 'full-v1' / 'web-en' / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['counts']['train_bytes'] = 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='Insufficient data'):
        sources.publish_recipe_v3(tmp_path)


# --- held-out rule, ids, record format -----------------------------------------------------------------
def test_held_out_rule_on_recorded_ids():
    namespace = load_original()
    excerpts = {}
    tree = ast.parse((REPO / 'scripts' / 'extract_corpus_samples.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'EXCERPTS':
            excerpts = ast.literal_eval(node.value)
    hash_buckets = 0
    for bucket, (identifier, _start, _end) in excerpts.items():
        size, spec = sources.bucket_specs()[bucket]
        if sources.source_info(spec)[2] in sources.SEPARATE_EVAL_POLICIES:
            continue
        key = sources.split_key_of(spec['kind'], identifier)
        assert sources.held_out(key) and namespace['held_out'](key), (bucket, identifier)
        hash_buckets += 1
    assert hash_buckets == 15  # notes corpus.md 3.4
    for key in ('', 'a', '<urn:uuid:x>', '中文'):
        assert sources.held_out(key) == (hashlib.sha256(key.encode()).digest()[0] < 26)


def test_split_key_inverts_record_fields():
    for kind, record in (('code', {'repo_name': 'own/repo', 'path': 'a/b/c.py', 'code': 'x'}),
                         ('qa', {'id': 'i', 'metadata': {'provenance': 'se-0040.json.gz:7'}, 'text': 'x'}),
                         ('books', {'id': 'gutenberg:12:3', 'split_key': 'gutenberg:12', 'text': 'x'}),
                         ('fineweb', {'id': '<urn:uuid:1>', 'text': 'x'}),
                         ('math', {'url': 'https://u/v', 'text': 'x'})):
        _text, identifier, split_key = sources.record_fields({'kind': kind}, record)
        assert sources.split_key_of(kind, identifier) == split_key
    with pytest.raises(ValueError):
        sources.split_key_of('code', 'norepo')
    with pytest.raises(ValueError):
        sources.split_key_of('books', 'other:1')


def test_write_record_format():
    import io
    buffer = io.BytesIO()
    sources._write_record(buffer, 'tëxt "q"\n\x00🙂', 'id/1')
    assert buffer.getvalue() == '{"id": "id/1", "text": "tëxt \\"q\\"\\n\\u0000🙂"}\n'.encode('utf-8')


# --- adapters: original vs port on fake upstream data ---------------------------------------------------
def _shard_records(name, count, text_bytes):
    return [{'id': f'{name}:{i}', 'text': 'x' * text_bytes, 'metadata': {'provenance': f'{name}:{i}', 'language': 'en'}}
            for i in range(count)]


def _first(generator, n):
    out = []
    for record in generator:
        out.append(record)
        if len(out) == n:
            break
    generator.close()
    return out


def test_common_pile_and_specialty_shard_order_and_requests(monkeypatch):
    def script(url, call):
        return gz_lines(_shard_records(url.rsplit('/', 1)[-1], 3, 200))

    fake = fake_requests(script)
    install(monkeypatch, 'requests', fake)
    namespace = load_original()
    for kind in ('qa', 'books', 'medical', 'legal'):
        fake.calls.clear()
        port = _first(sources.records({'kind': kind}, 'train'), 30)
        port_calls = list(fake.calls)
        fake.calls.clear()
        original = _first(namespace['records']({'kind': kind}, 'train'), 30)
        assert port == original
        assert [c['url'] for c in port_calls] == [c['url'] for c in fake.calls]
        assert [c['timeout'] for c in port_calls] == [c['timeout'] for c in fake.calls]
        assert all(c['stream'] is True and c['headers'] == {} for c in port_calls)
        assert all(c['stream'] is True and c['headers'] is None for c in fake.calls)
    fake.calls.clear()
    _first(sources.records({'kind': 'qa'}, 'train'), 12)
    shards = [int(c['url'].rsplit('-', 1)[-1].split('.')[0]) for c in fake.calls]
    assert shards[:4] == [40, 39, 10, 25]  # notes corpus.md 3.3
    fake.calls.clear()
    _first(sources.records({'kind': 'medical'}, 'train'), 9)
    assert [c['url'].rsplit('/', 1)[-1] for c in fake.calls][:3] == [
        '00014_pubmedcentral.jsonl.gz', '00002_pubmedcentral.jsonl.gz', '00004_pubmedcentral.jsonl.gz']
    assert all('/resolve/648b8cfc93953ca0663a9c96a8d842a91b98fb64/data/' in c['url'] for c in fake.calls)
    fake.calls.clear()
    _first(sources.records({'kind': 'legal'}, 'train'), 6)
    assert fake.calls[0]['url'].endswith('/CAP-Dolma-0001.json.gz') and fake.calls[0]['timeout'] == (30, 180)
    fake.calls.clear()
    _first(sources.records({'kind': 'books'}, 'train'), 3)
    assert fake.calls[0]['url'].endswith('/project_gutenberg-dolma-0007.json.gz')


def test_shard_cut_at_90_and_100_megabytes(monkeypatch):
    """Records after the shard's byte budget are not read (counted over every record, before filters)."""
    big = 45_000_000

    def script(url, call):
        return gz_lines([{'id': f'r{i}', 'text': 'y' * big if i < 3 else 'z' * 150,
                          'metadata': {'provenance': f'p{i}', 'language': 'en'}} for i in range(4)])

    install(monkeypatch, 'requests', fake_requests(script))
    namespace = load_original()
    port = _first(sources._common_pile_records('common-pile/stackexchange_filtered', 'stackexchange-dolma', 2), 4)
    original = _first(namespace['_common_pile_records']('common-pile/stackexchange_filtered', 'stackexchange-dolma', 2), 4)
    # 90 MB: records r0, r1 reach 90,000,000 bytes, so the next record comes from the next shard
    assert [r['id'] for r in port] == [r['id'] for r in original] == ['r0', 'r1', 'r0', 'r1']
    port = _first(sources._specialty_records('medical'), 6)
    assert [r['id'] for r in port] == ['r0', 'r1', 'r2', 'r0', 'r1', 'r2']  # 100 MB needs three 45 MB records


def test_paper_records_and_token_header(monkeypatch):
    def script(url, call):
        return gz_lines([{'id': f'{url[-30:]}-{i}', 'text': 'p' * 150} for i in range(3)])

    fake = fake_requests(script)
    install(monkeypatch, 'requests', fake)
    namespace = load_original()
    for split in ('train', 'validation'):
        fake.calls.clear()
        assert list(sources._paper_records(split)) == list(namespace['_paper_records'](split))
    assert fake.calls[0]['url'].endswith('/636a503e44a3ca1b58e01fb61eab0825cd574de0/data/v2/validation-00000-of-00002.json.gz')
    monkeypatch.setenv('HF_TOKEN', 'hf_secret_value')
    fake.calls.clear()
    list(sources._paper_records('train'))
    assert fake.calls[0]['headers'] == {'Authorization': 'Bearer hf_secret_value'}
    assert fake.calls[0]['url'].endswith('/data/v2/train-00000-of-00020.json.gz')


def test_open_url_retries_only_the_request(monkeypatch):
    sleeps = []
    monkeypatch.setattr(sources.time, 'sleep', lambda s: sleeps.append(s))
    outcomes = {}

    def script(url, call):
        return outcomes[url][call - 1]

    fake = fake_requests(script)
    install(monkeypatch, 'requests', fake)
    outcomes['u1'] = [fake.ConnectionError('reset'), 503, 429, b'body']
    response = sources._open_url('u1', timeout=(30, 120))
    assert response.status_code == 200 and response.raw.read() == b'body'
    assert len(fake.calls) == 4 and sleeps == [15.0, 30.0, 60.0]
    outcomes['u2'] = [404]
    with pytest.raises(sources.SourceUnavailable, match='no longer exists upstream'):
        sources._open_url('u2', timeout=(30, 120))
    outcomes['u3'] = [403]
    with pytest.raises(sources.SourceUnavailable, match='HF_TOKEN'):
        sources._open_url('u3', timeout=(30, 120))
    outcomes['u4'] = [fake.Timeout('slow')] * sources.OPEN_ATTEMPTS
    with pytest.raises(fake.Timeout):
        sources._open_url('u4', timeout=(30, 120))
    outcomes['u5'] = [500] * sources.OPEN_ATTEMPTS
    response = sources._open_url('u5', timeout=(30, 120))
    assert response.status_code == 500  # last attempt is returned; the adapter's raise_for_status raises
    with pytest.raises(Exception, match='HTTP 500'):
        response.raise_for_status()


def test_open_url_cannot_retry_after_the_body_was_read(monkeypatch):
    """A failure while reading a shard propagates (the bucket restarts from zero); no second request."""
    class Broken:
        def __init__(self):
            self.status_code = 200
            self.raw = self

        def read(self, *args):
            raise ConnectionResetError('mid-stream')

        def raise_for_status(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = fake_requests(lambda url, call: b'')
    fake.get = lambda url, **kw: (fake.calls.append(url), Broken())[1]
    install(monkeypatch, 'requests', fake)
    with pytest.raises(ConnectionResetError):
        list(sources._paper_records('train'))
    assert len(fake.calls) == 1


def test_hf_and_code_adapters(monkeypatch):
    rng = random.Random(3)

    def table(args, kwargs):
        if args[0] == 'parquet':
            index = int(kwargs['data_files'].rsplit('train-', 1)[1][:5])
            return [{'repo_name': f'o/r{index}', 'path': f'f{i}', 'code': 'c' * 120,
                     'license': rng.choice(['MIT', 'gpl-2.0', 'isc']), 'language': rng.choice(['GO', 'Go', 'Rust'])}
                    for i in range(5)]
        return [{'id': f'{kwargs["name"]}-{kwargs["split"]}-{i}', 'text': 't' * 150} for i in range(3)]

    fake = fake_datasets(table)
    install(monkeypatch, 'datasets', fake)
    namespace = load_original()
    for spec in ({'kind': 'fineweb'}, {'kind': 'fineweb2', 'config': 'jpn_Jpan'}, {'kind': 'wiki', 'language': 'hi'},
                 {'kind': 'math'}):
        for split in ('train', 'test'):
            fake.calls.clear()
            port = list(sources.records(spec, split))
            port_calls = list(fake.calls)
            fake.calls.clear()
            assert port == list(namespace['records'](spec, split))
            assert port_calls == fake.calls
    fake.calls.clear()
    list(sources.records({'kind': 'fineweb2', 'config': 'arb_Arab'}, 'test'))
    assert fake.calls == [(('HuggingFaceFW/fineweb-2',), {'name': 'arb_Arab', 'split': 'test', 'streaming': True,
                                                         'revision': 'af9c13333eb981300149d5ca60a8e9d659b276b9'})]
    fake.calls.clear()
    list(sources.records({'kind': 'wiki', 'language': 'zh'}, 'test'))
    assert fake.calls[0][1]['name'] == '20231101.zh' and fake.calls[0][1]['split'] == 'train'
    rng.seed(3)
    fake.calls.clear()
    port = list(sources.records({'kind': 'code'}, 'train'))
    port_calls = list(fake.calls)
    rng.seed(3)
    fake.calls.clear()
    assert port == list(namespace['records']({'kind': 'code'}, 'train'))
    assert port_calls == fake.calls
    assert all(r['license'].lower() in sources.PERMISSIVE_CODE_LICENSES and r['language'] in sources.CODE_LANGUAGES
               for r in port)
    urls = [kwargs['data_files'] for args, kwargs in port_calls]
    assert len(urls) == 32 and urls[0].endswith('/c48d40f9e70f0196f8236901ee35807f7d6c44c0/data/train-00000-of-00880.parquet')
    assert urls[-1].endswith('train-00031-of-00880.parquet')


def test_probe_selection_matches_original():
    namespace = load_original()
    tree = ast.parse(ORIGINAL.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'probe_sources')
    examples = ast.literal_eval(node.body[1].value)
    assert [dict(e) for e in sources.PROBE_EXAMPLES] == examples
    assert [s['name'] for s in sources.probe_selection('extra')] == ['math-en', 'qa-en', 'books-en']
    assert [s['name'] for s in sources.probe_selection('specialty')] == ['medical-en', 'legal-en']
    assert namespace['records'] is not None


# --- prepare_bucket: original vs port, byte for byte ----------------------------------------------------
def _run_both(monkeypatch, tmp_path, spec, streams):
    """Run original and port prepare_bucket on the same fake stream; return both result dirs."""
    freeze_gzip_time(monkeypatch)
    namespace = load_original()

    def fake_records(spec_, split):
        yield from (copy.deepcopy(r) for r in streams[split])

    namespace['records'] = fake_records
    monkeypatch.setattr(sources, 'records', fake_records)
    outcome = {}
    for label, function in (('original', namespace['prepare_bucket']), ('port', sources.prepare_bucket)):
        root = tmp_path / label
        try:
            outcome[label] = function(root, dict(spec))
        except Exception as exc:  # noqa: BLE001 -- compared below
            outcome[label] = exc
    return namespace, outcome, tmp_path / 'original' / spec['name'], tmp_path / 'port' / spec['name']


@pytest.mark.parametrize('kind,extra', [('fineweb', {}), ('code', {}), ('qa', {}), ('books', {}), ('math', {}),
                                        ('fineweb2', {'config': 'spa_Latn', 'language': 'es'}), ('papers', {})])
def test_prepare_bucket_is_byte_identical_to_the_original(monkeypatch, tmp_path, kind, extra):
    spec = bucket_spec(kind, **extra)
    # a stable seed (hash() of a tuple of strings differs in every Python process)
    streams = {split: list(synthetic_records(kind, 400, seed=zlib.crc32(f'{kind}/{split}'.encode()) % 1000, prefix=split))
               for split in ('train', 'test', 'validation')}
    namespace, outcome, original_dir, port_dir = _run_both(monkeypatch, tmp_path, spec, streams)
    assert not isinstance(outcome['original'], Exception), outcome['original']
    assert outcome['port'] == outcome['original']  # same manifest (spec, counts, file hashes)
    for name in ('train.jsonl.gz', 'eval.jsonl.gz', 'manifest.json'):
        assert (port_dir / name).read_bytes() == (original_dir / name).read_bytes(), name
    assert sorted(p.name for p in port_dir.iterdir()) == ['eval.jsonl.gz', 'manifest.json', 'train.jsonl.gz']
    with gzip.open(port_dir / 'train.jsonl.gz', 'rb') as handle:
        handle.read()
    assert b'train.jsonl.gz.tmp' in (port_dir / 'train.jsonl.gz').read_bytes()[:40]  # gzip FNAME as the original
    counts = outcome['port']['counts']
    assert counts['train_bytes'] >= spec['train_target'] and counts['eval_bytes'] >= spec['eval_target']


def _ids(held: bool, count: int, prefix: str) -> list:
    """Record ids on one side of the held-out rule (fineweb: the split key is the id)."""
    found, i = [], 0
    while len(found) < count:
        identifier = f'{prefix}-{i}'
        if sources.held_out(identifier) == held:
            found.append(identifier)
        i += 1
    return found


def _boundary_streams(case):
    """Streams that hit every boundary of prepare_bucket (corpus-3): sizes 99/100/1,000,000/1,000,001 bytes,
    a text with an empty id, and an exact hit of each target followed by one more document of that side."""
    train, held = _ids(False, 8, 'train'), _ids(True, 4, 'eval')

    def rec(identifier, size):
        return {'id': identifier, 'text': 'a' * size}

    sizes = [rec(train[0], 99), rec(train[1], 100), {'id': '', 'text': 'b' * 200}, rec(train[2], 1_000_000),
             rec(train[3], 1_000_001)]
    if case == 'eval-target-first':      # eval exactly at its target while train is open, then one more held-out doc
        stream = [rec(held[0], 600), rec(held[1], 400), rec(held[2], 300)] + sizes + [rec(train[4], 500), rec(train[5], 700)]
        return bucket_spec('fineweb', train_target=1_000_600, eval_target=1000), {'train': stream}
    if case == 'train-target-first':     # train exactly at its target while eval is open, then one more train doc
        stream = sizes + [rec(train[4], 500), rec(train[5], 300), rec(held[0], 600), rec(held[1], 400), rec(held[2], 300)]
        return bucket_spec('fineweb', train_target=1_000_600, eval_target=1000), {'train': stream}
    # separate held-out split (papers: upstream-validation): exact hit in the second loop, then one more doc
    stream = sizes + [rec(train[4], 500), rec(train[5], 300)]
    validation = [rec('v-0', 99), rec('v-1', 600), {'id': '', 'text': 'c' * 150}, rec('v-2', 400), rec('v-3', 300)]
    return bucket_spec('papers', train_target=1_000_600, eval_target=1000), {'train': stream, 'validation': validation}


@pytest.mark.parametrize('case', ['eval-target-first', 'train-target-first', 'separate-eval'])
def test_prepare_bucket_boundaries_match_the_original(monkeypatch, tmp_path, case):
    spec, streams = _boundary_streams(case)
    _, outcome, original_dir, port_dir = _run_both(monkeypatch, tmp_path, spec, streams)
    assert not isinstance(outcome['original'], Exception), outcome['original']
    assert outcome['port'] == outcome['original']
    for name in ('train.jsonl.gz', 'eval.jsonl.gz', 'manifest.json'):
        assert (port_dir / name).read_bytes() == (original_dir / name).read_bytes(), name
    counts = outcome['port']['counts']
    # every boundary was reached: exact targets, 100 and 1,000,000 bytes kept, 99 / 1,000,001 / empty id skipped
    assert counts['train_bytes'] == 1_000_600 and counts['eval_bytes'] == 1000
    assert counts['train_docs'] == 3 and counts['eval_docs'] == 2
    assert counts['skipped_docs'] == (5 if case == 'separate-eval' else 3)


class _StripAdditions(ast.NodeTransformer):
    """Remove exactly the additions the port declares for prepare_bucket (SCRATCH/review/corpus/ast_prepare_bucket.py)."""

    def __init__(self):
        self.removed = []

    def visit_body(self, body):
        out, i = [], 0
        while i < len(body):
            node = body[i]
            text = ast.unparse(node)
            if (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'stream'
                    and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'records'
                    and i + 1 < len(body) and isinstance(body[i + 1], ast.Try)):
                wrapper = body[i + 1]
                assert len(wrapper.body) == 1 and isinstance(wrapper.body[0], ast.For)
                assert not wrapper.handlers and not wrapper.orelse
                assert [ast.unparse(n) for n in wrapper.finalbody] == ['_close(stream)']
                loop = wrapper.body[0]
                assert ast.unparse(loop.iter) == 'stream'
                loop.iter = node.value
                self.removed.append('stream/try/finally/_close wrapper')
                out.append(self.visit(loop))
                i += 2
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'report':
                self.removed.append(text)
            elif isinstance(node, ast.FunctionDef) and node.name == 'report':
                self.removed.append('def report')
            elif text in ('last_report = time.monotonic()', 'manifest.update(extra_manifest)',
                          'from hyak.common.io import atomic_write_text', 'extra_manifest = dict(extra_manifest or {})',
                          "reserved = {'spec', 'dataset', 'config', 'revision', 'split_policy', 'counts', 'files'}"):
                self.removed.append(text)
            elif isinstance(node, ast.If) and ast.unparse(node.test) == 'reserved & set(extra_manifest)':
                self.removed.append('if reserved & set(extra_manifest): raise')
            else:
                out.append(self.visit(node))
            i += 1
        return out

    def generic_visit(self, node):
        for field in ('body', 'orelse', 'finalbody'):
            value = getattr(node, field, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                setattr(node, field, self.visit_body(value))
        for field, value in ast.iter_fields(node):
            if field in ('body', 'orelse', 'finalbody'):
                continue
            if isinstance(value, list):
                setattr(node, field, [self.visit(v) if isinstance(v, ast.AST) else v for v in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node.func) == 'atomic_write_text' and ast.unparse(node.args[0]) == 'manifest_path':
            self.removed.append('atomic_write_text(manifest_path, X) -> manifest_path.write_text(X)')
            return ast.Call(func=ast.Attribute(value=ast.Name('manifest_path', ast.Load()), attr='write_text',
                                               ctx=ast.Load()), args=node.args[1:], keywords=[])
        return node


def test_prepare_bucket_is_the_original_plus_only_the_declared_additions():
    """corpus-3: AST of the port minus its declared additions == AST of the original (no docstrings)."""
    def body_only(node):
        node = copy.deepcopy(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body = node.body[1:]
        return node

    original = body_only(function_node(ORIGINAL, 'prepare_bucket'))
    port = body_only(function_node(REPO / 'hyak' / 'corpus' / 'sources.py', 'prepare_bucket'))
    assert [a.arg for a in port.args.kwonlyargs] == ['progress', 'extra_manifest']
    port.args.kwonlyargs, port.args.kw_defaults = [], []
    stripper = _StripAdditions()
    port = ast.fix_missing_locations(stripper.visit(port))
    assert stripper.removed == [
        'from hyak.common.io import atomic_write_text', 'extra_manifest = dict(extra_manifest or {})',
        "reserved = {'spec', 'dataset', 'config', 'revision', 'split_policy', 'counts', 'files'}",
        'if reserved & set(extra_manifest): raise', 'last_report = time.monotonic()', 'def report',
        'stream/try/finally/_close wrapper', "report('train')", 'stream/try/finally/_close wrapper', "report('eval')",
        'manifest.update(extra_manifest)', 'atomic_write_text(manifest_path, X) -> manifest_path.write_text(X)']
    assert ast.dump(port) == ast.dump(original), '\n'.join(
        __import__('difflib').unified_diff(ast.unparse(original).splitlines(), ast.unparse(port).splitlines(), lineterm=''))


def test_prepare_bucket_exhausted_source_is_identical_and_cleans_up(monkeypatch, tmp_path):
    spec = bucket_spec('fineweb', train_target=10 ** 9)
    streams = {'train': list(synthetic_records('fineweb', 50))}
    _, outcome, original_dir, port_dir = _run_both(monkeypatch, tmp_path, spec, streams)
    assert isinstance(outcome['port'], RuntimeError) and str(outcome['port']) == str(outcome['original'])
    assert 'Source exhausted before target' in str(outcome['port'])
    assert list(port_dir.iterdir()) == [] and list(original_dir.iterdir()) == []


def test_prepare_bucket_skip_and_settings_check(monkeypatch, tmp_path):
    spec = bucket_spec('fineweb')
    streams = {'train': list(synthetic_records('fineweb', 400))}
    monkeypatch.setattr(sources, 'records', lambda s, split: iter(copy.deepcopy(streams[split])))
    first = sources.prepare_bucket(tmp_path, spec, extra_manifest={'hyak': {'x': 1}})
    assert first['hyak'] == {'x': 1}
    assert sources.prepare_bucket(tmp_path, spec) == first  # skipped, returns the stored manifest
    with pytest.raises(RuntimeError, match='different settings'):
        sources.prepare_bucket(tmp_path, dict(spec, train_target=7000))
    with pytest.raises(ValueError, match='may not replace'):
        sources.prepare_bucket(tmp_path / 'other', spec, extra_manifest={'counts': {}})


def test_prepare_bucket_closes_iterators_and_reports_progress(monkeypatch, tmp_path):
    closed = []

    def stream(spec, split):
        try:
            for record in synthetic_records('fineweb', 10_000):
                yield record
        finally:
            closed.append(split)

    monkeypatch.setattr(sources, 'records', stream)
    monkeypatch.setattr(sources, 'PROGRESS_SECONDS', 0.0)
    seen = []
    sources.prepare_bucket(tmp_path, bucket_spec('fineweb'), progress=lambda counts, phase: seen.append((phase, counts)))
    assert closed == ['train']
    assert seen and seen[0][0] == 'train' and set(seen[-1][1]) >= {'train_bytes', 'eval_bytes'}


def test_verify_corpus_port_matches_original_on_a_fixture(monkeypatch, tmp_path):
    spec = bucket_spec('fineweb')
    streams = {'train': list(synthetic_records('fineweb', 400))}
    monkeypatch.setattr(sources, 'records', lambda s, split: iter(copy.deepcopy(streams[split])))
    root = tmp_path / 'full-v1'
    manifest = sources.prepare_bucket(root, spec)
    (root / 'summary.json').write_text(sources.summary_text('full', [manifest]))
    summary = json.loads((root / 'summary.json').read_text())
    assert summary == {'size': 'full', 'total_train_bytes': manifest['counts']['train_bytes'],
                       'total_eval_bytes': manifest['counts']['eval_bytes'], 'buckets': ['test-bucket']}
    assert sources.verify_corpus(tmp_path, 'full')['verified_buckets'] == 1
    with gzip.open(root / 'test-bucket' / 'eval.jsonl.gz', 'ab') as handle:
        handle.write(b'{"id": "x", "text": "extra"}\n')
    with pytest.raises(RuntimeError, match='Checksum mismatch'):
        sources.verify_corpus(tmp_path, 'full')
