"""hyak.corpus.verify: anchor extraction from the repository and the scorecard arithmetic."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import zlib

import pytest

from hyak.corpus import sources, verify
from hyak.tests.test_corpus_fixtures import REPO, write_pool


@pytest.fixture(scope='module')
def anchors():
    return verify.load_anchors(REPO)


def test_repository_anchors(anchors):
    assert anchors.problems == []  # every hash-bucket anchor id also satisfies held_out(), as the notes found
    assert all(anchors.files.values()), [k for k, v in anchors.files.items() if not v]
    counts = anchors.counts()
    assert counts == {'buckets_with_document_anchors': 23, 'document_ids': 3149, 'document_sha256': 1592,
                      'excerpts': 34, 'train_available_bytes': 23, 'eval_file_sizes': 23, 'eval_document_counts': 5,
                      'original_train_files': 23, 'original_corpus_manifest': 1, 'eval_gzip_times': 23,
                      'reference_content_hashes': 23}
    assert anchors.train_files['wiki-zh'] == {'bytes': 61_422_998, 'sha256': anchors.train_files['wiki-zh']['sha256']}
    assert anchors.joint_manifest == {'bytes': 5535, 'sha256': '363fbd074905cc60e490109fd3017513f3f54f234d092475786643a9b92c304d'}
    assert anchors.eval_gzip_mtime['web-en'] == 1792695906 and anchors.quota['web-en'] == 1_607_142_857
    assert anchors.content['code']['train']['lines'] == 124644
    assert anchors.available['web-en'] == 2_500_029_021 and anchors.available['legal-en'] == 800_006_243
    assert anchors.eval_files['code'] == {'bytes': 1_103_751, 'sha256': '135e3dabd9b37eacceef78b06714a91bb7b61e756e84ee4cd2cf2e007c8d6a5c'}
    assert anchors.eval_docs == {'web-en': 1653, 'code': 655, 'math-en': 890, 'medical-en': 148, 'legal-en': 441}
    first = anchors.documents['web-en']['<urn:uuid:c47f1461-a8d1-46d5-be9d-f4bdfbae526a>']
    assert first['sha256'] == {'5dc3e5c0938df17cdbf310cdeb348711e962e922c70ea7f2691f647c16bdbad1'}
    assert first['exact_chars'] == 3856  # reward table window [0, 3856) < 4096 is the whole document
    assert anchors.documents['wiki-ru']['9']['min_chars'] == 45472  # extract_corpus_samples.py EXCERPTS
    assert anchors.documents['medical-en']['PMC10208914']['min_chars'] >= 12288 + 256  # '::chars-12288' window
    assert anchors.documents['qa-en'] == {'stackexchange-dolma-0040.json.gz:9655': anchors.documents['qa-en'][
        'stackexchange-dolma-0040.json.gz:9655']}
    rejected = [e for e in anchors.documents['legal-en'].values() if e['max_chars'] == 1023]
    assert len(rejected) == 153  # natural-transfer initial preparation: documents under 1,024 characters
    sources_seen = set().union(*(e['sources'] for d in anchors.documents.values() for e in d.values()))
    assert {verify.REWARD_TABLE, verify.NATURAL_COMPARISON, verify.RANDOM_POOL, verify.SWEEP_SPEC, verify.PANEL_SPEC,
            verify.BROAD_SCREEN, verify.RL_PER_DOCUMENT, verify.EXCERPTS_FILE, verify.PASSAGES} <= sources_seen
    offsets = [e for e in anchors.excerpts if e['offset'] is not None]
    assert len(offsets) == 10 and all(e['source'] == verify.NATURAL_EXAMPLES for e in offsets)


def test_window_ids():
    assert verify._window_id('PMC1::chars-4096') == ('PMC1', 4352)
    assert verify._window_id('<urn:uuid:x>') == ('<urn:uuid:x>', 0)


def _eval_file(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as handle:
        for identifier, text in rows:
            handle.write(json.dumps({'id': identifier, 'text': text}, ensure_ascii=False).encode() + b'\n')


def test_score_bucket_arithmetic(tmp_path):
    text_a = 'alpha ' * 50 + 'PREFIX|CONTINUATION' + ' omega' * 20
    text_b = 'beta ' * 300
    rows = [('a', text_a), ('b', text_b), ('c', 'gamma ' * 40), ('dup', 'x' * 120), ('dup', 'y' * 120)]
    directory = tmp_path / 'full-v1' / 'web-en'
    _eval_file(directory / 'eval.jsonl.gz', rows)
    (directory / 'manifest.json').write_text(json.dumps({'counts': {'train_bytes': 2_500_029_021, 'eval_bytes': 5}}))
    anchors = verify.Anchors()
    anchors.available['web-en'] = 2_500_029_021
    anchors.eval_files['web-en'] = {'bytes': (directory / 'eval.jsonl.gz').stat().st_size, 'sha256': 'x'}
    anchors.eval_docs['web-en'] = 5
    anchors.add('web-en', 'a', 's1', sha256=hashlib.sha256(text_a.encode()).hexdigest(), min_chars=10)
    anchors.add('web-en', 'b', 's1', sha256='0' * 64)                       # different text
    anchors.add('web-en', 'c', 's2', max_chars=100)                        # too long
    anchors.add('web-en', 'missing', 's2', sha256='1' * 64)
    anchors.add('web-en', 'dup', 's3')
    offset = text_a.index('PREFIX')
    anchors.excerpts += [{'bucket': 'web-en', 'id': 'a', 'text': 'PREFIX|CONTINUATION', 'offset': offset, 'source': 's'},
                         {'bucket': 'web-en', 'id': 'a', 'text': 'PREFIX|', 'offset': offset + 1, 'source': 's'},
                         {'bucket': 'web-en', 'id': 'b', 'text': 'nowhere', 'offset': None, 'source': 's'}]
    row = verify.score_bucket(tmp_path, 'web-en', anchors)
    assert row['state'] == 'built' and row['train_bytes_equal'] is True and row['train_bytes_delta'] == 0
    assert row['eval_size_equal'] is True and row['eval_docs'] == 5 and row['eval_docs_equal'] is True
    a = row['anchors']
    assert (a['ids_total'], a['ids_found'], a['ids_found_once']) == (5, 4, 3)
    assert (a['sha256_total'], a['sha256_equal']) == (3, 1)
    assert a['sha256_mismatch_examples'] == ['b'] and a['missing_examples'] == ['missing']
    assert a['length_violations'] == 1 and a['duplicated_examples'] == ['dup']
    assert (a['excerpts_total'], a['excerpts_equal'], a['excerpt_offsets_total'], a['excerpt_offsets_equal']) == (3, 2, 2, 1)
    assert verify.score_bucket(tmp_path, 'web-fr', anchors)['state'] == 'missing'


def test_scorecard_on_a_production_like_pool(tmp_path):
    """Counts equal to the original: train bytes 23/23 and a byte-identical regenerated recipe."""
    write_pool(tmp_path)
    report = verify.scorecard(tmp_path, REPO)
    assert report['totals']['buckets_built'] == 23 and report['totals']['train_bytes_equal'] == 23
    assert report['recipe']['regenerated'] and report['recipe']['byte_identical'] is True
    assert report['recipe']['differences'] == []
    assert report['verdict'].startswith('differs')  # synthetic held-out text cannot contain the stored documents
    assert report['totals']['train_files_read'] == 23 and report['totals']['train_file_size_equal'] == 0
    assert report['original_corpus_manifest']['rebuilt'] is False  # tiny pool files hold less than the quotas
    manifest = tmp_path / 'full-v1' / 'web-hi' / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['counts']['train_bytes'] += 1
    manifest.write_text(json.dumps(data))
    report = verify.scorecard(tmp_path, REPO)
    assert report['totals']['train_bytes_equal'] == 22 and report['buckets']['web-hi']['train_bytes_delta'] == 1
    assert report['recipe']['byte_identical'] is False
    assert report['recipe']['differences'] == ['web-hi.train_available_bytes: 642860651 vs 642860650']
    text = verify.render(report)
    assert 'web-hi' in text and 'regenerated recipe byte-identical' in text


def _exact_report(**changes):
    totals = {'buckets_built': 23, 'test_scale_buckets': 0, 'train_bytes_equal': 23, 'ids_found': 5,
              'ids_total': 5, 'sha256_equal': 2, 'sha256_total': 2, 'excerpts_equal': 1, 'excerpts_total': 1,
              'length_violations': 0, 'eval_docs_equal': 5, 'eval_docs_anchors': 5, 'eval_size_equal': 23,
              'eval_identical': 23, 'eval_content_equal': 23, 'train_content_equal': 23, 'train_file_size_equal': 23}
    totals.update(changes)
    return {'totals': totals, 'recipe': {'byte_identical': True}, 'zlib': {'comparable': True},
            'train_files_checked': True, 'original_corpus_manifest': {'rebuilt': True, 'identical': True}}


def test_verdict_for_incomplete_and_exact():
    base = {'totals': {'buckets_built': 3, 'test_scale_buckets': 0}, 'recipe': {}}
    assert verify.verdict(base).startswith('incomplete')
    assert verify.verdict(_exact_report()).startswith(verify.EXACT_PREFIX)
    # every comparable anchor is needed: one training-file size, one held-out file, one content hash
    for key in ('train_file_size_equal', 'eval_identical', 'train_content_equal', 'eval_content_equal', 'eval_size_equal'):
        text = verify.verdict(_exact_report(**{key: 22}))
        assert text.startswith('differs') and '22/23' in text, key
    report = _exact_report()
    report['original_corpus_manifest'] = {'rebuilt': True, 'identical': False}
    assert 'corpus/manifest.json' in verify.verdict(report)
    # another zlib: size and file-identity anchors are not comparable, content anchors still decide
    report = _exact_report(train_file_size_equal=0, eval_identical=0, eval_size_equal=0)
    report['zlib'] = {'comparable': False}
    text = verify.verdict(report)
    assert text.startswith(verify.EXACT_PREFIX) and 'not comparable' in text
    report['totals']['train_content_equal'] = 20
    assert verify.verdict(report).startswith('differs')
    # --skip-train-files never gives an exact verdict
    report = _exact_report()
    report['train_files_checked'] = False
    assert verify.verdict(report).startswith('not checked completely')


def test_zlib_known_answer_test(monkeypatch):
    kat = verify.zlib_known_answer()
    assert kat['sample_sha256'] == verify.KAT_SAMPLE_SHA256
    if zlib.ZLIB_RUNTIME_VERSION.startswith('1.2.') or zlib.ZLIB_RUNTIME_VERSION.startswith('1.3'):
        assert kat['comparable'] is True, kat  # classic zlib (1.2.12 here and on the cluster's Python)
    monkeypatch.setattr(verify, 'ZLIB_KNOWN_ANSWERS', dict(verify.ZLIB_KNOWN_ANSWERS, level1='0' * 64))
    assert verify.zlib_known_answer()['comparable'] is False


def _original_staged_file(path: Path, pool: Path, quota: int):
    """The original compact copy: gzip text mode, level 1, temporary name train.jsonl.partial (the file name
    lands in the gzip header), the cut rule of scripts/tokenizer_pool.py:112-131 (stop at remaining <= 0)."""
    partial = path.parent / 'train.jsonl.partial'
    used = documents = 0
    with gzip.open(pool, 'rt', encoding='utf-8') as source, \
            gzip.open(partial, 'wt', encoding='utf-8', compresslevel=1) as target:
        for line in source:
            text = json.loads(line)['text']
            remaining = quota - used
            encoded = text.encode('utf-8')
            if len(encoded) > remaining:
                text = encoded[:remaining].decode('utf-8', errors='ignore')
                encoded = text.encode('utf-8')
            if text:
                target.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                documents += 1
            used += len(encoded)
            if used >= quota:
                break
    partial.rename(path)
    return documents, used


def _pool_file(path: Path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as handle:
        for i, text in enumerate(texts):
            handle.write(json.dumps({'id': f'd{i}', 'text': text}, ensure_ascii=False).encode() + b'\n')


@pytest.mark.parametrize('case', ['ascii-cut', 'multibyte-cut'])
def test_joint_file_reproduces_the_original_staged_file_size(tmp_path, case):
    """joint_file gives the exact size of a file written the original way (corpus-v1, readme-v1)."""
    if case == 'ascii-cut':
        texts = ['plain text ' * (40 + i) for i in range(60)]
        quota = sum(len(t.encode()) for t in texts[:30]) + 123   # cut inside an ASCII document
    else:
        # a cut inside a 3-byte character; the next documents start with multi-byte characters, so their
        # 1- and 2-byte cuts are empty (skipped) until a cut of whole characters fits the remaining bytes
        texts = ['中文维基百科' * 30 for _ in range(10)] + ['日本語' * 5, 'é' * 50, '中' * 7, 'ab' * 40, 'tail ' * 30]
        quota = len(texts[0].encode()) * 10 - 1
    pool = tmp_path / 'pool' / 'train.jsonl.gz'
    _pool_file(pool, texts)
    original = tmp_path / 'original' / 'train.jsonl.gz'
    original.parent.mkdir()
    documents, used = _original_staged_file(original, pool, quota)
    result = verify.joint_file(pool, quota)
    assert result['complete'] and result['raw_bytes'] == used == quota
    assert result['documents'] == documents
    assert result['compressed_bytes'] == original.stat().st_size
    assert result['pool_lines'] == len(texts)
    assert result['pool_decompressed_sha256'] == hashlib.sha256(gzip.decompress(pool.read_bytes())).hexdigest()
    if case == 'multibyte-cut':
        assert result['empty_cuts_skipped'] >= 1 and result['documents_after_crossing'] >= 2
    short = verify.joint_file(pool, sum(len(t.encode()) for t in texts) + 10)
    assert short['complete'] is False


def test_rebuild_joint_manifest_and_eval_identity(tmp_path):
    anchors = verify.Anchors()
    anchors.quota = {'a': 10, 'b': 20}
    anchors.train_files = {'a': {'bytes': 5, 'sha256': 'x' * 64}, 'b': {'bytes': 6, 'sha256': 'y' * 64}}
    entries = {'a': {'compressed_bytes': 5, 'documents': 2, 'name': 'a', 'quota_bytes': 10, 'raw_bytes': 10, 'sha256': 'x' * 64},
               'b': {'compressed_bytes': 6, 'documents': 3, 'name': 'b', 'quota_bytes': 20, 'raw_bytes': 20, 'sha256': 'y' * 64}}
    data = (json.dumps(entries, indent=2, sort_keys=True) + '\n').encode()
    anchors.joint_manifest = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    joint = {'a': {'complete': True, 'documents': 2, 'raw_bytes': 10}, 'b': {'complete': True, 'documents': 3, 'raw_bytes': 20}}
    assert verify.rebuild_joint_manifest(joint, anchors)['identical'] is True
    joint['b']['documents'] = 4      # one document more in one bucket breaks the hash
    assert verify.rebuild_joint_manifest(joint, anchors)['identical'] is False
    joint['b']['complete'] = False
    assert verify.rebuild_joint_manifest(joint, anchors)['rebuilt'] is False
    # held-out byte identity: the regenerated file with the original header time is the original file
    path = tmp_path / 'eval.jsonl.gz'
    _eval_file(path, [('a', 'text ' * 40)])
    original = bytearray(path.read_bytes())
    original[4:8] = (1792695906).to_bytes(4, 'little')
    record = {'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
    assert verify.eval_fingerprint(path, 1792695906, record)['identical_with_original_time'] is True
    assert verify.eval_fingerprint(path, 1792695907, record)['identical_with_original_time'] is False


def test_render_states_the_expectation_and_the_stop(tmp_path):
    write_pool(tmp_path)
    report = verify.scorecard(tmp_path, REPO, workers=1)
    assert report['exact'] is False and report['format'] == verify.SCORECARD_FORMAT
    assert set(report['bucket_manifests']) == set(sources.bucket_specs()) and all(report['bucket_manifests'].values())
    text = verify.render(report)
    assert f'expected: {verify.EXACT_PREFIX}' in text and 'STOP before stage-inputs and the pool' in text
    assert '--accept-corpus-scorecard' in text and 'mean the same document sequence' not in text
    assert 'strong evidence, not proof' in text and 'another zlib build' in text
    report['exact'] = True
    assert 'OK: stage-inputs may run' in verify.render(report)


def test_cli_writes_scorecard_and_never_fails(tmp_config):
    write_pool(tmp_config.paths.corpus_root)
    monitor = tmp_config.paths.monitor_dir('corpus')
    monitor.mkdir(parents=True, exist_ok=True)
    (monitor / 'status.json').write_text(json.dumps({'state': 'complete', 'stage': 'complete', 'completed': 23,
                                                     'total': 23, 'error': None, 'buckets': {'web-en': {}}}))
    assert verify.main(['--workers', '2']) == 0
    for directory in (tmp_config.paths.corpus_root, monitor):
        data = json.loads((directory / 'scorecard.json').read_text())
        assert data['totals']['buckets_built'] == 23
        assert (directory / 'scorecard.txt').read_text().startswith('Corpus fidelity scorecard')
    events = [json.loads(line) for line in (monitor / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'scorecard' and events[-1]['exact'] is False and events[-1]['verdict'].startswith('differs')
    status = json.loads((monitor / 'status.json').read_text())
    assert status['state'] == 'complete' and status['completed'] == 23 and status['buckets'] == {'web-en': {}}
    assert status['scorecard']['exact'] is False and status['scorecard']['checks']['train_bytes'] == '23/23'
    empty = tmp_config.base / 'empty'
    assert verify.main(['--corpus-root', str(empty), '--output-dir', str(tmp_config.base / 'out')]) == 0
    assert json.loads((tmp_config.base / 'out' / 'scorecard.json').read_text())['verdict'].startswith('incomplete')


def test_live_miniature_matches_every_stored_document_it_contains(scratch_dir, anchors):
    """The live test-scale build (SCRATCH/mini-corpus): every stored held-out document it contains is identical."""
    root = scratch_dir / 'mini-corpus'
    if not (root / 'full-v1' / 'web-en' / 'manifest.json').is_file():
        pytest.skip(f'no miniature corpus at {root}')
    report = verify.scorecard(root, REPO, anchors)
    totals = report['totals']
    assert totals['buckets_built'] == 23 and totals['test_scale_buckets'] == 23
    assert totals['ids_found'] >= 30 and totals['sha256_equal'] >= 10
    mismatches = {name: row['anchors']['sha256_mismatch_examples'] for name, row in report['buckets'].items()
                  if row['anchors']['sha256_mismatch_examples']}
    assert mismatches == {}
    assert totals['length_violations'] == 0 and totals['ids_found'] == totals['ids_found_once']
