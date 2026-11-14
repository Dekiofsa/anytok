"""hyak.pool.scorecard: anchor loading, scoring arithmetic, and the three levels on real SmolLM2 prefixes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import hyak.pool as pool
from hyak.pool import scorecard
from hyak.pool.scorecard import load_reward_rows, load_spec_hashes, load_truncated_forms, score, summary_text
from hyak.tests.test_pool_support import make_test_plan, tokenizer_bytes

REPO = Path(__file__).resolve().parents[2]
#: Bytes 0x00-0x0b in the byte-level alphabet; not merged by SmolLM2 (control characters).
CONTROL12 = ''.join(chr(0x100 + i) for i in range(12))


def test_anchors_of_the_original_pool():
    plan, _ = pool.load_plan(REPO / pool.PLAN_REL)
    hashes = load_spec_hashes(REPO, plan)
    assert len(hashes) == 1000
    assert hashes[0] == 'a032cfaf90502af30fbfa0f87018f90318f736ce5c66871b3da05e361ed81293'
    assert hashes[86] == 'e656a3bd6279ee3b66bbd76a265e58fe4aedab4df0753bffccd533e7b24a8e23'
    truncated = load_truncated_forms(REPO)
    assert sorted(truncated) == list(range(1000))                     # every index occurs in uniform.csv
    assert (truncated[433], truncated[858], truncated[85]) == (588, 332, 51)   # NOTES/gaps.md N1
    rows = load_reward_rows(REPO)
    assert len(rows) == 576 and all(len(r['native_prefix_lengths']) == 1000 for r in rows)
    table = json.loads(__import__('gzip').decompress((REPO / pool.REWARD_TABLE_REL).read_bytes()))
    assert table['tokenizer_hashes'] == hashes


def test_score_arithmetic():
    plan = {'configs': [{'index': i, 'algorithm': a, 'pretokenizer': 'regex', 'vocab_size': 32000}
                        for i, a in enumerate(['bpe', 'wordpiece', 'unigram', 'bpe'])]}
    rows = [{'id': f'r{j}', 'native_prefix_lengths': {str(i): 10 + j for i in range(4)}} for j in range(3)]
    measurements = [
        {'index': 0, 'sha256': 'x', 'truncated_forms': 5, 'native_prefix_lengths': [10, 11, 12]},
        {'index': 1, 'sha256': 'B', 'truncated_forms': 7, 'native_prefix_lengths': [10, 13, 12]},
        {'index': 2, 'missing': True},
        {'index': 3, 'sha256': 'D', 'truncated_forms': 1, 'native_prefix_lengths': [9, 11, 12]},
    ]
    report = score(plan, measurements, spec_hashes=['A', 'B', 'C', 'D'], truncated={0: 5, 1: 8, 3: 1},
                   reward_rows=rows)
    by_index = {r['index']: r for r in report['tokenizers']}
    assert by_index[0]['level1']['match'] is False and by_index[1]['level1']['match'] is True
    assert by_index[0]['level2']['match'] and not by_index[1]['level2']['match']
    assert by_index[0]['level3']['all_match'] and by_index[1]['level3']['rows_match'] == 2
    assert by_index[1]['level3']['max_abs_difference'] == 2 and by_index[1]['level3']['mismatched_rows'] == ['r1']
    assert by_index[3]['level3']['mean_difference'] == round(-1 / 3, 4)
    assert by_index[2]['missing']
    totals = report['totals']
    assert totals['present'] == 3 and totals['missing'] == 1
    assert totals['level1_match'] == 2 and totals['level2_match'] == 2 and totals['level2_checked'] == 3
    assert totals['level3_rows'] == 9 and totals['level3_rows_match'] == 7 and totals['level3_all_rows_match'] == 1
    assert report['families']['bpe/regex']['tokenizers'] == 2
    report.update(created_at='now', pool_dir='/p', planned=4)
    text = summary_text(report)
    assert 'bpe/regex' in text and 'ALL' in text


def _byte_pool(tmp_path, extra_tokens=()):
    plan_path = tmp_path / 'plan.json'
    vocab = {'bpe': 257, 'wordpiece': 258, 'unigram': 258}
    plan = make_test_plan(plan_path, count=3, vocab=lambda c: vocab[c['algorithm']] + len(extra_tokens))
    pool_dir = tmp_path / 'pool'
    for config in plan['configs']:
        folder = pool_dir / 'tokenizers' / f'{config["index"]:04d}'
        folder.mkdir(parents=True)
        (folder / 'tokenizer.json').write_bytes(
            tokenizer_bytes(config['algorithm'], config['pretokenizer'], config['vocab_size'], extra_tokens))
    return plan_path, plan, pool_dir


def test_levels_2_and_3_on_real_prefixes_with_byte_tokenizers(tmp_path, scratch_assets):
    """Byte-symbol vocabularies: one native token per UTF-8 byte, and exactly one >10-piece entry."""
    model_dir = scratch_assets / 'full'
    plan_path, plan, pool_dir = _byte_pool(tmp_path, extra_tokens=(CONTROL12,))
    from hyak.eval import setup_imports
    from hyak.eval.rebuild_panel import rebuild
    setup_imports(REPO / 'runtime/anytok_pool_20k', REPO)
    rows, report = rebuild('pool', model_dir / 'tokenizer.json', repo=REPO)
    assert report['ok'] and report['verified_rows'] == 576
    prefixes = [r['prefix'] for r in rows]
    scorecard._init_worker({'tokenizers_dir': str(pool_dir / 'tokenizers'), 'level2': True,
                            'level3_prefixes': prefixes, 'model_dir': str(model_dir),
                            'code_dir': str(REPO / 'runtime/anytok_pool_20k'), 'repo': str(REPO)})
    measurements = [scorecard.measure(i) for i in range(3)]
    for item in measurements:
        assert item['native_prefix_lengths'] == [len(p.encode('utf-8')) for p in prefixes]
        assert item['truncated_forms'] == 1
    expected_rows = [{'id': r['id'], 'native_prefix_lengths': {str(i): len(r['prefix'].encode()) for i in range(3)}}
                     for r in rows]
    expected_rows[5]['native_prefix_lengths']['1'] += 1
    result = score(plan, measurements, truncated={0: 1, 1: 1, 2: 0}, reward_rows=expected_rows)
    by_index = {r['index']: r for r in result['tokenizers']}
    assert by_index[0]['level3']['all_match'] and not by_index[1]['level3']['all_match']
    assert by_index[1]['level3']['mismatched_rows'] == [rows[5]['id']]
    assert [by_index[i]['level2']['match'] for i in range(3)] == [True, True, False]


@pytest.mark.slow
def test_command_line_with_worker_processes(tmp_path, scratch_assets, tmp_config):
    plan_path, plan, pool_dir = _byte_pool(tmp_path)
    out = tmp_path / 'report' / 'scorecard.json'
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.scorecard', '--plan', str(plan_path), '--allow-test-plan',
                             '--pool-dir', str(pool_dir), '--model-dir', str(scratch_assets / 'full'),
                             '--out', str(out), '--workers', '2'],
                            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report['totals']['present'] == 3 and report['levels'] == [1, 2, 3]
    assert report['panel']['panel_sha256_matches'] and report['panel']['verified_rows'] == 576
    for row in report['tokenizers']:
        assert row['level1']['match'] is False and row['level3']['rows'] == 576
    assert out.with_suffix('.txt').read_text().startswith('Pool scorecard')


def test_missing_canonical_tokenizer_is_an_input_error(tmp_path):
    plan_path, plan, pool_dir = _byte_pool(tmp_path)
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.scorecard', '--plan', str(plan_path), '--allow-test-plan',
                             '--pool-dir', str(pool_dir), '--model-dir', str(tmp_path / 'nothing'),
                             '--out', str(tmp_path / 'o.json'), '--levels', '2', '--workers', '1'],
                            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 2 and 'tokenizer.json' in result.stderr


def test_level_1_only_needs_no_model(tmp_path):
    plan_path, plan, pool_dir = _byte_pool(tmp_path)
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.scorecard', '--plan', str(plan_path), '--allow-test-plan',
                             '--pool-dir', str(pool_dir), '--model-dir', str(tmp_path / 'nothing'),
                             '--out', str(tmp_path / 'o.json'), '--levels', '1', '--workers', '1'],
                            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'o.json').read_text())
    sha = hashlib.sha256((pool_dir / 'tokenizers' / '0000' / 'tokenizer.json').read_bytes()).hexdigest()
    assert report['tokenizers'][0]['level1']['sha256'] == sha


# --- review fixes (pool-10, readme-2) ---------------------------------------------------------------------------
def test_empty_pool_with_the_pilot_and_binding_preset(tmp_path):
    """pool-10 (a): no tokenizer yet -> exit 0 and a text summary (was KeyError 'level1_match')."""
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    out = tmp_path / 'o.json'
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.scorecard', '--indices', 'pilots+binding', '--levels', '1',
                             '--pool-dir', str(tmp_path / 'empty'), '--model-dir', str(tmp_path / 'nothing'),
                             '--out', str(out), '--workers', '1'],
                            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report['indices'] == 19 and report['totals']['missing'] == 19
    assert report['binding']['present'] == 0
    text = out.with_suffix('.txt').read_text()
    assert 'tokenizers present 0 / 1000' in text and 'binding tokenizers 64 100 632 687 728 821 914' in text


def test_protected_rows_over_the_cap_and_binding_summary():
    plan = {'configs': [{'index': i, 'algorithm': 'unigram', 'pretokenizer': 'regex', 'vocab_size': 32000}
                        for i in range(1000)]}
    rows = [{'id': 'p', 'original_joint_encoding_matches': True, 'native_prefix_lengths': {'64': 95, '100': 95}},
            {'id': 'q', 'original_joint_encoding_matches': False, 'native_prefix_lengths': {'64': 95, '100': 95}}]
    hashes = ['h%d' % i for i in range(1000)]
    measurements = [{'index': 64, 'sha256': 'h64', 'native_prefix_lengths': [95, 99]},     # identical; q may exceed
                    {'index': 100, 'sha256': 'other', 'native_prefix_lengths': [96, 95]}]  # p over the cap: stop
    report = score(plan, measurements, spec_hashes=hashes, reward_rows=rows)
    by_index = {r['index']: r for r in report['tokenizers']}
    assert by_index[64]['level3']['protected_rows_over_cap'] == 0 and by_index[64]['level3']['over_native_cap_95'] == 1
    assert by_index[100]['level3']['protected_rows_over_cap'] == 1 and by_index[100]['level3']['protected_rows_over_cap_ids'] == ['p']
    assert report['totals']['level3_protected_over_cap'] == 1 and report['totals']['level3_tokenizers_over_cap'] == 1
    binding = report['binding']
    assert (binding['present'], binding['level1_identical'], binding['not_identical']) == (2, 1, [100])
    assert binding['protected_rows_over_cap'] == 1
    report.update(created_at='now', pool_dir='/p', planned=1000)
    text = summary_text(report)
    assert 'identical to the original (level 1): 1 of 2' in text and 'validate_prefixes --panel pool, exit 3' in text
    assert 'BPE families' in text


def test_one_bad_tokenizer_is_reported_and_the_others_are_scored(tmp_path, monkeypatch):
    """pool-10 (b): an exception in level 2/3 of one index is recorded, not fatal."""
    import types
    plan_path, plan, pool_dir = _byte_pool(tmp_path)
    package = types.ModuleType('hf_hypernet')
    inputs = types.ModuleType('hf_hypernet.tokenizer_inputs')

    calls = []

    def build_surface_forms(pieces, source, max_pieces, fallback, specials):
        calls.append(len(pieces))
        if len(calls) == 2:                     # the second tokenizer has an unrepresentable piece
            raise ValueError('piece cannot be represented')
        return None, 3

    inputs.build_surface_forms = build_surface_forms
    monkeypatch.setitem(sys.modules, 'hf_hypernet', package)
    monkeypatch.setitem(sys.modules, 'hf_hypernet.tokenizer_inputs', inputs)
    monkeypatch.setattr(scorecard, '_STATE', {'tokenizers_dir': str(pool_dir / 'tokenizers'), 'level2': True,
                                              'level3_prefixes': ['hello world'], 'source': None, 'fallback': {}})
    measured = [scorecard.measure(config['index']) for config in plan['configs']]
    errors = [m for m in measured if 'level2_error' in m]
    assert [m['index'] for m in errors] == [1] and 'cannot be represented' in errors[0]['level2_error']
    assert all('native_prefix_lengths' in m for m in measured)
    ok = [m for m in measured if 'truncated_forms' in m]
    assert ok and all(m['truncated_forms'] == 3 for m in ok), measured
    report = score(plan, measured, truncated={c['index']: 3 for c in plan['configs']})
    assert sorted(e['index'] for e in report['errors']) == sorted(m['index'] for m in errors)
    report.update(created_at='now', pool_dir='/p', planned=3)
    assert summary_text(report).splitlines()[2].startswith('ERRORS: level 2/3 could not be computed')


def test_skip_if_complete(tmp_path):
    plan = {'plan_sha256': 'P', 'configs': [{}] * 3}
    out = tmp_path / 'pool-scorecard.json'
    report = {'levels': [1, 2, 3], 'indices': 3, 'planned': 3, 'totals': {'present': 3}, 'plan_sha256': 'P', 'errors': []}
    out.write_text(json.dumps(report))
    assert scorecard.existing_report_is_complete(out, plan, tmp_path)
    index = tmp_path / 'full-results.json'
    index.write_text('{}')
    old = out.stat().st_mtime - 10
    os.utime(out, (old, old))
    assert not scorecard.existing_report_is_complete(out, plan, tmp_path)       # older than the index
    out.write_text(json.dumps(dict(report, levels=[1])))
    assert not scorecard.existing_report_is_complete(out, plan, tmp_path)       # a pilot (level 1) report
    out.write_text(json.dumps(dict(report, totals={'present': 2})))
    assert not scorecard.existing_report_is_complete(out, plan, tmp_path)
