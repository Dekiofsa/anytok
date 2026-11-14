"""hyak.eval.rebuild_panel: both stored panels decode exactly and every hash verifies.

This is the first execution of the decode-based reconstruction (gaps.md 4.1 item 10).
Measured here (tokenizers 0.22.2, pinned SmolLM2 tokenizer.json): pool 576/576 rows,
panel sha256 30ee5e0d...; natural 2,032/2,032 rows, panel sha256 5c8ad1c6...
"""
from __future__ import annotations

import json

import pytest

from hyak.eval import PANEL_SPECS, EvalError, digest, setup_imports
from hyak.tests.test_eval_support import REPO, asset_tokenizer

setup_imports()

from hyak.eval import rebuild_panel as R  # noqa: E402


@pytest.fixture(scope='module')
def tokenizer_file():
    return asset_tokenizer()


@pytest.fixture(scope='module')
def rebuilt(tokenizer_file):
    return {panel: R.rebuild(panel, tokenizer_file) for panel in ('pool', 'natural')}


@pytest.mark.parametrize('panel,rows,expected', [
    ('pool', 576, '30ee5e0d25b2c77fd3d743d8a1a0635670572bf086c3adeef22bf192834e7fc9'),
    ('natural', 2032, '5c8ad1c63adf655cd9e2610500cf5ec58e9cc9190af3e16142fd7868ccec2f64'),
])
def test_every_row_and_the_whole_panel_verify(rebuilt, panel, rows, expected):
    panel_rows, report = rebuilt[panel]
    assert report['ok'] is True
    assert report['rows'] == rows and report['verified_rows'] == rows and report['failed_rows'] == []
    assert report['panel_problems'] == []
    assert report['panel_sha256'] == expected == PANEL_SPECS[panel].panel_sha256
    assert digest(panel_rows) == expected
    assert all(isinstance(r['prefix'], str) and r['prefix'] for r in panel_rows)


def test_pool_panel_structure(rebuilt):
    rows, report = rebuilt['pool']
    assert report['splits'] == {'screen': 288, 'confirm': 288}
    assert set(report['domains'].values()) == {64}
    stored = json.loads(__import__('gzip').decompress((REPO / PANEL_SPECS['pool'].source_rel).read_bytes()))
    assert [{k: v for k, v in r.items() if k != 'prefix'} for r in rows] == stored['rows']


def test_decoded_prefixes_equal_the_published_raw_text(rebuilt):
    """22 of the 23 published passages keep their context in the corrected panel; 10/10 natural examples."""
    rows, _ = rebuilt['pool']
    published = json.loads((REPO / 'reports/rl-pool-passages-20k/passages.json').read_text())['examples']
    by_original = {(r['domain'], r['original_id']): r for r in rows}
    same_context, equal = 0, 0
    for example in published:
        row = by_original[(example['domain'], example['id'])]
        if digest([example['prefix'], row['canonical_continuation_ids']]) == row['context_sha256']:
            same_context += 1
            equal += example['prefix'] == row['prefix']
    assert (same_context, equal) == (22, 22)
    natural, _ = rebuilt['natural']
    by_id = {r['id']: r for r in natural}
    examples = json.loads((REPO / 'reports/natural-transfer-20k/examples.json').read_text())
    assert len(examples) == 10
    for example in examples:
        assert by_id[example['id']]['prefix'] == example['prefix']


def test_natural_continuations_equal_published_text(rebuilt, tokenizer_file):
    from uniform_trajectory import CanonicalTokenizer
    canonical = CanonicalTokenizer(tokenizer_file)
    natural, _ = rebuilt['natural']
    by_id = {r['id']: r for r in natural}
    for example in json.loads((REPO / 'reports/natural-transfer-20k/examples.json').read_text()):
        row = by_id[example['id']]
        assert canonical.decode(row['canonical_continuation_ids']) == example['continuation']
        assert len(example['continuation'].encode()) == row['continuation_bytes']


def test_auto_tokenizer_from_the_model_directory_decodes_identically(rebuilt, tokenizer_file):
    """The GPU scorers load the canonical tokenizer with AutoTokenizer (config.json + tokenizer.json only)."""
    from transformers import AutoTokenizer
    auto = AutoTokenizer.from_pretrained(tokenizer_file.parent, local_files_only=True)
    assert len(auto) == 49152
    for panel in ('pool', 'natural'):
        rows, _ = rebuilt[panel]
        assert all(auto.decode(r['canonical_prefix_ids'], clean_up_tokenization_spaces=False) == r['prefix']
                   for r in rows)
        assert all(auto(r['prefix'], add_special_tokens=False).input_ids == r['canonical_prefix_ids'] for r in rows)


def test_a_corrupted_row_fails_loudly(tokenizer_file, monkeypatch, tmp_path):
    stored, expected, plan, sources = R.load_stored_rows('pool')
    bad = [dict(r) for r in stored]
    bad[5] = {**bad[5], 'canonical_prefix_ids': bad[5]['canonical_prefix_ids'][1:]}
    monkeypatch.setattr(R, 'load_stored_rows', lambda panel, repo=None: (bad, expected, plan, sources))
    rows, report = R.rebuild('pool', tokenizer_file)
    assert report['ok'] is False and report['verified_rows'] == 575
    assert report['failed_rows'][0]['id'] == bad[5]['id']
    assert 'context_sha256 mismatch' in report['failed_rows'][0]['problems']
    assert report['panel_sha256_matches'] is False
    with pytest.raises(EvalError, match='did not verify'):
        R.write_outputs('pool', rows, report, tmp_path)
    assert not (tmp_path / 'pool' / 'rebuilt.json').exists()
    assert json.loads((tmp_path / 'pool' / 'rebuild-report.json').read_text())['ok'] is False


def test_an_unpinned_tokenizer_is_refused(tokenizer_file, tmp_path):
    copy = tmp_path / 'tokenizer.json'
    copy.write_bytes(tokenizer_file.read_bytes() + b'\n')
    with pytest.raises(EvalError, match='not the pinned SmolLM2'):
        R.rebuild('pool', copy)


def test_cli_writes_once_and_refuses_different_content(tokenizer_file, tmp_path, capsys):
    out = tmp_path / 'panels'
    assert R.main(['--panel', 'natural', '--tokenizer', str(tokenizer_file), '--out-dir', str(out)]) == 0
    target = out / 'natural' / 'rebuilt.json'
    assert digest(json.loads(target.read_text())) == PANEL_SPECS['natural'].panel_sha256
    mtime = target.stat().st_mtime_ns
    assert R.main(['--panel', 'natural', '--tokenizer', str(tokenizer_file), '--out-dir', str(out)]) == 0
    assert target.stat().st_mtime_ns == mtime
    rows = json.loads(target.read_text())
    rows[0]['prefix'] += ' '
    target.write_text(json.dumps(rows))
    assert R.main(['--panel', 'natural', '--tokenizer', str(tokenizer_file), '--out-dir', str(out)]) == 2
    assert 'different content' in capsys.readouterr().err
    assert R.main(['--panel', 'pool', '--tokenizer', str(tokenizer_file), '--check-only']) == 0


def test_cli_exit_4_when_a_stored_panel_does_not_rebuild(tokenizer_file, tmp_path, monkeypatch):
    """Exit-code contract: a failed check is 4 (1 is reserved for crashes)."""
    real = R.rebuild

    def broken(panel, tokenizer):
        rows, report = real(panel, tokenizer)
        return rows, {**report, 'ok': False}

    monkeypatch.setattr(R, 'rebuild', broken)
    assert R.main(['--panel', 'pool', '--tokenizer', str(tokenizer_file), '--check-only']) == 4
    assert R.main(['--panel', 'pool', '--tokenizer', str(tokenizer_file), '--out-dir', str(tmp_path / 'p')]) == 4
    assert not (tmp_path / 'p' / 'pool' / 'rebuilt.json').exists()
