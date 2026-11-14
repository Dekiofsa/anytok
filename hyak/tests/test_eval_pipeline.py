"""prepare_features, validate_prefixes and score (with a fake GPU scorer) on small real tokenizers.

Three tokenizers are trained with the stock tokenizers library in the pool's file
format (BPE/regex, WordPiece/line256, Unigram/no-regex; 6,000 entries each) on the
decoded panel text, installed at pool indices 0-2 of a synthetic 1,000-entry pool
index that the trainer's TokenizerPool reader accepts.

Observed behaviour of the joint trimming loop with them on macOS (only properties are
asserted below, because Unigram training is not bit-reproducible across platforms):
natural panel: round 1 trims 67 rows (a 6,000-entry vocabulary needs more than 95
native tokens), round 2 trims 3 more (token counts are not monotone in the trim),
round 3 is clean -> converged, 1,965 rows unchanged.  Pool panel: rounds 10 -> 1 -> 0
trimmed rows, then two of the trimmed rows were valid in the original panel, so the
port stops exactly where scripts/modal_frozen_policy_cache.py:121-122 would (no panel
is written) unless the user accepted those trims (--accept-valid-context-trims).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hyak.common.io import read_json
from hyak.eval import EvalError, EvalLayout, digest, load_canonical, load_pool_spec, setup_imports
from hyak.tests.test_eval_support import (TOKENIZERS, FakeScorer, asset_dir, asset_tokenizer, fake_checkpoint,
                                          install_model_tokenizer, install_rebuilt, make_layout, train_tokenizer,
                                          write_pool)

INFO = setup_imports()

from hyak.eval import prepare_features as PF  # noqa: E402
from hyak.eval import score as SC  # noqa: E402
from hyak.eval import validate_prefixes as V  # noqa: E402
from hyak.eval.rebuild_panel import rebuild  # noqa: E402


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Rebuilt panels, three trained tokenizers, a fake pool and features, all in one temporary root."""
    tokenizer = asset_tokenizer()
    base = tmp_path_factory.mktemp('evalworld')
    paths = make_layout(base)
    layout = EvalLayout(paths)
    panels = {}
    for panel in ('pool', 'natural'):
        rows, report = rebuild(panel, tokenizer)
        assert report['ok']
        install_rebuilt(layout, panel, rows)
        panels[panel] = rows
    texts = [r['prefix'] for rows in panels.values() for r in rows]
    files = []
    for name, algorithm, pretokenizer, vocab in TOKENIZERS:
        path = base / 'trained' / f'{name}.json'
        train_tokenizer(algorithm, pretokenizer, vocab, texts, path)
        files.append((path, algorithm, pretokenizer))
    write_pool(paths, files)
    install_model_tokenizer(paths, asset_dir())
    spec = load_pool_spec(paths.pool_dir, reference_plan=None)
    counts = PF.prepare(layout, spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0, 1, 2], workers=2,
                        code_dir=INFO['code_dir'], claim_timeout=600, min_remaining=0)
    assert counts['built'] == 3 and counts['done'] == 3
    return {'paths': paths, 'layout': layout, 'spec': spec, 'panels': panels, 'base': base}


def test_features_have_the_original_format(world):
    layout, spec = world['layout'], world['spec']
    canonical, _ = load_canonical(world['paths'].model_dir / 'tokenizer.json')
    fallbacks = PF.expected_fallbacks(canonical)
    assert len(fallbacks) == 21 and min(fallbacks.values()) == 49153
    public = Path(asset_dir()).parent / 'public-zett' / 'hn_config_step500.json'
    if public.is_file():   # the original run's checkpoint config lists the same 21 symbols in the same order
        assert list(fallbacks) == json.loads(public.read_text())['hn_byte_fallback_symbols']
    for index in (0, 1, 2):
        folder = layout.feature_dir(index)
        manifest = read_json(folder / 'manifest.json')
        assert set(PF.MANIFEST_KEYS) <= set(manifest)
        assert manifest['tokenizer_sha256'] == spec['tokenizers'][index]['sha256']
        assert manifest['max_pieces'] == 10 and manifest['fallbacks'] == fallbacks
        forms = np.load(folder / 'forms.npy')
        priors = np.load(folder / 'priors.npy')
        vocab = spec['tokenizers'][index]['vocab_size']
        assert forms.dtype == np.int32 and forms.shape == (vocab, 10)
        assert priors.dtype == np.float32 and priors.shape == (vocab,)
        assert (forms[:, 0] != 49152).all() and forms.max() <= 49152 + 21
        assert PF.feature_problem(folder, spec['tokenizers'][index], fallbacks, deep=True) is None
    assert (np.load(layout.feature_dir(0) / 'priors.npy') == 0).all()      # BPE: no priors
    assert (np.load(layout.feature_dir(2) / 'priors.npy') != 0).any()      # Unigram: its scores


def test_features_are_deterministic_and_tampering_is_detected(world, tmp_path):
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    canonical, _ = load_canonical(paths.model_dir / 'tokenizer.json')
    fallbacks = PF.expected_fallbacks(canonical)
    again = PF.build_features(spec['tokenizers'][1], paths.pool_tokenizer_dir(1) / 'tokenizer.json', canonical,
                              fallbacks, tmp_path / 'again')
    original = read_json(layout.feature_dir(1) / 'manifest.json')
    assert again['forms_sha256'] == original['forms_sha256'] and again['priors_sha256'] == original['priors_sha256']
    copy = tmp_path / 'copy'
    import shutil
    shutil.copytree(layout.feature_dir(1), copy)
    np.save(copy / 'priors.npy', np.ones(spec['tokenizers'][1]['vocab_size'], dtype=np.float32))
    assert PF.feature_problem(copy, spec['tokenizers'][1], fallbacks) is None          # shallow: manifest only
    assert 'priors.npy' in PF.feature_problem(copy, spec['tokenizers'][1], fallbacks, deep=True)
    assert PF.feature_problem(copy, spec['tokenizers'][0], fallbacks) == 'built for another tokenizer file'
    with pytest.raises(EvalError, match='Feature arrays changed'):
        SC.check_features(copy, fallbacks, spec['tokenizers'][1]['sha256'])


def test_prepare_is_idempotent_and_respects_live_claims(world):
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    counts = PF.prepare(layout, spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0, 1, 2], workers=1,
                        code_dir=INFO['code_dir'], claim_timeout=600, min_remaining=0)
    assert counts['built'] == 0 and counts['existing'] == 3


def test_natural_panel_validation_trims_and_reports(world):
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    result = V.validate(layout, 'natural', spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0, 1, 2],
                        workers=2, code_info=INFO)
    report = result['report']
    needing = [r['rows_needing_trim'] for r in report['rounds']]
    assert needing[-1] == 0 and all(n > 0 for n in needing[:-1]) and len(needing) >= 2   # trimmed, then converged
    comparison = report['comparison']
    assert comparison['rows_trimmed'] > 0
    assert comparison['rows_identical_to_stored'] + comparison['rows_trimmed'] == 2032
    first = report['rounds'][0]['rows'][0]
    assert first['start_chars'] > 0 and first['tokenizers'] and set(first['tokenizers']) <= {'0', '1', '2'}
    rows, plan = result['rows'], result['plan']
    assert len(rows) == 2032 and digest(rows) == plan['panel_sha256'] != report['stored_panel_sha256']
    for row in rows:
        assert set(row['native_prefix_lengths']) == {'0', '1', '2'}
        assert max(row['native_prefix_lengths'].values()) <= 95 and 16 <= len(row['canonical_prefix_ids']) <= 64
        assert row['maximum_positions'] <= 128
    trimmed = {t['id'] for t in comparison['trimmed']}
    stored = {r['id']: r for r in world['panels']['natural']}
    for row in rows:
        before = stored[row['id']]
        assert row['canonical_continuation_ids'] == before['canonical_continuation_ids']
        assert row['continuation_bytes'] == before['continuation_bytes']
        assert before['prefix'].endswith(row['prefix'])
        assert (row['prefix'] != before['prefix']) == (row['id'] in trimmed)
        if row['id'] in trimmed:
            assert row['parent_context_sha256'] == before['context_sha256'] and row['pool_trim_chars'] > 0
    assert plan['indices'] == [0, 1, 2] and plan['models'] == []
    assert plan['hyak']['pool_index_sha256'] == spec['index_sha256']
    assert plan['hyak']['trimmed_rows'] == comparison['rows_trimmed']
    assert plan['hyak']['trimmed_row_ids'] == [t['id'] for t in comparison['trimmed']]          # panel order
    assert plan['hyak']['trimmed_chars'] == sum(t['trimmed_chars'] for t in comparison['trimmed']) > 0
    assert plan['hyak']['scripts_sha256'] == INFO['scripts_sha256'] and 'accepted_valid_context_trims' not in plan['hyak']
    assert plan['context_selected_ids'] == read_json(Path(V.repo_root()) / 'plans/natural-model-comparison-v1.json')[
        'context_selected_ids']
    assert V.write_validated(layout, 'natural', result) is True
    assert V.write_validated(layout, 'natural', result) is False            # identical: unchanged
    assert V.up_to_date(layout, 'natural', spec, [0, 1, 2])
    assert not V.up_to_date(layout, 'natural', spec, [0, 1])
    other = {**result, 'plan': {**plan, 'indices': [0, 1]}}
    with pytest.raises(EvalError, match='different validated panel'):
        V.write_validated(layout, 'natural', other)


def test_validation_with_unchanged_tokenizers_is_a_fixed_point(world):
    """Re-validating the validated rows with the same tokenizers needs no trim and gives the same lengths."""
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    rows = read_json(layout.panel('natural'))
    result = V.run_round(rows, [0, 1, 2], spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', workers=2,
                         code_dir=INFO['code_dir'])
    assert not any(result['starts']) and result['failures'] == []
    for index in ('0', '1', '2'):
        assert result['lengths'][index] == [r['native_prefix_lengths'][index] for r in rows]


@pytest.mark.parametrize('panel,indices', [('pool', [0]), ('natural', [0, 1, 2])])
def test_validation_reproduces_a_panel_made_with_the_same_tokenizers(world, tmp_path, monkeypatch, panel, indices):
    """The property that makes the original pool reproduce the stored panels bit for bit.

    First validate with some tokenizers (pool with index 0 alone needs 2 trims of originally
    invalid contexts and passes), then treat that validated panel as the stored one and
    validate again with the same tokenizers: no trim, identical rows, identical panel hash.
    """
    import dataclasses
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    first = V.validate(layout, panel, spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', indices,
                       workers=2, code_info=INFO)
    rows = first['rows']
    second_paths = make_layout(tmp_path)
    second = EvalLayout(second_paths)
    install_rebuilt(second, panel, rows)
    monkeypatch.setitem(V.PANEL_SPECS, panel, dataclasses.replace(V.PANEL_SPECS[panel], panel_sha256=digest(rows)))
    again = V.validate(second, panel, spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', indices, workers=2,
                       code_info=INFO)
    assert [r['rows_needing_trim'] for r in again['report']['rounds']] == [0]
    assert again['rows'] == rows and again['report']['panel_sha256_equals_stored'] is True
    assert again['report']['comparison']['native_lengths_equal_stored']['tokenizers_all_rows_equal'] == len(indices)


def test_pool_panel_stops_when_an_originally_valid_context_would_change(world):
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    with pytest.raises(V.ValidationStop, match='--accept-valid-context-trims') as info:
        V.validate(layout, 'pool', spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0, 1, 2], workers=2,
                   code_info=INFO)
    report = info.value.report
    assert report['rounds'][-1]['rows_needing_trim'] == 0                      # converged; the stop is the protocol
    assert report['problems'] and {p['error'] for p in report['problems']} == {'An already-valid original prefix changed'}
    assert {p['kind'] for p in report['problems']} == {'valid_context_trim'}
    refused = [p['id'] for p in report['problems']]
    trims = report['valid_context_trims']
    assert [t['id'] for t in trims] == refused
    for t in trims:
        assert t['trimmed_chars'] == t['stored_prefix_chars'] - t['prefix_chars'] > 0 and t['tokenizers']
        assert set(t['tokenizers']) <= {'0', '1', '2'}
    assert 'original pool' in report['decision'] and str(layout.validation_report('pool')) in report['decision']
    assert "user's decision" in str(info.value)
    assert not layout.panel('pool').exists() and not layout.plan('pool').exists()
    assert read_json(layout.validation_report('pool'))['ok'] is False
    # Review eval-1: the user's explicit acceptance validates the panel with exactly those trims, recorded.
    result = V.validate(layout, 'pool', spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0, 1, 2],
                        workers=2, code_info=INFO, accept_valid_context_trims=True)
    plan, rows = result['plan'], result['rows']
    assert [t['id'] for t in plan['hyak']['accepted_valid_context_trims']] == refused
    assert result['report']['accepted_valid_context_trims'] == plan['hyak']['accepted_valid_context_trims']
    assert set(refused) <= set(plan['hyak']['trimmed_row_ids'])
    stored = world['panels']['pool']
    assert len(rows) == 576 and [r['id'] for r in rows] == [r['id'] for r in stored]
    for before, after in zip(stored, rows):
        assert after['canonical_continuation_ids'] == before['canonical_continuation_ids']
        assert after['continuation_bytes'] == before['continuation_bytes'] and before['prefix'].endswith(after['prefix'])
        assert max(after['native_prefix_lengths'].values()) <= 95 and after['maximum_positions'] <= 128


def test_unsatisfiable_rows_stop_validation_and_are_listed(world, tmp_path):
    """A tokenizer that cannot round-trip spaces leaves rows without any valid prefix: stop, drop nothing."""
    from tokenizers import normalizers
    paths = make_layout(tmp_path)
    layout = EvalLayout(paths)
    install_rebuilt(layout, 'pool', world['panels']['pool'])
    texts = [r['prefix'] for r in world['panels']['pool']]
    lossy = tmp_path / 'lossy.json'
    train_tokenizer('bpe', 'regex', 3000, texts, lossy, normalizer=normalizers.Replace(' ', '_'))
    write_pool(paths, [(lossy, 'bpe', 'regex')])
    install_model_tokenizer(paths, asset_dir())
    spec = load_pool_spec(paths.pool_dir, reference_plan=None)
    with pytest.raises(V.ValidationStop, match='No shared lossless prefix') as info:
        V.validate(layout, 'pool', spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0], workers=1,
                   code_info=INFO, accept_valid_context_trims=True)       # the acceptance never lifts this stop
    failures = info.value.report['rounds'][0]['unsatisfiable']
    assert failures and all(f['index'] == 0 and f['error'].startswith('No shared lossless prefix') for f in failures)
    assert not layout.panel('pool').exists()
    assert read_json(layout.validation_report('pool'))['stopped'].startswith('No shared lossless prefix')


# --- scoring with a fake scorer ------------------------------------------------------------------------
@pytest.fixture
def scoring(world, tmp_path, monkeypatch):
    """A fresh scores directory for the validated natural panel, plus a fake final checkpoint."""
    layout, spec, paths = world['layout'], world['spec'], world['paths']
    if not V.up_to_date(layout, 'natural', spec, [0, 1, 2]):
        result = V.validate(layout, 'natural', spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', [0, 1, 2],
                            workers=2, code_info=INFO)
        V.write_validated(layout, 'natural', result)
    import shutil
    shutil.rmtree(layout.scores_dir('uniform', 'natural'), ignore_errors=True)
    shutil.rmtree(layout.claims('scores-uniform-natural'), ignore_errors=True)
    checkpoint = fake_checkpoint(tmp_path / 'runs' / 'uniform' / 'step-020000')
    canonical, _ = load_canonical(paths.model_dir / 'tokenizer.json')
    monitor = tmp_path / 'monitor'
    monkeypatch.setenv('HYAK_MONITOR_ROOT', str(monitor))
    from hyak.common.status import StatusWriter
    status = StatusWriter('eval-uniform', None, 'tokenizers', platform='Test · CPU', monitor_dir=monitor / 'eval-uniform')
    return dict(layout=layout, spec=spec, paths=paths, checkpoint=checkpoint, status=status, monitor=monitor,
                fallbacks=PF.expected_fallbacks(canonical))


def make_job(s, **kw):
    return SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, s['checkpoint'],
                       status=s['status'], min_remaining=0, **kw)


def test_score_job_scores_resumes_and_completes(scoring):
    s = scoring
    scorer = FakeScorer(s['fallbacks'])
    job = make_job(s)
    counts = job.run(lambda: scorer)
    assert counts['scored'] == 3 and counts['complete'] is True and scorer.control_calls == 1
    out = s['layout'].scores_dir('uniform', 'natural')
    plan = read_json(out / 'plan.json')
    assert plan['models'][0]['name'] == 'uniform20k-hyak'
    assert plan['models'][0]['manifest']['files']['lm.pt']['bytes'] == len(b'lm weights')
    controls = read_json(out / 'canonical.json')
    assert controls['plan_sha256'] == digest(plan) and np.asarray(controls['canonical']).shape == (2032, 32)
    for index in (0, 1, 2):
        result = read_json(out / f'000{index}.json')
        assert result['plan_sha256'] == digest(plan) and result['index'] == index
        assert result['config'] == s['spec']['tokenizers'][index]['config']
        assert np.asarray(result['transferred']).shape == (2032, 32)
        assert result['tokenizer_sha256'] == s['spec']['tokenizers'][index]['sha256']
    assert read_json(out / 'complete.json')['tokenizers'] == 3
    status = read_json(s['monitor'] / 'eval-uniform' / 'status.json')
    # The eval-uniform stage covers both panels: the natural panel alone does not complete it.
    assert status['state'] == 'running' and status['stage'] == 'natural panel complete; pool panel not complete yet'
    assert status['completed'] == 3 and status['total'] == 3 and status['panel'] == 'natural'
    # A second job finds everything done and never loads a model.
    again = make_job(s).run(lambda: pytest.fail('model must not be loaded'))
    assert again['complete'] is True and again['scored'] == 0
    assert read_json(s['monitor'] / 'eval-uniform' / 'status.json')['state'] == 'running'


def test_stage_status_is_complete_only_when_both_panels_are(scoring):
    """Regression (e2e evaluation run): the shared eval-<run> status said 'complete' after one panel."""
    s = scoring
    pool = s['layout'].scores_dir('uniform', 'pool')
    import shutil
    shutil.rmtree(pool, ignore_errors=True)
    job = make_job(s)
    job.run(lambda: FakeScorer(s['fallbacks']))
    status_path = s['monitor'] / 'eval-uniform' / 'status.json'
    assert read_json(status_path)['state'] == 'running'
    # The pool panel of ANOTHER checkpoint (another training manifest) does not count.
    other = {**job.plan, 'models': [{**job.plan['models'][0], 'manifest_sha256': 'another checkpoint'}]}
    from hyak.common.io import atomic_write_json
    atomic_write_json(pool / 'plan.json', other)
    atomic_write_json(pool / 'complete.json', {'plan_sha256': digest(other)})
    make_job(s).run(lambda: pytest.fail('model must not be loaded'))
    assert read_json(status_path)['state'] == 'running'
    # The pool panel of this checkpoint is complete: now the stage is.
    same = {**job.plan, 'panel_sha256': 'pool-panel-plan'}
    atomic_write_json(pool / 'plan.json', same)
    atomic_write_json(pool / 'complete.json', {'plan_sha256': digest(same)})
    make_job(s).run(lambda: pytest.fail('model must not be loaded'))
    status = read_json(status_path)
    assert status['state'] == 'complete' and status['stage'] == 'complete'
    shutil.rmtree(pool)


def test_eta_counts_other_jobs_not_this_jobs_starting_row(scoring):
    """Regression (e2e evaluation run): the batch script's 'starting' row of the SAME job (another pid) was
    counted as a second worker, so the dashboard ETA of a single scoring job was half the real one."""
    s = scoring
    from hyak.common.io import atomic_write_json
    status = s['status']
    workers = status.workers_dir
    row = status.worker_row(task='scoring the uniform checkpoint', state='Starting')
    atomic_write_json(workers / f'{status.job}__{status.host}__{os.getppid()}.json',
                      {'job_id': status.job, 'host': status.host, 'pid': os.getppid(), 'rows': [row]})
    job = make_job(s)
    job.durations = [10.0]
    job._emit('scoring tokenizers')
    assert read_json(s['monitor'] / 'eval-uniform' / 'status.json')['eta_seconds'] == 30.0   # 3 left x 10 s / 1 job
    other = {**row, 'id': 'other-job', 'job_id': 'other-job'}
    atomic_write_json(workers / f'other-job__{status.host}__{os.getppid()}.json',
                      {'job_id': 'other-job', 'host': status.host, 'pid': os.getppid(), 'rows': [other]})
    job._emit('scoring tokenizers')
    assert read_json(s['monitor'] / 'eval-uniform' / 'status.json')['eta_seconds'] == 15.0   # shared by 2 jobs


def test_score_job_refuses_another_checkpoint_in_the_same_directory(scoring, tmp_path):
    s = scoring
    make_job(s)
    other = fake_checkpoint(tmp_path / 'other' / 'step-020000', lm=b'LM weights')
    with pytest.raises(EvalError, match=r'Comparison plan changed.*checkpoint \(training manifest'):
        SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, other, min_remaining=0)


def test_an_archived_copy_of_the_checkpoint_continues_the_comparison(scoring, tmp_path):
    """Review readme-1: the plan identifies the checkpoint by its manifest, not by its absolute path."""
    import shutil
    s = scoring
    first = make_job(s)
    first.run(lambda: FakeScorer(s['fallbacks']))
    out = s['layout'].scores_dir('uniform', 'natural')
    (out / '0002.json').unlink()
    (out / 'complete.json').unlink()
    archive = tmp_path / 'project' / 'checkpoints' / 'uniform' / 'step-020000'
    shutil.copytree(s['checkpoint'], archive)
    job = SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, archive, min_remaining=0)
    assert job.plan == first.plan and 'checkpoint_path' not in job.plan['models'][0]
    counts = job.run(lambda: FakeScorer(s['fallbacks']))
    assert counts['scored'] == 1 and counts['complete'] is True
    assert read_json(out / '0002.json')['checkpoint_path'] == str(archive)        # kept as information
    assert read_json(out / '0000.json')['checkpoint_path'] == str(s['checkpoint'])


def test_score_job_refuses_a_non_final_checkpoint_without_the_flag(scoring, tmp_path):
    s = scoring
    early = fake_checkpoint(tmp_path / 'runs' / 'uniform' / 'step-002500', step=2500)
    with pytest.raises(EvalError, match='--allow-non-final-checkpoint'):
        SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, early, min_remaining=0,
                    verify_only=True)
    SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, early, min_remaining=0,
                verify_only=True, allow_non_final=True)


def test_score_job_refuses_other_scoring_code(scoring, monkeypatch):
    """Review eval-6: scripts and hyak/eval code hashes are recorded and compared on later jobs."""
    s = scoring
    job = make_job(s)
    assert set(job.plan['scoring_code_sha256']) == {'hyak/eval/__init__.py', 'hyak/eval/score.py'}
    from hyak.eval import SCORING_SCRIPTS
    assert set(job.plan['hyak']['scripts_sha256']) == {f'scripts/{n}' for n in SCORING_SCRIPTS}
    changed = {**job.plan['scoring_code_sha256'], 'hyak/eval/score.py': '0' * 64}
    monkeypatch.setattr(SC, 'scoring_code_hashes', lambda: changed)
    with pytest.raises(EvalError, match='scoring code: hyak/eval/score.py'):
        make_job(s)
    monkeypatch.undo()
    import hyak.eval as E
    current = E.current_code_hashes()
    monkeypatch.setattr(E, 'current_code_hashes', lambda: {
        'hf_hypernet': current['hf_hypernet'],
        'scripts': {**current['scripts'], 'scripts/pool_regime.py': '0' * 64}})
    with pytest.raises(EvalError, match='changed: scripts/pool_regime.py'):
        make_job(s)


def test_verify_code_tree_compares_with_the_pins_not_with_the_repository(tmp_path):
    """Review eval-6: an edited hf_hypernet file is refused even when the repository copy has the same edit."""
    import shutil
    from hyak.eval import verify_code_tree
    source = Path(V.repo_root()) / 'runtime/anytok_pool_20k'
    copy = tmp_path / 'tree'
    shutil.copytree(source / 'hf_hypernet', copy / 'hf_hypernet', ignore=shutil.ignore_patterns('__pycache__'))
    assert set(verify_code_tree(copy)) == {f'hf_hypernet/{n}' for n in
                                           __import__('hyak.code.pinned', fromlist=['x']).HF_HYPERNET_FILES}
    with open(copy / 'hf_hypernet' / 'modeling_hypernet.py', 'a') as stream:
        stream.write('# hot fix\n')
    with pytest.raises(EvalError, match='modeling_hypernet.py is not the pinned file'):
        verify_code_tree(copy)


def test_verify_only_writes_no_plan(scoring, tmp_path):
    """Regression (e2e evaluation run): --verify-only created plan.json and so bound the scores directory to the
    checkpoint it verified; a later job with the real checkpoint was refused ("Comparison plan changed")."""
    s = scoring
    out = s['layout'].scores_dir('uniform', 'natural')
    other = fake_checkpoint(tmp_path / 'other' / 'step-020000', lm=b'LM weights')
    SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, other, min_remaining=0,
                verify_only=True)
    assert not (out / 'plan.json').exists()
    make_job(s)                                  # the real job creates the plan
    assert (out / 'plan.json').is_file()
    with pytest.raises(EvalError, match='Comparison plan changed'):   # and verification still compares with it
        SC.ScoreJob(s['layout'], 'uniform', 'natural', s['spec'], s['paths'].pool_dir, other, min_remaining=0,
                    verify_only=True)


def test_score_job_detects_changed_checkpoint_files(scoring):
    s = scoring
    (s['checkpoint'] / 'lm.pt').write_bytes(b'LM weights')   # same size, different bytes
    with pytest.raises(EvalError, match='Checkpoint changed'):
        make_job(s)


def test_missing_training_manifest_needs_an_explicit_flag(scoring):
    s = scoring
    (s['checkpoint'] / 'manifest.json').unlink()
    with pytest.raises(EvalError, match='--compute-manifest'):
        make_job(s)
    job = make_job(s, compute_manifest=True)
    assert job.plan['models'][0]['manifest']['source'].startswith('computed')


def test_claimed_tokenizers_are_skipped_and_taken_over_when_their_owner_dies(scoring):
    s = scoring
    from hyak.eval import Claims
    root = s['layout'].claims('scores-uniform-natural')
    holder = Claims(root, 3600, job='local-x', host=SC.Claims(root, 1).host, pid=os.getpid())
    assert holder.try_acquire('0001')
    scorer = FakeScorer(s['fallbacks'])
    sleeps = []
    job = make_job(s, claim_timeout=60, poll_seconds=10, sleep=sleeps.append)
    counts = job.run(lambda: scorer)
    # Review eval-3: the job waited (bounded: claim timeout + one poll without a new result), then gave up.
    assert counts['scored'] == 2 and counts['claimed_elsewhere'] == 1 and counts['complete'] is False
    assert counts['incomplete'] == [1] and sum(sleeps) == 70 and counts['stopped'] is None
    assert SC.exit_code(job, counts) == 75
    status = read_json(s['monitor'] / 'eval-uniform' / 'status.json')
    assert status['state'] == 'running' and status['completed'] == 2 and status['stage'].startswith('not complete')
    owner = json.loads((root / '0001' / 'owner.json').read_text())
    owner['pid'] = __import__('hyak.tests.test_eval_claims', fromlist=['dead_pid']).dead_pid()
    (root / '0001' / 'owner.json').write_text(json.dumps(owner))
    counts = make_job(s).run(lambda: FakeScorer(s['fallbacks']))
    assert counts['scored'] == 1 and counts['complete'] is True and len(counts['takeovers']) == 1


def test_a_job_waits_for_claims_of_other_jobs_and_finishes_its_work(scoring):
    """Review eval-3: an index released by its holder during the wait is scored in the same run."""
    s = scoring
    from hyak.eval import Claims
    root = s['layout'].claims('scores-uniform-natural')
    holder = Claims(root, 3600, job='local-z', host=Claims(root, 1).host, pid=os.getpid())
    assert holder.try_acquire('0002')

    def holder_dies_without_a_result(_seconds):
        holder.release('0002')

    job = make_job(s, poll_seconds=5, sleep=holder_dies_without_a_result)
    counts = job.run(lambda: FakeScorer(s['fallbacks']))
    assert counts['scored'] == 3 and counts['complete'] is True and counts['waited_seconds'] == 5
    assert counts['incomplete'] == [] and SC.exit_code(job, counts) == 0


def test_stop_request_and_waiting_for_controls(scoring):
    s = scoring
    stop = SC.Stop()
    stop.requested, stop.reason = True, 'SIGTERM'
    counts = make_job(s, stop=stop).run(lambda: FakeScorer(s['fallbacks']))
    assert counts['scored'] == 0 and counts['stopped'] == 'SIGTERM'
    # Controls are being computed by another (live) job: wait, then use its file.
    out = s['layout'].scores_dir('uniform', 'natural')
    (out / 'canonical.json').unlink()
    from hyak.eval import Claims, create_once
    root = s['layout'].claims('scores-uniform-natural')
    holder = Claims(root, 3600, job='local-y', host=Claims(root, 1).host, pid=os.getpid())
    assert holder.try_acquire('controls')
    job = make_job(s, poll_seconds=1)
    scorer = FakeScorer(s['fallbacks'])

    def other_job_finishes(_seconds):
        canonical, no_prefix = scorer.controls(job.rows)
        create_once(out / 'canonical.json', dict(plan_sha256=job.plan_sha256, canonical=canonical,
                                                 no_prefix=no_prefix), indent=None)
        holder.release('controls')

    job.sleep = other_job_finishes
    counts = job.run(lambda: scorer)
    assert counts['complete'] is True and scorer.control_calls == 1   # computed once, by the "other job"
    # Without the other job, waiting ends with a clear error.
    (out / 'canonical.json').unlink()
    for name in ('0000', '0001', '0002'):
        (out / f'{name}.json').unlink()
    holder.try_acquire('controls')
    job = make_job(s, poll_seconds=1, controls_wait=2)
    job.sleep = lambda seconds: None
    with pytest.raises(EvalError, match='Waited'):
        job.run(lambda: FakeScorer(s['fallbacks']))
    # Regression (e2e evaluation run): a stop signal while waiting is a graceful stop, not an error (exit 2 made
    # 09_eval_score.sbatch record a failed job).
    stop = SC.Stop()
    job = make_job(s, poll_seconds=1, stop=stop)

    def signal_arrives(_seconds):
        stop.requested, stop.reason = True, 'SIGTERM'

    job.sleep = signal_arrives
    counts = job.run(lambda: FakeScorer(s['fallbacks']))
    assert counts['stopped'] == 'SIGTERM' and counts['scored'] == 0 and counts['complete'] is False
    assert not (out / 'canonical.json').exists()
    holder.release('controls')


def test_native_prefix_and_feature_checks(world):
    from uniform_trajectory import load_candidate
    rows = read_json(world['layout'].panel('natural'))
    tokenizer, _, _, _, forbidden = load_candidate(world['paths'].pool_tokenizer_dir(0) / 'tokenizer.json', 'pool')
    SC.check_native_prefixes(rows, tokenizer, forbidden, 0)
    bad = [dict(r) for r in rows]
    bad[3] = {**bad[3], 'native_prefix_lengths': {**bad[3]['native_prefix_lengths'], '0': 1}}
    with pytest.raises(EvalError, match='Native prefix changed'):
        SC.check_native_prefixes(bad, tokenizer, forbidden, 0)
    canonical, _ = load_canonical(world['paths'].model_dir / 'tokenizer.json')
    fallbacks = PF.expected_fallbacks(canonical)
    sha = world['spec']['tokenizers'][0]['sha256']
    SC.check_features(world['layout'].feature_dir(0), fallbacks, sha)
    with pytest.raises(EvalError, match='mapping changed'):
        SC.check_features(world['layout'].feature_dir(0), {**fallbacks, 'x': 1}, sha)


def test_torch_scorer_refuses_to_run_without_cuda(tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip('CUDA present')
    with pytest.raises(EvalError, match='CUDA is not available'):
        SC.TorchScorer(tmp_path, tmp_path)


def test_command_lines_with_a_config(world, make_tmp_config, monkeypatch, tmp_path):
    """prepare_features / validate_prefixes main() on a configured root (pool loader without the repo-plan check)."""
    cfg = make_tmp_config()
    paths = cfg.paths
    layout = EvalLayout(paths)
    import shutil
    shutil.copytree(world['paths'].pool_dir, paths.pool_dir)
    install_model_tokenizer(paths, asset_dir())
    install_rebuilt(layout, 'natural', world['panels']['natural'])
    loader = lambda pool_dir, **kw: load_pool_spec(pool_dir, reference_plan=None)  # noqa: E731
    monkeypatch.setattr(PF, 'load_pool_spec', loader)
    monkeypatch.setattr(V, 'load_pool_spec', loader)
    assert PF.main(['--indices', '0-2', '--workers', '2']) == 0
    status = read_json(Path(os.environ['HYAK_MONITOR_ROOT']) / 'eval-prepare' / 'status.json')
    assert status['completed'] == 3 and status['state'] == 'running' and 'features 3/1000' in status['stage']
    assert V.main(['--panel', 'natural', '--indices', '0-2', '--workers', '2']) == 0
    assert V.up_to_date(layout, 'natural', load_pool_spec(paths.pool_dir, reference_plan=None), [0, 1, 2])
    assert V.main(['--panel', 'natural', '--indices', '0-2']) == 0           # up to date: no work
    assert V.main(['--panel', 'natural', '--indices', '0-1']) == 2           # different index list: refused
    # Review eval-1/eval-2: the pool panel stops (exit 3, report names the option); the natural panel is independent.
    install_rebuilt(layout, 'pool', world['panels']['pool'])
    assert V.main(['--panel', 'pool', '--indices', '0-2', '--workers', '2']) == 3
    report = read_json(layout.validation_report('pool'))
    assert report['valid_context_trims'] and '--accept-valid-context-trims' in report['decision']
    assert not layout.plan('pool').exists()
    status = read_json(Path(os.environ['HYAK_MONITOR_ROOT']) / 'eval-prepare' / 'status.json')
    assert status['state'] == 'failed' and status['completed'] > 0          # a stop keeps the progress count
    assert V.main(['--panel', 'natural', '--indices', '0-2', '--recheck', '--workers', '2']) == 0
    assert V.main(['--panel', 'natural', '--indices', '0-2', '--accept-valid-context-trims']) == 0  # ignored
    assert V.main(['--panel', 'pool', '--indices', '0-2', '--workers', '2', '--accept-valid-context-trims']) == 0
    plan = read_json(layout.plan('pool'))
    assert [t['id'] for t in plan['hyak']['accepted_valid_context_trims']] == \
        [t['id'] for t in report['valid_context_trims']]
    assert V.main(['--panel', 'pool', '--indices', '0-2']) == 0              # up to date (acceptance recorded)
    with pytest.raises(SystemExit) as exit_info:
        SC.main(['--help'])
    assert exit_info.value.code == 0


@pytest.mark.slow
def test_torch_scorer_cpu_rehearsal(world, tmp_path):
    """The real TorchScorer and score_rows on a tiny Llama with SmolLM2's tokenizer, on the CPU.

    A test-only shim maps 'cuda' to the CPU in a separate process.  This runs the
    exact loading sequence (untie, strict lm.pt load, both-table equality check,
    hypernetwork reload with byte fallbacks), the controls, the full-vocabulary
    hypernetwork forward on the prepared features and score_rows with its 32-target
    and 1e-5 Hugging Face loss checks.  It says nothing about GPU numerics or speed.
    """
    import subprocess
    import sys
    layout, paths, spec = world['layout'], world['paths'], world['spec']
    if not V.up_to_date(layout, 'natural', spec, [0, 1, 2]):
        V.write_validated(layout, 'natural', V.validate(layout, 'natural', spec, paths.pool_dir,
                                                         paths.model_dir / 'tokenizer.json', [0, 1, 2], workers=2,
                                                         code_info=INFO))
    out = tmp_path / 'out.json'
    args = {'work': str(tmp_path / 'tiny'), 'tokenizer': str(asset_tokenizer()), 'panel': str(layout.panel('natural')),
            'rows': 32, 'out': str(out),
            'tokenizers': {str(i): [str(layout.feature_dir(i)), str(paths.pool_tokenizer_dir(i) / 'tokenizer.json')]
                           for i in (0, 1, 2)},
            'sha256': {str(i): spec['tokenizers'][i]['sha256'] for i in (0, 1, 2)}}
    env = {**os.environ, 'PYTHONPATH': str(Path(V.repo_root())), 'PYTHONDONTWRITEBYTECODE': '1'}
    code = 'import sys; from hyak.tests.test_eval_support import rehearsal_main; sys.exit(rehearsal_main())'
    result = subprocess.run([sys.executable, '-c', code, json.dumps(args)], env=env, capture_output=True, text=True,
                            timeout=600, cwd=str(Path(V.repo_root())))
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(out.read_text())
    canonical, _ = load_canonical(paths.model_dir / 'tokenizer.json')
    assert data['fallbacks'] == PF.expected_fallbacks(canonical)
    for name in ('canonical', 'no_prefix'):
        array = np.asarray(data[name])
        assert array.shape == (32, 32) and np.isfinite(array).all() and (array >= 0).all()
    for index in ('0', '1', '2'):
        array = np.asarray(data['transferred'][index])
        assert array.shape == (32, 32) and np.isfinite(array).all() and (array >= 0).all()
    # Scoring-protocol properties (review tests-1): each fails for a protocol mutation of score.py that the
    # shape checks above let through.
    assert data['lm_differences'] == [], 'lm.pt is not what the scorer scores with'
    assert data['controls_differ'] is True, 'the BOS-only control equals the canonical control'
    for index in ('0', '1', '2'):
        assert data['transferred'][index] == data['reference'][index], \
            f'tokenizer {index}: transferred scores differ from the deterministic reference recomputation'
        assert data['transferred'][index] != data['canonical'], 'the native path was not used'


# --- the scorer's per-tokenizer guards are really called (review tests-1) -------------------------------
def test_score_job_checks_native_prefixes(scoring):
    s = scoring
    job = make_job(s)
    # Changed in memory after load_panel (editing panel.json would be refused earlier as 'Corrected panel changed').
    job.rows[3] = {**job.rows[3], 'native_prefix_lengths': {**job.rows[3]['native_prefix_lengths'], '0': 1}}
    with pytest.raises(EvalError, match='Native prefix changed: tokenizer 0'):
        job.run(lambda: FakeScorer(s['fallbacks']))


def test_score_job_checks_the_feature_mapping(scoring):
    s = scoring
    with pytest.raises(EvalError, match='Hypernetwork feature mapping changed'):
        make_job(s).run(lambda: FakeScorer({**s['fallbacks'], 'x': 1}))


def test_score_job_checks_the_tokenizer_file(scoring):
    s = scoring
    job = make_job(s)
    entries = [dict(entry) for entry in job.spec['tokenizers']]
    entries[0]['sha256'] = '0' * 64
    job.spec = {**job.spec, 'tokenizers': entries}
    with pytest.raises(EvalError, match='Native tokenizer changed'):
        job.run(lambda: FakeScorer(s['fallbacks']))


def test_valid_context_stop_names_the_job_variable_of_the_prepare_job():
    """The opt-in flag reaches validate_prefixes through 08_eval_prepare.sbatch's HYAK_EVAL_PREPARE_ARGS."""
    text = V.decision_text([{'trimmed_chars': 3, 'tokenizers': [7]}], '/x/validation-report.json')
    assert 'HYAK_EVAL_PREPARE_ARGS' in text and V.ACCEPT_FLAG in text
    assert 'job_extra_args HYAK_EVAL_PREPARE_ARGS' in (Path(__file__).resolve().parents[1]
                                                      / 'slurm/08_eval_prepare.sbatch').read_text()


def test_e2e_shim_marks_the_injected_checkpoint_as_non_final(monkeypatch):
    """The e2e harness scores a miniature (non-final) checkpoint; score refuses it without the explicit flag."""
    import importlib.util
    shim_path = Path(__file__).resolve().parent / 'e2e' / 'shim' / 'hyak_e2e_cpu_shim.py'
    spec = importlib.util.spec_from_file_location('hyak_e2e_cpu_shim_under_test', shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)                 # definitions only; install() is not called
    argv = ['--run', 'uniform', '--panel', 'natural']
    monkeypatch.delenv('HYAK_E2E_EVAL_CHECKPOINT_UNIFORM', raising=False)
    assert shim.eval_argv('hyak.eval.score', argv) == argv
    monkeypatch.setenv('HYAK_E2E_EVAL_CHECKPOINT_UNIFORM', '/w/runs/uniform/step-000003')
    assert shim.eval_argv('hyak.eval.score', argv) == argv + ['--checkpoint', '/w/runs/uniform/step-000003',
                                                              '--allow-non-final-checkpoint']
    explicit = argv + ['--checkpoint', '/elsewhere']
    assert shim.eval_argv('hyak.eval.score', explicit) == explicit          # the caller's choice is kept
    assert shim.eval_argv('hyak.eval.pack', argv) == argv
