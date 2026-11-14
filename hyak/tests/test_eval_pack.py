"""hyak.eval.pack: the original metrics, the reward-table format and frozen_policy_search compatibility.

* The published complete-pool numbers (plans/frozen-policy-search-v1.md:77-80) and the
  natural-panel numbers (plans/natural-model-comparison-v1.md:57-74) are recomputed
  from the stored bundles with the packer's summary functions.
* Packing the stored losses gives a table whose JSON is byte-identical to the stored
  reward table, and that table passes scripts/frozen_policy_search.py's own loading,
  validation and policy set-up (not the 20,000-update search).
* The file-based path (per-tokenizer result files -> reward table / natural bundle)
  reproduces the stored per-passage sums exactly, using the real per-token RL20k
  losses of reports/natural-transfer-20k for the natural panel.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from hyak.common.io import atomic_write_json, read_json
from hyak.common.paths import pool_index_name
from hyak.eval import PANEL_SPECS, EvalError, EvalLayout, current_code_hashes, digest, read_bundle, setup_imports
from hyak.tests.test_eval_support import REPO, make_layout

setup_imports()

import frozen_policy_search as F  # noqa: E402
from hyak.eval import pack as P  # noqa: E402
from hyak.eval.validate_prefixes import derived_plan  # noqa: E402

REWARD = REPO / PANEL_SPECS['pool'].source_rel
HF = current_code_hashes()['hf_hypernet']


@pytest.fixture(scope='module')
def stored_cache():
    return json.loads(gzip.decompress(REWARD.read_bytes()))


@pytest.fixture(scope='module')
def stored_rebuilt_pool():
    """Stored pool rows with the decoded prefix (needed for panel.json); requires the SmolLM2 tokenizer."""
    from hyak.eval.rebuild_panel import rebuild
    from hyak.tests.test_eval_support import asset_tokenizer
    rows, report = rebuild('pool', asset_tokenizer())
    assert report['ok']
    return rows


def test_published_complete_pool_numbers(stored_cache):
    summary = P.pool_summary(stored_cache)
    expected = {'screen': (0.96692, 0.06690, 635, 0.10396), 'confirm': (0.96534, 0.08883, 635, 0.13784)}
    analysis = read_json(REPO / 'reports/frozen-policy-search-20k/analysis.json')['splits']
    for split, (canonical, gap, worst, worst_gap) in expected.items():
        s = summary[split]
        assert round(s['canonical_bpb'], 5) == canonical
        assert round(s['uniform_mean_gap_bpb'], 5) == gap
        assert s['best_index'] == worst and round(s['best_gap_bpb'], 5) == worst_gap
        # Exactly the values frozen_policy_search wrote into the committed analysis.
        assert s['canonical_bpb'] == analysis[split]['canonical_bpb']
        assert s['uniform_bpb'] == analysis[split]['uniform_bpb']
        assert s['best_bpb'] == analysis[split]['best_bpb']
        ours = [(r['index'], r['rank'], r['bpb'], r['gap_bpb']) for r in s['ranking']]
        theirs = [(r['index'], r['rank'], r['bpb'], r['gap_bpb']) for r in analysis[split]['ranking']]
        assert ours == theirs


def test_published_natural_panel_numbers():
    reference = P.original_natural_reference()
    assert reference['relation'] == 'different_runs' and reference['reproduction_target'] is False
    assert 'never a reproduction target' in reference['note']
    weak = {s['model']: s for s in reference['summaries']
            if s['regime'] == 'context' and s['split'] == 'confirm' and s['cohort'] == 'weak_pretokenization'}
    assert {m: round(s['median_gap_bpb'], 5) for m, s in weak.items()} == \
        {'zett20k': 0.41835, 'uniform20k': 0.14240, 'rl20k': 0.13437}
    paired = next(c for c in reference['paired_comparisons'] if c['comparison'] == 'rl20k-minus-uniform20k'
                  and c['regime'] == 'context' and c['split'] == 'confirm' and c['cohort'] == 'weak_pretokenization')
    assert round(paired['mean_gap_change_bpb'], 5) == -0.00809
    assert [round(x, 5) for x in paired['mean_gap_change_ci95']] == [-0.01236, -0.00320]
    assert round(paired['mean_absolute_bpb_change'], 5) == -0.00764
    # Identical to the committed analysis of scripts/analyze_natural_model_comparison.py: point estimates
    # exactly; bootstrap interval ends within 1e-12 (they go through a BLAS matrix product, whose last bit
    # depends on the BLAS build).
    committed = read_json(REPO / 'reports/natural-model-comparison-20k/analysis.json')
    keyed = {(s['model'], s['regime'], s['split'], s['cohort']): s for s in committed['summaries']}
    for s in reference['summaries']:
        theirs = keyed[(s['model'], s['regime'], s['split'], s['cohort'])]
        assert {k: v for k, v in s.items() if k != 'median_gap_ci95'} == \
            {k: v for k, v in theirs.items() if k != 'median_gap_ci95'}
        assert np.allclose(s['median_gap_ci95'], theirs['median_gap_ci95'], rtol=0, atol=1e-12)


def _scores_from_cache(cache):
    base = np.zeros((576, 32))
    base[:, 0] = cache['canonical_nats']
    losses = np.zeros((1000, 576, 32))
    losses[:, :, 0] = np.asarray(cache['transferred_nats'])
    return dict(plan=cache['plan'], indices=list(range(1000)), canonical=base, transferred=losses,
                configs=cache['configs'], tokenizer_sha256=cache['tokenizer_hashes'])


def test_reward_table_round_trip_is_byte_identical_json(stored_cache):
    cache = P.reward_table(_scores_from_cache(stored_cache), stored_cache['rows'])
    packed = P.reward_table_bytes(cache)
    # gzip output depends on the zlib build; the serialised JSON is identical byte for byte.
    assert gzip.decompress(packed) == gzip.decompress(REWARD.read_bytes())


def test_packed_table_passes_frozen_policy_search_loading(stored_cache, tmp_path):
    path = tmp_path / 'reward_table.json.gz'
    path.write_bytes(P.reward_table_bytes(P.reward_table(_scores_from_cache(stored_cache), stored_cache['rows'])))
    # The tool's own load line for a file argument (scripts/frozen_policy_search.py:219).
    cache = json.loads(gzip.decompress(path.read_bytes()))
    refs = F.reference(cache)
    assert refs['screen']['bpb'].shape == (1000,) and refs['confirm']['bpb'].shape == (1000,)
    assert F.file_sha(REPO / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py') == \
        cache['plan']['source_policy_sha256']
    settings = cache['plan']['policy']
    entropy0 = read_json(REPO / 'plans/frozen-policy-entropy0-v1.json')
    assert F.variant_settings(cache, entropy0) == {**settings, 'entropy_weight': 0.0}
    Policy = F.policy_class()
    policy = Policy(cache['configs'], **{k: settings[k] for k in ['seed', 'learning_rate', 'entropy_weight',
                                                                    'baseline_decay']})
    record = F.summarize(policy, refs, cache['configs'], np.zeros(1000, dtype=int))
    assert record['step'] == 0 and record['unique_tokenizers'] == 0
    action, _ = F.learner_step(policy, refs['screen']['bpb'])
    assert 0 <= action < 1000 and policy.updates == 1


def _write_pool_scores(layout, run, plan_rows, panel_plan, cache, *, shift=0.0, missing=()):
    """Per-tokenizer result files in score.py's format whose per-passage sums equal ``cache``."""
    out = layout.scores_dir(run, 'pool')
    plan = {**panel_plan, 'models': [{'name': P.MODEL_NAMES[run], 'run': run, 'checkpoint_path': '/x',
                                      'manifest': {'files': {}}}]}
    atomic_write_json(out / 'plan.json', plan)
    sha = digest(plan)
    pad = [0.0] * 31
    atomic_write_json(out / 'canonical.json', dict(plan_sha256=sha, canonical=[[v] + pad for v in cache['canonical_nats']],
                                                   no_prefix=[[v + 1.0] + pad for v in cache['canonical_nats']]),
                      indent=None)
    for i in range(1000):
        if i in missing:
            continue
        values = [[v + shift] + pad for v in cache['transferred_nats'][i]]
        atomic_write_json(out / f'{pool_index_name(i)}.json', dict(
            model=P.MODEL_NAMES[run], index=i, config=cache['configs'][i], transferred=values, plan_sha256=sha,
            elapsed_seconds=1.0, tokenizer_sha256=cache['tokenizer_hashes'][i], gpu='Fake GPU'), indent=None)
    return out


def _install_prepared(layout, panel, rows, plan):
    atomic_write_json(layout.panel(panel), rows)
    atomic_write_json(layout.plan(panel), plan)
    atomic_write_json(layout.preparation(panel), dict(plan=plan, panel_sha256=plan['panel_sha256'],
                      metadata=[{k: v for k, v in r.items() if k != 'prefix'} for r in rows]))


@pytest.mark.slow
def test_pack_pool_from_result_files(stored_cache, stored_rebuilt_pool, tmp_path):
    """Score files -> reward table: the stored per-passage sums come back exactly; uniform-vs-RL comparison."""
    layout = EvalLayout(make_layout(tmp_path))
    spec_file = read_json(REPO / 'reports/rl-pool-sweep-20k/spec.json')
    spec = {'index_sha256': 'synthetic', 'plan_sha256': 'synthetic',
            'tokenizers': [{'index': t['index'], 'sha256': t['sha256'], 'vocab_size': t['vocab_size'],
                            'config': t['config']} for t in spec_file['tokenizers']]}
    original_plan = read_json(REPO / PANEL_SPECS['pool'].plan_rel)
    panel_plan = derived_plan('pool', original_plan, stored_rebuilt_pool, list(range(1000)), spec,
                              stored_hash=PANEL_SPECS['pool'].panel_sha256, canonical_sha='x', code_hashes=HF,
                              rounds=1, trimmed=0)
    assert panel_plan['panel_sha256'] == PANEL_SPECS['pool'].panel_sha256  # unchanged rows -> unchanged hash
    _install_prepared(layout, 'pool', stored_rebuilt_pool, panel_plan)
    _write_pool_scores(layout, 'rl', stored_rebuilt_pool, panel_plan, stored_cache)
    _write_pool_scores(layout, 'uniform', stored_rebuilt_pool, panel_plan, stored_cache, shift=0.5)
    out = tmp_path / 'reports'
    assert P.pack_pool(layout, ['uniform', 'rl'], out) == {'uniform': 'complete', 'rl': 'complete'}
    table = json.loads(gzip.decompress((out / 'rl' / 'reward_table.json.gz').read_bytes()))
    assert set(table) == {'plan', 'rows', 'canonical_nats', 'transferred_nats', 'configs', 'tokenizer_hashes'}
    for key in ('rows', 'canonical_nats', 'transferred_nats', 'configs', 'tokenizer_hashes'):
        assert table[key] == stored_cache[key], key
    assert table['plan']['models'][0]['name'] == 'rl20k-straight-entropy0-hyak' and table['plan']['indices'] == list(range(1000))
    refs = F.reference(table)  # the policy-search tool accepts the packed file
    assert float(refs['screen']['canonical_bpb']) == F.reference(stored_cache)['screen']['canonical_bpb']
    summary = read_json(out / 'rl' / 'summary.json')
    assert round(summary['splits']['screen']['uniform_mean_gap_bpb'], 5) == 0.06690
    assert summary['splits']['screen']['bos_only_bpb'] > summary['splits']['screen']['canonical_bpb']
    comparison = read_json(out / 'comparison.json')
    screen = comparison['uniform_vs_rl']['screen']
    assert screen['tokenizers'] == 1000 and screen['mean_gap_change_rl_minus_uniform'] < 0
    assert comparison['reference_original']['splits']['confirm']['best_index'] == 635
    # D8: the stored original rl20k is a reference from a different run, never a reproduction target
    reference = comparison['reference_original']
    assert reference['relation'] == 'different_run' and reference['reproduction_target'] is False
    assert reference['compared_with'] == P.MODEL_NAMES['rl'] == 'rl20k-straight-entropy0-hyak'
    assert 'never a reproduction target' in reference['note'] and 'entropy 0.01' in reference['note']
    assert 'warmup initialization' in reference['note'] and 'original example cuts' in reference['note']
    assert (out / 'rl' / 'rankings.csv').read_text().count('\n') == 1001
    # Review eval-7: an untrimmed panel says so in every packed summary.
    for block in (summary['panel'], comparison['panel']):
        assert block['trimmed_rows'] == 0 and block['trimmed_row_ids'] == [] and block['equals_stored'] is True
        assert block['accepted_valid_context_trims'] == []


def test_pack_pool_refuses_incomplete_scores_but_summarises_partial(stored_cache, stored_rebuilt_pool, tmp_path):
    layout = EvalLayout(make_layout(tmp_path))
    spec = {'index_sha256': 's', 'plan_sha256': 's',
            'tokenizers': [{'index': i, 'sha256': h, 'vocab_size': c['vocab_size'], 'config': c}
                           for i, (h, c) in enumerate(zip(stored_cache['tokenizer_hashes'], stored_cache['configs']))]}
    panel_plan = derived_plan('pool', read_json(REPO / PANEL_SPECS['pool'].plan_rel), stored_rebuilt_pool,
                              list(range(1000)), spec, stored_hash='s', canonical_sha='x', code_hashes=HF,
                              rounds=1, trimmed=0)
    _install_prepared(layout, 'pool', stored_rebuilt_pool, panel_plan)
    _write_pool_scores(layout, 'rl', stored_rebuilt_pool, panel_plan, stored_cache, missing=set(range(3, 1000)))
    assert P.pack_pool(layout, ['rl', 'uniform'], tmp_path / 'out') == {
        'rl': 'incomplete (3/1000 tokenizers)', 'uniform': 'not scored'}
    assert not (tmp_path / 'out' / 'rl').exists()
    assert read_json(tmp_path / 'out' / 'comparison.json')['runs']['rl'].startswith('incomplete')
    with pytest.raises(EvalError, match='997 of 1000 tokenizers have no result'):
        P.load_run_scores(layout.scores_dir('rl', 'pool'), stored_rebuilt_pool, panel_plan)
    P.pack_pool(layout, ['rl'], tmp_path / 'out', partial=True)
    summary = read_json(tmp_path / 'out' / 'rl' / 'summary.json')
    assert summary['partial'] is True and summary['splits']['screen']['tokenizers'] == 3
    assert not (tmp_path / 'out' / 'rl' / 'reward_table.json.gz').exists()
    # A result stamped with another plan is refused (modal_random_pool_50.py:172-173).
    target = layout.scores_dir('rl', 'pool') / '0001.json'
    result = read_json(target)
    atomic_write_json(target, {**result, 'plan_sha256': 'other'})
    with pytest.raises(EvalError, match='another comparison'):
        P.pack_pool(layout, ['rl'], tmp_path / 'out', partial=True)


def test_pack_natural_with_real_per_token_scores(tmp_path, make_tmp_config):
    """The stored RL20k per-token losses packed by hyak.eval.pack equal the stored per-passage sums."""
    from hyak.tests.test_eval_support import asset_tokenizer
    from hyak.eval.rebuild_panel import rebuild
    rows, report = rebuild('natural', asset_tokenizer())
    assert report['ok']
    layout = EvalLayout(make_layout(tmp_path))
    transfer = read_bundle(REPO / PANEL_SPECS['natural'].source_rel)
    comparison = read_bundle(REPO / PANEL_SPECS['natural'].twin_rel)
    results = sorted(transfer['results'], key=lambda r: r['index'])
    configs = {r['index']: r['config'] for r in results}
    spec = {'index_sha256': 's', 'plan_sha256': 's',
            'tokenizers': [{'index': i, 'sha256': next((r['tokenizer_sha256'] for r in results if r['index'] == i), '0'),
                            'vocab_size': 0, 'config': configs.get(i, {})} for i in range(1000)]}
    indices = [r['index'] for r in results]
    panel_plan = derived_plan('natural', read_json(REPO / PANEL_SPECS['natural'].plan_rel), rows, indices, spec,
                              stored_hash=PANEL_SPECS['natural'].panel_sha256, canonical_sha='x', code_hashes=HF,
                              rounds=1, trimmed=0)
    _install_prepared(layout, 'natural', rows, panel_plan)
    out = layout.scores_dir('rl', 'natural')
    plan = {**panel_plan, 'models': [{'name': 'rl20k-straight-entropy0-hyak', 'run': 'rl', 'checkpoint_path': '/x', 'manifest': {}}]}
    atomic_write_json(out / 'plan.json', plan)
    sha = digest(plan)
    atomic_write_json(out / 'canonical.json', dict(plan_sha256=sha, canonical=transfer['canonical'],
                                                   no_prefix=transfer['no_prefix']), indent=None)
    for r in results:
        atomic_write_json(out / f'{pool_index_name(r["index"])}.json', dict(
            model='rl20k-straight-entropy0-hyak', index=r['index'], config=r['config'], transferred=r['transferred'], plan_sha256=sha,
            elapsed_seconds=r['elapsed_seconds'], tokenizer_sha256=r['tokenizer_sha256'], gpu=r['gpu']), indent=None)
    assert P.pack_natural(layout, ['rl', 'uniform'], tmp_path / 'reports') == {'rl': 'complete',
                                                                               'uniform': 'not scored'}
    data = read_bundle(tmp_path / 'reports' / 'results.json.gz')
    ours, theirs = data['models']['rl20k-straight-entropy0-hyak'], comparison['models']['rl20k']
    assert ours['canonical_nats'] == theirs['canonical_nats']
    assert ours['transferred_nats'] == theirs['transferred_nats']
    assert data['metadata'] == comparison['metadata'] and data['tokenizers'] == comparison['tokenizers']
    summary = read_json(tmp_path / 'reports' / 'summary.json')
    original = {(s['regime'], s['split'], s['cohort']): s for s in P.original_natural_reference()['summaries']
                if s['model'] == 'rl20k'}
    for s in summary['summaries']:
        key = (s['regime'], s['split'], s['cohort'])
        if key in original:
            assert {k: v for k, v in s.items() if k != 'model'} == {k: v for k, v in original[key].items() if k != 'model'}
    # Command line on a configured root (ANYTOK_ROOT = <tmp>/root, the layout used above).
    cfg = make_tmp_config()
    assert cfg.paths.root == layout.paths.root
    assert P.main(['--panel', 'natural', '--runs', 'rl', '--out-dir', str(tmp_path / 'cli')]) == 0
    assert summary['panel']['equals_stored'] is True and summary['panel']['trimmed_rows'] == 0
    assert summary['partial'] is False
    assert P.main(['--panel', 'natural', '--out-dir', str(tmp_path / 'cli')]) == 75       # uniform not scored
    assert P.main(['--panel', 'pool', '--out-dir', str(tmp_path / 'cli')]) == 2           # pool panel not validated
    # Regression (e2e evaluation run): no complete run yet is "not complete" (exit 75), not a failure (exit 2);
    # 09_eval_score.sbatch fails the job on exit 2, e.g. a job that stopped claiming before its time limit
    # or the earlier of two --count 2 jobs while the other still scores the last natural tokenizer.
    assert P.main(['--panel', 'natural', '--runs', 'uniform', '--out-dir', str(tmp_path / 'none')]) == 75
    assert not (tmp_path / 'none' / 'results.json.gz').exists()
    # Review eval-6: a panel validated with other scoring-script code is refused.
    other = {**panel_plan, 'hyak': {**panel_plan['hyak'],
                                    'scripts_sha256': {**panel_plan['hyak']['scripts_sha256'],
                                                       'scripts/natural_transfer.py': '0' * 64}}}
    _install_prepared(layout, 'natural', rows, other)
    with pytest.raises(EvalError, match='changed: scripts/natural_transfer.py'):
        P.load_prepared(layout, 'natural')
    assert P.main(['--panel', 'natural', '--runs', 'rl', '--out-dir', str(tmp_path / 'cli')]) == 2
    # Review eval-7: a trimmed panel is reported in the summary (ids, rows, characters, accepted trims).
    trimmed = derived_plan('natural', read_json(REPO / PANEL_SPECS['natural'].plan_rel), rows, indices, spec,
                           stored_hash='stored', canonical_sha='x', code_hashes=HF, rounds=2, trimmed=2,
                           trimmed_row_ids=[rows[5]['id'], rows[9]['id']], trimmed_chars=17)
    block = P.panel_block(trimmed)
    assert block['trimmed_rows'] == 2 and block['trimmed_row_ids'] == [rows[5]['id'], rows[9]['id']]
    assert block['trimmed_chars'] == 17 and block['equals_stored'] is False


def test_pack_natural_refuses_a_panel_validated_for_a_subset(tmp_path):
    """Review eval-8: a natural panel validated by hand for a few indices is not packed as if complete."""
    from hyak.tests.test_eval_support import asset_tokenizer
    from hyak.eval.rebuild_panel import rebuild
    rows, report = rebuild('natural', asset_tokenizer())
    layout = EvalLayout(make_layout(tmp_path))
    spec = {'index_sha256': 's', 'plan_sha256': 's',
            'tokenizers': [{'index': i, 'sha256': '0', 'vocab_size': 0, 'config': {}} for i in range(1000)]}
    plan = derived_plan('natural', read_json(REPO / PANEL_SPECS['natural'].plan_rel), rows, [3, 4, 11], spec,
                        stored_hash=PANEL_SPECS['natural'].panel_sha256, canonical_sha='x', code_hashes=HF,
                        rounds=1, trimmed=0)
    _install_prepared(layout, 'natural', rows, plan)
    with pytest.raises(EvalError, match='validated for 3 of the 38 plan indices'):
        P.pack_natural(layout, ['rl'], tmp_path / 'out')
    # With --partial it packs and marks the summary partial with the missing plan indices.
    transfer = read_bundle(REPO / PANEL_SPECS['natural'].source_rel)
    out = layout.scores_dir('rl', 'natural')
    scoring = {**plan, 'models': [{'name': 'rl20k-straight-entropy0-hyak', 'run': 'rl', 'manifest': {}}]}
    atomic_write_json(out / 'plan.json', scoring)
    sha = digest(scoring)
    atomic_write_json(out / 'canonical.json', dict(plan_sha256=sha, canonical=transfer['canonical'],
                                                   no_prefix=transfer['no_prefix']), indent=None)
    for r in transfer['results']:
        if r['index'] in (3, 4, 11):
            atomic_write_json(out / f'{pool_index_name(r["index"])}.json', dict(
                model='rl20k-straight-entropy0-hyak', index=r['index'], config=r['config'], transferred=r['transferred'],
                plan_sha256=sha, elapsed_seconds=1.0, tokenizer_sha256='0', gpu='g'), indent=None)
    assert P.pack_natural(layout, ['rl'], tmp_path / 'out', partial=True) == {'rl': 'complete'}
    summary = read_json(tmp_path / 'out' / 'summary.json')
    assert summary['partial'] is True and len(summary['missing_indices']) == 35 and 3 not in summary['missing_indices']


def test_pack_refuses_the_repository_reports_directory():
    with pytest.raises(EvalError, match='inside the repository reports'):
        P._refuse_repo_reports(REPO / 'reports' / 'frozen-policy-search-20k')


def test_an_unexpected_error_is_a_crash_with_a_traceback(make_tmp_config, monkeypatch, capsys):
    """Exit-code contract: 1 only for crashes, with the full traceback on stderr (2 stays usage/input errors)."""
    make_tmp_config()

    def boom(*args, **kwargs):
        raise ZeroDivisionError('bug')

    monkeypatch.setattr(P, 'pack_pool', boom)
    assert P.main(['--panel', 'pool']) == 1
    err = capsys.readouterr().err
    assert 'Traceback' in err and 'ZeroDivisionError: bug' in err and 'crashed' in err
