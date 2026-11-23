"""Raw-BPB full-pool ranking: both rankings, the report's additions and its refusals, on synthetic scores."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from package_training_inputs import sha256
import prepare_rawbpb_pool_ranking as prepare
import report_rawbpb_pool_ranking as report
from report_rawbpb_pool_ranking import (SELECTED, analyze, analyze_raw_and_reference, check_identity,
                                        check_provenance, load_reference, panel_paragraphs, private_files, readme)
from score_rawbpb_pool_ranking import pair_batch

# The spurious Accelerate matmul flags that main() turns off (see the report script's docstring).
pytestmark = pytest.mark.filterwarnings('ignore:.*encountered in matmul:RuntimeWarning')


def config(index):
    """Tokenizers 0-217 form the step-6,500 bottom group: 115 regex BPE, 102 regex WordPiece, regex Unigram #217."""
    if index < 218:
        return dict(index=index, algorithm='bpe' if index < 115 else 'wordpiece' if index < 217 else 'unigram',
                    pretokenizer='regex')
    return dict(index=index, algorithm=['unigram', 'wordpiece', 'bpe'][index % 3], pretokenizer='no-regex')


def tables():
    """Gap i/1000 for tokenizer i (+.01 on odd batches); canonical BPB 1, or 1.5 for even i."""
    index, batch = np.meshgrid(np.arange(1000), np.arange(32), indexing='ij')
    return index / 1000 + (batch % 2) * .01, 1 + .5 * (index % 2 == 0)


def manifest(step=20000, panel='fresh', **changes):
    """A fresh-panel manifest in which wiki-ja and wiki-zh are in their second pass (91 of 4,096 items)."""
    epochs = {'web-en': 0, 'wiki-ja': 1, 'wiki-zh': 1}
    return dict(dict(checkpoint_step=step, panel_sha256=panel, aggregation='mean of 32 per-update BPB',
                     panel_mode='fresh', examples=4096, domains={'web-en': 4005, 'wiki-ja': 50, 'wiki-zh': 41},
                     panel_source='fresh: synthetic', corpus_stream=dict(
                         epochs_at_checkpoint=epochs, epochs_after_items=epochs, epochs_after_banks=epochs),
                     cross_check=dict(from_step=18000, updates_replayed=2000, logged_hashes_matched=2000,
                                      rejected_examples=7, identical_panel=True)), **changes)


def records(gaps, canonical, favorite=0, step=20000, panel='fresh', counts=None):
    """Score records as the scorer writes them, a policy with all mass on favorite, and the panel manifest."""
    probabilities = np.zeros(1000); probabilities[favorite] = 1
    if counts is None:
        counts = np.zeros(1000, dtype=int); counts[favorite] = step
    rows = []
    for index in range(1000):
        batches = [dict(index=batch, split='screen' if batch < 16 else 'confirm', valid_examples=128,
                        maximum_positions=128, scored_bytes=100 if batch % 2 == 0 else 10000,
                        dropout_seed=2026111500 + batch, raw_bpb=float(canonical[index, batch] + gaps[index, batch]),
                        canonical_bpb=float(canonical[index, batch]), gap_bpb=float(gaps[index, batch]),
                        rejected_examples=0) for batch in range(32)]
        rows.append(dict(index=index, checkpoint_step=step, panel_sha256=panel, config=config(index),
                         tokenizer_sha256=hashlib.sha256(b'tokenizer %d' % index).hexdigest(), vocabulary=32000,
                         policy_probability=float(probabilities[index]), batches=batches))
    policy = dict(step=step, probabilities=probabilities.tolist(), counts=[int(count) for count in counts])
    return rows, policy, manifest(step, panel)


def pool(rows):
    return dict(tokenizers=[dict(config=row['config'], tokenizer_sha256=row['tokenizer_sha256']) for row in rows])


def write_reference(folder, rows, policy, index, panel=None):
    """A --reference folder; a panel manifest gets the pool index hash unless it names one."""
    folder.mkdir()
    (folder / 'policy.json').write_text(json.dumps(policy))
    (folder / 'pool-index.json').write_text(json.dumps(index))
    with gzip.open(folder / 'all-scores.json.gz', 'wt', compresslevel=1) as stream:
        json.dump(rows, stream)
    if panel is not None:
        panel = {'pool_index_sha256': sha256(folder / 'pool-index.json'), **panel}
        (folder / 'panel-manifest.json').write_text(json.dumps(panel))
    return folder


@pytest.fixture(scope='module')
def reference_folder(tmp_path_factory):
    """The step-6,500 ranking: gap i/1000, so tokenizers 0-217 hold its ranks 783-1,000; favorite #771."""
    rows, policy, panel = records(*tables(), favorite=771, step=6500, panel='published')
    return write_reference(tmp_path_factory.mktemp('reference') / 'step-6500', rows, policy, pool(rows), panel)


@pytest.fixture(scope='module')
def step6500(reference_folder):
    return load_reference(reference_folder, pool(records(*tables())[0]))


def ranking(reference, gaps, canonical, favorite=0, counts=None):
    """analyze() and analyze_raw_and_reference() of step-20,000 records, as main() runs them."""
    rows, policy, panel = records(gaps, canonical, favorite, counts=counts)
    result, table, scores = analyze(rows, policy, panel, 20000)
    return result, analyze_raw_and_reference(rows, policy, result, scores, reference), table, panel


def test_gap_ranking_matches_mean_update_reward_not_pooled_continuation_bytes():
    result, table, scores = analyze(*records(*tables()), 20000)
    gap = result['comparisons']['all']
    assert (gap['dominant_index'], gap['dominant_rank'], gap['hardest_index']) == (0, 1000, 999)
    assert gap['policy_expected_gap'] == pytest.approx(.005)  # pooled by continuation bytes it would be ~.0099
    assert gap['uniform_mean_gap'] == pytest.approx(.5045)
    confirmation = result['confirmation']
    assert confirmation['discovery_winner'] == 999 and confirmation['mean_gap_difference'] == pytest.approx(.999)
    assert confirmation['paired_batch_bootstrap_95pct'] == pytest.approx([.999, .999])
    assert confirmation['interpretation'] == 'Discovery winner has higher regret on confirmation batches.'
    assert [row['index'] for row in table[:2]] == [999, 998] and scores.shape == (1000, 32)
    assert (table[-1]['index'], table[-1]['canonical_bpb']) == (0, 1.5)
    assert table[-1]['raw_bpb'] == pytest.approx(1.505) and table[-1]['gap_bpb'] == pytest.approx(.005)


def test_raw_bpb_ranking_and_policy_comparison_are_kept_apart_from_the_gap_ranking(step6500):
    result, extra, table, panel = ranking(step6500, *tables())
    raw = extra['comparisons']['all']
    # Even tokenizers carry +.5 canonical BPB: #0 is the easiest by gap but has 749 harder by raw BPB.
    assert (raw['dominant_index'], raw['dominant_rank'], raw['hardest_index']) == (0, 750, 998)
    assert extra['policy_favorite']['gap_rank'] == 1000 and extra['policy_favorite']['raw_bpb_rank'] == 750
    assert (extra['peaks']['gap']['index'], extra['peaks']['gap']['raw_bpb_rank']) == (999, 251)
    assert (extra['peaks']['raw_bpb']['index'], extra['peaks']['raw_bpb']['gap_rank']) == (998, 2)
    assert extra['expected']['gap'] == pytest.approx(dict(policy=.005, uniform=.5045, maximum=1.004,
                                                          policy_minus_uniform=-.4995, maximum_minus_policy=.999))
    assert extra['expected']['raw_bpb'] == pytest.approx(dict(policy=1.505, uniform=1.7545, maximum=2.503,
                                                              policy_minus_uniform=-.2495, maximum_minus_policy=.998))
    confirmation = extra['confirmation']
    assert (confirmation['discovery_winner'], confirmation['dominant_index']) == (998, 0)
    assert confirmation['mean_raw_bpb_difference'] == pytest.approx(.998)
    assert confirmation['paired_batch_bootstrap_95pct'] == pytest.approx([.998, .998])
    assert confirmation['interpretation'] == 'Discovery winner has higher raw BPB on confirmation batches.'
    assert result['rank_stability_spearman'] == pytest.approx(1) and extra['rank_stability_spearman'] == pytest.approx(1)
    assert 0 < extra['gap_raw_bpb_spearman'] < 1


def test_readme_reports_both_rankings_and_requires_the_verdict_of_analyze(step6500):
    result, extra, table, panel = ranking(step6500, *tables())
    text = readme(result, extra, table, panel, 20000, 'rawbpb_pool_ranking_20000.png')
    for line in ['The saved policy favorite is **#0**, with **100.00%** probability and estimated gap rank **1000/1,000**.',
                 'Ranked by mean raw BPB, the policy favorite **#0** is **750/1,000** at **1.505000 BPB**.',
                 'The raw-BPB peak is **#998** (bpe / no-regex / 32,000): raw BPB **2.503000**, gap **+1.003000** (gap rank 2).',
                 'The gap peak **#999** (unigram / no-regex / 32,000) has raw BPB **2.004000** (raw-BPB rank 251).',
                 'on a fresh panel, the next 32 training batches the run would have drawn after step 20,000 (see What '
                 'was scored); panel items from wiki-ja and wiki-zh repeat text the run trained on in an earlier pass.',
                 'Raw-BPB discovery winner **#998** minus policy favorite on the 16 held-out confirmation batches',
                 'positive in 0 of 32 batches), 32-batch bootstrap 95% interval **[-0.499500, -0.499500]**. '
                 'The policy trails uniform sampling on these batches.',
                 '![Policy versus measured gap](rawbpb_pool_ranking_20000.png)', '[PDF](rawbpb_pool_ranking_20000.pdf)']:
        assert line in text
    result['confirmation']['interpretation'] = 'The difference is unresolved on the confirmation batches.'
    with pytest.raises(ValueError, match='verdict'):
        readme(result, extra, table, panel, 20000, 'rawbpb_pool_ranking_20000.png')


def test_policy_minus_uniform_bootstraps_all_32_batches(step6500):
    gaps, canonical = tables()
    gaps[500] += np.where(np.arange(32) < 16, .0123, -.0123)  # the favorite: harder on discovery, easier on confirmation
    result, extra, table, panel = ranking(step6500, gaps, canonical, favorite=500)
    gap, raw = extra['policy_minus_uniform']['gap'], extra['policy_minus_uniform']['raw_bpb']
    difference = gaps[500] - gaps.mean(0)  # per batch: the one-hot policy's expectation minus the uniform mean
    assert (gap['batches'], gap['resamples'], gap['batches_positive']) == (32, 10000, 16)
    assert gap['mean'] == pytest.approx(difference.mean()) and gap['screen'] == pytest.approx(difference[:16].mean())
    low, high = gap['paired_batch_bootstrap_95pct']
    assert gap['confirm'] < low < 0 < high < gap['screen']  # a bootstrap of either half alone would exclude zero
    draws = np.random.default_rng(20261115).integers(0, 32, size=(10000, 32))
    assert [low, high] == pytest.approx(np.quantile(difference[draws].mean(1), [.025, .975]))
    assert raw['batches_positive'] == 32 and raw['paired_batch_bootstrap_95pct'][0] > 0
    text = readme(result, extra, table, panel, 20000, 'figure.png')
    assert 'positive in 16 of 32 batches), 32-batch bootstrap 95% interval' in text
    assert 'The policy is not distinguishable from uniform sampling on these batches.' in text
    assert 'The policy beats uniform sampling on these batches.' in text


def test_hardest_overlaps_compare_the_two_metrics_and_the_two_halves(step6500):
    gaps, canonical = tables()
    top = np.arange(980, 1000)
    gaps[top, 16:] -= ((top - 980) * .00105)[:, None]  # confirmation half: the 20 hardest by gap in reverse order
    result, extra, table, panel = ranking(step6500, gaps, canonical)
    assert [row['index'] for row in table[:20]] == list(range(999, 979, -1))  # over all 32 batches still by index
    assert extra['gap_raw_bpb_hardest_overlap'] == {'10': 5, '50': 25, '100': 50}  # raw BPB puts even tokenizers first
    assert extra['half_hardest_overlap'] == dict(gap={'10': 0, '50': 50, '100': 100},
                                                 raw_bpb={'10': 10, '50': 50, '100': 100})
    text = readme(result, extra, table, panel, 20000, 'figure.png')
    assert 'their hardest 10 / 50 / 100 share **5 / 25 / 50** tokenizers.' in text
    assert "The halves' hardest 10 / 50 / 100 share **0 / 50 / 100** tokenizers by gap and **10 / 50 / 100**" in text


def test_pick_counts_tell_the_most_picked_tokenizer_from_the_favorite(step6500):
    counts = np.zeros(1000, dtype=int); counts[SELECTED] = [17, 9, 30, 8, 7, 6, 5, 4, 3]  # #824 17, #771 30
    result, extra, table, panel = ranking(step6500, *tables(), favorite=824, counts=counts)
    assert extra['picks'] == dict(total=89, most_picked=771, most_picked_count=30, favorite_picks=17,
                                  selected_are_most_picked=True)
    assert [(row['index'], row['picks']) for row in extra['selected']] == list(zip(SELECTED, counts[SELECTED].tolist()))
    assert (extra['selected'][2]['reference_gap_rank'], extra['selected'][2]['reference_policy_probability']) == (229, 1)
    text = readme(result, extra, table, panel, 20000, 'figure.png')
    assert "The rows are the run's 9 most-picked tokenizers over its 89 updates." in text
    assert ('the most-picked tokenizer over the run is #771 (30 picks), while the step-20,000 favorite #824 was '
            'picked 17 times.') in text
    counts[5] = 10  # a tokenizer outside the table is now among the 9 most picked
    result, extra, table, panel = ranking(step6500, *tables(), favorite=824, counts=counts)
    assert not extra['picks']['selected_are_most_picked']
    assert 'most-picked tokenizers over its' not in readme(result, extra, table, panel, 20000, 'figure.png')
    counts[824] = 40
    result, extra, table, panel = ranking(step6500, *tables(), favorite=824, counts=counts)
    assert ('the favorite #824 is also the most-picked tokenizer over the run (40 picks).'
            in readme(result, extra, table, panel, 20000, 'figure.png'))


def test_bottom_group_is_one_block_plus_singletons_and_the_outside_spearman_leaves_it_out(step6500):
    gaps, canonical = tables()
    rest = np.arange(218, 1000)
    gaps[rest] = gaps[1217 - rest]  # outside the step-6,500 bottom group the order is reversed here
    gaps[217] += .0015              # regex Unigram #217 moves between #999 and #998
    result, extra, table, panel = ranking(step6500, gaps, canonical)
    reference = extra['reference']; group = reference['bottom_group']
    bpe, wordpiece = (dict(pretokenizer='regex', algorithm=name, tokenizers=count)
                      for name, count in [('bpe', 115), ('wordpiece', 102)])
    assert (group['tokenizers'], group['reference_ranks']) == (218, [783, 1000])
    assert group['composition'] == [bpe, wordpiece, dict(pretokenizer='regex', algorithm='unigram', tokenizers=1,
                                                         indices=[217])]
    assert group['block'] == dict(kinds=[bpe, wordpiece], tokenizers=217, gap_ranks_here=[784, 1000, True],
                                  gap_ranks_reference=[784, 1000, True])
    assert group['singletons'] == [dict(index=217, pretokenizer='regex', algorithm='unigram', gap_rank_here=782,
                                        gap_rank_reference=783)]
    assert group['gap_ranks_here'] == dict(minimum=782, median=891.5, maximum=1000)
    assert group['in_bottom_here'] == 217 and group['hardest_here']['index'] == 217
    assert reference['gap_spearman_outside_bottom_group'] == pytest.approx(-1)
    assert reference['gap_pearson_outside_bottom_group'] == pytest.approx(-1)
    assert reference['gap_spearman'] > 0  # only the shared bottom group keeps the all-pool Spearman positive
    assert (reference['checkpoint_step'], reference['same_panel'], reference['hardest_100_overlap']) == (6500, False, 0)
    text = readme(result, extra, table, panel, 20000, 'figure.png')
    assert ('The 217 regex BPE and regex WordPiece tokenizers fill ranks 784–1,000 in both rankings; '
            '#217 (regex Unigram) is rank 782 here and 783 at step 6,500. ') in text
    assert 'Over the 782 tokenizers outside the bottom group described next, the gap Spearman is **-1.000**' in text
    gaps[0] += 2  # a block member becomes the hardest tokenizer here
    result, extra, table, panel = ranking(step6500, gaps, canonical)
    assert extra['reference']['bottom_group']['block']['gap_ranks_here'] == [1, 1000, False]
    assert ('The 217 regex BPE and regex WordPiece tokenizers lie within ranks 1–1,000 here and fill ranks '
            '784–1,000 at step 6,500; #217 (regex Unigram) is rank 783 here and 783 at step 6,500. '
            in readme(result, extra, table, panel, 20000, 'figure.png'))


@pytest.mark.parametrize('change,message', [
    (lambda rows: rows.pop(), 'Missing or duplicated'),
    (lambda rows: rows.__setitem__(999, rows[998]), 'Missing or duplicated'),
    (lambda rows: rows[7].update(checkpoint_step=18000), 'provenance'),
    (lambda rows: rows[7].update(panel_sha256='published'), 'provenance'),
    (lambda rows: rows[7].update(policy_probability=.5), 'provenance'),
    (lambda rows: rows[7]['batches'].pop(), 'provenance'),
    (lambda rows: rows[7]['batches'][3].update(valid_examples=127), 'protocol'),
    (lambda rows: rows[7]['batches'][3].update(maximum_positions=129), 'protocol'),
    (lambda rows: rows[7]['batches'][3].update(scored_bytes=0), 'protocol'),
    (lambda rows: rows[7]['batches'][3].update(dropout_seed=2026111500), 'protocol'),
    (lambda rows: rows[7]['batches'][20].update(split='screen'), 'protocol'),
    (lambda rows: rows[7]['batches'][3].update(gap_bpb=.5), 'protocol'),
    (lambda rows: rows[7]['batches'][3].update(raw_bpb=float('nan'), gap_bpb=float('nan')), 'Nonfinite'),
], ids=['missing-tokenizer', 'duplicated-tokenizer', 'wrong-step', 'other-panel', 'other-policy', 'missing-batch',
        'wrong-valid-examples', 'position-cap', 'no-bytes', 'reused-dropout-seed', 'moved-split',
        'gap-not-raw-minus-canonical', 'nonfinite'])
def test_incomplete_or_mismatched_scores_are_refused(change, message):
    rows, policy, panel = records(*tables())
    change(rows)
    with pytest.raises(ValueError, match=message):
        analyze(rows, policy, panel, 20000)


def test_complete_scores_of_another_step_are_refused():
    with pytest.raises(ValueError, match='provenance'):
        analyze(*records(*tables()), 18000)


def test_reference_must_be_the_step_6500_ranking_of_the_same_pool(tmp_path, step6500):
    rows, policy, panel = records(*tables(), favorite=771, step=6500, panel='published')
    index = pool(rows)
    other = json.loads(json.dumps(index)); other['tokenizers'][5]['config']['algorithm'] = 'unigram'

    def refused(name, message, policy=policy, rows=rows[:1], reference_pool=index, panel=panel):
        with pytest.raises(ValueError, match=message):
            load_reference(write_reference(tmp_path / name, rows, policy, reference_pool, panel), index)

    refused('policy-step', 'policy.json belongs to another checkpoint step', policy=dict(policy, step=20000))
    refused('panel-step', 'must hold the step-6,500 ranking', panel=dict(panel, checkpoint_step=8000))
    refused('score-step', 'must hold the step-6,500 ranking', rows=[dict(rows[0], checkpoint_step=8000)], panel=None)
    refused('pool', 'different tokenizer pool', reference_pool=other)
    refused('pool-hash', 'pool index metadata changed', panel=dict(panel, pool_index_sha256='0' * 64))
    refused('identity', 'identity mismatch', rows=[dict(rows[0], tokenizer_sha256='0' * 64)])
    assert step6500['result']['checkpoint_step'] == 6500 and step6500['result']['panel_sha256'] == 'published'
    assert set(step6500['files']) == {'all-scores.json.gz', 'policy.json', 'pool-index.json', 'panel-manifest.json'}


def test_provenance_and_native_identity_mismatches_are_refused(tmp_path):
    rows, policy, panel = records(*tables())
    policy = dict(policy, policy_sha256='a' * 64)
    (tmp_path / 'panel-manifest.json').write_text(json.dumps(panel))
    check_provenance(tmp_path, panel, policy, 20000)
    for changed_panel, changed_policy in [(dict(panel, checkpoint_step=18000), policy), (panel, dict(policy, step=18000))]:
        with pytest.raises(ValueError, match='another checkpoint step'):
            check_provenance(tmp_path, changed_panel, changed_policy, 20000)
    plan = dict(checkpoint_step=20000, panel_manifest_sha256=sha256(tmp_path / 'panel-manifest.json'))
    (tmp_path / 'checkpoint-manifest.json').write_text(json.dumps(dict(files={'policy.pt': dict(sha256='a' * 64)})))
    for changed in [dict(plan, checkpoint_step=18000), dict(plan, panel_manifest_sha256='0' * 64)]:
        (tmp_path / 'run-plan.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='Run plan differs'):
            check_provenance(tmp_path, panel, policy, 20000)
    (tmp_path / 'run-plan.json').write_text(json.dumps(plan))
    check_provenance(tmp_path, panel, policy, 20000)
    with pytest.raises(ValueError, match='frozen checkpoint'):
        check_provenance(tmp_path, panel, dict(policy, policy_sha256='b' * 64), 20000)
    check_identity(rows, pool(rows))
    for change in [dict(tokenizer_sha256='0' * 64), dict(config=dict(rows[3]['config'], algorithm='unigram'))]:
        with pytest.raises(ValueError, match='identity mismatch'):
            check_identity(rows[:3] + [dict(rows[3], **change)] + rows[4:], pool(rows))


def test_fresh_panel_text_discloses_the_cross_check_and_second_readings():
    fresh, cross, later = panel_paragraphs(manifest(), 20000)
    assert fresh.startswith('**Fresh panel.** The next 32 training batches of 128 windows (4,096 initial examples, '
                            'all 3 corpus buckets) that the run would have drawn after its step-20,000 full restart')
    assert cross.endswith('2,000 of 2,000 logged paired-batch hashes and rejected-example counts matched (7 rejected '
                          'examples in total), and the replayed stream drew an identical panel.')
    assert later == ('**Not all unseen.** wiki-ja and wiki-zh were past the first pass over their training quota when '
                     'the panel was drawn (passes completed at step 20,000 → after the panel items: wiki-ja 1 → 1, '
                     'wiki-zh 1 → 1). Windows drawn in a later pass repeat text the run trained on in an earlier '
                     'pass; these buckets hold 91 of the 4,096 panel items.')
    first = {'web-en': 0, 'wiki-ja': 0, 'wiki-zh': 0}
    lines = panel_paragraphs(manifest(cross_check=None, corpus_stream=dict(
        epochs_at_checkpoint=first, epochs_after_items=first, epochs_after_banks=dict(first, **{'web-en': 1}))), 20000)
    assert lines[1].startswith('**Cross-check:** none.')
    assert lines[2] == ('**Unseen text.** Every panel item comes from the first pass over its bucket, after the windows '
                        'the run read up to step 20,000. The replacement banks, used only after a rejection, also '
                        'reach a later pass in web-en.')
    wrapped = panel_paragraphs(manifest(corpus_stream=dict(epochs_at_checkpoint=first, epochs_after_items=dict(
        first, **{'wiki-ja': 1}), epochs_after_banks=dict(first, **{'wiki-ja': 1}))), 20000)[2]
    assert wrapped.endswith('these buckets hold 50 of the 4,096 panel items. A bucket that wrapped during the draw '
                            'repeats text only after its wrap.')
    published = manifest(panel_mode='published', panel_source='published: the published panel of the step-6,500 ranking')
    assert panel_paragraphs(published, 20000) == ['**Published panel.** The published panel of the step-6,500 ranking.']


def test_report_text_with_a_private_path_email_or_account_name_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr('getpass.getuser', lambda: 'tester')
    for name, text in {'clean.md': 'See reports/rawbpb-pool-ranking-20000/README.md and the testers note.',
                       'path.json': '{"state": "/gpfs/projects/stf/someone/run"}', 'home.md': 'cd ~/anytok',
                       'email.csv': 'owner,someone@example.org', 'account.md': 'scored by tester on the cluster',
                       'notes.txt': '/home/someone/ is not in a report text file'}.items():
        (tmp_path / name).write_text(text)
    assert private_files(tmp_path) == ['account.md', 'email.csv', 'home.md', 'path.json']


def state_folder(folder, rows, policy, panel):
    """A copied sweep state as main() reads it, with a run plan and checkpoint manifest that agree."""
    folder.mkdir()
    (folder / 'pool-index.json').write_text(json.dumps(pool(rows)))
    (folder / 'panel-manifest.json').write_text(json.dumps(dict(panel, pool_index_sha256=sha256(folder / 'pool-index.json'))))
    (folder / 'policy.json').write_text(json.dumps(dict(policy, policy_sha256='a' * 64)))
    (folder / 'run-plan.json').write_text(json.dumps(dict(checkpoint_step=20000, panel_manifest_sha256=sha256(
        folder / 'panel-manifest.json'))))
    (folder / 'checkpoint-manifest.json').write_text(json.dumps(dict(files={'policy.pt': dict(sha256='a' * 64)})))
    (folder / 'preparation-status.json').write_text(json.dumps(dict(stage='prepared', panel_mode='fresh')))
    with gzip.open(folder / 'all-scores.json.gz', 'wt', compresslevel=1) as stream:
        json.dump(rows, stream)
    return folder


def test_main_writes_the_report_and_withholds_its_manifest_for_private_text(tmp_path, monkeypatch, capsys,
                                                                            reference_folder):
    monkeypatch.setattr('getpass.getuser', lambda: 'tester')
    monkeypatch.setattr(report, 'plot', lambda result, rows, gaps, step, stem, folder: [
        (folder / stem.with_suffix(suffix).name).write_bytes(b'figure') for suffix in ['.png', '.pdf']])
    state, folder = state_folder(tmp_path / 'state', *records(*tables())), tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['report', '--state', str(state), '--report', str(folder),
                                      '--plots', str(tmp_path / 'figure'), '--reference', str(reference_folder)])
    with np.errstate():
        report.main()
    files = json.loads((folder / 'manifest.json').read_text())['files']
    assert set(files) == set(report.INPUTS) - {'input-verification.json'} | {
        'README.md', 'analysis.json', 'tokenizer-ranking.csv', 'figure.png', 'figure.pdf'}
    assert all(files[name]['sha256'] == sha256(folder / name) for name in files)
    analysis = json.loads((folder / 'analysis.json').read_text())
    assert analysis['checkpoint_step'] == 20000 and analysis['raw_bpb_and_reference']['reference']['checkpoint_step'] == 6500
    gap, extra = map(json.loads, capsys.readouterr().out.splitlines())
    assert gap['dominant_rank'] == 1000 and extra['raw_bpb']['dominant_rank'] == 750 and 'ranks' not in extra['reference']
    (folder / 'manifest.json').unlink()
    (state / 'preparation-status.json').write_text(json.dumps(dict(stage='failed', error='/gpfs/projects/stf/someone/x')))
    with np.errstate(), pytest.raises(ValueError, match='preparation-status.json; manifest.json not written'):
        report.main()
    assert not (folder / 'manifest.json').exists()
    (state / 'run-plan.json').write_text(json.dumps(dict(checkpoint_step=18000)))
    with np.errstate(), pytest.raises(ValueError, match='Run plan differs'):
        report.main()
    (state / 'pool-index.json').write_text('{}')
    with np.errstate(), pytest.raises(ValueError, match='Pool index metadata changed'):
        report.main()


def pair(raw, native, canonical, min_chars):
    """Rejects windows that start with '!'; otherwise a pair whose targets and bytes match."""
    return None if raw.startswith('!') else (['native', raw, len(raw)], ['canonical', raw, len(raw)],
                                              dict(window=raw, maximum_positions=64))


def batch(texts, replacements=()):
    return dict(items=[dict(text=text, raw_sha256=hashlib.sha256(text.encode()).hexdigest(), domain='web-en')
                       for text in texts], replacements={'web-en': list(replacements)})


def test_rejected_items_take_the_next_replacements_in_order_and_count_every_draw():
    texts = ['!a', '!b'] + [f'item {i}' for i in range(126)]
    examples, controls, proofs, rejected = pair_batch(batch(texts, ['!c', 'first', 'second', 'third']), None, None, pair)
    assert len(examples) == len(controls) == len(proofs) == 128 and rejected == 3
    assert [proof['window'] for proof in proofs[:3]] == ['first', 'second', 'item 0']


def test_pairing_refuses_changed_text_too_many_rejections_lost_examples_and_broken_pairs():
    texts = [f'item {i}' for i in range(128)]
    changed = batch(texts); changed['items'][5]['text'] = 'edited'
    with pytest.raises(ValueError, match='Fixed raw input changed'):
        pair_batch(changed, None, None, pair)
    assert pair_batch(batch(['!x'] + texts[1:], ['!'] * 127 + ['last']), None, None, pair)[3] == 128
    with pytest.raises(ValueError, match='rejection limit exceeded'):
        pair_batch(batch(['!x'] + texts[1:], ['!'] * 128), None, None, pair)
    with pytest.raises(ValueError, match='Complete 128-example update'):
        pair_batch(batch(texts[:127]), None, None, pair)
    for broken in [lambda raw, native, canonical, min_chars: (['n', raw, 1], ['c', raw, 1], dict(maximum_positions=129)),
                   lambda raw, native, canonical, min_chars: (['n', raw, 1], ['c', raw, 2], dict(maximum_positions=64))]:
        with pytest.raises(ValueError, match='Paired targets/bytes or position cap'):
            pair_batch(batch(texts), None, None, broken)


def test_real_tokenizers_reconstruct_the_replacements_of_rejected_items():
    pytest.importorskip('torch'); pytest.importorskip('tokenizers'); pytest.importorskip('transformers')
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    spec = importlib.util.spec_from_file_location(
        'pairing', ROOT / 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py')
    pairing = importlib.util.module_from_spec(spec); spec.loader.exec_module(pairing)
    vocab = {'<eos>': 0, 'a': 1, 'b': 2, 'c': 3, ' ': 4, 'ab': 5}
    native = Tokenizer(models.BPE(vocab=vocab, merges=[])); native.decoder = decoders.Fuse()
    canonical = PreTrainedTokenizerFast(tokenizer_object=native, bos_token='<eos>', eos_token='<eos>')
    replacements = ['a b ' * 20, 'b c ' * 20] + ['ab c ' * 20] * 126
    examples, controls, proofs, rejected = pair_batch(batch(['a', 'b'] + ['ab c ' * 20] * 126, replacements),
                                                      native, canonical, pairing.make_paired_prefix_example)
    assert len(examples) == len(controls) == len(proofs) == 128 and rejected == 2
    for index in [0, 1]:
        reconstructed = canonical.decode(controls[index][0] + controls[index][1], clean_up_tokenization_spaces=False)
        assert reconstructed == replacements[index][:proofs[index]['window_chars']]
    assert all(a[1:] == b[1:] for a, b in zip(examples, controls))


class Corpus:
    """Windows read in order per bucket; a bucket's pass count rises every quota reads."""

    def __init__(self, names, weights, quota=1000):
        self.names, self.weights, self.quota = names, weights, quota
        self.reads, self.epochs = dict.fromkeys(names, 0), dict.fromkeys(names, 0)

    def next_text(self, name, chars):
        self.reads[name] += 1; self.epochs[name] = self.reads[name] // self.quota
        return (f'{name} {self.reads[name]} ' * 300)[:chars]


def test_panel_draw_is_reproducible_and_counts_passes_before_the_banks():
    names, progress = ['web-en', 'wiki-ja', 'wiki-zh'], []
    batches, coverage, epochs = prepare.draw_panel(Corpus(names, [8, 1, 1]), random.Random(42), progress.append)
    items = [item for batch in batches for item in batch['items']]
    assert [batch['split'] for batch in batches] == ['screen'] * 16 + ['confirm'] * 16
    assert [batch['dropout_seed'] for batch in batches] == [2026111500 + index for index in range(32)]
    assert len(items) == sum(coverage.values()) == 4096 and set(coverage) == set(names) and progress == list(range(1, 33))
    assert items[129]['id'] == 'batch-01/slot-001' and all(len(item['text']) == 512 for item in items)
    assert all(item['raw_sha256'] == hashlib.sha256(item['text'].encode()).hexdigest() for item in items)
    assert all(len(batch['replacements'][name]) == 128 for batch in batches for name in names)
    assert epochs == {name: coverage[name] // 1000 for name in names}  # after the items, before the banks
    corpus = Corpus(names, [8, 1, 1])
    again, _, _ = prepare.draw_panel(corpus, random.Random(42), lambda done: None)
    assert again == batches and prepare.canonical_sha256(again) == prepare.canonical_sha256(batches)
    assert corpus.epochs == {name: (coverage[name] + 32 * 128) // 1000 for name in names} != epochs
    with pytest.raises(ValueError, match='did not cover'):
        prepare.draw_panel(Corpus(names, [1, 1, 0]), random.Random(42), lambda done: None)


def test_replay_reproduces_the_logged_stream_and_refuses_a_different_log():
    def paired(raw, tokenizer, canonical, min_chars):  # rejects every 50th window
        read = int(raw.split()[1])
        return None if read % 50 == 0 else ([read], [read], dict(read=read, tokenizer=tokenizer))
    trainer, tokenizers = SimpleNamespace(make_paired_prefix_example=paired), SimpleNamespace(load=lambda i: (f'#{i}', None))
    # Update 1 reads windows 1-128 and replaces its rejected 50th and 100th windows with windows 129 and 130.
    reads = [*range(1, 50), 129, *range(51, 100), 130, *range(101, 129)]
    digest = hashlib.sha256(json.dumps([dict(read=read, tokenizer='#3') for read in reads], sort_keys=True).encode())
    log = dict(tokenizer_pool_index=3, paired_batch_sha256=digest.hexdigest(), rejected_examples=2)
    progress = []
    replay = lambda row: prepare.replay_training_stream(trainer, Corpus(['web-en'], [1]), random.Random(0), tokenizers,
                                                        None, {1: row}, 1, 1, lambda *done: progress.append(done))
    assert replay(log) == 2 and progress == [(1, 2)]
    for change in [dict(rejected_examples=1), dict(tokenizer_pool_index=4)]:
        with pytest.raises(ValueError, match='replay mismatch at 1'):
            replay(dict(log, **change))


def test_checkpoint_files_staged_code_and_preparation_status_are_checked(tmp_path, monkeypatch, capsys):
    step = tmp_path / 'step-020000'; step.mkdir()
    (step / 'policy.pt').write_bytes(b'policy')
    files = {'policy.pt': dict(bytes=6, sha256=sha256(step / 'policy.pt'))}
    (step / 'manifest.json').write_text(json.dumps(dict(step=20000, kind='full_restart_checkpoint', files=files)))
    checkpoint = prepare.full_restart_manifest(step, 20000)
    prepare.verify_checkpoint_file(step, checkpoint, 'policy.pt')
    (step / 'policy.pt').write_bytes(b'Policy')  # same size, other bytes
    with pytest.raises(ValueError, match='Saved checkpoint changed'):
        prepare.verify_checkpoint_file(step, checkpoint, 'policy.pt')
    with pytest.raises(ValueError, match='not the full restart checkpoint of step 18000'):
        prepare.full_restart_manifest(step, 18000)
    (step / 'manifest.json').write_text(json.dumps(dict(step=20000, kind='weights_checkpoint', files=files)))
    with pytest.raises(ValueError, match='not the full restart checkpoint'):
        prepare.full_restart_manifest(step, 20000)
    code = tmp_path / 'code'; code.mkdir(); (code / 'source-manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='not the code this run trained with'):
        prepare.load_trainer(code)
    assert prepare.canonical_sha256(dict(b=1, a=[2])) == hashlib.sha256(b'{"a":[2],"b":1}').hexdigest()
    monkeypatch.setattr('getpass.getuser', lambda: 'tester')
    assert (prepare.without_account('No such file: /gpfs/projects/stf/tester/run (tester)')
            == 'No such file: /gpfs/projects/stf/<user>/run (tester)')
    prepare.Status(tmp_path)('drawing_panel', 'fixed replacement banks', completed=3, total=32)
    status = json.loads((tmp_path / 'preparation-status.json').read_text())
    assert {key: status[key] for key in ['stage', 'detail', 'completed', 'total']} == dict(
        stage='drawing_panel', detail='fixed replacement banks', completed=3, total=32)
    assert json.loads(capsys.readouterr().out) == status


def test_status_assembles_all_scores_only_when_all_four_shards_are_complete(tmp_path):
    (tmp_path / 'status').mkdir(); (tmp_path / 'scores').mkdir()
    for index in range(999):
        (tmp_path / 'scores' / f'{index:04d}.json').write_text(json.dumps(dict(index=index)))
    for shard in range(4):
        (tmp_path / 'status' / f'{shard}.json').write_text(json.dumps(dict(stage='complete', completed=250)))
    (tmp_path / 'submissions.json').write_text(json.dumps(dict(prepare=1, score=2)))
    status = lambda: json.loads(subprocess.run(
        [sys.executable, ROOT / 'scripts/rawbpb_pool_ranking_status.py', '--state', tmp_path],
        capture_output=True, text=True, check=True).stdout)
    first = status()
    assert (first['completed'], first['complete'], first['preparation']) == (999, False, None)
    assert first['submissions'] == dict(prepare=1, score=2) and first['workers']['3']['stage'] == 'complete'
    (tmp_path / 'scores/0999.json').write_text(json.dumps(dict(index=999)))
    (tmp_path / 'status/3.json').write_text(json.dumps(dict(stage='scoring', completed=249)))
    assert not status()['complete'] and not (tmp_path / 'all-scores.json.gz').exists()
    (tmp_path / 'status/3.json').write_text(json.dumps(dict(stage='complete', completed=250)))
    assert status()['complete']
    with gzip.open(tmp_path / 'all-scores.json.gz', 'rt') as stream:
        assert [row['index'] for row in json.load(stream)] == list(range(1000))
