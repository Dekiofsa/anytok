"""Report the frozen full-pool ranking of the raw-BPB run's checkpoint.

Port of scripts/report_training_pool_ranking.py at 5f431ea on zett-modal-20k, the
report of the regret run's step-6,500 ranking (reports/regret-pool-ranking-6500 on
zett-modal-20k). Only these differ:

- The state, report, figure and reference locations and the checkpoint step are
  arguments. analyze() keeps that script's logic and checks; the step it fixed at
  6500 is passed in. Its gap ranking (gap_bpb = transferred minus canonical BPB,
  called regret there) stays the primary ranking.
- plot() keeps that figure's layout, axes and styling; only titles, labels and
  file names change. monitoring/web/Manrope.ttf is a variable font: matplotlib
  3.11, which drew the step-6,500 figure, used its Regular instance, while
  older matplotlib draws the default ExtraLight. plot() therefore draws with a
  static Regular instance made by fontTools (a matplotlib dependency) in a
  temporary folder, so the weight matches under any matplotlib version.
- New, beside analyze(): analyze_raw_and_reference() repeats the policy
  comparison, the screen/confirm split and the paired confirmation bootstrap on
  raw_bpb (this run's training reward) and compares the gap ranking with the
  step-6,500 ranking in --reference. analysis.json holds it under
  'raw_bpb_and_reference'.
- The README describes the panel recorded in panel-manifest.json and both
  metrics. main() also checks the step, policy and run plan against each other,
  requires every number in analysis.json to be finite, and refuses to write
  manifest.json while a report text file holds an absolute path, an email
  address or the local account name.
- Floating-point warnings are off: numpy 2.0's Accelerate BLAS on macOS raises
  spurious divide/overflow/invalid flags in matmul (p @ gaps). Matmul results
  can differ from another BLAS in the last bits, as in policy_expected_gap.
- Added after an independent check of the step-20,000 report (2026-11-21); no
  earlier number changes. The README calls the confirmation batches held out,
  prints policy mass on the hardest tokenizers in scientific notation, and adds
  a 32-batch bootstrap of policy minus uniform for each metric, the hardest
  10/50/100 shared between gap and raw BPB and between the two halves, each
  selected tokenizer's pick count, the bottom group's rank blocks and the gap
  Spearman outside it. load_reference() also requires the reference policy's
  step to be 6,500. Smaller differences from his script: INPUTS also copies
  preparation-status.json, plot() selects the Agg backend, and main() prints a
  second JSON line.

--reference is a folder holding the step-6,500 report's all-scores.json.gz,
pool-index.json and policy.json (its panel-manifest.json is used when present).
To fill it from git, for each NAME:

  git cat-file blob origin/zett-modal-20k:reports/regret-pool-ranking-6500/NAME > FOLDER/NAME
"""
import argparse
from collections import Counter
import csv
import getpass
import gzip
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from package_training_inputs import atomic_json, sha256
from pool_policy_ranking import average_ranks, compare_policy

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ['panel-manifest.json', 'policy.json', 'run-plan.json', 'checkpoint-manifest.json', 'all-scores.json.gz',
          'input-verification.json', 'pool-index.json', 'preparation-status.json']
SPLITS = [('all', slice(None)), ('screen', slice(0, 16)), ('confirm', slice(16, 32))]
REFERENCE_STEP = 6500
REFERENCE_NAME = "the regret run's step-6,500 ranking (reports/regret-pool-ranking-6500 on zett-modal-20k)"
REFERENCE_BOTTOM = (783, 1000)  # Bottom group of the step-6,500 chart: its 217 regex BPE and regex WordPiece
                                # tokenizers (ranks 784-1,000) and regex Unigram #169 (rank 783).
SELECTED = [824, 170, 771, 103, 873, 281, 314, 792, 941]
ALGORITHMS = dict(bpe='BPE', wordpiece='WordPiece', unigram='Unigram')


def analyze(records, policy, manifest, step):
    records = sorted(records, key=lambda row: row['index'])
    if [row['index'] for row in records] != list(range(1000)):
        raise ValueError('Missing or duplicated tokenizer scores')
    p = np.asarray(policy['probabilities'], dtype=float)
    for row in records:
        if (row['checkpoint_step'] != step or row['panel_sha256'] != manifest['panel_sha256']
                or len(row['batches']) != 32 or row['policy_probability'] != p[row['index']]):
            raise ValueError('Score provenance or complete batch coverage mismatch')
        for index, batch in enumerate(row['batches']):
            if (batch['index'] != index or batch['valid_examples'] != 128
                    or batch['maximum_positions'] > 128 or batch['scored_bytes'] <= 0
                    or batch['dropout_seed'] != 2026111500 + index
                    or batch['split'] != ('screen' if index < 16 else 'confirm')
                    or abs(batch['gap_bpb'] - (batch['raw_bpb'] - batch['canonical_bpb'])) > 1e-12):
                raise ValueError('Training reward protocol changed')
    gaps = np.asarray([[batch['gap_bpb'] for batch in row['batches']] for row in records])
    raw = np.asarray([[batch['raw_bpb'] for batch in row['batches']] for row in records])
    canonical = np.asarray([[batch['canonical_bpb'] for batch in row['batches']] for row in records])
    if (not np.isfinite(gaps).all() or not np.isfinite(raw).all() or not np.isfinite(canonical).all()
            or (raw < 0).any() or (canonical < 0).any()):
        raise ValueError('Nonfinite score; no tokenizer may be excluded')
    summaries = {name: compare_policy(gaps[:, selection].mean(1), p) for name, selection in
                 [('all', slice(None)), ('screen', slice(0, 16)), ('confirm', slice(16, 32))]}
    dominant = int(p.argmax()); winner = summaries['screen']['hardest_index']
    difference = gaps[winner, 16:] - gaps[dominant, 16:]
    rng = np.random.default_rng(20261115)
    draws = rng.integers(0, 16, size=(10000, 16))
    means = difference[draws].mean(1)
    interval = np.quantile(means, [.025, .975]).tolist()
    confirmation = dict(discovery_winner=winner, dominant_index=dominant,
        mean_gap_difference=float(difference.mean()), paired_batch_bootstrap_95pct=interval,
        batches=16, resamples=10000,
        interpretation=('The dominant action is also the discovery winner.' if winner == dominant else
            'Discovery winner has higher regret on confirmation batches.' if interval[0] > 0 else
            'The difference is unresolved on the confirmation batches.' if interval[1] >= 0 else
            'Discovery winner does not beat the dominant action on confirmation batches.'))
    result = dict(checkpoint_step=step, tokenizers=1000, batches_per_tokenizer=32,
        initial_text_examples=4096, corpus_buckets=23, dropout_active=True,
        aggregation=manifest['aggregation'], panel_sha256=manifest['panel_sha256'],
        comparisons=summaries, confirmation=confirmation,
        rank_stability_spearman=float(np.corrcoef(summaries['screen']['ranks'], summaries['confirm']['ranks'])[0, 1]))
    rows = []
    for index, row in enumerate(records):
        c = row['config']
        rows.append(dict(index=index, rank=summaries['all']['ranks'][index], algorithm=c['algorithm'],
            pretokenizer=c['pretokenizer'], vocab_size=row['vocabulary'], policy_probability=p[index],
            raw_bpb=float(raw[index].mean()), canonical_bpb=float(canonical[index].mean()),
            gap_bpb=float(gaps[index].mean()), screen_gap_bpb=float(gaps[index, :16].mean()),
            confirm_gap_bpb=float(gaps[index, 16:].mean()),
            screen_rank=summaries['screen']['ranks'][index], confirm_rank=summaries['confirm']['ranks'][index],
            replacement_draws=sum(batch['rejected_examples'] for batch in row['batches'])))
    return result, sorted(rows, key=lambda row: row['rank']), gaps


def manrope_regular(folder):
    """Static Regular (wght 400) instance of the variable monitoring/web/Manrope.ttf, saved in folder."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    path = Path(folder) / 'Manrope-Regular.ttf'
    # recalcTimestamp=False keeps head.modified fixed, so the PDF that embeds this font is byte-stable
    instancer.instantiateVariableFont(TTFont(ROOT / 'monitoring/web/Manrope.ttf', recalcTimestamp=False), {'wght': 400},
                                      updateFontNames=True).save(path)
    return path


def plot(result, rows, gaps, step, target_stem, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = tempfile.TemporaryDirectory()  # holds the Regular instance until the figure is saved
    font = manrope_regular(fonts.name)
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Manrope', 'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    p = np.array([row['policy_probability'] for row in sorted(rows, key=lambda row: row['index'])])
    summary = result['comparisons']['all']; dominant = summary['dominant_index']
    order = np.argsort(-gaps.mean(1), kind='stable'); ranks = np.arange(1, 1001)
    ax = axes[0, 0]
    ax.plot(ranks, gaps.mean(1)[order], color='#87919D', lw=1)
    ax.scatter(summary['dominant_rank'], gaps[dominant].mean(), color='#36967B', s=55, zorder=3)
    ax.annotate(f'Policy favorite #{dominant}: rank {summary["dominant_rank"]}',
                (summary['dominant_rank'], gaps[dominant].mean()), xytext=(8, 9), textcoords='offset points', fontsize=8)
    ax.set(title='Measured gap across the complete pool', xlabel='Estimated hardness rank (1 = hardest)', ylabel='Mean gap, BPB')
    ax = axes[0, 1]
    ax.plot(ranks, np.cumsum(p[order]), color='#36967B', label='Saved policy')
    ax.plot(ranks, ranks / 1000, color='#87919D', ls='--', label='Uniform sampling')
    ax.set(title='Policy probability on the hardest tokenizers', xlabel='Number of hardest tokenizers included', ylabel='Cumulative probability', ylim=(0, 1.02))
    ax.legend(frameon=False)
    ax = axes[1, 0]
    configs = {row['index']:row for row in rows}
    for algorithm, color in [('bpe','#2F8FD0'), ('wordpiece','#36967B'), ('unigram','#8657BC')]:
        ids = [i for i in range(1000) if configs[i]['algorithm'] == algorithm]
        ax.scatter(gaps[ids, :16].mean(1), gaps[ids, 16:].mean(1), s=9, alpha=.5, color=color, label=algorithm)
    low = min(gaps[:, :16].mean(1).min(), gaps[:, 16:].mean(1).min())
    high = max(gaps[:, :16].mean(1).max(), gaps[:, 16:].mean(1).max())
    ax.plot([low,high],[low,high],color='#B7B7B7',ls=':',lw=.8)
    ax.scatter(gaps[dominant,:16].mean(),gaps[dominant,16:].mean(),s=90,marker='*',color='#242424',zorder=4)
    ax.set(title='Does hardness carry over to confirmation batches?', xlabel='Discovery gap (16 batches)', ylabel='Confirmation gap (16 batches)')
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1, 1]
    winner = result['confirmation']['discovery_winner']
    values = np.vstack([gaps[:,16:].mean(0), p @ gaps[:,16:], gaps[dominant,16:], gaps[winner,16:]])
    rng = np.random.default_rng(20261115);draws=rng.integers(0,16,size=(10000,16))
    means = values.mean(1);bounds=np.quantile(values[:,draws].mean(2),[.025,.975],axis=1)
    ax.errorbar(np.arange(4),means,yerr=np.vstack([means-bounds[0],bounds[1]-means]),fmt='o',color='#2F8FD0',capsize=4)
    ax.set(xticks=np.arange(4),xticklabels=['Uniform','Saved policy','Policy favorite','Discovery winner'],
           title='Confirmation gap and paired-batch uncertainty',ylabel='Mean gap, BPB')
    ax.tick_params(axis='x',labelsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', color='#E9EDF1', lw=.6);ax.set_axisbelow(True)
    fig.suptitle(f'Frozen raw-BPB step {step:,} · all 1,000 tokenizers · gap = transferred − canonical BPB',fontsize=13)
    target_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png','pdf']:
        target=target_stem.with_suffix('.'+suffix);temporary=target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(temporary,dpi=170,facecolor='white',metadata={'CreationDate':None} if suffix=='pdf' else None)
        temporary.replace(target)
        shutil.copy2(target,report/target.name)
    plt.close(fig)
    fonts.cleanup()


def verdict(winner, dominant, interval, metric):
    """analyze()'s confirmation interpretation for any metric; 'regret' gives its exact text."""
    return ('The dominant action is also the discovery winner.' if winner == dominant else
            f'Discovery winner has higher {metric} on confirmation batches.' if interval[0] > 0 else
            'The difference is unresolved on the confirmation batches.' if interval[1] >= 0 else
            'Discovery winner does not beat the dominant action on confirmation batches.')


def spearman(a, b):
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def check_identity(records, index):
    """main()'s native tokenizer identity check of every score record against its pool index."""
    expected = {r['config']['index']: r for r in index['tokenizers']}
    for record in records:
        row = expected[record['index']]
        if record['tokenizer_sha256'] != row['tokenizer_sha256'] or record['config'] != row['config']:
            raise ValueError('Native tokenizer identity mismatch')


def check_provenance(state, manifest, policy, step):
    """Added: the step, policy, run plan and checkpoint manifest of the sweep agree."""
    if manifest.get('checkpoint_step', step) != step or policy.get('step', step) != step:
        raise ValueError('Panel manifest or policy belongs to another checkpoint step')
    if (state / 'run-plan.json').exists():
        plan = json.loads((state / 'run-plan.json').read_text())
        if (plan.get('checkpoint_step', step) != step
                or plan.get('panel_manifest_sha256', sha256(state / 'panel-manifest.json'))
                != sha256(state / 'panel-manifest.json')):
            raise ValueError('Run plan differs from the panel manifest or the step')
    if (state / 'checkpoint-manifest.json').exists() and 'policy_sha256' in policy:
        files = json.loads((state / 'checkpoint-manifest.json').read_text()).get('files', {})
        if 'policy.pt' in files and files['policy.pt']['sha256'] != policy['policy_sha256']:
            raise ValueError('Policy differs from the frozen checkpoint')


def load_reference(folder, index):
    """The step-6,500 sweep in --reference: same pool, main()'s identity checks, ranked by analyze()."""
    policy = json.loads((folder / 'policy.json').read_text())
    if policy.get('step', REFERENCE_STEP) != REFERENCE_STEP:
        raise ValueError('--reference policy.json belongs to another checkpoint step')
    pool = json.loads((folder / 'pool-index.json').read_text())
    identity = lambda value: [(row['tokenizer_sha256'], row['config'])
                              for row in sorted(value['tokenizers'], key=lambda row: row['config']['index'])]
    if identity(pool) != identity(index):
        raise ValueError('The reference ranks a different tokenizer pool')
    with gzip.open(folder / 'all-scores.json.gz', 'rt') as stream:
        records = json.load(stream)
    check_identity(records, pool)
    if (folder / 'panel-manifest.json').exists():
        manifest = json.loads((folder / 'panel-manifest.json').read_text())
        if sha256(folder / 'pool-index.json') != manifest['pool_index_sha256']:
            raise ValueError('Reference pool index metadata changed')
    else:  # analyze() then requires every record to share the first record's panel and step.
        manifest = dict(panel_sha256=records[0]['panel_sha256'], aggregation=None,
                        checkpoint_step=records[0]['checkpoint_step'])
    if manifest['checkpoint_step'] != REFERENCE_STEP:
        raise ValueError('--reference must hold the step-6,500 ranking')
    result, rows, gaps = analyze(records, policy, manifest, REFERENCE_STEP)
    files = {name: sha256(folder / name) for name in
             ['all-scores.json.gz', 'policy.json', 'pool-index.json', 'panel-manifest.json'] if (folder / name).exists()}
    return dict(result=result, gaps=gaps, files=files, probabilities=np.asarray(policy['probabilities'], dtype=float),
                raw=np.array([row['raw_bpb'] for row in sorted(rows, key=lambda row: row['index'])]))


def analyze_raw_and_reference(records, policy, result, gaps, reference):
    """New analysis beside analyze(): raw BPB, this run's reward, and the step-6,500 comparison."""
    records = sorted(records, key=lambda row: row['index'])
    p = np.asarray(policy['probabilities'], dtype=float)
    raw = np.asarray([[batch['raw_bpb'] for batch in row['batches']] for row in records])
    # (a) analyze()'s policy comparison, split and confirmation bootstrap, on raw BPB.
    summaries = {name: compare_policy(raw[:, selection].mean(1), p) for name, selection in SPLITS}
    dominant = int(p.argmax()); winner = summaries['screen']['hardest_index']
    difference = raw[winner, 16:] - raw[dominant, 16:]
    rng = np.random.default_rng(20261115)
    draws = rng.integers(0, 16, size=(10000, 16))
    interval = np.quantile(difference[draws].mean(1), [.025, .975]).tolist()
    confirmation = dict(discovery_winner=winner, dominant_index=dominant,
        mean_raw_bpb_difference=float(difference.mean()), paired_batch_bootstrap_95pct=interval,
        batches=16, resamples=10000, interpretation=verdict(winner, dominant, interval, 'raw BPB'))
    metrics = {'gap': result['comparisons']['all'], 'raw_bpb': summaries['all']}
    ranks = {name: np.asarray(summary['ranks']) for name, summary in metrics.items()}
    means = {'gap': gaps.mean(1), 'raw_bpb': raw.mean(1)}

    def versus_uniform(values):
        """Per batch, policy expectation minus uniform mean: its mean, halves and a 32-batch bootstrap."""
        difference = values.T @ p - values.mean(0)
        draws32 = np.random.default_rng(20261115).integers(0, 32, size=(10000, 32))
        return dict(mean=float(difference.mean()), screen=float(difference[:16].mean()),
                    confirm=float(difference[16:].mean()), batches_positive=int((difference > 0).sum()),
                    paired_batch_bootstrap_95pct=np.quantile(difference[draws32].mean(1), [.025, .975]).tolist(),
                    batches=32, resamples=10000)

    def overlap(a, b):
        """How many tokenizers two rankings share among their hardest 10, 50 and 100."""
        a, b = np.asarray(a), np.asarray(b)
        return {str(k): int(((a <= k) & (b <= k)).sum()) for k in (10, 50, 100)}

    def tokenizer(i):
        config = records[i]['config']
        return dict(index=i, algorithm=config['algorithm'], pretokenizer=config['pretokenizer'],
                    vocab_size=records[i]['vocabulary'], policy_probability=float(p[i]),
                    gap_rank=int(ranks['gap'][i]), mean_gap_bpb=float(means['gap'][i]),
                    raw_bpb_rank=int(ranks['raw_bpb'][i]), mean_raw_bpb=float(means['raw_bpb'][i]))

    # (e) The step-6,500 ranking: its gap ranks come from analyze() on its own scores.
    reference_gap = reference['gaps'].mean(1)
    reference_ranks = np.asarray(reference['result']['comparisons']['all']['ranks'])
    first, last = REFERENCE_BOTTOM
    inside = (reference_ranks >= first) & (reference_ranks <= last)
    group, rest = np.flatnonzero(inside), np.flatnonzero(~inside)
    here, here_raw = ranks['gap'][group], ranks['raw_bpb'][group]
    kinds = Counter((records[i]['config']['pretokenizer'], records[i]['config']['algorithm']) for i in group)
    composition = [dict(pretokenizer=pre, algorithm=alg, tokenizers=count,
                        **({'indices': [int(i) for i in group if (records[i]['config']['pretokenizer'],
                                                                   records[i]['config']['algorithm']) == (pre, alg)]}
                           if count <= 3 else {}))
                   for (pre, alg), count in sorted(kinds.items(), key=lambda item: (-item[1], item[0]))]
    # The group's large kinds as one block, its small kinds (at most 3 tokenizers) one by one, ranked here and there.
    size = lambda i: kinds[(records[i]['config']['pretokenizer'], records[i]['config']['algorithm'])]
    large = [int(i) for i in group if size(i) > 3]
    span = lambda values: [int(values.min()), int(values.max()), bool(values.max() - values.min() + 1 == len(values))]
    block = dict(kinds=[dict(pretokenizer=pre, algorithm=alg, tokenizers=count)
                        for (pre, alg), count in sorted(kinds.items(), key=lambda item: (-item[1], item[0])) if count > 3],
                 tokenizers=len(large),  # ranks: [first, last, whether the block fills every rank between]
                 gap_ranks_here=span(ranks['gap'][large]), gap_ranks_reference=span(reference_ranks[large])) if large else None
    singletons = [dict(index=int(i), pretokenizer=records[i]['config']['pretokenizer'],
                       algorithm=records[i]['config']['algorithm'], gap_rank_here=int(ranks['gap'][i]),
                       gap_rank_reference=int(reference_ranks[i])) for i in group if size(i) <= 3]
    spread = lambda values: dict(minimum=int(values.min()), median=float(np.median(values)), maximum=int(values.max()))
    bottom_group = dict(reference_ranks=[first, last], tokenizers=int(len(group)), composition=composition,
        block=block, singletons=singletons,
        gap_ranks_here=spread(here), raw_bpb_ranks_here=spread(here_raw),
        in_bottom_here=int((here > 1000 - len(group)).sum()),  # members among the last len(group) gap ranks here
        hardest_here=tokenizer(int(group[here.argmin()])),
        mean_gap_here=float(means['gap'][group].mean()), mean_gap_rest_here=float(means['gap'][rest].mean()),
        mean_gap_reference=float(reference_gap[group].mean()),
        mean_gap_rest_reference=float(reference_gap[rest].mean()))
    comparison = dict(checkpoint_step=reference['result']['checkpoint_step'],
        panel_sha256=reference['result']['panel_sha256'],
        same_panel=reference['result']['panel_sha256'] == result['panel_sha256'], files=reference['files'],
        gap_spearman=spearman(means['gap'], reference_gap),
        gap_pearson=float(np.corrcoef(means['gap'], reference_gap)[0, 1]),
        raw_bpb_spearman=spearman(means['raw_bpb'], reference['raw']),
        raw_bpb_pearson=float(np.corrcoef(means['raw_bpb'], reference['raw'])[0, 1]),
        hardest_100_overlap=int(((ranks['gap'] <= 100) & (reference_ranks <= 100)).sum()),
        gap_spearman_outside_bottom_group=spearman(means['gap'][rest], reference_gap[rest]),
        gap_pearson_outside_bottom_group=float(np.corrcoef(means['gap'][rest], reference_gap[rest])[0, 1]),
        bottom_group=bottom_group, ranks=reference_ranks.tolist())
    counts = [int(count) for count in policy['counts']]
    most = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
    picks = dict(total=sum(counts), most_picked=most[0], most_picked_count=counts[most[0]],
                 favorite_picks=counts[dominant], selected_are_most_picked=set(most[:len(SELECTED)]) == set(SELECTED))
    return dict(
        note=('comparisons holds compare_policy() output on mean raw BPB, so its *_gap fields are raw BPB; '
              'the top-level comparisons and confirmation are on gap_bpb, which they call regret'),
        comparisons=summaries, confirmation=confirmation,
        rank_stability_spearman=float(np.corrcoef(summaries['screen']['ranks'], summaries['confirm']['ranks'])[0, 1]),
        gap_raw_bpb_spearman=spearman(means['gap'], means['raw_bpb']),
        gap_raw_bpb_hardest_overlap=overlap(ranks['gap'], ranks['raw_bpb']),
        half_hardest_overlap=dict(
            gap=overlap(result['comparisons']['screen']['ranks'], result['comparisons']['confirm']['ranks']),
            raw_bpb=overlap(summaries['screen']['ranks'], summaries['confirm']['ranks'])),
        policy_minus_uniform=dict(gap=versus_uniform(gaps), raw_bpb=versus_uniform(raw)),
        picks=picks,
        # (b) policy favorite and the peak tokenizer of each metric
        policy_favorite=tokenizer(dominant),
        peaks={name: tokenizer(summary['hardest_index']) for name, summary in metrics.items()},
        # (c) policy mass on the hardest 1, 10, 50 and 100 by each metric
        probability_on_hardest={name: summary['probability_on_hardest'] for name, summary in metrics.items()},
        # (d) expected value under the policy (sum p x), uniform (mean) and the maximum
        expected={name: dict(policy=summary['policy_expected_gap'], uniform=summary['uniform_mean_gap'],
                             maximum=summary['hardest_gap'], policy_minus_uniform=summary['policy_minus_uniform'],
                             maximum_minus_policy=summary['oracle_minus_policy'])
                  for name, summary in metrics.items()},
        reference=comparison,
        # (f) selected tokenizers
        selected=[dict(tokenizer(i), reference_gap_rank=int(reference_ranks[i]),
                       reference_mean_gap_bpb=float(reference_gap[i]),
                       reference_policy_probability=float(reference['probabilities'][i]), picks=counts[i])
                  for i in SELECTED])


def listing(names):
    names = list(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else ''.join(names)


def later_buckets(manifest):
    """Fresh panel: buckets past their first pass when the items were drawn (prepare's panel_source rule)."""
    stream = manifest.get('corpus_stream') or {}
    return sorted(name for name, epochs in stream.get('epochs_after_items', {}).items() if epochs)


def panel_paragraphs(manifest, step):
    mode, buckets, examples = manifest.get('panel_mode'), len(manifest['domains']), manifest['examples']
    if mode == 'fresh':
        lines = [f'**Fresh panel.** The next 32 training batches of 128 windows ({examples:,} initial examples, '
                 f'all {buckets} corpus buckets) that the run would have drawn after its step-{step:,} full restart, '
                 'and the ordered 128-window replacement banks per bucket and batch, drawn from that checkpoint\'s '
                 'saved corpus stream and Python RNG with no replay. It is not the panel of the step-6,500 ranking.']
        cross = manifest.get('cross_check')
        if cross:
            lines.append(f'**Cross-check:** the step-{cross["from_step"]:,} full restart was restored and the data '
                         f'stream of the {cross["updates_replayed"]:,} logged updates {cross["from_step"] + 1:,}–{step:,} '
                         'was replayed (corpus draws, Python RNG and each update\'s paired examples for its logged '
                         f'tokenizer; no model updates); {cross["logged_hashes_matched"]:,} of '
                         f'{cross["updates_replayed"]:,} logged paired-batch hashes and rejected-example counts matched '
                         f'({cross["rejected_examples"]:,} rejected examples in total), and the replayed stream drew '
                         f'{"an identical" if cross["identical_panel"] else "a different"} panel.')
        else:
            lines.append('**Cross-check:** none. The panel was prepared without the replay cross-check '
                         '(cross_check is null in panel-manifest.json).')
        stream = manifest.get('corpus_stream')
        if not stream:
            lines.append(f'Panel source as recorded in panel-manifest.json: {manifest["panel_source"]}')
            return lines
        at, items, banks = (stream[key] for key in ['epochs_at_checkpoint', 'epochs_after_items', 'epochs_after_banks'])
        later = later_buckets(manifest)
        if later:
            text = (f'**Not all unseen.** {listing(later)} {"were" if len(later) > 1 else "was"} past the first pass '
                    f'over {"their" if len(later) > 1 else "its"} training quota when the panel was drawn (passes completed at step '
                    f'{step:,} → after the panel items: '
                    + ', '.join(f'{name} {at[name]} → {items[name]}' for name in later)
                    + f'). Windows drawn in a later pass repeat text the run trained on in an earlier pass; these '
                    f'buckets hold {sum(manifest["domains"].get(name, 0) for name in later):,} of the '
                    f'{examples:,} panel items.')
            if any(not at[name] for name in later):
                text += ' A bucket that wrapped during the draw repeats text only after its wrap.'
        else:
            text = (f'**Unseen text.** Every panel item comes from the first pass over its bucket, after the '
                    f'windows the run read up to step {step:,}.')
        banks_only = sorted(name for name, epochs in banks.items() if epochs and not items.get(name))
        if banks_only:
            text += (f' The replacement banks, used only after a rejection, also reach a later pass in '
                     f'{listing(banks_only)}.')
        lines.append(text)
        return lines
    if mode == 'published':
        source = manifest['panel_source'].split(': ', 1)[-1]
        return [f'**Published panel.** {source[:1].upper()}{source[1:]}.']
    updates = manifest.get('replayed_updates') or []
    return [f'The original corpus/Python state was replayed from full step {updates[0] - 1:,} to {updates[-1]:,} '
            f'and all {len(updates):,} actual paired-batch hashes matched. The next 32 recipe-sampled batches were '
            f'frozen, covering all {buckets} corpus buckets.' if updates else
            f'The panel holds 32 recipe-sampled batches covering all {buckets} corpus buckets.']


def readme(result, extra, rows, manifest, step, figure):
    a, c = result['comparisons']['all'], result['confirmation']
    r, rc = extra['comparisons']['all'], extra['confirmation']
    if verdict(c['discovery_winner'], c['dominant_index'], c['paired_batch_bootstrap_95pct'], 'regret') != c['interpretation']:
        raise ValueError('README verdict differs from analyze()')
    favorite, peaks, reference = extra['policy_favorite'], extra['peaks'], extra['reference']
    group = reference['bottom_group']
    mass = lambda values: ' / '.join(f'{values[k]:.2e}' for k in ['1', '10', '50', '100'])
    shared = lambda values: ' / '.join(str(values[k]) for k in ['10', '50', '100'])
    name = lambda item: f'{item["pretokenizer"]} {ALGORITHMS.get(item["algorithm"], item["algorithm"])}'

    def versus(item):
        low, high = item['paired_batch_bootstrap_95pct']
        return (f'Policy minus uniform: **{item["mean"]:+.6f} BPB** (discovery half {item["screen"]:+.6f}, confirmation '
                f'half {item["confirm"]:+.6f}; positive in {item["batches_positive"]} of {item["batches"]} batches), '
                f'{item["batches"]}-batch bootstrap 95% interval **[{low:+.6f}, {high:+.6f}]**. '
                + ('The policy is not distinguishable from uniform sampling on these batches.' if low <= 0 <= high else
                   f'The policy {"beats" if low > 0 else "trails"} uniform sampling on these batches.'))

    def blocks(group):
        parts, block = [], group.get('block')
        where = lambda span: f'{"fill" if span[2] else "lie within"} ranks {span[0]:,}–{span[1]:,}'
        if block:
            here, there = block['gap_ranks_here'], block['gap_ranks_reference']
            parts.append(f'The {block["tokenizers"]} {listing(name(kind) for kind in block["kinds"])} tokenizers '
                         + (f'{where(here)} in both rankings' if here == there
                            else f'{where(here)} here and {where(there)} at step 6,500'))
        parts += [f'#{item["index"]} ({name(item)}) is rank {item["gap_rank_here"]:,} here and '
                  f'{item["gap_rank_reference"]:,} at step 6,500' for item in group.get('singletons', [])]
        return '; '.join(parts) + '. ' if parts else ''

    picks = extra['picks']
    picked = ((f'The rows are the run\'s {len(extra["selected"])} most-picked tokenizers over its {picks["total"]:,} '
               'updates. ' if picks['selected_are_most_picked'] else '')
              + 'Picks counts how often the run sampled each tokenizer (policy.json counts). '
              + f'The step-{step:,} probabilities are a snapshot: '
              + (f'the favorite #{favorite["index"]} is also the most-picked tokenizer over the run '
                 f'({picks["favorite_picks"]:,} picks).' if picks['most_picked'] == favorite['index'] else
                 f'the most-picked tokenizer over the run is #{picks["most_picked"]} ({picks["most_picked_count"]:,} '
                 f'picks), while the step-{step:,} favorite #{favorite["index"]} was picked {picks["favorite_picks"]:,} times.'))
    config = lambda row: f'{row["algorithm"]} / {row["pretokenizer"]} / {row["vocab_size"]:,}'
    kind = lambda item: (f'{item["tokenizers"]} {item["pretokenizer"]} {ALGORITHMS.get(item["algorithm"], item["algorithm"])}'
                         + (', ' + ', '.join(f'#{i}' for i in item['indices']) if 'indices' in item else ''))
    mode = manifest.get('panel_mode')
    if mode == 'fresh':
        panel = (f'a fresh panel, the next 32 training batches the run would have drawn after step {step:,} '
                 '(see What was scored)')
        if later_buckets(manifest):
            panel += (f'; panel items from {listing(later_buckets(manifest))} repeat text the run trained on '
                      'in an earlier pass')
    else:
        panel = ('the published panel of the step-6,500 ranking (see What was scored)' if mode == 'published'
                 else 'a replayed panel')
    stability = f'gap **{result["rank_stability_spearman"]:.3f}**, raw BPB **{extra["rank_stability_spearman"]:.3f}**'
    halves = extra['half_hardest_overlap']
    stability += (f'. The halves\' hardest 10 / 50 / 100 share **{shared(halves["gap"])}** tokenizers by gap and '
                  f'**{shared(halves["raw_bpb"])}** by raw BPB, so close ranks among the hardest are not decisive')
    lines = ['# Frozen raw-BPB policy versus complete pool hardness', '',
        f'Frozen checkpoint **{step:,}** of the raw-BPB run, all **1,000** native pool tokenizers, **32 × 128** '
        f'training-style examples per tokenizer, on {panel}. The primary ranking is by **gap** (transferred minus '
        f'canonical BPB), the metric of {REFERENCE_NAME}, which calls it regret, so ranks compare with that chart; '
        'absolute values do not, because the run, checkpoint, reward and panel all differ. '
        '**Raw BPB** (transferred BPB), this run\'s training reward, is reported alongside.', '',
        '## Gap, as in the step-6,500 ranking', '',
        f'The saved policy favorite is **#{a["dominant_index"]}**, with **{100*a["dominant_probability"]:.2f}%** probability and estimated gap rank **{a["dominant_rank"]}/1,000**. Its mean gap is **{a["dominant_gap"]:+.6f} BPB**.', '',
        f'Uniform expected gap: **{a["uniform_mean_gap"]:+.6f} BPB**. Policy expected gap: **{a["policy_expected_gap"]:+.6f} BPB**. Estimated maximum: **{a["hardest_gap"]:+.6f} BPB**, tokenizer **#{a["hardest_index"]}**. Estimated oracle-minus-policy headroom: **{a["oracle_minus_policy"]:.6f} BPB**.', '',
        versus(extra['policy_minus_uniform']['gap']), '',
        f'Policy probability on the estimated hardest 1 / 10 / 50 / 100: **{mass(a["probability_on_hardest"])}**.', '',
        f'Discovery winner **#{c["discovery_winner"]}** minus policy favorite on the 16 held-out confirmation batches: **{c["mean_gap_difference"]:+.6f} BPB**, paired-batch bootstrap 95% interval **[{c["paired_batch_bootstrap_95pct"][0]:+.6f}, {c["paired_batch_bootstrap_95pct"][1]:+.6f}]**. '
        + verdict(c['discovery_winner'], c['dominant_index'], c['paired_batch_bootstrap_95pct'], 'gap'), '',
        '## Raw BPB, this run\'s reward', '',
        f'Ranked by mean raw BPB, the policy favorite **#{favorite["index"]}** is **{favorite["raw_bpb_rank"]}/1,000** at **{favorite["mean_raw_bpb"]:.6f} BPB**. '
        f'The raw-BPB peak is **#{peaks["raw_bpb"]["index"]}** ({config(peaks["raw_bpb"])}): raw BPB **{peaks["raw_bpb"]["mean_raw_bpb"]:.6f}**, gap **{peaks["raw_bpb"]["mean_gap_bpb"]:+.6f}** (gap rank {peaks["raw_bpb"]["gap_rank"]}). '
        f'The gap peak **#{peaks["gap"]["index"]}** ({config(peaks["gap"])}) has raw BPB **{peaks["gap"]["mean_raw_bpb"]:.6f}** (raw-BPB rank {peaks["gap"]["raw_bpb_rank"]}). '
        f'Over all 1,000 tokenizers the gap and raw-BPB rankings have Spearman **{extra["gap_raw_bpb_spearman"]:.3f}**, '
        f'but they agree less on which are hardest: their hardest 10 / 50 / 100 share '
        f'**{shared(extra["gap_raw_bpb_hardest_overlap"])}** tokenizers.', '',
        f'Uniform expected raw BPB: **{r["uniform_mean_gap"]:.6f}**. Policy expected raw BPB: **{r["policy_expected_gap"]:.6f}**. Estimated maximum: **{r["hardest_gap"]:.6f}**, tokenizer **#{r["hardest_index"]}**. Estimated oracle-minus-policy headroom: **{r["oracle_minus_policy"]:.6f} BPB**.', '',
        versus(extra['policy_minus_uniform']['raw_bpb']), '',
        f'Policy probability on the estimated hardest 1 / 10 / 50 / 100 by raw BPB: **{mass(r["probability_on_hardest"])}**.', '',
        f'Raw-BPB discovery winner **#{rc["discovery_winner"]}** minus policy favorite on the 16 held-out confirmation batches: **{rc["mean_raw_bpb_difference"]:+.6f} BPB**, paired-batch bootstrap 95% interval **[{rc["paired_batch_bootstrap_95pct"][0]:+.6f}, {rc["paired_batch_bootstrap_95pct"][1]:+.6f}]**. {rc["interpretation"]}', '',
        f'Rank stability between the discovery and confirmation halves (Spearman): {stability}.', '',
        '## Against the step-6,500 ranking', '',
        f'Per-tokenizer mean gap here against {REFERENCE_NAME}: Spearman **{reference["gap_spearman"]:.3f}**, Pearson **{reference["gap_pearson"]:.3f}** '
        f'(mean raw BPB: Spearman {reference["raw_bpb_spearman"]:.3f}, Pearson {reference["raw_bpb_pearson"]:.3f}). '
        f'Over the {1000 - group["tokenizers"]} tokenizers outside the bottom group described next, the gap Spearman is '
        f'**{reference["gap_spearman_outside_bottom_group"]:.3f}** (Pearson {reference["gap_pearson_outside_bottom_group"]:.3f}). '
        f'The two hardest-100 sets by gap share **{reference["hardest_100_overlap"]}** tokenizers. '
        f'That chart\'s bottom group, its ranks {group["reference_ranks"][0]:,}–{group["reference_ranks"][1]:,} '
        f'({group["tokenizers"]} tokenizers: {listing(kind(item) for item in group["composition"])}), ranks '
        f'**{group["gap_ranks_here"]["minimum"]:,}–{group["gap_ranks_here"]["maximum"]:,}** here (median {group["gap_ranks_here"]["median"]:,.1f}); '
        f'**{group["in_bottom_here"]} of {group["tokenizers"]}** are among the last {group["tokenizers"]} here. '
        f'{blocks(group)}'
        f'Its mean gap is **{group["mean_gap_here"]:+.6f}** here against **{group["mean_gap_rest_here"]:+.6f}** for the other {1000 - group["tokenizers"]} '
        f'(step 6,500: {group["mean_gap_reference"]:+.6f} against {group["mean_gap_rest_reference"]:+.6f}). '
        + ('Both rankings use the same panel.' if reference['same_panel'] else
           'The two rankings differ in model and panel together, so a difference cannot be assigned to either.'), '',
        '## Selected tokenizers', '',
        picked, '',
        f'| Tokenizer | Algorithm / pretokenizer | Vocab | Gap rank | Mean gap | Raw-BPB rank | Mean raw BPB | Step-6,500 rank | Policy probability (step {step:,}) | Picks (of {picks["total"]:,}) |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in extra['selected']:
        lines.append(f'| #{row["index"]} | {row["algorithm"]} / {row["pretokenizer"]} | {row["vocab_size"]:,} | {row["gap_rank"]} | {row["mean_gap_bpb"]:+.6f} | {row["raw_bpb_rank"]} | {row["mean_raw_bpb"]:.6f} | {row["reference_gap_rank"]} | {row["policy_probability"]:.5g} | {row["picks"]:,} |')
    lines += ['', 'Ranks: 1 = hardest of 1,000.', '', '## What was scored', '']
    for paragraph in panel_paragraphs(manifest, step):
        lines += [paragraph, '']
    lines += [
        'All actions used the same primary text draws and ordered replacement banks. Original rejected-example behavior and limits were preserved.', '',
        'Each action used its native prefix, a variable canonical continuation, matched canonical control, batch 128, microbatch 32 and the original 128-total-position cap. Canonical targets/bytes match each transferred pair; cuts can differ between actions exactly as in training. Canonical controls were recomputed for each action. Hypernetwork dropout stayed active with fixed independent per-batch seeds; a new embedding table was generated for every batch. No model or policy updates occurred.', '',
        'Gap and raw BPB are byte-weighted within each update, then averaged equally across updates. Raw BPB matches this run\'s policy reward; gap is the regret of the step-6,500 ranking, kept as the primary ranking. Lexical loss is excluded. No fixed-32-token or deterministic-dropout substitute was used.', '',
        '## Interpretation limits', '',
        'These are finite estimates of stochastic training-distribution rewards, not exact population ranks or a held-out generalization evaluation. The all-batch maximum is selected from 1,000 noisy estimates. The discovery winner is therefore also compared on held-out confirmation batches. Bootstrap intervals describe variation across these fixed training-reward batches; sequential corpus windows can be correlated. Close ranks should not be treated as decisive.', '',
        'This policy was trained on raw BPB, not on gap; the gap ranking is kept for comparison with the step-6,500 chart. That chart ranks another run\'s model at another checkpoint, so its ranks describe that model.', '',
        f'![Policy versus measured gap]({figure})', '',
        f'[PDF]({Path(figure).with_suffix(".pdf")}) · [Every tokenizer](tokenizer-ranking.csv) · [Analysis](analysis.json)', '',
        '| Rank | Tokenizer | Algorithm / pretokenizer | Vocab | Policy probability | Mean gap |',
        '| ---: | ---: | --- | ---: | ---: | ---: |']
    for row in rows[:10]:
        lines.append(f'| {row["rank"]} | #{row["index"]} | {row["algorithm"]} / {row["pretokenizer"]} | {row["vocab_size"]:,} | {row["policy_probability"]:.5g} | {row["gap_bpb"]:+.6f} |')
    return '\n'.join(lines) + '\n'


def private_files(report):
    """Report text files holding an absolute path, an email address or the local account name."""
    patterns = [r'(?<![\w.~-])/(?:Users|home|gpfs|mmfs1|private|tmp|var|root|scratch|mnt|opt|projects|data)/',
                r'(?<![\w.])~/', r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+']
    try:
        user = getpass.getuser()
    except Exception:
        user = ''
    if len(user) > 2:
        patterns.append(r'(?<![\w-])' + re.escape(user) + r'(?![\w-])')
    pattern = re.compile('|'.join(patterns))
    return [str(path.relative_to(report)) for path in sorted(report.rglob('*'))
            if path.is_file() and path.suffix in {'.md', '.json', '.csv'} and pattern.search(path.read_text(errors='replace'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state', type=Path, default=ROOT / 'modal_results/rawbpb_pool_ranking_20000',
                        help='Copied sweep results (default modal_results/rawbpb_pool_ranking_20000)')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/rawbpb-pool-ranking-20000',
                        help='Report folder (default reports/rawbpb-pool-ranking-20000)')
    parser.add_argument('--plots', type=Path, default=ROOT / 'plots/rawbpb_pool_ranking_20000',
                        help='Figure path without suffix; .png and .pdf are written (default plots/rawbpb_pool_ranking_20000)')
    parser.add_argument('--step', type=int, default=20000, help='Frozen checkpoint step of every score (default 20000)')
    parser.add_argument('--reference', type=Path, required=True,
                        help="Folder with the step-6,500 report's all-scores.json.gz, pool-index.json, policy.json "
                             "and panel-manifest.json (git cat-file blob origin/zett-modal-20k:reports/regret-pool-ranking-6500/NAME)")
    args = parser.parse_args()
    np.seterr(divide='ignore', over='ignore', invalid='ignore')  # spurious Accelerate matmul flags; see docstring
    STATE, REPORT = args.state, args.report
    manifest=json.loads((STATE/'panel-manifest.json').read_text());policy=json.loads((STATE/'policy.json').read_text())
    if sha256(STATE/'pool-index.json') != manifest['pool_index_sha256']:
        raise ValueError('Pool index metadata changed')
    check_provenance(STATE, manifest, policy, args.step)
    index=json.loads((STATE/'pool-index.json').read_text())
    with gzip.open(STATE/'all-scores.json.gz','rt') as stream:records=json.load(stream)
    check_identity(records, index)
    result,rows,gaps=analyze(records,policy,manifest,args.step)
    extra=analyze_raw_and_reference(records, policy, result, gaps, load_reference(args.reference, index))
    analysis = dict(result, raw_bpb_and_reference=extra)
    try:
        json.dumps(analysis, allow_nan=False)
    except ValueError:
        raise ValueError('Nonfinite number in the analysis') from None
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'analysis.json',analysis)
    with (REPORT/'tokenizer-ranking.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in INPUTS:
        if (STATE/name).exists():shutil.copy2(STATE/name,REPORT/name)
    plot(result,rows,gaps,args.step,args.plots,REPORT)
    (REPORT/'README.md').write_text(readme(result, extra, rows, manifest, args.step, args.plots.with_suffix('.png').name))
    private = private_files(REPORT)
    if private:
        raise ValueError('Absolute path, email address or account name in ' + ', '.join(private) + '; manifest.json not written')
    atomic_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps({k:v for k,v in result['comparisons']['all'].items() if k!='ranks'}))
    print(json.dumps(dict(raw_bpb={k: v for k, v in extra['comparisons']['all'].items() if k != 'ranks'},
                          reference={k: v for k, v in extra['reference'].items() if k not in ('ranks', 'bottom_group', 'files')})))


if __name__=='__main__':main()
