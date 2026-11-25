"""Report the held-out full-pool eval of every raw-BPB save (protocol heldout-pool-4992).

Qingyun's held-out eval (948787e, reports/heldout-pool-sweep on zett-modal-20k) scored all 42
saves of the raw-BPB run (steps 0 to 20,000 every 500, plus 16,409) on all 1,000 pool
tokenizers, on Tillicum H200s (plans/rawbpb-heldout-pool.md). This script summarizes those
scores and sets them beside the complete checkpoints in his public archive: the uniform, regret
and predictor runs, scored on Rift L40S GPUs under the same frozen plan with another checkpoint
list.

Inputs (defaults under modal_results/rawbpb_heldout_pool/):
  scores/                       ranking.csv, complete.json and canonical.json of each of the 42
                                checkpoints, copied from Tillicum
  passage-means.json            heldout_pool_passage_means.py over the Tillicum score folders
  reference-passage-means.json  the same over his archive's run/scores
  sacct.txt                     sacct -X -P -n --format=JobID,JobName,QOS,State,ExitCode,
                                ElapsedRaw,Start,End,AllocTRES for the run's jobs
  hf-upload/archives.json       path, size and SHA-256 of the 42 per-checkpoint score archives
                                uploaded to Hugging Face (SCORES_ON_HUB)
  --reference                   his archive restored by restore_rift_pool_archive.py; its
                                run/scores/<label>/{ranking.csv,complete.json} and
                                run/scoring-panel.json are read
Writes README.md, checkpoint-summary.csv, comparison.csv, domains.csv, groups.csv, jobs.csv,
rankings.csv.gz, passage-means.json.gz, analysis.json, manifest.json and the figure to --report.

Gap is transferred minus canonical BPB; for one tokenizer it is the mean of the nine domain gaps
(his balanced_bpb). The pool mean over the 1,000 tokenizers is linear in the per-passage losses,
so the per-passage pool means give it by domain and allow a passage bootstrap: passages are
resampled within each domain, with the same draws for both sides of a paired difference. The
bootstrap covers passage sampling only, not the GPU type or run-to-run training variation.
"""
import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from package_training_inputs import atomic_json, sha256
from report_rawbpb_pool_ranking import manrope_regular, private_files, spearman

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/rawbpb_heldout_pool'
PICKS = ROOT / 'reports/rawbpb-pool-ranking-20000'  # the run's step-20,000 policy.json and its pool index
OURS = 'raw-bpb'
HIS = ('uniform', 'regret', 'predictor')
PLANS = {OURS: '8d982055d80dbf07cf8430b118bd88b8f41ffae950785a1cbca820cc32d61ae8',
         **dict.fromkeys(HIS, 'd1269e6fee0c5875d475cfda7584b65f3f0c0cc9cb8a6161b518b756627a6964')}
PLATFORM = {OURS: 'Tillicum H200', **dict.fromkeys(HIS, 'Rift L40S')}
NAMES = {OURS: 'raw-BPB', 'uniform': 'uniform', 'regret': 'regret', 'predictor': 'predictor'}
COLORS = {OURS: '#2a78d6', 'uniform': '#eb6834', 'regret': '#1baf7a', 'predictor': '#eda100'}
STEPS = sorted([*range(0, 20001, 500), 16409])
RECEIPT = dict(completed=1000, documents=156, scored_tokens=4992, scored_bytes=14429)
PANEL_SHA256 = 'cf76016fa6adc9b58f9537d70955fc900bee45a3ad611b8ce05782221adca703'
ARCHIVE = dict(dataset='qingy2024/anytok-rift-eval-resume', revision='38176d7d654b33f513dcc7e4f3f19e6bfc142d85',
               file='paused-20261122T143018Z/rift-sweep-resume.tar.gz',
               sha256='7d23d37dc3f4a7554c4b612cb07ac00b5100a19d1686ee7f08ad2c3757bd9631')
# The per-tokenizer score files, one archive per checkpoint, uploaded from Tillicum on 2026-11-24. The Hub
# limits that repo to 20,000 files, too few for the 42,126 loose files.
SCORES_ON_HUB = dict(repo='Dekiofsa/RLtokenizers', revision='fb8f2a00409a98e41ea02337784eda5c2c956e5f',
                     folder='raw-bpb-rl-20k/diagnostics/heldout-pool-4992')
FINAL = 20000
SELECTED = (0, 500, 1000, 2000, 5000, 10000, 15000, 16409, 20000)
MATCHED = (500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)
REPS = 10000
SEED = 20261124
USD_PER_GPU_HOUR = 0.90
# hyakusage for account stf, read by hand on 2026-11-24 before the scoring array and after it ended
HYAKUSAGE = dict(before_array=70.55, after_array=83.19, budget=250.0)
JOB_FIELDS = ['JobID', 'JobName', 'QOS', 'State', 'ExitCode', 'ElapsedRaw', 'Start', 'End', 'AllocTRES']


def label_step(label):
    run, step = label.rsplit('-', 1)
    return run, int(step)


def read_checkpoint(folder, plan):
    """One checkpoint's ranking.csv, checked against its complete.json receipt."""
    label = folder.name
    receipt = json.loads((folder / 'complete.json').read_text())
    if receipt != dict(plan_sha256=plan, checkpoint=label, **RECEIPT):
        raise ValueError(f'{label}: unexpected receipt {receipt}')
    with (folder / 'ranking.csv').open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row['tokenizer']))
    if [int(row['tokenizer']) for row in rows] != list(range(1000)) or {row['checkpoint'] for row in rows} != {label}:
        raise ValueError(f'{label}: ranking.csv must cover the 1,000 tokenizers once')
    canonical = {row['canonical_bpb'] for row in rows}
    if len(canonical) != 1:
        raise ValueError(f'{label}: more than one canonical BPB')
    return dict(label=label, rows=rows, canonical=float(canonical.pop()),
                gap=np.array([float(row['gap_bpb']) for row in rows]),
                transferred=np.array([float(row['transferred_bpb']) for row in rows]))


def load_runs(scores, reference):
    runs = {run: {} for run in (OURS, *HIS)}
    folders = [path for path in sorted(scores.iterdir()) if path.is_dir()]
    folders += [path for path in sorted((reference / 'run' / 'scores').iterdir()) if (path / 'complete.json').is_file()]
    for folder in folders:
        run, step = label_step(folder.name)
        if run not in runs or (run == OURS) != (folder.parent == scores):
            raise ValueError(f'Unexpected checkpoint folder {folder.name}')
        runs[run][step] = read_checkpoint(folder, PLANS[run])
    if sorted(runs[OURS]) != STEPS or not all(runs[run] for run in HIS):
        raise ValueError('Expected the 42 raw-BPB checkpoints and complete checkpoints of each archived run')
    pool = [(row['algorithm'], row['pretokenizer'], int(row['vocab'])) for row in runs[OURS][0]['rows']]
    for checkpoints in runs.values():
        for checkpoint in checkpoints.values():
            if [(row['algorithm'], row['pretokenizer'], int(row['vocab'])) for row in checkpoint['rows']] != pool:
                raise ValueError(f"{checkpoint['label']}: different pool metadata")
    return runs, pool


def domain_groups(reference):
    path = reference / 'run' / 'scoring-panel.json'
    if sha256(path) != PANEL_SHA256:
        raise ValueError('The scoring panel is not the audited heldout-pool-4992 panel')
    panel = json.loads(path.read_text())
    groups = []
    for domain in sorted({row['domain'] for row in panel}):
        indices = np.array([i for i, row in enumerate(panel) if row['domain'] == domain])
        groups.append((domain, indices, np.array([panel[i]['continuation_bytes'] for i in indices], dtype=float)))
    if len(panel) != 156 or len(groups) != 9 or sum(size.sum() for *_, size in groups) != 14429:
        raise ValueError('Expected 156 passages, 9 domains and 14,429 scored bytes')
    return panel, groups


def balanced(values, groups):
    """His balanced_bpb: per-domain byte-weighted BPB, averaged equally over the nine domains."""
    return float(np.mean([values[indices].sum() / (size.sum() * math.log(2)) for _, indices, size in groups]))


def attach_passage_means(runs, path, which, groups):
    """Per-passage pool means, checked to reproduce each checkpoint's ranking.csv."""
    data = json.loads(path.read_text())
    expected = {checkpoint['label'] for run in which for checkpoint in runs[run].values()}
    if set(data) != expected:
        raise ValueError(f'{path.name}: expected exactly the {len(expected)} complete checkpoints')
    for run in which:
        for checkpoint in runs[run].values():
            entry = data[checkpoint['label']]
            canonical = np.array(entry['canonical_nats'])
            gap = np.array(entry['mean_transferred_nats']) - canonical
            if (entry['plan_sha256'] != PLANS[run] or entry['tokenizers'] != 1000 or len(canonical) != 156
                    or abs(balanced(canonical, groups) - checkpoint['canonical']) > 1e-12
                    or abs(balanced(gap, groups) - checkpoint['gap'].mean()) > 1e-9):
                raise ValueError(f"{checkpoint['label']}: passage means disagree with ranking.csv")
            checkpoint['canonical_nats'], checkpoint['gap_nats'] = canonical, gap
    return data


def hardest(gap):
    """His ranking order: largest gap first, ties by tokenizer index."""
    return sorted(range(len(gap)), key=lambda index: (-gap[index], index))


def summary_row(run, step, checkpoint):
    gap = checkpoint['gap']
    order = hardest(gap)
    return dict(run=run, platform=PLATFORM[run], checkpoint=checkpoint['label'], step=step,
                canonical_bpb=checkpoint['canonical'], mean_transferred_bpb=float(checkpoint['transferred'].mean()),
                mean_gap_bpb=float(gap.mean()), median_gap_bpb=float(np.median(gap)),
                p10_gap_bpb=float(np.percentile(gap, 10)), p90_gap_bpb=float(np.percentile(gap, 90)),
                min_gap_bpb=float(gap.min()), max_gap_bpb=float(gap.max()), negative_gaps=int((gap < 0).sum()),
                hardest_tokenizer=order[0], easiest_tokenizer=order[-1])


def bootstrap_draws(groups, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, len(indices), size=(reps, len(indices))) for _, indices, _ in groups]


def resampled(values, groups, draws):
    """Pool-mean gap of each bootstrap replicate, passages resampled within each domain."""
    per_domain = [values[indices][draw].sum(1) / (size[draw].sum(1) * math.log(2))
                  for (_, indices, size), draw in zip(groups, draws)]
    return np.mean(per_domain, axis=0)


def compare(runs, groups, draws):
    rows = []
    for run in HIS:
        for step in sorted(set(runs[run]) & set(runs[OURS])):
            ours, his = runs[OURS][step], runs[run][step]
            replicates = resampled(ours['gap_nats'], groups, draws) - resampled(his['gap_nats'], groups, draws)
            low, high = np.percentile(replicates, [2.5, 97.5])
            shared = set(hardest(ours['gap'])[:100]) & set(hardest(his['gap'])[:100])
            rows.append(dict(step=step, run=run, raw_bpb_mean_gap=float(ours['gap'].mean()),
                             other_mean_gap=float(his['gap'].mean()),
                             difference=float(ours['gap'].mean() - his['gap'].mean()),
                             ci95_low=float(low), ci95_high=float(high),
                             raw_bpb_canonical_bpb=ours['canonical'], other_canonical_bpb=his['canonical'],
                             spearman=spearman(ours['gap'], his['gap']), hardest100_shared=len(shared)))
    return rows


def domain_rows(runs, groups):
    rows = []
    for run, checkpoints in runs.items():
        for step, checkpoint in sorted(checkpoints.items()):
            for domain, indices, size in groups:
                scale = size.sum() * math.log(2)
                rows.append(dict(run=run, step=step, domain=domain, passages=len(indices), scored_bytes=int(size.sum()),
                                 canonical_bpb=float(checkpoint['canonical_nats'][indices].sum() / scale),
                                 pool_mean_gap_bpb=float(checkpoint['gap_nats'][indices].sum() / scale)))
    return rows


def run_picks(reference):
    """How often the raw-BPB run drew each pool tokenizer over its 20,000 updates (its step-20,000 policy.json)."""
    plan = json.loads((reference / 'plan.json').read_text())
    if sha256(PICKS / 'pool-index.json') != plan['pool_index_sha256']:
        raise ValueError('The pick counts index a different pool')
    policy = json.loads((PICKS / 'policy.json').read_text())
    counts = np.array([int(count) for count in policy['counts']])
    if policy['step'] != FINAL or len(counts) != 1000 or counts.sum() != FINAL:
        raise ValueError('Expected the 20,000 picks of the step-20,000 policy')
    return counts


def group_rows(runs, pool, picks, step=FINAL):
    rows = []
    for algorithm, pretokenizer in sorted({(a, p) for a, p, _ in pool}):
        members = [i for i, (a, p, _) in enumerate(pool) if (a, p) == (algorithm, pretokenizer)]
        ours, uniform = runs[OURS][step]['gap'][members].mean(), runs['uniform'][step]['gap'][members].mean()
        rows.append(dict(algorithm=algorithm, pretokenizer=pretokenizer, tokenizers=len(members), step=step,
                         pool_share=len(members) / len(pool), update_share=float(picks[members].sum() / picks.sum()),
                         raw_bpb_mean_gap=float(ours), uniform_mean_gap=float(uniform), difference=float(ours - uniform)))
    return sorted(rows, key=lambda row: (-row['raw_bpb_mean_gap'], row['algorithm'], row['pretokenizer']))


def hardest_rows(runs, pool, step=FINAL, count=10):
    ours, uniform = runs[OURS][step]['gap'], runs['uniform'][step]['gap']
    uniform_rank = {index: rank for rank, index in enumerate(hardest(uniform), 1)}
    return [dict(rank=rank, tokenizer=index, algorithm=pool[index][0], pretokenizer=pool[index][1], vocab=pool[index][2],
                 gap=float(ours[index]), uniform_rank=uniform_rank[index], uniform_gap=float(uniform[index]))
            for rank, index in enumerate(hardest(ours)[:count], 1)]


def stability(runs):
    ours = runs[OURS]
    consecutive = [dict(previous=a, step=b, spearman=spearman(ours[a]['gap'], ours[b]['gap']))
                   for a, b in zip(STEPS, STEPS[1:])]
    later = [row['spearman'] for row in consecutive if row['previous'] >= 1000]
    return dict(consecutive=consecutive, first_step=consecutive[0]['spearman'], from_1000_min=min(later),
                from_1000_median=float(np.median(later)),
                step10000_vs_20000=spearman(ours[10000]['gap'], ours[20000]['gap']),
                step0_vs_20000=spearman(ours[0]['gap'], ours[20000]['gap']))


def read_jobs(path):
    rows = [dict(zip(JOB_FIELDS, line.split('|'))) for line in path.read_text().splitlines() if line.strip()]
    array = [row for row in rows if row['JobID'].startswith('518632_')]
    setup = [row for row in rows if '_' not in row['JobID']]
    if (len(array) != 42 or len(setup) != 5 or any(row['State'] != 'COMPLETED' or row['ExitCode'] != '0:0' for row in rows)
            or any('gres/gpu=1' not in row['AllocTRES'] for row in rows)):
        raise ValueError('Expected 5 setup jobs and 42 array tasks, each completed on one GPU')
    hours = {name: sum(int(row['ElapsedRaw']) for row in group) / 3600 for name, group in (('array', array), ('setup', setup))}
    elapsed = sorted(int(row['ElapsedRaw']) for row in array)
    return rows, dict(array_job='518632', tasks=len(array), setup_jobs=len(setup), array_gpu_hours=hours['array'],
                      setup_gpu_hours=hours['setup'], array_usd=hours['array'] * USD_PER_GPU_HOUR,
                      setup_usd=hours['setup'] * USD_PER_GPU_HOUR, task_seconds_min=elapsed[0], task_seconds_max=elapsed[-1],
                      first_start=min(row['Start'] for row in array), last_end=max(row['End'] for row in array),
                      hyakusage=HYAKUSAGE)


def hub_archives(path):
    """The upload's record of the 42 score archives: each checkpoint's folder (1,003 files) in SCORES_ON_HUB."""
    record = json.loads(path.read_text())
    labels = [f'{OURS}-{step:06d}' for step in STEPS]
    if sorted(record) != labels or any(entry['path'] != f"{SCORES_ON_HUB['folder']}/{label}.tar.gz"
                                       or entry['files'] != 1003 for label, entry in record.items()):
        raise ValueError(f'{path} does not describe one archive per checkpoint')
    return {label: dict(size=record[label]['size'], sha256=record[label]['sha256']) for label in labels}


def plot(runs, comparison, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = tempfile.TemporaryDirectory()  # holds the Regular instance until the figure is saved
    font_manager.fontManager.addfont(str(manrope_regular(fonts.name)))
    ink, secondary, grid, axis, surface = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
    plt.rcParams.update({'font.family': 'Manrope', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': axis, 'axes.labelcolor': secondary, 'xtick.color': secondary,
                         'ytick.color': secondary, 'axes.titlecolor': ink, 'axes.titlesize': 10, 'axes.grid': True,
                         'grid.color': grid, 'grid.linewidth': 0.6, 'figure.facecolor': surface,
                         'axes.facecolor': surface, 'savefig.facecolor': surface, 'legend.frameon': False,
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    spec = fig.add_gridspec(2, 2)
    top, diff, canon = fig.add_subplot(spec[0, :]), fig.add_subplot(spec[1, 0]), fig.add_subplot(spec[1, 1])
    steps = sorted(runs[OURS])
    top.fill_between(steps, [np.percentile(runs[OURS][s]['gap'], 10) for s in steps],
                     [np.percentile(runs[OURS][s]['gap'], 90) for s in steps], color=COLORS[OURS], alpha=0.14, lw=0,
                     label='raw-BPB, 10th to 90th percentile over tokenizers')
    ends = {OURS: -8, 'uniform': 8, 'regret': 8, 'predictor': -8}
    for run in (OURS, *HIS):
        steps = sorted(runs[run])
        mean = [runs[run][s]['gap'].mean() for s in steps]
        top.plot(steps, mean, color=COLORS[run], lw=1.6, zorder=3 if run == OURS else 2,
                 label=f'{NAMES[run]} ({PLATFORM[run]})')
        top.annotate(NAMES[run], (steps[-1], mean[-1]), xytext=(5, ends[run]), textcoords='offset points',
                     va='center', fontsize=8, color=ink)
    top.set(title='Mean gap over the 1,000 pool tokenizers', xlabel='Training step',
            ylabel='Mean gap (transferred minus canonical), BPB', xlim=(0, 21400), ylim=(0, None))
    top.legend(loc='upper right')
    offsets = {'uniform': 0, 'regret': -150, 'predictor': 150}
    for run in HIS:
        rows = [row for row in comparison if row['run'] == run]
        x = np.array([row['step'] for row in rows]) + offsets[run]
        y = np.array([row['difference'] for row in rows])
        error = np.array([y - [row['ci95_low'] for row in rows], [row['ci95_high'] for row in rows] - y])
        diff.errorbar(x, y, yerr=error, fmt='o', ms=4, color=COLORS[run], elinewidth=1, capsize=0,
                      label=f'raw-BPB minus {run}')
    diff.axhline(0, color=axis, lw=1)
    diff.set(title='Raw-BPB minus each archived run at the same step', xlabel='Training step',
             ylabel='Difference in mean gap, BPB (95% passage bootstrap)')
    diff.legend(loc='upper right')
    for run in (OURS, *HIS):
        steps = sorted(runs[run])
        canon.plot(steps, [runs[run][s]['canonical'] for s in steps], color=COLORS[run], lw=1.6, label=NAMES[run],
                   zorder=3 if run == OURS else 2)
    canon.set(title='Canonical BPB on the same 156 passages', xlabel='Training step', ylabel='Canonical BPB')
    canon.legend(loc='upper right')
    for axes in (top, diff, canon):
        axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f'{value:,.0f}'))
    fig.savefig(stem.with_suffix('.png'), dpi=160, metadata={'Software': None})
    fig.savefig(stem.with_suffix('.pdf'), metadata={'CreationDate': None})
    plt.close(fig)


def signed(value):
    return f'{value:+.4f}'


def interval(row):
    return f"{signed(row['difference'])} [{signed(row['ci95_low'])}, {signed(row['ci95_high'])}]"


def readme(analysis, summary, comparison, groups, domains, hardest10, jobs, figure):
    ours = {row['step']: row for row in summary if row['run'] == OURS}
    by_run = {(row['run'], row['step']): row for row in summary}
    versus = {(row['run'], row['step']): row for row in comparison}
    first, last = ours[0], ours[FINAL]
    uniform = [row for row in comparison if row['run'] == 'uniform']
    above = [row for row in uniform if row['ci95_low'] > 0]
    below = [row for row in uniform if row['ci95_high'] < 0]
    largest = max(uniform, key=lambda row: row['difference'])
    smallest = min(uniform, key=lambda row: row['difference'])
    stable = analysis['stability']
    final_uniform = versus[('uniform', FINAL)]
    canon5000 = {run: by_run[(run, 5000)]['canonical_bpb'] for run in (OURS, *HIS)}
    canon_around = {run: (by_run[(run, 4000)]['canonical_bpb'], by_run[(run, 6000)]['canonical_bpb']) for run in HIS}
    if not all(canon5000[run] > max(canon_around[run]) for run in HIS) or canon5000[OURS] > canon5000['uniform']:
        raise ValueError('The step-5,000 paragraph no longer describes the data')
    if any(row['negative_gaps'] for row in ours.values()):
        raise ValueError('Some tokenizer beats canonical; the Result paragraph no longer describes the data')
    lower = [row for row in groups if row['difference'] < 0]
    higher = [row for row in groups if row['difference'] > 0]
    worst = max(groups, key=lambda row: row['difference'])

    def name(row):
        return f"{row['algorithm']} / {row['pretokenizer']}"
    lines = [
        '# Held-out full-pool eval of the raw-BPB run',
        '',
        "Qingyun's held-out full-pool eval (**heldout-pool-4992**, commit 948787e, reports/heldout-pool-sweep on "
        'zett-modal-20k) on **every save of the raw-BPB run**: steps 0 to 20,000 every 500 plus 16,409, so '
        '**42 checkpoints, each scored on all 1,000 pool tokenizers**, on Tillicum H200s on 2026-11-24. Every '
        'tokenizer is scored on the same 156 frozen passages from nine domains, 32 canonical continuation tokens '
        'each (4,992 tokens and 14,429 bytes), with its own native prefix, the hypernetwork\'s dropout off and one '
        'generated embedding table per tokenizer. A tokenizer\'s **gap** is its transferred BPB minus the '
        "checkpoint's canonical BPB, averaged over the nine domains. Only the checkpoint list differs from his "
        f"frozen plan (his `{PLANS['uniform'][:12]}…`, this run's `{PLANS[OURS][:12]}…`).",
        '',
        '## Result',
        '',
        f"The mean gap over the 1,000 tokenizers falls from **{signed(first['mean_gap_bpb'])} BPB** at step 0 to "
        f"**{signed(last['mean_gap_bpb'])} BPB** at step 20,000. No tokenizer matches canonical at any save: the "
        f"smallest gap at step 20,000 is {signed(last['min_gap_bpb'])} (#{last['easiest_tokenizer']}) and the largest "
        f"{signed(last['max_gap_bpb'])} (#{last['hardest_tokenizer']}). Canonical BPB on the same passages falls from "
        f"{first['canonical_bpb']:.4f} to {last['canonical_bpb']:.4f}, so the gap shrinks while the canonical "
        'baseline it is measured against also improves.',
        '',
        '| Step | Canonical BPB | Mean gap | 10th percentile | 90th percentile | Largest gap |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {s:,} | {ours[s]['canonical_bpb']:.4f} | {signed(ours[s]['mean_gap_bpb'])} | {signed(ours[s]['p10_gap_bpb'])} "
          f"| {signed(ours[s]['p90_gap_bpb'])} | {signed(ours[s]['max_gap_bpb'])} (#{ours[s]['hardest_tokenizer']}) |"
          for s in SELECTED],
        '',
        'Every save is in [checkpoint-summary.csv](checkpoint-summary.csv).',
        '',
        f'![Mean gap, differences from the archived runs, and canonical BPB]({figure}.png)',
        '',
        f'[PDF]({figure}.pdf)',
        '',
        "## Against Qingyun's runs on the same eval",
        '',
        f"His public archive ({ARCHIVE['dataset']} at revision {ARCHIVE['revision'][:12]}…) holds complete scores of "
        f"three runs under the same frozen plan: **uniform** ({len(analysis['archived']['uniform'])} saves, 500 to "
        f"20,000), **regret** ({len(analysis['archived']['regret'])} saves, 500 to 8,000) and **predictor** "
        f"({len(analysis['archived']['predictor'])} saves, 500 to 7,000). They were scored on Rift L40S GPUs and "
        'this run on H200s; his protocol notes that another GPU type can shift BPB slightly.',
        '',
        '| Step | Raw-BPB | Uniform | Regret | Predictor | Raw-BPB minus uniform [95% interval] |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {s:,} | {signed(ours[s]['mean_gap_bpb'])} | "
          + ' | '.join(signed(by_run[(run, s)]['mean_gap_bpb']) if (run, s) in by_run else '' for run in HIS)
          + f" | {interval(versus[('uniform', s)])} |" for s in MATCHED],
        '',
        f"Over the {len(uniform)} matching saves, raw-BPB minus uniform runs from {signed(smallest['difference'])} "
        f"(step {smallest['step']:,}) to {signed(largest['difference'])} (step {largest['step']:,}). The 95% passage "
        f"bootstrap interval lies above zero at {len(above)} steps and below zero at {len(below)}. At step 20,000 "
        f"the difference is {interval(final_uniform)}. Per-tokenizer gaps at step 20,000 rank the pool similarly in "
        f"both runs (Spearman {final_uniform['spearman']:.3f}); their hardest 100 share "
        f"{final_uniform['hardest100_shared']} tokenizers. Every comparison is in [comparison.csv](comparison.csv).",
        '',
        f"At step 5,000 all three archived runs have canonical BPB between {min(canon5000[run] for run in HIS):.4f} and "
        f"{max(canon5000[run] for run in HIS):.4f}, above their own steps 4,000 and 6,000 (for example uniform "
        f"{canon_around['uniform'][0]:.4f} and {canon_around['uniform'][1]:.4f}); this run has "
        f"{canon5000[OURS]:.4f}. Their lower mean gaps at that step come with that higher canonical BPB.",
        '',
        '## Where the gap remains at step 20,000',
        '',
        '| Algorithm / pretokenizer | Tokenizers | Share of the run\'s updates | Raw-BPB mean gap | Uniform mean gap '
        '| Raw-BPB minus uniform |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {row['algorithm']} / {row['pretokenizer']} | {row['tokenizers']} | {100 * row['update_share']:.1f}% | "
          f"{signed(row['raw_bpb_mean_gap'])} | {signed(row['uniform_mean_gap'])} | {signed(row['difference'])} |"
          for row in groups],
        '',
        f"Raw-BPB is below uniform in {len(lower)} groups and above it in {len(higher)} "
        f"({', '.join(name(row) for row in higher)}), most for {name(worst)} ({signed(worst['difference'])}). The "
        f"{len(lower)} lower groups are {100 * sum(row['pool_share'] for row in lower):.1f}% of the pool and drew "
        f"{100 * sum(row['update_share'] for row in lower):.1f}% of the run's 20,000 updates; the {len(higher)} higher "
        f"ones are {100 * sum(row['pool_share'] for row in higher):.1f}% of the pool and drew "
        f"{100 * sum(row['update_share'] for row in higher):.1f}%. Update shares come from the run's pick counts in "
        'reports/rawbpb-pool-ranking-20000/policy.json, which indexes the same pool.',
        '',
        'Pool-mean gap by domain (byte-weighted within the domain):',
        '',
        '| Domain | Passages | Raw-BPB step 0 | Raw-BPB step 10,000 | Raw-BPB step 20,000 | Uniform step 20,000 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {domain} | {domains[(OURS, 0, domain)]['passages']} | "
          + ' | '.join(signed(domains[key]['pool_mean_gap_bpb'])
                       for key in ((OURS, 0, domain), (OURS, 10000, domain), (OURS, FINAL, domain), ('uniform', FINAL, domain)))
          + ' |' for domain in analysis['domains']],
        '',
        'The ten tokenizers with the largest gap at step 20,000, with their place in the uniform run at step 20,000:',
        '',
        '| Rank | Tokenizer | Algorithm / pretokenizer | Vocab | Gap | Uniform rank | Uniform gap |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: |',
        *[f"| {row['rank']} | #{row['tokenizer']} | {row['algorithm']} / {row['pretokenizer']} | {row['vocab']:,} | "
          f"{signed(row['gap'])} | {row['uniform_rank']} | {signed(row['uniform_gap'])} |" for row in hardest10],
        '',
        'Ranks: 1 = largest gap of the 1,000. Every tokenizer of every save is in [rankings.csv.gz](rankings.csv.gz).',
        '',
        '## How the ranking moves during training',
        '',
        f"Spearman correlation of the per-tokenizer gaps between consecutive saves is {stable['first_step']:.3f} from "
        f"step 0 to 500, then at least {stable['from_1000_min']:.3f} between consecutive saves from step 1,000 on "
        f"(median {stable['from_1000_median']:.3f}). Step 10,000 against step 20,000: {stable['step10000_vs_20000']:.3f}; "
        f"step 0 against step 20,000: {stable['step0_vs_20000']:.3f}.",
        '',
        '## Checks',
        '',
        '- Every one of the 42,000 per-tokenizer score files was read on Tillicum: each carries this run\'s plan hash, '
        'its checkpoint, its index and its pool tokenizer\'s SHA-256, 156 finite non-negative passage losses, and a '
        "BPB and gap that recompute from those losses. Each checkpoint's ranking.csv lists the 1,000 tokenizers once "
        'in his order and matches those files, and each complete.json is his receipt (1,000 tokenizers, 156 passages, '
        '4,992 tokens, 14,429 bytes).',
        '- The scoring panel on Tillicum is byte-identical to the one in his archive, and the per-passage pool means '
        'used here reproduce every checkpoint\'s mean gap and canonical BPB.',
        f"- Slurm array {jobs['array_job']}: {jobs['tasks']} tasks, all COMPLETED with exit code 0, "
        f"{jobs['task_seconds_min'] / 60:.0f} to {jobs['task_seconds_max'] / 60:.0f} minutes each, one H200 per task. "
        'The logs hold only model-loading progress bars and one Triton message per task (Python.h missing when '
        'Triton builds its driver helper); plans/rawbpb-heldout-pool.md explains why that message cannot change a score.',
        f"- Cost: {jobs['array_gpu_hours']:.2f} GPU-hours, ${jobs['array_usd']:.2f} for the array, plus "
        f"${jobs['setup_usd']:.2f} for the {jobs['setup_jobs']} setup jobs (smoke test, panel audit, preflight and two "
        f"diagnostics). hyakusage for the stf account moved from ${HYAKUSAGE['before_array']:.2f} to "
        f"${HYAKUSAGE['after_array']:.2f} of ${HYAKUSAGE['budget']:.0f} over the array. Every job is in [jobs.csv](jobs.csv).",
        '',
        '## Limits',
        '',
        '- One fixed panel. The intervals resample its passages within each domain; they do not cover run-to-run '
        'training variation (each curriculum here is a single run) or the GPU type.',
        '- The archived runs were scored on L40S GPUs and this run on H200s. The size of that shift was not measured, '
        'so differences of a few ten-thousandths of a BPB between runs should not be read as real.',
        "- Each gap is measured against its own checkpoint's canonical BPB, which changes during training, so a "
        'smaller gap can come from better transfer or from a worse canonical baseline.',
        '',
        '## Files',
        '',
        '- [checkpoint-summary.csv](checkpoint-summary.csv): one row per checkpoint of all four runs.',
        '- [comparison.csv](comparison.csv): raw-BPB against each archived run at every matching step, with the '
        'bootstrap interval, Spearman correlation and shared hardest 100.',
        '- [domains.csv](domains.csv) and [groups.csv](groups.csv): the pool-mean gap by domain for every checkpoint, '
        'and by algorithm and pretokenizer at step 20,000.',
        '- [rankings.csv.gz](rankings.csv.gz): the 42 ranking.csv files of this run, concatenated.',
        '- [passage-means.json.gz](passage-means.json.gz): the panel\'s passage domains and bytes, and each '
        "checkpoint's canonical and pool-mean transferred losses per passage, for all four runs.",
        '- [analysis.json](analysis.json) holds every number above; [manifest.json](manifest.json) the inputs\' and '
        'files\' hashes.',
        '- The per-tokenizer score files (losses for every passage of every pair) are on Hugging Face, one archive '
        f"per checkpoint: [{SCORES_ON_HUB['folder']}](https://huggingface.co/{SCORES_ON_HUB['repo']}/tree/"
        f"{SCORES_ON_HUB['revision']}/{SCORES_ON_HUB['folder']}) in {SCORES_ON_HUB['repo']}, revision "
        f"{SCORES_ON_HUB['revision'][:8]}. Each raw-bpb-NNNNNN.tar.gz unpacks to the scorer's folder "
        'raw-bpb-NNNNNN/; [manifest.json](manifest.json) lists their SHA-256.',
        '',
        'Made by scripts/report_rawbpb_heldout_pool.py from the scores copied off Tillicum and his archive.',
        '',
    ]
    return '\n'.join(lines)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gzip_bytes(path, payload):
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state', type=Path, default=STATE, help='Copied results (default modal_results/rawbpb_heldout_pool)')
    parser.add_argument('--reference', type=Path, required=True, help="His restored archive folder")
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/rawbpb-heldout-pool',
                        help='Report folder (default reports/rawbpb-heldout-pool)')
    args = parser.parse_args()
    runs, pool = load_runs(args.state / 'scores', args.reference)
    panel, groups = domain_groups(args.reference)
    means = {**attach_passage_means(runs, args.state / 'passage-means.json', (OURS,), groups),
             **attach_passage_means(runs, args.state / 'reference-passage-means.json', HIS, groups)}
    draws = bootstrap_draws(groups)
    summary = [summary_row(run, step, checkpoint) for run, checkpoints in runs.items()
               for step, checkpoint in sorted(checkpoints.items())]
    comparison = compare(runs, groups, draws)
    domains = domain_rows(runs, groups)
    groups_table = group_rows(runs, pool, run_picks(args.reference))
    hardest10 = hardest_rows(runs, pool)
    job_rows, jobs = read_jobs(args.state / 'sacct.txt')
    archives = hub_archives(args.state / 'hf-upload' / 'archives.json')
    analysis = dict(protocol='heldout-pool-4992', plans=PLANS, platform=PLATFORM, archive=ARCHIVE,
                    panel_sha256=PANEL_SHA256, bootstrap=dict(reps=REPS, seed=SEED, resample='passages within domain'),
                    domains=[domain for domain, *_ in groups], archived={run: sorted(runs[run]) for run in HIS},
                    summary=summary, comparison=comparison, groups=groups_table, hardest=hardest10,
                    stability=stability(runs), jobs=jobs)
    numbers = json.dumps(analysis)
    if any(token in numbers for token in ('NaN', 'Infinity')):
        raise ValueError('Every number in analysis.json must be finite')
    report = args.report
    report.mkdir(parents=True, exist_ok=True)
    figure = 'rawbpb_heldout_pool'
    plot(runs, comparison, report / figure)
    write_csv(report / 'checkpoint-summary.csv', summary)
    write_csv(report / 'comparison.csv', comparison)
    write_csv(report / 'domains.csv', domains)
    write_csv(report / 'groups.csv', groups_table)
    write_csv(report / 'jobs.csv', [{key: row[key] for key in JOB_FIELDS[:-1]} for row in job_rows])
    rankings = io.StringIO()
    writer = csv.DictWriter(rankings, fieldnames=list(runs[OURS][0]['rows'][0]), lineterminator='\r\n')
    writer.writeheader()
    for step in STEPS:
        writer.writerows(sorted(runs[OURS][step]['rows'], key=lambda row: int(row['rank'])))
    gzip_bytes(report / 'rankings.csv.gz', rankings.getvalue().encode())
    gzip_bytes(report / 'passage-means.json.gz', json.dumps(dict(
        panel=[dict(id=row['id'], domain=row['domain'], continuation_bytes=row['continuation_bytes']) for row in panel],
        checkpoints=means), sort_keys=True).encode())
    atomic_json(report / 'analysis.json', analysis)
    domain_index = {(row['run'], row['step'], row['domain']): row for row in domains}
    (report / 'README.md').write_text(readme(analysis, summary, comparison, groups_table, domain_index, hardest10, jobs,
                                             figure))
    leaks = private_files(report)
    if leaks:
        raise ValueError(f'Report files hold a private path, email or account name: {leaks}')
    inputs = {name: sha256(args.state / name)
              for name in ('passage-means.json', 'reference-passage-means.json', 'sacct.txt', 'hf-upload/archives.json')}
    inputs.update({f'rawbpb-pool-ranking-20000/{name}': sha256(PICKS / name) for name in ('policy.json', 'pool-index.json')})
    inputs['receipts'] = {label: sha256(args.state / 'scores' / label / 'complete.json')
                          for label in sorted(p.name for p in (args.state / 'scores').iterdir() if p.is_dir())}
    outputs = {path.name: sha256(path) for path in sorted(report.iterdir()) if path.name != 'manifest.json'}
    scripts = {name: sha256(ROOT / 'scripts' / name)
               for name in ('report_rawbpb_heldout_pool.py', 'heldout_pool_passage_means.py')}
    atomic_json(report / 'manifest.json', dict(inputs=inputs, outputs=outputs, scripts=scripts, archive=ARCHIVE,
                                               panel_sha256=PANEL_SHA256,
                                               scores_on_hub=dict(SCORES_ON_HUB, archives=archives)))
    final = {row['run']: row for row in comparison if row['step'] == FINAL}
    print(json.dumps(dict(report=str(report.relative_to(ROOT)), checkpoints=len(summary), comparisons=len(comparison),
                          raw_bpb_final=analysis['summary'][STEPS.index(FINAL)]['mean_gap_bpb'],
                          minus_uniform_final=final['uniform']['difference'])))


if __name__ == '__main__':
    main()
