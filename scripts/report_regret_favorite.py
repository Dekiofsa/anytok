"""Validate identical scored examples and plot the policy favorite's learning."""
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_favorite_trajectory'
REPORT = ROOT / 'reports/regret-favorite-trajectory'
PLOT = ROOT / 'plots/regret_favorite_trajectory'


def analyze(records, inventory):
    records = sorted(records, key=lambda row: row['step'])
    steps = [row['step'] for row in records]
    if not records or len(set(steps)) != len(steps) or not set(steps) <= set(inventory['steps']):
        raise ValueError('Missing, duplicated or unexpected trajectory checkpoint')
    reference = {row['step']: row for row in inventory['checkpoints']}
    proofs = None; gpu = None; torch = None; identity = None; previous_count = -1
    for row in records:
        if (row['tokenizer_index'] != 771 or len(row['batches']) != 32
                or row['panel_sha256'] != inventory['panel_sha256']
                or row['source_manifest_sha256'] != reference[row['step']]['manifest_sha256']
                or not row['frozen_parameter_versions_unchanged']):
            raise ValueError('Incomplete or inconsistent trajectory scoring')
        if (not np.isfinite(row['policy_probability']) or not 0 <= row['policy_probability'] <= 1
                or not previous_count <= row['cumulative_selections'] <= row['step']):
            raise ValueError('Invalid policy probability or selection counts')
        previous_count = row['cumulative_selections']
        current_identity = (row['tokenizer_sha256'], row['worker_sha256'], row['staged_source_manifest_sha256'])
        if identity is not None and current_identity != identity:
            raise ValueError('Tokenizer identity or scoring implementation changed')
        identity = current_identity
        signature = []
        for index, batch in enumerate(row['batches']):
            if (batch['index'] != index or batch['valid_examples'] != 128
                    or batch['maximum_positions'] > 128
                    or batch['dropout_seed'] != 2026111500 + index
                    or batch['scored_bytes'] <= 0 or batch['scored_tokens'] <= 0
                    or abs(batch['raw_bpb'] - batch['canonical_bpb'] - batch['gap_bpb']) > 1e-12):
                raise ValueError('Batch protocol mismatch')
            signature.append(tuple(batch[key] for key in ['paired_batch_sha256', 'scored_tokens',
                'scored_bytes', 'valid_examples', 'maximum_positions', 'rejected_examples', 'dropout_seed']))
        if proofs is not None and signature != proofs:
            raise ValueError('Scored contexts, targets, token counts or bytes changed across checkpoints')
        if gpu is not None and (row['gpu'] != gpu or row['torch'] != torch):
            raise ValueError('Local hardware/software changed within the trajectory')
        proofs = signature; gpu = row['gpu']; torch = row['torch']
    scores = {key: np.asarray([[batch[key] for batch in row['batches']] for row in records])
              for key in ['raw_bpb', 'canonical_bpb', 'gap_bpb']}
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Nonfinite scores; no batch may be excluded')
    rng = np.random.default_rng(20261115)
    draws = rng.integers(0, 32, size=(10000, 32))
    table = []
    for index, row in enumerate(records):
        summary = dict(step=row['step'], policy_probability=row['policy_probability'],
                       cumulative_selections=row['cumulative_selections'])
        for key, values in scores.items():
            average = values[index].mean()
            bounds = np.quantile(values[index][draws].mean(1), [.025, .975])
            summary[key] = float(average)
            summary[key + '_low'], summary[key + '_high'] = map(float, bounds)
        table.append(summary)
    first, last = steps[0], steps[-1]
    result = dict(tokenizer=771, checkpoints_scored=len(steps), checkpoints_total=len(inventory['steps']),
        complete=steps == inventory['steps'], first_step=first, last_step=last,
        gpu=gpu, torch=torch, examples=4096, batches=32, maximum_positions=128,
        scored_tokens=sum(batch['scored_tokens'] for batch in records[0]['batches']),
        scored_bytes=sum(batch['scored_bytes'] for batch in records[0]['batches']),
        panel_sha256=inventory['panel_sha256'], scored_examples_identical_across_checkpoints=True,
        aggregation='arithmetic mean of per-update byte-weighted BPB; same as policy reward',
        dropout='active; same fixed independent per-batch seeds across every checkpoint', endpoint_change={})
    for key, values in scores.items():
        differences = values[-1] - values[0]
        bounds = np.quantile(differences[draws].mean(1), [.025, .975])
        result['endpoint_change'][key] = dict(last_minus_first=float(differences.mean()),
            paired_batch_bootstrap_95pct=bounds.tolist())
    result['latest_minus_each_checkpoint'] = {}
    for index, step in enumerate(steps):
        changes = {}
        for key, values in scores.items():
            differences = values[-1] - values[index]
            changes[key] = dict(mean=float(differences.mean()),
                paired_batch_bootstrap_95pct=np.quantile(differences[draws].mean(1), [.025, .975]).tolist())
        result['latest_minus_each_checkpoint'][str(step)] = changes
    return result, table, scores, draws


def plot(result, table, scores, draws):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT / 'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    steps = np.array([row['step'] for row in table])
    green, blue, purple, grey = '#36967B', '#2F8FD0', '#8657BC', '#87919D'
    ax = axes[0, 0]
    for key, label, color in [('raw_bpb','Transferred #771',green), ('canonical_bpb','Trained canonical',grey)]:
        ax.plot(steps, scores[key].mean(1), '-o', ms=3.5, color=color, label=label)
    ax.set(title='A · Is #771 becoming easier on fixed text?', ylabel='Absolute BPB · lower is better')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    means = scores['gap_bpb'].mean(1)
    bounds = np.quantile(scores['gap_bpb'][:, draws].mean(2), [.025, .975], axis=1)
    ax.plot(steps, means, '-o', ms=3.5, color=purple)
    ax.fill_between(steps, bounds[0], bounds[1], color=purple, alpha=.12, linewidth=0)
    ax.axhline(0, color=grey, ls=':', lw=.8)
    ax.set(title='B · Is its transfer gap shrinking?', ylabel='Transferred − trained canonical · BPB')
    ax = axes[1, 0]
    for key, label, color in [('raw_bpb','Transferred #771',green), ('canonical_bpb','Trained canonical',grey)]:
        values = scores[key] - scores[key][0]
        bounds = np.quantile(values[:, draws].mean(2), [.025, .975], axis=1)
        ax.plot(steps, values.mean(1), '-o', ms=3.5, color=color, label=label)
        ax.fill_between(steps,bounds[0],bounds[1],color=color,alpha=.12,linewidth=0)
    ax.axhline(0, color=grey, ls=':', lw=.8)
    ax.set(title=f'C · Change from step {steps[0]:,} on identical batches', ylabel='BPB change · negative means improvement')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    ax.plot(steps,[row['policy_probability'] for row in table],'-o',ms=3.5,color=blue,label='Saved selection probability')
    if any(row.get('last_500_selection_fraction') is not None for row in table):
        ax.plot(steps,[row.get('last_500_selection_fraction',np.nan) for row in table],
                '-o',ms=3.5,color=green,label='Fraction of previous 500 updates')
    ax.set(title='D · How much training did the policy assign to #771?', ylabel='Selection probability / update fraction', ylim=(0,1.03))
    ax.legend(frameon=False,fontsize=8)
    ax.text(.02,.97,f"Selected {table[-1]['cumulative_selections']:,} times by step {steps[-1]:,}",
            transform=ax.transAxes,va='top',fontsize=8,color='#525C67')
    for ax in axes.flat:
        ax.set(xlabel='Joint training step',xlim=(400,8100))
        ax.axvline(5288,color='#D5DCE3',ls=':',lw=.8)
        ax.grid(axis='y',color='#E9EDF1',lw=.6);ax.set_axisbelow(True)
    prefix = 'Complete' if result['complete'] else f"{len(table)}/{result['checkpoints_total']} checkpoints"
    fig.suptitle(f'Tokenizer #771 · WordPiece / no regex / 127k · {prefix}',fontsize=13)
    fig.supxlabel('32 fixed batches × 128 examples · variable continuations · 128 total positions · active dropout\nBands: 95% batch bootstrap; dotted vertical line: Modal → Tillicum continuation',fontsize=8)
    PLOT.parent.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:
        target=PLOT.with_suffix('.'+suffix);temporary=target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(temporary,dpi=170,facecolor='white');temporary.replace(target)
        shutil.copy2(target,REPORT/target.name)
    plt.close(fig)


def main():
    inventory=json.loads((STATE/'inventory.json').read_text())
    if sha256(STATE/'training-events.jsonl') != inventory['training_events_sha256']:
        raise ValueError('Committed training history changed')
    records=[json.loads(path.read_text()) for path in sorted((STATE/'scores').glob('*.json'))]
    result,table,scores,draws=analyze(records,inventory)
    history={row['step']:row for row in map(json.loads,(STATE/'training-events.jsonl').read_text().splitlines())
             if row.get('stage')=='train'}
    for row in table:
        window=[history[step] for step in range(max(1,row['step']-499),row['step']+1) if step in history]
        if len(window)!=min(row['step'],500):
            raise ValueError('Missing committed training updates for selection frequency')
        row['last_500_selection_fraction']=sum(r['tokenizer_pool_index']==771 for r in window)/len(window)
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'analysis.json',result)
    atomic_json(REPORT/'checkpoint-sources.json',inventory)
    with (REPORT/'learning-curve.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with gzip.open(REPORT/'all-batch-scores.json.gz','wt') as stream:json.dump(records,stream)
    shutil.copy2(STATE/'panel-manifest.json',REPORT/'panel-manifest.json')
    plot(result,table,scores,draws)
    first,last=table[0],table[-1]
    lines=['# Does the model improve on the tokenizer the policy keeps selecting?','',
        f"**{'Complete' if result['complete'] else 'In progress'}: {len(table)} of {len(inventory['steps'])} saved checkpoints scored.** Tokenizer **#771** is the original **WordPiece / no regex / 127k** pool tokenizer. No model or policy updates occur in this evaluation.",'',
        f"On these identical batches, transferred BPB changes from **{first['raw_bpb']:.6f}** at step **{first['step']:,}** to **{last['raw_bpb']:.6f}** at **{last['step']:,}**. Its canonical gap changes from **{first['gap_bpb']:+.6f}** to **{last['gap_bpb']:+.6f} BPB**.",'',
        '![Learning on tokenizer 771](regret_favorite_trajectory.png)','',
        '[PDF](regret_favorite_trajectory.pdf) · [Every checkpoint](learning-curve.csv) · [Analysis and uncertainty](analysis.json)','',
        '## What the evaluation measures','',
        'Every checkpoint sees the same complete 32 × 128 training-style examples from the frozen step-6,500 ranking panel, covering all 23 original training corpus buckets. Native #771 prefixes and canonical continuations use the unchanged paired cut helper and fit BOS + prefix + variable continuation within 128 positions. Continuations, token counts, byte counts, contexts and pairing hashes are verified identical across checkpoints. Each saved LM also scores its matching canonical control.','',
        'The LM and hypernetwork stay frozen. Hypernetwork dropout remains active, with the same independent CUDA seed per batch across checkpoints and a fresh full-vocabulary embedding table each batch. Batch 128, microbatch 32, BF16 autocast and the validated GPU-only memory implementation match the training-reward diagnostic. BPB is byte-weighted within each update and then averaged equally across the 32 updates.','',
        'Panel A shows absolute loss, B shows the transfer gap relative to each trained canonical LM, C shows paired changes from the earliest scored checkpoint, and D shows policy focus. The shaded bands resample the same 32 batches 10,000 times. Negative changes in panel C mean improvement.','',
        '## Checkpoint lineage and limits','',
        'The full sweep includes all saved 500-step checkpoints through 8,000, along with the 2,912 and 5,288 handoff checkpoints. The earliest available saved checkpoint is 500; no step-zero result is reconstructed. The earlier Modal continuation beyond 5,288 was discarded when training moved to Tillicum. Its obsolete 5,500 checkpoint is excluded; 5,500 onward uses the resumed committed Tillicum trajectory. The dotted vertical line marks that continuation.','',
        'All trajectory points run on the same local RTX PRO 6000 with torch 2.15.0. Cross-device bitwise equality with the separate H200 ranking is not claimed. Corpus/tokenizer cloud originals and checkpoints remain intact. Only a temporary weight-only diagnostic copy is downloaded for each checkpoint, without optimizer state.','',
        'This checks progress on a fixed training-distribution panel. It is not a held-out generalization test: the panel is drawn after step 6,500 and its corpus region can be seen in later training. Bootstrap intervals describe variation across these 32 fixed batches; sequential corpus windows can be correlated. Policy concentration and improvement on #771 alone do not establish improvement on other tokenizers. The curve is observational and does not isolate the causal effect of adversarial selection.','',
        'A lower absolute transferred BPB means the model scores this tokenizer better. A smaller canonical gap means transfer is also improving relative to that checkpoint’s canonical LM. The gap depends on the LM and hypernetwork together; it does not isolate the hypernetwork alone.','',
        '| Step | Transferred BPB | Canonical BPB | Gap BPB | Policy probability | Selections so far |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in table:
        lines.append(f"| {row['step']:,} | {row['raw_bpb']:.6f} | {row['canonical_bpb']:.6f} | {row['gap_bpb']:+.6f} | {row['policy_probability']:.4%} | {row['cumulative_selections']:,} |")
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(REPORT/'manifest.json',dict(files={str(path.relative_to(REPORT)):dict(bytes=path.stat().st_size,sha256=sha256(path))
        for path in REPORT.rglob('*') if path.is_file() and path.name!='manifest.json'}))
    print(json.dumps(dict(checkpoints=len(table),latest=last,endpoint_change=result['endpoint_change'])))


if __name__=='__main__':main()
