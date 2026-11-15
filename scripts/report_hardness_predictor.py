"""Live plots and a transparent report; reference labels are evaluator-only."""
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from package_training_inputs import atomic_json, sha256
from pool_policy_ranking import average_ranks

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/hardness_predictor_6500'
REPORT=ROOT/'reports/hardness-predictor-6500'
PLOT=ROOT/'plots/hardness_predictor_6500'
REFERENCE=ROOT/'reports/regret-pool-ranking-6500'


def main():
    snapshots=[json.loads(path.read_text()) for path in sorted((STATE/'snapshots').glob('*.json'))]
    if not snapshots:return
    records=[json.loads(line) for line in (STATE/'observations.jsonl').read_text().splitlines()]
    last=snapshots[-1];step=last['step']
    stop_path=STATE/'stop-audit.json'
    stopped=json.loads(stop_path.read_text()) if stop_path.exists() else None
    if stopped and stopped['completed']!=step:
        raise ValueError('Stopped checkpoint and final monitoring snapshot disagree')
    records=[row for row in records if row['step']<=step]
    if [row['step'] for row in records]!=list(range(1,step+1)):
        raise ValueError('Live reward observations are incomplete or duplicated')
    if any(row['valid_examples']!=128 or row['maximum_positions']>128 or row['scored_bytes']<=0
           or abs(row['gap_bpb']-row['raw_bpb']+row['canonical_bpb'])>1e-12 for row in records):
        raise ValueError('Reward scoring protocol changed')
    reference=np.empty(1000);algorithms=['']*1000
    with (REFERENCE/'tokenizer-ranking.csv').open() as stream:
        for row in csv.DictReader(stream):
            index=int(row['index']);reference[index]=float(row['gap_bpb']);algorithms[index]=row['algorithm']
    expected=json.loads((REFERENCE/'analysis.json').read_text())['comparisons']['all']
    predicted=np.asarray(last['predictions'])
    if predicted.shape!=(1000,) or not np.isfinite(predicted).all():
        raise ValueError('All 1,000 finite predictions must be evaluated')
    ranks=np.empty(1000,dtype=int);ranks[np.argsort(-reference,kind='stable')]=np.arange(1,1001)
    ids=last['greedy_indices']
    result=dict(completed=step,total=20000,frozen_checkpoint_step=6500,
        stage='stopped' if stopped else ('complete' if step==20000 else 'running'),
        observations=len(records),real_reward_evaluations=True,reference_used_for_learning=False,
        greedy_indices=ids,greedy_reference_ranks=[int(ranks[i]) for i in ids],
        greedy_reference_gap=last['greedy_reference_gap'],uniform_reference_gap=expected['uniform_mean_gap'],
        saved_policy_reference_gap=expected['policy_expected_gap'],oracle_reference_gap=expected['hardest_gap'],
        greedy_mass_in_hardest100=last['greedy_mass_in_hardest100'],prediction_rmse=last['prediction_rmse'],
        unique_visited=last['unique_visited'],unique_exploration=last['unique_exploration'],
        exploration_observations=sum(row['mode']=='explore' for row in records),
        exploitation_observations=sum(row['mode']=='exploit' for row in records),
        prediction_rank_spearman=float(np.corrcoef(average_ranks(predicted),average_ranks(reference))[0,1])
            if predicted.std()>0 else None)
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'analysis.json',result)
    curve=[]
    for snapshot in snapshots:
        end=snapshot['step'];window=records[max(0,end-100):end]
        row={key:snapshot[key] for key in ['step','greedy_reference_gap','nominal_75_25_reference_gap',
            'greedy_mass_in_hardest100','prediction_rmse','unique_visited','unique_exploration']}
        for mode in ['explore','exploit']:
            values=[r['gap_bpb'] for r in window if r['mode']==mode]
            row[mode+'_observed_gap']=float(np.mean(values)) if values else None
        curve.append(row)
    with (REPORT/'learning-curve.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
    with (REPORT/'current-predictions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['index','predicted_gap','reference_gap','reference_rank',
            'algorithm','visits','exploration_visits']);writer.writeheader()
        for i in range(1000):writer.writerow(dict(index=i,predicted_gap=predicted[i],reference_gap=reference[i],
            reference_rank=int(ranks[i]),algorithm=algorithms[i],visits=last['counts'][i],
            exploration_visits=last['exploration_counts'][i]))
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT/'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    x=[row['step'] for row in curve];green,blue,purple,grey='#36967B','#2F8FD0','#8657BC','#87919D'
    ax=axes[0,0]
    for mode,color in [('exploit',green),('explore',blue)]:
        ax.plot(x,[row[mode+'_observed_gap'] for row in curve],color=color,label=mode.capitalize())
    ax.set(title='A · Actual rewards from fresh full-batch forwards',ylabel='Mean paired regret, BPB · 100-step display windows')
    ax.legend(frameon=False)
    ax=axes[0,1]
    ax.plot(x,[row['greedy_reference_gap'] for row in curve],color=green,label='Learned exploitation choice')
    for value,color,label in [(expected['uniform_mean_gap'],grey,'Uniform'),
        (expected['policy_expected_gap'],blue,'Saved 6.5k policy'),(expected['hardest_gap'],purple,'Measured pool maximum')]:
        ax.axhline(value,color=color,ls=':',lw=1,label=label)
    ax.set(title='B · Does learned selection target harder tokenizers?',ylabel='Reference-panel expected regret, BPB')
    ax.legend(frameon=False,fontsize=7)
    ax=axes[1,0]
    ax.plot(x,[row['unique_visited'] for row in curve],color=green,label='Any visit')
    ax.plot(x,[row['unique_exploration'] for row in curve],color=blue,label='Exploration visit')
    ax.set(title='C · Coverage of the complete 1,000-tokenizer pool',ylabel='Distinct tokenizers',ylim=(0,1020))
    ax.legend(frameon=False)
    ax=axes[1,1]
    for algorithm,color in [('bpe',blue),('wordpiece',green),('unigram',purple)]:
        mask=np.array(algorithms)==algorithm
        ax.scatter(reference[mask],predicted[mask],s=8,alpha=.4,color=color,label=algorithm)
    lo=min(float(reference.min()),float(predicted.min()));hi=max(float(reference.max()),float(predicted.max()))
    ax.plot([lo,hi],[lo,hi],color=grey,ls=':',lw=.8)
    ax.scatter(reference[ids],predicted[ids],marker='*',s=65,color='#242424',label='Current greedy choice',zorder=3)
    ax.set(title=f'D · Predictions across all tokenizers at step {step:,}',
           xlabel='Measured reference regret, BPB',ylabel='Predicted regret, BPB')
    ax.legend(frameon=False,fontsize=7)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set(xlabel='Predictor update',xlim=(0,max(100,step)))
    for ax in axes.flat:ax.grid(axis='y',color='#E9EDF1',lw=.6);ax.set_axisbelow(True)
    suffix=' · stopped early' if stopped else ''
    fig.suptitle(f'Frozen 6.5k · online hardness predictor · {step:,} / 20,000 live rewards{suffix}',fontsize=13)
    fig.supxlabel('One observation → one MSE update · every fourth step explores · LM/HN frozen with active dropout\nReference scores are evaluator-only estimates from the H200 panel; learner rewards are measured locally.',fontsize=8)
    for suffix in ['png','pdf']:
        target=PLOT.with_suffix('.'+suffix);temporary=target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(temporary,dpi=170,facecolor='white');temporary.replace(target);shutil.copy2(target,REPORT/target.name)
    plt.close(fig)
    for name in ['plan.json','preflight.json','completion-audit.json','stop-audit.json']:
        if (STATE/name).exists():shutil.copy2(STATE/name,REPORT/name)
    lines=['# Online hardness predictor on the frozen 6.5k model','',
        (f'**Stopped at the user’s request after {step:,} / 20,000 observations.** The pilot ended early at a saved checkpoint; this is not a completed 20k experiment.' if stopped else f'**{step:,} / 20,000 observations complete.**')+' The LM and hypernetwork stay frozen. Only the small CPU settings predictor learns.','',
        'The shared 18→64→64→1 Tanh network predicts paired BPB gap. After each real reward evaluation, Adam at 0.001 makes one squared-error update on that observation. Every fourth step visits the next tokenizer in a shuffled complete-pool cycle; other steps choose the largest prediction, breaking ties uniformly. There is no initial warmup, replay buffer, entropy term or age weighting. A full 20k run gives each tokenizer exactly five exploration visits.','',
        'Each reward uses a fresh 128-example recipe-sampled training batch, the unchanged paired cut helper, variable canonical continuations, BOS + prefix + continuation ≤128 positions, and microbatch32. Canonical and transferred targets/bytes match exactly. Hypernetwork dropout stays active, with a fresh full native embedding table per observation. The validated memory implementation preserves full vocabularies.','',
        'The original 23-bucket corpus and complete tokenizer pool are checksum verified. Data sampling uses private Python seed42 and selector initialization/permutations use private seed43. Exact raw windows reserved for the reference panel and replacement banks are excluded from learner batches by content hashes while retaining recipe bucket choices.','',
        'The existing all-1,000 step-6,500 ranking is used only after learner updates for monitoring and assessment. Its labels never enter predictor training or action selection. It is a finite H200 estimate of the reward distribution; local Blackwell measurements are not asserted bitwise equal across devices. The same-device scorer and exact-resume checks are recorded in preflight.json.','',
        f"Current greedy choices: **{ids}**, reference ranks **{result['greedy_reference_ranks']}**. Expected reference gap: **{result['greedy_reference_gap']:+.6f} BPB**; saved policy **{result['saved_policy_reference_gap']:+.6f}**; uniform **{result['uniform_reference_gap']:+.6f}**; measured maximum **{result['oracle_reference_gap']:+.6f}**.",'',
        'Panel A plots actual observed rewards separately for exploration/exploitation, with display-only 100-step averaging. Panel B assesses the learned exploitation choice on the reference; mandatory exploration is excluded from that choice. C checks coverage. D checks the predictions across every pool tokenizer.','',
        '![Live hardness-predictor plots](hardness_predictor_6500.png)','',
        '[PDF](hardness_predictor_6500.pdf) · [Learning curve](learning-curve.csv) · [All current predictions](current-predictions.csv)','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    if step==20000 or stopped:
        with gzip.open(REPORT/'observations.jsonl.gz','wt') as stream:
            for row in records:stream.write(json.dumps(row)+'\n')
        with gzip.open(REPORT/'snapshots.json.gz','wt') as stream:json.dump(snapshots,stream)
    atomic_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=sha256(p))
        for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(result))


if __name__=='__main__':main()
