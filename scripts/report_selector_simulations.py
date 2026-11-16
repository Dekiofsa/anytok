"""Report empirical reward-table selector stress tests, including synthetic limits."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from package_training_inputs import atomic_json, sha256
from selector_simulation import ROOT, SCENARIOS, METHODS, LABELS, load_inputs

STATE = ROOT/'modal_results/selector_simulations'
REPORT = ROOT/'reports/selector-simulations'
NAMES = dict(uniform='Uniform', reinforce='Old REINFORCE · entropy 0', predictor='Hardness predictor + coverage')
COLORS = dict(uniform='#87919D', reinforce='#2F8FD0', predictor='#36967B')


def main():
    if not (STATE/'plan.json').exists():
        return
    plan = json.loads((STATE/'plan.json').read_text())
    REPORT.mkdir(parents=True, exist_ok=True)
    curves = []; cells = []; trace_paths = []
    for folder in sorted((STATE/'cells').glob('*')):
        if (folder/'complete.json').exists():
            cells.append(json.loads((folder/'complete.json').read_text()))
            trace_paths.append(folder/'trace.npz')
        path = folder/'events.jsonl'
        if path.exists():
            lines = path.read_bytes().splitlines(keepends=True)
            for line in lines:
                if line.endswith(b'\n'):
                    curves.append(json.loads(line))
    if not curves:
        return
    with (REPORT/'learning-curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(curves)
    grouped = {}
    for row in curves:
        grouped.setdefault((row['scenario'], row['method'], row['step']), []).append(row)
    aggregate = []
    for (scenario, method, step), values in sorted(grouped.items()):
        row = dict(scenario=scenario, method=method, step=step, seeds=len(values))
        for key in ['missed_hardness', 'mean_hardness', 'mean_oracle', 'top100_fraction', 'unique_visited', 'largest_recent_share']:
            numbers = np.array([r[key] for r in values])
            row[key] = float(numbers.mean()); row[key+'_sd'] = float(numbers.std(ddof=1)) if len(numbers)>1 else 0.
        aggregate.append(row)
    with (REPORT/'aggregate-curves.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregate[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(aggregate)
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            rows = [row for row in cells if row['scenario']==scenario and row['method']==method]
            if not rows:
                continue
            row = dict(scenario=scenario, method=method, completed_seeds=len(rows))
            for key in ['mean_missed_hardness', 'final_2000_missed_hardness', 'final_2000_top100_fraction', 'unique_visited']:
                values=np.array([r[key] for r in rows]); row[key]=float(values.mean())
                row[key+'_sd']=float(values.std(ddof=1)) if len(values)>1 else 0.
            summary.append(row)
    atomic_json(REPORT/'analysis.json', dict(completed_cells=len(cells), total_cells=75,
        summary=summary, metric='Current-world maximum mean minus selected-action mean, BPB; lower is better',
        uncertainty='Between-seed sample standard deviation; not a confidence interval',
        interpretation='Practice worlds depend on each selector’s choices; comparisons are local selection shortfall, not shared fixed-world absolute reward.'))
    if summary:
        with (REPORT/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n')
            writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT/'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        for method in METHODS:
            rows=[r for r in aggregate if r['scenario']==scenario and r['method']==method]
            # 500-step display windows; do not mix incompletely observed seeds at a step.
            rows=[r for r in rows if r['seeds']==5]
            windows=[]
            for end in range(500,20001,500):
                block=[r for r in rows if end-500<r['step']<=end]
                if len(block)==5:
                    # SD of the per-seed window average, retaining within-seed correlations.
                    vals=[]
                    for seed in plan['seeds']:
                        own=[r['missed_hardness'] for r in curves if r['scenario']==scenario and r['method']==method and r['seed']==seed and end-500<r['step']<=end]
                        vals.append(float(np.mean(own)))
                    windows.append((end,float(np.mean(vals)),float(np.std(vals,ddof=1))))
            if windows:
                x,y,sd=np.array(windows).T
                ax.plot(x,y,color=COLORS[method],label=NAMES[method])
                ax.fill_between(x,np.maximum(0,y-sd),y+sd,color=COLORS[method],alpha=.12,lw=0)
        if scenario in ('gradual','abrupt'):
            ax.axvline(10000,color='#B9A8C9',ls=':',lw=1)
        ax.set(title=LABELS[scenario],xlabel='Selection / observed reward',ylabel='Missed hardness, BPB ↓',xlim=(0,20000),ylim=(0,None))
        ax.grid(axis='y',color='#E9EDF1',lw=.6);ax.set_axisbelow(True)
    ax=axes.flat[-1];ax.axis('off')
    ax.text(0,.95,'What this measures',fontsize=12,weight='bold',va='top')
    waiting='\n\nMissing lines await five complete seed windows.' if len(cells)<75 else ''
    ax.text(0,.83,'Current hardest mean − selected mean\nLower means better selection.\n\nAll 1,000 tokenizers · 32 measured rewards each\n20,000 observations · five seeds per method\n\nLines: 500-step windows, averaged across seeds\nShading: ± one between-seed standard deviation\n\nChanging environments are synthetic.\nNo LM or hypernetwork is trained here.'+waiting,va='top',linespacing=1.5)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(f'Selector stress tests · {len(cells)} / 75 runs complete · CPU only',fontsize=14)
    for suffix in ('png','pdf'):
        target=ROOT/'plots'/f'selector_simulations.{suffix}';tmp=target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(tmp,dpi=170,facecolor='white');tmp.replace(target);shutil.copy2(target,REPORT/target.name)
    plt.close(fig)
    if len(cells)==75:
        configs,_,_=load_inputs()
        algorithms=np.array([row['algorithm'] for row in configs])
        diagnostics=[]
        for path in trace_paths:
            cell=json.loads((path.parent/'complete.json').read_text())
            with np.load(path) as trace:
                for label,start,end in [('before_change',8000,10000),('first_2k_after_change',10000,12000),('final_2k',18000,20000)]:
                    for selection in (['all','exploitation_only'] if cell['method']=='predictor' else ['all']):
                        mask=np.zeros(20000,dtype=bool);mask[start:end]=True
                        if selection=='exploitation_only':mask&=trace['exploration']==0
                        actions=trace['actions'][mask];counts=np.bincount(actions,minlength=1000)
                        diagnostics.append(dict(scenario=cell['scenario'],method=cell['method'],seed=cell['seed'],
                            window=label,selection=selection,observations=int(mask.sum()),
                            missed_hardness=float((trace['oracle_mean'][mask]-trace['chosen_mean'][mask]).mean()),
                            top100_fraction=float(np.mean(trace['selected_rank'][mask]<=100)),
                            unique_selected=int(np.unique(actions).size),most_selected_index=int(counts.argmax()),
                            largest_share=float(counts.max()/counts.sum()),
                            unigram_share=float(np.mean(algorithms[actions]=='unigram')),
                            wordpiece_share=float(np.mean(algorithms[actions]=='wordpiece')),
                            bpe_share=float(np.mean(algorithms[actions]=='bpe'))))
        with (REPORT/'window-diagnostics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(diagnostics[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(diagnostics)
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for ax,scenario in zip(axes.flat,SCENARIOS):
            for method in METHODS:
                rows=[r for r in aggregate if r['scenario']==scenario and r['method']==method and r['seeds']==5]
                x=[r['step'] for r in rows];y=np.array([r['largest_recent_share'] for r in rows]);sd=np.array([r['largest_recent_share_sd'] for r in rows])
                ax.plot(x,y,color=COLORS[method],label=NAMES[method])
                ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=COLORS[method],alpha=.12,lw=0)
            if scenario in ('gradual','abrupt'):ax.axvline(10000,color='#8657BC',ls=':',lw=1)
            ax.set(title=LABELS[scenario],xlabel='Selection step',ylabel='Largest one-tokenizer share · trailing 1k',xlim=(0,20000),ylim=(0,1))
            ax.grid(axis='y',color='#E9EDF1',lw=.6)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0,.9,'Concentration needs context',fontsize=13,weight='bold')
        axes.flat[-1].text(0,.8,'A high share can mean useful exploitation\nor getting stuck on an outdated favorite.\nRead this alongside current missed hardness\nand the algorithm-selection heatmap.\n\nThe predictor reserves 25% of choices for\ncomplete-pool exploration, so its maximum\nlong-window share is approximately 75%.\n\nShading: ± one between-seed SD.',va='top',linespacing=1.6)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle('Does the selector keep choosing the same tokenizer?',fontsize=14)
        for suffix in ('png','pdf'):fig.savefig(REPORT/f'selection_concentration.{suffix}',dpi=170,facecolor='white')
        plt.close(fig)
        fig,axes=plt.subplots(3,5,figsize=(15,7),constrained_layout=True,sharex=True,sharey=True)
        for column,scenario in enumerate(SCENARIOS):
            for row,method in enumerate(METHODS):
                shares=[]
                for path in trace_paths:
                    cell=json.loads((path.parent/'complete.json').read_text())
                    if cell['scenario']!=scenario or cell['method']!=method:continue
                    with np.load(path) as trace:actions=trace['actions']
                    shares.append(np.array([[np.mean(algorithms[actions[start:start+100]]==name) for start in range(0,20000,100)] for name in ('bpe','wordpiece','unigram')]))
                ax=axes[row,column]
                image=ax.imshow(np.mean(shares,axis=0),aspect='auto',origin='upper',extent=(0,20000,2.5,-.5),vmin=0,vmax=1,cmap='Greens')
                if row==0:ax.set_title(LABELS[scenario],fontsize=8)
                if column==0:ax.set_ylabel(NAMES[method]);ax.set_yticks([0,1,2],['BPE','WordPiece','Unigram'])
                if row==2:ax.set_xlabel('Selection step')
                if scenario in ('gradual','abrupt'):ax.axvline(10000,color='#8657BC',ls=':',lw=1)
        fig.colorbar(image,ax=axes,fraction=.02,label='Share of selections · 100-step windows · mean of five seeds')
        fig.suptitle('Where selectors spend observations · all 1,000 actions remain eligible',fontsize=13)
        for suffix in ('png','pdf'):
            fig.savefig(REPORT/f'selection_families.{suffix}',dpi=170,facecolor='white')
        plt.close(fig)
        traces=REPORT/'traces';traces.mkdir(exist_ok=True)
        for path in trace_paths:shutil.copy2(path,traces/(path.parent.name+'.npz'))
    shutil.copy2(STATE/'plan.json',REPORT/'plan.json')
    cal=plan['calibration']
    lines=['# Selector simulations: can the selector track changing weaknesses?','',
        f'**{len(cells)} / 75 runs complete.** Five environments × three selectors × five seeds, each with 20,000 observations. All work is local CPU; the real LM and hypernetwork remain unchanged.','',
        '## What is measured and what is simulated','',
        'The fixed environment resamples the complete 1,000 × 32 reward table measured at model checkpoint 6,500. A batch index is drawn uniformly with replacement at every step; all methods within a seed see the same batch-index sequence. Each reward is the selected tokenizer’s paired BPB gap on that batch. These are finite empirical reward distributions, not fresh language-model evaluations or estimates of all future text.','',
        f'The changing environments retain each tokenizer’s centered measured noise and change its mean. Calibration uses tokenizer #771: gap {cal["start_gap"]:.6f} at 6,500 to {cal["end_gap"]:.6f} at 8,000, a drop of {cal["observed_drop"]:.6f} over 1,500 updates and {cal["selected_visits"]} observed selections. This is an observational curve with joint-model changes and cross-tokenizer effects. It does **not** identify a causal learning rate. The following are explicit stress-test assumptions.','',
        '| Environment | Rule |','| --- | --- |',
        '| Fixed | No mean changes. All 32 measured rewards remain available per action. |',
        '| Gradual | Starting at selection 10,000, reduce every Unigram mean by the observed gap drop, linearly over 1,500 steps; hold it there. |',
        '| Abrupt | At 10,000, immediately reduce every Unigram mean by twice that drop. This is deliberately stronger than the observed change. |',
        '| Practice: tokenizer | Every selection exponentially reduces that tokenizer’s mean; the observed gap ratio is reproduced after the observed number of selections. |',
        '| Practice: family | The same update reduces the means of all tokenizers in the selected algorithm family. This assumes complete within-family transfer. |','',
        '## Selectors and fairness','',
        'Every cell starts with fresh original weights and optimizer state. The new predictor is imported unchanged: the shared 18→64→64→1 settings MLP, float64, Adam 0.001, one MSE update on the current observation, and one shuffled full-pool exploration choice every four steps. It receives no reference labels beyond its selected action’s reward. The old baseline imports the exact original SettingsTokenizerBandit: sampled softmax, REINFORCE, EMA baseline 0.99, Adam 0.001 and entropy coefficient **0**. Uniform selects independently from all 1,000 actions. These compare the complete selector designs; they do not isolate regression from forced exploration. No method gets a warm start from the frozen pilot.','',
        'A seed pairs the environment’s batch-index stream across methods and is also used for learner initialization. The simulator can inspect all current means for evaluation, but the selector receives exactly one selected-action reward per step. No counterfactual scores enter learner updates. Source hashes and exact parameters are in plan.json.','',
        '## How to read the results','',
        '**Missed hardness = largest current mean reward − selected tokenizer’s current mean reward.** Lower is better. This is selection shortfall, not the BPB-gap reward itself. It includes the cost of the predictor’s mandatory exploration. The comparator knows current means, not future noisy rewards or long-run consequences. Selecting an easy tokenizer can improve the model too; this metric tests immediate hard-example mining, not optimal future learning.','',
        'In the practice environments, each selector creates its own evolving environment. Absolute rewards and maximum rewards can therefore differ because a selector has already reduced weaknesses. Compare selection shortfall in each current world; do not interpret lower raw reward as worse selection. Shading is between-seed standard deviation, not a confidence interval.','',
        'The full traces retain all actions, observed rewards, true selected means, true current maxima, ranks and exploration flags. They permit checking concentration, switches and transient failures instead of judging the final average alone. Synthetic success would justify a joint-training test; it would not establish language-model generalization.','',
        '![Selector stress tests](selector_simulations.png)','',
        '[PDF](selector_simulations.pdf) · [Summary](summary.csv) · [Per-seed learning curves](learning-curves.csv) · [Aggregated curves](aggregate-curves.csv)','']
    if len(cells)==75:lines+=['![Algorithm selection over time](selection_families.png)','',
        '![Selection concentration](selection_concentration.png)','',
        '[Before/after window diagnostics](window-diagnostics.csv) separate the predictor’s exploitation choices from its mandatory exploration. Per-step numeric traces for every cell are in `traces/` (NumPy NPZ).','']
    if summary:
        lines+=['| Environment | Selector | Complete seeds | Final 2k missed hardness, BPB ↓ |','| --- | --- | ---: | ---: |']
        for row in summary:lines.append(f'| {LABELS[row["scenario"]]} | {NAMES[row["method"]]} | {row["completed_seeds"]} | {row["final_2000_missed_hardness"]:.6f} ± {row["final_2000_missed_hardness_sd"]:.6f} |')
    if len(cells)==75:
        lines+=['','## What happened','',
            'The predictor is not uniformly better than REINFORCE. REINFORCE has smaller average final selection shortfall in the fixed, gradual-change and tokenizer-only-practice environments. The predictor has smaller average shortfall in the abrupt-change and family-practice environments. These comparisons include all exploration choices and use the final 2,000 observations, averaged across five seeds.','',
            'The family-practice result is especially variable for REINFORCE: read the individual seeds and concentration plots, not only the mean. The predictor repeatedly switches among families, whereas some REINFORCE runs spend observations on families they have already made easier. Under tokenizer-only improvement, the shared predictor has to distinguish many similar settings with different selection histories; this is a weakness of its current settings-only input. These are interpretations of controlled simulations, not established explanations of the real training run.','',
            'Continued exploration has a visible price when the world is stable: the predictor deliberately spends 25% of observations on shuffled coverage. The exploitation-only rows in window-diagnostics.csv help separate this price from prediction quality. This comparison does not isolate the benefit of regression from the benefit of exploration.','',
            'The useful next real-world measurement is how broadly improvement transfers to neighboring tokenizers while training. That determines which simulated response is relevant. These results support testing adaptation during joint training; they do not prove the new selector will outperform REINFORCE on real LM training or held-out generalization.','',
            '## Verification and reproduction','',
            'The audit independently reconstructs every observed reward and current maximum for all 1,500,000 steps, checks pinned source and trace hashes, and verifies five exploration visits per tokenizer in every predictor run. See [completion-audit.json](completion-audit.json). Re-run with the existing project CPU environment:','',
            '```bash','CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-memory/bin/python scripts/run_selector_simulations.py',
            '.venv-memory/bin/python scripts/audit_selector_simulations.py',
            '.venv-memory/bin/python scripts/report_selector_simulations.py','```','',
            'The controller preserves completed runs after verifying their trace hashes; interrupted runs repeat deterministically from their seed. It refuses changed pinned inputs or simulator code in the same run directory. The real LM weights are not needed.','']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(complete_cells=len(cells),curve_rows=len(curves))))


if __name__=='__main__':main()
