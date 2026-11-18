"""Live boundary plots and complete independent audits for synthetic student sweeps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

from generalized_student import SHARING,SPEEDS,METHODS,group_ids
from package_training_inputs import atomic_json,sha256
from selector_simulation import ROOT,load_inputs

STATE=ROOT/'modal_results/student_generalization'
REPORT=ROOT/'reports/student-generalization'


def initialize_audit():
    global INPUTS
    INPUTS=load_inputs()


def audit_cell(task):
    world,folder,plan_hash=task;folder=Path(folder)
    receipt=json.loads((folder/'complete.json').read_text());path=folder/'trace.npz'
    if receipt['plan_sha256']!=plan_hash or sha256(path)!=receipt['trace_sha256']:
        raise ValueError('Plan or raw trace changed')
    audit_source=sha256(Path(__file__))
    cache=STATE/'audits'/(folder.name+'.json')
    if cache.exists():
        saved=json.loads(cache.read_text())
        if (saved['trace_sha256']==receipt['trace_sha256'] and saved['plan_sha256']==plan_hash
            and saved.get('audit_source_sha256')==audit_source):return saved
    configs,rewards,calibration=INPUTS
    base=rewards.mean(1);residuals=rewards-base[:,None]
    if world['starting_gaps']=='flat':base=np.full_like(base,base.mean())
    elif world['starting_gaps']=='shuffled':base=np.random.default_rng(20261118).permutation(base)
    groups=group_ids(configs,world['grouping']);sizes=np.bincount(groups)
    counts=np.zeros(1000,dtype=np.int64);rho=world['sharing']
    rate=-math.log((calibration['end_gap']/calibration['start_gap']-.5)/.5)/calibration['selected_visits']*world['speed']
    batches=np.random.default_rng(10000+receipt['seed']).integers(32,size=receipt['steps'])
    with np.load(path) as data:t={k:data[k] for k in data.files}
    if any(v.shape!=(receipt['steps'],) or not np.isfinite(v).all() for v in t.values()):
        raise ValueError('Missing or invalid observations')
    stale=[]
    for j in range(receipt['steps']):
        # Independently reconstruct group counts from the complete action history.
        group_visits=np.bincount(groups,weights=counts,minlength=len(sizes))
        practice=(1-rho)*counts+rho*group_visits[groups]
        if world['budget']=='shared':practice/=1+rho*(sizes[groups]-1)
        x=rate*practice
        remaining=(np.exp(-x) if world['curve']=='exponential' else
                   np.maximum(0.,1-x) if world['curve']=='linear' else 1/(1+x))
        floor=base*world['floor']
        means=base if rate==0 else floor+(base-floor)*remaining
        action=int(t['action'][j]);favorite=int(t['favorite'][j])
        if not 0<=action<1000 or not 0<=favorite<1000:raise ValueError('Invalid action')
        expected=dict(reward=means[action]+world['noise']*residuals[action,batches[j]],
            chosen_mean=means[action],best_mean=means.max(),student_mean=means.mean(),
            mastery=((base-means)/(base*(1-world['floor']))).mean())
        for field,value in expected.items():
            if abs(t[field][j]-value)>1e-12:raise ValueError(f'{folder.name}: {field} mismatch at {j+1}')
        rank=1+int((means>means[action]).sum());favorite_rank=1+int((means>means[favorite]).sum())
        if rank!=t['selected_rank'][j] or favorite_rank!=t['favorite_rank'][j]:
            raise ValueError('Current true ranking differs from the trace')
        if receipt['method']=='current_hardest' and rank!=1:raise ValueError('Current-hardest reference chose incorrectly')
        epsilon=.25 if receipt['method']=='coverage' else 0.
        if abs(t['favorite_behavior_probability'][j]-((1-epsilon)*t['favorite_probability'][j]+epsilon/1000))>1e-12:
            raise ValueError('Exploration mixture differs from recorded sampling')
        stale.append(t['favorite_behavior_probability'][j]>=.5 and favorite_rank>100)
        counts[action]+=1
    np.testing.assert_array_equal(t['stale'],stale)
    if receipt['persistent_failure']!=bool(t['stale'][-2000:].mean()>.5):raise ValueError('Failure summary mismatch')
    result=dict(plan_sha256=plan_hash,trace_sha256=receipt['trace_sha256'],steps=receipt['steps'],audit_source_sha256=audit_source,
        rewards_student_transitions_and_ranks_verified=True)
    atomic_json(cache,result);return result


def records(plan):
    rows=[]
    for path in sorted((STATE/'cells').glob('*/complete.json')):
        row=json.loads(path.read_text())
        if row['plan_sha256']!=plan['plan_sha256'] or row['steps']!=plan['steps']:
            raise ValueError('Receipt belongs to another frozen study')
        rows.append(row)
    return rows


def summarize(plan,rows):
    result=[]
    metrics=['final_choice_gap','final_learned_gap','whole_run_choice_gap','final_stale_fraction',
             'final_student_mean','final_mastery']
    for world in plan['worlds']:
        for method in plan['methods']:
            group=[r for r in rows if r['world_id']==world['id'] and r['method']==method]
            if not group:continue
            value=dict(**world,method=method,repetitions=len(group),
                failures=sum(r['persistent_failure'] for r in group))
            for field in metrics:
                values=np.array([r[field] for r in group]);value[field]=float(values.mean())
                value[field+'_sd']=float(values.std(ddof=1)) if len(values)>1 else None
            control=next(w for w in plan['worlds'] if w['speed']==0 and w['noise']==world['noise']
                         and w['starting_gaps']==world['starting_gaps'])
            paired=[]
            for row in group:
                baseline=next((r for r in rows if r['world_id']==control['id'] and r['method']==method
                    and r['seed']==row['seed']),None)
                if baseline:paired.append((row['persistent_failure'],baseline['persistent_failure']))
            value.update(paired_frozen_controls=len(paired),
                new_failures_vs_frozen=sum(a and not b for a,b in paired),
                recoveries_vs_frozen=sum(b and not a for a,b in paired))
            result.append(value)
    return result


def plots(plan,summary,completed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    font_manager.fontManager.addfont(str(ROOT/'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'})
    cmap=LinearSegmentedColormap.from_list('teacher_failure',['#36967B','#F2E3B5','#C75E57'])
    cmap.set_bad('#ECEFF1')
    full=completed==plan['cells']
    subtitle=('All runs complete' if full else f'Live results · {completed:,} / {plan["cells"]:,} runs complete')
    frozen=[]
    for method,label in [('reinforce','original'),('coverage','with random exploration')]:
        item=next((s for s in summary if s['speed']==0 and s['noise']==1 and s['starting_gaps']=='measured'
                   and s['method']==method and s['repetitions']==5),None)
        frozen.append(label+': '+(f'{item["failures"]}/5' if item else 'pending'))
    for field,filename,title in [('failure','student_generalization','When does the teacher get stuck?'),
        ('final_choice_gap','student_generalization_choices','How much harder a choice was available'),
        ('final_learned_gap','student_generalization_focus','How well the learned choices keep up · before random exploration')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.12,1,.88))
        for row,method in enumerate(['reinforce','coverage']):
            for col,budget in enumerate(['extra','shared']):
                matrix=np.full((len(SHARING),len(SPEEDS)),np.nan);counts={}
                for i,sharing in enumerate(SHARING):
                    for j,speed in enumerate(SPEEDS):
                        actual_budget='extra' if sharing==0 else budget
                        item=next((s for s in summary if s['study']=='map' and s['sharing']==sharing
                            and s['speed']==speed and s['budget']==actual_budget and s['method']==method
                            and s['repetitions']==len(plan['seeds'])),None)
                        if item:
                            matrix[i,j]=item['failures']/len(plan['seeds']) if field=='failure' else item[field]
                            counts[i,j]=item['failures']
                ax=axes[row,col]
                image=ax.imshow(matrix,aspect='auto',cmap=cmap if field=='failure' else 'YlOrRd',
                    vmin=0,vmax=1 if field=='failure' else .06)
                ax.set_xticks(range(len(SPEEDS)),[f'{x:g}×' for x in SPEEDS])
                ax.set_yticks(range(len(SHARING)),[f'{x:.0%}' for x in SHARING])
                ax.set_xlabel('How fast the student learns · reference speed = 1×')
                if col==0:ax.set_ylabel('How much practice helps related skills')
                heading=('Related skills get extra practice' if budget=='extra' else 'One practice update spread across the group')
                ax.set_title(heading+'\n'+('Original teacher' if method=='reinforce' else 'Teacher + 25% random exploration'),fontsize=9)
                if field=='failure':
                    for (i,j),n in counts.items():
                        ax.text(j,i,f'{n}/5',ha='center',va='center',color='white' if n in [0,5] else '#303945',fontsize=8)
        fig.suptitle(title+'\n'+subtitle,fontsize=12)
        fig.colorbar(image,ax=axes,label='Fraction of repetitions ending stuck' if field=='failure' else 'Synthetic difficulty difference · lower is better',shrink=.75)
        fig.text(.5,.015,'No-learning control · '+', '.join(frozen)+'. Grey cells await all five repetitions.\n“Stuck” means a majority favorite outside the current hardest 100. Sharing 0% uses the same control in both columns.\nThese are hypothetical students, not real LM results.',ha='center',va='bottom',fontsize=8,color='#67717D')
        for ext in ['png','pdf']:
            target=REPORT/(filename+'.'+ext);temporary=target.with_name(target.stem+'.tmp.'+ext)
            fig.savefig(temporary,dpi=170,bbox_inches='tight');temporary.replace(target)
            shutil.copy2(target,ROOT/'plots'/target.name)
        plt.close(fig)


def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true')
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    plan=json.loads((STATE/'plan.json').read_text());rows=records(plan)
    from pool_generalization import digest
    if digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Frozen specification checksum changed')
    if not args.partial:
        expected={(w['id'],m,s) for w in plan['worlds'] for m in plan['methods'] for s in plan['seeds']}
        if {(r['world_id'],r['method'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):
            raise ValueError('Full study requires every predeclared world, method and seed')
        for path,digest in plan['source_sha256'].items():
            if sha256(ROOT/path)!=digest:raise ValueError('Scientific inputs or implementation changed')
        lookup={w['id']:w for w in plan['worlds']}
        tasks=[(lookup[r['world_id']],str(STATE/'cells'/f'{r["world_id"]}-{r["method"]}-seed{r["seed"]}'),plan['plan_sha256']) for r in rows]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_audit) as pool:
            for i,result in enumerate(pool.map(audit_cell,tasks),1):
                if i%20==0:print(json.dumps(dict(stage='audit',completed_cells=i,total_cells=len(rows))),flush=True)
        atomic_json(REPORT/'audit.json',dict(complete=True,cells=len(rows),observations=len(rows)*plan['steps'],
            every_reward_student_transition_and_rank_reconstructed=True,plan_sha256=plan['plan_sha256']))
    summary=summarize(plan,rows)
    atomic_json(REPORT/'analysis.json',dict(complete=not args.partial,cells=len(rows),total_cells=plan['cells'],
        summary=summary,uncertainty='Five independent teacher initializations, paired across worlds/methods; SD, not confidence intervals.',
        limitation='Synthetic rules test mechanisms, not real model generalization. Extra/shared budgets conserve different practice mass; neither fixes exact loss improvement.'))
    csv_file(REPORT/'summary.csv',summary);csv_file(REPORT/'per-run-results.csv',rows)
    shutil.copy2(STATE/'plan.json',REPORT/'plan.json');plots(plan,summary,len(rows))
    completion=(f'**{len(rows):,} / {plan["cells"]:,} runs complete.** This is a local CPU study; real training stays paused. '
        'Live plots are provisional until every planned observation passes the independent audit.' if args.partial else
        f'**{len(rows):,} / {plan["cells"]:,} runs complete and audited.** '
        f'All {plan["observations"]:,} observations passed independent reconstruction. Real training stays paused.')
    lines=['# How far does the stale-teacher failure generalize?','',
        '[Live progress and plots](http://100.84.104.59:1337/?run=student-generalization). Repository snapshots can lag the running study.','',
        completion,'',
        'We test whether an adversarial teacher keeps choosing problems that used to be hard after a synthetic student improves. The first study used either no sharing or complete sharing across a tokenizer algorithm. This study fills in the gap between those extremes and varies student speed independently.','',
        'Practice can help related skills by 0%, 1%, 10% or 100% of its direct effect. Student speeds range from 100 times slower to 256 times faster than the earlier reference rule. We compare giving neighbours extra practice with spreading one fixed practice update across the group. The latter conserves practice weight, not the exact amount of loss reduction.','',
        'The follow-up checks remove or double observation noise, change learning plateaus and curve shapes, use smaller or randomly shuffled skill groups, share across all skills, and flatten or shuffle initial difficulty. Matched frozen controls accompany noise and initial-difficulty changes. Every condition is declared before results.','',
        'All 1,000 tokenizer settings and all 32 measured batch residuals remain present. Five new teacher-network seeds (48–52) are paired across conditions. The teacher sees only its selected reward. The original REINFORCE network is compared with its correctly differentiated 25% uniform mixture, uniform choices, and an evaluator-only reference that always chooses a currently hardest problem. That reference is not an optimal long-term teacher.','',
        '![Where teachers get stuck](student_generalization.png)','',
        '[Actual choice quality](student_generalization_choices.png) · [Learned choice quality before exploration](student_generalization_focus.png)','',
        'A repetition ends stuck when, for most of its final 2,000 choices, at least half the sampling probability is on a favorite outside the current hardest 100. This binary summary is accompanied by continuous choice-quality scores, so the conclusion need not depend on that particular cutoff. Each method creates its own student history; compare a choice with the alternatives available in that same world.','',
        'The summary also pairs each repetition with its frozen-student control using the same noise, starting difficulty and teacher seed. This distinguishes failures introduced by learning from failures that already occur without student change.','',
        'The uniform-mixture intervention changes actual sampling and differentiates its actual probability. Random observations can carry little policy-gradient credit when the learned probability is already tiny. More exploration is therefore not assumed to be a universal cure.','',
        'These are controlled hypothetical learning rules, not estimates that a real LM improves by this amount or transfers knowledge within these groups. The plots should identify hypotheses for real-world tests, including regions where our explanation fails.','',
        '## Data and reproduction','',
        '`python scripts/run_student_generalization.py` runs the frozen full plan. `python scripts/report_student_generalization.py --partial` refreshes the live views; without --partial it independently reconstructs every saved reward, student update and rank. Per-step traces and receipts remain under `modal_results/student_generalization/cells/`; full summary metadata are in this folder. No tokenizers, corpora or model checkpoints are downloaded.','']
    if (REPORT/'FINDINGS.md').exists():lines.insert(2,'Start with [the findings in plain language](FINDINGS.md).\n')
    (REPORT/'README.md').write_text('\n'.join(lines))
    if not args.partial:
        atomic_json(REPORT/'trace-manifest.json',dict(files={str(p.relative_to(STATE)):dict(bytes=p.stat().st_size,sha256=sha256(p))
            for p in (STATE/'cells').glob('*/trace.npz')}))
        atomic_json(REPORT/'manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p))
            for p in REPORT.iterdir() if p.is_file() and p.name!='manifest.json'}))


if __name__=='__main__':main()
