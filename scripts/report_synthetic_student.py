"""Audit every synthetic student transition, then report all paired continuations."""
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from package_training_inputs import atomic_json,sha256
from selector_simulation import ROOT,load_inputs
from synthetic_student import SyntheticStudent,SCENARIOS,INITIALIZATIONS,COVERAGES

STATE=ROOT/'modal_results/synthetic_student'
REPORT=ROOT/'reports/synthetic-student'
LABELS=dict(frozen='Student does not learn',individual='Practice helps one tokenizer',family='Practice helps its algorithm family')


def audit_cell(folder,configs,rewards,calibration):
    receipt=json.loads((folder/'complete.json').read_text())
    if sha256(folder/'trace.npz')!=receipt['trace_sha256']:raise ValueError('Trace checksum changed')
    with np.load(folder/'trace.npz') as loaded:trace={k:loaded[k] for k in loaded.files}
    n=receipt['steps']
    if any(v.shape!=(n,) for v in trace.values()) or any(not np.isfinite(v).all() for v in trace.values()):
        raise ValueError('Incomplete synthetic trace')
    env=SyntheticStudent(rewards,configs,calibration,receipt['scenario'],receipt['seed'],n)
    def close(observed,expected,field,step):
        if abs(float(observed)-float(expected))>1e-12:
            raise ValueError(f'{folder.name}: {field} mismatch at step {step}')
    stale=[]
    for offset in range(n):
        step=offset+1;means=env.means();action=int(trace['action'][offset]);favorite=int(trace['favorite'][offset])
        if not 0<=action<1000 or not 0<=favorite<1000:raise ValueError('Action outside complete pool')
        expected=dict(chosen_mean=means[action],favorite_mean=means[favorite],best_mean=means.max(),
            best_action=int(means.argmax()),selected_rank=1+int((means>means[action]).sum()),
            favorite_rank=1+int((means>means[favorite]).sum()),batch=int(env.batches[offset]),
            student_mean_gap=means.mean(),mean_mastery=((env.base-means)/(env.base-env.floor)).mean())
        expected['observed_reward']=env.observe(action,step)
        for field,value in expected.items():close(trace[field][offset],value,field,step)
        if not 0<=trace['action_probability'][offset]<=1 or not 0<=trace['action_probability_after'][offset]<=1:
            raise ValueError('Invalid recorded probabilities')
        qfavorite=(1-receipt['coverage'])*trace['favorite_probability'][offset]+receipt['coverage']/1000
        close(trace['favorite_behavior_probability'][offset],qfavorite,'sampling mixture',step)
        stale.append(qfavorite>=.5 and expected['favorite_rank']>100)
    np.testing.assert_array_equal(trace['stale_focus'],stale)
    close(receipt['final_student_mean_gap'],env.means().mean(),'final student',n)
    close(receipt['final_2000_missed_hardness'],
        (trace['best_mean'][-2000:]-trace['chosen_mean'][-2000:]).mean(),'final choice quality',n)
    close(receipt['final_2000_policy_missed_hardness'],
        (trace['best_mean'][-2000:]-trace['policy_expected_mean'][-2000:]).mean(),'final learned policy',n)
    return receipt,trace


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def plot(curves,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT/'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'})
    plots=[('synthetic_student','actual_shortfall','How much harder a choice was available · BPB',
            'Actual selected choices, including the exploration cost'),
        ('synthetic_student_focus','policy_shortfall','How much harder a choice was available · BPB',
            'Learned policy alone, before its uniform exploration component'),
        ('synthetic_student_stale','stale_fraction','Time spent on an outdated majority favorite',
            'Outdated = over 50% sampling probability, outside current hardest 100'),
        ('synthetic_student_learning','student_gap','Remaining synthetic mean gap · BPB',
            'Each teacher creates its own student history; this is not real LM generalization')]
    for filename,field,ylabel,subtitle in plots:
        fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained',sharex=True)
        for row,initial in enumerate(INITIALIZATIONS):
            for column,scenario in enumerate(SCENARIOS):
                ax=axes[row,column]
                for coverage,color,label in [(0.,'#2F8FD0','Original REINFORCE'),(.25,'#36967B','REINFORCE + 25% coverage')]:
                    x=[];mean=[];sd=[]
                    for end in range(500,plan['steps']+1,500):
                        values=[c[field] for c in curves if c['initial']==initial and c['scenario']==scenario
                            and c['coverage']==coverage and c['step']==end]
                        if len(values)!=len(plan['seeds']):continue
                        x.append(end);mean.append(np.mean(values));sd.append(np.std(values,ddof=1))
                    mean=np.asarray(mean);sd=np.asarray(sd)
                    ax.plot(x,mean,color=color,label=label)
                    ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.12,lw=0)
                ax.set_title(LABELS[scenario]+('\nFresh teacher' if initial=='fresh' else '\nReal collapsed teacher at 6.5k'),fontsize=9)
                if row==1:ax.set_xlabel('Synthetic training choices')
                if column==0:ax.set_ylabel(ylabel)
                ax.set_xlim(0,plan['steps']);ax.grid(axis='y',alpha=.15)
                if field=='stale_fraction':ax.set_ylim(0,1)
                elif field!='student_gap':ax.set_ylim(bottom=0)
        fig.suptitle(subtitle,fontsize=12)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
        for suffix in ['png','pdf']:
            target=REPORT/(filename+'.'+suffix);fig.savefig(target,dpi=170,bbox_inches='tight')
            shutil.copy2(target,ROOT/'plots'/target.name)
        plt.close(fig)


def main():
    REPORT.mkdir(parents=True,exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    plan=json.loads((STATE/'plan.json').read_text());configs,rewards,calibration=load_inputs()
    for relative,expected in plan['source_sha256'].items():
        if sha256(ROOT/relative)!=expected:raise ValueError('Pinned simulation input or implementation changed')
    receipts=[];curves=[];mechanisms=[];seen=set()
    for folder in sorted((STATE/'cells').iterdir()):
        if not (folder/'complete.json').exists():continue
        receipt,trace=audit_cell(folder,configs,rewards,calibration)
        key=(receipt['scenario'],receipt['initial'],receipt['coverage'],receipt['seed'])
        if key in seen:raise ValueError('Duplicate experimental condition')
        seen.add(key);receipts.append(receipt)
        stale=trace['stale_focus']
        better=(trace['chosen_mean']>trace['favorite_mean']) & stale
        positive=better & (trace['advantage']>0)
        gained=trace['action_probability_after']>trace['action_probability']
        epsilon=receipt['coverage']
        probability=trace['action_probability'][positive]
        # d log(q_a) / d log(pi_a): uniform observations can carry very little
        # gradient credit when the trainable policy already suppresses that arm.
        credit=(1-epsilon)*probability/((1-epsilon)*probability+epsilon/1000)
        mechanisms.append(dict(scenario=receipt['scenario'],initial=receipt['initial'],
            coverage=receipt['coverage'],seed=receipt['seed'],stale_steps=int(stale.sum()),
            harder_alternative_observations_while_stale=int(better.sum()),
            harder_alternative_with_nonpositive_advantage=int((better & ~positive).sum()),
            harder_alternative_with_positive_advantage=int(positive.sum()),
            positive_advantage_without_probability_gain=int((positive & ~gained).sum()),
            median_mixture_gradient_credit_on_positive_alternatives=float(np.median(credit)) if len(credit) else None))
        print(json.dumps(dict(stage='audit',completed_cells=len(receipts),total_cells=60,cell=folder.name)),flush=True)
        for end in range(500,plan['steps']+1,500):
            sl=slice(end-500,end)
            curves.append(dict(scenario=receipt['scenario'],initial=receipt['initial'],coverage=receipt['coverage'],
                seed=receipt['seed'],step=end,
                actual_shortfall=float((trace['best_mean'][sl]-trace['chosen_mean'][sl]).mean()),
                policy_shortfall=float((trace['best_mean'][sl]-trace['policy_expected_mean'][sl]).mean()),
                stale_fraction=float(trace['stale_focus'][sl].mean()),student_gap=float(trace['student_mean_gap'][sl].mean()),
                mean_mastery=float(trace['mean_mastery'][sl].mean()),
                probability_on_hardest100=float(trace['hardest100_probability'][sl].mean())))
    expected={(s,i,c,k) for s in SCENARIOS for i in INITIALIZATIONS for c in COVERAGES for k in plan['seeds']}
    if seen!=expected:raise ValueError('All 60 runs are required; do not drop seeds or conditions')
    metrics=['final_2000_missed_hardness','final_2000_policy_missed_hardness',
        'final_2000_stale_focus_fraction','final_2000_top100_probability','final_student_mean_gap','final_mastery']
    summary=[];paired=[]
    for scenario in SCENARIOS:
        for initial in INITIALIZATIONS:
            for coverage in COVERAGES:
                group=[r for r in receipts if (r['scenario'],r['initial'],r['coverage'])==(scenario,initial,coverage)]
                item=dict(scenario=scenario,initial=initial,coverage=coverage,repetitions=len(group))
                for metric in metrics:
                    values=np.array([r[metric] for r in group]);item[metric]=float(values.mean());item[metric+'_sd']=float(values.std(ddof=1))
                summary.append(item)
            for metric in metrics:
                values=[]
                for seed in plan['seeds']:
                    a=next(r for r in receipts if (r['scenario'],r['initial'],r['coverage'],r['seed'])==(scenario,initial,0.,seed))
                    b=next(r for r in receipts if (r['scenario'],r['initial'],r['coverage'],r['seed'])==(scenario,initial,.25,seed))
                    values.append(b[metric]-a[metric])
                paired.append(dict(scenario=scenario,initial=initial,metric=metric,
                    coverage_minus_original=float(np.mean(values)),paired_sd=float(np.std(values,ddof=1)),
                    negative_changes=sum(v<0 for v in values),positive_changes=sum(v>0 for v in values),
                    differences=values))
    atomic_json(REPORT/'analysis.json',dict(summary=summary,paired=paired,
        interpretation='Hypothetical student rules; five future continuations per condition, one real collapsed initialization. SD across continuations, not a confidence interval or evidence about real-world student generalization.'))
    atomic_json(REPORT/'audit.json',dict(complete=True,cells=60,observations=1200000,
        every_reward_rank_and_student_transition_reconstructed=True,replay=json.loads((STATE/'replay-verification.json').read_text())))
    write_csv(REPORT/'summary.csv',summary);write_csv(REPORT/'paired-comparisons.csv',paired)
    write_csv(REPORT/'learning-curves.csv',curves);write_csv(REPORT/'per-run-results.csv',receipts)
    write_csv(REPORT/'mechanism-observations.csv',mechanisms)
    traces=REPORT/'traces';traces.mkdir(exist_ok=True)
    for folder in sorted((STATE/'cells').iterdir()):
        if (folder/'complete.json').exists():shutil.copy2(folder/'trace.npz',traces/(folder.name+'.npz'))
    # The committed training log contains scalar observations and hashes, not
    # corpus text or model weights. Preserve it so the initial policy is replayable.
    history=ROOT/'modal_results/anytok_pool_regret_rl_20k/committed-train.log'
    (REPORT/'policy-history.jsonl.gz').write_bytes(gzip.compress(history.read_bytes(),mtime=0))
    shutil.copy2(STATE/'replay-verification.json',REPORT/'replay-verification.json')
    if (STATE/'runtime-verification.json').exists():shutil.copy2(STATE/'runtime-verification.json',REPORT/'runtime-verification.json')
    shutil.copy2(STATE/'plan.json',REPORT/'plan.json');plot(curves,plan)
    lines=['# Can an adversarial teacher keep up with a learning student?','',
        'This CPU-only experiment isolates exploration in the original REINFORCE teacher. The synthetic student begins with the measured difficulty of all 1,000 tokenizers at model step 6,500. It can stay unchanged, improve only on the selected tokenizer, or improve on every tokenizer using the selected algorithm. The learner sees only its selected tokenizer’s noisy reward.','',
        'Practice removes a learnable deficit and eventually reaches a floor. We set that floor to half the initial difficulty, and choose the learning rate once from the observed relative improvement of tokenizer #771. Both the floor and the extent of shared improvement are hypothetical assumptions, not fitted causal properties of the real model. The original 32 measured noise values per tokenizer are retained.','',
        'We compare original entropy-zero REINFORCE with the same network and optimizer sampling from 75% learned probability plus 25% uniform probability. The exploratory version differentiates the probability distribution it actually samples. There is no hardness predictor in this experiment.','',
        'Each condition starts either from the original uniform policy or from the actual collapsed policy at step 6,500. The latter is reconstructed from every recorded action and reward; every replayed action and advantage must match, along with final probabilities and counts. Five paired future sampling and noise streams branch from the same initial state. These repetitions do not represent five independently trained real models.','',
        '## Results','',
        'The table measures the final 2,000 choices. “Harder choice available” is the current maximum mean gap minus the selected tokenizer’s mean gap. Lower means the teacher chose closer to the hardest remaining alternative. It includes the immediate cost of uniform exploration. The learned-policy-only view is reported separately. Values are means ± sample standard deviations across five paired continuation streams, not confidence intervals.','',
        '| Synthetic student | Teacher start | Original REINFORCE | With 25% coverage |',
        '| --- | --- | ---: | ---: |']
    for scenario in SCENARIOS:
        for initial in INITIALIZATIONS:
            cells=[next(r for r in summary if (r['scenario'],r['initial'],r['coverage'])==(scenario,initial,c)) for c in COVERAGES]
            values=[f'{r["final_2000_missed_hardness"]:.5f} ± {r["final_2000_missed_hardness_sd"]:.5f}' for r in cells]
            lines.append(f'| {LABELS[scenario]} | {initial} | {values[0]} | {values[1]} |')
    def ended_stale(scenario,initial,coverage):
        return sum(r['final_2000_stale_focus_fraction']>.5 for r in receipts
            if (r['scenario'],r['initial'],r['coverage'])==(scenario,initial,coverage))
    lines+=['',f'With a fresh teacher and no coverage, an outdated majority persists through most of the final window in {ended_stale("frozen","fresh",0.)}/5 frozen-student continuations, {ended_stale("individual","fresh",0.)}/5 individual-learning continuations and {ended_stale("family","fresh",0.)}/5 family-learning continuations. Starting from the real collapsed policy, it persists in {ended_stale("frozen","collapsed",0.)}/5 continuations even after the student is frozen.','',
        'Those comparisons separate two questions: whether the student’s learning rule can produce the failure from a fresh teacher, and whether a teacher with a real collapsed history recovers once the student stops changing. Neither result alone identifies the cause of the original real training collapse.']
    lines+=['',
        '![Actual teacher choices](synthetic_student.png)','',
        '[Learned choices without exploration](synthetic_student_focus.png) · [Outdated majority favorites](synthetic_student_stale.png) · [Synthetic student progress](synthetic_student_learning.png)','',
        'A majority is labeled outdated only when its tokenizer has at least 50% actual selection probability and lies outside the current hardest 100. Concentration on a tokenizer that remains hard is not counted as failure.','',
        'Uniform mixing mechanically reduces the maximum sampling probability. A smaller outdated-majority fraction by itself does not prove that the learned ranking improved; read it alongside the learned-policy-only scores and favorite ranks.','',
        'Student histories diverge across selectors. Compare each choice with the alternatives available to that teacher at that moment. A lower observed reward can mean the student has already improved, rather than that the teacher selected badly. Synthetic student progress is not evidence of real LM generalization.','',
        'Hardest-task selection and useful teaching are separate objectives. A tokenizer can remain hard because it is near an irreducible floor. Selecting it can maximize immediate reward while doing little to improve the student. The mastery plot tracks the synthetic learnable deficits separately.','',
        'mechanism-observations.csv counts what happens while a teacher has an outdated majority favorite: how often it observes a genuinely harder alternative, whether that observation produces a positive policy advantage, and whether the update then increases that alternative’s probability. A nonpositive advantage can reflect batch noise or a stale reward baseline; this count alone does not distinguish them.','',
        'The exploration intervention is specifically an on-policy uniform mixture. Its gradient on a sampled action is weighted by `(1-epsilon) * pi(action) / q(action)`. An alternative can therefore be observed through the uniform component yet receive little gradient credit when its learned probability is tiny. The mechanism table records this factor. These results do not test a shuffled-coverage learner with another way of using exploration observations.','',
        '## Audit and reproduction','',
        'All 60 runs and all 1.2 million observations are retained. The audit reconstructs every reward, chosen/favorite rank and student update from the saved actions. Source and trace hashes are pinned. No world, seed or method is selected for favorable results. The original 75-run simulator remains separate and unchanged.','',
        '`python scripts/run_synthetic_student.py` runs the experiment and report. `python scripts/report_synthetic_student.py` independently repeats the trace audit and rebuilds the report. Every per-step trace is included in traces/, with receipt hashes in per-run-results.csv. Original working copies remain in `modal_results/synthetic_student/cells/*/trace.npz`. policy-history.jsonl.gz contains the original scalar training log needed to reconstruct the initial collapsed policy; decompress it to `modal_results/anytok_pool_regret_rl_20k/committed-train.log` when reproducing in a fresh checkout. Original full-pool reward inputs are in `reports/regret-pool-ranking-6500/`. There are no real LM updates or GPU allocations.','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    atomic_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=sha256(p))
        for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(stage='report_complete',cells=60,observations=1200000)),flush=True)


if __name__=='__main__':main()
