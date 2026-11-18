"""Fork fresh and real collapsed policies into controlled CPU-only synthetic students."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime,timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from hardness_predictor import _features
from package_training_inputs import atomic_json,sha256
from selector_simulation import ROOT,load_inputs
from synthetic_student import SyntheticStudent,SCENARIOS,INITIALIZATIONS,COVERAGES,mixed_probabilities,update_policy,restore

STATE=ROOT/'modal_results/synthetic_student'
STEPS=20000
SEEDS=(43,44,45,46,47)
LOG=ROOT/'modal_results/anytok_pool_regret_rl_20k/committed-train.log'


def emit(stage,completed=0,workers=None,**fields):
    value=dict(stage=stage,completed=completed,total=1200000,platform='Local · CPU only',
        workers=workers or [],at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    atomic_json(STATE/'status.json',value)
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def replay(configs):
    """Reconstruct the actual 6.5k policy and optimizer from observed real rewards."""
    target=STATE/'replayed-policy.pt';receipt_path=STATE/'replay-verification.json'
    if target.exists() and receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if sha256(target)!=receipt['state_sha256']:raise ValueError('Replayed state changed')
        return
    policy=_features.SettingsTokenizerBandit(configs,seed=43,entropy_weight=0.)
    rows={}
    for line in LOG.read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if row.get('stage')=='train' and 1<=row.get('step',0)<=6500:
            if row['step'] in rows:raise ValueError('Duplicate committed training update')
            rows[row['step']]=row
    if sorted(rows)!=list(range(1,6501)):raise ValueError('All actual 6,500 training observations required')
    max_advantage_error=0.
    for step in range(1,6501):
        row=rows[step];action=policy.sample()
        if action!=row['tokenizer_pool_index']:
            raise ValueError(f'Actual policy replay action mismatch at update {step}')
        result=policy.update(action,row['policy_reward_bpb'])
        error=abs(result['policy_advantage_bpb']-row['policy_advantage_bpb'])
        max_advantage_error=max(max_advantage_error,error)
        if error>1e-10:raise ValueError('Actual advantage replay differs')
        if step%1000==0:emit('Reconstructing the real collapsed policy',replay_step=step)
    reference=json.loads((ROOT/'reports/regret-pool-ranking-6500/policy.json').read_text())
    error=float(np.max(np.abs(policy.probabilities()-reference['probabilities'])))
    if error>1e-9 or policy.counts.tolist()!=reference['counts']:
        raise ValueError('Replayed policy differs from independently saved 6.5k checkpoint')
    state=copy.deepcopy(policy.state_dict())
    torch.save(state,target)
    atomic_json(receipt_path,dict(updates=6500,all_actual_actions_match=True,max_probability_error=error,
        max_advantage_error=max_advantage_error,counts_match=True,state_sha256=sha256(target),
        observed_favorite=int(policy.probabilities().argmax()),
        observed_favorite_probability=float(policy.probabilities().max())))


def initialize():
    global INPUTS,STATES
    torch.set_num_threads(1)
    INPUTS=load_inputs();configs=INPUTS[0]
    STATES=dict(collapsed=torch.load(STATE/'replayed-policy.pt',weights_only=False,map_location='cpu'),
        fresh=copy.deepcopy(_features.SettingsTokenizerBandit(configs,seed=43,entropy_weight=0.).state_dict()))


def cell_key(task):
    scenario,initial,coverage,seed=task
    return f'{scenario}-{initial}-coverage{int(coverage*100)}-seed{seed}'


def run_cell(task):
    scenario,initial,coverage,seed=task;key=cell_key(task)
    folder=STATE/'cells'/key;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'complete.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value['trace_sha256']!=sha256(folder/'trace.npz'):raise ValueError('Completed trace changed')
        return value
    configs,rewards,calibration=INPUTS
    env=SyntheticStudent(rewards,configs,calibration,scenario,seed,STEPS)
    policy=restore(configs,STATES[initial],seed)
    names=['observed_reward','chosen_mean','best_mean','policy_expected_mean','behavior_expected_mean',
        'favorite_mean','favorite_probability','favorite_behavior_probability','action_probability',
        'action_probability_after','advantage','student_mean_gap','mean_mastery','hardest100_probability']
    trace={name:np.empty(STEPS,dtype=np.float64) for name in names}
    for name in ['action','selected_rank','favorite','favorite_rank','best_action','batch']:
        trace[name]=np.empty(STEPS,dtype=np.int16)
    trace['stale_focus']=np.empty(STEPS,dtype=np.bool_)
    started=time.monotonic();events=folder/'events.jsonl';events.write_text('')
    for offset in range(STEPS):
        step=offset+1;means=env.means();p,q=mixed_probabilities(policy,coverage)
        action=int(policy.rng.choice(1000,p=q));favorite=int(p.argmax());best=float(means.max())
        rank=1+int((means>means[action]).sum());favorite_rank=1+int((means>means[favorite]).sum())
        reward=env.observe(action,step)
        result=update_policy(policy,action,reward,coverage)
        values=dict(action=action,selected_rank=rank,favorite=favorite,favorite_rank=favorite_rank,
            best_action=int(means.argmax()),batch=int(env.batches[offset]),observed_reward=reward,
            chosen_mean=means[action],best_mean=best,policy_expected_mean=float(p@means),
            behavior_expected_mean=float(q@means),favorite_mean=means[favorite],favorite_probability=p[favorite],
            favorite_behavior_probability=q[favorite],action_probability=p[action],action_probability_after=result['policy_selected_probability'],
            advantage=result['policy_advantage_bpb'],student_mean_gap=float(means.mean()),
            mean_mastery=float(((env.base-means)/(env.base-env.floor)).mean()),
            hardest100_probability=float(p[means>=np.partition(means,-100)[-100]].sum()),
            stale_focus=bool(q[favorite]>=.5 and favorite_rank>100))
        for name,value in values.items():trace[name][offset]=value
        if step%100==0:
            window=slice(step-100,step)
            row=dict(step=step,scenario=scenario,initial=initial,coverage=coverage,seed=seed,
                at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
                missed_hardness=float((trace['best_mean'][window]-trace['chosen_mean'][window]).mean()),
                policy_missed_hardness=float((trace['best_mean'][window]-trace['policy_expected_mean'][window]).mean()),
                stale_focus_fraction=float(trace['stale_focus'][window].mean()),
                favorite_probability=float(trace['favorite_probability'][window].mean()),
                hardest100_probability=float(trace['hardest100_probability'][window].mean()),
                student_mean_gap=float(means.mean()),mean_mastery=values['mean_mastery'])
            atomic_json(folder/'status.json',dict(stage='running',completed=step,**row))
            with events.open('a') as f:f.write(json.dumps(row)+'\n')
    temporary=folder/'trace.tmp.npz';np.savez_compressed(temporary,**trace);temporary.replace(folder/'trace.npz')
    end=env.means();late=slice(-2000,None)
    summary=dict(scenario=scenario,initial=initial,coverage=coverage,seed=seed,steps=STEPS,
        trace_sha256=sha256(folder/'trace.npz'),seconds=time.monotonic()-started,
        final_2000_missed_hardness=float((trace['best_mean'][late]-trace['chosen_mean'][late]).mean()),
        final_2000_policy_missed_hardness=float((trace['best_mean'][late]-trace['policy_expected_mean'][late]).mean()),
        final_2000_stale_focus_fraction=float(trace['stale_focus'][late].mean()),
        final_2000_top100_probability=float(trace['hardest100_probability'][late].mean()),
        final_student_mean_gap=float(end.mean()),final_mastery=float(((env.base-end)/(env.base-env.floor)).mean()),
        total_stale_steps=int(trace['stale_focus'].sum()),unique_selected=int(np.unique(trace['action']).size),
        first_stale_step=int(np.flatnonzero(trace['stale_focus'])[0]+1) if trace['stale_focus'].any() else None)
    atomic_json(receipt,summary);atomic_json(folder/'status.json',dict(stage='complete',completed=STEPS,**summary))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);configs,rewards,calibration=load_inputs()
    sources=['scripts/synthetic_student.py','scripts/run_synthetic_student.py','scripts/selector_simulation.py',
        'scripts/hardness_predictor.py','runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py',
        'reports/regret-pool-ranking-6500/all-scores.json.gz','reports/regret-pool-ranking-6500/pool-index.json',
        'reports/regret-pool-ranking-6500/policy.json','reports/regret-favorite-trajectory/learning-curve.csv',
        str(LOG.relative_to(ROOT))]
    plan=dict(version=1,steps=STEPS,seeds=list(SEEDS),scenarios=list(SCENARIOS),
        initializations=list(INITIALIZATIONS),coverages=list(COVERAGES),cells=60,observations=1200000,
        tokenizers=1000,batches_per_tokenizer=32,floor_fraction=.5,calibration=calibration,
        mean_rule='floor + (initial mean - floor) * exp(-rate * visits); floor = 0.5 * initial mean',
        sharing='Frozen: no learning. Individual: chosen tokenizer only. Family: every tokenizer with the same algorithm.',
        reward='Pre-update synthetic mean plus unchanged centered measured batch residual; negative observations retained',
        coverage='q = (1-epsilon) * softmax(logits) + epsilon/1000; sample q and differentiate log q',
        update='Original shared settings network, Adam .001, EMA .99, entropy zero. No predictor or regression intervention.',
        initialization='Fresh original seed43 or replayed real step6500 weights, Adam and EMA. Five paired future sampling/noise streams, not five independently trained real models.',
        limits='Explicit hypothetical student rules, no real LM updates; full within-family transfer and 50% floors are assumptions, not measured causal effects.',
        source_sha256={p:sha256(ROOT/p) for p in sources})
    target=STATE/'plan.json'
    if target.exists() and json.loads(target.read_text())!=plan:raise ValueError('Frozen simulation plan changed')
    atomic_json(target,plan);emit('Verifying inputs and reconstructing starting policy');replay(configs)
    tasks=[(scenario,initial,coverage,seed) for scenario in SCENARIOS for initial in INITIALIZATIONS
        for coverage in COVERAGES for seed in SEEDS]
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(run_cell,t):t for t in tasks};pending=set(futures)
        while pending:
            for f in list(pending):
                if f.done():f.result();pending.remove(f)
            completed=0;workers=[]
            for task in tasks:
                p=STATE/'cells'/cell_key(task)/'status.json'
                if not p.exists():continue
                row=json.loads(p.read_text());completed+=row['completed']
                if row['stage']=='running':workers.append(dict(id=cell_key(task),profile='local',
                    platform='Local · CPU',state=f"{task[0]} / {task[1]} / coverage {task[2]:.0%} / {row['completed']:,} of 20,000"))
            elapsed=time.monotonic()-started
            emit('Simulating student learning and teacher choices',completed,workers,
                elapsed_seconds=elapsed,eta_seconds=(1200000-completed)*elapsed/completed if completed else None)
            if pending:time.sleep(5)
    emit('Auditing every simulated observation',1200000)
    subprocess.run([sys.executable,'scripts/report_synthetic_student.py'],cwd=ROOT,check=True)
    emit('complete',1200000)


if __name__=='__main__':
    try:main()
    except Exception as error:
        STATE.mkdir(parents=True,exist_ok=True);emit('failed',error=str(error));raise
