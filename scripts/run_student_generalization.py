"""Run the complete, predeclared synthetic-student generalization study on CPUs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from generalized_student import GeneralizedStudent,worlds,METHODS,SEEDS,STEPS
from hardness_predictor import _features
from package_training_inputs import atomic_json,sha256
from selector_simulation import ROOT,load_inputs
from synthetic_student import mixed_probabilities,update_policy

STATE=ROOT/'modal_results/student_generalization'


def emit(stage,completed,total,workers=None,**fields):
    value=dict(stage=stage,completed=completed,total=total,workers=workers or [],
        at_utc=datetime.now(timezone.utc).isoformat(),platform='Local · CPU only',**fields)
    atomic_json(STATE/'status.json',value)
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def initialize():
    global INPUTS,PLAN
    torch.set_num_threads(1);INPUTS=load_inputs()
    PLAN=json.loads((STATE/'plan.json').read_text())


def key(task):
    world,method,seed=task
    return f'{world["id"]}-{method}-seed{seed}'


def run_cell(task):
    world,method,seed=task;folder=STATE/'cells'/key(task);folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'complete.json';identity=dict(world_id=world['id'],method=method,seed=seed,
        steps=STEPS,plan_sha256=PLAN['plan_sha256'])
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if any(saved.get(k)!=v for k,v in identity.items()) or saved['trace_sha256']!=sha256(folder/'trace.npz'):
            raise ValueError('Saved cell belongs to another plan or its trace changed')
        return saved
    configs,rewards,calibration=INPUTS
    env=GeneralizedStudent(rewards,configs,calibration,world,seed)
    policy=(_features.SettingsTokenizerBandit(configs,seed=seed,entropy_weight=0.)
        if method in ['reinforce','coverage'] else None)
    rng=policy.rng if policy is not None else np.random.default_rng(seed)
    epsilon=.25 if method=='coverage' else 0.
    floats=['reward','chosen_mean','best_mean','policy_mean','behavior_mean','student_mean',
        'mastery','favorite_probability','favorite_behavior_probability','advantage',
        'action_probability','action_probability_after']
    trace={k:np.empty(STEPS,dtype=np.float64) for k in floats}
    for name in ['action','favorite','favorite_rank','selected_rank']:
        trace[name]=np.empty(STEPS,dtype=np.int16)
    trace['stale']=np.empty(STEPS,dtype=np.bool_)
    started=time.monotonic();events=folder/'events.jsonl';events.write_text('')
    for offset in range(STEPS):
        step=offset+1;means=env.means();best=float(means.max())
        if policy is not None:p,q=mixed_probabilities(policy,epsilon)
        elif method=='uniform':p=q=np.full(1000,.001)
        else:
            # This evaluator-only comparator knows current means, not future
            # student progress. Its labels never enter the learned selectors.
            p=q=(means==best).astype(float);p/=p.sum()
        action=int(rng.choice(1000,p=q));favorite=int(p.argmax())
        reward=env.observe(action,step,means)
        result=(update_policy(policy,action,reward,epsilon) if policy is not None else
                dict(policy_advantage_bpb=0.,policy_selected_probability=p[action]))
        favorite_rank=1+int((means>means[favorite]).sum())
        values=dict(action=action,favorite=favorite,favorite_rank=favorite_rank,
            selected_rank=1+int((means>means[action]).sum()),reward=reward,chosen_mean=means[action],
            best_mean=best,policy_mean=float(p@means),behavior_mean=float(q@means),
            student_mean=float(means.mean()),mastery=float(((env.base-means)/(env.base-env.floor)).mean()),
            favorite_probability=p[favorite],favorite_behavior_probability=q[favorite],
            advantage=result['policy_advantage_bpb'],action_probability=p[action],
            action_probability_after=result['policy_selected_probability'],
            stale=bool(q[favorite]>=.5 and favorite_rank>100))
        for name,value in values.items():trace[name][offset]=value
        if step%100==0:
            sl=slice(step-100,step)
            row=dict(**identity,step=step,at_utc=datetime.now(timezone.utc).isoformat(),
                seconds=time.monotonic()-started,
                choice_gap=float((trace['best_mean'][sl]-trace['chosen_mean'][sl]).mean()),
                learned_gap=float((trace['best_mean'][sl]-trace['policy_mean'][sl]).mean()),
                stale_fraction=float(trace['stale'][sl].mean()),student_mean=float(means.mean()))
            atomic_json(folder/'status.json',dict(stage='running',completed=step,**row))
            with events.open('a') as f:f.write(json.dumps(row)+'\n')
    temporary=folder/'trace.tmp.npz';np.savez_compressed(temporary,**trace);temporary.replace(folder/'trace.npz')
    late=slice(-2000,None)
    summary=dict(**identity,trace_sha256=sha256(folder/'trace.npz'),seconds=time.monotonic()-started,
        final_choice_gap=float((trace['best_mean'][late]-trace['chosen_mean'][late]).mean()),
        final_learned_gap=float((trace['best_mean'][late]-trace['policy_mean'][late]).mean()),
        whole_run_choice_gap=float((trace['best_mean']-trace['chosen_mean']).mean()),
        final_stale_fraction=float(trace['stale'][late].mean()),
        persistent_failure=bool(trace['stale'][late].mean()>.5),
        first_stale_step=int(np.flatnonzero(trace['stale'])[0]+1) if trace['stale'].any() else None,
        total_stale_steps=int(trace['stale'].sum()),final_student_mean=float(env.means().mean()),
        final_mastery=float(((env.base-env.means())/(env.base-env.floor)).mean()),
        unique_selected=int(np.unique(trace['action']).size))
    atomic_json(receipt,summary);atomic_json(folder/'status.json',dict(stage='complete',completed=STEPS,**summary))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--scope',choices=['full','map'],default='full')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from pool_generalization import digest
    selected=[w for w in worlds() if args.scope=='full' or w['study']=='map']
    inputs=['reports/regret-pool-ranking-6500/all-scores.json.gz',
        'reports/regret-pool-ranking-6500/tokenizer-ranking.csv','reports/regret-pool-ranking-6500/pool-index.json',
        'reports/regret-favorite-trajectory/learning-curve.csv','scripts/generalized_student.py',
        'scripts/run_student_generalization.py','scripts/selector_simulation.py','scripts/synthetic_student.py',
        'scripts/hardness_predictor.py','runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py']
    plan=dict(version=1,scope=args.scope,worlds=selected,methods=list(METHODS),seeds=list(SEEDS),
        steps=STEPS,cells=len(selected)*len(METHODS)*len(SEEDS),
        observations=len(selected)*len(METHODS)*len(SEEDS)*STEPS,tokenizers=1000,batches=32,
        teacher_initialization='Five new independent original-network initializations, seeds48–52; paired across conditions',
        objective='Current difficulty selection; the current-hardest comparator is not an optimal long-term curriculum',
        practice_control='Shared budget conserves total effective practice weight, not necessarily the exact loss reduction',
        rate_control='Reference rate fixed using the prior 50% floor; varying floors does not recalibrate rate',
        grouping_seed=20261118,layout_seed=20261118,
        runtime={p:importlib.metadata.version(p) for p in ['torch','numpy']},
        source_sha256={p:sha256(ROOT/p) for p in inputs})
    plan['plan_sha256']=digest(plan)
    target=STATE/'plan.json'
    if target.exists() and json.loads(target.read_text())!=plan:raise ValueError('Frozen study changed')
    atomic_json(target,plan)
    tasks=[(w,m,s) for w in selected for m in METHODS for s in SEEDS]
    total=plan['observations'];emit('Preparing controlled student worlds',0,total)
    if args.prepare_only:return
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    started=time.monotonic();last_report=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(run_cell,t):t for t in tasks};pending=set(futures)
        while pending:
            for f in list(pending):
                if f.done():f.result();pending.remove(f)
            completed=0;workers=[]
            for task in tasks:
                p=STATE/'cells'/key(task)/'status.json'
                if not p.exists():continue
                row=json.loads(p.read_text());completed+=row['completed']
                if row['stage']=='running':workers.append(dict(id=key(task),profile='local',platform='Local · CPU',
                    state=f"{task[0]['label']} / {task[1]} / {row['completed']:,} of 20,000"))
            elapsed=time.monotonic()-started
            emit('Testing where the teacher keeps up or gets stuck',completed,total,workers,
                elapsed_seconds=elapsed,eta_seconds=(total-completed)*elapsed/completed if completed else None)
            if time.monotonic()-last_report>=60:
                result=subprocess.run([sys.executable,'scripts/report_student_generalization.py','--partial'],cwd=ROOT)
                if result.returncode:
                    atomic_json(STATE/'plot-error.json',dict(error='Live plot refresh failed; simulations continue',
                        at_utc=datetime.now(timezone.utc).isoformat()))
                last_report=time.monotonic()
            if pending:time.sleep(10)
    emit('Auditing all simulated observations',total,total)
    subprocess.run([sys.executable,'scripts/report_student_generalization.py','--workers',str(args.workers)],cwd=ROOT,check=True)
    emit('complete',total,total)


if __name__=='__main__':
    try:main()
    except Exception as error:
        STATE.mkdir(parents=True,exist_ok=True)
        p=STATE/'status.json';old=json.loads(p.read_text()) if p.exists() else {}
        emit('failed',old.get('completed',0),old.get('total',0),error=str(error));raise
