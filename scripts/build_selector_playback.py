"""Recover pre-choice beliefs by exact replay of the completed CPU simulations."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from hardness_predictor import HardnessPredictor, _features
from package_training_inputs import atomic_json, sha256
from selector_simulation import ROOT, SCENARIOS, METHODS, load_inputs

STATE=ROOT/'modal_results/selector_playback_25'
REPORT=ROOT/'reports/selector-simulations'
FRAMES=np.r_[1,np.arange(25,20001,25)]


def initialize():
    global CONFIGS
    torch.set_num_threads(1)
    CONFIGS=load_inputs()[0]


def replay(task):
    scenario,method,seed=task
    key=f'{scenario}-{method}-{seed}'
    folder=STATE/key;folder.mkdir(parents=True,exist_ok=True)
    source=REPORT/'traces'/f'{key}.npz'
    fingerprint=dict(trace_sha256=sha256(source),
        predictor_sha256=sha256(ROOT/'scripts/hardness_predictor.py'),
        policy_sha256=sha256(ROOT/'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'),
        simulation_plan_sha256=sha256(REPORT/'plan.json'),
        builder_sha256=sha256(Path(__file__)),torch=str(torch.__version__))
    if (folder/'complete.json').exists():
        previous=json.loads((folder/'complete.json').read_text())
        if previous['source']!=fingerprint or previous['values_sha256']!=sha256(folder/'values.npz'):
            raise ValueError('Playback source or cached beliefs changed')
        return key
    with np.load(source) as file:
        actions=file['actions'];rewards=file['rewards'];exploration=file['exploration']
    learner=(HardnessPredictor(CONFIGS,seed) if method=='predictor' else
             _features.SettingsTokenizerBandit(CONFIGS,seed=seed,entropy_weight=0.) if method=='reinforce' else None)
    values=np.empty((len(FRAMES),1000));frame=0
    rng=np.random.default_rng(seed)
    for offset in range(20000):
        step=offset+1
        if frame<len(FRAMES) and step==FRAMES[frame]:
            values[frame]=(learner.predictions() if method=='predictor' else
                           learner.probabilities() if method=='reinforce' else np.full(1000,.001))
            frame+=1
        if method=='predictor':
            action=learner.select()
            if action['index']!=int(actions[offset]) or int(action['mode']=='explore')!=int(exploration[offset]):
                raise ValueError(f'Replayed predictor action differs: {key} step {step}')
            learner.observe(action['index'],float(rewards[offset]))
        elif method=='reinforce':
            action=learner.sample()
            if action!=int(actions[offset]):raise ValueError(f'Replayed REINFORCE action differs: {key} step {step}')
            learner.update(action,float(rewards[offset]))
        elif int(rng.integers(1000))!=int(actions[offset]):
            raise ValueError('Uniform RNG replay differs')
        if step%1000==0:
            atomic_json(folder/'status.json',dict(completed=step,stage='running',method=method,scenario=scenario,seed=seed))
    if frame!=len(FRAMES) or not np.isfinite(values).all():raise ValueError('Incomplete beliefs')
    target=folder/'values.tmp.npz';np.savez_compressed(target,steps=FRAMES,values=values);target.replace(folder/'values.npz')
    atomic_json(folder/'complete.json',dict(source=fingerprint,values_sha256=sha256(folder/'values.npz'),
        steps_replayed=20000,frames=len(FRAMES),actions_match_every_step=True,
        timing='Beliefs before selecting at the displayed step; reward at that step has not entered learning.'))
    atomic_json(folder/'status.json',dict(completed=20000,stage='complete',method=method,scenario=scenario,seed=seed))
    return key


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[(scenario,method,seed) for scenario in SCENARIOS for seed in range(43,48) for method in METHODS]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        pending={pool.submit(replay,task) for task in tasks}
        while True:
            for future in list(pending):
                if future.done():future.result();pending.remove(future)
            completed=0;workers=[]
            for scenario,method,seed in tasks:
                path=STATE/f'{scenario}-{method}-{seed}'/'status.json'
                if not path.exists():continue
                state=json.loads(path.read_text());completed+=state['completed']
                if state['stage']=='running':workers.append(dict(id=f'{scenario}-{method}-{seed}',profile='local',platform='Local · CPU',state=f'{scenario} · {method} · seed {seed}'))
            row=dict(completed=completed,total=1500000,stage='running' if pending else 'complete',
                elapsed_seconds=time.monotonic()-started,platform='Local · CPU only',workers=workers,
                at_utc=datetime.now(timezone.utc).isoformat(),eta_seconds=(1500000-completed)*(time.monotonic()-started)/completed if completed else None)
            atomic_json(STATE/'status.json',row)
            with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if not pending:break
            time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE/'status.json',dict(stage='failed',state='failed',error=str(error),completed=0,total=1500000,workers=[],at_utc=datetime.now(timezone.utc).isoformat()))
        raise
