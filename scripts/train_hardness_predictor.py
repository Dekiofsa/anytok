"""20,000 live-reward observations; only the small CPU predictor learns."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from frozen_training_reward import FrozenTrainingReward, FreshRewardStream, STATE, ROOT, REFERENCE
from hardness_predictor import HardnessPredictor
from package_training_inputs import atomic_json, sha256
from score_training_pool_ranking import pair_batch


def emit(stage,step,started,**fields):
    elapsed=time.monotonic()-started
    row=dict(stage=stage,completed=step,total=20000,at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed,platform='Local · RTX PRO 6000 96 GB',
        workers=[] if stage in ['complete','stopped','failed','scoring_complete'] else [dict(
            id='local-hardness-predictor',profile='local',platform='Local · RTX PRO 6000 96 GB',
            state=stage,task=f'Frozen 6.5k · predictor {step:,} / 20,000')],**fields)
    atomic_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def evaluate_reference(predictor,step):
    # The learner API never receives this table. These are evaluator-only metrics.
    receipt=json.loads((STATE/'artifacts.json').read_text())
    if sha256(REFERENCE/'tokenizer-ranking.csv')!=receipt['reference_csv_sha256']:
        raise ValueError('Evaluator reference changed')
    expected=np.empty(1000);seen=[]
    with (REFERENCE/'tokenizer-ranking.csv').open() as stream:
        for row in csv.DictReader(stream):
            index=int(row['index']);expected[index]=float(row['gap_bpb']);seen.append(index)
    if sorted(seen)!=list(range(1000)):raise ValueError('Complete reference required')
    values=predictor.predictions();ties=np.flatnonzero(values==values.max())
    top=set(np.argsort(-expected)[:100].tolist())
    gap=float(expected[ties].mean())
    record=dict(step=step,predictions=values.tolist(),counts=predictor.counts.tolist(),
        exploration_counts=predictor.exploration_counts.tolist(),greedy_indices=ties.tolist(),
        greedy_reference_gap=gap,nominal_75_25_reference_gap=.75*gap+.25*float(expected.mean()),
        greedy_mass_in_hardest100=sum(i in top for i in ties)/len(ties),
        prediction_rmse=float(np.sqrt(np.mean((values-expected)**2))),
        unique_visited=int((predictor.counts>0).sum()),
        unique_exploration=int((predictor.exploration_counts>0).sum()))
    atomic_json(STATE/'snapshots'/f'{step:06d}.json',record)
    return record


def one_update(predictor,stream,scorer):
    action=predictor.select();step=predictor.updates+1
    batch=stream.batch(action['index'])
    metrics=scorer.score(action['index'],*batch,2026111600+step-1)
    mse=predictor.observe(action['index'],metrics['gap_bpb'])
    return dict(step=step,index=action['index'],mode=action['mode'],
        predicted_gap_before=action['predicted_gap'],prediction_squared_error=mse,
        reserved_windows_skipped=stream.reserved_skipped,**metrics)


def preflight(scorer):
    started=time.monotonic();emit('preflight_parity',0,started)
    path=ROOT/'modal_results/regret_favorite_trajectory/training-panel.json.gz'
    with gzip.open(path,'rt') as stream:batch=json.load(stream)[0]
    native,_=scorer.pool.load(771)
    pairs=pair_batch(batch,native,scorer.canonical,scorer.trainer.make_paired_prefix_example)
    actual=scorer.score(771,*pairs,batch['dropout_seed'])
    previous=json.loads((ROOT/'modal_results/regret_favorite_trajectory/scores/006500.json').read_text())['batches'][0]
    keys=['raw_bpb','canonical_bpb','gap_bpb','transferred_nats','canonical_nats',
          'scored_tokens','scored_bytes','maximum_positions','paired_batch_sha256']
    differences={key:[previous[key],actual[key]] for key in keys if previous[key]!=actual[key]}
    if differences:raise ValueError('Same-GPU reward parity failed: '+json.dumps(differences))
    configs=[row['config'] for row in scorer.pool.entries]
    predictor=HardnessPredictor(configs);stream=FreshRewardStream(scorer)
    for _ in range(2):one_update(predictor,stream,scorer)
    saved=copy.deepcopy(predictor.state_dict());stream_state=copy.deepcopy(stream.state_dict())
    uninterrupted=one_update(predictor,stream,scorer)
    final=copy.deepcopy(predictor.state_dict());stream.close()
    predictor=HardnessPredictor(configs);predictor.load_state_dict(saved)
    stream=FreshRewardStream(scorer,stream_state)
    resumed=one_update(predictor,stream,scorer);stream.close()
    if uninterrupted!=resumed or any(not torch.equal(value,predictor.network.state_dict()[name])
                                    for name,value in final['network'].items()):
        raise ValueError('Real-reward checkpoint/resume failed to reproduce the next update')
    scorer.assert_frozen()
    atomic_json(STATE/'preflight.json',dict(same_gpu_reward_bitwise_equal=True,
        real_reward_resume_bitwise_equal=True,frozen_models_unchanged=True,
        parity_metrics=actual,real_forward_calls=scorer.calls,seconds=time.monotonic()-started,
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),plan_sha256=sha256(STATE/'plan.json')))
    emit('preflight_passed',0,started)


def save_checkpoint(predictor,stream,scorer,started):
    scorer.assert_frozen()
    step=predictor.updates;folder=STATE/'checkpoints'/f'step-{step:06d}'
    folder.mkdir(parents=True,exist_ok=True)
    payload=dict(step=step,predictor=predictor.state_dict(),stream=stream.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        plan_sha256=sha256(STATE/'plan.json'),elapsed_seconds=time.monotonic()-started)
    temporary=folder/'state.pt.tmp'
    torch.save(payload,temporary);temporary.replace(folder/'state.pt')
    atomic_json(folder/'manifest.json',dict(step=step,plan_sha256=payload['plan_sha256'],
        frozen_checkpoint_step=6500,files={'state.pt':dict(bytes=(folder/'state.pt').stat().st_size,
                                                       sha256=sha256(folder/'state.pt'))}))


def restore(predictor,scorer):
    folders=sorted((STATE/'checkpoints').glob('step-*'),reverse=True)
    for folder in folders:
        if not (folder/'manifest.json').exists():continue
        manifest=json.loads((folder/'manifest.json').read_text())
        state_path=folder/'state.pt'
        if sha256(state_path)!=manifest['files']['state.pt']['sha256']:
            raise ValueError('Predictor restart checksum changed')
        payload=torch.load(state_path,map_location='cpu',weights_only=True)
        if payload['plan_sha256']!=sha256(STATE/'plan.json'):
            raise ValueError('Resume settings or implementation changed')
        predictor.load_state_dict(payload['predictor'])
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        return FreshRewardStream(scorer,payload['stream']),payload['elapsed_seconds']
    return FreshRewardStream(scorer),0.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    lock=(STATE/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((STATE/'plan.json').read_text())
    for name,expected in plan['code_sha256'].items():
        if sha256(ROOT/'scripts'/name)!=expected:
            raise ValueError('The frozen pilot implementation changed: '+name)
    if (plan['updates']!=20000 or plan['checkpoint_step']!=6500 or plan['exploration_period']!=4
            or plan['warmup_updates']!=0 or plan['replay']):
        raise ValueError('This worker implements only the approved simplified pilot')
    started=time.monotonic();emit('loading_frozen_models',0,started)
    scorer=FrozenTrainingReward()
    if args.preflight_only:
        preflight(scorer);return
    if not (STATE/'preflight.json').exists():raise ValueError('Full-workload preflight required')
    predictor=HardnessPredictor([row['config'] for row in scorer.pool.entries])
    stream,previous_elapsed=restore(predictor,scorer);started=time.monotonic()-previous_elapsed
    journal=STATE/'observations.jsonl'
    if journal.exists():
        # Discard only uncheckpointed tail observations, which are re-evaluated
        # with real forwards after restoring the exact stream and selector state.
        rows=[json.loads(line) for line in journal.read_text().splitlines()]
        rows=[row for row in rows if row['step']<=predictor.updates]
        if [row['step'] for row in rows]!=list(range(1,predictor.updates+1)):
            raise ValueError('Committed observation journal has missing or duplicated updates')
        journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif predictor.updates:raise ValueError('Restart state exists without its observation journal')
    for path in (STATE/'snapshots').glob('*.json'):
        if int(path.stem)>predictor.updates:path.unlink()
    evaluate_reference(predictor,predictor.updates)
    recent=[]
    with journal.open('a',buffering=1) as log:
        while predictor.updates<20000:
            before=time.monotonic();row=one_update(predictor,stream,scorer)
            row.update(seconds=time.monotonic()-before,at_utc=datetime.now(timezone.utc).isoformat())
            # Reference labels enter only the separate monitoring snapshots.
            log.write(json.dumps(row)+'\n');recent.append(row['seconds']);recent=recent[-100:]
            step=predictor.updates
            emit('learning_predictor',step,started,index=row['index'],mode=row['mode'],
                observed_gap=row['gap_bpb'],predicted_gap=row['predicted_gap_before'],
                unique_visited=int((predictor.counts>0).sum()),
                exploration_visits=int(predictor.exploration_counts.sum()),
                eta_seconds=float(np.median(recent))*(20000-step),
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            if step%100==0:
                log.flush();os.fsync(log.fileno())
                snapshot=evaluate_reference(predictor,step)
                save_checkpoint(predictor,stream,scorer,started)
                subprocess.run([sys.executable,'scripts/report_hardness_predictor.py'],cwd=ROOT,check=True)
    scorer.assert_frozen();stream.close()
    if predictor.exploration_counts.tolist()!=[5]*1000:
        raise ValueError('The complete 20k pilot must give every tokenizer exactly five exploration visits')
    atomic_json(STATE/'completion-audit.json',dict(updates=predictor.updates,
        real_reward_forwards_this_process=scorer.calls,all_tokenizers_explored_five_times=True,
        frozen_parameter_versions_unchanged=True,checkpoint_step=6500,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),reference_labels_used_for_learning=False))
    emit('scoring_complete',20000,started,eta_seconds=0)


if __name__=='__main__':main()
