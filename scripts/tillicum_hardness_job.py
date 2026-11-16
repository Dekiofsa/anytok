"""One H200: exact-resume preflight, fresh 20k training, verified cloud backups."""
from collections import deque
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from package_training_inputs import atomic_json,sha256
from run_hardness_training import ROOT,STATE,INPUT,command
from check_checkpoint_rounding import compare_safetensors
sys.path.insert(0,str(ROOT))

RUN=STATE/'run'
REPO='cminst/anytok-hardness-predictor-20k'
STOP=False


def status(stage,step=0,**fields):
    row=dict(stage=stage,step=step,total=20000,platform='Tillicum · H200',
        at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    row['workers']=[] if stage in ['complete','stopped','failed'] else [dict(
        id=os.environ.get('SLURM_JOB_ID','preparing'),profile='zlab',platform='Tillicum · H200',state=stage)]
    atomic_json(STATE/'status.json',row)
    print(json.dumps(row),flush=True)


def train_rows(folder):
    return [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()
            if json.loads(line).get('stage')=='train']


def scientific(row):
    return {key:value for key,value in row.items() if key not in ['seconds','peak_gib']}


def run(cmd,phase,reference=None):
    global STOP
    process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    def signal_child(received,_):
        global STOP
        STOP=True
        if process.poll() is None:process.send_signal(received)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,signal_child)
    step=0;error=None
    for line in process.stdout:
        print(line,end='',flush=True)
        try:row=json.loads(line)
        except ValueError:continue
        if not isinstance(row,dict) or 'stage' not in row:continue
        if phase=='training':
            step=row.get('step',step)
            if reference and row['stage']=='train' and step<=3:
                if scientific(row)!=scientific(reference[step-1]):
                    error=f'Fresh training does not reproduce preflight at step {step}'
                    process.send_signal(signal.SIGTERM)
                elif step==3:
                    atomic_json(STATE/'launch-verification.json',dict(first_three_training_updates_match_preflight=True,
                        source_manifest_sha256=sha256(CODE/'source-manifest.json')))
            status('training' if row['stage'] in ['train','eval','checkpoint'] else row['stage'],step)
        else:status('gpu_preflight',0,preflight_phase=phase,preflight_step=row.get('step',0))
    result=process.wait()
    if result or error:raise RuntimeError(error or f'{phase} trainer exited {result}')


def preflight():
    receipt=STATE/'preflight-verification.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value['source_manifest_sha256']!=sha256(CODE/'source-manifest.json'):
            raise ValueError('Preflight code changed')
        return train_rows(STATE/'preflight/uninterrupted')
    uninterrupted=STATE/'preflight/uninterrupted'
    resumed=STATE/'preflight/resumed'
    if uninterrupted.exists() or resumed.exists():
        for folder in [uninterrupted,resumed]:
            final=folder/'step-000003'
            if not (final/'manifest.json').exists():
                raise RuntimeError('Interrupted preflight is incomplete; preserve it for inspection')
            manifest=json.loads((final/'manifest.json').read_text())
            for name,expected in manifest['files'].items():
                if sha256(final/name)!=expected['sha256']:raise ValueError('Original preflight checkpoint changed')
    else:
        run(command(uninterrupted,CODE,stop=3),'uninterrupted_three')
        if STOP:raise RuntimeError('Preflight interrupted; no full training launched')
        run(command(resumed,CODE,stop=2),'two_before_resume')
        if STOP:raise RuntimeError('Preflight interrupted; no full training launched')
        run(command(resumed,CODE,stop=3,resume=True),'third_after_resume')
    a,b=train_rows(uninterrupted),train_rows(resumed)
    if len(a)!=3 or len(b)!=3 or [scientific(r) for r in a]!=[scientific(r) for r in b]:
        raise ValueError('GPU checkpoint resume changed input/reward/gradient/predictor metrics')
    for row in a:
        if row['paired_valid_examples']!=128 or row['paired_maximum_positions']>128:
            raise ValueError('Full-workload paired scoring preflight failed')
        if abs(row['train_bpb']-row['paired_canonical_bpb']-row['selector_reward_bpb'])>1e-12:
            raise ValueError('Regret reward changed')
    import torch
    for name in ['lm.pt','policy.pt']:
        paths=[folder/'step-000003'/name for folder in [uninterrupted,resumed]]
        values=[torch.load(path,map_location='cpu',weights_only=True,mmap=True) for path in paths]
        key='network' if name=='policy.pt' else None
        x,y=[v[key] if key else v for v in values]
        if x.keys()!=y.keys() or any(not torch.equal(x[k],y[k]) for k in x):
            raise ValueError('GPU resume changed saved model/predictor weights: '+name)
        del values,x,y
    checkpoint=lambda folder:folder/'step-000003/hypernet/model.safetensors'
    rounding=compare_safetensors(checkpoint(uninterrupted),checkpoint(resumed))
    repeat_rounding=None
    if not rounding['bitwise_identical']:
        repeat=STATE/'preflight/independent_uninterrupted'
        if repeat.exists():raise ValueError('Independent rounding diagnostic already exists; inspect it before retrying')
        run(command(repeat,CODE,stop=3),'independent_rounding_check')
        if [scientific(row) for row in train_rows(repeat)]!=[scientific(row) for row in a]:
            raise ValueError('Independent GPU run changed scientific metrics')
        repeat_rounding=dict(versus_uninterrupted=compare_safetensors(checkpoint(uninterrupted),checkpoint(repeat)),
            versus_resumed=compare_safetensors(checkpoint(resumed),checkpoint(repeat)))
    value=dict(source_manifest_sha256=sha256(CODE/'source-manifest.json'),
        full_batches=128,microbatch=32,maximum_positions=128,
        first_three_metrics_identical_after_resume=True,lm_and_predictor_weights_bitwise_identical=True,
        hypernetwork_rounding=rounding,independent_repeat=repeat_rounding,
        actual_reward='paired pre-update BPB gap',cuda=torch.version.cuda,torch=str(torch.__version__),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip(),
        at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(receipt,value);status('preflight_passed',0)
    return a


def latest_restart():
    return max([int(p.name.removeprefix('step-')) for p in RUN.glob('step-*')
                if (p/'state.pt').exists() and (p/'manifest.json').exists()],default=-1)


def backup(api,folder):
    receipt=STATE/'backups'/(folder.name+'.json')
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if finish_summary(folder,record):
            latest=sorted((STATE/'backups').glob('step-*.json'))[-1]
            atomic_json(STATE/'backup-status.json',dict(stage='verified',**json.loads(latest.read_text())))
        return
    manifest=json.loads((folder/'manifest.json').read_text());step=manifest['step']
    for name,expected in manifest['files'].items():
        if (folder/name).stat().st_size!=expected['bytes'] or sha256(folder/name)!=expected['sha256']:
            raise ValueError('Checkpoint changed before backup: '+name)
    commit=api.upload_folder(repo_id=REPO,folder_path=folder,path_in_repo='checkpoints/'+folder.name,
        commit_message=f'Checkpoint {step}: {manifest["kind"]}')
    files={r.rfilename:r for r in api.model_info(REPO,revision=commit.oid,files_metadata=True).siblings}
    for name,expected in manifest['files'].items():
        stored=files['checkpoints/'+folder.name+'/'+name]
        if stored.size!=expected['bytes'] or (stored.lfs and stored.lfs.sha256!=expected['sha256']):
            raise ValueError('Cloud checkpoint verification failed: '+name)
        if not stored.lfs:
            payload=(folder/name).read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if stored.blob_id!=blob:raise ValueError('Cloud metadata checksum failed: '+name)
    record=dict(step=step,kind=manifest['kind'],repo_id=REPO,prefix='checkpoints/'+folder.name,
        revision=commit.oid,manifest_sha256=sha256(folder/'manifest.json'),at_utc=datetime.now(timezone.utc).isoformat())
    from huggingface_hub import CommitOperationAdd
    operations=[CommitOperationAdd(path_in_repo='metadata/'+name,path_or_fileobj=str(RUN/name))
                for name in ['run_config.json','heldout_tokenizers.json','initial_eval.json'] if (RUN/name).exists()]
    if manifest['kind']=='full_restart_checkpoint':
        operations.append(CommitOperationAdd(path_in_repo='latest_restart.json',path_or_fileobj=(json.dumps(record,indent=2)+'\n').encode()))
    api.create_commit(repo_id=REPO,operations=operations,commit_message=f'Verified checkpoint metadata {step}')
    atomic_json(receipt,record);atomic_json(STATE/'backup-status.json',dict(stage='verified',**record))
    finish_summary(folder,record)


def finish_summary(folder,record):
    """A verified payload must not prevent retrying derived metadata after an error."""
    step=record['step']
    if step:
        path=STATE/'policy-monitor.json';previous=json.loads(path.read_text()) if path.exists() else dict(snapshots=[])
        if any(row['step']==step for row in previous['snapshots']):return
        import torch
        from monitoring.policy_summary import summarize
        value=torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
        configs=json.loads((INPUT/'pool/plan.json').read_text())['configs']
        frequencies=(value['recent_counts'].double()/value['recent_steps']).tolist()
        summary=summarize(configs,frequencies,value['counts'].tolist(),step)
        summary.update(kind='recent_actual_selection_frequencies',recent_steps=value['recent_steps'],
            predictions=value['predictions'].tolist(),exploration_visits=int(value['exploration_counts'].sum()))
        snapshots={row['step']:row for row in previous['snapshots']};snapshots[step]=summary
        atomic_json(path,dict(snapshots=[snapshots[s] for s in sorted(snapshots)],error=None,checked_at=datetime.now(timezone.utc).isoformat()))
        return True


def main():
    global CODE
    STATE.mkdir(parents=True,exist_ok=True)
    if (STATE/'stop-request.json').exists():status('stopped');return
    lock=(STATE/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for package,expected in [('torch','2.15.0'),('transformers','5.15.1'),('tokenizers','0.22.2')]:
        if importlib.metadata.version(package).split('+')[0]!=expected:
            raise ValueError('Validated runtime version changed: '+package)
    plan=json.loads((STATE/'plan.json').read_text());CODE=Path(plan['code'])
    if sha256(CODE/'source-manifest.json')!=plan['source_manifest_sha256']:raise ValueError('Source manifest changed')
    for name,digest in json.loads((CODE/'source-manifest.json').read_text()).items():
        if sha256(CODE/name)!=digest:raise ValueError('Pinned training source changed: '+name)
    atomic_json(STATE/'current-job.json',dict(job_id=os.environ['SLURM_JOB_ID'],at_utc=datetime.now(timezone.utc).isoformat()))
    from huggingface_hub import HfApi
    api=HfApi(token=(STATE/'.hf-token').read_text().strip())
    api.model_info(REPO)
    reference=preflight()
    if STOP:raise RuntimeError('Stop requested during preflight')
    stop_backup=threading.Event()
    def uploads():
        while not stop_backup.is_set():
            try:
                for folder in sorted(RUN.glob('step-*')):
                    if (folder/'manifest.json').exists():backup(api,folder)
            except Exception as error:
                atomic_json(STATE/'backup-status.json',dict(stage='retrying',error=str(error),at_utc=datetime.now(timezone.utc).isoformat()))
            stop_backup.wait(20)
    uploader=threading.Thread(target=uploads,daemon=True);uploader.start()
    start=latest_restart()
    try:run(command(RUN,CODE,resume=start>=0),'training',reference=reference if start<3 else None)
    finally:stop_backup.set();uploader.join(timeout=180)
    if uploader.is_alive():raise TimeoutError('Checkpoint upload still active; preserve allocation state')
    for folder in sorted(RUN.glob('step-*')):
        if (folder/'manifest.json').exists():backup(api,folder)
    step=latest_restart()
    if step>=20000:
        atomic_json(STATE/'complete.json',dict(step=step,repo=REPO,at_utc=datetime.now(timezone.utc).isoformat()))
        status('complete',step)
    elif STOP and not (STATE/'stop-request.json').exists():
        if step<=start:raise RuntimeError('No durable progress; no automatic continuation')
        result=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+os.environ['SLURM_JOB_ID'],
            'scripts/tillicum_hardness_train.sbatch'],cwd=ROOT,capture_output=True,text=True,check=True)
        atomic_json(STATE/'current-job.json',dict(job_id=result.stdout.strip().split(';')[0],resume_step=step,
            reason='Graceful walltime handoff',at_utc=datetime.now(timezone.utc).isoformat()))
        status('waiting_for_next_allocation',step)
    else:status('stopped',step)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',max(0,latest_restart()),error=str(error))
        raise
