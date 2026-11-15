"""Prepare the exact local copy and supervise the approved frozen pilot."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from huggingface_hub import HfApi, hf_hub_download
from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/hardness_predictor_6500'
INPUT = ROOT / 'modal_results/local_memory/artifacts/input'
REPO = 'cminst/anytok-regret-rl-20k'
STARTED = time.monotonic()


def emit(stage, completed=0, **fields):
    row = dict(stage=stage, completed=completed, total=20000,
        at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-STARTED,
        platform='Local · RTX PRO 6000 96 GB',
        workers=[] if stage in ['complete','failed'] else [dict(id='local-hardness-predictor',
            profile='local',platform='Local · RTX PRO 6000 96 GB',state=stage)], **fields)
    STATE.mkdir(parents=True,exist_ok=True)
    atomic_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def prepare():
    receipt = STATE/'artifacts.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    emit('verifying_original_inputs')
    audit=json.loads((INPUT/'rl-input-audit.json').read_text())
    for number,row in enumerate(audit['files'],1):
        relative=Path(row['path'])
        if relative.parts[0]!='input' or '..' in relative.parts:
            raise ValueError('Invalid source artifact path')
        path=INPUT/relative.relative_to('input')
        if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Original input changed: '+str(relative))
        if number%100==0:emit('verifying_original_inputs',files_verified=number)
    inventory=json.loads((ROOT/'modal_results/regret_favorite_trajectory/inventory.json').read_text())
    source=next(row for row in inventory['checkpoints'] if row['step']==6500)
    checkpoint=STATE/'artifacts'/source['prefix']
    for name in ['lm.pt','hypernet/config.json','hypernet/model.safetensors']:
        target=checkpoint/name;expected=source['manifest']['files'][name]
        if not (target.exists() and target.stat().st_size==expected['bytes'] and sha256(target)==expected['sha256']):
            emit('downloading_frozen_weights',file=name,file_bytes=expected['bytes'])
            hf_hub_download(source['repo'],source['prefix']+'/'+name,revision=source['revision'],
                            local_dir=STATE/'artifacts')
        if target.stat().st_size!=expected['bytes'] or sha256(target)!=expected['sha256']:
            raise ValueError('Frozen checkpoint download mismatch: '+name)
    reference=ROOT/'reports/regret-pool-ranking-6500'
    panel=ROOT/'modal_results/regret_favorite_trajectory/training-panel.json.gz'
    panel_manifest=json.loads((reference/'panel-manifest.json').read_text())
    if sha256(panel)!=panel_manifest['panel_sha256']:
        raise ValueError('Reference raw-window metadata changed')
    data=dict(checkpoint=str(checkpoint),source=source,input_audit_sha256=sha256(INPUT/'rl-input-audit.json'),
        input_files_verified=len(audit['files']),reference_csv_sha256=sha256(reference/'tokenizer-ranking.csv'),
        reference_analysis_sha256=sha256(reference/'analysis.json'),panel_sha256=sha256(panel),
        verified_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(receipt,data);emit('artifacts_ready')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if (STATE/'stop-request.json').exists() and not args.prepare_only:
        parser.error('This pilot was stopped by the user; preserve its saved state and use an explicitly approved new run.')
    STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare()
    if args.prepare_only:return
    code_files=['hardness_predictor.py','frozen_training_reward.py','train_hardness_predictor.py']
    hashes={name:sha256(ROOT/'scripts'/name) for name in code_files}
    plan=dict(version=1,checkpoint_step=6500,updates=20000,tokenizers=1000,
        batch_size=128,microbatch_size=32,maximum_positions=128,continuation='variable',
        dropout_active=True,data_seed=42,predictor_seed=43,cuda_seed_base=2026111600,
        exploration_period=4,warmup_updates=0,replay=False,learning_rate=.001,
        loss='one squared-error update on the current observed paired-regret reward',
        reward='pre-update transferred BPB minus paired canonical BPB',code_sha256=hashes,
        ranking_role='evaluator only; never supplied to the learner',
        artifact_receipt_sha256=sha256(STATE/'artifacts.json'))
    path=STATE/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Approved pilot settings or implementation changed')
    atomic_json(path,plan)
    env={**os.environ,'TOKENIZERS_PARALLELISM':'false','PYTORCH_ALLOC_CONF':'expandable_segments:True',
         'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    python=ROOT/'.venv-memory/bin/python'
    stop=threading.Event()
    def backups():
        api=HfApi()
        while not stop.is_set():
            try:
                for folder in sorted((STATE/'checkpoints').glob('step-*')):
                    step=int(folder.name.removeprefix('step-'))
                    receipt=STATE/'backups'/(folder.name+'.json')
                    if step%500 or receipt.exists() or not (folder/'manifest.json').exists():continue
                    manifest=json.loads((folder/'manifest.json').read_text())
                    for name,expected in manifest['files'].items():
                        if sha256(folder/name)!=expected['sha256']:raise ValueError('Pilot checkpoint changed before backup')
                    revision=api.upload_folder(repo_id=REPO,folder_path=folder,
                        path_in_repo='diagnostics/hardness-predictor-step-006500/checkpoints/'+folder.name,
                        commit_message=f'Online hardness predictor: resumable step {step}')
                    atomic_json(receipt,dict(step=step,revision=revision.oid))
                atomic_json(STATE/'backup-status.json',dict(stage='ok',at_utc=datetime.now(timezone.utc).isoformat()))
            except Exception as error:
                atomic_json(STATE/'backup-status.json',dict(stage='retrying',error=str(error),
                    at_utc=datetime.now(timezone.utc).isoformat()))
            stop.wait(20)
    uploader=threading.Thread(target=backups,daemon=True);uploader.start()
    try:
        for extra in [['--preflight-only'],[]]:
            if extra and (STATE/'preflight.json').exists():
                previous=json.loads((STATE/'preflight.json').read_text())
                if previous.get('plan_sha256')==sha256(path):continue
            command=[str(python),'-u','scripts/train_hardness_predictor.py',*extra]
            subprocess.run(command,cwd=ROOT,env=env,check=True)
        deadline=time.monotonic()+600
        while not (STATE/'backups/step-020000.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Final predictor checkpoint is local; cloud backup has not completed')
            time.sleep(5)
    finally:
        stop.set();uploader.join(timeout=120)
    subprocess.run([str(python),'scripts/report_hardness_predictor.py'],cwd=ROOT,check=True)
    report=ROOT/'reports/hardness-predictor-6500'
    HfApi().upload_folder(repo_id=REPO,folder_path=report,
        path_in_repo='diagnostics/hardness-predictor-step-006500/report',
        commit_message='Complete real-reward frozen hardness-predictor pilot')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='zett-modal-20k':raise RuntimeError('Report ready; branch changed before publication')
    subprocess.run(['git','add','reports/hardness-predictor-6500'],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(not name.startswith('reports/hardness-predictor-6500/') for name in staged):
        raise RuntimeError('Report ready; unrelated staged changes prevent publication')
    subprocess.run(['git','commit','-m','Report frozen hardness-predictor pilot with live rewards'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin',branch],cwd=ROOT,check=True)
    emit('complete',20000,report='reports/hardness-predictor-6500')


if __name__=='__main__':
    try:main()
    except Exception as error:
        old=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        emit('failed',old.get('completed',0),error=str(error));raise
