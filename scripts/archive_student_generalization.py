"""Persist synthetic measurement traces to HF after the local CPU study completes."""
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import tarfile
import time
import subprocess

from package_training_inputs import atomic_json,sha256

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/student_generalization'
REPORT=ROOT/'reports/student-generalization'
REPO='cminst/anytok-hardness-predictor-20k'
PREFIX='diagnostics/student-generalization'


def monitor(stage,**fields):
    atomic_json(STATE/'archive-status.json',dict(stage=stage,at_utc=datetime.now(timezone.utc).isoformat(),**fields))
    path=STATE/'status.json'
    value=json.loads(path.read_text())
    value.update(stage='complete' if stage=='complete' else 'Results complete · '+
        ('archive needs retry' if stage=='failed' else 'archiving measurements'),archive_state=stage,
        at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(path,value)


def main():
    while True:
        p=STATE/'status.json';status=json.loads(p.read_text()) if p.exists() else {}
        if status.get('stage')=='failed':return
        if status.get('stage')=='complete':break
        time.sleep(30)
    from huggingface_hub import HfApi
    api=HfApi();folder=STATE/'archive';folder.mkdir(exist_ok=True)
    paths=sorted((STATE/'cells').glob('*/trace.npz'))+sorted((STATE/'cells').glob('*/complete.json'))
    paths+=sorted((STATE/'audits').glob('*.json'))
    parts=[];current=[];size=0
    for path in paths:
        if current and size+path.stat().st_size>128*2**20:
            parts.append(current);current=[];size=0
        current.append(path);size+=path.stat().st_size
    if current:parts.append(current)
    receipts=[]
    for i,files in enumerate(parts):
        path=folder/f'traces-{i:03d}.tar';receipt=folder/f'traces-{i:03d}.json'
        if receipt.exists():
            receipts.append(json.loads(receipt.read_text()));continue
        if not path.exists():
            temporary=path.with_suffix('.partial')
            with tarfile.open(temporary,'w') as archive:
                for source in files:archive.add(source,arcname=str(source.relative_to(STATE)),recursive=False)
            temporary.replace(path)
        commit=api.upload_file(repo_id=REPO,path_or_fileobj=path,path_in_repo=PREFIX+'/'+path.name,
            commit_message='Synthetic student generalization: measurement trace archive')
        expected=sha256(path)
        stored=next(f for f in api.model_info(REPO,revision=commit.oid,files_metadata=True).siblings
                    if f.rfilename==PREFIX+'/'+path.name)
        if stored.size!=path.stat().st_size or (stored.lfs and stored.lfs.sha256!=expected):
            raise ValueError('Cloud measurement archive verification failed')
        if not stored.lfs:
            payload=path.read_bytes()
            if stored.blob_id!=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest():
                raise ValueError('Cloud archive Git checksum differs')
        record=dict(path=PREFIX+'/'+path.name,bytes=path.stat().st_size,sha256=expected,revision=commit.oid)
        atomic_json(receipt,record);receipts.append(record)
        monitor('uploading',completed=i+1,total=len(parts))
    result=dict(repo_id=REPO,prefix=PREFIX,verified=True,parts=receipts)
    atomic_json(REPORT/'archive-receipt.json',result)
    path=REPORT/'README.md';text=path.read_text()
    text+='\nRaw per-step traces are preserved in the verified cloud archive listed in archive-receipt.json; each trace checksum is in trace-manifest.json.\n'
    path.write_text(text)
    atomic_json(REPORT/'manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p))
        for p in REPORT.iterdir() if p.is_file() and p.name!='manifest.json'}))
    api.upload_folder(repo_id=REPO,folder_path=REPORT,path_in_repo=PREFIX+'/report',
        commit_message='Audited synthetic student generalization report')
    # Publish only this generated report. Never include unrelated user changes,
    # force-push, or switch branches in a background experiment.
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='zett-modal-20k':raise RuntimeError('Cloud archive verified; report publication awaits the original branch')
    relative=str(REPORT.relative_to(ROOT))
    subprocess.run(['git','add','--',relative],cwd=ROOT,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet','--',relative],cwd=ROOT)
    if changed.returncode:
        subprocess.run(['git','commit','--only','-m','Report audited student generalization boundary study','--',relative],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','zett-modal-20k'],cwd=ROOT,check=True)
    atomic_json(STATE/'publication.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        report=relative,cloud_archive_verified=True))
    monitor('complete',completed=len(parts),total=len(parts),verified=True)
    print(json.dumps(dict(stage='complete',repo_id=REPO,prefix=PREFIX,parts=len(parts))),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        STATE.mkdir(parents=True,exist_ok=True)
        monitor('failed',error=str(error))
        raise
