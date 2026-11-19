"""Archive the trajectory report directly from Tillicum to persistent HF storage."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi
from package_training_inputs import atomic_json, sha256

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/predictor_trajectory'
REPORT=ROOT/'reports/predictor-uniform-trajectory'
REPO='cminst/anytok-hardness-predictor-20k'
PREFIX='diagnostics/predictor-uniform-trajectory'


def main():
    api=HfApi(token=(ROOT/'modal_results/tillicum_hardness/.hf-token').read_text().strip())
    manifest=json.loads((REPORT/'manifest.json').read_text())
    for name,row in manifest['files'].items():
        path=REPORT/name
        if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Report changed before archival: '+name)
    commit=api.upload_folder(repo_id=REPO,folder_path=REPORT,path_in_repo=PREFIX,
        commit_message='Matched predictor/uniform checkpoints at 2k, 4k, 6k and 8k')
    stored={r.rfilename:r for r in api.model_info(REPO,revision=commit.oid,files_metadata=True).siblings}
    verified={}
    for path in REPORT.iterdir():
        if not path.is_file():continue
        entry=stored[PREFIX+'/'+path.name]; content=path.read_bytes()
        if entry.size!=len(content):raise ValueError('Archived size mismatch')
        expected=hashlib.sha256(content).hexdigest()
        if entry.lfs:
            if entry.lfs.sha256!=expected:raise ValueError('Archived SHA mismatch')
        elif entry.blob_id!=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest():
            raise ValueError('Archived Git blob mismatch')
        verified[path.name]=dict(bytes=len(content),sha256=expected)
    receipt=dict(repo_id=REPO,prefix=PREFIX,revision=commit.oid,verified=True,files=verified)
    atomic_json(STATE/'archive-receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
