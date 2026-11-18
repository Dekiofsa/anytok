"""Archive evaluation losses and provenance directly from Tillicum to Hugging Face."""
import json
from pathlib import Path

from huggingface_hub import HfApi
from package_training_inputs import atomic_json,sha256

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/four_way_8k'
REPO='cminst/anytok-hardness-predictor-20k'
PREFIX='diagnostics/four-way-8k'


def main():
    api=HfApi(token=(ROOT/'modal_results/tillicum_hardness/.hf-token').read_text().strip())
    receipt=STATE/'archive-receipt.json'
    if receipt.exists():
        print(receipt.read_text(),flush=True);return
    records={}
    for name in ['results.json.gz','plan.json','preparation.json','panel-metadata.json']:
        path=STATE/name
        commit=api.upload_file(repo_id=REPO,path_or_fileobj=path,path_in_repo=PREFIX+'/'+name,
            commit_message='Eight-thousand-step four-model evaluation: '+name)
        stored=next(f for f in api.model_info(REPO,revision=commit.oid,files_metadata=True).siblings
                    if f.rfilename==PREFIX+'/'+name)
        expected=sha256(path)
        if stored.size!=path.stat().st_size or (stored.lfs and stored.lfs.sha256!=expected):
            raise ValueError('Evaluation archive checksum mismatch')
        if not stored.lfs:
            import hashlib
            data=path.read_bytes()
            if stored.blob_id!=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest():
                raise ValueError('Evaluation metadata checksum mismatch')
        records[name]=dict(bytes=path.stat().st_size,sha256=expected,revision=commit.oid)
    record=dict(repo_id=REPO,prefix=PREFIX,files=records,verified=True)
    atomic_json(receipt,record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
