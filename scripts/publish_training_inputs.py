"""Upload an already verified cloud archive directly to Hugging Face."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from package_training_inputs import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.folder / 'bundle-receipt.json'
    deadline = time.monotonic() + 1800
    while not receipt_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Verified archive was not completed within thirty minutes')
        time.sleep(5)
    from huggingface_hub import HfApi
    receipt = json.loads(receipt_path.read_text())
    archive = args.folder / receipt['archive']
    if archive.stat().st_size != receipt['bytes'] or sha256(archive) != receipt['sha256']:
        raise ValueError('Archive changed after verification')
    api = HfApi(token=args.token_file.read_text().strip())
    print(json.dumps(dict(stage='uploading', at_utc=datetime.now(timezone.utc).isoformat(),
                         repo_id=args.repo_id, bytes=receipt['bytes'])), flush=True)
    commit = api.upload_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=args.folder,
        allow_patterns=['*.tar.gz', '*.sha256', 'BUNDLE_MANIFEST.json', 'bundle-receipt.json',
                        'README.md', 'training-config.json'],
        commit_message='Publish exact AnyTok 1,000-tokenizer training input bundle')
    info = api.dataset_info(args.repo_id, revision=commit.oid, files_metadata=True)
    stored = next(item for item in info.siblings if item.rfilename == archive.name)
    if not stored.lfs or stored.lfs.sha256 != receipt['sha256'] or stored.lfs.size != receipt['bytes']:
        raise ValueError('Published archive hash/size differs from verified cloud archive')
    result = dict(stage='complete', at_utc=datetime.now(timezone.utc).isoformat(), repo_id=args.repo_id,
                  revision=commit.oid, repo_url=f'https://huggingface.co/datasets/{args.repo_id}',
                  archive_url=f'https://huggingface.co/datasets/{args.repo_id}/resolve/{commit.oid}/{archive.name}',
                  sha256=receipt['sha256'], bytes=receipt['bytes'], input_files=receipt['input_files'])
    atomic_json(args.folder / 'publication.json', result)
    args.token_file.unlink()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
