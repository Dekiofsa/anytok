"""Direct Modal block download of a full restart; run on the cloud target."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import threading
import time

import requests
from package_training_inputs import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []; pending = []; completed = 0
    block = manifest['block_bytes']; thread = threading.local()
    for row in manifest['files']:
        path = args.output / row['path']
        if Path(row['path']).is_absolute() or '..' in Path(row['path']).parts:
            raise ValueError('Invalid restart path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256']:
            completed += row['bytes']; continue
        partial = path.with_suffix(path.suffix + '.partial')
        with partial.open('wb') as stream:
            stream.truncate(row['bytes'])
        jobs.extend((partial, i, url, min(block, row['bytes'] - i * block)) for i, url in enumerate(row['urls']))
        pending.append((row, path, partial))
    def fetch(item):
        path, index, url, length = item
        if not hasattr(thread, 'session'):
            thread.session = requests.Session()
        for attempt in range(5):
            try:
                response = thread.session.get(url, timeout=(15, 90)); response.raise_for_status()
                data = response.content
                if len(data) != length:
                    raise ValueError('Incomplete restart block')
                fd = os.open(path, os.O_WRONLY)
                try:
                    written = 0
                    while written < len(data):
                        written += os.pwrite(fd, data[written:], index * block + written)
                finally:
                    os.close(fd)
                return length
            except Exception:
                if attempt == 4:
                    raise RuntimeError('Restart block transfer failed') from None
                time.sleep(min(8, 2 ** attempt))
    def status(stage):
        atomic_json(args.output.parent / 'transfer-status.json', dict(stage=stage, completed_bytes=completed,
            total_bytes=sum(row['bytes'] for row in manifest['files']), source=manifest['source'], step=manifest['step']))
    status('copying_restart')
    with ThreadPoolExecutor(max_workers=16) as pool:
        last = time.monotonic()
        for future in as_completed([pool.submit(fetch, item) for item in jobs]):
            completed += future.result()
            if time.monotonic() - last > 5:
                status('copying_restart'); last = time.monotonic()
    status('verifying_restart')
    for row, path, partial in pending:
        if partial.stat().st_size != row['bytes'] or sha256(partial) != row['sha256']:
            raise ValueError('Restart checksum mismatch: ' + row['path'])
        partial.replace(path)
    # Keep the interrupted trajectory intact, but continue the committed history.
    events = args.output / 'events.jsonl'
    original = args.output / 'modal-events-through-5815.jsonl'
    if not original.exists():
        events.replace(original)
        rows = [json.loads(line) for line in original.read_text().splitlines()]
        events.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows
                                  if row.get('step', 0) <= manifest['step']))
    receipt = {k: v for k, v in manifest.items() if k != 'files'}
    receipt['files'] = [{k: v for k, v in row.items() if k != 'urls'} for row in manifest['files']]
    atomic_json(args.output.parent / 'restart-receipt.json', receipt)
    args.manifest.unlink()
    status('complete')
    print(json.dumps(dict(stage='complete', step=manifest['step'], files=len(manifest['files']), bytes=completed)), flush=True)


if __name__ == '__main__':
    main()
