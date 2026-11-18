"""Run on Tillicum: transfer only required evaluation artifacts directly from Modal."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

import requests
from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/four_way_8k'


def main():
    signed = STATE / 'signed-files.json'
    plan = json.loads(signed.read_text())
    block = plan['block_bytes']; jobs = []; pending = []; completed = 0
    total = sum(r['bytes'] for r in plan['files'])
    def emit(stage):
        atomic_json(STATE / 'transfer-status.json', dict(stage=stage, completed_bytes=completed,
            total_bytes=total, at_utc=datetime.now(timezone.utc).isoformat()))
    for row in plan['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid cloud path')
        path = STATE / relative; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == row['bytes'] and ('sha256' not in row or sha256(path) == row['sha256']):
            completed += row['bytes']; continue
        partial = path.with_name(path.name + '.partial')
        with partial.open('wb') as f:
            f.truncate(row['bytes'])
        pending.append((row, path, partial))
        if len(row['urls']) != (row['bytes'] + block - 1) // block:
            raise ValueError('Invalid source block layout')
        jobs.extend((partial, i, url, min(block, row['bytes'] - i * block)) for i,url in enumerate(row['urls']))
    thread = threading.local()
    def fetch(job):
        partial, index, url, length = job
        if not hasattr(thread, 'session'):
            thread.session = requests.Session()
        for attempt in range(5):
            try:
                response = thread.session.get(url, timeout=(15,90)); response.raise_for_status()
                data = response.content
                if len(data) != length:
                    raise ValueError('Incomplete cloud block')
                fd = os.open(partial, os.O_WRONLY)
                try:
                    written = 0
                    while written < length:
                        written += os.pwrite(fd, data[written:], index * block + written)
                finally:
                    os.close(fd)
                return length
            except Exception:
                if attempt == 4:
                    raise RuntimeError('Direct cloud transfer failed') from None
                time.sleep(min(8,2**attempt))
    emit('copying_cloud_artifacts')
    with ThreadPoolExecutor(max_workers=16) as pool:
        last = time.monotonic()
        for future in as_completed([pool.submit(fetch,j) for j in jobs]):
            completed += future.result()
            if time.monotonic() - last > 5:
                emit('copying_cloud_artifacts'); last = time.monotonic()
    emit('verifying_cloud_artifacts')
    for row,path,partial in pending:
        if partial.stat().st_size != row['bytes'] or ('sha256' in row and sha256(partial) != row['sha256']):
            raise ValueError('Cloud artifact checksum mismatch: ' + row['path'])
        partial.replace(path)
    atomic_json(STATE / 'transfer-receipt.json', dict(files=[{k:v for k,v in row.items() if k != 'urls'}
        for row in plan['files']], bytes=total, transport='Modal directly to Tillicum'))
    signed.unlink(); emit('complete')


if __name__ == '__main__':
    main()
