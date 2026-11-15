"""Read bounded training metadata on Tillicum; no checkpoint payloads returned."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/tillicum_regret'


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    events = STATE / 'run/events.jsonl'
    chunk = b''; offset = args.offset
    if events.exists():
        if events.stat().st_size < offset:
            offset = 0
        with events.open('rb') as stream:
            stream.seek(offset); chunk = stream.read(1 << 20)
        if chunk and not chunk.endswith(b'\n'):
            chunk = chunk.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in chunk else b''
    job = read(STATE / 'current-job.json')
    queue = ''
    if job:
        queue = subprocess.run(['squeue', '-h', '-j', job['job_id'], '-o', '%T|%M|%R'],
                               capture_output=True, text=True, timeout=15).stdout.strip()
        if not queue:
            rows = subprocess.run(['sacct', '-n', '-P', '-j', job['job_id'],
                                    '--format=State,ExitCode'], capture_output=True, text=True,
                                   timeout=15).stdout.strip().splitlines()
            queue = rows[0] if rows else ''
    result = dict(status=read(STATE / 'run/status.json'), transfer=read(STATE / 'transfer-status.json'),
        backup=read(STATE / 'backup-status.json'), policy=read(STATE / 'policy-monitor.json'),
        job=job, queue=queue, complete=read(STATE / 'complete.json'),
        pause_request=read(STATE / 'pause-request.json'), paused=read(STATE / 'paused.json'),
        pause_status=read(STATE / 'pause-status.json'),
        event_offset=offset, next_offset=offset + len(chunk), event_text=chunk.decode())
    if job:
        log = STATE / ('slurm-' + job['job_id'] + '.log')
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 24000)); result['logs'] = stream.read().decode(errors='replace')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
