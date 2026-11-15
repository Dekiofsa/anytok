"""Stop only the requested allocation after its full checkpoint is committed.

Run on the cloud login host. Checkpoints never pass through the local caller.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/tillicum_regret'


def committed_checkpoint(folder, target):
    path = folder / 'manifest.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if (manifest['step'] != target or manifest['kind'] != 'full_restart_checkpoint'
            or 'state.pt' not in manifest['files']):
        raise ValueError('Pause requires the requested full resumable checkpoint')
    for name, metadata in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid checkpoint member')
        if (folder / relative).stat().st_size != metadata['bytes']:
            raise ValueError('Checkpoint member is not fully committed: ' + name)
    return manifest


def main():
    lock = (STATE / 'pause.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = json.loads((STATE / 'pause-request.json').read_text())
    target = request['target_step']; job = request['job_id']
    if not isinstance(target, int) or target % 2000 or not str(job).isdigit():
        raise ValueError('Expected a scheduled full checkpoint and one exact job ID')
    if (STATE / 'paused.json').exists():
        raise ValueError('Run already paused; do not repeat job control')
    def emit(stage, **fields):
        row = dict(stage=stage, target_step=target, job_id=job,
                   at_utc=datetime.now(timezone.utc).isoformat(), **fields)
        atomic_json(STATE / 'pause-status.json', row)
        with (STATE / 'pause-events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    emit('waiting_for_checkpoint')
    folder = STATE / 'run' / f'step-{target:06d}'
    next_queue_check = 0
    while not committed_checkpoint(folder, target):
        current = json.loads((STATE / 'current-job.json').read_text())
        if current['job_id'] != job:
            raise RuntimeError('Training allocation changed; refusing to stop another job')
        if time.monotonic() >= next_queue_check:
            queue = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                                   capture_output=True, text=True, check=True).stdout.strip()
            if queue not in ['RUNNING', 'COMPLETING']:
                raise RuntimeError('Training ended before the requested checkpoint: ' + queue)
            next_queue_check = time.monotonic() + 20
        time.sleep(.5)
    # The manifest is written last by the trainer. Release the GPU immediately
    # after that durable save; this also prevents the old live driver's handoff.
    emit('releasing_training_gpu', manifest_sha256=sha256(folder / 'manifest.json'))
    subprocess.run(['scancel', job], check=True)
    # First cancel the allocation (blocking afterok dependents), then stop all
    # processes so the old driver cannot schedule its own walltime successor.
    subprocess.run(['scancel', '--full', '--signal=KILL', job], check=False)
    deadline = time.monotonic() + 120
    while subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'], capture_output=True,
                         text=True, check=True).stdout.strip():
        if time.monotonic() > deadline:
            raise TimeoutError('Checkpoint saved, but training allocation has not exited')
        time.sleep(1)
    emit('verifying_cloud_backup', gpu_released=True)
    # Hash verification and upload run without a new GPU allocation. Retrying
    # uses the existing verified backup receipts and upload lock.
    from huggingface_hub import HfApi
    from tillicum_regret_job import backup_checkpoint
    api = HfApi(token=(STATE / '.hf-token').read_text().strip())
    for attempt in range(5):
        try:
            backup_checkpoint(api, folder)
            break
        except Exception as error:
            emit('backup_retry', gpu_released=True, attempt=attempt + 1, error=str(error))
            if attempt == 4:
                raise
            time.sleep(20)
    receipt = json.loads((STATE / 'backups' / f'step-{target:06d}.json').read_text())
    if receipt['step'] != target or receipt['kind'] != 'full_restart_checkpoint':
        raise ValueError('Backup receipt is not for the requested full checkpoint')
    paused = dict(step=target, job_id=job, request=request, checkpoint_backup=receipt,
                  gpu_released=True, at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(STATE / 'paused.json', paused)
    emit('paused', **{key: value for key, value in paused.items() if key not in ['job_id', 'at_utc']})


if __name__ == '__main__':
    main()
