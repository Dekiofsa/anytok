"""Run the continued regret experiment, durable backups and safe walltime handoff."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import subprocess
import sys
import threading
import time

from package_training_inputs import atomic_json, sha256
from run_raw_bpb_training import command, prepare_code, run_and_monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE = ROOT / 'modal_results/tillicum_regret'
RUN = STATE / 'run'
INPUT = ROOT / 'modal_results/local_memory/artifacts/input'
REPO_ID = 'cminst/anytok-regret-rl-20k'


def latest_restart():
    paths = sorted(p for p in RUN.glob('step-*') if (p / 'state.pt').is_file()
                   and (p / 'manifest.json').is_file())
    if not paths:
        raise ValueError('No full checkpoint for the existing regret run')
    return int(paths[-1].name.removeprefix('step-'))


def backup_checkpoint(api, folder):
    with (STATE / 'backup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _backup_checkpoint(api, folder)


def _backup_checkpoint(api, folder):
    step = int(folder.name.removeprefix('step-'))
    receipt_path = STATE / 'backups' / (folder.name + '.json')
    if receipt_path.exists():
        return
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['step'] != step:
        raise ValueError('Checkpoint step mismatch')
    for name, expected in manifest['files'].items():
        if (folder / name).stat().st_size != expected['bytes'] or sha256(folder / name) != expected['sha256']:
            raise ValueError('Checkpoint changed before backup: ' + name)
    prefix = 'checkpoints/' + folder.name
    commit = api.upload_folder(repo_id=REPO_ID, repo_type='model', folder_path=folder,
                              path_in_repo=prefix, commit_message=f'Checkpoint {step}: {manifest["kind"]}')
    info = api.model_info(REPO_ID, revision=commit.oid, files_metadata=True)
    siblings = {row.rfilename: row for row in info.siblings}
    for name, expected in manifest['files'].items():
        stored = siblings[prefix + '/' + name]
        if stored.size != expected['bytes']:
            raise ValueError('Backup size mismatch: ' + name)
        if stored.lfs and stored.lfs.sha256 != expected['sha256']:
            raise ValueError('Backup hash mismatch: ' + name)
    receipt = dict(step=step, kind=manifest['kind'], revision=commit.oid, repo_id=REPO_ID,
                   manifest_sha256=sha256(folder / 'manifest.json'), prefix=prefix,
                   at_utc=datetime.now(timezone.utc).isoformat())
    # Publish configuration and the latest pointer only after payload verification.
    from huggingface_hub import CommitOperationAdd
    operations = [CommitOperationAdd(path_in_repo='metadata/' + name, path_or_fileobj=RUN / name)
                  for name in ['run_config.json', 'run_config.modal.json', 'initial_eval.json',
                               'heldout_tokenizers.json', 'events.jsonl'] if (RUN / name).exists()]
    previous_full = [json.loads(p.read_text())['step'] for p in (STATE / 'backups').glob('step-*.json')
                     if json.loads(p.read_text())['kind'] == 'full_restart_checkpoint']
    if manifest['kind'] == 'full_restart_checkpoint' and step >= max(previous_full, default=0):
        operations.append(CommitOperationAdd(path_in_repo='latest_restart.json',
                                             path_or_fileobj=(json.dumps(receipt, indent=2) + '\n').encode()))
    api.create_commit(repo_id=REPO_ID, repo_type='model', operations=operations,
                      commit_message=f'Verified metadata for checkpoint {step}')
    atomic_json(receipt_path, receipt)
    # Only derived distribution metadata leaves the cloud worker.
    import torch
    torch.set_num_threads(1)
    from monitoring.policy_summary import summarize
    state = torch.load(folder / 'policy.pt', map_location='cpu', weights_only=True)
    configs = json.loads((INPUT / 'pool/plan.json').read_text())['configs']
    summary = summarize(configs, state['logits'].softmax(0).tolist(), state['counts'].tolist(), step)
    path = STATE / 'policy-monitor.json'
    previous = json.loads(path.read_text()) if path.exists() else dict(snapshots=[])
    merged = {r['step']: r for r in previous['snapshots']}
    merged[step] = {**summary, 'profile': 'zlab', 'policy_sha256': manifest['files']['policy.pt']['sha256']}
    atomic_json(path, dict(snapshots=[merged[s] for s in sorted(merged)], error=None,
                          checked_at=datetime.now(timezone.utc).isoformat()))
    atomic_json(STATE / 'backup-status.json', dict(stage='verified', **receipt))
    print(json.dumps(dict(stage='backup_verified', **receipt)), flush=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from huggingface_hub import HfApi
    api = HfApi(token=(STATE / '.hf-token').read_text().strip())
    # Durable destination before consuming GPU time for more updates.
    start_step = latest_restart()
    backup_checkpoint(api, RUN / f'step-{start_step:06d}')
    code = prepare_code(STATE)
    args = SimpleNamespace(inputs=INPUT, output=RUN, stop_at_update=20000, resume=True,
                           reward_objective='regret', memory_execution='optimized')
    cmd = command(args, code)
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/resume_regret_tillicum.py'), '--code', str(code), '--', *cmd[3:]]
    stop_backup = threading.Event()
    def backups():
        while not stop_backup.is_set():
            try:
                for folder in sorted(RUN.glob('step-*')):
                    if (folder / 'manifest.json').is_file():
                        backup_checkpoint(api, folder)
            except Exception as error:
                atomic_json(STATE / 'backup-status.json', dict(stage='retrying', error=str(error),
                    at_utc=datetime.now(timezone.utc).isoformat()))
            stop_backup.wait(20)
    worker = threading.Thread(target=backups, daemon=True)
    worker.start()
    result = run_and_monitor(cmd, args, {**os.environ, 'TOKENIZERS_PARALLELISM':'false',
        'ANYTOK_RUN_PLATFORM':'Tillicum · H200', 'ANYTOK_RUN_PROFILE':'zlab'})
    stop_backup.set(); worker.join(timeout=180)
    if worker.is_alive():
        raise TimeoutError('Backup has not finished; no next job will be submitted')
    if result:
        raise RuntimeError(f'Trainer exited {result}; preserve all checkpoint files and investigate')
    for folder in sorted(RUN.glob('step-*')):
        if (folder / 'manifest.json').is_file():
            backup_checkpoint(api, folder)
    step = latest_restart()
    if step <= start_step:
        raise RuntimeError('No committed training progress; refusing another allocation')
    if step < 20000:
        # Graceful walltime stop only; no retries after training errors.
        submission = subprocess.run(['sbatch', '--parsable', '--dependency=afterok:' + os.environ['SLURM_JOB_ID'],
                                    'scripts/tillicum_regret_train.sbatch'],
                                    cwd=ROOT, text=True, capture_output=True, check=True)
        atomic_json(STATE / 'current-job.json', dict(job_id=submission.stdout.strip(), resume_step=step,
                    at_utc=datetime.now(timezone.utc).isoformat(), reason='Graceful walltime handoff'))
    else:
        atomic_json(STATE / 'complete.json', dict(step=step, repo_id=REPO_ID,
                    at_utc=datetime.now(timezone.utc).isoformat()))
        (STATE / '.hf-token').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
