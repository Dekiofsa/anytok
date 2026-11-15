"""Observe the complete frozen sweep, then audit, plot and publish its report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from package_training_inputs import atomic_json
from run_tillicum_smoke import ssh, remote_text, HOST, SOCKET, REMOTE

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_pool_ranking'


def emit(stage, completed, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed,
               total=1000, platform='Tillicum · H200', **fields)
    atomic_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def copy_metadata(relative, target):
    subprocess.run(['scp', '-o', 'ControlPath=' + SOCKET, HOST + ':' + REMOTE + '/' + relative,
                    str(target)], check=True, timeout=60, capture_output=True)


def main():
    master = json.loads((STATE / 'submission.json').read_text())['job_id']
    jobs = {str(i): master for i in range(4)}
    pointer = STATE / 'worker-jobs.json'
    if pointer.exists():
        jobs = json.loads(pointer.read_text())
    retry_pointer = STATE / 'worker-retries.json'
    retries = (json.loads(retry_pointer.read_text()) if retry_pointer.exists()
               else {str(i): 0 for i in range(4)})
    failures = 0
    while True:
        try:
            response = ssh('cd ' + shlex.quote(REMOTE) + ' && .venv-memory/bin/python scripts/pool_ranking_status.py')
            if response.returncode:
                raise ConnectionError(response.stderr[-1000:])
            data = json.loads(response.stdout)
            workers = []; remaining = []
            for shard in range(4):
                key = str(shard); item = data['workers'][key] or {}
                if item.get('stage') == 'complete':
                    continue
                job = jobs[key] + '_' + key
                query = ssh('squeue -r -h -j ' + job + ' -o ' + shlex.quote('%T|%M|%R'))
                queue = query.stdout.strip()
                if not queue:
                    account = ssh('sacct -n -P -j ' + job + ' --format=State,ExitCode').stdout
                    if 'TIMEOUT' in account and retries[key] < 2:
                        resumed = ssh('cd ' + shlex.quote(REMOTE) + ' && sbatch --parsable --array=' + key +
                                      ' scripts/score_training_pool_ranking.sbatch')
                        if resumed.returncode:
                            raise RuntimeError(resumed.stderr)
                        jobs[key] = resumed.stdout.strip(); retries[key] += 1
                        atomic_json(pointer, jobs); atomic_json(retry_pointer, retries)
                        queue = 'PENDING|0:00|Resuming saved batches'
                    elif 'TIMEOUT' in account:
                        raise RuntimeError('Scoring worker exhausted its two saved-batch resumptions: ' + job)
                    elif any(word in account for word in ['FAILED','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']):
                        raise RuntimeError('Scoring worker ended: ' + job + ' ' + account.strip())
                batches = item.get('batches_scored', 0)
                done = item.get('completed', 0)
                if done and item.get('elapsed_seconds'):
                    remaining.append(item['elapsed_seconds'] / done * (250 - done))
                if queue.startswith('RUNNING'):
                    workers.append(dict(id=job, profile='zlab', platform='Tillicum · H200',
                        state=f"Tokenizer #{item.get('index','—')} · batch {batches}/32",
                        task=f'{done}/250 complete', elapsed_seconds=item.get('elapsed_seconds')))
            eta = max(remaining) if remaining else None
            emit('scoring' if not data['complete'] else 'analyzing', data['completed'], workers=workers,
                 eta_seconds=eta, worker_jobs=jobs, checkpoint_step=6500)
            failures = 0
            if data['complete']:
                break
            time.sleep(20)
        except (ConnectionError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
            failures += 1
            print(json.dumps(dict(stage='observation_retry', error=str(error))), flush=True)
            if failures >= 20:
                raise RuntimeError('Repeated inability to observe scoring; remote jobs remain independent') from error
            time.sleep(15)
    copy_metadata('modal_results/regret_pool_ranking/all-scores.json.gz', STATE / 'all-scores.json.gz')
    copy_metadata('modal_results/local_memory/artifacts/input/pool/full-results.json', STATE / 'pool-index.json')
    subprocess.run([sys.executable, 'scripts/report_training_pool_ranking.py'], cwd=ROOT, check=True)
    report = ROOT / 'reports/regret-pool-ranking-6500'
    # Corpus snippets and replacement banks upload only from the cloud host.
    code = "from pathlib import Path;from huggingface_hub import HfApi;root=Path(" + repr(REMOTE) + ");api=HfApi(token=(root/'modal_results/tillicum_regret/.hf-token').read_text().strip());c=api.upload_folder(repo_id='cminst/anytok-regret-rl-20k',repo_type='model',folder_path=root/'modal_results/regret_pool_ranking',path_in_repo='diagnostics/policy-ranking-step-006500',allow_patterns=['training-panel.json.gz','panel-manifest.json','policy.json','run-plan.json','all-scores.json.gz']);print(c.oid)"
    publication = subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes',HOST,
        'cd '+shlex.quote(REMOTE)+' && .venv-memory/bin/python -c '+shlex.quote(code)],
        text=True,capture_output=True,timeout=600)
    if publication.returncode:
        raise RuntimeError('Cloud diagnostic persistence failed: ' + publication.stderr[-1000:])
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id='cminst/anytok-regret-rl-20k', repo_type='model', folder_path=report,
        path_in_repo='diagnostics/policy-ranking-step-006500/report', commit_message='Complete frozen training-reward policy ranking')
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch != 'zett-modal-20k':
        raise RuntimeError('Report ready; branch changed, so automatic Git publication stopped')
    subprocess.run(['git','add','reports/regret-pool-ranking-6500'],cwd=ROOT,check=True)
    staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(not path.startswith('reports/regret-pool-ranking-6500/') for path in staged):
        raise RuntimeError('Report ready; unrelated staged changes prevent automatic commit')
    subprocess.run(['git','commit','-m','Report complete step-6500 policy versus full-pool training hardness'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','zett-modal-20k'],cwd=ROOT,check=True)
    emit('complete',1000,workers=[],eta_seconds=0,checkpoint_step=6500,
         report='reports/regret-pool-ranking-6500',cloud_revision=publication.stdout.strip())


if __name__=='__main__':
    try:main()
    except Exception as error:
        old=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        emit('failed',old.get('completed',0),workers=[],error=str(error))
        raise
