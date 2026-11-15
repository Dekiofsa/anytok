"""Mirror only metadata into the existing regret run's dashboard and plots."""
from datetime import datetime, timezone
from collections import deque
import json
from pathlib import Path
import shlex
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pool_generalization import save_json
from run_tillicum_smoke import ssh, REMOTE
from monitoring.state import tail

STATE = ROOT / 'modal_results/anytok_pool_regret_rl_20k'
LOCAL = ROOT / 'modal_results/tillicum_regret'


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    pointer = LOCAL / 'mirror-offset.json'
    offset = json.loads(pointer.read_text())['offset'] if pointer.exists() else 0
    previous = None
    timings = deque(maxlen=100)
    for line in tail(STATE / 'committed-train.log', limit=150000).splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('stage') == 'train' and row.get('step', 0) > 5288:
            timings.append(row['seconds'])
    while True:
        try:
            response = ssh('cd ' + shlex.quote(REMOTE) + ' && .venv-memory/bin/python scripts/tillicum_regret_status.py --offset ' + str(offset))
            if response.returncode:
                raise RuntimeError(response.stderr[-800:])
            remote = json.loads(response.stdout)
            if remote['event_text']:
                for line in remote['event_text'].splitlines():
                    row = json.loads(line)
                    if row.get('stage') == 'train' and row.get('step', 0) > 5288:
                        timings.append(row['seconds'])
                with (STATE / 'committed-train.log').open('w' if remote['event_offset'] == 0 else 'a') as stream:
                    stream.write(remote['event_text'])
                offset = remote['next_offset']; save_json(pointer, dict(offset=offset))
            if remote.get('policy'):
                save_json(STATE / 'policy-monitor.json', remote['policy'])
            if remote.get('logs'):
                (STATE / 'tillicum-train.log').write_text(remote['logs'])
            data = json.loads((STATE / 'status.json').read_text())
            data.update(remote['status'] or {})
            data.update(at_utc=datetime.now(timezone.utc).isoformat(), profile='zlab', profiles=['zlab'],
                platform='Tillicum · H200', final_step=20000, state='running', error=None,
                observation_error=None, workers=[])
            job = remote.get('job') or {}; queue = remote.get('queue', '')
            data['slurm_job_id'] = job.get('job_id')
            data['slurm_state'] = queue
            data['checkpoint_backup'] = remote.get('backup')
            data['status_note'] = 'Resumed from full Modal step 5,288 on Tillicum H200; the later uncommitted Modal trajectory is archived. Reward remains regret, entropy weight 0.'
            data.pop('max_seconds', None)
            data['job_wall_seconds'] = 86400
            data['seconds_per_step'] = None
            data['eta_seconds'] = None
            if remote.get('backup') and remote['backup'].get('step'):
                data['latest_weights_step'] = remote['backup']['step']
                if remote['backup']['kind'] == 'full_restart_checkpoint':
                    data['latest_resumable_step'] = remote['backup']['step']
            if len(timings) >= 10:
                data['seconds_per_step'] = statistics.median(timings)
                data['eta_seconds'] = (20000 - data.get('step', 5288)) * data['seconds_per_step']
            if remote.get('complete'):
                data.update(stage='complete', state='complete', step=20000, eta_seconds=0)
            elif queue.startswith(('PENDING', 'CONFIGURING')):
                data.update(stage='waiting_for_gpu', state='queued')
            elif queue.startswith('RUNNING'):
                data['workers'] = [dict(id=job['job_id'], profile='zlab', platform='Tillicum · H200',
                    task=f"Regret step {data.get('step',5288):,} / 20,000", state='Training' if remote['status'] else 'Loading checkpoint',
                    log=str(STATE / 'tillicum-train.log'))]
                if not remote['status']:
                    data['stage'] = 'loading_checkpoint'
            elif any(word in queue for word in ['FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL']):
                data.update(stage='failed', state='failed', error='Slurm: ' + queue)
            elif not job:
                data['stage'] = 'preparing_tillicum_resume'
            request = remote.get('pause_request')
            paused = remote.get('paused')
            pause_status = remote.get('pause_status') or {}
            if request:
                data['pause_target_step'] = request['target_step']
                data['status_note'] = f"User requested a pause at full checkpoint {request['target_step']:,}; no automatic continuation."
                if data.get('seconds_per_step'):
                    data['eta_seconds'] = max(0, request['target_step'] - data.get('step', 0)) * data['seconds_per_step']
                for worker in data['workers']:
                    worker['task'] = f"Step {data.get('step', 0):,}; pauses at {request['target_step']:,}"
            if paused:
                data.update(stage='paused', state='stopped', step=paused['step'], workers=[],
                            eta_seconds=None, error=None, checkpoint_backup=paused['checkpoint_backup'],
                            latest_resumable_step=paused['step'], latest_weights_step=paused['step'])
            elif request and pause_status.get('gpu_released'):
                data.update(stage='pause_backup', state='stopped', step=request['target_step'],
                            workers=[], eta_seconds=None, error=None,
                            status_note='Training GPU released after the full checkpoint; verifying its cloud backup.')
            signature = (data.get('step'), data['stage'], data.get('slurm_job_id'),
                         json.dumps(data.get('checkpoint_backup'), sort_keys=True))
            save_json(STATE / 'status.json', data)
            if signature != previous:
                with (STATE / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(data) + '\n')
                print(json.dumps(data), flush=True); previous = signature
            if remote.get('complete'):
                return
            time.sleep(1 if remote['event_text'] else 15)
        except Exception as error:
            data = json.loads((STATE / 'status.json').read_text())
            data.update(observation_error=str(error), at_utc=datetime.now(timezone.utc).isoformat())
            save_json(STATE / 'status.json', data)
            print(json.dumps(dict(stage='observation_retry', error=str(error))), flush=True)
            time.sleep(15)


if __name__ == '__main__':
    main()
