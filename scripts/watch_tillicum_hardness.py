"""Read-only metadata mirror for the independent Tillicum predictor run."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import statistics
import time

from package_training_inputs import atomic_json
from run_tillicum_smoke import ssh

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tillicum_hardness'
REMOTE='/gpfs/scrubbed/qxli2/anytok-hardness-20k'


def remote_status(offset):
    code='''import json,subprocess
from pathlib import Path
r=Path('modal_results/tillicum_hardness')
def read(name):
 p=r/name
 return json.loads(p.read_text()) if p.exists() else None
p=r/'run/events.jsonl';offset=OFFSET;chunk=b''
if p.exists():
 if p.stat().st_size<offset:offset=0
 with p.open('rb') as f:f.seek(offset);chunk=f.read(1<<20)
 if chunk and not chunk.endswith(b'\\n'):chunk=chunk.rsplit(b'\\n',1)[0]+b'\\n' if b'\\n' in chunk else b''
job=read('current-job.json');queue=''
if job:
 queue=subprocess.run(['squeue','-h','-j',str(job['job_id']),'-o','%T|%M|%R'],capture_output=True,text=True,timeout=15).stdout.strip()
 if not queue:
  rows=subprocess.run(['sacct','-n','-P','-j',str(job['job_id']),'--format=State,ExitCode'],capture_output=True,text=True,timeout=15).stdout.strip().splitlines()
  queue=rows[0] if rows else ''
log=r/('slurm-'+str(job['job_id'])+'.log') if job else r/'none'
tail=''
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-16000));tail=f.read().decode(errors='replace')
print(json.dumps(dict(status=read('status.json'),backup=read('backup-status.json'),job=job,queue=queue,
 complete=read('complete.json'),policy=read('policy-monitor.json'),preflight=read('preflight-verification.json'),
 launch=read('launch-verification.json'),event_offset=offset,next_offset=offset+len(chunk),event_text=chunk.decode(),logs=tail)))
'''.replace('OFFSET',str(offset))
    result=ssh('cd '+shlex.quote(REMOTE)+' && .venv-memory/bin/python -c '+shlex.quote(code))
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    return json.loads(result.stdout)


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    pointer=STATE/'mirror-offset.json';offset=json.loads(pointer.read_text())['offset'] if pointer.exists() else 0
    while True:
        try:
            data=remote_status(offset)
            if data['event_text']:
                with (STATE/'train.log').open('w' if data['event_offset']==0 else 'a') as stream:stream.write(data['event_text'])
                offset=data['next_offset'];atomic_json(pointer,dict(offset=offset))
            for key,name in [('policy','policy-monitor.json'),('preflight','preflight-verification.json'),('launch','launch-verification.json')]:
                if data.get(key):atomic_json(STATE/name,data[key])
            (STATE/'slurm.log').write_text(data['logs'])
            status=data['status'] or dict(stage='waiting_for_gpu',step=0)
            status.update(state='running',total=20000,platform='Tillicum · H200',profile='zlab',profiles=['zlab'],
                at_utc=datetime.now(timezone.utc).isoformat(),checkpoint_backup=data['backup'],slurm_state=data['queue'],
                slurm_job_id=(data['job'] or {}).get('job_id'),observation_error=None)
            if data['queue'].startswith(('PENDING','CONFIGURING')):status.update(stage='waiting_for_gpu',state='queued',workers=[])
            elif any(s in data['queue'] for s in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']):status.update(stage='failed',state='failed',workers=[])
            elif status['stage'] in ['complete','failed','stopped']:status.update(state=status['stage'],workers=[])
            if data['complete']:status.update(stage='complete',state='complete',step=20000,workers=[],eta_seconds=0)
            if (STATE/'train.log').exists():
                rows=[]
                for line in (STATE/'train.log').read_text().splitlines()[-300:]:
                    row=json.loads(line)
                    if row.get('stage')=='train':rows.append(row['seconds'])
                if len(rows)>=10 and status['stage']=='training':
                    seconds=statistics.median(rows[-100:]);status.update(seconds_per_step=seconds,eta_seconds=seconds*(20000-status.get('step',0)))
            atomic_json(STATE/'status.json',status)
            with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(status)+'\n')
            print(json.dumps(status),flush=True)
            if data['complete']:return
            time.sleep(2 if data['event_text'] else 15)
        except Exception as error:
            previous=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else dict(stage='observing',step=0,total=20000,workers=[])
            previous.update(observation_error=str(error),at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(STATE/'status.json',previous);print(json.dumps(dict(observation_error=str(error))),flush=True);time.sleep(15)


if __name__=='__main__':main()
