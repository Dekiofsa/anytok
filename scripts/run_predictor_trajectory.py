"""Monitor staging, evaluate six earlier checkpoints, and publish complete metadata."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from package_training_inputs import atomic_json
from run_tillicum_smoke import ssh, HOST, SOCKET

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/predictor_trajectory'
REMOTE='/gpfs/scrubbed/qxli2/anytok-hardness-20k'
JOBS=[f'{n}-{s:06d}' for s in [2000,4000,6000] for n in ['uniform','predictor']]


def status(stage,completed=124,**fields):
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=completed,
        total=496,cached_comparisons=124,platform='Tillicum · H200',profiles=['zlab'],workers=[])
    row.update(fields)
    atomic_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def remote(code):
    r=ssh('cd '+shlex.quote(REMOTE)+' && python3 -c '+shlex.quote(code))
    if r.returncode:raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout)


def launch(script,arguments,log):
    code='''import subprocess,pathlib,json
root=pathlib.Path(%r)
with (root/%r).open('ab') as stream:
 p=subprocess.Popen([str(root/'.venv-memory/bin/python'),'-u',%r]+%r,
  cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid)))''' % (REMOTE,log,script,arguments)
    return remote(code)


def snapshot():
    return remote('''import json,pathlib,subprocess
p=pathlib.Path('modal_results/predictor_trajectory')
def read(x):return json.loads(x.read_text()) if x.exists() else None
submission=read(p/'submission.json')
data=dict(transfer=read(p/'transfer-status.json'),preparation=read(p/'preparation.json'),
 statuses={x.stem:read(x) for x in (p/'status').glob('*.json')},submission=submission)
logs=[]
for f in [p/'transfer.log',p/'prepare.log']+sorted(p.glob('slurm-*.log')):
 if f.exists():
  with f.open('rb') as s:s.seek(max(0,f.stat().st_size-1500));logs.append(f.name+'\\n'+s.read().decode(errors='replace'))
data['logs']='\\n'.join(logs)
data['queue']=subprocess.run(['squeue','-h','-j',submission['job_id'],'-o','%i|%T|%M|%R'],capture_output=True,text=True).stdout if submission else ''
print(json.dumps(data))''')


def collect():
    r=ssh('cd '+shlex.quote(REMOTE)+' && .venv-memory/bin/python -u scripts/predictor_trajectory.py --collect')
    if r.returncode:raise RuntimeError(r.stderr[-1500:])
    for name in ['results.json.gz','plan.json','preparation.json']:
        subprocess.run(['scp','-o','ControlPath='+SOCKET,
            HOST+':'+REMOTE+'/modal_results/predictor_trajectory/'+name,str(STATE/name)],check=True)
    atomic_json(ROOT/'plans/predictor-uniform-trajectory-v1.json',json.loads((STATE/'plan.json').read_text()))
    subprocess.run([sys.executable,'scripts/report_predictor_trajectory.py'],cwd=ROOT,check=True)
    report=ROOT/'reports/predictor-uniform-trajectory'
    r=ssh('mkdir -p '+REMOTE+'/reports/predictor-uniform-trajectory')
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    subprocess.run(['scp','-o','ControlPath='+SOCKET,*[str(p) for p in report.iterdir() if p.is_file()],
        HOST+':'+REMOTE+'/reports/predictor-uniform-trajectory/'],check=True)
    launch('scripts/archive_predictor_trajectory.py',[],'modal_results/predictor_trajectory/archive.log')
    started=time.monotonic()
    while True:
        receipt=remote('from pathlib import Path;p=Path("modal_results/predictor_trajectory/archive-receipt.json");print(p.read_text() if p.exists() else "null")')
        if receipt:
            if not receipt['verified']:raise ValueError('Cloud archive not verified')
            atomic_json(STATE/'archive-receipt.json',receipt);return
        if time.monotonic()-started>900:raise TimeoutError('Results complete; archive verification exceeded 15 minutes')
        status('Scores and plots complete · verifying cloud archive',496);time.sleep(10)


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    try:
        while True:
            data=snapshot();(STATE/'remote.log').write_text(data['logs'])
            atomic_json(STATE/'remote-status.json',{k:v for k,v in data.items() if k!='logs'})
            transfer=data['transfer']
            if not transfer or transfer['stage']!='complete':
                workers=[dict(id='cloud-transfer',profile='zlab',platform='Modal → Tillicum',
                    state='Copying checkpoint weights',completed=(transfer or {}).get('completed_bytes',0),
                    total=(transfer or {}).get('total_bytes',25875483186))]
                status('Copying three uniform checkpoints directly from Modal to Tillicum',transfer=transfer,workers=workers)
                time.sleep(10);continue
            if not data['preparation']:
                started=STATE/'preparation-started.json'
                if not started.exists():
                    record=launch('scripts/predictor_trajectory.py',['--prepare'],'modal_results/predictor_trajectory/prepare.log')
                    atomic_json(started,dict(**record,at_utc=datetime.now(timezone.utc).isoformat()))
                prep=data['statuses'].get('preparation',{})
                if prep.get('stage')=='failed':raise RuntimeError(prep.get('error','Preparation failed'))
                status('Verifying exact checkpoints, original panel and full tokenizer inputs',preparation=prep)
                time.sleep(10);continue
            if not data['submission']:
                r=ssh('cd '+shlex.quote(REMOTE)+' && sbatch --parsable scripts/predictor_trajectory.sbatch')
                if r.returncode:raise RuntimeError(r.stderr[-1500:])
                record=dict(job_id=r.stdout.strip().split(';')[0],jobs=JOBS,kind='evaluation_only',
                    gpu='H200',array='0-5',at_utc=datetime.now(timezone.utc).isoformat())
                remote('from pathlib import Path;Path("modal_results/predictor_trajectory/submission.json").write_text('+repr(json.dumps(record))+');print("{}")')
                atomic_json(STATE/'submission.json',record)
                status('Queued · six earlier-checkpoint H200 evaluations',job_id=record['job_id']);time.sleep(10);continue
            atomic_json(STATE/'submission.json',data['submission'])
            workers=[];completed=124
            for job in JOBS:
                row=data['statuses'].get(job,{})
                if row.get('stage')=='failed':raise RuntimeError(job+': '+row.get('error','Evaluation failed'))
                completed+=row.get('completed',0)
                workers.append(dict(id=job,profile='zlab',platform='Tillicum · H200',state=row.get('stage','queued'),
                    completed=row.get('completed',0),total=62,eta_seconds=row.get('eta_seconds'),tokenizer=row.get('tokenizer')))
            if all(data['statuses'].get(j,{}).get('stage')=='complete' for j in JOBS):
                status('Auditing scores and plotting training trajectories',496);collect();status('complete',496);return
            if not data['queue'].strip():
                time.sleep(2);later=snapshot()
                if all(later['statuses'].get(j,{}).get('stage')=='complete' for j in JOBS):continue
                raise RuntimeError('Evaluation allocation ended before all checkpoints completed')
            etas=[w['eta_seconds'] for w in workers if w['eta_seconds'] is not None]
            status('Evaluating the identical 62-tokenizer panel',completed,workers=workers,
                job_id=data['submission']['job_id'],eta_seconds=max(etas) if etas else None)
            time.sleep(10)
    except Exception as error:
        previous=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        status('failed',previous.get('completed',124),error=str(error));raise


if __name__=='__main__':main()
