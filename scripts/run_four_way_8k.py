"""Prepare, submit once, and monitor only the approved four-checkpoint evaluation."""
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from package_training_inputs import atomic_json
from run_tillicum_smoke import ssh,HOST,SOCKET

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/four_way_8k'
REMOTE='/gpfs/scrubbed/qxli2/anytok-hardness-20k'
MODELS=['uniform','raw_bpb','regret','predictor']


def status(stage,completed=0,**fields):
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=completed,
        total=248,platform='Tillicum · H200',profile='zlab',profiles=['zlab'],workers=[])
    row.update(fields)
    atomic_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def remote(code):
    result=ssh('cd '+shlex.quote(REMOTE)+' && python3 -c '+shlex.quote(code))
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    return json.loads(result.stdout)


def snapshot():
    return remote('''import json,pathlib,subprocess
p=pathlib.Path('modal_results/four_way_8k')
def read(path):return json.loads(path.read_text()) if path.exists() else None
data=dict(transfer=read(p/'transfer-status.json'),prepared=read(p/'preparation.json'),
 statuses={f.stem:read(f) for f in (p/'status').glob('*.json')},submission=read(p/'submission.json'))
logs=[]
for f in [p/'transfer.log',p/'prepare.log']+sorted(p.glob('slurm-*.log')):
 if f.exists():
  with f.open('rb') as s:s.seek(max(0,f.stat().st_size-2000));logs.append(f.name+'\\n'+s.read().decode(errors='replace'))
data['logs']='\\n'.join(logs)
data['queue']=subprocess.run(['squeue','-h','-u','qxli2','-o','%i|%T|%M|%R|%j'],capture_output=True,text=True).stdout
print(json.dumps(data))''')


def collect():
    remote('''import gzip,json,pathlib
p=pathlib.Path('modal_results/four_way_8k');plan=json.loads((p/'plan.json').read_text())
models={}
for name in ['uniform','raw_bpb','regret','predictor']:
 f=p/'scores'/name
 models[name]=dict(canonical=json.loads((f/'canonical.json').read_text()),
  tokenizers=[json.loads((f/(c['id']+'.json')).read_text()) for c in plan['candidates']])
data=dict(plan=plan,preparation=json.loads((p/'preparation.json').read_text()),
 rows=json.loads((p/'panel-metadata.json').read_text()),models=models)
(p/'results.json.gz').write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
print(json.dumps(dict(bytes=(p/'results.json.gz').stat().st_size)))''')
    subprocess.run(['scp','-o','ControlPath='+SOCKET,HOST+':'+REMOTE+'/modal_results/four_way_8k/results.json.gz',
                    str(STATE/'results.json.gz')],check=True)
    subprocess.run([sys.executable,'scripts/report_four_way_8k.py'],cwd=ROOT,check=True)
    # Results are metadata, but archive the original loss bundle cloud-to-cloud.
    result=ssh('cd '+shlex.quote(REMOTE)+' && nohup .venv-memory/bin/python -u scripts/archive_four_way_8k.py > '+
        shlex.quote(REMOTE+'/modal_results/four_way_8k/archive.log')+' 2>&1 < /dev/null &')
    if result.returncode:raise RuntimeError('Could not start cloud evaluation archive')
    started=time.monotonic()
    while True:
        archived=remote('import pathlib,json; p=pathlib.Path("modal_results/four_way_8k/archive-receipt.json");print(p.read_text() if p.exists() else "null")')
        if archived:
            atomic_json(STATE/'archive-receipt.json',archived);break
        if time.monotonic()-started>900:
            raise TimeoutError('Scores and plots saved; cloud archive has not verified within 15 minutes')
        status('Scores complete · verifying cloud archive',248)
        time.sleep(10)


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    try:
        while True:
            data=snapshot();(STATE/'remote.log').write_text(data['logs'])
            atomic_json(STATE/'remote-status.json',{k:v for k,v in data.items() if k!='logs'})
            transfer=data['transfer']
            if not transfer or transfer['stage']!='complete':
                status('Preparing · copying checkpoint and tokenizer files directly from Modal',
                    transfer=transfer);time.sleep(10);continue
            if not data['prepared']:
                if not (STATE/'preparation-started.json').exists():
                    code='nohup '+REMOTE+'/.venv-memory/bin/python -u '+REMOTE+'/scripts/four_way_8k.py --prepare > '+REMOTE+'/modal_results/four_way_8k/prepare.log 2>&1 < /dev/null & echo $!'
                    result=ssh(code)
                    if result.returncode:raise RuntimeError(result.stderr[-1000:])
                    atomic_json(STATE/'preparation-started.json',dict(pid=result.stdout.strip()))
                prep=data['statuses'].get('preparation',{})
                if prep.get('stage')=='failed':raise RuntimeError(prep.get('error','Preparation failed'))
                status('Preparing · '+prep.get('stage','auditing shared panel'),preparation=prep)
                time.sleep(10);continue
            atomic_json(STATE/'preparation.json',data['prepared'])
            if not data['submission']:
                result=ssh('cd '+shlex.quote(REMOTE)+' && sbatch --parsable scripts/four_way_8k.sbatch')
                if result.returncode:raise RuntimeError(result.stderr[-1000:])
                job=result.stdout.strip().split(';')[0]
                record=dict(job_id=job,at_utc=datetime.now(timezone.utc).isoformat(),kind='evaluation_only',
                    models=MODELS,steps=8000,gpu='H200',array='0-3',wall_limit='01:00:00')
                remote('import pathlib,json; p=pathlib.Path("modal_results/four_way_8k/submission.json");p.write_text('+repr(json.dumps(record))+');print("{}")')
                atomic_json(STATE/'submission.json',record)
                status('Queued · four frozen-checkpoint H200 evaluations',job_id=job);time.sleep(10);continue
            atomic_json(STATE/'submission.json',data['submission'])
            workers=[];completed=0
            for name in MODELS:
                row=data['statuses'].get(name,{})
                if row.get('stage')=='failed':raise RuntimeError(name+': '+row.get('error','Evaluation failed'))
                completed+=row.get('completed',0)
                workers.append(dict(id=name,profile='zlab',platform='Tillicum · H200',
                    state=row.get('stage','queued'),completed=row.get('completed',0),total=62,
                    eta_seconds=row.get('eta_seconds'),tokenizer=row.get('tokenizer')))
            if all(data['statuses'].get(n,{}).get('stage')=='complete' for n in MODELS):
                status('Collecting scores and building comparison plots',completed)
                collect();status('complete',248);return
            if not data['queue'].strip():
                # A job may finish between reading its status and reading Slurm.
                time.sleep(2)
                refreshed=snapshot()
                if all(refreshed['statuses'].get(n,{}).get('stage')=='complete' for n in MODELS):
                    continue
                raise RuntimeError('Evaluation allocation ended before all 248 pairs completed; inspect Slurm logs')
            etas=[r.get('eta_seconds') for r in workers if r.get('eta_seconds') is not None]
            stage=('Scoring the fixed 62-tokenizer panel' if any(w['state']!='queued' for w in workers)
                   else 'Queued · waiting for H200 allocations')
            status(stage,completed,workers=workers,
                eta_seconds=max(etas) if etas else None,job_id=data['submission']['job_id'])
            time.sleep(10)
    except Exception as error:
        previous=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        status('failed',previous.get('completed',0),error=str(error));raise


if __name__=='__main__':main()
