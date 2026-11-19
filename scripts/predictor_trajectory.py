"""Evaluate two training trajectories on the exact frozen four-way 8k panel."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from package_training_inputs import atomic_json, sha256
from pool_generalization import digest

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'modal_results/predictor_trajectory'
PARENT = ROOT/'modal_results/four_way_8k'
INPUT = ROOT/'modal_results/local_memory/artifacts/input'
STEPS = [2000, 4000, 6000, 8000]
NAMES = ['uniform', 'predictor']
FILES = ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']


def emit(job, stage, completed=0, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
        completed=completed, total=62, platform='Tillicum · H200', job=job, **fields)
    atomic_json(STATE/'status'/f'{job}.json', row)
    path = STATE/'events'/f'{job}.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def check_sources(plan):
    for relative, expected in plan['parent']['evaluator_sources'].items():
        if sha256(ROOT/'scripts'/relative) != expected:
            raise ValueError('Original evaluator source changed: '+relative)
    for package, version in plan['parent']['runtime'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Original runtime changed: '+package)
    code = Path(plan['parent']['code'])
    if sha256(code/'source-manifest.json') != plan['parent']['code_manifest_sha256']:
        raise ValueError('Pinned model source manifest changed')
    for relative, expected in json.loads((code/'source-manifest.json').read_text()).items():
        if sha256(code/relative) != expected:
            raise ValueError('Pinned model implementation changed')
    for relative, expected in plan['source_hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError('Trajectory evaluator changed after preparation')


def prepare():
    source = json.loads((STATE/'source-plan.json').read_text())
    parent = json.loads((PARENT/'plan.json').read_text())
    if digest(parent) != source['parent_plan_sha256']:
        raise ValueError('Original 8k plan changed')
    rows = json.loads((PARENT/'panel.json').read_text())
    if digest(rows) != parent['panel_sha256'] or digest(rows) != source['parent_panel_sha256']:
        raise ValueError('Original prefixes or targets changed')
    models = source['models']
    for step in STEPS[:-1]:
        checkpoint = ROOT/f'modal_results/tillicum_hardness/run/step-{step:06d}'
        manifest = json.loads((checkpoint/'manifest.json').read_text())
        models.append(dict(name='predictor', step=step, job=f'predictor-{step:06d}',
            manifest=manifest, checkpoint=str(checkpoint), source=dict(platform='Tillicum')))
    for name in NAMES:
        old = next(m for m in parent['models'] if m['name'] == name)
        models.append(dict(name=name, step=8000, job=f'{name}-008000',
            manifest=old['manifest'], checkpoint=old['checkpoint'],
            cached=True, source=dict(parent_plan_sha256=digest(parent))))
    if {(m['name'], m['step']) for m in models} != {(n,s) for n in NAMES for s in STEPS}:
        raise ValueError('All eight checkpoints required')
    for m in models:
        emit('preparation', 'verifying_checkpoints', checkpoint=m['job'])
        if m['manifest']['step'] != m['step']:
            raise ValueError('Wrong checkpoint step')
        for relative in FILES:
            p = Path(m['checkpoint'])/relative; expected = m['manifest']['files'][relative]
            if p.stat().st_size != expected['bytes'] or sha256(p) != expected['sha256']:
                raise ValueError('Checkpoint verification failed: '+m['job']+'/'+relative)
    for relative in ['model/tokenizer.json', 'model/config.json', 'model/model.safetensors']:
        if sha256(INPUT/relative) != parent['input_hashes'][relative]:
            raise ValueError('Original model input changed')
    for c in parent['candidates']:
        if sha256(c['path']) != c['sha256']:
            raise ValueError('Tokenizer changed: '+c['id'])
        for field in ['forms', 'priors']:
            if sha256(PARENT/'features'/c['id']/(field+'.npy')) != c['features'][field+'_sha256']:
                raise ValueError('Native feature map changed')
    plan = dict(source, models=sorted(models,key=lambda m:(m['step'],m['name'])), parent=parent,
        panel_sha256=parent['panel_sha256'], panel_audit_sha256=parent['panel_audit_sha256'],
        source_hashes={'scripts/predictor_trajectory.py':sha256(__file__)})
    check_sources(plan)
    atomic_json(STATE/'plan.json', plan)
    atomic_json(STATE/'panel-metadata.json', json.loads((PARENT/'panel-metadata.json').read_text()))
    atomic_json(STATE/'preparation.json', dict(stage='complete', plan_sha256=digest(plan),
        panel_sha256=parent['panel_sha256'], documents=576, continuation_tokens=18432,
        continuation_bytes=53707, comparisons=496, new_comparisons=372,
        cached_comparisons=124, training_resumed=False))
    emit('preparation', 'complete')


def score(job):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    plan = json.loads((STATE/'plan.json').read_text())
    check_sources(plan)
    code = Path(plan['parent']['code']); sys.path.insert(0, str(code))
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from uniform_trajectory import load_candidate
    from natural_transfer import score_rows, validate_scores
    torch.set_num_threads(1)
    if 'H200' not in torch.cuda.get_device_name():
        raise ValueError('Original H200 evaluation runtime required')
    m = next(m for m in plan['models'] if m['job'] == job)
    if m.get('cached'):
        raise ValueError('Use existing verified 8k scores')
    rows = json.loads((PARENT/'panel.json').read_text())
    if digest(rows) != plan['panel_sha256']:
        raise ValueError('Frozen panel changed')
    checkpoint = Path(m['checkpoint'])
    for relative in FILES:
        if sha256(checkpoint/relative) != m['manifest']['files'][relative]['sha256']:
            raise ValueError('Checkpoint changed after verification')
    emit(job,'loading_checkpoint')
    canonical = AutoTokenizer.from_pretrained(INPUT/'model',local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(INPUT/'model',local_files_only=True,
        dtype=torch.float32,low_cpu_mem_usage=True).to('cuda')
    separate_input_embeddings_and_freeze_output(model)
    original = model.get_input_embeddings().weight.detach().clone()
    model.load_state_dict(torch.load(checkpoint/'lm.pt',map_location='cpu',weights_only=True,mmap=True))
    model.eval().requires_grad_(False)
    source = model.get_input_embeddings().weight.detach()
    if not torch.equal(source,original) or not torch.equal(model.get_output_embeddings().weight,original):
        raise ValueError('Canonical embedding tables changed')
    del original
    hn = ZettHypernet.from_pretrained(checkpoint/'hypernet',local_files_only=True).to('cuda').eval()
    hn.requires_grad_(False); enable_byte_fallbacks(hn,canonical)
    output = STATE/'scores'/job; output.mkdir(parents=True,exist_ok=True)
    key = dict(plan_sha256=digest(plan),panel_sha256=plan['panel_sha256'],model=m['name'],step=m['step'])
    base = score_rows(model,source,source,rows); validate_scores(rows,base)
    atomic_json(output/'canonical.json',dict(**key,canonical=base))
    begin = time.monotonic(); timings=[]
    for i,c in enumerate(plan['parent']['candidates']):
        target = output/(c['id']+'.json')
        if target.exists():
            old = json.loads(target.read_text())
            if any(old.get(k)!=v for k,v in key.items()) or old['tokenizer_sha256']!=c['sha256']:
                raise ValueError('Existing score has different provenance')
            validate_scores(rows,old['transferred']); emit(job,'scoring',i+1,tokenizer=c['id']); continue
        started = time.monotonic(); path = Path(c['path'])
        if sha256(path)!=c['sha256']:
            raise ValueError('Native tokenizer changed')
        native,_,_,_,forbidden = load_candidate(path,c['kind'])
        for row in rows:
            ids = native.encode(row['prefix'],add_special_tokens=False).ids
            if (len(ids)!=row['native_prefix_lengths'][c['id']] or (forbidden|{0}).intersection(ids)
                or native.decode(ids,skip_special_tokens=False)!=row['prefix']):
                raise ValueError('Native prefix proof changed')
        feature = PARENT/'features'/c['id']
        for field in ['forms','priors']:
            if sha256(feature/(field+'.npy'))!=c['features'][field+'_sha256']:
                raise ValueError('Native feature input changed')
        forms = torch.from_numpy(np.load(feature/'forms.npy').astype(np.int64)).to('cuda')
        priors = torch.from_numpy(np.load(feature/'priors.npy')).to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            generated,out,_ = hn(forms,priors,source,None,deterministic=True)
        if out is not None or generated.shape!=(c['vocab_size'],source.shape[1]) or not torch.isfinite(generated).all():
            raise ValueError('Invalid full-vocabulary embeddings')
        transferred = score_rows(model,generated,source,rows,native); validate_scores(rows,transferred)
        seconds=time.monotonic()-started; timings.append(seconds)
        atomic_json(target,dict(**key,tokenizer=c['id'],tokenizer_sha256=c['sha256'],
            transferred=transferred,elapsed_seconds=seconds,gpu=torch.cuda.get_device_name()))
        del forms,priors,generated; torch.cuda.empty_cache()
        emit(job,'scoring',i+1,tokenizer=c['id'],elapsed_seconds=time.monotonic()-begin,
            eta_seconds=(61-i)*sum(timings)/len(timings),peak_gib=torch.cuda.max_memory_allocated()/2**30)
    emit(job,'complete',62,elapsed_seconds=time.monotonic()-begin)


def collect():
    import gzip
    plan=json.loads((STATE/'plan.json').read_text()); models=[]
    for m in plan['models']:
        folder=PARENT/'scores'/m['name'] if m.get('cached') else STATE/'scores'/m['job']
        models.append(dict(name=m['name'],step=m['step'],cached=bool(m.get('cached')),
            canonical=json.loads((folder/'canonical.json').read_text()),
            tokenizers=[json.loads((folder/(c['id']+'.json')).read_text()) for c in plan['parent']['candidates']]))
    data=dict(plan=plan,preparation=json.loads((STATE/'preparation.json').read_text()),
        rows=json.loads((STATE/'panel-metadata.json').read_text()),models=models)
    (STATE/'results.json.gz').write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
    print(json.dumps(dict(bytes=(STATE/'results.json.gz').stat().st_size)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--collect',action='store_true')
    parser.add_argument('--job'); args=parser.parse_args()
    try:
        if args.prepare: prepare()
        elif args.collect: collect()
        elif args.job: score(args.job)
        else: parser.error('Choose prepare, job or collect')
    except Exception as error:
        emit(args.job or 'preparation','failed',error=str(error)); raise


if __name__=='__main__': main()
