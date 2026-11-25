"""Shared-prefix preparation and frozen four-model scoring at step 8,000."""
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
STATE = ROOT / 'modal_results/four_way_8k'
INPUT = ROOT / 'modal_results/local_memory/artifacts/input'
MODELS = ['uniform', 'raw_bpb', 'regret', 'predictor']


def emit(name, stage, completed=0, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed,
        total=62, model=name, platform='Tillicum · H200', **fields)
    atomic_json(STATE / 'status' / (name + '.json'), row)
    with (STATE / 'events' / (name + '.jsonl')).open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def common_panel(parent, canonical, tokenizers, forbidden):
    """Preserve all targets; find a single minimally trimmed prefix for all encoders."""
    rows = []
    for original in parent:
        target = original['canonical_continuation_ids']
        suffix = canonical.decode(target, clean_up_tokenization_spaces=False)
        if len(target) != 32 or len(suffix.encode()) != original['continuation_bytes']:
            raise ValueError('Invalid frozen continuation')
        for start in range(len(original['prefix'])):
            prefix = original['prefix'][start:]
            ids = canonical(prefix, add_special_tokens=False).input_ids
            if not 16 <= len(ids) <= 64 or 0 in ids:
                continue
            if (canonical.decode(ids, clean_up_tokenization_spaces=False) != prefix or
                canonical(prefix + suffix, add_special_tokens=False).input_ids != ids + target):
                continue
            lengths = {}
            for key, tokenizer in tokenizers.items():
                native = tokenizer.encode(prefix, add_special_tokens=False).ids
                if (not native or len(native) > 95 or forbidden[key].intersection(native)
                    or tokenizer.decode(native, skip_special_tokens=False) != prefix):
                    break
                lengths[key] = len(native)
            else:
                rows.append(dict(original, prefix=prefix, canonical_prefix_ids=ids,
                    native_prefix_lengths=lengths, additional_trim_chars=start,
                    original_prefix_sha256=digest(original['prefix']), prefix_sha256=digest(prefix),
                    canonical_joint_encoding_matches=True,
                    maximum_positions=33 + max(len(ids), max(lengths.values())),
                    context_sha256=digest([prefix, target])))
                break
        else:
            raise ValueError('No common valid prefix; no passage removed: ' + original['id'])
    return rows


def setup_paths():
    trained = json.loads((ROOT / 'modal_results/tillicum_hardness/plan.json').read_text())
    code = Path(trained['code'])
    sys.path.insert(0, str(code))
    return code


def prepare():
    import numpy as np
    from transformers import AutoTokenizer
    code = setup_paths()
    from hf_hypernet.tokenizer_inputs import build_surface_forms, missing_byte_symbols
    from uniform_trajectory import load_candidate
    source = json.loads((STATE / 'source-plan.json').read_text())
    emit('preparation','verifying_checkpoints')
    parent = json.loads((STATE / 'parent-panel.json').read_text())
    if digest(parent) != source['parent_panel_sha256']:
        raise ValueError('Original corrected panel changed')
    if len(parent) != 576 or len({(r['split'],r['domain'],r['id']) for r in parent}) != 576:
        raise ValueError('All original documents required')
    for split, byte_count in [('screen',27039),('confirm',26668)]:
        rows = [r for r in parent if r['split'] == split]
        if len(rows) != 288 or sum(r['continuation_bytes'] for r in rows) != byte_count:
            raise ValueError('Original split changed')
        domains = {r['domain'] for r in rows}
        if len(domains) != 9 or any(sum(r['domain']==d for r in rows) != 32 for d in domains):
            raise ValueError('Original domain coverage changed')
    for name, folder in [('regret', ROOT.parent / 'anytok-h200-smoke/modal_results/tillicum_regret/run'),
                         ('predictor', ROOT / 'modal_results/tillicum_hardness/run')]:
        checkpoint = folder / 'step-008000'
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        config = json.loads((folder / 'run_config.json').read_text())
        source['models'].append(dict(name=name, checkpoint=str(checkpoint), manifest=manifest, config=config))
    models = {r['name']:r for r in source['models']}
    if set(models) != set(MODELS):
        raise ValueError('All four checkpoints required')
    for name in MODELS:
        model = models[name]; checkpoint = STATE / model['checkpoint']
        if model['manifest']['step'] != 8000:
            raise ValueError('Wrong checkpoint step')
        for relative in ['lm.pt','hypernet/config.json','hypernet/model.safetensors']:
            expected = model['manifest']['files'][relative]; path = checkpoint / relative
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError('Checkpoint checksum mismatch: ' + name + '/' + relative)
        model['checkpoint'] = str(checkpoint)
    for relative in ['model/tokenizer.json','model/model.safetensors','model/config.json']:
        if sha256(INPUT / relative) != source['input_hashes'][relative]:
            raise ValueError('Original LM input changed')
    canonical = AutoTokenizer.from_pretrained(INPUT / 'model', local_files_only=True)
    symbols = missing_byte_symbols(canonical)
    fallbacks = {s:len(canonical)+1+i for i,s in enumerate(symbols)}
    for model in models.values():
        config = json.loads((Path(model['checkpoint'])/'hypernet/config.json').read_text())
        if (config['hn_byte_fallback_symbols'] != symbols or config['hn_surface_maxlen'] != 10
            or config['original_vocab_size'] != len(canonical)):
            raise ValueError('Different hypernetwork feature mapping')
    natives, forbidden = {}, {}
    for candidate in source['candidates']:
        emit('preparation','preparing_native_features',tokenizer=candidate['id'])
        path = (INPUT / candidate['path'].removeprefix('input/') if candidate['kind']=='pool'
                else STATE / candidate['path'])
        if sha256(path) != candidate['sha256']:
            raise ValueError('Native tokenizer changed: ' + candidate['id'])
        native, tokens, priors, specials, controls = load_candidate(path,candidate['kind'])
        if len(tokens) != candidate['vocab_size']:
            raise ValueError('Native vocabulary size changed')
        candidate['path'] = str(path)
        natives[candidate['id']] = native; forbidden[candidate['id']] = controls | {0}
        features = STATE / 'features' / candidate['id']; features.mkdir(parents=True,exist_ok=True)
        forms, truncated = build_surface_forms(tokens,canonical,10,fallbacks,specials)
        np.save(features/'forms.npy',forms)
        np.save(features/'priors.npy',np.asarray(priors,dtype=np.float32))
        candidate['features'] = dict(forms_sha256=sha256(features/'forms.npy'),
            priors_sha256=sha256(features/'priors.npy'), truncated_forms=truncated)
    rows = common_panel(parent,canonical,natives,forbidden)
    for before,after in zip(parent,rows):
        for key in ['id','split','domain','canonical_continuation_ids','continuation_bytes']:
            if before[key] != after[key]:
                raise ValueError('Frozen targets changed')
    files = ['four_way_8k.py','natural_transfer.py','pool_generalization.py','uniform_trajectory.py',
             'uniform_weakness.py','package_training_inputs.py']
    source.update(models=[models[n] for n in MODELS], panel_sha256=digest(rows),
        panel_audit_sha256=digest([[r['id'],r['prefix_sha256'],r['canonical_continuation_ids'],r['continuation_bytes']] for r in rows]),
        code=str(code), code_manifest_sha256=sha256(code/'source-manifest.json'),
        evaluator_sources={n:sha256(ROOT/'scripts'/n) for n in files},
        runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','numpy']},
        trimmed_passages=sum(r['additional_trim_chars']>0 for r in rows),
        documents=576, comparisons=248, training_resumed=False)
    atomic_json(STATE/'panel.json',rows)
    atomic_json(STATE/'plan.json',source)
    fields = ['id','original_id','split','domain','cluster_id','canonical_prefix_ids',
        'canonical_continuation_ids','continuation_bytes','maximum_positions',
        'native_prefix_lengths','additional_trim_chars','original_prefix_sha256',
        'prefix_sha256','context_sha256','canonical_joint_encoding_matches']
    metadata = [{k:r[k] for k in fields if k in r} for r in rows]
    atomic_json(STATE/'panel-metadata.json',metadata)
    atomic_json(STATE/'preparation.json',dict(stage='complete', panel_sha256=source['panel_sha256'],
        plan_sha256=digest(source), trimmed_passages=source['trimmed_passages'], documents=576,
        continuation_tokens=18432, continuation_bytes=53707, comparisons=248))
    print(json.dumps(json.loads((STATE/'preparation.json').read_text())),flush=True)


def score(name):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    plan = json.loads((STATE/'plan.json').read_text()); setup_paths()
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from uniform_trajectory import load_candidate
    from natural_transfer import score_rows, validate_scores
    torch.set_num_threads(1)
    if 'H200' not in torch.cuda.get_device_name():
        raise ValueError('H200 required for common evaluation runtime')
    for relative,expected in plan['evaluator_sources'].items():
        if sha256(ROOT/'scripts'/relative) != expected:
            raise ValueError('Evaluator source changed after panel preparation')
    for package,version in plan['runtime'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Evaluation runtime changed')
    for relative,expected in json.loads((Path(plan['code'])/'source-manifest.json').read_text()).items():
        if sha256(Path(plan['code'])/relative) != expected:
            raise ValueError('Pinned model implementation changed')
    rows = json.loads((STATE/'panel.json').read_text())
    if digest(rows) != plan['panel_sha256']:
        raise ValueError('Shared panel changed')
    checkpoint = Path(next(m for m in plan['models'] if m['name']==name)['checkpoint'])
    manifest = next(m for m in plan['models'] if m['name']==name)['manifest']
    for relative in ['lm.pt','hypernet/config.json','hypernet/model.safetensors']:
        if sha256(checkpoint/relative) != manifest['files'][relative]['sha256']:
            raise ValueError('Checkpoint changed after verification')
    emit(name,'loading_checkpoint')
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
    output = STATE/'scores'/name; output.mkdir(parents=True,exist_ok=True)
    key = dict(plan_sha256=digest(plan),panel_sha256=plan['panel_sha256'],model=name,step=8000)
    base = score_rows(model,source,source,rows); validate_scores(rows,base)
    atomic_json(output/'canonical.json',dict(**key,canonical=base))
    begin = time.monotonic(); timings=[]
    for i,candidate in enumerate(plan['candidates']):
        target = output/(candidate['id']+'.json')
        if target.exists():
            old=json.loads(target.read_text())
            if any(old.get(k)!=v for k,v in key.items()):
                raise ValueError('Existing scores belong to another evaluation')
            validate_scores(rows,old['transferred'])
            emit(name,'scoring',i+1,tokenizer=candidate['id']);continue
        started = time.monotonic()
        path = Path(candidate['path'])
        if sha256(path)!=candidate['sha256']:
            raise ValueError('Native tokenizer changed during scoring')
        native,_,_,_,forbidden=load_candidate(path,candidate['kind'])
        for row in rows:
            ids=native.encode(row['prefix'],add_special_tokens=False).ids
            if (len(ids)!=row['native_prefix_lengths'][candidate['id']] or (forbidden|{0}).intersection(ids)
                or native.decode(ids,skip_special_tokens=False)!=row['prefix']):
                raise ValueError('Native prefix proof changed')
        feature=STATE/'features'/candidate['id']
        for field in ['forms','priors']:
            if sha256(feature/(field+'.npy'))!=candidate['features'][field+'_sha256']:
                raise ValueError('Hypernetwork input changed')
        forms=torch.from_numpy(np.load(feature/'forms.npy').astype(np.int64)).to('cuda')
        priors=torch.from_numpy(np.load(feature/'priors.npy')).to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            generated,out,_=hn(forms,priors,source,None,deterministic=True)
        if out is not None or generated.shape!=(candidate['vocab_size'],source.shape[1]) or not torch.isfinite(generated).all():
            raise ValueError('Invalid full-vocabulary embeddings')
        transferred=score_rows(model,generated,source,rows,native);validate_scores(rows,transferred)
        seconds=time.monotonic()-started;timings.append(seconds)
        atomic_json(target,dict(**key,tokenizer=candidate['id'],tokenizer_sha256=candidate['sha256'],
            transferred=transferred,elapsed_seconds=seconds,gpu=torch.cuda.get_device_name()))
        del forms,priors,generated;torch.cuda.empty_cache()
        emit(name,'scoring',i+1,tokenizer=candidate['id'],elapsed_seconds=time.monotonic()-begin,
            seconds_per_tokenizer=sum(timings)/len(timings),
            eta_seconds=(61-i)*sum(timings)/len(timings),peak_gib=torch.cuda.max_memory_allocated()/2**30)
    emit(name,'complete',62,elapsed_seconds=time.monotonic()-begin)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    (STATE/'status').mkdir(parents=True,exist_ok=True)
    (STATE/'events').mkdir(parents=True,exist_ok=True)
    try:
        if args.prepare:prepare()
        elif args.model:score(args.model)
        else:parser.error('Choose --prepare or --model')
    except Exception as error:
        emit(args.model or 'preparation','failed',error=str(error));raise


if __name__=='__main__':
    main()
