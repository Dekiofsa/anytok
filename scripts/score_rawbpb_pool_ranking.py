"""Frozen all-pool ranking with the exact training reward and active dropout.

Copy of scripts/score_training_pool_ranking.py at 5f431ea for the raw-BPB run's
frozen full-restart checkpoint. Only these differ: the state, input, checkpoint
and staged-code paths and the checkpoint step are required arguments that must
match run-plan.json (input, checkpoint and code relative to the state directory,
so no absolute path is frozen); the staged code is the copy the run trained
with, checked file by file against its source manifest, not a new copy under
the state folder; and --max-tokenizers stops a smoke test after that many newly
scored tokenizers (stage smoke_stop, never complete). The scoring computation is
unchanged.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from package_training_inputs import atomic_json, sha256


def pair_batch(batch, native, canonical, helper):
    examples = []; controls = []; proofs = []
    positions = {name: 0 for name in batch['replacements']}
    rejected = 0
    for item in batch['items']:
        raw = item['text']
        if hashlib.sha256(raw.encode()).hexdigest() != item['raw_sha256']:
            raise ValueError('Fixed raw input changed')
        paired = helper(raw, native, canonical, min_chars=32)
        while paired is None and rejected < 128:
            name = item['domain']; index = positions[name]
            raw = batch['replacements'][name][index]; positions[name] += 1; rejected += 1
            paired = helper(raw, native, canonical, min_chars=32)
        if paired is None:
            raise ValueError('Original training rejection limit exceeded; no passages or tokenizer removed')
        example, control, proof = paired
        if example[1:] != control[1:] or proof['maximum_positions'] > 128:
            raise ValueError('Paired targets/bytes or position cap changed')
        examples.append(example); controls.append(control); proofs.append(proof)
    if len(examples) != 128:
        raise ValueError('Complete 128-example update required')
    return examples, controls, proofs, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--max-tokenizers', type=int, default=None,
                        help='Smoke test: stop after this many newly scored tokenizers')
    args = parser.parse_args()
    if args.shards != 4 or not 0 <= args.shard < args.shards:
        parser.error('The approved complete sweep uses four nonoverlapping shards')
    if args.max_tokenizers is not None and args.max_tokenizers < 1:
        parser.error('--max-tokenizers must be positive')
    STATE, INPUT, CHECKPOINT = args.state.resolve(), args.input.resolve(), args.checkpoint.resolve()
    if not (STATE / 'run-plan.json').is_file():
        parser.error('--state must be the prepared evaluation directory (run-plan.json missing)')
    shard = args.shard; started = time.monotonic()
    status_path = STATE / 'status' / f'{shard}.json'
    def emit(stage, completed, **fields):
        row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed,
                   total=250, shard=shard, elapsed_seconds=time.monotonic() - started, **fields)
        atomic_json(status_path, row); print(json.dumps(row), flush=True)
    emit('loading_frozen_checkpoint', 0)
    plan = json.loads((STATE / 'panel-manifest.json').read_text())
    run_plan = json.loads((STATE / 'run-plan.json').read_text())
    if sha256(Path(__file__)) != run_plan['scorer_sha256']:
        raise ValueError('Scoring implementation changed after the sweep was frozen')
    if ([os.path.relpath(path, STATE) for path in [INPUT, CHECKPOINT, args.code.resolve()]] + [args.step]
            != [run_plan[key] for key in ['input_dir', 'checkpoint_dir', 'code_dir', 'checkpoint_step']]):
        raise ValueError('Arguments differ from the frozen run plan')
    panel_path = STATE / 'training-panel.json.gz'
    if sha256(panel_path) != plan['panel_sha256']:
        raise ValueError('Fixed panel changed')
    with gzip.open(panel_path, 'rt', encoding='utf-8') as stream:
        batches = json.load(stream)
    if len(batches) != 32 or any(len(batch['items']) != 128 for batch in batches):
        raise ValueError('All 32 full training batches required')
    policy = json.loads((STATE / 'policy.json').read_text())
    if sha256(CHECKPOINT / 'policy.pt') != policy['policy_sha256'] or policy['step'] != args.step:
        raise ValueError('Policy checkpoint changed')
    for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
        if sha256(CHECKPOINT / name) != plan['checkpoint_manifest']['files'][name]['sha256']:
            raise ValueError('Frozen model changed: ' + name)
    code = Path(args.code).resolve()
    if sha256(code / 'source-manifest.json') != run_plan['staged_source_manifest_sha256']:
        raise ValueError('Staged training helpers changed after panel preparation')
    for name, digest in json.loads((code / 'source-manifest.json').read_text()).items():
        if sha256(code / name) != digest:
            raise ValueError('Staged training helper changed: ' + name)
    sys.path[:0] = [str(code / 'scripts'), str(code)]
    path = code / 'scripts/train_joint_smollm_input.py'
    spec = importlib.util.spec_from_file_location('ranking_trainer', path)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    import numpy as np
    import torch
    from pool_training_memory import apply_gpu_memory_fix
    torch.set_num_threads(1)
    if 'H200' not in torch.cuda.get_device_name():
        raise ValueError('Full H200 required for this sweep')
    canonical = trainer.AutoTokenizer.from_pretrained(INPUT / 'model', local_files_only=True)
    pool = trainer.TokenizerPool(INPUT / 'pool')
    if pool.index_hash != plan['pool_index_sha256'] or pool.plan_hash != plan['pool_plan_sha256']:
        raise ValueError('Original complete tokenizer pool changed')
    model = trainer.AutoModelForCausalLM.from_pretrained(INPUT / 'model', local_files_only=True,
        dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
    trainer.separate_input_embeddings_and_freeze_output(model)
    original = model.get_input_embeddings().weight.detach().clone()
    model.load_state_dict(torch.load(CHECKPOINT / 'lm.pt', map_location='cpu', weights_only=True, mmap=True))
    source = model.get_input_embeddings().weight.detach()
    if not torch.equal(original, source) or not torch.equal(original, model.get_output_embeddings().weight):
        raise ValueError('Canonical input/output tables changed')
    del original
    hn = trainer.ZettHypernet.from_pretrained(CHECKPOINT / 'hypernet', local_files_only=True).to('cuda')
    fallback = trainer.enable_byte_fallbacks(hn, canonical); trainer.freeze_rescalers(hn)
    apply_gpu_memory_fix(hn)
    model.requires_grad_(False); hn.requires_grad_(False)
    model.train(); hn.train()  # Same dropout behavior as the training reward.
    versions = {prefix + name: p._version for prefix, module in [('lm.', model), ('hn.', hn)]
                for name, p in module.named_parameters()}
    order = sorted(range(1000), key=lambda i: (-policy['probabilities'][i], i))
    work = order[shard::args.shards]
    completed = 0
    newly_scored = 0
    for index in work:
        target = STATE / 'scores' / f'{index:04d}.json'
        key = dict(index=index, checkpoint_step=args.step, panel_sha256=plan['panel_sha256'],
                   tokenizer_sha256=pool.entries[index]['tokenizer_sha256'])
        if target.exists():
            result = json.loads(target.read_text())
            if any(result.get(k) != v for k, v in key.items()) or len(result['batches']) != 32:
                raise ValueError('Existing score belongs to another sweep')
            completed += 1; emit('scoring', completed, index=index); continue
        partial = STATE / 'partial' / f'{index:04d}.json'
        result = json.loads(partial.read_text()) if partial.exists() else dict(**key, batches=[])
        if any(result.get(k) != v for k, v in key.items()):
            raise ValueError('Partial scores belong to another sweep')
        native, vocab = pool.load(index)
        forms_np, truncated = pool.forms(index, canonical, fallback)
        forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
        priors = torch.tensor([score for _, score in vocab], device='cuda', dtype=torch.float32)
        token_started = time.monotonic()
        for batch in batches[len(result['batches']):]:
            examples, controls, proofs, rejected = pair_batch(batch, native, canonical,
                                                              trainer.make_paired_prefix_example)
            scored_tokens = sum(len(item[1]) for item in examples)
            scored_bytes = sum(item[2] for item in examples)
            torch.cuda.manual_seed(batch['dropout_seed'])
            with torch.inference_mode():
                base_nats = trainer.canonical_loss_nats(model, controls, source, 32, trainer.make_lm_microbatch)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated, output, _ = hn(forms, priors, source, None, deterministic=False)
                if output is not None or not torch.isfinite(generated).all():
                    raise ValueError('Invalid native embedding table')
                leaf = generated.detach().float()
                nats = 0.
                for start in range(0, 128, 32):
                    micro = examples[start:start + 32]
                    embeds, mask, labels, positions = trainer.make_lm_microbatch(micro, leaf, source, 'cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                                     labels=labels, use_cache=False).loss
                    nats += float(loss) * sum(len(item[1]) for item in micro)
                    del loss, embeds, mask, labels, positions
                raw, canonical_bpb, gap = trainer.paired_gap_bpb(nats, base_nats, scored_bytes)
                del leaf, generated
            record = dict(index=batch['index'], split=batch['split'], raw_bpb=raw,
                canonical_bpb=canonical_bpb, gap_bpb=gap, transferred_nats=nats, canonical_nats=base_nats,
                scored_tokens=scored_tokens, scored_bytes=scored_bytes, valid_examples=128,
                maximum_positions=max(proof['maximum_positions'] for proof in proofs), rejected_examples=rejected,
                dropout_seed=batch['dropout_seed'],
                paired_batch_sha256=hashlib.sha256(json.dumps(proofs, sort_keys=True).encode()).hexdigest())
            result['batches'].append(record); atomic_json(partial, result)
            emit('scoring', completed, index=index, batches_scored=len(result['batches']), batch_total=32)
        result.update(config=pool.entries[index]['config'], vocabulary=len(vocab), truncated_forms=truncated,
            policy_probability=policy['probabilities'][index], seconds=time.monotonic() - token_started)
        atomic_json(target, result); partial.unlink(missing_ok=True)
        completed += 1; emit('scoring', completed, index=index)
        del forms, priors
        pool._forms.pop(index, None)
        torch.cuda.empty_cache()
        newly_scored += 1
        if args.max_tokenizers is not None and newly_scored >= args.max_tokenizers:
            break
    if versions != {prefix + name: p._version for prefix, module in [('lm.', model), ('hn.', hn)]
                    for name, p in module.named_parameters()}:
        raise ValueError('Model parameters changed during frozen scoring')
    if args.max_tokenizers is not None:  # A smoke test never reports the shard complete.
        emit('smoke_stop', completed, frozen_parameter_versions_unchanged=True,
             newly_scored=newly_scored, max_tokenizers=args.max_tokenizers)
        return
    emit('complete', completed, frozen_parameter_versions_unchanged=True)


if __name__ == '__main__':
    main()
