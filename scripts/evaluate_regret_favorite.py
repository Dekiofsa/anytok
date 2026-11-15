"""Score one saved regret checkpoint on #771 and the unchanged full panel."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from package_training_inputs import atomic_json, sha256
from run_raw_bpb_training import prepare_code
from score_training_pool_ranking import pair_batch

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_favorite_trajectory'
INPUT = ROOT / 'modal_results/local_memory/artifacts/input'
INDEX = 771
WEIGHTS = ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors', 'policy.pt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    if torch.__version__.split('+')[0] != '2.15.0':
        raise ValueError('Use the pinned .venv-memory torch 2.15.0 environment')
    torch.set_num_threads(1)
    started = time.monotonic()
    checkpoint = args.checkpoint
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    inventory = json.loads((STATE / 'inventory.json').read_text())
    reference = next(row for row in inventory['checkpoints'] if row['step'] == args.step)
    if manifest != reference['manifest'] or manifest['step'] != args.step:
        raise ValueError('Checkpoint step mismatch')
    for name in WEIGHTS:
        expected = manifest['files'][name]
        if (checkpoint / name).stat().st_size != expected['bytes'] or sha256(checkpoint / name) != expected['sha256']:
            raise ValueError('Checkpoint checksum mismatch: ' + name)
    plan = json.loads((STATE / 'panel-manifest.json').read_text())
    if sha256(STATE / 'training-panel.json.gz') != plan['panel_sha256']:
        raise ValueError('Frozen full panel changed')
    with gzip.open(STATE / 'training-panel.json.gz', 'rt') as stream:
        batches = json.load(stream)
    if len(batches) != 32 or any(len(batch['items']) != 128 for batch in batches):
        raise ValueError('All 32 full training-style batches are required')
    code = prepare_code(STATE)
    source_manifest = sha256(code / 'source-manifest.json')
    original_plan = json.loads((STATE / 'ranking-run-plan.json').read_text())
    if source_manifest != original_plan['staged_source_manifest_sha256']:
        raise ValueError('The paired helper or scoring dependencies changed')
    sys.path[:0] = [str(code / 'scripts'), str(code)]
    spec = importlib.util.spec_from_file_location('favorite_trainer', code / 'scripts/train_joint_smollm_input.py')
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    from pool_training_memory import apply_gpu_memory_fix
    canonical = trainer.AutoTokenizer.from_pretrained(INPUT / 'model', local_files_only=True)
    pool = trainer.TokenizerPool(INPUT / 'pool')
    if pool.index_hash != plan['pool_index_sha256'] or pool.plan_hash != plan['pool_plan_sha256']:
        raise ValueError('Original pool metadata changed')
    native, vocab = pool.load(INDEX)
    lm = trainer.AutoModelForCausalLM.from_pretrained(INPUT / 'model', local_files_only=True,
        dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
    trainer.separate_input_embeddings_and_freeze_output(lm)
    original = lm.get_input_embeddings().weight.detach().clone()
    lm.load_state_dict(torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True, mmap=True))
    source = lm.get_input_embeddings().weight.detach()
    if not torch.equal(original, source) or not torch.equal(original, lm.get_output_embeddings().weight):
        raise ValueError('Fixed canonical input/output embeddings changed')
    del original
    hn = trainer.ZettHypernet.from_pretrained(checkpoint / 'hypernet', local_files_only=True).to('cuda')
    fallback = trainer.enable_byte_fallbacks(hn, canonical); trainer.freeze_rescalers(hn)
    apply_gpu_memory_fix(hn)
    lm.requires_grad_(False); hn.requires_grad_(False)
    lm.train(); hn.train()
    versions = {prefix + name: parameter._version for prefix, model in [('lm.', lm), ('hn.', hn)]
                for name, parameter in model.named_parameters()}
    forms_np, truncated = pool.forms(INDEX, canonical, fallback)
    forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
    priors = torch.tensor([score for _, score in vocab], device='cuda', dtype=torch.float32)
    policy = torch.load(checkpoint / 'policy.pt', map_location='cpu', weights_only=True)
    if policy['updates'] != args.step or int(policy['counts'].sum()) != args.step:
        raise ValueError('Policy snapshot does not match the model checkpoint')
    key = dict(step=args.step, tokenizer_index=INDEX, panel_sha256=plan['panel_sha256'],
               checkpoint_manifest_sha256=sha256(checkpoint / 'manifest.json'),
               source_manifest_sha256=reference['manifest_sha256'],
               tokenizer_sha256=pool.entries[INDEX]['tokenizer_sha256'],
               staged_source_manifest_sha256=source_manifest, worker_sha256=sha256(Path(__file__)))
    partial = STATE / 'partial' / f'{args.step:06d}.json'
    result = json.loads(partial.read_text()) if partial.exists() else dict(**key, batches=[])
    if any(result.get(name) != value for name, value in key.items()):
        raise ValueError('Partial scores belong to a different experiment')
    for batch in batches[len(result['batches']):]:
        examples, controls, proofs, rejected = pair_batch(batch, native, canonical,
                                                          trainer.make_paired_prefix_example)
        count = sum(len(example[1]) for example in examples)
        size = sum(example[2] for example in examples)
        torch.cuda.manual_seed(batch['dropout_seed'])
        with torch.inference_mode():
            base_nats = trainer.canonical_loss_nats(lm, controls, source, 32, trainer.make_lm_microbatch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                generated, output, _ = hn(forms, priors, source, None, deterministic=False)
            if output is not None or not torch.isfinite(generated).all():
                raise ValueError('Invalid generated embeddings')
            leaf = generated.detach().float(); nats = 0.
            for start in range(0, 128, 32):
                micro = examples[start:start + 32]
                embeds, mask, labels, positions = trainer.make_lm_microbatch(micro, leaf, source, 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = lm(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                              labels=labels, use_cache=False).loss
                nats += float(loss) * sum(len(example[1]) for example in micro)
                del loss, embeds, mask, labels, positions
            raw, base, gap = trainer.paired_gap_bpb(nats, base_nats, size)
            del leaf, generated
        result['batches'].append(dict(index=batch['index'], split=batch['split'],
            raw_bpb=raw, canonical_bpb=base, gap_bpb=gap, transferred_nats=nats, canonical_nats=base_nats,
            scored_tokens=count, scored_bytes=size, valid_examples=128,
            maximum_positions=max(proof['maximum_positions'] for proof in proofs),
            rejected_examples=rejected, dropout_seed=batch['dropout_seed'],
            paired_batch_sha256=hashlib.sha256(json.dumps(proofs, sort_keys=True).encode()).hexdigest()))
        atomic_json(partial, result)
        print(json.dumps(dict(stage='scoring', step=args.step, batches_scored=len(result['batches']),
                              at_utc=datetime.now(timezone.utc).isoformat())), flush=True)
    if versions != {prefix + name: parameter._version for prefix, model in [('lm.', lm), ('hn.', hn)]
                    for name, parameter in model.named_parameters()}:
        raise ValueError('Frozen model parameters changed')
    result.update(config=pool.entries[INDEX]['config'], vocabulary=len(vocab), truncated_forms=truncated,
        policy_probability=float(policy['logits'].softmax(0)[INDEX]),
        cumulative_selections=int(policy['counts'][INDEX]), seconds=time.monotonic() - started,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        frozen_parameter_versions_unchanged=True)
    atomic_json(STATE / 'scores' / f'{args.step:06d}.json', result)
    partial.unlink(missing_ok=True)
    print(json.dumps(dict(stage='checkpoint_scored', step=args.step, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
