"""Reconstruct actual training text after a frozen checkpoint, without updates."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

from package_training_inputs import atomic_json, sha256
from run_raw_bpb_training import prepare_code

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_pool_ranking'
RUN = ROOT / 'modal_results/tillicum_regret/run'
INPUT = ROOT / 'modal_results/local_memory/artifacts/input'


def status(stage, **fields):
    atomic_json(STATE / 'preparation-status.json', dict(stage=stage, **fields))
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'panel-manifest.json').exists():
        print('Verified panel already prepared'); return
    code = prepare_code(STATE)
    sys.path[:0] = [str(code / 'scripts'), str(code)]
    path = code / 'scripts/train_joint_smollm_input.py'
    spec = importlib.util.spec_from_file_location('ranking_trainer', path)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    import torch
    torch.set_num_threads(1)
    for step in [6000, 6500]:
        manifest = json.loads((RUN / f'step-{step:06d}/manifest.json').read_text())
        for name in (['state.pt'] if step == 6000 else ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors', 'policy.pt']):
            file = RUN / f'step-{step:06d}' / name
            if sha256(file) != manifest['files'][name]['sha256']:
                raise ValueError('Saved checkpoint changed: ' + str(file))
    saved = torch.load(RUN / 'step-006000/state.pt', map_location='cpu', weights_only=True)
    assert saved['step'] == saved['bandit']['updates'] == 6000
    canonical = trainer.AutoTokenizer.from_pretrained(INPUT / 'model', local_files_only=True)
    pool = trainer.TokenizerPool(INPUT / 'pool')
    corpus = trainer.BalancedTextStream(INPUT / 'recipe.json', INPUT / 'corpus', saved['corpus'])
    rng = random.Random(); rng.setstate(saved['python_rng'])
    logs = {row['step']: row for row in map(json.loads, (RUN / 'events.jsonl').read_text().splitlines())
            if row.get('stage') == 'train'}
    for step in range(6001, 6501):
        row = logs[step]
        names = rng.choices(corpus.names, weights=corpus.weights, k=128)
        texts = [corpus.next_text(name, chars=2048) for name in names]
        tokenizer, _ = pool.load(row['tokenizer_pool_index'])
        proofs = []; rejected = 0
        for name, raw in zip(names, texts):
            paired = trainer.make_paired_prefix_example(raw[:512], tokenizer, canonical, min_chars=32)
            while paired is None and rejected < 128:
                rejected += 1; raw = corpus.next_text(name, chars=2048)
                paired = trainer.make_paired_prefix_example(raw[:512], tokenizer, canonical, min_chars=32)
            if paired is None:
                raise ValueError('Cannot reconstruct logged paired training batch')
            proofs.append(paired[2])
        digest = hashlib.sha256(json.dumps(proofs, sort_keys=True).encode()).hexdigest()
        if digest != row['paired_batch_sha256'] or rejected != row['rejected_examples']:
            raise ValueError(f'Training stream replay mismatch at {step}')
        if step % 25 == 0:
            status('replaying_saved_training_stream', replayed=step - 6000, total_replay=500)
    batches = []
    for batch in range(32):
        names = rng.choices(corpus.names, weights=corpus.weights, k=128)
        items = []
        for slot, name in enumerate(names):
            raw = corpus.next_text(name, chars=2048)[:512]
            items.append(dict(id=f'batch-{batch:02d}/slot-{slot:03d}', domain=name, text=raw,
                              raw_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        batches.append(dict(index=batch, split='screen' if batch < 16 else 'confirm',
                            dropout_seed=2026111500 + batch, items=items, replacements={}))
    # Same ordered replacement bank for every action; use only when the original
    # paired-example helper rejects an item, respecting its 128-rejection cap.
    for batch in batches:
        for name in corpus.names:
            batch['replacements'][name] = [corpus.next_text(name, chars=2048)[:512] for _ in range(128)]
        status('preparing_fixed_replacement_banks', batches=batch['index'] + 1, total_batches=32)
    corpus.close()
    coverage = Counter(item['domain'] for batch in batches for item in batch['items'])
    if set(coverage) != set(corpus.names):
        raise ValueError('Fixed training draw did not cover all 23 corpus buckets')
    target = STATE / 'training-panel.json.gz'
    with gzip.open(target, 'wt', encoding='utf-8') as stream:
        json.dump(batches, stream)
    policy = torch.load(RUN / 'step-006500/policy.pt', map_location='cpu', weights_only=True)
    if policy['updates'] != 6500 or int(policy['counts'].sum()) != 6500:
        raise ValueError('Policy does not match frozen model checkpoint')
    probabilities = policy['logits'].softmax(0).tolist()
    atomic_json(STATE / 'policy.json', dict(step=6500, probabilities=probabilities,
        counts=policy['counts'].tolist(), settings=policy['settings'],
        policy_sha256=sha256(RUN / 'step-006500/policy.pt')))
    checkpoint = json.loads((RUN / 'step-006500/manifest.json').read_text())
    manifest = dict(version=1, checkpoint_step=6500, checkpoint_manifest=checkpoint,
        panel_sha256=sha256(target), corpus_manifest_sha256=sha256(INPUT / 'corpus/manifest.json'),
        recipe_sha256=sha256(INPUT / 'recipe.json'), pool_index_sha256=pool.index_hash,
        pool_plan_sha256=pool.plan_hash, batches=32, batch_size=128, microbatch_size=32,
        examples=4096, domains=dict(coverage), screen_batches=16, confirmation_batches=16,
        scored_continuation='variable length: unchanged make_paired_prefix_example',
        maximum_total_positions=128, snippet_chars=512, min_chars=32,
        dropout='active as in training; fixed independent seed per batch, shared across actions',
        aggregation='arithmetic mean of per-update byte-weighted regret BPB',
        replayed_updates=list(range(6001, 6501)),
        paired_helper_sha256=sha256(ROOT / 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py'))
    atomic_json(STATE / 'panel-manifest.json', manifest)
    status('complete', examples=4096, batches=32, checkpoint_step=6500, panel_sha256=manifest['panel_sha256'])


if __name__ == '__main__':
    main()
