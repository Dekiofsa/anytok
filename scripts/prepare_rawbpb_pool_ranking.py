"""Freeze a raw-BPB full-restart checkpoint and the training batches it is ranked on.

Adapted from scripts/prepare_training_reward_panel.py at 5f431ea. The
corpus reader, the paired-example helper and the tokenizer pool come from the
staged code the run trained with, imported the way the scorer imports it.

  --panel fresh      The next 32 training batches of 128 windows (and the fixed
                     replacement banks) the run would have drawn after the frozen
                     full restart, from that checkpoint's saved corpus stream and
                     Python RNG; nothing is replayed. Cross-checked by default:
                     restore an earlier full restart (--cross-check-from, default
                     18000), replay every logged update up to the frozen step
                     (each logged paired_batch_sha256 and rejected_examples must
                     match), draw again and require an identical panel. Buckets
                     past their first pass are named in panel_source.
  --panel published  The published panel of the step-6,500 ranking, copied
                     unchanged after its sha256 is checked.

Writes to --state: training-panel.json.gz, policy.json, checkpoint-manifest.json,
run-plan.json and, last, panel-manifest.json, which is published only after its
bytes match the hash in run-plan.json and never over an existing one;
preparation-status.json follows every stage. Frozen files hold no absolute path:
run-plan.json gives the other directories relative to --state. No model update
and no GPU work: torch only reads the checkpoint.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys

from package_training_inputs import atomic_json, sha256

# sha256 of source-manifest.json of the staged code this run trained with; it is
# also staged_source_manifest_sha256 of the step-6,500 ranking.
TRAINED_SOURCE_MANIFEST_SHA256 = 'c4714e6e32dd9b8c8823f92717b3c25abea71fec9c9bcb14c390303b6cf597db'
# panel_sha256 of the published step-6,500 panel.
PUBLISHED_PANEL_SHA256 = '4afcd96282ec8700d6cdfc0b275138769d614cc2d633e44b0f340b1921c0951d'
CHECKPOINT_FILES = ['state.pt', 'policy.pt', 'lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']
DEFAULT_CROSS_CHECK_FROM = 18000


class Status:
    """preparation-status.json, rewritten atomically at every stage."""

    def __init__(self, state):
        self.path = state / 'preparation-status.json'
        self.stage = None

    def __call__(self, stage, detail, **fields):
        self.stage = stage
        row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, detail=detail, **fields)
        atomic_json(self.path, row)
        print(json.dumps(row), flush=True)


def terminated(signum, _frame):
    raise SystemExit(128 + signum)  # Slurm's TERM: record 'failed' before the job ends.


def without_account(text):
    """Error text for the status file, with the account name in paths written <user> as in the reports."""
    try:
        user = getpass.getuser()
    except Exception:
        return text
    return re.sub(r'(?<=/)' + re.escape(user) + r'(?![\w.-])', '<user>', text) if user else text


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_checkpoint_file(directory, manifest, name):
    expected = manifest['files'][name]
    path = directory / name
    if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
        raise ValueError(f'Saved checkpoint changed: {directory.name}/{name}')


def full_restart_manifest(directory, step):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('step') != step or manifest.get('kind') != 'full_restart_checkpoint':
        raise ValueError(f'{directory.name} is not the full restart checkpoint of step {step}')
    return manifest


def load_trainer(code):
    """The staged training code, verified file by file and imported as the scorer imports it."""
    if sha256(code / 'source-manifest.json') != TRAINED_SOURCE_MANIFEST_SHA256:
        raise ValueError('Staged code is not the code this run trained with (source-manifest.json differs)')
    for name, digest in json.loads((code / 'source-manifest.json').read_text()).items():
        if not (code / name).is_file() or sha256(code / name) != digest:
            raise ValueError('Staged training code changed: ' + name)
    sys.path[:0] = [str(code / 'scripts'), str(code)]
    path = code / 'scripts/train_joint_smollm_input.py'
    spec = importlib.util.spec_from_file_location('ranking_trainer', path)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    return trainer


def draw_panel(corpus, rng, progress):
    """The original draw (L70-89 at 5f431ea), unchanged: 32 recipe-sampled batches, then the banks.

    Also returns each bucket's pass count (corpus epochs) after the items, before the banks.
    """
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
    epochs_after_items = dict(corpus.epochs)
    # Same ordered replacement bank for every action; use only when the original
    # paired-example helper rejects an item, respecting its 128-rejection cap.
    for batch in batches:
        for name in corpus.names:
            batch['replacements'][name] = [corpus.next_text(name, chars=2048)[:512] for _ in range(128)]
        progress(batch['index'] + 1)
    coverage = Counter(item['domain'] for batch in batches for item in batch['items'])
    if set(coverage) != set(corpus.names):
        raise ValueError('Fixed training draw did not cover all 23 corpus buckets')
    return batches, coverage, epochs_after_items


def replay_training_stream(trainer, corpus, rng, pool, canonical, logs, first, last, progress):
    """The original replay (L51-69 at 5f431ea): logged tokenizer, the trainer's rejection loop."""
    rejected_total = 0
    for step in range(first, last + 1):
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
            raise ValueError(f'Training stream replay mismatch at {step}: paired_batch_sha256 {digest} '
                             f'(logged {row["paired_batch_sha256"]}), rejected_examples {rejected} '
                             f'(logged {row["rejected_examples"]})')
        rejected_total += rejected
        if step % 25 == 0 or step == last:
            progress(step - first + 1, rejected_total)
    return rejected_total


def cross_check(trainer, torch, run, inputs, pool, origin_step, step, batches, status):
    """Replay origin_step+1..step from an earlier full restart and require the same panel."""
    total = step - origin_step
    origin = (run / f'step-{origin_step:06d}').resolve()
    status('cross_check_replay', f'verifying step-{origin_step:06d}/state.pt', completed=0, total=total)
    manifest = full_restart_manifest(origin, origin_step)
    verify_checkpoint_file(origin, manifest, 'state.pt')
    earlier = torch.load(origin / 'state.pt', map_location='cpu', weights_only=True)
    if not earlier['step'] == earlier['bandit']['updates'] == origin_step:
        raise ValueError('Earlier full restart counters differ from its step')
    # Last row per update (updates 4,001-5,000 of this run are logged twice).
    logs = {row['step']: row for row in map(json.loads, (run / 'events.jsonl').read_text().splitlines())
            if row.get('stage') == 'train'}
    missing = [update for update in range(origin_step + 1, step + 1) if update not in logs]
    if missing:
        raise ValueError(f'{len(missing)} replayed updates have no logged training row, first {missing[0]}')
    status('cross_check_replay', f'restoring the corpus stream and Python RNG of step {origin_step}',
           completed=0, total=total)
    canonical = trainer.AutoTokenizer.from_pretrained(inputs / 'model', local_files_only=True)
    corpus = trainer.BalancedTextStream(inputs / 'recipe.json', inputs / 'corpus', earlier['corpus'])
    rng = random.Random(); rng.setstate(earlier['python_rng'])
    rejected = replay_training_stream(
        trainer, corpus, rng, pool, canonical, logs, origin_step + 1, step,
        lambda done, rejected: status('cross_check_replay', f'replaying logged updates {origin_step + 1}-{step}',
                                      completed=done, total=total, rejected_examples=rejected))
    again, _, _ = draw_panel(corpus, rng, lambda done: status(
        'cross_check_replay', 'drawing the panel again from the replayed stream (replacement banks)',
        completed=total, total=total, banks=done, total_banks=32))
    corpus.close()
    if again != batches or canonical_sha256(again) != canonical_sha256(batches):
        raise ValueError(f'The stream replayed from step {origin_step} gives a different panel '
                         f'than the saved step-{step} stream')
    return dict(from_step=origin_step, state_sha256=manifest['files']['state.pt']['sha256'],
                updates_replayed=total, logged_hashes_matched=total, rejected_examples=rejected,
                identical_panel=True)


def prepare(args, state, status, cross_from):
    run, inputs, code = args.run.resolve(), args.input.resolve(), args.code.resolve()
    step = args.step
    checkpoint = (run / f'step-{step:06d}').resolve()

    status('verifying_code', 'checking every staged training file against source-manifest.json')
    trainer = load_trainer(code)
    import torch
    torch.set_num_threads(1)

    status('hashing_checkpoint', f'step-{step:06d}/manifest.json', completed=0, total=len(CHECKPOINT_FILES))
    checkpoint_manifest = full_restart_manifest(checkpoint, step)
    for number, name in enumerate(CHECKPOINT_FILES):
        status('hashing_checkpoint', f'step-{step:06d}/{name}', completed=number, total=len(CHECKPOINT_FILES))
        verify_checkpoint_file(checkpoint, checkpoint_manifest, name)
    saved = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=True)
    policy = torch.load(checkpoint / 'policy.pt', map_location='cpu', weights_only=True)
    if not saved['step'] == saved['bandit']['updates'] == policy['updates'] == step:
        raise ValueError('Saved step and policy update counts differ from --step')
    if int(policy['counts'].sum()) != step:
        raise ValueError('Policy does not match frozen model checkpoint')
    pool = trainer.TokenizerPool(inputs / 'pool')
    probabilities = policy['logits'].softmax(0).tolist()
    if len(probabilities) != len(pool.entries):
        raise ValueError('Policy size differs from the tokenizer pool')

    target = state / 'training-panel.json.gz'
    temporary = state / 'training-panel.tmp.json.gz'
    cross = None
    if args.panel == 'fresh':
        status('drawing_panel', 'checking the corpus files against corpus/manifest.json')
        trainer.verify_corpus(inputs / 'corpus', inputs / 'corpus/manifest.json')
        status('drawing_panel', f'restoring the corpus stream and Python RNG of step {step} (no replay)')
        corpus = trainer.BalancedTextStream(inputs / 'recipe.json', inputs / 'corpus', saved['corpus'])
        rng = random.Random(); rng.setstate(saved['python_rng'])
        batches, coverage, epochs_after_items = draw_panel(corpus, rng, lambda done: status(
            'drawing_panel', 'fixed replacement banks', completed=done, total=32))
        # Passes over each bucket's quota; a later pass re-reads the same windows as the first.
        corpus_stream = dict(epochs_at_checkpoint=saved['corpus']['epochs'],
                             used_bytes_at_checkpoint=saved['corpus']['used_bytes'],
                             epochs_after_items=epochs_after_items, epochs_after_banks=dict(corpus.epochs))
        corpus.close()
        if cross_from is not None:
            cross = cross_check(trainer, torch, run, inputs, pool, cross_from, step, batches, status)
        status('writing', 'training-panel.json.gz')
        with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
            json.dump(batches, stream)
        temporary.replace(target)
        with gzip.open(target, 'rt', encoding='utf-8') as stream:
            if json.load(stream) != batches:
                raise ValueError('Written panel differs from the drawn panel')
        source = (f'fresh: the next 32 training batches of 128 windows and the 128-window replacement banks '
                  f'per bucket per batch that the run would have drawn after its step-{step} full restart, '
                  f'restored from that checkpoint\'s saved corpus stream and Python RNG with no replay')
        later = sorted(name for name, epochs in epochs_after_items.items() if epochs)
        if later:
            source += (f'; not all unseen: items drawn in a later pass over a bucket repeat windows the run '
                       f'drew for training in an earlier pass, which affects these buckets (epochs at step '
                       f'{step} -> after the items): ' + ', '.join(
                           f'{name} {corpus_stream["epochs_at_checkpoint"][name]}->{epochs_after_items[name]}'
                           for name in later))
        if cross:
            source += (f'; cross-checked: replaying logged updates {cross_from + 1}-{step} from the '
                       f'step-{cross_from} full restart matched every logged paired_batch_sha256 and '
                       f'rejected_examples and gave an identical panel')
        replayed = dict(first=cross_from + 1, last=step, count=step - cross_from) if cross else []
    else:
        status('writing', 'copying and checking the published panel')
        if sha256(args.published_panel) != PUBLISHED_PANEL_SHA256:
            raise ValueError('The given panel is not the published step-6,500 panel')
        shutil.copyfile(args.published_panel, temporary)
        temporary.replace(target)
        if sha256(target) != PUBLISHED_PANEL_SHA256:
            raise ValueError('Copied panel differs from the published step-6,500 panel')
        with gzip.open(target, 'rt', encoding='utf-8') as stream:
            batches = json.load(stream)
        if len(batches) != 32 or any(len(batch['items']) != 128 for batch in batches):
            raise ValueError('All 32 full training batches required')
        names = [bucket['name'] for bucket in json.loads((inputs / 'recipe.json').read_text())['buckets']]
        coverage = Counter(item['domain'] for batch in batches for item in batch['items'])
        if set(coverage) != set(names):
            raise ValueError('Published panel does not cover all 23 corpus buckets')
        source = (f'published: the published panel of the step-6,500 ranking (sha256 {PUBLISHED_PANEL_SHA256}), '
                  f'copied unchanged: the 32 batches after the regret run\'s step 6,500, the same text this '
                  f'run drew at its updates 6,501-6,532; not unseen by the step-{step} model')
        replayed = []
        corpus_stream = None

    status('writing', 'policy.json, checkpoint-manifest.json, run-plan.json, panel-manifest.json')
    atomic_json(state / 'policy.json', dict(step=step, probabilities=probabilities,
        counts=policy['counts'].tolist(), settings=policy['settings'],
        policy_sha256=sha256(checkpoint / 'policy.pt')))
    atomic_json(state / 'checkpoint-manifest.json', checkpoint_manifest)
    manifest = dict(version=1, checkpoint_step=step, checkpoint_manifest=checkpoint_manifest,
        panel_sha256=sha256(target), panel_json_sha256=canonical_sha256(batches),
        corpus_manifest_sha256=sha256(inputs / 'corpus/manifest.json'),
        recipe_sha256=sha256(inputs / 'recipe.json'), pool_index_sha256=pool.index_hash,
        pool_plan_sha256=pool.plan_hash, batches=32, batch_size=128, microbatch_size=32,
        examples=4096, domains=dict(coverage), screen_batches=16, confirmation_batches=16,
        scored_continuation='variable length: unchanged make_paired_prefix_example',
        maximum_total_positions=128, snippet_chars=512, min_chars=32,
        dropout='active as in training; fixed independent seed per batch, shared across actions',
        aggregation=('arithmetic mean over the 32 batches of per-update byte-weighted BPB (token NLL '
                     'summed over each 128-example update, divided by its continuation bytes and ln 2), '
                     'reported both as gap_bpb = transferred minus canonical BPB (the regret of the '
                     'step-6,500 ranking) and as raw_bpb = transferred BPB (this run\'s training reward)'),
        panel_mode=args.panel, panel_source=source, replayed_updates=replayed, cross_check=cross,
        corpus_stream=corpus_stream, paired_helper_sha256=sha256(code / 'hf_hypernet/paired_reward.py'))
    # The exact bytes atomic_json writes, so run-plan.json can be written before
    # panel-manifest.json, which marks a completed preparation.
    manifest_sha256 = hashlib.sha256((json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    # Directories relative to the state directory: no absolute path or account name is frozen.
    atomic_json(state / 'run-plan.json', dict(checkpoint_step=step, dropout_active=True, reward='raw_bpb',
        shards=4, indices=list(range(1000)), panel_manifest_sha256=manifest_sha256,
        scorer_sha256=sha256(args.scorer),
        staged_source_manifest_sha256=sha256(code / 'source-manifest.json'),
        paths_relative_to='state directory', input_dir=os.path.relpath(inputs, state),
        checkpoint_dir=os.path.relpath(checkpoint, state), code_dir=os.path.relpath(code, state),
        run_dir=os.path.relpath(run, state), panel_mode=args.panel))
    # The completion marker appears only once its bytes match run-plan.json; os.link
    # refuses an existing name, so a manifest is never replaced.
    pending = state / 'panel-manifest.pending.json'
    atomic_json(pending, manifest)
    if sha256(pending) != manifest_sha256:
        raise ValueError('panel-manifest.json differs from the hash in run-plan.json')
    os.link(pending, state / 'panel-manifest.json')
    pending.unlink()
    status('prepared', f'{args.panel} panel and step-{step} checkpoint frozen for the full-pool ranking',
           panel_mode=args.panel, checkpoint_step=step, examples=4096, batches=32,
           panel_sha256=manifest['panel_sha256'], panel_manifest_sha256=manifest_sha256,
           replayed_updates=replayed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state', type=Path, required=True, help='Evaluation state directory (outputs)')
    parser.add_argument('--run', type=Path, required=True, help='Training run directory (events.jsonl, step-*)')
    parser.add_argument('--input', type=Path, required=True, help='The verified training input bundle')
    parser.add_argument('--code', type=Path, required=True, help='The staged code the run trained with')
    parser.add_argument('--step', type=int, default=20000, help='Frozen full-restart step (default 20000)')
    parser.add_argument('--scorer', type=Path, required=True,
                        help='The scorer file that will run; its sha256 is frozen in run-plan.json')
    parser.add_argument('--panel', choices=['fresh', 'published'], required=True)
    parser.add_argument('--published-panel', type=Path,
                        help='published mode: the training-panel.json.gz of the step-6,500 ranking')
    parser.add_argument('--cross-check-from', type=int,
                        help=f'fresh mode: earlier full restart replayed to --step (default {DEFAULT_CROSS_CHECK_FROM})')
    parser.add_argument('--no-cross-check', action='store_true', help='fresh mode: skip the replay cross-check')
    args = parser.parse_args()
    if args.step < 1:
        parser.error('--step must be positive')
    cross_from = None
    if args.panel == 'published':
        if args.published_panel is None or not args.published_panel.is_file():
            parser.error('--panel published requires an existing --published-panel file')
        if args.cross_check_from is not None or args.no_cross_check:
            parser.error('The replay cross-check applies only to --panel fresh')
    else:
        if args.published_panel is not None:
            parser.error('--published-panel applies only to --panel published')
        if args.no_cross_check and args.cross_check_from is not None:
            parser.error('Choose either --cross-check-from or --no-cross-check')
        if not args.no_cross_check:
            cross_from = DEFAULT_CROSS_CHECK_FROM if args.cross_check_from is None else args.cross_check_from
            if not 0 <= cross_from < args.step:
                parser.error('--cross-check-from must be an earlier step than --step')
    if not args.scorer.is_file():
        parser.error('--scorer must name the scorer file that will run')
    state = args.state.resolve()
    state.mkdir(parents=True, exist_ok=True)
    if (state / 'panel-manifest.json').exists():
        raise SystemExit(f'Verified panel already prepared in {state}; refusing to run again')
    status = Status(state)
    signal.signal(signal.SIGTERM, terminated)
    try:
        prepare(args, state, status, cross_from)
    except BaseException as error:
        status('failed', f'failed during {status.stage}', failed_stage=status.stage,
               error=without_account(f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
