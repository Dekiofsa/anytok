"""Shared helpers for the hyak.train tests (no tests here).

* ``build_world``: a complete fake cluster layout in a temporary directory -- inputs that pass
  ``run_segment.verify_inputs`` (real recipe and plan, tiny corpus/model files), and staged code trees
  whose ``scripts/train_joint_smollm_input.py`` is a FAKE trainer (``FAKE_TRAINER_SOURCE``).  The fake
  reproduces the pinned trainers' argument checks, resume rules, ``save_checkpoint`` signatures and
  cadence, uniform pruning (``prune_checkpoints`` copied verbatim) and events, and can be told to stop
  (as on SIGTERM) or to "die" (``FakeHardKill``) at given steps through ``$HYAK_FAKE_CONTROL``.  The RL
  variant mimics the straight-BPB-reward trainer (D8): the policy flags, ``run_config.json``'s ``policy``
  and ``policy_reward``, the paired metrics of every update, and sanity results on corrected cuts (their
  byte counts differ from the uniform fake's); ``$HYAK_FAKE_CONTROL`` can break each of them.
* ``fake_memory_module``: stand-in for ``hyak.memory.chunked_hypernet``.
* ``isolated_imports``: every simulated segment is a new "process" for the trainer import.
* ``restart_available``: the invariant "a verified full restart exists".
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

from hyak.common.io import read_json
from hyak.train import retention, run_segment

#: Variant-specific fake of the pinned trainer; @@VARIANT@@ is replaced by 'uniform' or 'rl'.
FAKE_TRAINER_SOURCE = r'''
"""FAKE pinned trainer for hyak.train tests (variant @@VARIANT@@).  Not the real trainer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import tempfile
import time

import torch

from hyak.tests.test_train_support import FakeHardKill, invariant_hook

VARIANT = '@@VARIANT@@'
REPO = Path(__file__).resolve().parents[1]
CONTROL = {}
STOP_SEEN = []


class ZettHypernet:
    def __call__(self, forms, priors=None, source=None, lang_index=None, deterministic=True):
        return None


class FakeBandit:
    def __init__(self, size=1000, seed=43):
        self.rng = random.Random(seed)
        self.counts = torch.zeros(size, dtype=torch.int64)
        self.logits = torch.zeros(size, dtype=torch.float64)
        self.updates = 0
        self.settings = {'kind': 'shared_settings_mlp', 'size': size, 'seed': seed}

    def sample(self):
        return self.rng.randrange(len(self.counts))

    def update(self, index, reward):
        self.counts[index] += 1
        self.updates += 1
        self.logits[index] += 1e-3
        p = self.logits.softmax(0)
        return {'policy_reward_bpb': reward, 'policy_entropy': float(-(p * p.log()).sum()),
                'policy_effective_tokenizers': 999.0, 'policy_max_probability': float(p.max()),
                'policy_unique_tokenizers': int((self.counts > 0).sum()), 'policy_updates': self.updates}

    def weights_dict(self):
        return {'logits': self.logits.clone(), 'counts': self.counts.clone(), 'updates': self.updates,
                'settings': dict(self.settings)}

    def state_dict(self):
        return {**self.weights_dict(), 'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        self.logits = state['logits'].clone()
        self.counts = state['counts'].clone()
        self.updates = state['updates']
        self.rng.setstate(state['rng'])


def prune_checkpoints(output, protected_steps, before_save=False):
    """Verbatim copy of the uniform trainer's prune_checkpoints (188-206)."""
    completed = sorted(
        p for p in Path(output).glob("step-*")
        if p.is_dir() and (p / "state.pt").is_file()
    )
    if not protected_steps:
        if not before_save:
            for stale in completed[:-2]:
                shutil.rmtree(stale)
        return
    protected = [p for p in completed
                 if int(p.name.removeprefix("step-")) in protected_steps]
    disposable = [p for p in completed if p not in protected]
    stale = disposable if before_save and protected else disposable[:-1]
    for path in stale:
        shutil.rmtree(path)


def _real_sigterm():
    """SIGTERM to this process; returns once the trainer's handler has run (at most 5 s)."""
    os.kill(os.getpid(), signal.SIGTERM)
    deadline = time.monotonic() + 5
    while not STOP_SEEN and time.monotonic() < deadline:
        time.sleep(0.01)


def _crash(step, where):
    crash = CONTROL.get('crash') or {}
    return crash.get('step') == step and crash.get('where') == where


def save_checkpoint(output, step, lm, hn, optimizer, corpus, queue_batches,
                    py_rng, noise_rng, metrics, protected_steps=(), bandit=None,
                    full_restart=True):
    output = Path(output)
    final = output / f"step-{step:06d}"
    if final.exists():
        raise FileExistsError(final)
    if VARIANT == 'uniform':
        prune_checkpoints(output, protected_steps, before_save=True)
    if _crash(step, 'in_write'):
        stage = Path(tempfile.mkdtemp(prefix='.writing-', dir=output))
        (stage / 'lm.pt').write_bytes(b'partial write of a killed job')
        raise FakeHardKill(f'killed while writing step {step}')
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=output) as temporary:
        stage = Path(temporary)
        (stage / 'hypernet').mkdir()
        (stage / 'hypernet' / 'config.json').write_text(json.dumps({'step': step}))
        (stage / 'hypernet' / 'model.safetensors').write_bytes(f'hypernet {step}'.encode())
        torch.save({'lm': step}, stage / 'lm.pt')
        state = {'step': step, 'python_rng': py_rng.getstate(), 'metrics': metrics}
        if VARIANT == 'rl':
            torch.save(bandit.weights_dict(), stage / 'policy.pt')
            state['bandit'] = bandit.state_dict()
        if VARIANT == 'uniform' or full_restart:
            torch.save({'optimizer': step}, stage / 'optimizer.pt')
            torch.save(state, stage / 'state.pt')
        os.replace(stage, final)
    if CONTROL.get('sigterm_in_save') == step:
        _real_sigterm()                                 # a real SIGTERM while this save runs (after the rename)
    slow = CONTROL.get('slow_save') or {}
    if slow.get('step') == step:
        time.sleep(slow.get('seconds', 0.5))
    if _crash(step, 'after_replace'):
        raise FakeHardKill(f'killed after renaming step {step}, before the manifest hook')
    if VARIANT == 'uniform':
        prune_checkpoints(output, protected_steps)


def record(output, event):
    line = json.dumps(event, sort_keys=True)
    print(line, flush=True)
    with (Path(output) / "events.jsonl").open("a") as handle:
        handle.write(line + "\n")


def main():
    global CONTROL
    path = os.environ.get('HYAK_FAKE_CONTROL')
    CONTROL = json.loads(Path(path).read_text()) if path else {}
    parser = argparse.ArgumentParser()
    for name in ('--recipe', '--data-dir', '--data-manifest', '--eval-dir', '--output',
                 '--warmup-checkpoint', '--tokenizer-pool'):
        parser.add_argument(name, type=Path, required=True)
    if VARIANT == 'rl':
        parser.add_argument('--eval-tokenizers', type=Path, required=True)
        parser.add_argument('--policy-lr', type=float, default=0.001)
        parser.add_argument('--policy-entropy', type=float, default=0.0)
        parser.add_argument('--policy-baseline-decay', type=float, default=0.99)
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-1.7B')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--microbatch-size', type=int, default=4)
    parser.add_argument('--max-updates-this-run', type=int, default=None)
    parser.add_argument('--stop-at-update', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--protected-steps', type=int, nargs='*', default=[])
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if CONTROL.get('parser_error'):
        parser.error(CONTROL['parser_error'])
    # trainer 324-350 (minus CUDA)
    if any(step % args.save_every for step in args.protected_steps):
        parser.error("Protected steps must be divisible by --save-every")
    if args.stop_at_update and any(step > args.stop_at_update for step in args.protected_steps):
        parser.error("Protected steps must not exceed --stop-at-update")
    if args.max_updates_this_run is not None and args.stop_at_update is not None:
        parser.error("Choose only one update limit")
    if args.resume:
        if not args.output.is_dir():
            parser.error("--resume requires an existing run directory")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty unless --resume is set")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train(args, parser)
    finally:
        lock_handle.close()   # a real process releases the lock when it exits


def _train(args, parser):
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        STOP_SEEN.append(signum)
        print(json.dumps({"stage": "stop_requested", "signal": signum}), flush=True)

    if CONTROL.get('sleep_before_handlers'):
        time.sleep(CONTROL['sleep_before_handlers'])     # the real trainer's start-up before its handlers
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = {'model': args.model, 'warmup_checkpoint': str(args.warmup_checkpoint.resolve()),
              'microbatch_size': args.microbatch_size, 'seed': args.seed, 'variant': VARIANT}
    if args.protected_steps:
        config['protected_steps'] = sorted(args.protected_steps)
    if VARIANT == 'rl':
        config['fixed_eval_sha256'] = hashlib.sha256(args.eval_tokenizers.read_bytes()).hexdigest()
        config['policy'] = {'kind': 'shared_settings_mlp', 'size': 1000, 'seed': args.seed + 1,
                            'learning_rate': args.policy_lr, 'entropy_weight': args.policy_entropy,
                            'baseline_decay': args.policy_baseline_decay}
        config['policy_reward'] = CONTROL.get('policy_reward', 'paired_pre_model_update_transferred_bpb')
        config['pairing_protocol'] = 'nearest_valid_canonical_midpoint_boundary_joint_128_positions_v1'
        baseline = json.loads((args.tokenizer_pool.parent / 'uniform_run_config.json').read_text())
        if any(baseline[key] != config[key] for key in ('microbatch_size', 'seed')):
            raise ValueError('RL settings differ from the uniform baseline')
    config_path = args.output / "run_config.json"
    if args.resume:
        completed = sorted(p for p in args.output.glob("step-*") if p.is_dir() and (p / "state.pt").is_file())
        if not completed:
            raise ValueError("No complete checkpoint to resume")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume settings differ from the original run")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    completed = sorted(p for p in args.output.glob("step-*") if p.is_dir() and (p / "state.pt").is_file())
    last = completed[-1] if args.resume else None
    py_rng = random.Random(args.seed)
    bandit = FakeBandit() if VARIANT == 'rl' else None
    start_update = 0
    if last:
        state = torch.load(last / "state.pt", map_location="cpu", weights_only=True)
        start_update = state["step"]
        py_rng.setstate(state["python_rng"])
        if bandit is not None:
            bandit.load_state_dict(state['bandit'])
            if bandit.updates != start_update:
                raise ValueError('Policy and model update counters differ')
    specs_path = args.output / "heldout_tokenizers.json"
    initial_eval_path = args.output / "initial_eval.json"
    if last:
        initial_results = json.loads(initial_eval_path.read_text())
    else:
        if VARIANT == 'rl':
            eval_specs = json.loads(args.eval_tokenizers.read_text())
        else:
            eval_specs = [[["<|endoftext|>", 0.0], ["a", -1.5]], [["<|endoftext|>", 0.0], ["b", -2.5]]]
        specs_path.write_text(json.dumps(eval_specs, ensure_ascii=False) + "\n")
        initial_results = [{"generated_bpb": 3.25, "canonical_bpb": 1.0633, "examples": 92, "bytes": 10452,
                            "truncated_forms": 596},
                           {"generated_bpb": 3.5, "canonical_bpb": 1.0611, "examples": 92, "bytes": 10610,
                            "truncated_forms": 88}]
        if VARIANT == 'rl' and not CONTROL.get('original_cuts'):
            # the corrected paired cuts move the boundaries: other continuation bytes than the uniform run's
            initial_results = [{**r, "bytes": r["bytes"] + 37, "generated_bpb": r["generated_bpb"] - 0.01}
                               for r in initial_results]
        initial_eval_path.write_text(json.dumps(initial_results, indent=2) + "\n")
        record(args.output, {"stage": "initial_eval", "results": initial_results})
    if CONTROL.get('sigterm_after_initial_eval'):
        _real_sigterm()                                 # a real stop signal right after the (initial) evaluation
    if args.stop_at_update is not None:
        stop_update = args.stop_at_update
    elif args.max_updates_this_run is not None:
        stop_update = min(190_000, start_update + args.max_updates_this_run)
    else:
        stop_update = 190_000
    if stop_update <= start_update:
        raise ValueError("Requested stopping update has already been reached")
    if CONTROL.get('return_without_save') and last:
        return                                          # a trainer that returns without saving anything new
    saved = last is not None
    if VARIANT == 'rl' and not last:
        save_checkpoint(args.output, 0, None, None, None, None, [], py_rng, None,
                        {'stage': 'initial_checkpoint', 'step': 0}, args.protected_steps, bandit=bandit,
                        full_restart=True)
        saved = True
    stops = set(CONTROL.get('stop_at', []))
    for update in range(start_update, stop_update):
        if saved:
            invariant_hook(args.output, VARIANT)
        step = update + 1
        index = bandit.sample() if bandit is not None else py_rng.randrange(1000)
        py_rng.random()
        metrics = {"stage": "train", "step": step, "loss_nats_per_token": 2.0, "input_lexical_loss": 0.5,
                   "scored_tokens": 8000, "hn_lr": 0.0, "lm_lr": 0.0, "gradient_norm_before_clip": 1.0,
                   "seconds": float(CONTROL.get('seconds', 2.0)), "peak_gib": 70.5,
                   "tokenizer_pool_index": index, "tokenizer_config_index": index,
                   "tokenizer_vocab_size": 32000 + 1000 * (index % 97), "truncated_forms": 3,
                   "rejected_examples": 0}
        if bandit is not None:
            # the straight-BPB-reward trainer: the policy receives train_bpb; the paired gap is a diagnostic
            train_bpb, canonical_bpb = 1.25, 1.0
            gap = train_bpb - canonical_bpb + CONTROL.get('gap_error', 0.0)
            policy = bandit.update(index, gap if CONTROL.get('reward_gap') else train_bpb)
            metrics.update(scored_bytes=27000, train_bpb=train_bpb, paired_canonical_bpb=canonical_bpb,
                           paired_gap_bpb=gap, paired_valid_examples=CONTROL.get('paired_valid_examples', 128),
                           paired_maximum_positions=CONTROL.get('paired_maximum_positions', 127),
                           paired_batch_sha256='f' * 64, policy_action_probability=0.001, **policy)
        record(args.output, metrics)
        if CONTROL.get('sleep'):
            time.sleep(CONTROL['sleep'])
        if step in stops:
            stop_requested = True       # as if SIGTERM arrived during this update
        if CONTROL.get('kill_at') == step:
            raise FakeHardKill(f'killed during update {step}')
        if CONTROL.get('raise_at') == step:
            raise RuntimeError(f'ordinary trainer exception at update {step} (simulated)')
        if bandit is not None and CONTROL.get('bandit_extra_update') and step in stops:
            bandit.updates += 1                          # policy counter off by one before the stop save
        should_eval = step % args.eval_every == 0 or step == stop_update or stop_requested
        should_save = step % args.save_every == 0 or step == stop_update or stop_requested
        if should_eval:
            if CONTROL.get('sigterm_in_eval') == step:
                _real_sigterm()                             # a real stop signal during this evaluation
            record(args.output, {"stage": "eval", "step": step, "results": initial_results})
        if should_save:
            extra = {}
            if VARIANT == 'rl':
                extra = {'bandit': bandit,
                         'full_restart': (step % 2000 == 0 or step == stop_update or stop_requested)}
            save_checkpoint(args.output, step, None, None, None, None, [], py_rng, None, metrics,
                            args.protected_steps, **extra)
            saved = True
            invariant_hook(args.output, VARIANT)
            record(args.output, {"stage": "checkpoint", "step": step})
        if stop_requested:
            stop_update = step
            break
    if CONTROL.get('no_final_event'):
        return
    record(args.output, {"stage": "complete" if stop_update == 190_000 else "paused_at", "step": stop_update})
'''


class FakeHardKill(BaseException):
    """Simulates SIGKILL / node failure inside the trainer: nothing after it runs in that "process"."""


#: Number of invariant checks performed (tests assert it is non-zero).
INVARIANT = {'checks': 0, 'enabled': False}


def restart_available(run_dir, run) -> int:
    """The newest directory with state.pt has a full-restart manifest whose files exist; returns its step."""
    full = retention.full_restarts(run_dir)
    assert full, f'no full restart in {run_dir}'
    step, path = full[-1]
    manifest = read_json(path / 'manifest.json')
    assert manifest['kind'] == retention.FULL and manifest['step'] == step, manifest
    assert set(manifest['files']) == retention.full_files(run), sorted(manifest['files'])
    for name in manifest['files']:
        assert (path / name).is_file(), path / name
    return step


def invariant_hook(run_dir, run):
    if INVARIANT['enabled']:
        restart_available(run_dir, run)
        INVARIANT['checks'] += 1


@dataclass
class World:
    """A fake cluster layout (see build_world)."""

    tmp: object
    cfg: dict
    paths: object
    expected_code: dict
    init_bytes: int
    control_file: Path
    model_expected: dict = None


def _gz_jsonl(path: Path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for i, text in enumerate(texts):
            stream.write(json.dumps({'id': str(i), 'text': text}, ensure_ascii=False) + '\n')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fake_code_tree(code_dir: Path, run: str) -> dict:
    """Staged-layout tree with the fake trainer; returns ``{rel: sha256}`` (the test's pinned expectation)."""
    if code_dir.exists():
        shutil.rmtree(code_dir)
    files = {retention.TRAINER_REL: FAKE_TRAINER_SOURCE.replace('@@VARIANT@@', run),
             'hf_hypernet/__init__.py': '',
             'hf_hypernet/joint_training.py': '# fake helpers (tests)\n',
             'hf_hypernet/pool_sampling.py': '# fake pool (tests)\n'}
    if run == 'rl':
        files[retention.POLICY_REL] = '# fake settings policy (tests)\n'
    for rel, text in files.items():
        path = code_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    so = code_dir / run_segment.SO_REL
    so.parent.mkdir(parents=True, exist_ok=True)
    so.write_bytes(b'\x7fELF fake rust library for tests')
    return {rel: _sha(code_dir / rel) for rel in [*files, run_segment.SO_REL]}


#: Model files a production ``input/manifest.json`` lists (hyak.corpus.stage_inputs); a test-scale stage has none.
STAGED_MODEL_FILES = ('model/config.json', 'model/model.safetensors', 'model/tokenizer.json')

#: The recipe a fake ``init_manifest.json`` records (option (b), the default of hyak.model.init_hypernet).
FAKE_INIT_RECIPE = {'option': '(b) calibrated with the zero bias, then bias drawn (default)', 'seed': 42,
                    'prior_bias_std': 0.02, 'calibration': 'before_bias',
                    'rescalers_from_fixture': {'file': 'original_in_scaler.npz', 'sha256': 'f' * 64},
                    'prior_bias_seed': 1042, 'rescaler_init': True}


def write_init_manifest(paths, *, recipe=None, outputs=None) -> dict:
    """``input/hypernet_init/init_manifest.json`` for the files there (or the given ``outputs``)."""
    if outputs is None:
        outputs = {name: {'bytes': (paths.hypernet_init / name).stat().st_size, 'sha256': _sha(paths.hypernet_init / name)}
                   for name in ('config.json', 'model.safetensors')}
    manifest = {'format': 'fake', 'complete': True, 'recipe': dict(FAKE_INIT_RECIPE if recipe is None else recipe),
                'outputs': outputs}
    paths.hypernet_init_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def write_inputs(paths, repo_root: Path, *, sparse_init: bool = False) -> int:
    """Inputs that pass verify_inputs; returns the size of hypernet_init/model.safetensors."""
    paths.input.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_root / 'data/balanced-full-v3.json', paths.recipe)
    recipe = read_json(paths.recipe)
    corpus = {}
    for bucket in recipe['buckets']:
        name = bucket['name']
        train = paths.input_corpus_train(name)
        _gz_jsonl(train, [f'{name} document {i} ' + 'lorem ipsum dolor sit amet ' * 8 for i in range(5)])
        corpus[name] = {'compressed_bytes': train.stat().st_size, 'sha256': _sha(train)}
        _gz_jsonl(paths.input_eval_file(name), [f'{name} held-out text ' * 10 for _ in range(4)])
    paths.input_corpus_manifest.write_text(json.dumps(corpus, indent=2, sort_keys=True) + '\n')
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        (paths.model_dir / name).write_text(f'fake {name}\n')
    paths.hypernet_init.mkdir(parents=True, exist_ok=True)
    (paths.hypernet_init / 'config.json').write_text('{"fake": "hypernetwork config"}\n')
    model = paths.hypernet_init / 'model.safetensors'
    if sparse_init:
        with open(model, 'wb') as stream:
            stream.truncate(run_segment.HYPERNET_INIT_BYTES)
    else:
        model.write_bytes(b'fake initial hypernetwork weights')
        # as hyak.model.init_hypernet writes it (outputs + recipe); a sparse world has none (one warning)
        write_init_manifest(paths)
    paths.pool_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_root / 'plans/tokenizer-pool-1000-v1.json', paths.pool_plan)
    plan_sha = read_json(paths.pool_plan)['plan_sha256']
    paths.pool_results.write_text(json.dumps({'plan_sha256': plan_sha, 'completed': 1000, 'tokenizers': [
        {'status': 'complete', 'config': {'index': i}} for i in range(1000)]}) + '\n')
    # as hyak.corpus.stage_inputs writes a production stage: recipe, corpus, held-out files and the three
    # model files (hypernet_init is verified through init_sha256, the pool through its plan and index)
    listed = {}
    for path in sorted(paths.input.rglob('*')):
        rel = path.relative_to(paths.input).as_posix()
        if path.is_file() and path != paths.input_manifest and (rel.split('/')[0] in ('recipe.json', 'corpus', 'eval')
                                                                 or rel in STAGED_MODEL_FILES):
            listed[rel] = {'bytes': path.stat().st_size, 'sha256': _sha(path)}
    paths.input_manifest.write_text(json.dumps({'kind': 'anytok_hyak_inputs', 'files': listed}, indent=2) + '\n')
    return model.stat().st_size


def model_hashes(paths) -> dict:
    """``{name: {"bytes", "sha256"}}`` of the fake model files (the test stand-in for fetch_model's pins)."""
    return {name: {'bytes': (paths.model_dir / name).stat().st_size, 'sha256': _sha(paths.model_dir / name)}
            for name in ('config.json', 'model.safetensors', 'tokenizer.json')}


def build_world(make_tmp_config, repo_root: Path, *, sparse_init: bool = False, **overrides) -> World:
    tmp = make_tmp_config(**overrides)
    paths = tmp.paths
    init_bytes = write_inputs(paths, repo_root, sparse_init=sparse_init)
    expected = {run: write_fake_code_tree(paths.code_dir(run), run) for run in ('uniform', 'rl')}
    paths.code_manifest.write_text(json.dumps({'kind': 'test code manifest'}) + '\n')
    return World(tmp=tmp, cfg=tmp.cfg, paths=paths, expected_code=expected, init_bytes=init_bytes,
                 control_file=tmp.base / 'fake-control.json', model_expected=model_hashes(paths))


def fake_memory_module():
    """Stand-in for hyak.memory.chunked_hypernet with the DESIGN 5.D interface."""
    module = types.ModuleType('hyak.memory.chunked_hypernet')
    module.patched = {}

    def install(hypernet_class, chunk_rows):
        if module.patched.get(hypernet_class, chunk_rows) != chunk_rows:
            raise RuntimeError('different chunk setting')
        module.patched[hypernet_class] = chunk_rows

    def is_installed(hypernet_class):
        return hypernet_class in module.patched

    def settings(hypernet_class=None):
        rows = next(iter(module.patched.values()), 0)
        return {'chunk_rows': rows, 'use_checkpoint': True, 'installed': bool(module.patched),
                'module_sha256': 'c' * 64, 'algorithm_version': 1}

    module.install, module.is_installed, module.settings = install, is_installed, settings
    return module


_ISOLATED_PREFIXES = ('scripts', 'hf_hypernet', 'rust_utils')


@contextmanager
def isolated_imports():
    """Forget the trainer/hf_hypernet modules and restore sys.path afterwards (a new 'process')."""
    def purge() -> dict:
        removed = {}
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + '.') for p in _ISOLATED_PREFIXES):
                removed[name] = sys.modules.pop(name)
        return removed
    saved_path = list(sys.path)
    saved = purge()
    try:
        yield
    finally:
        # forget what the simulated process imported, then give the rest of the test session back the
        # modules it had before (a purge must not leak into other test files)
        purge()
        sys.modules.update(saved)
        sys.path[:] = saved_path


class H200Adapter(run_segment.TrainerAdapter):
    """The real adapter, reporting an H200 so tests see no GPU warning."""

    def gpu_info(self):
        info = super().gpu_info()
        return {**info, 'name': 'NVIDIA H200', 'count': 1}


def run_once(world: World, run: str, monkeypatch, *, control=None, remaining=None, smoke_phase=None,
             max_seconds=None, adapter=H200Adapter, memory=None, **kwargs):
    """One simulated process: run_segment.run_segment with the fake trainer; returns its outcome."""
    world.control_file.write_text(json.dumps(control or {}))
    monkeypatch.setenv('HYAK_FAKE_CONTROL', str(world.control_file))
    monkeypatch.setattr('hyak.common.slurm.seconds_remaining', lambda default=None, now=None: remaining)
    monkeypatch.setitem(sys.modules, 'hyak.memory.chunked_hypernet', memory or fake_memory_module())
    kwargs.setdefault('model_expected', world.model_expected)
    with isolated_imports():
        return run_segment.run_segment(run, smoke_phase=smoke_phase, max_seconds=max_seconds,
                                       adapter_factory=adapter, expected_code=world.expected_code[run],
                                       init_bytes=world.init_bytes, **kwargs)


def trainer_events(run_dir: Path) -> list:
    return [json.loads(line) for line in (Path(run_dir) / 'events.jsonl').read_text().splitlines() if line.strip()]


def monitor_events(world: World, stage: str) -> list:
    path = world.paths.monitor_dir(stage) / 'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def step_kinds(run_dir: Path) -> dict:
    """``{step: kind}`` from each step directory's manifest (None when missing)."""
    rows = {}
    for step, path in retention.step_dirs(run_dir):
        manifest = path / 'manifest.json'
        rows[step] = read_json(manifest)['kind'] if manifest.is_file() else None
    return rows
