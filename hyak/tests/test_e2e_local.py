"""Opt-in end-to-end check: the REAL pinned trainers through the production wrapper on CPU (stage A).

Skipped unless ``HYAK_TEST_E2E=1`` and the local scratch assets exist (macOS build of the pinned Rust
crate, the miniature corpus, the SmolLM2 tokenizer, three smoke tokenizers; see
``hyak/tests/e2e/world.py``).  It builds a fresh miniature world in the pytest temporary directory and
runs exactly what the cluster runs, as child processes of ``python -m hyak.train.run_segment``:

1. ``--run uniform --smoke``  (bootstrap of one update, then a resume to update 3 in a new process);
2. ``--run uniform``          (the production bootstrap segment, which freezes the uniform/RL pair);
3. ``--run rl --smoke``       (RL fresh start to update 2, then a resume to update 3; policy counters 3; the
   staged RL trainer is the straight-BPB-reward variant of the joint regret trainer (D8) and each phase must
   pass the paired-protocol receipt: entropy 0, ``policy_reward`` straight, 128 paired examples, at most 128
   positions, ``paired_gap_bpb == train_bpb - paired_canonical_bpb``, ``policy_reward_bpb == train_bpb``).

Then the evaluation entry points on the uniform smoke checkpoint the real trainer wrote (step 3):
``hyak.eval.rebuild_panel --panel natural``, ``prepare_features`` and ``validate_prefixes --panel natural``
for pool index 3, ``score --run uniform --panel natural --checkpoint <step-000003>`` (the real TorchScorer
on the CPU shim, with ``--allow-non-final-checkpoint``) and ``pack --panel natural --runs uniform --partial``
(only one of the 38 natural-panel indices is validated).  The pool panel cannot be validated with the
world's 4,000-entry smoke tokenizers (29 originally valid contexts would need a trim: exit 3), so the
natural panel is the one exercised here (see ``hyak/tests/e2e/eval_world.py``).

The CUDA-to-CPU shim (``hyak/tests/e2e/shim``) is activated only through the children's ``PYTHONPATH``.
About 4-8 minutes on a 10-core Mac (2 trainer threads), plus about 2 minutes of evaluation.  ``HYAK_TEST_E2E_KEEP=DIR`` builds the world in
DIR instead (kept for inspection).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hyak.code import pinned
from hyak.tests.e2e import world as W

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(os.environ.get('HYAK_TEST_E2E') != '1',
                                 reason='end-to-end CPU run of the real trainers; set HYAK_TEST_E2E=1')]

SHIM_MODULE = 'hyak_e2e_cpu_shim'
#: The 21 settings the RL trainer compares with input/uniform_run_config.json (RL trainer paired_keys).
PAIRED_KEYS = ('recipe_sha256', 'data_manifest_sha256', 'main_updates', 'batch_size', 'microbatch_size',
               'pool_plan_sha256', 'pool_size', 'pool_index_sha256', 'pool_sampling_sha256', 'pool_priors', 'max_pieces',
               'hn_peak_lr', 'lm_peak_lr', 'lr_ramp_updates', 'lr_final_fraction', 'lexical_loss_weight',
               'sequence_length', 'sampler_text_chars', 'data_snippet_chars', 'min_data_chars', 'seed')


def _segment(world: Path, *args, attempts: int = 3) -> dict:
    """One production CLI call; a fresh start may be retried when the miniature held-out set is too small."""
    env = W.world_env(world, {'HYAK_E2E_TORCH_THREADS': os.environ.get('HYAK_E2E_TORCH_THREADS', '2')})
    for _ in range(attempts):
        result = subprocess.run([sys.executable, '-m', 'hyak.train.run_segment', *args], env=env, cwd=str(W.REPO),
                                capture_output=True, text=True, timeout=3600)
        if result.returncode == 0 or 'usable held-out examples' not in result.stderr:
            break
    assert result.returncode == 0, result.stderr[-4000:]
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    final = [row for row in rows if row.get('stage') == 'segment_result']
    assert final, result.stdout[-4000:]
    return {'result': final[-1], 'rows': rows}


@pytest.fixture(scope='module')
def e2e_world(tmp_path_factory):
    missing = W.missing_assets()
    if missing:
        pytest.skip(f'local e2e assets missing: {missing}')
    keep = os.environ.get('HYAK_TEST_E2E_KEEP')
    world = Path(keep) if keep else tmp_path_factory.mktemp('e2e') / 'world'
    W.build(world, force=True, log=lambda *_: None)
    return world


def test_production_modules_never_import_the_shim():
    for path in (W.REPO / 'hyak').rglob('*.py'):
        if 'tests' in path.relative_to(W.REPO / 'hyak').parts:
            continue
        text = path.read_text(encoding='utf-8')
        assert SHIM_MODULE not in text and 'hyak.tests' not in text, path


def test_stage_a_smoke_sequence_with_the_real_trainers(e2e_world):
    world = e2e_world
    runs = world / 'runs'
    uniform = _segment(world, '--run', 'uniform', '--smoke')
    assert uniform['result']['outcome'] == 'smoke_verified'
    assert uniform['result']['checkpoints'] == [1, 3] and uniform['result']['train_steps'] == [1, 2, 3]
    segment = json.loads((runs / 'smoke/uniform/hyak_segment.json').read_text())
    assert segment['chunk_identity']['chunk_rows'] == W.CHUNK_ROWS and 'e2e test shim' in segment['gpu_name']
    for step in (1, 3):
        manifest = json.loads((runs / f'smoke/uniform/step-{step:06d}/manifest.json').read_text())
        assert manifest['kind'] == 'full_restart_checkpoint'
        assert sorted(manifest['files']) == ['hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt',
                                             'optimizer.pt', 'state.pt']

    bootstrap = _segment(world, '--run', 'uniform')
    assert bootstrap['result']['outcome'] == 'bootstrap_done' and bootstrap['result']['step'] == 1
    assert any(row.get('stage') == 'pairing_frozen' for row in bootstrap['rows'])
    assert (world / 'root/input/pairing.json').is_file()

    rl = _segment(world, '--run', 'rl', '--smoke')
    assert rl['result']['outcome'] == 'smoke_verified'
    assert rl['result']['checkpoints'] == [3] and rl['result']['policy_updates'] == 3
    # D8: the real straight-BPB-reward trainer ran through the production wrapper and passed the receipt
    receipts = [row for row in rl['rows'] if row.get('stage') == 'paired_protocol_verified']
    assert [row['step'] for row in receipts] == [2, 3]
    final = rl['result']['paired_protocol']
    assert final['step'] == 3 and final['metrics_checked'] and final['entropy_weight'] == 0
    assert final['policy_reward'] == 'paired_pre_model_update_transferred_bpb'
    assert final['paired_valid_examples'] == 128 and final['paired_maximum_positions'] <= 128
    assert abs(final['policy_reward_bpb'] - final['train_bpb']) <= 1e-12
    assert abs(final['paired_gap_bpb'] - (final['train_bpb'] - final['paired_canonical_bpb'])) <= 1e-10
    assert rl['result']['sanity_tokenizers_equal_uniform'] is True and 'narrowed (D8)' in rl['result']['initial_eval_check']
    segment = json.loads((runs / 'smoke/rl/hyak_segment.json').read_text())
    assert segment['trainer_sha256'] == pinned.RL_TRAINER_SHA256 == segment['trainer_variant']['sha256']
    assert segment['trainer_variant']['base_sha256'] == pinned.REGRET_TRAINER_SHA256
    config = json.loads((runs / 'smoke/rl/run_config.json').read_text())
    assert config['training_script_sha256'] == pinned.RL_TRAINER_SHA256
    assert config['policy']['entropy_weight'] == 0 and config['policy']['learning_rate'] == 0.001
    assert config['policy']['baseline_decay'] == 0.99 and config['policy']['seed'] == 43
    assert config['pairing_protocol'] == 'nearest_valid_canonical_midpoint_boundary_joint_128_positions_v1'
    assert config['paired_reward_source_sha256'] == pinned.PAIRED_REWARD_SHA256
    # the trainer's own paired-key check passed against the uniform bootstrap's config (it raises otherwise)
    uniform_config = json.loads((world / 'root/input/uniform_run_config.json').read_text())
    for key in PAIRED_KEYS:
        assert config[key] == uniform_config[key], key
    train = [json.loads(line) for line in (runs / 'smoke/rl/events.jsonl').read_text().splitlines()
             if json.loads(line).get('stage') == 'train']
    assert [row['step'] for row in train] == [1, 2, 3]
    assert all(row['policy_reward_bpb'] == row['train_bpb'] and row['paired_valid_examples'] == 128 for row in train)
    manifest = json.loads((runs / 'smoke/rl/step-000003/manifest.json').read_text())
    assert manifest['kind'] == 'full_restart_checkpoint' and 'policy.pt' in manifest['files']
    pruned = [row for row in rl['rows'] if row.get('stage') == 'pruned_restarts']
    assert [row['deleted'] for row in pruned] == [[0], [2]]


def _eval(world: Path, module: str, *args) -> subprocess.CompletedProcess:
    """One production evaluation CLI call (CPU shim and the world's explicit test parameters via world_env)."""
    return subprocess.run([sys.executable, '-m', f'hyak.eval.{module}', *args], env=W.world_env(world),
                          cwd=str(W.REPO), capture_output=True, text=True, timeout=1800)


def test_evaluation_scores_one_tokenizer_on_a_real_trainer_checkpoint(e2e_world):
    world = e2e_world
    checkpoint = world / 'runs' / 'smoke' / 'uniform' / 'step-000003'
    if not (checkpoint / 'manifest.json').is_file():
        pytest.skip('stage A did not leave the uniform smoke checkpoint')
    result = _eval(world, 'rebuild_panel', '--panel', 'natural')
    assert result.returncode == 0 and '2032/2032 rows verified' in result.stdout, result.stderr[-3000:]
    result = _eval(world, 'prepare_features', '--indices', '3', '--workers', '1', '--checkpoint', str(checkpoint),
                   '--no-status')
    assert result.returncode == 0, result.stderr[-3000:]
    result = _eval(world, 'validate_prefixes', '--panel', 'natural', '--indices', '3', '--workers', '1',
                   '--no-status')
    assert result.returncode == 0, result.stderr[-3000:]
    # step 3 of the smoke run is not the final step: the explicit flag of hyak.eval.score (tests, diagnostics)
    result = _eval(world, 'score', '--run', 'uniform', '--panel', 'natural', '--checkpoint', str(checkpoint),
                   '--allow-non-final-checkpoint')
    assert result.returncode == 0 and '0003: scored in' in result.stdout, result.stderr[-3000:]
    out = world / 'root' / 'eval' / 'scores' / 'uniform' / 'natural'
    plan = json.loads((out / 'plan.json').read_text())
    model = plan['models'][0]
    # the plan names the checkpoint by its training manifest, not by its location (an archived copy continues
    # the same comparison); the path is kept in each result file as information
    assert model['name'] == 'uniform20k-hyak' and model['run'] == 'uniform' and 'checkpoint_path' not in model
    assert model['manifest']['source'] == 'training manifest.json' and model['manifest']['step'] == 3
    scores = json.loads((out / '0003.json').read_text())
    assert scores['checkpoint_path'] == str(checkpoint)
    controls = json.loads((out / 'canonical.json').read_text())
    for values in (scores['transferred'], controls['canonical'], controls['no_prefix']):
        assert len(values) == 2032 and all(len(row) == 32 for row in values)
        assert all(0 <= x < 100 for row in values for x in row)
    assert 'e2e test shim' in scores['gpu'] and (out / 'complete.json').is_file()
    status = json.loads((world / 'monitor' / 'eval-uniform' / 'status.json').read_text())
    assert status['completed'] == 1 and status['total'] == 1 and status['panel'] == 'natural'
    assert status['state'] == 'running' and 'pool panel not complete' in status['stage']
    # only index 3 of the 38 natural-panel indices was validated here: pack refuses that without --partial
    result = _eval(world, 'pack', '--panel', 'natural', '--runs', 'uniform')
    assert result.returncode == 2 and '--partial' in result.stderr, result.stderr[-3000:]
    result = _eval(world, 'pack', '--panel', 'natural', '--runs', 'uniform', '--partial')
    assert result.returncode == 0, result.stderr[-3000:]
    summary = json.loads((world / 'root' / 'eval' / 'reports' / 'natural' / 'summary.json').read_text())
    assert summary['models'] == ['uniform20k-hyak'] and summary['runs'] == {'uniform': 'complete'}
    assert summary['partial'] is True and 3 not in summary['missing_indices']
