"""hyak.train.pairing (D6): every branch of freezing and verifying the uniform files the RL run uses."""
from __future__ import annotations

import json
import os

import pytest

from hyak.common.io import read_json, sha256_file
from hyak.train import pairing as P
from hyak.train import retention as R
from hyak.tests.test_train_retention import make_checkpoint

SEGMENT = {'run': 'uniform', 'chunk_identity': {'chunk_rows': 16384, 'use_checkpoint': True,
                                                 'module_sha256': 'c' * 64, 'algorithm_version': 1},
           'init_sha256': {'config.json': 'a' * 64, 'model.safetensors': 'b' * 64},
           'hf_hypernet_sha256': {'hf_hypernet/joint_training.py': '1' * 64, 'hf_hypernet/pool_sampling.py': '2' * 64}}


def bootstrap_uniform(paths, step=1):
    """A uniform run directory as the bootstrap segment leaves it."""
    run = paths.run_dir('uniform')
    make_checkpoint(run, step)
    (run / 'run_config.json').write_text(json.dumps({'seed': 42, 'microbatch_size': 32}, indent=2) + '\n')
    (run / 'heldout_tokenizers.json').write_text(json.dumps([[['<|endoftext|>', 0.0]]]) + '\n')
    (run / R.SEGMENT_RECORD).write_text(json.dumps(SEGMENT))
    return run


def test_uniform_not_started(tmp_config):
    with pytest.raises(P.PairingError, match='no full restart yet'):
        P.ensure(tmp_config.paths)
    assert not P.is_frozen(tmp_config.paths)


def test_uniform_bootstrapped_freezes_once(tmp_config):
    paths = tmp_config.paths
    run = bootstrap_uniform(paths)
    events = []
    record = P.ensure(paths, emit=lambda name, **f: events.append(name))
    assert events == ['pairing_frozen'] and P.is_frozen(paths)
    assert paths.uniform_run_config.read_bytes() == (run / 'run_config.json').read_bytes()
    assert paths.uniform_heldout.read_bytes() == (run / 'heldout_tokenizers.json').read_bytes()
    assert record['uniform_full_restart_step_at_freeze'] == 1
    assert record['uniform_run_config']['sha256'] == sha256_file(run / 'run_config.json')
    assert record['uniform_segment'] == SEGMENT
    assert read_json(paths.pairing) == record
    # frozen copies present and equal: verified, nothing rewritten
    stamp = os.stat(paths.pairing).st_mtime_ns
    assert P.ensure(paths, role='uniform') == record
    assert os.stat(paths.pairing).st_mtime_ns == stamp
    assert P.verify_uniform_unchanged(paths) == record


def test_frozen_copies_or_uniform_files_differ(tmp_config):
    paths = tmp_config.paths
    run = bootstrap_uniform(paths)
    P.ensure(paths)
    original = paths.uniform_heldout.read_bytes()
    paths.uniform_heldout.write_bytes(original + b' ')
    with pytest.raises(P.PairingError, match='changed after the freeze'):
        P.ensure(paths)
    paths.uniform_heldout.write_bytes(original)
    (run / 'run_config.json').write_text('{"seed": 43}\n')       # uniform restarted from scratch
    with pytest.raises(P.PairingError, match='restarted from scratch'):
        P.ensure(paths)
    with pytest.raises(P.PairingError):
        P.verify_uniform_unchanged(paths)


def test_uniform_dir_wiped_after_freeze(tmp_config):
    paths = tmp_config.paths
    run = bootstrap_uniform(paths)
    P.ensure(paths)
    for path in sorted(run.rglob('*'), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    with pytest.raises(P.PairingError, match='wiped after the pair was frozen'):
        P.ensure(paths)
    assert P.is_frozen(paths)


def test_rl_started_before_freeze(tmp_config):
    paths = tmp_config.paths
    bootstrap_uniform(paths)
    rl = paths.run_dir('rl')
    rl.mkdir(parents=True)
    (rl / 'run_config.json').write_text('{}')
    with pytest.raises(P.RLDirectoryNotEmpty, match='started before the uniform pair was frozen'):
        P.ensure(paths)
    with pytest.raises(P.RLDirectoryNotEmpty):
        P.ensure(paths, role='uniform')
    assert not paths.pairing.exists() and not paths.uniform_run_config.exists()


def test_interrupted_freeze_is_completed_or_refused(tmp_config):
    paths = tmp_config.paths
    run = bootstrap_uniform(paths)
    paths.input.mkdir(parents=True, exist_ok=True)
    paths.uniform_run_config.write_bytes((run / 'run_config.json').read_bytes())   # died after one copy
    assert P.is_frozen(paths)
    P.verify_uniform_unchanged(paths)
    record = P.ensure(paths)
    assert record['uniform_run_config']['sha256'] == sha256_file(run / 'run_config.json')
    paths.pairing.unlink()
    paths.uniform_heldout.write_text('something else\n')
    with pytest.raises(P.PairingError, match='exists without'):
        P.ensure(paths)
    with pytest.raises(P.PairingError, match='interrupted freeze'):
        P.verify_uniform_unchanged(paths)


def test_manifestless_uniform_restart_and_moved_runs(tmp_config, make_tmp_config):
    paths = tmp_config.paths
    run = bootstrap_uniform(paths)
    (run / R.step_name(1) / 'manifest.json').unlink()
    with pytest.raises(P.PairingError, match='no manifest yet'):
        P.ensure(paths)
    make_checkpoint(run, 1)
    P.ensure(paths)
    moved = make_tmp_config(ANYTOK_RUNS=str(tmp_config.base / 'other-runs'))
    with pytest.raises(P.PairingError, match='ANYTOK_RUNS changed'):
        P.ensure(moved.paths)


def test_check_rl_settings():
    record = {'uniform_segment': SEGMENT}
    rl = {**SEGMENT, 'run': 'rl',
          'hf_hypernet_sha256': {**SEGMENT['hf_hypernet_sha256'], 'hf_hypernet/settings_policy.py': '9' * 64}}
    P.check_rl_settings(record, rl)
    with pytest.raises(P.PairingError, match='chunk_identity'):
        P.check_rl_settings(record, {**rl, 'chunk_identity': {**SEGMENT['chunk_identity'], 'chunk_rows': 8192}})
    with pytest.raises(P.PairingError, match='init_sha256'):
        P.check_rl_settings(record, {**rl, 'init_sha256': {'config.json': 'x'}})
    with pytest.raises(P.PairingError, match='hf_hypernet sources differ'):
        P.check_rl_settings(record, {**rl, 'hf_hypernet_sha256': {'hf_hypernet/joint_training.py': '0' * 64}})


def test_pairing_records_the_rl_trainer_variant_and_refuses_another(tmp_config):
    """D8: pairing.json names the RL trainer (the straight-BPB-reward variant of the joint regret trainer)."""
    from hyak.code import pinned
    paths = tmp_config.paths
    bootstrap_uniform(paths)
    record = P.ensure(paths)
    assert record['rl_trainer_variant'] == pinned.RL_TRAINER_VARIANT
    assert record['rl_trainer_variant']['base_path'] == 'runtime/anytok_pool_regret_rl_20k/scripts/train_joint_smollm_input.py'
    stale = {**record, 'rl_trainer_variant': {**record['rl_trainer_variant'], 'sha256': pinned.ORIGINAL_RL_TRAINER_SHA256}}
    paths.pairing.write_text(json.dumps(stale))
    with pytest.raises(P.PairingError, match='restore the checkout the pair was frozen with'):
        P.ensure(paths)
    legacy = {k: v for k, v in record.items() if k != 'rl_trainer_variant'}   # a record without the field: verified
    paths.pairing.write_text(json.dumps(legacy))
    assert P.ensure(paths) == legacy


def test_paired_reward_is_an_rl_only_source():
    uniform = {**SEGMENT}
    rl = {**SEGMENT, 'hf_hypernet_sha256': {**SEGMENT['hf_hypernet_sha256'], 'hf_hypernet/settings_policy.py': '3' * 64,
                                            'hf_hypernet/paired_reward.py': '4' * 64}}
    assert P.check_rl_settings({'uniform_segment': uniform}, rl) == []
    assert 'hf_hypernet/paired_reward.py' in P.RL_ONLY_SOURCES
