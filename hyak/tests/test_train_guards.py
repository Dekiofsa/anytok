"""Guards of hyak.train.run_segment that must not be lost (review finding tests-2).

Each test removes nothing from the wrapper; it drives the fake trainer into the one state the guard
exists for and expects the guard's refusal.  Written before the fix round changed run_segment.py.
"""
from __future__ import annotations

import json

import pytest

from hyak.common.io import read_json, sha256_file
from hyak.train import pairing
from hyak.train import run_segment as S
from hyak.tests import test_train_support as T


@pytest.fixture
def world(make_tmp_config, repo_root):
    return T.build_world(make_tmp_config, repo_root)


def _memory_with_module(sha):
    module = T.fake_memory_module()
    module.settings = lambda cls=None: {'chunk_rows': 16384, 'use_checkpoint': True, 'module_sha256': sha,
                                        'algorithm_version': 1}
    return module


def test_rl_segment_checks_the_pair_settings(world, monkeypatch):
    """(a) pairing.check_rl_settings is called by the RL segment, not only unit-tested."""
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(pairing.PairingError, match='chunk_identity'):
        T.run_once(world, 'rl', monkeypatch, memory=_memory_with_module('d' * 64))
    assert not S.retention.full_restarts(world.paths.run_dir('rl'))


def test_resume_refuses_a_rebuilt_initial_hypernetwork(world, monkeypatch):
    """(b) init_sha256 is a compared record field (same size, other bytes)."""
    paths = world.paths
    T.run_once(world, 'uniform', monkeypatch)
    model = paths.hypernet_init / 'model.safetensors'
    data = bytearray(model.read_bytes())
    data[0] ^= 1
    model.write_bytes(bytes(data))
    manifest = read_json(paths.input_manifest)
    if 'hypernet_init/model.safetensors' in manifest['files']:
        manifest['files']['hypernet_init/model.safetensors']['sha256'] = sha256_file(model)
        paths.input_manifest.write_text(json.dumps(manifest))
    with pytest.raises(S.SegmentError, match='init_sha256'):
        T.run_once(world, 'uniform', monkeypatch)


def test_resume_refuses_changed_fixed_paths(world, monkeypatch):
    """(b) fixed_paths is a compared record field."""
    T.run_once(world, 'uniform', monkeypatch)
    real = S.fixed_paths
    monkeypatch.setattr(S, 'fixed_paths', lambda paths, run, output: {**real(paths, run, output), 'model': '/elsewhere'})
    with pytest.raises(S.SegmentError, match='fixed_paths'):
        T.run_once(world, 'uniform', monkeypatch)


def test_rl_policy_counters_must_equal_the_step(world, monkeypatch):
    """(c) the launcher's RL post-check 'Policy did not resume with the model'."""
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(S.SegmentError, match='Policy did not resume with the model'):
        T.run_once(world, 'rl', monkeypatch, control={'stop_at': [3], 'bandit_extra_update': True})


def test_trainer_must_end_with_paused_at(world, monkeypatch):
    """(d) the trainer's last event must be paused_at/complete at the newest restart."""
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(S.SegmentError, match="trainer's last event"):
        T.run_once(world, 'uniform', monkeypatch, control={'stop_at': [4], 'no_final_event': True})


def test_segment_must_make_checkpoint_progress(world, monkeypatch):
    """(e) a resumed trainer that returns without saving is a failure."""
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(S.SegmentError, match='made no checkpoint progress'):
        T.run_once(world, 'uniform', monkeypatch, control={'return_without_save': True})


def test_stop_before_the_trainer_handlers_is_resent(world, monkeypatch):
    """(f) a stop that lands after the trainer started but before its handlers exist is re-sent."""
    T.run_once(world, 'uniform', monkeypatch)
    monkeypatch.setattr(S, 'MIN_MAX_SECONDS', 0)
    outcome = T.run_once(world, 'uniform', monkeypatch, max_seconds=0.2,
                         control={'sleep_before_handlers': 0.5, 'sleep': 0.01})
    assert outcome['outcome'] == 'paused' and 1 < outcome['step'] < 1000
