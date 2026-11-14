"""Regression tests of the review fix round for hyak.train.run_segment / retention / pairing (owner B).

Each test names the finding it covers.  All use the fake trainer of test_train_support.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from hyak.common.io import read_json, sha256_file
from hyak.train import pairing
from hyak.train import retention as R
from hyak.train import run_segment as S
from hyak.tests import test_train_support as T
from hyak.tests.test_train_retention import code_dir, make_checkpoint


@pytest.fixture
def world(make_tmp_config, repo_root):
    return T.build_world(make_tmp_config, repo_root)


def events(world, stage, name):
    return [e for e in T.monitor_events(world, stage) if e.get('event') == name]


def _edited_chunk_module(sha='d' * 64):
    """hyak/memory/chunked_hypernet.py edited mid-run: same rows, another module hash (chunk_identity changes)."""
    module = T.fake_memory_module()
    module.settings = lambda cls=None: {'chunk_rows': 16384, 'use_checkpoint': True, 'module_sha256': sha,
                                        'algorithm_version': 1}
    return module


def cli_world(world, monkeypatch, control=None):
    """Make ``S.main([...])`` run against the fake world (the CLI cannot take test parameters)."""
    from hyak.model import fetch_model
    world.control_file.write_text(json.dumps(control or {}))
    monkeypatch.setenv('HYAK_FAKE_CONTROL', str(world.control_file))
    monkeypatch.setattr('hyak.common.slurm.seconds_remaining', lambda default=None, now=None: None)
    monkeypatch.setitem(sys.modules, 'hyak.memory.chunked_hypernet', T.fake_memory_module())
    monkeypatch.setattr(S, 'expected_code_files', lambda run, repo=None: world.expected_code[run])
    monkeypatch.setattr(fetch_model, 'expected_files', lambda repo=None: world.model_expected)
    monkeypatch.setattr(S.run_segment, '__kwdefaults__', {**S.run_segment.__kwdefaults__,
                                                          'init_bytes': world.init_bytes, 'lock_wait_seconds': 0.2})


# --- train-lifecycle-1: tracebacks reach the job log ---------------------------------------------------------
def test_ordinary_trainer_exception_prints_the_traceback(world, monkeypatch, capsys):
    T.run_once(world, 'uniform', monkeypatch)
    cli_world(world, monkeypatch, control={'raise_at': 3})
    with T.isolated_imports():
        code = S.main(['--run', 'uniform'])
    err = capsys.readouterr().err
    assert code == 1
    assert 'Traceback (most recent call last)' in err and 'train_joint_smollm_input.py' in err
    assert err.strip().splitlines()[-1].startswith('ERROR: RuntimeError: ordinary trainer exception at update 3')
    failed = events(world, 'train-uniform', 'segment_failed')[-1]
    assert 'Traceback' in failed['traceback'] and failed['error'].startswith('RuntimeError')


def test_deliberate_refusals_use_the_contract_codes(world, monkeypatch, capsys):
    T.run_once(world, 'uniform', monkeypatch)
    cli_world(world, monkeypatch)
    monkeypatch.setitem(sys.modules, 'hyak.memory.chunked_hypernet', _edited_chunk_module())
    with T.isolated_imports():
        assert S.main(['--run', 'uniform']) == 3                   # changed chunk setting: the user's decision
    err = capsys.readouterr().err
    assert 'Traceback' not in err and '--accept-chunk-change' in err
    world.paths.input_manifest.unlink()
    with T.isolated_imports():
        assert S.main(['--run', 'uniform']) == 2                   # missing input: remedy named
    assert 'stage_inputs' in capsys.readouterr().err


# --- train-lifecycle-2 / train-state-1: stop during an RL weights-only save -------------------------------------
def test_rl_stop_during_a_weights_only_save_is_a_pause(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch)
    rl = world.paths.run_dir('rl')
    assert T.run_once(world, 'rl', monkeypatch, control={'stop_at': [2100]})['step'] == 2100
    outcome = T.run_once(world, 'rl', monkeypatch, control={'sigterm_in_save': 2500})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 2100
    assert T.trainer_events(rl)[-1] == {'stage': 'paused_at', 'step': 2500}
    warnings = events(world, 'train-rl', 'warning')
    assert warnings[-1]['paused_step'] == 2500 and warnings[-1]['restart_step'] == 2100
    assert not events(world, 'train-rl', 'segment_failed')
    # after a full restart was passed (4000), a stop inside the save of 4500 pauses at 4000
    outcome = T.run_once(world, 'rl', monkeypatch, control={'sigterm_in_save': 4500})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 4000
    moved = events(world, 'train-rl', 'moved_interrupted')
    assert moved[-1]['steps'] == [2500] and moved[-1]['restart_step'] == 2100
    # the next segment replays from 4000 and the grid stays exact
    assert T.run_once(world, 'rl', monkeypatch, control={'stop_at': [5100]})['step'] == 5100
    kinds = T.step_kinds(rl)
    assert kinds[4500] == R.WEIGHTS_ONLY and kinds[4000] == R.FULL and kinds[2500] == R.WEIGHTS_ONLY


def test_paused_at_without_a_checkpoint_still_fails(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch)
    rl = world.paths.run_dir('rl')
    T.run_once(world, 'rl', monkeypatch, control={'stop_at': [600]})
    with (rl / 'events.jsonl').open('a') as stream:          # paused_at 1000 but no step-001000 directory
        stream.write(json.dumps({'stage': 'paused_at', 'step': 1000}) + '\n')
    full = R.full_restarts(rl)
    with pytest.raises(S.SegmentError, match="trainer's last event"):
        S.post_checks('rl', rl, T.H200Adapter(world.paths.code_dir('rl')), start_step=None,
                      emit=S.Emitter('train-rl', 'rl', world.cfg))
    assert full[-1][0] == 600


def test_stop_timer_waits_for_a_weights_only_save():
    activity = R.SaveActivity()
    activity.begin(2500, False)
    sent = []
    timer = S.StopTimer(0.05, activity, poll=0.02, send=lambda: sent.append(time.monotonic())).start()
    time.sleep(0.3)
    assert not sent and timer.deferred_step == 2500           # deferred while the save runs
    activity.end()
    deadline = time.monotonic() + 2
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(sent) == 1 and timer.deferred_seconds >= 0.2 and timer.fired_at is not None
    full = R.SaveActivity()
    full.begin(2000, True)                                    # a full-restart save is no reason to wait
    sent.clear()
    S.StopTimer(0.01, full, poll=0.02, send=lambda: sent.append(1)).start()
    time.sleep(0.2)
    assert sent == [1]


def test_stop_timer_fires_before_slurms_signal_and_the_stop_is_measured(world, monkeypatch):
    assert S.stop_budget(3 * 3600, 45 * 60, lead=S.STOP_TIMER_LEAD_SECONDS) == 3 * 3600 - 45 * 60 - 300
    assert S.stop_budget(3 * 3600, 45 * 60, 600, lead=300) == 600       # --max-seconds is not shortened
    assert S.stop_budget(45 * 60 + 20 * 60 - 1, 45 * 60, lead=300) is None
    T.run_once(world, 'uniform', monkeypatch)
    monkeypatch.setattr(S, 'MIN_MAX_SECONDS', 0)
    T.run_once(world, 'uniform', monkeypatch, control={'sleep': 0.01}, max_seconds=0.4)
    end = events(world, 'train-uniform', 'segment_end')[-1]
    assert end['stop_source'] == 'timer' and isinstance(end['stop_seconds'], float) and end['stop_seconds'] < 5
    start = events(world, 'train-uniform', 'segment_start')[-1]
    assert set(start['env']) == set(S.RECORDED_ENV_VARS) and start['environment']['python']


def test_rl_weights_only_saves_report_their_activity(world, monkeypatch):
    seen = []
    real = R.rl_save_hook

    def spy(original, run_dir, code_dir, **kwargs):
        activity = kwargs['save_activity']

        def original_spy(*args, **kw):
            seen.append((args[1], activity.weights_only))
            return original(*args, **kw)
        return real(original_spy, run_dir, code_dir, **kwargs)

    monkeypatch.setattr(R, 'rl_save_hook', spy)
    T.run_once(world, 'uniform', monkeypatch)
    T.run_once(world, 'rl', monkeypatch, control={'stop_at': [1100]})
    assert seen == [(0, False), (500, True), (1000, True), (1100, False)]


# --- train-lifecycle-3: RL before the uniform bootstrap (see also test_train_segment) -------------------------
def test_rl_cli_before_the_bootstrap_exits_not_now(world, monkeypatch, capsys):
    cli_world(world, monkeypatch)
    with T.isolated_imports():
        assert S.main(['--run', 'rl']) == 75
    assert not (world.paths.monitor_dir('train-rl') / 'HALTED.json').exists()
    assert 'uniform_bootstrap_pending' in capsys.readouterr().out


# --- train-lifecycle-6 / train-state-5: the run lock is waited for -------------------------------------------
def _hold_lock(path: Path, seconds: float, ready: threading.Event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'pid': 'helper', 'job': 'predecessor'}) + '\n')
        handle.flush()
        ready.set()
        time.sleep(seconds)


def test_segment_waits_for_the_run_lock(world, monkeypatch):
    lock = R.wrapper_lock_path(world.paths, 'uniform')
    ready = threading.Event()
    holder = threading.Thread(target=_hold_lock, args=(lock, 1.0, ready))
    holder.start()
    ready.wait(5)
    monkeypatch.setattr(S, 'LOCK_POLL_SECONDS', 0.1)
    outcome = T.run_once(world, 'uniform', monkeypatch, lock_wait_seconds=10)
    holder.join()
    assert outcome['outcome'] == 'bootstrap_done'
    waiting = events(world, 'train-uniform', 'waiting_for_run_lock')
    assert waiting and waiting[0]['holder']['job'] == 'predecessor'
    assert events(world, 'train-uniform', 'run_lock_acquired')[0]['waited_seconds'] > 0.3


def test_segment_gives_up_on_a_held_lock_with_not_now(world, monkeypatch):
    lock = R.wrapper_lock_path(world.paths, 'uniform')
    ready = threading.Event()
    holder = threading.Thread(target=_hold_lock, args=(lock, 2.0, ready))
    holder.start()
    ready.wait(5)
    outcome = T.run_once(world, 'uniform', monkeypatch, lock_wait_seconds=0.3)
    holder.join()
    assert outcome['outcome'] == 'run_locked' and outcome['outcome'] in S.NOT_NOW_OUTCOMES
    assert not world.paths.run_dir('uniform').exists()
    assert not events(world, 'train-uniform', 'segment_failed')


# --- train-lifecycle-7 / corpus-7 / tests-10: test-scale corpus -------------------------------------------------
def _mark_test_scale(paths):
    manifest = read_json(paths.input_manifest)
    manifest['provenance'] = {'corpus': {'test_scale': True, 'shortfall_buckets': 23}}
    paths.input_manifest.write_text(json.dumps(manifest))


def test_test_scale_corpus_is_refused_without_the_test_parameter(world, monkeypatch):
    _mark_test_scale(world.paths)
    with pytest.raises(S.SegmentError, match='test-scale corpus') as exc:
        T.run_once(world, 'uniform', monkeypatch)
    assert exc.value.exit_code == 2
    with pytest.raises(S.SegmentError, match='test-scale corpus'):     # a smoke phase is no exception
        T.run_once(world, 'uniform', monkeypatch, smoke_phase='first')
    outcome = T.run_once(world, 'uniform', monkeypatch, allow_test_scale=True)
    assert outcome['outcome'] == 'bootstrap_done'
    assert read_json(world.paths.run_dir('uniform') / R.SEGMENT_RECORD)['corpus_test_scale'] is True


# --- train-lifecycle-8 / env-1 / readme-4: the model directory is verified ---------------------------------------
def test_model_files_are_verified_at_every_segment(world, monkeypatch):
    paths = world.paths
    with pytest.raises(S.SegmentError, match='is not the pinned model'):          # the fake files vs the real pins
        S.verify_inputs(paths, 'uniform', S.HashCache(paths.runs / 'c.json'), init_bytes=world.init_bytes)
    T.run_once(world, 'uniform', monkeypatch)
    record = read_json(paths.run_dir('uniform') / R.SEGMENT_RECORD)
    assert record['model_sha256'] == {name: entry['sha256'] for name, entry in world.model_expected.items()}
    tokenizer = paths.model_dir / 'tokenizer.json'
    original = tokenizer.read_bytes()
    tokenizer.write_bytes(original.upper())                                         # same size, other bytes
    with pytest.raises(S.SegmentError, match='tokenizer.json has SHA-256'):
        T.run_once(world, 'uniform', monkeypatch)
    tokenizer.write_bytes(original)
    (paths.model_dir / 'tokenizer_config.json').write_text('{}')
    with pytest.raises(S.SegmentError, match='unexpected entries'):
        T.run_once(world, 'uniform', monkeypatch)
    (paths.model_dir / 'tokenizer_config.json').unlink()
    assert read_json(paths.pairing)['model_sha256'] == record['model_sha256']


def test_rl_must_use_the_uniform_runs_model(world, monkeypatch):
    paths = world.paths
    T.run_once(world, 'uniform', monkeypatch)
    config = paths.model_dir / 'config.json'
    config.write_text(config.read_text().upper())
    other = T.model_hashes(paths)                    # an RL world whose pins would accept the changed file
    # without a new stage the input manifest (which lists model/*) already refuses the changed file
    with pytest.raises(S.SegmentError, match='Checksum mismatch: model/config.json'):
        T.run_once(world, 'rl', monkeypatch, model_expected=other)
    # ... and after a re-stage that records it, the pairing check still does
    manifest = read_json(paths.input_manifest)
    manifest['files']['model/config.json'] = other['config.json']
    paths.input_manifest.write_text(json.dumps(manifest))
    with pytest.raises(pairing.PairingError, match='model_sha256'):
        T.run_once(world, 'rl', monkeypatch, model_expected=other)


# --- chunk-change override (lead decision) --------------------------------------------------------------------------
def test_chunk_change_needs_the_explicit_recorded_override(world, monkeypatch):
    paths = world.paths
    run_dir = paths.run_dir('uniform')
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(S.SegmentError, match='--accept-chunk-change') as exc:
        T.run_once(world, 'uniform', monkeypatch, memory=_edited_chunk_module())
    assert exc.value.exit_code == 3
    outcome = T.run_once(world, 'uniform', monkeypatch, memory=_edited_chunk_module(), control={'stop_at': [3]},
                         accept_chunk_change='OOM at update 12,000 with 16384 rows')
    assert outcome['outcome'] == 'paused' and outcome['step'] == 3
    record = read_json(run_dir / R.SEGMENT_RECORD)
    assert record['chunk_identity']['module_sha256'] == 'd' * 64
    change = record['chunk_changes'][0]
    assert change['from']['module_sha256'] == 'c' * 64 and change['reason'].startswith('OOM') and change['step'] == 1
    assert events(world, 'train-uniform', 'chunk_change_accepted')[0]['to']['module_sha256'] == 'd' * 64
    # the new setting is now the run's setting: no flag needed any more, the old one is refused
    assert T.run_once(world, 'uniform', monkeypatch, memory=_edited_chunk_module(),
                      control={'stop_at': [4]})['step'] == 4
    with pytest.raises(S.SegmentError, match='chunk_identity'):
        T.run_once(world, 'uniform', monkeypatch)
    # the RL run against the frozen uniform record: refused, then accepted on the record
    with pytest.raises(pairing.PairingError, match='--accept-chunk-change'):
        T.run_once(world, 'rl', monkeypatch, memory=_edited_chunk_module(), control={'stop_at': [2]})
    assert T.run_once(world, 'rl', monkeypatch, memory=_edited_chunk_module(), control={'stop_at': [2]},
                      accept_chunk_change='same as the uniform run')['step'] == 2
    rl_record = read_json(paths.run_dir('rl') / R.SEGMENT_RECORD)
    assert 'chunk_identity' in rl_record['accepted_pairing_differences'][0]['difference']
    assert T.run_once(world, 'rl', monkeypatch, memory=_edited_chunk_module(), control={'stop_at': [3]})['step'] == 3


def test_cli_flag_reaches_the_segment(world, monkeypatch):
    seen = {}
    monkeypatch.setattr(S, 'run_segment', lambda run, **kw: seen.update(kw) or {'outcome': 'paused', 'step': 1})
    assert S.main(['--run', 'uniform', '--accept-chunk-change', 'oom']) == 0
    assert seen['accept_chunk_change'] == 'oom'
    with pytest.raises(SystemExit):
        S.main(['--run', 'uniform', '--accept-chunk-change', ' '])


# --- environment record (lead decision) -------------------------------------------------------------------------------
def test_environment_change_between_segments_is_a_loud_warning(world, monkeypatch):
    run_dir = world.paths.run_dir('uniform')
    T.run_once(world, 'uniform', monkeypatch)
    record = read_json(run_dir / R.SEGMENT_RECORD)
    assert set(S.ENVIRONMENT_FIELDS) <= set(record['environment'])

    class NewTorch(T.H200Adapter):
        def gpu_info(self):
            return {**super().gpu_info(), 'torch': '9.9.9'}

    assert T.run_once(world, 'uniform', monkeypatch, adapter=NewTorch, control={'stop_at': [3]})['step'] == 3
    changed = events(world, 'train-uniform', 'environment_changed')
    assert changed[0]['changes']['torch']['now'] == '9.9.9'
    assert read_json(run_dir / R.SEGMENT_RECORD)['environment_history'][-1]['torch'] == '9.9.9'
    T.run_once(world, 'uniform', monkeypatch, adapter=NewTorch, control={'stop_at': [4]})
    assert len(events(world, 'train-uniform', 'environment_changed')) == 1    # warned once per change


# --- train-lifecycle-v1: smoke phases warn when no stop timer exists under Slurm ---------------------------------
def test_smoke_phase_warns_about_a_missing_timer_only_under_slurm(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch, smoke_phase='first')
    assert not [w for w in events(world, 'smoke-train', 'warning') if 'no in-process stop timer' in w['message']]
    import shutil
    shutil.rmtree(world.paths.smoke_dir('uniform'))
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    T.run_once(world, 'uniform', monkeypatch, smoke_phase='first')
    assert [w for w in events(world, 'smoke-train', 'warning') if 'no in-process stop timer' in w['message']]


# --- monitoring-8: an earlier attempt's monitor files are rotated on a fresh start -------------------------------
def test_fresh_start_rotates_an_earlier_attempts_monitor_files(world, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '5009001')
    monitor = world.paths.monitor_dir('train-rl')
    monitor.mkdir(parents=True, exist_ok=True)
    (monitor / '5000999-train.log').write_text('{"stage": "uniform_bootstrap_pending"}\n')   # a 'not now' job
    assert T.run_once(world, 'rl', monkeypatch)['outcome'] == 'uniform_bootstrap_pending'
    for name in ('slurm-5001001.out', 'status.json'):
        (monitor / name).write_text('{}\n')
    (monitor / '5001001-train.log').write_text('{"stage": "initial_eval", "results": []}\n')
    (monitor / '5009001-train.log').write_text('this job\n')
    (monitor / 'slurm-5009001.out').write_text('this job\n')
    T.run_once(world, 'uniform', monkeypatch)
    T.run_once(world, 'rl', monkeypatch, control={'stop_at': [2]})
    previous = [p for p in monitor.iterdir() if p.name.startswith('previous-')]
    assert len(previous) == 1
    moved = sorted(p.name for p in previous[0].iterdir())
    assert moved == ['5000999-train.log', '5001001-train.log', 'events.jsonl', 'slurm-5001001.out', 'status.json']
    assert (monitor / '5009001-train.log').read_text() == 'this job\n' and (monitor / 'slurm-5009001.out').exists()
    assert events(world, 'train-rl', 'previous_attempt_rotated')
    # a resume rotates nothing
    T.run_once(world, 'rl', monkeypatch, control={'stop_at': [3]})
    assert len([p for p in monitor.iterdir() if p.name.startswith('previous-')]) == 1


# --- train-state-9: a stale receipt is retired when the run trains again ------------------------------------------
def test_final_audit_receipt_is_retired_before_training(world, monkeypatch):
    run_dir = world.paths.run_dir('uniform')
    T.run_once(world, 'uniform', monkeypatch)
    (run_dir / R.FINAL_AUDIT).write_text('{"final_step": 20000, "checkpoint_count": 40}\n')
    T.run_once(world, 'uniform', monkeypatch, control={'stop_at': [3]})
    assert not (run_dir / R.FINAL_AUDIT).exists() and list(run_dir.glob('final_audit.stale-*.json'))
    assert events(world, 'train-uniform', 'final_audit_retired')


def test_audit_receipt_names_the_final_manifest(tmp_path):
    from hyak.tests.test_train_retention import _finished_grid
    manifest = tmp_path / 'input.json'
    manifest.write_text('{}')
    run = tmp_path / 'uniform'
    _finished_grid(run, 'uniform')
    receipt = R.audit_final(run, 'uniform', manifest)
    assert receipt['final_manifest_sha256'] == sha256_file(run / 'step-020000' / 'manifest.json')
    assert R.audit_valid(run, 'uniform')
    (run / 'step-020000' / 'manifest.json').write_text((run / 'step-020000' / 'manifest.json').read_text() + ' ')
    assert not R.audit_valid(run, 'uniform')               # step 20,000 was replaced after the audit


def test_audit_checks_cheap_preconditions_before_hashing(tmp_path, monkeypatch):
    from hyak.tests.test_train_retention import _finished_grid
    run = tmp_path / 'uniform'
    _finished_grid(run, 'uniform')
    calls = []
    real = R.sha256_file
    monkeypatch.setattr(R, 'sha256_file', lambda path, *a: calls.append(path) or real(path, *a))
    with pytest.raises(R.CheckpointError, match='input manifest'):
        R.audit_final(run, 'uniform', tmp_path / 'absent.json')
    import shutil
    shutil.rmtree(run / 'step-012000')
    make_checkpoint(run, 19500, 'uniform', R.FULL)
    manifest = tmp_path / 'input.json'
    manifest.write_text('{}')
    with pytest.raises(R.CheckpointError, match=r'no manifest: \[12000\].*wrong kind: step-019500'):
        R.audit_final(run, 'uniform', manifest)
    assert calls == []


# --- train-state-2: the retention step of the last save is completed before the audit ----------------------------
def _replace_state(run_dir, state):
    import torch
    folder = run_dir / 'step-020000'
    torch.save(state, folder / 'state.pt')
    manifest = read_json(folder / 'manifest.json')
    manifest['files']['state.pt'] = {'bytes': (folder / 'state.pt').stat().st_size,
                                     'sha256': sha256_file(folder / 'state.pt')}
    (folder / 'manifest.json').write_text(json.dumps(manifest))


def test_final_retention_is_completed_before_the_audit(tmp_config):
    import torch
    from hyak.tests.test_train_retention import _finished_grid
    paths = tmp_config.paths
    paths.input_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.input_manifest.write_text('{"files": {}}\n')
    uniform = paths.run_dir('uniform')
    _finished_grid(uniform, 'uniform')
    make_checkpoint(uniform, 19500, 'uniform', R.FULL)            # killed before the last compaction
    _replace_state(uniform, {'step': 20000})
    names = []
    outcome = S._finish_complete('uniform', paths, uniform, S.TrainerAdapter(paths.code_dir('uniform')),
                                 lambda name, **f: names.append((name, f)), 20000, already=True)
    assert outcome['outcome'] == 'already_complete'
    assert ('finished_retention', {'step': 20000, 'compacted': [19500]}) in names
    assert R.verify_checkpoint(uniform, 19500, 'uniform', R.WEIGHTS_ONLY)
    rl = paths.run_dir('rl')
    _finished_grid(rl, 'rl')
    make_checkpoint(rl, 19300, 'rl', R.FULL)                      # killed before the last prune
    _replace_state(rl, {'step': 20000, 'bandit': {'updates': 20000, 'counts': torch.full((1000,), 20)}})
    outcome = S._finish_complete('rl', paths, rl, S.TrainerAdapter(paths.code_dir('rl')), lambda *a, **f: None,
                                 20000, already=True)
    assert outcome['final_audit']['extra_step_dirs'] == [] and not (rl / 'step-019300').exists()


# --- train-state-4: quarantine guard -------------------------------------------------------------------------------
def test_quarantine_refuses_older_steps_and_the_only_restart(tmp_config, capsys):
    run = tmp_config.paths.run_dir('uniform')
    make_checkpoint(run, 2000)
    make_checkpoint(run, 2600)
    assert R.main(['quarantine', '--run', 'uniform', '--step', '2000']) == 4
    assert 'below the newest full restart' in capsys.readouterr().err and (run / 'step-002000').is_dir()
    assert R.main(['quarantine', '--run', 'uniform', '--step', '2600']) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result['resume_step'] == 2000 and result['updates_given_up'] == 600 and result['undo'].startswith('mv ')
    assert 'resumes from step 2000' in out.err
    assert R.main(['quarantine', '--run', 'uniform', '--step', '2000']) == 4      # now the only full restart
    assert 'only full restart' in capsys.readouterr().err
    assert R.main(['quarantine', '--run', 'uniform', '--step', '2000', '--force']) == 0


# --- train-state-6: fsync before older restart state is removed ----------------------------------------------------
def test_new_checkpoint_is_synced_before_compaction(tmp_path, monkeypatch):
    code = code_dir(tmp_path)
    run = tmp_path / 'run'
    old = make_checkpoint(run, 500)
    order = []
    real_fsync, real_unlink = os.fsync, Path.unlink
    monkeypatch.setattr(os, 'fsync', lambda fd: order.append('fsync') or real_fsync(fd))
    monkeypatch.setattr(Path, 'unlink', lambda self, *a, **k: order.append(f'unlink {self.name}') or
                        real_unlink(self, *a, **k))

    def save(output, step, *args, **kwargs):
        make_checkpoint(Path(output), step, manifest=False)

    hook = R.uniform_save_hook(save, run, code)
    hook(run, 1000)
    assert not (old / 'state.pt').exists()
    first_unlink = next(i for i, item in enumerate(order) if item.startswith('unlink'))
    assert order[:first_unlink].count('fsync') >= len(R.full_files('uniform'))
    # an fsync error keeps the older restart
    old2 = make_checkpoint(run, 1500)
    monkeypatch.setattr(os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError(5, 'I/O error')))
    with pytest.raises(OSError):
        hook(run, 2500)
    assert (old2 / 'state.pt').exists()


# --- auto-adopt and train-state-8 / train-state-v2: pre-flight remedies ------------------------------------------
def test_preflight_adopts_a_complete_restart_and_refuses_a_remainder(tmp_path):
    code = code_dir(tmp_path)
    run = tmp_path / 'run'
    make_checkpoint(run, 2000)
    newest = make_checkpoint(run, 2600, manifest=False)
    with pytest.raises(R.ManifestMissing, match='adopt --run uniform --step 2600'):
        R.preflight(run, 'uniform', adopt_code_dir=code)             # no hyak_segment.json: not adoptable
    (run / R.SEGMENT_RECORD).write_text('{}')
    (newest / '.manifest.json.abc123.tmp').write_text('{')          # a kill inside the atomic manifest write
    result = R.preflight(run, 'uniform', adopt_code_dir=code)
    assert result.adopted == 2600 and result.newest_full_step == 2600 and result.newest_full_bytes > 0
    assert result.removed_manifest_tmp == ['step-002600/.manifest.json.abc123.tmp']
    R.verify_checkpoint(run, 2600, 'uniform', R.FULL)
    # the remainder of an interrupted deletion: only quarantine is named, never adopt
    half = make_checkpoint(run, 2700, manifest=False)
    for name in ('hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt'):
        (half / name).unlink()
    with pytest.raises(R.ManifestMissing, match='remainder of an interrupted deletion') as exc:
        R.preflight(run, 'uniform', adopt_code_dir=code)
    assert 'quarantine --run uniform --step 2700' in str(exc.value) and 'retention adopt' not in str(exc.value)


def test_preflight_never_names_quarantine_for_the_only_restart(tmp_path):
    for run_name in ('uniform', 'rl'):
        run = tmp_path / run_name
        only = make_checkpoint(run, 700, run_name)
        (only / 'state.pt').write_bytes(b'corrupt')
        with pytest.raises(R.CheckpointError) as exc:
            R.preflight(run, run_name, remedy_suffix=' --smoke')
        text = str(exc.value)
        assert 'quarantine --run' not in text and 'only full restart' in text
        assert f'summary --run {run_name} --smoke' in text


def test_adopt_hashes_once_and_ignores_a_stale_manifest_temporary(tmp_path, monkeypatch):
    from hyak.common import io
    code = code_dir(tmp_path)
    run = tmp_path / 'run'
    folder = make_checkpoint(run, 2600, manifest=False)
    (folder / '.manifest.json.zz.tmp').write_text('{')
    (run / R.SEGMENT_RECORD).write_text('{}')
    calls = []
    real = io.sha256_file
    monkeypatch.setattr(io, 'sha256_file', lambda path, *a: calls.append(Path(path).name) or real(path, *a))
    R.adopt(run, 'uniform', 2600, code)
    assert not list(folder.glob('.manifest.json.*.tmp'))
    for name in ('lm.pt', 'optimizer.pt', 'state.pt', 'model.safetensors', 'config.json'):
        assert calls.count(name) == 1, (name, calls)


# --- train-lifecycle-9: real signals at the fake trainer's other injection points ---------------------------------
def test_real_signal_after_the_initial_evaluation_and_during_an_evaluation(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch)
    rl = world.paths.run_dir('rl')
    outcome = T.run_once(world, 'rl', monkeypatch, control={'sigterm_after_initial_eval': True})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 1          # one update, then a full restart
    assert T.step_kinds(rl) == {1: R.FULL}                                   # step 0 pruned after it
    end = events(world, 'train-rl', 'segment_end')[-1]
    assert end['stop_source'] == 'external'
    outcome = T.run_once(world, 'rl', monkeypatch, control={'sigterm_in_eval': 500})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 500         # an evaluation precedes the save
    assert T.step_kinds(rl)[500] == R.FULL
