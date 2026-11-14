"""hyak.train.run_segment: argv, timer arithmetic, code/input checks, modes, refusals, timer, smoke."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys

import pytest

from hyak.common.io import read_json, sha256_file
from hyak.train import pairing
from hyak.train import retention as R
from hyak.train import run_segment as S
from hyak.tests import test_train_support as T

PROTECTED = [str(s) for s in range(500, 20001, 500)]


@pytest.fixture
def world(make_tmp_config, repo_root):
    return T.build_world(make_tmp_config, repo_root)


def segment_starts(world, stage):
    return [e for e in T.monitor_events(world, stage) if e.get('event') == 'segment_start']


# --- argv (modal_anytok_pool_20k.py:321-337, modal_anytok_pool_rl_20k.py:248-257) ----------------------
def expected_uniform(root, code, out, tail):
    return [f'{code}/scripts/train_joint_smollm_input.py',
            '--recipe', f'{root}/input/recipe.json', '--data-dir', f'{root}/input/corpus',
            '--data-manifest', f'{root}/input/corpus/manifest.json', '--eval-dir', f'{root}/input/eval',
            '--output', out, '--warmup-checkpoint', f'{root}/input/hypernet_init',
            '--model', f'{root}/input/model', '--tokenizer-pool', f'{root}/input/pool',
            '--microbatch-size', '32', '--save-every', '500', '--protected-steps', *PROTECTED,
            '--eval-every', '500', *tail]


def expected_rl(root, code, out, tail):
    return [f'{code}/scripts/train_joint_smollm_input.py',
            '--recipe', f'{root}/input/recipe.json', '--data-dir', f'{root}/input/corpus',
            '--data-manifest', f'{root}/input/corpus/manifest.json', '--eval-dir', f'{root}/input/eval',
            '--output', out, '--warmup-checkpoint', f'{root}/input/hypernet_init',
            '--model', f'{root}/input/model', '--tokenizer-pool', f'{root}/input/pool',
            '--eval-tokenizers', f'{root}/input/uniform_heldout_tokenizers.json',
            '--microbatch-size', '32', '--save-every', '500', '--eval-every', '500',
            # the joint regret launcher's explicit policy flags (entropy 0, D8)
            '--policy-entropy', '0', '--policy-lr', '0.001', '--policy-baseline-decay', '0.99',
            '--stop-at-update', '20000', '--seed', '42', '--protected-steps', *PROTECTED, *tail]


def test_argv_exact_strings(tmp_config):
    paths = tmp_config.paths
    root = str(paths.root)
    u_code, r_code = str(paths.code_dir('uniform')), str(paths.code_dir('rl'))
    u_out, r_out = str(paths.run_dir('uniform')), str(paths.run_dir('rl'))
    plan = S.plan_segment('uniform', paths, paths.code_dir('uniform'), paths.run_dir('uniform'), None)
    assert plan.bootstrap and plan.target_step == 1
    assert plan.argv == expected_uniform(root, u_code, u_out, ['--max-updates-this-run', '1', '--seed', '42'])
    plan = S.plan_segment('uniform', paths, paths.code_dir('uniform'), paths.run_dir('uniform'), 1)
    assert plan.argv == expected_uniform(root, u_code, u_out, ['--stop-at-update', '20000', '--seed', '42', '--resume'])
    plan = S.plan_segment('rl', paths, paths.code_dir('rl'), paths.run_dir('rl'), None)
    assert plan.argv == expected_rl(root, r_code, r_out, [])
    plan = S.plan_segment('rl', paths, paths.code_dir('rl'), paths.run_dir('rl'), 2000)
    assert plan.argv == expected_rl(root, r_code, r_out, ['--resume'])
    smoke = paths.smoke_dir('rl')
    first = S.plan_segment('rl', paths, paths.code_dir('rl'), smoke, None, 'first')
    assert '--protected-steps' not in first.argv and first.argv[first.argv.index('--stop-at-update') + 1] == '2'
    second = S.plan_segment('uniform', paths, paths.code_dir('uniform'), paths.smoke_dir('uniform'), 1, 'second')
    assert second.argv[-5:] == ['--stop-at-update', '3', '--seed', '42', '--resume']
    with pytest.raises(S.SegmentError):
        S.plan_segment('rl', paths, paths.code_dir('rl'), smoke, 1, 'second')


def test_stop_budget_arithmetic():
    margin = 45 * 60
    assert S.stop_budget(None, margin) == math.inf
    assert S.stop_budget(math.inf, margin, 600) == 600
    assert S.stop_budget(24 * 3600, margin) == 24 * 3600 - margin
    assert S.stop_budget(24 * 3600, margin, 3600) == 3600
    assert S.stop_budget(margin + 20 * 60, margin) == 20 * 60
    assert S.stop_budget(margin + 20 * 60 - 1, margin) is None
    assert S.stop_budget(30 * 60, margin) is None


# --- code and input verification ------------------------------------------------------------------------
def stage_real_tree(repo_root: Path, code: Path, run: str):
    """What hyak.code.stage_code builds: real copies of the pinned files."""
    base = repo_root / 'runtime/anytok_pool_20k'
    for path in sorted((base / 'hf_hypernet').glob('*.py')):
        (code / 'hf_hypernet').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, code / 'hf_hypernet' / path.name)
    (code / 'scripts').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_root / S.TRAINER_SOURCE[run], code / S.TRAINER_REL)
    if run == 'rl':
        shutil.copyfile(repo_root / 'runtime/anytok_pool_settings_rl_20k' / S.POLICY_REL, code / S.POLICY_REL)
        shutil.copyfile(repo_root / S.PAIRED_REWARD_SOURCE, code / S.PAIRED_REWARD_REL)
    (code / S.SO_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / S.SO_REL, code / S.SO_REL)


def test_verify_code_against_the_pinned_snapshots(tmp_path, repo_root):
    manifest = tmp_path / 'code_manifest.json'
    manifest.write_text('{"kind": "hyak_code_manifest"}\n')
    for run in ('uniform', 'rl'):
        code = tmp_path / run
        stage_real_tree(repo_root, code, run)
        (code / 'hf_hypernet' / '__pycache__').mkdir()
        (code / 'hf_hypernet' / '__pycache__' / 'x.cpython-312.pyc').write_bytes(b'ignored')
        info = S.verify_code(code, run, manifest)
        assert info['trainer_sha256'] == S.TRAINER_SHA256[run] and not info['rust_utils_rebuilt']
        assert (info['policy_sha256'] == S.POLICY_SHA256) == (run == 'rl')
        assert (info['hf_hypernet_sha256'].get(S.PAIRED_REWARD_REL) == S.PAIRED_REWARD_SHA256) == (run == 'rl')
    # the RL tree runs the straight-reward variant, never the original RL or the regret trainer
    assert S.TRAINER_SOURCE['rl'] == 'hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py'
    assert S.expected_code_files('rl', repo_root)[S.TRAINER_REL] == S.TRAINER_SHA256['rl']
    assert S.expected_code_files('rl', repo_root)[S.PAIRED_REWARD_REL] == S.PAIRED_REWARD_SHA256
    assert S.PAIRED_REWARD_REL not in S.expected_code_files('uniform', repo_root)
    code = tmp_path / 'rl'
    so = code / S.SO_REL
    original = so.read_bytes()
    so.write_bytes(b'rebuilt library')
    with pytest.raises(S.SegmentError, match='neither the committed binary'):
        S.verify_code(code, 'rl', manifest)
    # hyak.code.stage_code records a rebuild as trees/<run>/files/<rel>
    manifest.write_text(json.dumps({'trees': {'rl': {'files': {S.SO_REL: {'sha256': sha256_file(so)}}},
                                              'uniform': {'files': {}}}}))
    assert S.verify_code(code, 'rl', manifest)['rust_utils_rebuilt']
    so.write_bytes(original)
    trainer = code / S.TRAINER_REL
    trainer.write_text(trainer.read_text() + '\n# edited\n')
    with pytest.raises(S.SegmentError, match='differ from the pinned snapshots'):
        S.verify_code(code, 'rl', manifest)
    shutil.copyfile(repo_root / 'runtime/anytok_pool_regret_rl_20k' / S.TRAINER_REL, trainer)   # the regret trainer
    with pytest.raises(S.SegmentError, match='differ from the pinned snapshots'):
        S.verify_code(code, 'rl', manifest)
    shutil.copyfile(repo_root / S.TRAINER_SOURCE['rl'], trainer)
    (code / 'hf_hypernet' / 'extra.py').write_text('')
    with pytest.raises(S.SegmentError, match="unexpected \\['hf_hypernet/extra.py'\\]"):
        S.verify_code(code, 'rl', manifest)
    (code / 'hf_hypernet' / 'extra.py').unlink()
    trainer.unlink()
    trainer.symlink_to(repo_root / S.TRAINER_SOURCE['rl'])
    with pytest.raises(S.SegmentError, match='symlink'):
        S.verify_code(code, 'rl', manifest)
    with pytest.raises(S.SegmentError, match='is missing'):
        S.verify_code(tmp_path / 'absent', 'rl', manifest)


def test_verify_inputs_hashes_large_files_once(world, monkeypatch):
    paths = world.paths
    monkeypatch.setattr(S, 'LARGE_FILE_BYTES', 1)       # every input counts as large here
    calls = []
    real = S.sha256_file
    monkeypatch.setattr(S, 'sha256_file', lambda path, *a: calls.append(str(path)) or real(path, *a))
    cache_file = paths.runs / '.hyak-cache' / 'test.json'
    result = S.verify_inputs(paths, 'uniform', S.HashCache(cache_file), init_bytes=world.init_bytes,
                           model_expected=world.model_expected)
    assert result['file_count'] == len(read_json(paths.input_manifest)['files'])
    first = len(calls)
    calls.clear()
    S.verify_inputs(paths, 'uniform', S.HashCache(cache_file), init_bytes=world.init_bytes,
                           model_expected=world.model_expected)
    assert len(calls) < first / 4                       # only the always-hashed recipe/plan/manifest
    train = paths.input_corpus_train('web-en')
    data = bytearray(train.read_bytes())
    data[-5] ^= 1
    train.write_bytes(bytes(data))                      # same size, new mtime: re-hashed and caught
    with pytest.raises(S.SegmentError, match='Checksum mismatch: corpus/web-en/train.jsonl.gz'):
        S.verify_inputs(paths, 'uniform', S.HashCache(cache_file), init_bytes=world.init_bytes,
                           model_expected=world.model_expected)


def test_verify_inputs_hashes_each_model_file_once(world, monkeypatch):
    """A production manifest lists model/*; verify_model's digests serve it (no second read of the weights)."""
    paths = world.paths
    listed = read_json(paths.input_manifest)['files']
    assert set(T.STAGED_MODEL_FILES) <= set(listed)
    monkeypatch.setattr(S, 'LARGE_FILE_BYTES', 1)
    calls = []
    real = S.sha256_file
    monkeypatch.setattr(S, 'sha256_file', lambda path, *a: calls.append(os.path.realpath(path)) or real(path, *a))
    S.verify_inputs(paths, 'uniform', S.HashCache(paths.runs / 'c.json'), init_bytes=world.init_bytes,
                    model_expected=world.model_expected)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        assert calls.count(os.path.realpath(paths.model_dir / name)) == 1, name
    # a model file that changed after staging is still caught by the manifest check (same size, other bytes,
    # and the pins of this test follow the change, so only the manifest can catch it)
    tokenizer = paths.model_dir / 'tokenizer.json'
    tokenizer.write_text(tokenizer.read_text().upper())
    with pytest.raises(S.SegmentError, match='Checksum mismatch: model/tokenizer.json'):
        S.verify_inputs(paths, 'uniform', S.HashCache(paths.runs / 'c.json'), init_bytes=world.init_bytes,
                        model_expected=T.model_hashes(paths))


def test_init_recipe_is_read_from_a_matching_init_manifest(world):
    paths = world.paths
    cache = S.HashCache(paths.runs / 'c.json')
    kwargs = dict(init_bytes=world.init_bytes, model_expected=world.model_expected)
    result = S.verify_inputs(paths, 'uniform', cache, **kwargs)
    assert result['init_recipe'] == {key: T.FAKE_INIT_RECIPE[key] for key in S.INIT_RECIPE_KEYS}
    assert result['init_recipe_warning'] is None
    assert 'init_recipe' not in S.COMPARED_RECORD_FIELDS
    # wrong hash: None and a warning, never an exception
    manifest = read_json(paths.hypernet_init_manifest)
    manifest['outputs']['model.safetensors']['sha256'] = '0' * 64
    paths.hypernet_init_manifest.write_text(json.dumps(manifest))
    result = S.verify_inputs(paths, 'uniform', cache, **kwargs)
    assert result['init_recipe'] is None and 'model.safetensors' in result['init_recipe_warning']
    # unreadable manifest
    paths.hypernet_init_manifest.write_text('{not json')
    result = S.verify_inputs(paths, 'uniform', cache, **kwargs)
    assert result['init_recipe'] is None and 'could not be read' in result['init_recipe_warning']
    # no manifest
    paths.hypernet_init_manifest.unlink()
    result = S.verify_inputs(paths, 'uniform', cache, **kwargs)
    assert result['init_recipe'] is None and 'missing' in result['init_recipe_warning']


def test_init_recipe_reaches_the_segment_record_and_pairing(world, monkeypatch):
    paths = world.paths
    T.run_once(world, 'uniform', monkeypatch)
    expected = {key: T.FAKE_INIT_RECIPE[key] for key in S.INIT_RECIPE_KEYS}
    assert read_json(paths.run_dir('uniform') / R.SEGMENT_RECORD)['init_recipe'] == expected
    assert read_json(paths.pairing)['init_recipe'] == expected
    assert not [e for e in T.monitor_events(world, 'train-uniform')
                if e.get('event') == 'warning' and 'init_manifest' in e.get('message', '')]
    # a later segment without a matching manifest warns once and still runs (information only)
    paths.hypernet_init_manifest.unlink()
    T.run_once(world, 'uniform', monkeypatch, control={'stop_at': [1000]})
    warnings = [e for e in T.monitor_events(world, 'train-uniform')
                if e.get('event') == 'warning' and 'init_manifest' in e.get('message', '')]
    assert len(warnings) == 1 and warnings[0]['init_recipe'] is None
    assert read_json(paths.run_dir('uniform') / R.SEGMENT_RECORD)['init_recipe'] == expected


def test_verify_inputs_refuses_wrong_inputs(world):
    paths = world.paths
    cache = S.HashCache(paths.runs / 'c.json')
    with pytest.raises(S.SegmentError, match='expected 1376746800'):
        S.verify_inputs(paths, 'uniform', cache, model_expected=world.model_expected)
    results = read_json(paths.pool_results)
    results['completed'] = 999
    paths.pool_results.write_text(json.dumps(results))
    with pytest.raises(S.SegmentError, match='complete 1000-tokenizer pool'):
        S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes, model_expected=world.model_expected)
    paths.recipe.write_text(paths.recipe.read_text() + ' ')
    with pytest.raises(S.SegmentError, match='balanced-full-v3'):
        S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes, model_expected=world.model_expected)
    paths.input_manifest.unlink()
    with pytest.raises(S.SegmentError, match='input manifest'):
        S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes, model_expected=world.model_expected)


def test_test_plan_needs_the_explicit_test_parameter(world):
    """A miniature world's TEST plan is refused unless its SHA-256 is passed as the test parameter (e2e harness)."""
    paths = world.paths
    plan = read_json(paths.pool_plan)
    plan['distribution']['hyak_e2e_test'] = 'TEST plan'
    paths.pool_plan.write_text(json.dumps(plan))
    manifest = read_json(paths.input_manifest)
    manifest['files']['pool/plan.json'] = {'bytes': paths.pool_plan.stat().st_size, 'sha256': sha256_file(paths.pool_plan)}
    paths.input_manifest.write_text(json.dumps(manifest))
    cache = S.HashCache(paths.runs / 'c.json')
    with pytest.raises(S.SegmentError, match='is not plans/tokenizer-pool-1000-v1.json'):
        S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes, model_expected=world.model_expected)
    result = S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes,
                             plan_file_sha256=sha256_file(paths.pool_plan), model_expected=world.model_expected)
    assert result['file_count'] == len(manifest['files'])


# --- segments with the fake trainer -------------------------------------------------------------------------
def test_uniform_bootstrap_then_resume(world, monkeypatch):
    paths = world.paths
    run_dir = paths.run_dir('uniform')
    outcome = T.run_once(world, 'uniform', monkeypatch)
    assert outcome == {'outcome': 'bootstrap_done', 'step': 1, 'elapsed_seconds': outcome['elapsed_seconds'],
                       'start_step': None}
    assert [s for s, _ in R.step_dirs(run_dir)] == [1]
    R.verify_checkpoint(run_dir, 1, 'uniform', R.FULL)
    record = read_json(run_dir / R.SEGMENT_RECORD)
    assert record['chunk_identity'] == {'chunk_rows': 16384, 'use_checkpoint': True, 'module_sha256': 'c' * 64,
                                        'algorithm_version': 1}
    assert record['fixed_paths']['model'] == str(paths.model_dir)
    assert pairing.is_frozen(paths)                       # frozen right after the bootstrap (D6)
    assert read_json(paths.pairing)['uniform_full_restart_step_at_freeze'] == 1
    root, code = str(paths.root), str(paths.code_dir('uniform'))
    start = segment_starts(world, 'train-uniform')[-1]
    assert start['argv'] == expected_uniform(root, code, str(run_dir), ['--max-updates-this-run', '1', '--seed', '42'])
    assert start['chunk_rows'] == 16384 and start['bootstrap'] is True
    outcome = T.run_once(world, 'uniform', monkeypatch, control={'stop_at': [3]}, remaining=3 * 3600)
    assert outcome['outcome'] == 'paused' and outcome['step'] == 3 and outcome['start_step'] == 1
    start = segment_starts(world, 'train-uniform')[-1]
    assert start['argv'] == expected_uniform(root, code, str(run_dir),
                                             ['--stop-at-update', '20000', '--seed', '42', '--resume'])
    assert start['stop_timer_seconds'] == 3 * 3600 - 45 * 60 - S.STOP_TIMER_LEAD_SECONDS   # before Slurm's signal
    assert start['stop_timer_lead_seconds'] == 300
    # the trainer's own pruning keeps one off-grid restart after a save (prune_checkpoints, disposable[:-1])
    assert [s for s, _ in R.step_dirs(run_dir)] == [3]
    names = [e['event'] for e in T.monitor_events(world, 'train-uniform')]
    for name in ('verified_code', 'verified_inputs', 'verified_restart', 'committed_checkpoint', 'segment_end'):
        assert name in names


def test_resume_refuses_changed_settings(world, monkeypatch, make_tmp_config):
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(S.SegmentError, match='chunk_identity'):
        T.run_once(world, 'uniform', monkeypatch, memory=_memory_with_rows(8192))
    code = world.paths.code_dir('uniform')
    (code / 'hf_hypernet' / 'joint_training.py').write_text('# changed\n')
    world.expected_code['uniform']['hf_hypernet/joint_training.py'] = sha256_file(code / 'hf_hypernet/joint_training.py')
    with pytest.raises(S.SegmentError, match='code_tree_sha256'):
        T.run_once(world, 'uniform', monkeypatch)


def _memory_with_rows(rows):
    module = T.fake_memory_module()
    install = module.install
    module.install = lambda cls, chunk_rows: install(cls, rows)
    module.settings = lambda cls=None: {'chunk_rows': 16384, 'use_checkpoint': True, 'module_sha256': 'd' * 64,
                                        'algorithm_version': 1}
    return module


def test_dirty_directories(world, monkeypatch):
    paths = world.paths
    uniform = paths.run_dir('uniform')
    uniform.mkdir(parents=True)
    (uniform / 'run_config.json').write_text('{}')          # bootstrap died before its first save (OOM)
    (uniform / '.run.lock').write_text('')
    (uniform / '.writing-x').mkdir()
    assert T.run_once(world, 'uniform', monkeypatch)['outcome'] == 'bootstrap_done'
    cleared = [e for e in T.monitor_events(world, 'train-uniform') if e.get('event') == 'cleared_run_directory'][0]
    assert cleared['deleted'] == ['.writing-x'] and sorted(cleared['moved']) == ['.run.lock', 'run_config.json']
    assert (Path(cleared['archive']) / 'run_config.json').is_file()
    rl = paths.run_dir('rl')
    # the RL directory is not cleared before the pair exists (pairing refuses first), but after it:
    pairing.ensure(paths)
    rl.mkdir()
    (rl / 'events.jsonl').write_text('{"stage": "initial_eval"}\n')
    outcome = T.run_once(world, 'rl', monkeypatch, control={'stop_at': [2]})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 2
    (rl / 'step-000000').mkdir(exist_ok=True)
    for path in rl.glob('step-*'):
        (path / 'state.pt').unlink(missing_ok=True)
    with pytest.raises(S.SegmentError, match='neither resume nor start fresh'):
        T.run_once(world, 'rl', monkeypatch)


def test_uniform_never_restarts_from_scratch_once_frozen(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch)
    shutil.rmtree(world.paths.run_dir('uniform'))
    with pytest.raises(pairing.PairingError, match='wiped after the pair was frozen'):
        T.run_once(world, 'uniform', monkeypatch)
    world.paths.run_dir('uniform').mkdir()
    for name in ('run_config.json', 'heldout_tokenizers.json'):
        source = world.paths.uniform_run_config if name == 'run_config.json' else world.paths.uniform_heldout
        shutil.copyfile(source, world.paths.run_dir('uniform') / name)
    with pytest.raises(S.SegmentError, match='pair is frozen'):
        T.run_once(world, 'uniform', monkeypatch)


def test_rl_needs_the_uniform_bootstrap(world, monkeypatch, capsys):
    """RL before the uniform bootstrap froze the pair: 'not now' (exit 75, no failure), never a halt."""
    outcome = T.run_once(world, 'rl', monkeypatch)
    assert outcome['outcome'] == 'uniform_bootstrap_pending' and 'run the uniform bootstrap first' in outcome['reason']
    assert not world.paths.run_dir('rl').exists()
    assert not [e for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'segment_failed']
    monkeypatch.setattr('hyak.common.slurm.seconds_remaining', lambda default=None, now=None: 86400.0)
    capsys.readouterr()
    answer = S.should_start('rl')
    assert (answer['start'], answer['reason']) == (False, 'uniform_bootstrap_pending')
    assert S.main(['--run', 'rl', '--should-start']) == 75
    assert json.loads(capsys.readouterr().out)['reason'] == 'uniform_bootstrap_pending'
    T.run_once(world, 'uniform', monkeypatch)                          # bootstrap: the pair is frozen
    assert S.should_start('rl')['start'] is True


def test_insufficient_time_starts_nothing(world, monkeypatch):
    outcome = T.run_once(world, 'uniform', monkeypatch, remaining=45 * 60 + 19 * 60)
    assert outcome == {'outcome': 'insufficient_time', 'step': None}
    assert not world.paths.run_dir('uniform').exists()
    assert 'scripts.train_joint_smollm_input' not in sys.modules
    monkeypatch.setattr('hyak.common.slurm.seconds_remaining', lambda default=None, now=None: 600.0)
    assert S.should_start('uniform') == {'run': 'uniform', 'newest_full_step': None, 'start': False,
                                         'reason': 'insufficient_time', 'seconds_remaining': 600.0,
                                         'margin_seconds': 2700, 'minimum_seconds': 1200}
    monkeypatch.setattr('hyak.common.slurm.seconds_remaining', lambda default=None, now=None: 86400.0)
    assert S.should_start('uniform')['start'] is True
    assert S.main(['--run', 'uniform', '--should-start']) == 0


def test_stop_timer_stops_the_trainer_gracefully(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch)
    monkeypatch.setattr(S, 'MIN_MAX_SECONDS', 0)
    outcome = T.run_once(world, 'uniform', monkeypatch, control={'sleep': 0.01}, max_seconds=0.4)
    assert outcome['outcome'] == 'paused' and 1 < outcome['step'] < 1000
    events = T.trainer_events(world.paths.run_dir('uniform'))
    assert events[-1] == {'stage': 'paused_at', 'step': outcome['step']}
    assert {'stage': 'eval', 'step': outcome['step'], 'results': events[0]['results']} in events
    assert R.full_restarts(world.paths.run_dir('uniform'))[-1][0] == outcome['step']


def test_signal_before_the_trainer_starts(world, monkeypatch):
    real = S.verify_inputs

    def verify_then_signal(*args, **kwargs):
        result = real(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(S, 'verify_inputs', verify_then_signal)
    outcome = T.run_once(world, 'uniform', monkeypatch)
    assert outcome == {'outcome': 'stopped_before_start', 'step': None}
    assert not world.paths.run_dir('uniform').exists()


class SignalWhileImporting(T.H200Adapter):
    """Slurm's TERM arriving while the trainer module is imported (seconds with the real trainer)."""

    def load(self):
        module = super().load()
        os.kill(os.getpid(), signal.SIGTERM)
        return module


def test_signal_while_the_trainer_is_imported(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch)                                       # bootstrap (step 1)
    before = T.trainer_events(world.paths.run_dir('uniform'))
    outcome = T.run_once(world, 'uniform', monkeypatch, adapter=SignalWhileImporting, control={'stop_at': [5]})
    assert outcome == {'outcome': 'stopped_before_start', 'step': 1}
    assert T.trainer_events(world.paths.run_dir('uniform')) == before                # the trainer never ran
    assert [e['signal'] for e in T.monitor_events(world, 'train-uniform')
            if e.get('event') == 'stopped_before_start'] == [signal.SIGTERM]


def test_trainer_failures_are_loud(world, monkeypatch, capsys):
    with pytest.raises(S.SegmentError, match='exited with status 2'):
        T.run_once(world, 'uniform', monkeypatch, control={'parser_error': 'A CUDA GPU is required'})
    assert any(e.get('event') == 'segment_failed' for e in T.monitor_events(world, 'train-uniform'))
    monkeypatch.setitem(sys.modules, 'hyak.memory.chunked_hypernet', None)
    with pytest.raises(S.SegmentError, match='needs hyak.memory.chunked_hypernet'):
        with T.isolated_imports():
            S.run_segment('uniform', adapter_factory=T.H200Adapter, expected_code=world.expected_code['uniform'],
                          init_bytes=world.init_bytes, model_expected=world.model_expected)
    with pytest.raises(ValueError, match='at least 60'):
        S.run_segment('uniform', max_seconds=5)


def test_gpu_other_than_h200_warns(world, monkeypatch):
    T.run_once(world, 'uniform', monkeypatch, adapter=S.TrainerAdapter)
    warnings = [e for e in T.monitor_events(world, 'train-uniform') if e.get('event') == 'warning']
    assert any('not an H200' in w['message'] for w in warnings)


def test_hard_kill_during_bootstrap_save_is_cleared(world, monkeypatch):
    with pytest.raises(T.FakeHardKill):
        T.run_once(world, 'uniform', monkeypatch, control={'crash': {'step': 1, 'where': 'in_write'}})
    run_dir = world.paths.run_dir('uniform')
    assert list(run_dir.glob('.writing-*')) and not R.full_restarts(run_dir)
    assert not pairing.is_frozen(world.paths)
    assert T.run_once(world, 'uniform', monkeypatch)['outcome'] == 'bootstrap_done'


def test_first_save_without_manifest_restarts_the_run(world, monkeypatch):
    with pytest.raises(T.FakeHardKill):
        T.run_once(world, 'uniform', monkeypatch, control={'crash': {'step': 1, 'where': 'after_replace'}})
    assert [(s, (p / 'manifest.json').exists()) for s, p in R.step_dirs(world.paths.run_dir('uniform'))] == [(1, False)]
    assert T.run_once(world, 'uniform', monkeypatch)['outcome'] == 'bootstrap_done'
    with pytest.raises(T.FakeHardKill):
        T.run_once(world, 'rl', monkeypatch, control={'crash': {'step': 0, 'where': 'after_replace'}})
    outcome = T.run_once(world, 'rl', monkeypatch, control={'stop_at': [2]})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 2
    reasons = [e['reason'] for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'cleared_run_directory']
    assert len(reasons) == 1 and 'no manifest' in reasons[0]


# --- smoke (two child processes, as on the cluster) ------------------------------------------------------
CHILD = '''
import json, sys
sys.path.insert(0, {repo!r})
from hyak.tests import test_train_support as T
from hyak.train import run_segment
from hyak.model import fetch_model
expected = json.loads(open({expected!r}).read())
run_segment.expected_code_files = lambda run, repo=None: expected[run]
fetch_model.expected_files = lambda repo=None: expected['model']
sys.modules['hyak.memory.chunked_hypernet'] = T.fake_memory_module()
sys.exit(run_segment.main(sys.argv[1:]))
'''


@pytest.mark.slow
def test_smoke_sequences_in_child_processes(make_tmp_config, repo_root, monkeypatch, tmp_path):
    world = T.build_world(make_tmp_config, repo_root, sparse_init=True)
    expected = tmp_path / 'expected.json'
    expected.write_text(json.dumps({**world.expected_code, 'model': world.model_expected}))
    script = tmp_path / 'child.py'
    script.write_text(CHILD.format(repo=str(repo_root), expected=str(expected)))
    monkeypatch.setenv('HYAK_FAKE_CONTROL', str(world.control_file))
    world.control_file.write_text('{}')
    command = [sys.executable, str(script)]
    result = S.run_smoke('uniform', child_command=command)
    assert result['outcome'] == 'smoke_verified' and result['checkpoints'] == [1, 3]
    assert [p['outcome'] for p in result['phases']] == ['smoke_phase_done', 'smoke_phase_done']
    assert result['stop_timer_seconds'] == [None, None]              # no job end time outside Slurm
    assert read_json(world.paths.smoke_dir('uniform') / S.SMOKE_VERIFIED)['step'] == 3   # monitor: complete
    # the RL smoke needs the uniform pair: not now (exit 75), nothing started
    assert S.run_smoke('rl', child_command=command)['outcome'] == 'uniform_bootstrap_pending'
    T.run_once(world, 'uniform', monkeypatch)            # bootstrap the real uniform run
    result = S.run_smoke('rl', child_command=command)
    assert result['checkpoints'] == [3] and result['policy_updates'] == 3
    # D8: the corrected cuts change the RL sanity examples; only the sanity tokenizers must be the uniform pair
    assert result['initial_eval_equals_uniform'] is False and result['sanity_tokenizers_equal_uniform'] is True
    assert 'narrowed (D8)' in result['initial_eval_check']
    receipt = result['paired_protocol']
    assert receipt['step'] == 3 and receipt['metrics_checked'] and receipt['entropy_weight'] == 0.0
    assert receipt['policy_reward'] == 'paired_pre_model_update_transferred_bpb'
    assert receipt['policy_reward_bpb'] == receipt['train_bpb'] and receipt['paired_valid_examples'] == 128
    assert not any(e.get('event') == 'warning' and 'initial_eval' in e.get('message', '')
                   for e in T.monitor_events(world, 'smoke-train'))
    phases = [e for e in T.monitor_events(world, 'smoke-train') if e.get('event') == 'paired_protocol_verified']
    assert [e['step'] for e in phases] == [2, 3]
    smoke_events = T.monitor_events(world, 'smoke-train')
    assert sum(e.get('event') == 'smoke_verified' for e in smoke_events) == 2
    # the smoke never touches the real run directories
    assert not world.paths.run_dir('rl').exists()
    # a second smoke starts from a wiped directory
    assert S.run_smoke('rl', child_command=command)['checkpoints'] == [3]


@pytest.mark.slow
def test_rl_smoke_fails_with_exit_4_when_the_policy_gets_the_gap(make_tmp_config, repo_root, monkeypatch, tmp_path):
    world = T.build_world(make_tmp_config, repo_root, sparse_init=True)
    expected = tmp_path / 'expected.json'
    expected.write_text(json.dumps({**world.expected_code, 'model': world.model_expected}))
    script = tmp_path / 'child.py'
    script.write_text(CHILD.format(repo=str(repo_root), expected=str(expected)))
    monkeypatch.setenv('HYAK_FAKE_CONTROL', str(world.control_file))
    T.run_once(world, 'uniform', monkeypatch)
    world.control_file.write_text(json.dumps({'reward_gap': True}))
    with pytest.raises(S.SegmentError, match='smoke phase first exited with status 4') as error:
        S.run_smoke('rl', child_command=[sys.executable, str(script)])
    assert error.value.exit_code == S.EXIT_CHECK


# --- the RL paired-protocol receipt (D8) --------------------------------------------------------------------
GOOD_METRICS = {'stage': 'train', 'step': 7, 'train_bpb': 1.3, 'paired_canonical_bpb': 1.05,
                'paired_gap_bpb': 1.3 - 1.05, 'policy_reward_bpb': 1.3, 'paired_valid_examples': 128,
                'paired_maximum_positions': 128}
GOOD_CONFIG = {'policy': {'entropy_weight': 0.0, 'learning_rate': 0.001, 'baseline_decay': 0.99, 'seed': 43},
               'policy_reward': 'paired_pre_model_update_transferred_bpb'}


def _receipt(tmp_path, config=None, metrics=None, step=7):
    (tmp_path / 'run_config.json').write_text(json.dumps(GOOD_CONFIG if config is None else config))
    return S.validate_rl_protocol(tmp_path, {'step': step, 'metrics': GOOD_METRICS if metrics is None else metrics})


def test_rl_protocol_receipt_accepts_the_straight_reward(tmp_path):
    receipt = _receipt(tmp_path)
    assert receipt['metrics_checked'] and receipt['policy_reward_bpb'] == receipt['train_bpb'] == 1.3
    # step 0 (the initial checkpoint) has no update yet: only run_config.json is checked
    assert _receipt(tmp_path, metrics={'stage': 'initial_checkpoint', 'step': 0}, step=0)['metrics_checked'] is False


@pytest.mark.parametrize('change, message', [
    ({'config': {**GOOD_CONFIG, 'policy': {**GOOD_CONFIG['policy'], 'entropy_weight': 0.01}}}, 'entropy_weight'),
    ({'config': {**GOOD_CONFIG, 'policy': {**GOOD_CONFIG['policy'], 'learning_rate': 0.01}}}, 'learning_rate'),
    ({'config': {**GOOD_CONFIG, 'policy_reward':
                 'paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb'}}, 'policy_reward'),
    ({'config': {**GOOD_CONFIG, 'policy_reward': 'pre_model_update_continuation_nats / scored_utf8_bytes / ln(2)'}},
     'policy_reward'),
    ({'metrics': {**GOOD_METRICS, 'paired_valid_examples': 127}}, 'paired_valid_examples'),
    ({'metrics': {**GOOD_METRICS, 'paired_maximum_positions': 129}}, 'paired_maximum_positions'),
    ({'metrics': {**GOOD_METRICS, 'paired_gap_bpb': 0.25 + 1e-9}}, 'paired_gap_bpb differs'),
    ({'metrics': {**GOOD_METRICS, 'policy_reward_bpb': 0.25}}, 'straight transferred BPB'),
    ({'metrics': {**GOOD_METRICS, 'policy_reward_bpb': 1.3 + 1e-11}}, 'straight transferred BPB'),
    ({'metrics': {k: v for k, v in GOOD_METRICS.items() if k != 'paired_canonical_bpb'}}, 'lack finite'),
    ({'metrics': {**GOOD_METRICS, 'train_bpb': float('nan')}}, 'lack finite'),
])
def test_rl_protocol_receipt_refuses_each_violation(tmp_path, change, message):
    with pytest.raises(S.SegmentError, match=message) as error:
        _receipt(tmp_path, **change)
    assert error.value.exit_code == S.EXIT_CHECK


@pytest.mark.parametrize('control, message', [
    ({'reward_gap': True}, 'straight transferred BPB'),
    ({'policy_reward': 'paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb'}, 'policy_reward'),
    ({'paired_valid_examples': 120}, 'paired_valid_examples'),
    ({'paired_maximum_positions': 130}, 'paired_maximum_positions'),
    ({'gap_error': 1e-6}, 'paired_gap_bpb differs'),
])
def test_rl_segment_with_a_protocol_violation_fails_the_check(world, monkeypatch, capsys, control, message):
    T.run_once(world, 'uniform', monkeypatch)
    with pytest.raises(S.SegmentError, match=message) as error:
        T.run_once(world, 'rl', monkeypatch, control={**control, 'stop_at': [2]})
    assert S.exit_code_for(error.value) == 4
    assert any(e.get('event') == 'segment_failed' and 'paired-protocol check' in e['error']
               for e in T.monitor_events(world, 'train-rl'))


def test_rl_segment_records_the_trainer_variant_and_the_receipt(world, monkeypatch):
    from hyak.code import pinned
    T.run_once(world, 'uniform', monkeypatch)
    outcome = T.run_once(world, 'rl', monkeypatch, control={'stop_at': [3]})
    assert outcome['outcome'] == 'paused' and outcome['step'] == 3
    record = read_json(world.paths.run_dir('rl') / R.SEGMENT_RECORD)
    variant = record['trainer_variant']
    assert {k: variant[k] for k in pinned.RL_TRAINER_VARIANT} == pinned.RL_TRAINER_VARIANT
    assert variant['staged_sha256'] == record['trainer_sha256']
    uniform = read_json(world.paths.run_dir('uniform') / R.SEGMENT_RECORD)['trainer_variant']
    assert uniform['name'] == 'original_uniform' and uniform['sha256'] == pinned.UNIFORM_TRAINER_SHA256
    assert read_json(world.paths.pairing)['rl_trainer_variant'] == pinned.RL_TRAINER_VARIANT
    start = segment_starts(world, 'train-rl')[0]['argv']
    assert start[start.index('--policy-entropy'):start.index('--policy-entropy') + 6] == \
        ['--policy-entropy', '0', '--policy-lr', '0.001', '--policy-baseline-decay', '0.99']
    config = read_json(world.paths.run_dir('rl') / 'run_config.json')
    assert config['policy']['entropy_weight'] == 0.0 and config['policy']['seed'] == 43
    receipts = [e for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'paired_protocol_verified']
    assert [(e['step'], e['metrics_checked'], e['policy_reward_bpb'] == e['train_bpb']) for e in receipts] == \
        [(3, True, True)]
    pair = [e for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'initial_eval_pairing']
    assert len(pair) == 1 and pair[0]['initial_eval_equal'] is False and pair[0]['sanity_tokenizers_equal'] is True
    assert 'narrowed (D8)' in pair[0]['check']
    assert not [e for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'warning'
                and ('initial_eval' in e['message'] or 'sanity' in e['message'])]


def test_cli_exit_codes(world, monkeypatch, capsys):
    monkeypatch.setattr('hyak.common.slurm.seconds_remaining', lambda default=None, now=None: 60.0)
    assert S.main(['--run', 'uniform', '--should-start']) == 75  # do not start (not now)
    assert json.loads(capsys.readouterr().out)['reason'] == 'insufficient_time'
    assert S.main(['--run', 'uniform']) == 4                     # the fake tree is not the pinned layout
    assert 'differs from the pinned layout' in capsys.readouterr().err
    from hyak.model import fetch_model
    monkeypatch.setattr(fetch_model, 'expected_files', lambda repo=None: world.model_expected)
    monkeypatch.setattr(S, 'expected_code_files', lambda run, repo=None: world.expected_code[run])
    monkeypatch.setattr(S, 'HYPERNET_INIT_BYTES', world.init_bytes)
    monkeypatch.setattr(S.verify_inputs, '__kwdefaults__', {'init_bytes': world.init_bytes})
    monkeypatch.setattr(S.run_segment, '__kwdefaults__', {**S.run_segment.__kwdefaults__,
                                                          'init_bytes': world.init_bytes})
    assert S.main(['--run', 'uniform']) == 0                     # too little time: nothing starts
    assert 'insufficient_time' in capsys.readouterr().out
    monkeypatch.setenv('HYAK_CONFIG', str(world.tmp.base / 'missing.env'))
    assert S.main(['--run', 'uniform']) == 2
    with pytest.raises(SystemExit):
        S.main(['--run', 'uniform', '--max-seconds', '10'])


def test_interop_with_stage_code_and_the_chunked_forward(tmp_config, repo_root):
    """Trees staged by hyak.code.stage_code pass verify_code; the real patch installs through the adapter."""
    stage_code = pytest.importorskip('hyak.code.stage_code')
    chunked = pytest.importorskip('hyak.memory.chunked_hypernet')
    import types
    paths = tmp_config.paths
    stage_code.stage(paths, repo=repo_root, log=lambda *a: None)
    for run in ('uniform', 'rl'):
        info = S.verify_code(paths.code_dir(run), run, paths.code_manifest)
        assert info['trainer_sha256'] == S.TRAINER_SHA256[run] and not info['rust_utils_rebuilt']
    with T.isolated_imports():
        sys.path.insert(0, str(paths.code_dir('uniform')))
        from hf_hypernet.modeling_hypernet import ZettHypernet
        adapter = S.TrainerAdapter(paths.code_dir('uniform'))
        adapter.module = types.SimpleNamespace(ZettHypernet=ZettHypernet)
        try:
            settings = adapter.install_memory_patch(16384)
            assert chunked.is_installed(ZettHypernet)
            assert {k: settings[k] for k in ('chunk_rows', 'use_checkpoint', 'algorithm_version')} == \
                {'chunk_rows': 16384, 'use_checkpoint': True, 'algorithm_version': settings['algorithm_version']}
            assert settings['module_sha256'] == sha256_file(repo_root / 'hyak/memory/chunked_hypernet.py')
        finally:
            chunked.uninstall(ZettHypernet)
    assert not list(paths.code_root.rglob('__pycache__'))
