"""Tests of hyak.code.pinned and hyak.code.stage_code (DESIGN.md 5.A): pins match the
repository, staging reproduces the Modal image layout, tampering and drift are detected."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

import pytest

from hyak.code import pinned, stage_code
from hyak.code.stage_code import StageError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unlock(path: Path) -> None:
    """Make a staged (read-only) tree writable so a test can tamper with it."""
    for directory, dirnames, filenames in os.walk(path):
        os.chmod(directory, 0o755)
        for name in filenames:
            os.chmod(Path(directory, name), 0o644)


def _fake_repo(repo_root: Path, base: Path, chunked: str | None = 'print("chunked v1")\n') -> Path:
    """A copy of the three runtime snapshots and the RL trainer variant (+ an optional chunked_hypernet.py) outside git."""
    repo = base / 'fake-repo'
    for snapshot in pinned.RUNTIME_SNAPSHOTS:
        shutil.copytree(repo_root / snapshot, repo / snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    for rel in pinned.VARIANT_FILES:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / rel, repo / rel)
    if chunked is not None:
        target = repo / stage_code.CHUNKED_HYPERNET_REL
        target.parent.mkdir(parents=True)
        target.write_text(chunked)
    return repo


# --- pins ------------------------------------------------------------------------------------------
def test_every_pin_matches_the_repository(repo_root):
    assert pinned.check_files(repo_root) == []
    assert pinned.unexpected_snapshot_files(repo_root) == []


def test_named_hashes_are_the_recorded_original_values():
    # DESIGN.md non-negotiable 2 and NOTES/trainer-rl.md Appendix B.
    assert pinned.UNIFORM_TRAINER_SHA256 == 'c6406d263fb9cbc8638492d8c77327ce249e215eb74dc43e10335bc95a11ed52'
    assert pinned.ORIGINAL_RL_TRAINER_SHA256 == '36057349fd398b363a84ce6d8f12438e49cb17b401e4afdaf9e93a6333a691af'
    # Qingyun's joint regret RL20k snapshot at 3de64a5 and the port's straight-reward variant of its trainer
    assert pinned.REGRET_PINNED_AT_COMMIT == '3de64a55815ec5ebecb37f4e47b7addbe886cdde'
    assert pinned.REGRET_TRAINER_SHA256 == '76ff232402154653c5899826a4f63f8a4afadf5a443829f87919f8e75b31c6d4'
    assert pinned.PAIRED_REWARD_SHA256 == '406cfc58bf1c46cef0dc6785bbb313ca5797c0d35bf6cd786b3115a92e5523c1'
    assert pinned.RL_TRAINER == 'hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py'
    assert pinned.RL_TRAINER_SHA256 == '8dd3b8570dd6545123ab1aee5d338a8957d72ece26e48a759196d2619f43ff51'
    assert pinned.RL_TRAINER_VARIANT['base_sha256'] == pinned.REGRET_TRAINER_SHA256
    assert pinned.RL_TRAINER_VARIANT['policy_reward'] == 'paired_pre_model_update_transferred_bpb'
    assert (pinned.RL_POLICY_ENTROPY, pinned.RL_POLICY_LR, pinned.RL_POLICY_BASELINE_DECAY) == (0.0, 0.001, 0.99)
    assert pinned.SETTINGS_POLICY_SHA256 == '6dbfcbf8c6552ef98b56de08c09ae59bd35a2d7d1a75e7e295cfcf5346a90b8c'
    assert pinned.JOINT_TRAINING_SHA256.startswith('7ce18a9d')
    assert pinned.POOL_SAMPLING_SHA256.startswith('846446a0')
    assert pinned.TOKENIZER_POOL_SHA256.startswith('b194b104')
    assert pinned.RUST_LIBRARY_SHA256 == '9b485a745b374afbe20d846eba6e63c47fa251ff56c133f40c20c0fa7000e2f4'
    assert pinned.RUST_LIBRARY_BYTES == 3_810_872
    assert pinned.INPUTS_MANIFEST_SHA256 == '53d2968d9a1e1a7917b7f05716bca77acfcf9b61a06a08d35fa2bb3b4bbefb8c'


def test_pins_cover_every_file_of_the_runtime_snapshots(repo_root):
    files = set()
    for snapshot in pinned.RUNTIME_SNAPSHOTS:
        files |= {p.relative_to(repo_root).as_posix() for p in (repo_root / snapshot).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
    assert files == set(pinned.SNAPSHOT_FILES) | set(pinned.RUST_SOURCE_FILES)


def test_layout_is_the_union_of_the_modal_images():
    uniform, rl = pinned.tree_layout('uniform'), pinned.tree_layout('rl')
    hf = {f'hf_hypernet/{n}' for n in pinned.HF_HYPERNET_FILES}
    assert set(uniform) == hf | {pinned.TRAINER_REL, pinned.RUST_LIBRARY_REL}
    assert set(rl) == hf | {pinned.POLICY_REL, pinned.PAIRED_REWARD_REL, pinned.TRAINER_REL, pinned.RUST_LIBRARY_REL}
    assert uniform[pinned.TRAINER_REL] == 'runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py'
    # the regret launcher's image (scripts/modal_anytok_pool_rl_20k.py with ANYTOK_RL_EXPERIMENT=regret), with
    # the straight-reward variant in place of the regret trainer
    assert rl[pinned.TRAINER_REL] == 'hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py'
    assert rl[pinned.POLICY_REL] == 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
    assert rl[pinned.PAIRED_REWARD_REL] == 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py'
    assert set(rl.values()) <= set(pinned.STAGED_SOURCE_FILES) | {pinned.RUST_LIBRARY}
    assert pinned.ORIGINAL_RL_TRAINER not in rl.values() and pinned.REGRET_TRAINER not in rl.values()
    for rel in hf | {pinned.RUST_LIBRARY_REL}:
        assert uniform[rel] == rl[rel] and uniform[rel].startswith('runtime/anytok_pool_20k/')
    with pytest.raises(ValueError):
        pinned.tree_layout('zett')


def test_check_files_reports_changes_and_symlinks(tmp_path, repo_root):
    rel = 'runtime/anytok_pool_20k/hf_hypernet/pool_sampling.py'
    repo = tmp_path / 'r'
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_bytes((repo_root / rel).read_bytes() + b'#')
    problems = pinned.check_files(repo, {rel: pinned.SNAPSHOT_FILES[rel]})
    assert len(problems) == 1 and 'bytes' in problems[0]
    # Same length, one byte different: only the SHA-256 comparison can see it (the size branch is not reached).
    data = bytearray((repo_root / rel).read_bytes())
    data[len(data) // 2] ^= 0x01
    (repo / rel).write_bytes(bytes(data))
    assert len(data) == pinned.SNAPSHOT_FILES[rel][1]
    problems = pinned.check_files(repo, {rel: pinned.SNAPSHOT_FILES[rel]})
    assert len(problems) == 1 and 'sha256' in problems[0] and 'bytes' not in problems[0]
    (repo / rel).unlink()
    (repo / rel).symlink_to(repo_root / rel)
    assert 'symlink' in pinned.check_files(repo, {rel: pinned.SNAPSHOT_FILES[rel]})[0]
    assert 'missing' in pinned.check_files(repo, {'nope.py': ('0' * 64, 1)})[0]


def test_pinned_cli_check(repo_root):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(repo_root)}
    out = subprocess.run([sys.executable, '-m', 'hyak.code.pinned', '--check'], cwd=repo_root, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'OK 26 pinned files' in out.stdout
    out = subprocess.run([sys.executable, '-m', 'hyak.code.pinned', '--layout', 'rl'], cwd=repo_root, env=env,
                         capture_output=True, text=True)
    assert json.loads(out.stdout) == pinned.tree_layout('rl')


def test_environment_pins_match_the_setup_scripts(repo_root):
    train = (repo_root / 'hyak/env/setup_train_env.sh').read_text()
    pins = re.search(r'^PINS=\(([^)]*)\)', train, re.M).group(1).split()
    assert pins == [f'{n}=={v}' for n, v in pinned.TRAIN_PINS]
    assert re.search(r'^TEST_PINS=\(([^)]*)\)', train, re.M).group(1).split() == \
        [f'{n}=={v}' for n, v in pinned.TRAIN_TEST_PINS]
    data = (repo_root / 'hyak/env/setup_data_env.sh').read_text()
    data_pins = re.search(r'^DATA_PINS=\(([^)]*)\)', data, re.M).group(1).split()
    assert data_pins == [f'{n}=={v}' for n, v in pinned.DATA_PINS]
    requirements = re.search(r'^REQUIREMENTS=\((.*)\)$', data, re.M).group(1)
    assert requirements == '"${DATA_PINS[@]}" "maturin>=1.4,<2.0"'
    assert list(pinned.DATA_REQUIREMENTS) == data_pins + ['maturin>=1.4,<2.0']
    # the validated corpus-build set (corpus-2): nothing in it may become a range again
    assert dict(pinned.DATA_PINS) == {'datasets': '5.2.0', 'pyarrow': '26.0.1', 'huggingface-hub': '2.4.0',
                                      'hf-xet': '1.8.0', 'fsspec': '2026.9.0', 'requests': '2.34.2',
                                      'urllib3': '2.8.0'}
    # The six pins are those of the Modal training image (scripts/modal_anytok_pool_20k.py:37-42).
    launcher = (repo_root / 'scripts/modal_anytok_pool_20k.py').read_text()
    for name, version in pinned.TRAIN_PINS:
        assert f'"{name}=={version}"' in launcher
    assert re.search(r'^RUST_TOOLCHAIN=1\.96\.0$', data, re.M)


# --- staging ------------------------------------------------------------------------------------------
def test_stage_reproduces_the_layout_with_real_read_only_copies(tmp_config, repo_root):
    paths = tmp_config.paths
    result = stage_code.stage(paths, log=lambda m: None)
    assert result['trees_written'] == ['uniform', 'rl'] and result['manifest_written']
    for run in ('uniform', 'rl'):
        tree = paths.code_dir(run)
        found = {p.relative_to(tree).as_posix() for p in tree.rglob('*') if not p.is_dir()}
        assert found == set(pinned.tree_layout(run))
        for rel, source in pinned.tree_layout(run).items():
            staged = tree / rel
            assert not staged.is_symlink() and staged.is_file()
            assert staged.read_bytes() == (repo_root / source).read_bytes()
            assert stat.S_IMODE(staged.stat().st_mode) == 0o444
        assert stat.S_IMODE(tree.stat().st_mode) == 0o555
        assert not list(tree.rglob('__pycache__'))
    manifest = json.loads(paths.code_manifest.read_text())
    assert manifest['kind'] == 'hyak_code_manifest' and manifest['schema'] == 1
    assert manifest['identity_sha256'] == stage_code.identity_sha256(manifest)
    for run in ('uniform', 'rl'):
        files = manifest['trees'][run]['files']
        assert manifest['trees'][run]['tree_sha256'] == stage_code.tree_sha256(files)
        for rel, record in files.items():
            assert record['sha256'] == _sha(paths.code_dir(run) / rel)
            assert record['bytes'] == (paths.code_dir(run) / rel).stat().st_size
            assert record['source'] == pinned.tree_layout(run)[rel]
    assert manifest['rust_library']['source'] == 'committed'
    assert manifest['rust_library']['sha256'] == pinned.RUST_LIBRARY_SHA256
    chunked = repo_root / stage_code.CHUNKED_HYPERNET_REL
    expected = _sha(chunked) if chunked.is_file() else None
    assert manifest['chunked_hypernet'] == {'path': 'hyak/memory/chunked_hypernet.py', 'sha256': expected,
                                            'bytes': chunked.stat().st_size if chunked.is_file() else None}
    head = subprocess.run(['git', '-C', str(repo_root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode == 0:
        assert manifest['repository']['commit'] == head.stdout.strip()
    assert manifest['trainer_sha256'] == {'uniform': pinned.UNIFORM_TRAINER_SHA256, 'rl': pinned.RL_TRAINER_SHA256}
    assert manifest['rl_trainer_variant'] == pinned.RL_TRAINER_VARIANT
    assert manifest['paired_reward_sha256'] == pinned.PAIRED_REWARD_SHA256
    assert manifest['regret_pinned_at_commit'] == pinned.REGRET_PINNED_AT_COMMIT
    assert _sha(paths.code_dir('rl') / pinned.TRAINER_REL) == pinned.RL_TRAINER_SHA256
    assert _sha(paths.code_dir('uniform') / pinned.TRAINER_REL) == pinned.UNIFORM_TRAINER_SHA256
    assert stage_code.check_pairing(manifest) == []
    assert stage_code.verify_stage(paths) == []


def test_stage_is_idempotent_and_keeps_the_manifest(tmp_config):
    paths = tmp_config.paths
    stage_code.stage(paths, log=lambda m: None)
    before = paths.code_manifest.read_bytes()
    messages = []
    result = stage_code.stage(paths, log=messages.append)
    assert not result['changed'] and result['trees_written'] == [] and not result['manifest_written']
    assert paths.code_manifest.read_bytes() == before
    assert any('up to date' in m for m in messages)


def test_tampering_is_detected_and_refused_without_force(tmp_config):
    paths = tmp_config.paths
    stage_code.stage(paths, log=lambda m: None)
    victim = paths.code_dir('rl') / 'hf_hypernet' / 'joint_training.py'
    _unlock(paths.code_dir('rl'))
    victim.write_text(victim.read_text() + '\n# edited\n')
    problems = stage_code.verify_stage(paths)
    assert any('rl/hf_hypernet/joint_training.py' in p for p in problems)
    with pytest.raises(StageError, match='--force'):
        stage_code.stage(paths, log=lambda m: None)
    result = stage_code.stage(paths, force=True, log=lambda m: None)
    assert result['trees_written'] == ['rl'] and not result['manifest_written']
    assert stage_code.verify_stage(paths) == []


def test_same_length_tampering_is_detected_by_the_hash(tmp_config):
    """One byte replaced, the size unchanged: only a SHA-256 comparison notices (tests-8)."""
    paths = tmp_config.paths
    stage_code.stage(paths, log=lambda m: None)
    victim = paths.code_dir('rl') / 'hf_hypernet' / 'joint_training.py'
    _unlock(paths.code_dir('rl'))
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0x01
    victim.write_bytes(bytes(data))
    assert any('rl/hf_hypernet/joint_training.py' in p for p in stage_code.verify_stage(paths))
    with pytest.raises(StageError, match='--force'):
        stage_code.stage(paths, log=lambda m: None)
    assert stage_code.stage(paths, force=True, log=lambda m: None)['trees_written'] == ['rl']
    assert stage_code.verify_stage(paths) == []


def test_extra_files_symlinks_and_missing_trees_are_detected(tmp_config):
    paths = tmp_config.paths
    stage_code.stage(paths, log=lambda m: None)
    tree = paths.code_dir('uniform')
    _unlock(tree)
    (tree / 'hf_hypernet' / '__pycache__').mkdir()
    (tree / 'hf_hypernet' / '__pycache__' / 'x.pyc').write_bytes(b'0')
    (tree / 'scripts' / 'link.py').symlink_to(tree / 'scripts' / 'train_joint_smollm_input.py')
    problems = stage_code.verify_stage(paths)
    assert any('__pycache__/: unexpected directory' in p for p in problems)
    assert any('link.py: symlink' in p for p in problems)
    assert any('x.pyc: not part of the staged layout' in p for p in problems)
    stage_code.stage(paths, force=True, log=lambda m: None)
    assert stage_code.verify_stage(paths) == []
    stage_code._remove_tree(paths.code_dir('rl'))
    assert any('rl' in p and 'missing' in p for p in stage_code.verify_stage(paths))
    result = stage_code.stage(paths, log=lambda m: None)          # a missing tree is restaged without --force
    assert result['trees_written'] == ['rl'] and not result['manifest_written']


def test_force_is_refused_once_a_run_directory_is_in_use(tmp_config):
    paths = tmp_config.paths
    stage_code.stage(paths, log=lambda m: None)
    _unlock(paths.code_dir('uniform'))
    (paths.code_dir('uniform') / 'scripts' / 'train_joint_smollm_input.py').write_text('changed\n')
    paths.run_dir('uniform').mkdir(parents=True)
    (paths.run_dir('uniform') / 'run_config.json').write_text('{}')
    with pytest.raises(StageError, match='run directories already exist'):
        stage_code.stage(paths, force=True, log=lambda m: None)
    shutil.rmtree(paths.run_dir('uniform'))
    stage_code.stage(paths, force=True, log=lambda m: None)
    assert stage_code.verify_stage(paths) == []


def test_chunked_hypernet_hash_is_recorded_and_drift_detected(tmp_config, repo_root, tmp_path):
    paths = tmp_config.paths
    repo = _fake_repo(repo_root, tmp_path)
    result = stage_code.stage(paths, repo=repo, log=lambda m: None)
    chunked = repo / stage_code.CHUNKED_HYPERNET_REL
    assert result['manifest']['chunked_hypernet']['sha256'] == _sha(chunked)
    assert result['manifest']['repository']['commit'] is None        # not a git checkout: recorded, not fatal
    assert result['manifest']['repository']['error']
    assert stage_code.verify_stage(paths, repo=repo) == []
    chunked.write_text('print("chunked v2")\n')
    problems = stage_code.verify_stage(paths, repo=repo)
    assert len(problems) == 1 and 'changed since staging' in problems[0]
    with pytest.raises(StageError, match='chunked_hypernet.py'):
        stage_code.stage(paths, repo=repo, log=lambda m: None)
    result = stage_code.stage(paths, repo=repo, force=True, log=lambda m: None)
    assert result['manifest_written'] and result['trees_written'] == []
    assert result['manifest']['chunked_hypernet']['sha256'] == _sha(chunked)
    chunked.unlink()
    assert 'changed since staging' in stage_code.verify_stage(paths, repo=repo)[0]


def test_absent_chunked_hypernet_is_recorded_as_null(tmp_config, repo_root, tmp_path):
    repo = _fake_repo(repo_root, tmp_path, chunked=None)
    manifest = stage_code.stage(tmp_config.paths, repo=repo, log=lambda m: None)['manifest']
    assert manifest['chunked_hypernet'] == {'path': 'hyak/memory/chunked_hypernet.py', 'sha256': None, 'bytes': None}
    assert stage_code.verify_stage(tmp_config.paths, repo=repo) == []


def test_modified_snapshot_in_the_repository_is_refused(tmp_config, repo_root, tmp_path):
    repo = _fake_repo(repo_root, tmp_path)
    target = repo / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
    target.write_text(target.read_text() + '\n')
    with pytest.raises(StageError, match='settings_policy.py'):
        stage_code.stage(tmp_config.paths, repo=repo, log=lambda m: None)
    target.write_bytes((repo_root / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py').read_bytes())
    (repo / 'runtime/anytok_pool_20k/hf_hypernet/extra.py').write_text('')
    with pytest.raises(StageError, match='not pinned'):
        stage_code.stage(tmp_config.paths, repo=repo, log=lambda m: None)
    assert not tmp_config.paths.code_manifest.exists()


def test_modified_regret_snapshot_or_variant_is_refused(tmp_config, repo_root, tmp_path):
    repo = _fake_repo(repo_root, tmp_path)
    for rel in (pinned.PAIRED_REWARD, pinned.REGRET_TRAINER, pinned.RL_TRAINER):
        target = repo / rel
        original = target.read_bytes()
        target.write_bytes(original + b'\n')
        with pytest.raises(StageError, match=Path(rel).name):
            stage_code.stage(tmp_config.paths, repo=repo, log=lambda m: None)
        target.write_bytes(original)
    (repo / pinned.REGRET_SNAPSHOT / 'hf_hypernet' / 'extra.py').write_text('')
    with pytest.raises(StageError, match='not pinned'):
        stage_code.stage(tmp_config.paths, repo=repo, log=lambda m: None)
    assert not tmp_config.paths.code_manifest.exists()


def test_rl_tree_pairing_allows_only_the_policy_and_the_paired_reward(tmp_config):
    paths = tmp_config.paths
    manifest = stage_code.stage(paths, log=lambda m: None)['manifest']
    rl_only = sorted(set(manifest['trees']['rl']['files']) - set(manifest['trees']['uniform']['files']))
    assert rl_only == sorted(pinned.RL_ONLY_RELS)
    changed = json.loads(json.dumps(manifest))
    changed['trees']['rl']['files']['hf_hypernet/other.py'] = {'bytes': 1, 'sha256': '0' * 64}
    assert any('unexpected hf_hypernet files' in p for p in stage_code.check_pairing(changed))


def test_rebuilt_library_is_recorded_kept_and_switching_back_needs_force(tmp_config, tmp_path):
    paths = tmp_config.paths
    library = tmp_path / 'librust_utils.so'
    library.write_bytes(b'\x7fELF rebuilt test bytes')
    provenance = tmp_path / 'provenance.json'
    provenance.write_text(json.dumps({'sha256': _sha(library), 'variant': 'extension-module'}))
    manifest = stage_code.stage(paths, rust_library=str(library), rust_provenance=provenance,
                                log=lambda m: None)['manifest']
    record = manifest['rust_library']
    assert record['source'] == 'rebuilt' and record['sha256'] == _sha(library)
    assert record['provenance']['variant'] == 'extension-module'
    for run in ('uniform', 'rl'):
        assert (paths.code_dir(run) / pinned.RUST_LIBRARY_REL).read_bytes() == library.read_bytes()
    assert stage_code.verify_stage(paths) == []
    library.unlink()                                   # the default keeps the staged rebuilt library
    assert not stage_code.stage(paths, log=lambda m: None)['changed']
    with pytest.raises(StageError, match='librust_utils.so'):
        stage_code.stage(paths, rust_library='committed', log=lambda m: None)
    manifest = stage_code.stage(paths, rust_library='committed', force=True, log=lambda m: None)['manifest']
    assert manifest['rust_library']['source'] == 'committed'
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'sha256': '0' * 64}))
    other = tmp_path / 'other.so'
    other.write_bytes(b'other')
    with pytest.raises(StageError, match='describes sha256'):
        stage_code.stage(paths, rust_library=str(other), rust_provenance=bad, force=True, log=lambda m: None)


def test_hand_edited_manifest_is_rejected(tmp_config):
    paths = tmp_config.paths
    stage_code.stage(paths, log=lambda m: None)
    manifest = json.loads(paths.code_manifest.read_text())
    manifest['trees']['uniform']['files']['hf_hypernet/__init__.py']['sha256'] = 'f' * 64
    paths.code_manifest.write_text(json.dumps(manifest))
    with pytest.raises(StageError, match='identity_sha256'):
        stage_code.load_manifest(paths)
    assert 'identity_sha256' in stage_code.verify_stage(paths)[0]
    with pytest.raises(StageError, match='--force'):
        stage_code.stage(paths, log=lambda m: None)
    stage_code.stage(paths, force=True, log=lambda m: None)
    assert stage_code.verify_stage(paths) == []


def test_pairing_check_flags_divergent_trees(tmp_config):
    manifest = stage_code.stage(tmp_config.paths, log=lambda m: None)['manifest']
    manifest = json.loads(json.dumps(manifest))
    manifest['trees']['rl']['files']['hf_hypernet/modeling_hypernet.py']['sha256'] = '0' * 64
    assert stage_code.check_pairing(manifest) == ['hf_hypernet/modeling_hypernet.py differs between the uniform and rl trees']
    assert stage_code.hf_hypernet_hashes(manifest, 'uniform')['hf_hypernet/joint_training.py'] == \
        pinned.JOINT_TRAINING_SHA256


def test_cli_stage_verify_show(tmp_config, capsys):
    assert stage_code.main([]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line['stage'] == 'code_staged' and line['trees_written'] == ['uniform', 'rl']
    assert stage_code.main(['--verify']) == 0
    assert 'OK staged code verified' in capsys.readouterr().out
    assert stage_code.main(['--show']) == 0
    assert json.loads(capsys.readouterr().out)['kind'] == 'hyak_code_manifest'
    _unlock(tmp_config.paths.code_dir('uniform'))
    (tmp_config.paths.code_dir('uniform') / 'hf_hypernet' / '__init__.py').write_text('x = 1\n')
    assert stage_code.main(['--verify', '--run', 'rl']) == 0
    assert stage_code.main(['--verify']) == 1
    assert 'PROBLEM uniform/hf_hypernet/__init__.py' in capsys.readouterr().err
    assert stage_code.main([]) == 1


def test_cli_reports_config_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('HYAK_CONFIG', str(tmp_path / 'missing.env'))
    assert stage_code.main(['--verify']) == 2
    assert 'not found' in capsys.readouterr().err


_SYSTEM_PYTHON = shutil.which('python3', path='/usr/bin:/bin')


@pytest.mark.skipif(_SYSTEM_PYTHON is None, reason='no system python3')
def test_modules_run_on_the_login_node_python(repo_root, tmp_config):
    """pinned, stage_code, fetch_model and check_env must at least compile on Python >= 3.9 (login nodes)."""
    version = subprocess.run([_SYSTEM_PYTHON, '-c', 'import sys; print(sys.version_info[:2] >= (3, 9))'],
                             capture_output=True, text=True).stdout.strip()
    if version != 'True':
        pytest.skip('system python3 is older than 3.9')
    files = ['hyak/code/pinned.py', 'hyak/code/stage_code.py', 'hyak/model/fetch_model.py', 'hyak/env/check_env.py']
    code = 'import sys\nfor f in sys.argv[1:]:\n    compile(open(f).read(), f, "exec")\n'
    out = subprocess.run([_SYSTEM_PYTHON, '-c', code, *files], cwd=repo_root, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env.update(PYTHONPATH=str(repo_root), PYTHONDONTWRITEBYTECODE='1', HYAK_CONFIG=str(tmp_config.config_file))
    out = subprocess.run([_SYSTEM_PYTHON, '-m', 'hyak.code.stage_code'], cwd=repo_root, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    out = subprocess.run([_SYSTEM_PYTHON, '-m', 'hyak.code.stage_code', '--verify'], cwd=repo_root, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and 'OK staged code verified' in out.stdout, out.stderr


def test_leftovers_of_an_interrupted_staging_run_are_removed(tmp_config):
    paths = tmp_config.paths
    leftover = paths.code_root / '.staging-uniform-99999'
    (leftover / 'hf_hypernet').mkdir(parents=True)
    (leftover / 'hf_hypernet' / 'x.py').write_text('')
    os.chmod(leftover / 'hf_hypernet', 0o555)
    stage_code.stage(paths, log=lambda m: None)
    assert not leftover.exists()
    assert stage_code.verify_stage(paths) == []
