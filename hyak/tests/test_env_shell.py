"""Tests of the environment shell scripts (no conda, no pip, no downloads: --dry-run and guards only)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from hyak.code import pinned

SCRIPTS = ('setup_train_env.sh', 'setup_data_env.sh', 'build_rust_utils.sh')
BASHES = [b for b in ('bash', '/bin/bash') if shutil.which(b)]


def _run(repo_root, script, *args, env=None, bash='bash'):
    full = {**os.environ, **(env or {})}
    return subprocess.run([bash, str(repo_root / 'hyak' / 'env' / script), *args], capture_output=True,
                          text=True, env=full, cwd=str(repo_root), timeout=120)


@pytest.mark.parametrize('bash', BASHES)
@pytest.mark.parametrize('script', SCRIPTS)
def test_syntax(repo_root, script, bash):
    out = subprocess.run([bash, '-n', str(repo_root / 'hyak' / 'env' / script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    text = (repo_root / 'hyak' / 'env' / script).read_text()
    assert 'set -euo pipefail' in text
    for forbidden in ('.bashrc', '.zshrc', 'conda init', '>> ~/'):
        assert forbidden not in text.replace('~/.bashrc, ~/.condarc', '')


@pytest.mark.parametrize('script', SCRIPTS)
def test_help(repo_root, script):
    out = _run(repo_root, script, '--help', bash=BASHES[-1])
    assert out.returncode == 0 and 'Usage' in out.stdout and 'set -euo' not in out.stdout


def test_train_dry_run_prints_the_pinned_plan_and_changes_nothing(repo_root, tmp_config):
    out = _run(repo_root, 'setup_train_env.sh', '--dry-run', bash=BASHES[-1])
    assert out.returncode == 0, out.stderr
    prefix = tmp_config.paths.env_train
    pins = ' '.join(f'{n}=={v}' for n, v in pinned.TRAIN_PINS + pinned.TRAIN_TEST_PINS)
    assert f'packages:     {pins}' in out.stdout
    assert f'conda create --yes --prefix {prefix} --override-channels --channel conda-forge python=3.12 pip' in out.stdout
    assert f'{prefix}/bin/python -m pip install --index-url https://pypi.org/simple {pins}' in out.stdout
    assert 'from PyPI (CUDA 13.2' in out.stdout and 'dry run complete' in out.stdout
    assert not tmp_config.paths.env_root.exists()


def test_train_dry_run_with_a_pytorch_index(repo_root, tmp_config):
    url = 'https://download.pytorch.org/whl/cu134'
    out = _run(repo_root, 'setup_train_env.sh', '--dry-run', env={'HYAK_TORCH_INDEX_URL': url}, bash=BASHES[-1])
    assert out.returncode == 0, out.stderr
    assert (f'-m pip install --index-url {url} --extra-index-url https://pypi.org/simple '
            'torch==2.15.0+cu134') in out.stdout


@pytest.mark.parametrize('label', ['cu128', 'cu126', 'cpu', 'rocm7.14'])
def test_train_rejects_torch_builds_that_do_not_exist(repo_root, tmp_config, label):
    out = _run(repo_root, 'setup_train_env.sh', '--dry-run',
               env={'HYAK_TORCH_INDEX_URL': f'https://download.pytorch.org/whl/{label}'})
    assert out.returncode == 1
    assert 'only for: cu132 cu134' in out.stderr and 'no CUDA 12' in out.stderr


def _existing_env(tmp_config, scratch_dir):
    venv = scratch_dir / 'venv-train'
    if not (venv / 'bin' / 'python').exists():
        pytest.skip('local train venv not found')
    tmp_config.paths.env_root.mkdir(parents=True)
    tmp_config.paths.env_train.symlink_to(venv, target_is_directory=True)


def test_train_refuses_to_keep_a_different_torch_build(repo_root, tmp_config, scratch_dir):
    _existing_env(tmp_config, scratch_dir)          # an "existing" env whose torch is the plain 2.15.0
    out = _run(repo_root, 'setup_train_env.sh', '--dry-run',
               env={'HYAK_TORCH_INDEX_URL': 'https://download.pytorch.org/whl/cu132'})
    assert out.returncode == 1
    assert 'installed torch is 2.15.0 but this configuration asks for 2.15.0+cu132' in out.stderr
    assert '--recreate' in out.stderr
    ok = _run(repo_root, 'setup_train_env.sh', '--dry-run')
    assert ok.returncode == 0, ok.stderr
    assert 'environment exists (Python 3.12)' in ok.stdout and 'conda create' not in ok.stdout


def test_recreate_refuses_a_directory_that_is_not_a_conda_env(repo_root, tmp_config, scratch_dir):
    _existing_env(tmp_config, scratch_dir)
    out = _run(repo_root, 'setup_train_env.sh', '--dry-run', '--recreate')
    assert out.returncode == 1 and 'not a conda environment' in out.stderr
    assert (tmp_config.paths.env_train / 'bin' / 'python').exists()


def test_data_dry_run(repo_root, tmp_config):
    out = _run(repo_root, 'setup_data_env.sh', '--dry-run', bash=BASHES[-1])
    assert out.returncode == 0, out.stderr
    prefix = tmp_config.paths.env_data
    assert f'{prefix}/bin/python -m pip install --index-url https://pypi.org/simple' in out.stdout
    for requirement in pinned.DATA_REQUIREMENTS:
        assert requirement in out.stdout
    assert 'toolchain 1.96.0 (minimal)' in out.stdout
    assert f'{tmp_config.paths.cargo_home}/bin/rustup toolchain install 1.96.0 --profile minimal' in out.stdout
    assert 'static.rust-lang.org/rustup/dist/' in out.stdout
    assert not tmp_config.paths.root.joinpath('toolchain').exists()
    skip = _run(repo_root, 'setup_data_env.sh', '--dry-run', '--skip-rust')
    assert skip.returncode == 0 and 'rustup' not in skip.stdout


def test_build_rust_utils_dry_run_phases(repo_root, tmp_config):
    out = _run(repo_root, 'build_rust_utils.sh', '--dry-run', bash=BASHES[-1])
    assert out.returncode == 0, out.stderr
    build = tmp_config.paths.rust_utils_build
    text = out.stdout
    order = [text.index(marker) for marker in (
        '-m hyak.code.pinned --check --group rust-sources',
        f'fetch --locked --manifest-path {build}/crate/Cargo.toml',
        f'build --release --locked --offline --manifest-path {build}/crate/Cargo.toml --features defaut',
        f'-m hyak.env.check_env --only rust_library --rust-library {build}/out/librust_utils.so',
        f'-m hyak.code.stage_code --rust-library {build}/out/librust_utils.so --rust-provenance '
        f'{build}/out/provenance.json --force')]
    assert order == sorted(order)
    assert '+1.96.0' in text
    for name in ('Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'rust-toolchain', 'src/lib.rs'):
        assert f'runtime/anytok_pool_20k/rust_utils/{name} {build}/crate/{name}' in text
    assert not build.exists()
    only = _run(repo_root, 'build_rust_utils.sh', 'check', '--dry-run')
    assert only.returncode == 0 and 'check_env' in only.stdout and 'cargo' not in only.stdout.split('phases:')[1]
    assert _run(repo_root, 'build_rust_utils.sh', 'bogus').returncode == 1


def test_build_rust_utils_refuses_a_build_tree_inside_the_repository(repo_root, make_tmp_config):
    make_tmp_config()
    inside = repo_root / 'hyak' / 'tests' / '.not-created-root'
    out = _run(repo_root, 'build_rust_utils.sh', '--dry-run', env={'ANYTOK_ROOT': str(inside)})
    assert out.returncode == 1 and 'inside the repository' in out.stderr
    assert not inside.exists()


def test_scripts_do_not_need_the_caller_to_set_hyak_repo(repo_root, tmp_config, tmp_path):
    out = subprocess.run(['bash', str(repo_root / 'hyak/env/setup_data_env.sh'), '--dry-run', '--skip-rust'],
                         capture_output=True, text=True, cwd=str(tmp_path),
                         env={k: v for k, v in os.environ.items() if k != 'HYAK_REPO'})
    assert out.returncode == 0, out.stderr
    assert f'repository:   {repo_root}' in out.stdout


# --- a full run of the setup scripts against a stand-in interpreter (no conda, no pip, no network) ---------------------
STUB_PYTHON = r'''#!/bin/bash
# stand-in for $ANYTOK_ROOT/env/<name>/bin/python: records its calls and the pip-related environment
{ echo "CALL $*"; env | grep -E '^(PIP_|TMPDIR=)' | sed 's/^/ENV /'; } >> "$STUB_LOG"
case "$1" in
  -c) case "$2" in
        *version_info*) echo 3.12 ;;
        *torch.version.cuda*) echo 13.2 ;;
        *) echo "${STUB_TORCH:-2.15.0}" ;;
      esac ;;
  -m) case "$2 $3" in
        "pip check") [ -z "${STUB_PIP_CHECK:-}" ] || { printf '%s\n' "$STUB_PIP_CHECK"; exit 1; } ;;
        "pip freeze") printf '%s\n' "${STUB_FREEZE:-numpy==2.3.5}" ;;
      esac ;;
  -) cat > /dev/null ;;
esac
exit 0
'''
FAKE_CONDA = '#!/bin/sh\nexit 0\n'
PIP_CONFLICT = ('tokenizers 0.22.2 has requirement huggingface-hub<2.0,>=0.16.4, but you have huggingface-hub 2.4.0.')


@pytest.fixture
def stub_env(tmp_config, tmp_path):
    """env/{train,data} with a stand-in interpreter, a fake conda on PATH, and the variables that make pip misbehave."""
    log = tmp_path / 'stub.log'
    bindir = tmp_path / 'stubbin'
    bindir.mkdir()
    (bindir / 'conda').write_text(FAKE_CONDA)
    (bindir / 'conda').chmod(0o755)
    for which in ('train', 'data'):
        python = tmp_config.paths.env_root / which / 'bin' / 'python'
        python.parent.mkdir(parents=True)
        python.write_text(STUB_PYTHON)
        python.chmod(0o755)
    env = {'STUB_LOG': str(log), 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_MODULES': '',
           'PIP_EXTRA_INDEX_URL': 'https://download.pytorch.org/whl/cpu', 'PIP_FIND_LINKS': '/somewhere',
           'PIP_CONSTRAINT': '/c.txt', 'PIP_CONFIG_FILE': '/home/me/pip.conf'}
    return env, log


def _setup_calls(log):
    lines = log.read_text().splitlines()
    return [l[5:] for l in lines if l.startswith('CALL ')], [l[4:] for l in lines if l.startswith('ENV ')]


def test_data_setup_pins_the_validated_set_and_tolerates_the_patched_wheels_conflict(repo_root, tmp_config, stub_env):
    env, log = stub_env
    out = _run(repo_root, 'setup_data_env.sh', '--skip-rust', env={**env, 'STUB_PIP_CHECK': PIP_CONFLICT},
               bash=BASHES[-1])
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'only the accepted tokenizers / huggingface-hub conflict' in out.stdout
    calls, envs = _setup_calls(log)
    install = next(c for c in calls if c.startswith('-m pip install'))
    for pin in ('datasets==5.2.0', 'pyarrow==26.0.1', 'huggingface-hub==2.4.0', 'hf-xet==1.8.0', 'fsspec==2026.9.0',
                'requests==2.34.2', 'urllib3==2.8.0'):
        assert pin in install
    assert '>=' not in install.replace('maturin>=1.4,<2.0', '')
    # pip sees no configuration and none of the login shell's PIP_* variables; its temporary files are on project storage
    assert 'PIP_CONFIG_FILE=/dev/null' in envs and not [e for e in envs if e.startswith(('PIP_EXTRA', 'PIP_FIND', 'PIP_CONSTRAINT'))]
    assert any(e == f'TMPDIR={tmp_config.paths.env_root}/.tmp' for e in envs)
    assert not (tmp_config.paths.env_root / '.tmp').exists()              # cleaned up at the end
    assert (tmp_config.paths.env_data / 'hyak-libpython').is_dir()
    assert (tmp_config.paths.env_root / 'data.freeze.txt').read_text() == 'numpy==2.3.5\n'


def test_data_setup_stops_at_any_other_pip_conflict(repo_root, tmp_config, stub_env):
    env, log = stub_env
    other = PIP_CONFLICT + '\ndatasets 5.2.0 has requirement fsspec[http]<=2026.0, but you have fsspec 2026.9.0.'
    out = _run(repo_root, 'setup_data_env.sh', '--skip-rust', env={**env, 'STUB_PIP_CHECK': other}, bash=BASHES[-1])
    assert out.returncode == 1 and 'pip check found conflicts' in out.stderr
    assert 'datasets 5.2.0 has requirement' in out.stderr
    assert not (tmp_config.paths.env_root / 'data.freeze.txt').exists()


def test_freezes_are_kept_with_a_timestamp_and_refreshed_by_freeze_only(repo_root, tmp_config, stub_env):
    env, log = stub_env
    freeze = tmp_config.paths.env_root / 'data.freeze.txt'
    freeze.write_text('numpy==2.3.4\n')
    out = _run(repo_root, 'setup_data_env.sh', '--freeze-only', env={**env, 'STUB_FREEZE': 'numpy==2.3.5\ntokenizers==0.22.2'},
               bash=BASHES[-1])
    assert out.returncode == 0, out.stdout + out.stderr
    assert freeze.read_text() == 'numpy==2.3.5\ntokenizers==0.22.2\n'
    (kept,) = [p for p in freeze.parent.glob('data.freeze.*.txt')]
    assert re.fullmatch(r'data\.freeze\.\d{8}T\d{6}Z\.txt', kept.name) and kept.read_text() == 'numpy==2.3.4\n'
    calls, envs = _setup_calls(log)
    assert calls == ['-m pip freeze --all']                  # nothing installed, no conda, no module
    # the same freeze again changes nothing
    _run(repo_root, 'setup_data_env.sh', '--freeze-only', env={**env, 'STUB_FREEZE': 'numpy==2.3.5\ntokenizers==0.22.2'})
    assert len(list(freeze.parent.glob('data.freeze.*.txt'))) == 1


def test_train_setup_isolates_pip_keeps_tmp_off_the_login_node_and_keeps_old_freezes(repo_root, tmp_config, stub_env):
    env, log = stub_env
    freeze = tmp_config.paths.env_root / 'train.freeze.txt'
    freeze.write_text('numpy==2.3.4\n')
    out = _run(repo_root, 'setup_train_env.sh', env=env, bash=BASHES[-1])
    assert out.returncode == 0, out.stdout + out.stderr
    calls, envs = _setup_calls(log)
    assert any(c.startswith('-m pip install') for c in calls)
    assert 'PIP_CONFIG_FILE=/dev/null' in envs and not [e for e in envs if e.startswith(('PIP_EXTRA', 'PIP_FIND', 'PIP_CONSTRAINT'))]
    assert any(e == f'TMPDIR={tmp_config.paths.env_root}/.tmp' for e in envs) and not (tmp_config.paths.env_root / '.tmp').exists()
    assert len(list(freeze.parent.glob('train.freeze.*.txt'))) == 1 and freeze.read_text() == 'numpy==2.3.5\n'
    assert (tmp_config.paths.env_train / 'hyak-libpython').is_dir()


def test_build_rust_utils_warns_on_a_slurm_login_node(repo_root, tmp_config, stub_env, tmp_path):
    env, _ = stub_env
    (tmp_path / 'stubbin' / 'sbatch').write_text('#!/bin/sh\nexit 0\n')
    (tmp_path / 'stubbin' / 'sbatch').chmod(0o755)
    out = _run(repo_root, 'build_rust_utils.sh', 'build', env={k: v for k, v in env.items() if k != 'SLURM_JOB_ID'},
               bash=BASHES[-1])
    assert 'looks like a Slurm login node' in out.stderr                              # then it stops for other reasons
