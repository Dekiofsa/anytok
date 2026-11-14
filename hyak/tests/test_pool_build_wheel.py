"""hyak/pool/build_wheel.sh: constants, argument handling, fail-fast checks; slow tests use the local build."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

import hyak.pool as pool

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'hyak/pool/build_wheel.sh'


def run(*args, env=None, timeout=300):
    full = {**os.environ, 'HYAK_PYTHON': sys.executable, 'PYTHONDONTWRITEBYTECODE': '1', **(env or {})}
    result = subprocess.run(['bash', str(SCRIPT), *args], cwd=str(REPO), env=full, capture_output=True, text=True,
                            timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def test_syntax():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0


def test_constants_match_python_and_patch_headers():
    text = SCRIPT.read_text()
    assert re.search(r'^UPSTREAM_REVISION=(\w+)$', text, re.M).group(1) == pool.UPSTREAM_REVISION
    assert re.search(r'^UPSTREAM_TAG=(\S+)$', text, re.M).group(1) == pool.UPSTREAM_TAG
    assert re.search(r'^RUST_TOOLCHAIN_DEFAULT=(\S+)$', text, re.M).group(1) == pool.RUST_TOOLCHAIN
    assert tuple(re.search(r'^PATCH_FILES="([^"]+)"$', text, re.M).group(1).split()) == pool.PATCH_FILES
    assert re.search(r'^FORBIDDEN_PATCH=(\S+)$', text, re.M).group(1) == pool.FORBIDDEN_PATCH
    table = re.search(r'^BLOB_TABLE="([^"]+)"$', text, re.M).group(1).splitlines()
    rows = {line.split()[0]: line.split()[1:] for line in table}
    assert {path: base for path, (base, _) in rows.items()} == pool.UPSTREAM_BLOBS
    post = {}
    for name in pool.PATCH_FILES:
        patch = (REPO / 'patches' / name).read_text()
        post.update(dict(re.findall(r'^diff --git a/(\S+) b/\S+\nindex [0-9a-f]+\.\.([0-9a-f]+)', patch, re.M)))
    assert {path: after for path, (_, after) in rows.items()} == post


def test_arguments():
    code, out = run('--help')
    assert code == 0 and 'Phases' in out and 'merge-noise' in out
    code, out = run()
    assert code != 0 and 'at least one phase' in out
    code, out = run('fetch', '--bogus')
    assert code != 0 and "unknown argument '--bogus'" in out
    code, out = run('fetch', '--no-config', '--src', '/x')
    assert code != 0 and '--no-config requires' in out
    code, out = run('fetch', '--no-config', '--src', 'relative', '--wheels', '/w', '--python', '/p', '--toolchain', '/t')
    assert code != 0 and 'SRC must be an absolute path' in out


def test_missing_toolchain_fails_before_any_download(tmp_path):
    code, out = run('fetch', '--no-config', '--src', str(tmp_path / 'src'), '--wheels', str(tmp_path / 'w'),
                    '--python', sys.executable, '--toolchain', str(tmp_path / 'toolchain'))
    assert code != 0 and 'setup_data_env.sh' in out
    assert not (tmp_path / 'src').exists()


def _bin_without(tmp_path, excluded, fakes=()):
    """A PATH directory with every program of /bin and /usr/bin except ``excluded`` (+ fake ``fakes``)."""
    folder = tmp_path / 'bin'
    folder.mkdir()
    for source in (Path('/usr/bin'), Path('/bin')):
        for entry in source.iterdir():
            if entry.name in excluded or (folder / entry.name).exists():
                continue
            (folder / entry.name).symlink_to(entry)
    for name in fakes:
        (folder / name).write_text('#!/bin/sh\nexit 0\n')
        (folder / name).chmod(0o755)
    return str(folder)


COMPILERS = {'cc', 'c++', 'gcc', 'g++', 'clang', 'clang++', 'cpp', 'c89', 'c99'}


@pytest.mark.parametrize('case', ['gcc-without-g++', 'no-git'])
def test_build_phase_checks_compilers_and_git_first(tmp_path, case):
    """pool-8: the build phase needs git and both a C and a C++ compiler; it says so before any work."""
    if case == 'no-git':
        path = _bin_without(tmp_path, {'git'} | COMPILERS)
        expected = 'git is not installed'
    else:
        path = _bin_without(tmp_path, {'git'} | COMPILERS, fakes=('gcc', 'git'))
        expected = 'a C and a C++ compiler are needed'
    code, out = run('build', '--no-config', '--src', str(tmp_path / 'src'), '--wheels', str(tmp_path / 'w'),
                    '--python', sys.executable, '--toolchain', str(tmp_path / 'toolchain'),
                    env={'PATH': path})
    assert code != 0 and expected in out, out
    if case != 'no-git':
        assert 'C++: none' in out and 'module load gcc' in out


def test_install_refuses_the_train_environment(make_tmp_config):
    tmp = make_tmp_config()
    train_python = tmp.paths.env_train / 'bin' / 'python'
    train_python.parent.mkdir(parents=True)
    train_python.symlink_to(sys.executable)
    tmp.paths.wheels.mkdir(parents=True)
    (tmp.paths.wheels / 'provenance.json').write_text(json.dumps({'wheel': {'name': 'x.whl'}}))
    code, out = run('install', '--python', str(train_python), env={'HYAK_CONFIG': str(tmp.config_file)})
    assert code != 0 and 'refusing to install the patched wheel into the train environment' in out


def test_config_paths_come_from_hyak_common_paths(make_tmp_config):
    tmp = make_tmp_config()
    code, out = run('build', env={'HYAK_CONFIG': str(tmp.config_file)})
    assert code != 0
    assert str(tmp.paths.tokenizers_src) in out and str(tmp.paths.wheels) in out
    assert str(tmp.paths.env_data / 'bin' / 'python') in out


def _local_assets(scratch_dir):
    src = scratch_dir / 'wheel-build' / 'tokenizers-src'
    toolchain = scratch_dir / 'toolchain'
    if not (src / '.git').is_dir() or not (toolchain / 'cargo' / 'bin' / 'cargo').exists():
        pytest.skip('local tokenizers clone or Rust toolchain not present (run build_wheel.sh fetch first)')
    return src, toolchain


@pytest.mark.slow
def test_fetch_from_a_local_clone_and_tamper_detection(tmp_path, scratch_dir):
    upstream, toolchain = _local_assets(scratch_dir)
    src = tmp_path / 'src'
    common = ['--no-config', '--src', str(src), '--wheels', str(tmp_path / 'wheels'), '--python', sys.executable,
              '--toolchain', str(toolchain)]
    env = {'CARGO_NET_OFFLINE': 'true'}
    code, out = run('fetch', '--source', str(upstream), *common, env=env, timeout=900)
    assert code == 0, out
    marker = json.loads((src / '.anytok-source.json').read_text())
    assert marker['revision'] == pool.UPSTREAM_REVISION and marker['patches'] == pool.PATCH_SHA256
    assert sorted(marker['files']) == sorted(pool.UPSTREAM_BLOBS)
    record = json.loads((src / '.anytok-fetch.json').read_text())
    assert record['locks']['bindings/python/Cargo.lock']['tracked_upstream'] is True
    assert record['locks']['tokenizers/Cargo.lock']['tracked_upstream'] is False
    code, out = run('fetch', '--source', str(upstream), *common, env=env, timeout=900)   # idempotent
    assert code == 0 and 'reusing existing source' in out
    lock = src / 'tokenizers' / 'Cargo.lock'
    original_lock = lock.read_bytes()
    lock.write_bytes(original_lock + b'\n')
    code, out = run('build', *common, env=env)
    assert code != 0 and 'changed after the fetch phase' in out
    lock.write_bytes(original_lock)
    target = src / 'tokenizers/src/models/unigram/lattice.rs'
    target.write_text(target.read_text() + '\n// edited\n')
    code, out = run('build', *common, env=env)
    assert code != 0 and 'was edited' in out
    shutil.rmtree(src)


@pytest.mark.slow
def test_installed_local_wheel_verifies(scratch_dir):
    python = scratch_dir / 'venv-pool' / 'bin' / 'python'
    wheels = scratch_dir / 'wheel-build' / 'wheels'
    if not python.exists() or not (wheels / 'provenance.json').exists():
        pytest.skip('local patched wheel not built/installed (venv-pool)')
    code, out = run('verify', '--no-config', '--src', str(scratch_dir / 'wheel-build' / 'tokenizers-src'),
                    '--wheels', str(wheels), '--python', str(python), '--toolchain', str(scratch_dir / 'toolchain'),
                    timeout=600)
    assert code == 0, out
    assert '"tokenizers_version": "0.22.2"' in out
