"""Shared pytest configuration for hyak/tests.

* Puts the repository root on ``sys.path`` (``import hyak``, ``import monitoring``).
* Markers: ``network`` (skipped unless ``HYAK_TEST_NETWORK=1``) and ``slow`` (runs by
  default; skipped when ``HYAK_TEST_SKIP_SLOW=1``; deselect with ``-m "not slow"``).
* Autouse isolation: every test starts without config keys, per-stage overrides,
  ``HYAK_REPO``, ``HYAK_PYTHON``, ``HYAK_STAGE``, ``HYAK_RUN``, ``HYAK_ENV`` or ``SLURM_*`` variables in the
  environment (so a test run inside a configured Slurm job cannot touch real storage), with
  ``HYAK_CONFIG`` pointing at a file that does not exist (code that forgets to configure itself fails with
  "Config file ... not found" instead of reading the checkout's real ``hyak/config.env`` and acting on the
  user's storage), with ``HYAK_MONITOR_ROOT`` pointing at a per-test temporary directory (StatusWriter and
  lib.sh never write into the repository's modal_results/), and with ``HF_HOME`` / ``HF_HUB_CACHE`` in
  per-test temporary directories and ``HF_HUB_OFFLINE=1`` (unless ``HYAK_TEST_NETWORK=1``).  A test that needs
  any of these sets it itself (``monkeypatch.setenv``).
* Before any test module is imported (huggingface_hub reads these at import): ``HF_HOME`` and
  ``HF_HUB_CACHE`` in a temporary directory removed at exit, ``HF_HUB_OFFLINE=1`` unless
  ``HYAK_TEST_NETWORK=1``, and no bytecode (``sys.dont_write_bytecode``, ``PYTHONDONTWRITEBYTECODE``).
* The hub's own request in the first header it builds (``$HF_HOME/.agent_harnesses.json``) therefore never
  reaches the network or the user's home directory from a test.
* Fixtures: ``repo_root``, ``scratch_dir``, ``scratch_assets``, ``scratch_path``, ``tmp_config``,
  ``make_tmp_config``.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')  # children of tests (hyak/__init__ and this file are cached already)

# huggingface_hub evaluates HF_HOME, HF_HUB_CACHE and HF_HUB_OFFLINE when it is imported, which test modules do
# at collection time (transformers): a fixture is too late for the test process itself.
_HF_SESSION = tempfile.mkdtemp(prefix='hyak-tests-hf-')
atexit.register(shutil.rmtree, _HF_SESSION, ignore_errors=True)
os.environ['HF_HOME'] = _HF_SESSION
os.environ['HF_HUB_CACHE'] = str(Path(_HF_SESSION) / 'hub')
if os.environ.get('HYAK_TEST_NETWORK') != '1':
    os.environ['HF_HUB_OFFLINE'] = '1'
else:
    os.environ.pop('HF_HUB_OFFLINE', None)

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hyak.common import slurm  # noqa: E402
from hyak.common.config import STAGE_OVERRIDE_KEYS, parse_env_file  # noqa: E402
from hyak.common.paths import MONITOR_ROOT_ENV, Paths  # noqa: E402

CONFIG_KEYS = tuple(parse_env_file(REPO / 'hyak' / 'config.env.example'))
_ISOLATED = set(CONFIG_KEYS) | set(STAGE_OVERRIDE_KEYS) | {
    'HYAK_CONFIG', 'HYAK_STAGE', 'HYAK_RUN', 'HYAK_ENV', 'HYAK_CONFIG_FILE', 'HYAK_TERM_RECEIVED', 'HYAK_REPO',
    'HYAK_PYTHON'}


def pytest_configure(config):
    config.addinivalue_line('markers', 'network: needs internet access; runs only with HYAK_TEST_NETWORK=1')
    config.addinivalue_line('markers', 'slow: long or memory-heavy; skipped with HYAK_TEST_SKIP_SLOW=1')


def pytest_collection_modifyitems(config, items):
    network = os.environ.get('HYAK_TEST_NETWORK') == '1'
    skip_slow = os.environ.get('HYAK_TEST_SKIP_SLOW') == '1'
    for item in items:
        if 'network' in item.keywords and not network:
            item.add_marker(pytest.mark.skip(reason='network test; set HYAK_TEST_NETWORK=1 to run'))
        if 'slow' in item.keywords and skip_slow:
            item.add_marker(pytest.mark.skip(reason='slow test skipped by HYAK_TEST_SKIP_SLOW=1'))


@pytest.fixture(autouse=True)
def _hermetic_hyak_env(monkeypatch, tmp_path_factory):
    for key in list(os.environ):
        if key in _ISOLATED or key.startswith('SLURM_') or key == 'SLURMD_NODENAME':
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(MONITOR_ROOT_ENV, str(tmp_path_factory.mktemp('monitor')))
    monkeypatch.setenv('HYAK_CONFIG', str(tmp_path_factory.mktemp('no-config') / 'absent.env'))
    hf = tmp_path_factory.mktemp('hf-home')
    monkeypatch.setenv('HF_HOME', str(hf))
    monkeypatch.setenv('HF_HUB_CACHE', str(hf / 'hub'))
    if os.environ.get('HYAK_TEST_NETWORK') == '1':
        monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    else:
        monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    slurm.reset_job_listing()  # one cached squeue listing per process must not leak between tests
    yield
    slurm.reset_job_listing()


@pytest.fixture
def repo_root() -> Path:
    """The repository checkout under test."""
    return REPO


@pytest.fixture
def scratch_dir() -> Path:
    """Directory of local test assets: ``$HYAK_TEST_SCRATCH``, else the parent of the running venv.

    With the port's local venv (``<scratch>/venv-train``) this is the scratch directory that
    holds ``smollm2/`` and ``mini-corpus/``.  Tests must check that what they need exists.
    """
    value = os.environ.get('HYAK_TEST_SCRATCH')
    return Path(value) if value else Path(sys.prefix).parent


@pytest.fixture
def scratch_assets(scratch_dir) -> Path:
    """SmolLM2 test assets (``full/{config.json,tokenizer.json}``, ``embed_only/model.safetensors``).

    ``$HYAK_TEST_ASSETS`` if set, else ``<scratch_dir>/smollm2``; skips the test when absent.
    """
    value = os.environ.get('HYAK_TEST_ASSETS')
    path = Path(value) if value else scratch_dir / 'smollm2'
    if not path.is_dir():
        pytest.skip(f'test assets not found at {path} (set HYAK_TEST_ASSETS)')
    return path


@pytest.fixture
def scratch_path(scratch_dir):
    """Factory ``scratch_path(*parts) -> Path`` of an asset under the scratch directory; skips the test, naming
    the missing path and the variable that points elsewhere, when it does not exist (the builders' laptop
    assets are absent on the cluster; their tests must skip cleanly, not fail)."""
    def find(*parts) -> Path:
        path = scratch_dir.joinpath(*parts)
        if not path.exists():
            pytest.skip(f'test asset not found at {path} (a laptop-only asset; set HYAK_TEST_SCRATCH)')
        return path
    return find


@dataclass
class TmpConfig:
    """A written config file, its resolved dict and the Paths, all rooted in a temporary directory."""

    cfg: dict
    paths: Paths
    config_file: Path
    base: Path


def _write_config(base: Path, overrides: dict) -> Path:
    values = {'ANYTOK_ROOT': str(base / 'root'), 'ANYTOK_RUNS': str(base / 'runs'), **overrides}
    lines = []
    for key, value in values.items():
        if any(c in value for c in '"\\$`'):
            raise ValueError(f'test config value for {key} cannot be written: {value!r}')
        lines.append(f'{key}="{value}"')
    path = base / 'config.env'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


@pytest.fixture
def make_tmp_config(tmp_path, monkeypatch):
    """Factory ``make_tmp_config(**overrides) -> TmpConfig``.

    Writes ``<tmp>/config.env`` with ANYTOK_ROOT=<tmp>/root, ANYTOK_RUNS=<tmp>/runs (both
    created) plus ``overrides``, sets ``HYAK_CONFIG`` to it and ``HYAK_MONITOR_ROOT`` to
    ``<tmp>/monitor``, and returns the resolved config and Paths.
    """
    from hyak.common.config import load_config

    def make(**overrides) -> TmpConfig:
        file = _write_config(tmp_path, {k: str(v) for k, v in overrides.items()})
        monkeypatch.setenv('HYAK_CONFIG', str(file))
        monkeypatch.setenv(MONITOR_ROOT_ENV, str(tmp_path / 'monitor'))
        cfg = load_config()
        paths = Paths.from_config(cfg)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.runs.mkdir(parents=True, exist_ok=True)
        return TmpConfig(cfg=cfg, paths=paths, config_file=file, base=tmp_path)

    return make


@pytest.fixture
def tmp_config(make_tmp_config) -> TmpConfig:
    """Default ``make_tmp_config()``: ``.cfg`` (dict), ``.paths`` (Paths), ``.config_file``, ``.base``."""
    return make_tmp_config()
