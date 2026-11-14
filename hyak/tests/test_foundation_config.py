"""hyak.common.config and hyak.common.paths: grammar, resolution order, validation, layout."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hyak.common import config as C
from hyak.common import paths as P

DESIGN_DEFAULTS = {
    'ANYTOK_ROOT': '/gpfs/projects/CHANGE_ME/anytok', 'ANYTOK_RUNS': '/gpfs/scrubbed/CHANGE_ME/anytok-runs',
    'HYAK_PARTITION': 'gpu-h200', 'HYAK_QOS': 'normal', 'HYAK_QOS_DEBUG': 'debug', 'HYAK_ACCOUNT': '',
    'HYAK_GRES': 'gpu:1', 'HYAK_CPUS': '8', 'HYAK_MEM': '160G', 'HYAK_TIME': '24:00:00',
    'HYAK_TIME_DEBUG': '01:00:00', 'HYAK_MODULES': 'conda', 'HYAK_SEGMENT_MARGIN_MINUTES': '45',
    'HYAK_HN_CHUNK_ROWS': '16384', 'HYAK_POOL_WORKERS': '2', 'HYAK_POOL_THREADS': '4',
    'HYAK_CORPUS_WORKERS': '6', 'HYAK_PLATFORM': 'Hyak Tillicum · H200', 'HF_TOKEN': '',
    'HYAK_TORCH_INDEX_URL': '',
}


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding='utf-8')
    return path


def example_defaults(repo_root):
    return C.parse_env_file(repo_root / 'hyak' / 'config.env.example')


def test_example_defaults_match_design(repo_root):
    defaults = example_defaults(repo_root)
    # per-stage overrides (short limits for short stages ...) are the example's own tuning, not design constants
    stage_defaults = {k: defaults.pop(k) for k in list(defaults) if C.is_stage_override_key(k)}
    assert all(v for v in stage_defaults.values())
    # the example storage path is a placeholder that must be edited (and lies outside any checkout); the
    # memory default is either the cluster's per-GPU default (empty: no --mem flag) or an explicit size
    root = defaults.pop('ANYTOK_ROOT')
    assert root.startswith('/gpfs/projects/') and 'CHANGE_ME' in root
    assert defaults.pop('HYAK_MEM') in ('', '160G')
    assert defaults == {k: v for k, v in DESIGN_DEFAULTS.items() if k not in ('ANYTOK_ROOT', 'HYAK_MEM')}


def test_gitignore_hides_config_env(repo_root):
    result = subprocess.run(['git', 'check-ignore', '-v', 'hyak/config.env'], cwd=repo_root,
                            capture_output=True, text=True)
    if result.returncode == 128:
        pytest.skip('not a git checkout')
    assert result.returncode == 0 and 'hyak/.gitignore' in result.stdout
    tracked = subprocess.run(['git', 'check-ignore', 'hyak/config.env.example'], cwd=repo_root,
                             capture_output=True, text=True)
    assert tracked.returncode == 1


def test_grammar(tmp_path):
    file = write(tmp_path / 'c.env', '\n'.join([
        '# comment', '   ', 'A=plain/value:1,2@x%y+z=w-.', 'B="two words · ok"  # trailing',
        "C='single $ \\ quoted'", 'export D=exported', '  E=', 'F= # empty with comment', 'G=""',
    ]) + '\n')
    assert C.parse_env_file(file) == {'A': 'plain/value:1,2@x%y+z=w-.', 'B': 'two words · ok',
                                      'C': 'single $ \\ quoted', 'D': 'exported', 'E': '', 'F': '', 'G': ''}


@pytest.mark.parametrize('line', [
    'A=two words', 'A="has $var"', 'A="back\\slash"', 'A="tick`"', 'A=abc#nospace', 'A="x"#c',
    'A = spaced', 'A=~/home', "A='unterminated", '1A=x', 'A=x\r', 'just text', 'A=$(cmd)',
])
def test_grammar_rejects(tmp_path, line):
    file = write(tmp_path / 'c.env', line + '\n')
    with pytest.raises(C.ConfigError) as err:
        C.parse_env_file(file)
    assert str(file) in str(err.value) and ':1' in str(err.value)


def test_duplicate_key_rejected(tmp_path):
    file = write(tmp_path / 'c.env', 'A=1\nA=2\n')
    with pytest.raises(C.ConfigError, match='duplicate key A'):
        C.parse_env_file(file)


def test_no_trailing_newline(tmp_path):
    file = tmp_path / 'c.env'
    file.write_bytes(b'A=1\nB=2')
    assert C.parse_env_file(file) == {'A': '1', 'B': '2'}


def test_resolution_order(make_tmp_config, monkeypatch, repo_root):
    t = make_tmp_config(HYAK_CPUS='4', HYAK_TIME_POOL='12:00:00')
    cfg = t.cfg
    assert cfg['HYAK_CPUS'] == '4'                       # file over default
    assert cfg['HYAK_MEM'] == example_defaults(repo_root)['HYAK_MEM']   # default
    assert cfg['HYAK_TIME_POOL'] == '12:00:00'           # per-stage key from the file
    monkeypatch.setenv('HYAK_CPUS', '2')
    monkeypatch.setenv('HYAK_MEM', '')                   # set-but-empty environment value wins too
    monkeypatch.setenv('HYAK_GRES_CORPUS', 'gpu:2')      # per-stage key from the environment
    cfg = C.load_config()
    assert cfg['HYAK_CPUS'] == '2' and cfg['HYAK_MEM'] == '' and cfg['HYAK_GRES_CORPUS'] == 'gpu:2'
    assert set(DESIGN_DEFAULTS) <= set(cfg)


def test_missing_file_message(monkeypatch, tmp_path):
    monkeypatch.setenv('HYAK_CONFIG', str(tmp_path / 'absent.env'))
    with pytest.raises(C.ConfigError, match='not found. Remedy: cp'):
        C.load_config()


def test_placeholder_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv('HYAK_CONFIG', str(write(tmp_path / 'c.env', 'HYAK_CPUS=8\n')))
    with pytest.raises(C.ConfigError) as err:
        C.load_config()
    assert 'ANYTOK_ROOT, ANYTOK_RUNS still contain CHANGE_ME' in str(err.value)


def test_unknown_key_rejected(make_tmp_config):
    with pytest.raises(C.ConfigError, match='unknown key.*HYAK_PARTITON_POOL'):
        make_tmp_config(HYAK_PARTITON_POOL='x')
    with pytest.raises(C.ConfigError, match='unknown key.*HYAK_TIME_TRAIN'):
        make_tmp_config(HYAK_TIME_TRAIN='1:00:00')


@pytest.mark.parametrize('key,value,needle', [
    ('ANYTOK_ROOT', 'relative/path', 'must be an absolute path'),
    ('HYAK_CPUS', 'eight', 'HYAK_CPUS'),
    ('HYAK_CPUS', '0', 'HYAK_CPUS'),
    ('HYAK_TIME', '25 hours', 'HYAK_TIME'),
    ('HYAK_TIME', '0', 'non-zero'),
    ('HYAK_TIME_DEBUG', 'UNLIMITED', 'finite'),
    ('HYAK_MEM', '160GB', 'HYAK_MEM'),
    ('HYAK_GRES', 'gpu 1', 'HYAK_GRES'),
    ('HYAK_SEGMENT_MARGIN_MINUTES', '-5', 'HYAK_SEGMENT_MARGIN_MINUTES'),
    ('HYAK_HN_CHUNK_ROWS', '16k', 'HYAK_HN_CHUNK_ROWS'),
    ('HYAK_HN_CHUNK_ROWS', '4095', 'HYAK_HN_CHUNK_ROWS'),
    ('HYAK_HN_CHUNK_ROWS', '0', 'HYAK_HN_CHUNK_ROWS'),
    ('HYAK_POOL_WORKERS', '0', 'HYAK_POOL_WORKERS'),
    ('HYAK_PLATFORM', ' ', 'HYAK_PLATFORM'),
    ('HYAK_TORCH_INDEX_URL', 'download.pytorch.org', 'HYAK_TORCH_INDEX_URL'),
    ('HYAK_TIME_POOL', '3 days', 'HYAK_TIME_POOL'),
    ('HYAK_CPUS_PROBE', 'many', 'HYAK_CPUS_PROBE'),
])
def test_invalid_values(make_tmp_config, key, value, needle):
    with pytest.raises(C.ConfigError) as err:
        make_tmp_config(**{key: value})
    assert needle in str(err.value) and 'Remedy' in str(err.value)


def test_none_values_accepted(make_tmp_config):
    cfg = make_tmp_config(HYAK_ACCOUNT='none', HYAK_GRES_POOL='none', HYAK_CPUS_POOL='none').cfg
    assert cfg['HYAK_GRES_POOL'] == 'none'


def test_chunk_rows_floor_is_4096(make_tmp_config):
    assert make_tmp_config(HYAK_HN_CHUNK_ROWS='4096').cfg['HYAK_HN_CHUNK_ROWS'] == '4096'
    assert C.MIN_HN_CHUNK_ROWS == 4096


def test_empty_memory_means_no_mem_flag(make_tmp_config):
    """An empty HYAK_MEM (the cluster's per-GPU default memory) and empty / 'none' per-stage values are valid."""
    from hyak.common.slurm import sbatch_resources
    t = make_tmp_config(HYAK_MEM='', HYAK_MEM_POOL='none', HYAK_MEM_CORPUS='64G')
    assert sbatch_resources(t.cfg, 'train-uniform')['mem'] is None
    assert sbatch_resources(t.cfg, 'pool')['mem'] is None
    assert sbatch_resources(t.cfg, 'corpus')['mem'] == '64G'
    t = make_tmp_config(HYAK_MEM_POOL='')              # an empty per-stage value falls back to the global one
    assert sbatch_resources(t.cfg, 'pool')['mem'] == (t.cfg['HYAK_MEM'] or None)


@pytest.mark.parametrize('value,needle', [
    ('/gpfs/data/anytok/', 'normalised'), ('/gpfs//data', 'normalised'), ('//gpfs/data', 'normalised'),
    ('/gpfs/./data', 'normalised'), ('/gpfs/data/.', 'normalised'), ('/gpfs/x/../data', 'normalised'),
    ('/gpfs/data/..', 'normalised'),
])
@pytest.mark.parametrize('key', ['ANYTOK_ROOT', 'ANYTOK_RUNS'])
def test_storage_paths_must_be_normalised(make_tmp_config, key, value, needle):
    """slurm-3: conda normalises the prefix it is given, so any other spelling fails every job at activation."""
    with pytest.raises(C.ConfigError) as err:
        make_tmp_config(**{key: value})
    assert needle in str(err.value) and key in str(err.value) and 'Remedy' in str(err.value)
    assert repr(os.path.normpath(value)) in str(err.value) or value.startswith('//')


def test_storage_paths_may_not_lie_inside_the_checkout(make_tmp_config, repo_root, monkeypatch):
    """readme-v2: environments, corpus and checkpoints must not land in the git working tree."""
    for key in ('ANYTOK_ROOT', 'ANYTOK_RUNS'):
        for value in (str(repo_root), str(repo_root / 'runs'), str(repo_root / 'hyak' / 'x')):
            with pytest.raises(C.ConfigError, match='inside the repository checkout') as err:
                make_tmp_config(**{key: value})
            assert key in str(err.value)
    # a sibling whose name merely starts with the checkout's name is outside it
    sibling = str(repo_root) + '-data'
    assert make_tmp_config(ANYTOK_ROOT=sibling).cfg['ANYTOK_ROOT'] == sibling
    # HYAK_REPO moves the checkout (tests, relocated clones)
    monkeypatch.setenv('HYAK_REPO', str(repo_root))
    with pytest.raises(C.ConfigError, match='inside the repository checkout'):
        make_tmp_config(ANYTOK_RUNS=str(repo_root / 'hyak'))


def test_stage_value(make_tmp_config):
    cfg = make_tmp_config(HYAK_PLATFORM_POOL='klone · CPU', HYAK_SEGMENT_MARGIN_MINUTES_TRAIN_RL='60').cfg
    assert C.stage_value(cfg, 'HYAK_PLATFORM', 'pool') == 'klone · CPU'
    assert C.stage_value(cfg, 'HYAK_PLATFORM', 'corpus') == 'Hyak Tillicum · H200'
    assert C.int_value(cfg, 'HYAK_SEGMENT_MARGIN_MINUTES', 'train-rl') == 60
    assert C.int_value(cfg, 'HYAK_SEGMENT_MARGIN_MINUTES', 'train-uniform') == 45
    assert C.int_value(cfg, 'HYAK_HN_CHUNK_ROWS') == 16384
    with pytest.raises(ValueError):
        C.stage_value(cfg, 'HYAK_HN_CHUNK_ROWS', 'pool')
    with pytest.raises(ValueError):
        C.stage_value(cfg, 'HYAK_PLATFORM', 'not-a-stage')


def test_secrets_redacted(make_tmp_config, monkeypatch):
    cfg = make_tmp_config(HF_TOKEN='hf_secretvalue123').cfg
    assert C.redacted(cfg)['HF_TOKEN'] == '***'
    assert C.redacted({**cfg, 'HF_TOKEN': ''})['HF_TOKEN'] == ''
    assert C.redact_text('auth hf_secretvalue123 failed', cfg) == 'auth *** failed'
    out = subprocess.run([sys.executable, '-m', 'hyak.common.config', '--print'], capture_output=True, text=True,
                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, cwd=C.repo_root())
    assert out.returncode == 0, out.stderr
    assert 'hf_secretvalue123' not in out.stdout and 'HF_TOKEN=***' in out.stdout


def test_repo_root(monkeypatch, tmp_path, repo_root):
    assert C.repo_root() == repo_root
    monkeypatch.setenv('HYAK_REPO', str(tmp_path))
    assert C.repo_root() == tmp_path
    monkeypatch.setenv('HYAK_REPO', 'relative')
    with pytest.raises(C.ConfigError):
        C.repo_root()


def test_cli(tmp_config):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    run = lambda *a: subprocess.run([sys.executable, '-m', 'hyak.common.config', *a], capture_output=True,
                                    text=True, env=env, cwd=C.repo_root())
    assert run('--check').returncode == 0
    assert run('--get', 'HYAK_MEM').stdout.strip() == example_defaults(C.repo_root())['HYAK_MEM']
    resources = json.loads(run('--resources', 'probe').stdout)
    assert resources['qos'] == 'debug' and resources['time'] == '01:00:00'
    bad = run('--get', 'NOPE')
    assert bad.returncode == 2 and 'Unknown key' in bad.stderr          # an input error is 2, never 1


# --- paths -------------------------------------------------------------------------------------------
def test_paths_layout(tmp_config, repo_root, tmp_path):
    p, root, runs = tmp_config.paths, tmp_path / 'root', tmp_path / 'runs'
    assert p.root == root and p.runs == runs and p.repo == repo_root
    expected = {
        p.env_train: root / 'env/train', p.env_data: root / 'env/data',
        p.env_freeze('train'): root / 'env/train.freeze.txt', p.toolchain: root / 'toolchain',
        p.rustup_home: root / 'toolchain/rustup', p.cargo_home: root / 'toolchain/cargo',
        p.tokenizers_src: root / 'build/tokenizers-src', p.rust_utils_build: root / 'build/rust_utils',
        p.wheels: root / 'wheels', p.wheel_provenance: root / 'wheels/provenance.json',
        p.hf_cache: root / 'hf-cache', p.code_dir('uniform'): root / 'code/uniform',
        p.code_dir('rl'): root / 'code/rl', p.code_manifest: root / 'code/code_manifest.json',
        p.corpus_pool_dir('full'): root / 'corpus/full-v1',
        p.corpus_bucket_dir('specialty', 'code-en'): root / 'corpus/specialty-v1/code-en',
        p.recipe: root / 'input/recipe.json', p.pool_recipe: root / 'input/pool-recipe.json',
        p.input_corpus_train('web-en'): root / 'input/corpus/web-en/train.jsonl.gz',
        p.input_corpus_manifest: root / 'input/corpus/manifest.json',
        p.input_eval_file('web-en'): root / 'input/eval/web-en.jsonl.gz',
        p.model_dir: root / 'input/model', p.hypernet_init: root / 'input/hypernet_init',
        p.hypernet_init_manifest: root / 'input/hypernet_init/init_manifest.json',
        p.pool_dir: root / 'input/pool', p.pool_plan: root / 'input/pool/plan.json',
        p.pool_results: root / 'input/pool/full-results.json',
        p.pool_tokenizer_dir(7): root / 'input/pool/tokenizers/0007',
        p.pool_claim_dir(999): root / 'input/pool/claims/0999',
        p.uniform_run_config: root / 'input/uniform_run_config.json',
        p.uniform_heldout: root / 'input/uniform_heldout_tokenizers.json',
        p.pairing: root / 'input/pairing.json', p.input_manifest: root / 'input/manifest.json',
        p.eval_panels: root / 'eval/panels', p.eval_feature_dir(12): root / 'eval/features/0012',
        p.eval_scores_dir('rl'): root / 'eval/scores/rl', p.eval_reports: root / 'eval/reports',
        p.run_dir('uniform'): runs / 'uniform', p.run_dir('rl'): runs / 'rl', p.probe_dir: runs / 'probe',
        p.smoke_dir('rl'): runs / 'smoke/rl',
        p.monitor_dir('pool'): tmp_path / 'monitor/pool',
        p.status_file('train-rl'): tmp_path / 'monitor/train-rl/status.json',
        p.events_file('corpus'): tmp_path / 'monitor/corpus/events.jsonl',
        p.train_log('train-rl', '4242'): tmp_path / 'monitor/train-rl/4242-train.log',
        p.slurm_log('pool', '17'): tmp_path / 'monitor/pool/slurm-17.out',
        p.latest_slurm_log('pool'): tmp_path / 'monitor/pool/latest-slurm.out',
    }
    for actual, wanted in expected.items():
        assert actual == wanted
    assert p.monitor_rel('train-rl') == 'modal_results/hyak_tillicum/train-rl'


def test_paths_default_monitor_root(tmp_config, monkeypatch, repo_root):
    monkeypatch.delenv('HYAK_MONITOR_ROOT')
    p = P.Paths.from_config(tmp_config.cfg)
    assert p.monitor_dir('probe') == repo_root / 'modal_results/hyak_tillicum/probe'


def test_paths_validation(tmp_config):
    p = tmp_config.paths
    for call in (lambda: p.run_dir('other'), lambda: p.code_dir('UNIFORM'), lambda: p.monitor_dir('train'),
                 lambda: p.pool_tokenizer_dir(10000), lambda: p.pool_tokenizer_dir(True),
                 lambda: p.pool_tokenizer_dir('7'), lambda: p.input_eval_file('../x'),
                 lambda: p.corpus_pool_dir('huge'), lambda: p.train_log('train-rl', '../x'),
                 lambda: p.env_prefix('base')):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError, match='absolute'):
        P.Paths.from_config({'ANYTOK_ROOT': 'x', 'ANYTOK_RUNS': '/y'})
    with pytest.raises(Exception):
        p.root = Path('/elsewhere')


def test_paths_normalised_not_resolved(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    (tmp_path / 'link').symlink_to(real)
    p = P.Paths.from_config({'ANYTOK_ROOT': f'{tmp_path}/link//', 'ANYTOK_RUNS': f'{tmp_path}/link/./runs'})
    assert p.root == tmp_path / 'link' and p.runs == tmp_path / 'link' / 'runs'


def test_stage_tables_consistent():
    assert len(set(P.STAGES)) == len(P.STAGES) == 14
    assert set(P.SBATCH_FILES) == set(P.STAGES) == set(P.STAGE_PROGRESS)
    assert set(P.DASHBOARD_RUN_IDS) <= set(P.STAGES)
    assert all(v.startswith('hyak-') for v in P.DASHBOARD_RUN_IDS.values())
    assert len(set(P.DASHBOARD_RUN_IDS.values())) == len(P.DASHBOARD_RUN_IDS)
    assert P.DASHBOARD_RUN_IDS['train-uniform'] == 'hyak-uniform-training'
    assert P.DASHBOARD_RUN_IDS['train-rl'] == 'hyak-rl-training'
    assert P.DASHBOARD_RUN_IDS['probe'] == 'hyak-memory-probe'
    assert P.DASHBOARD_RUN_IDS['corpus'] == 'hyak-corpus'
    assert P.DASHBOARD_RUN_IDS['pool'] == 'hyak-tokenizer-pool'
    assert P.DASHBOARD_RUN_IDS['eval-uniform'] == 'hyak-eval-uniform'
    assert P.DASHBOARD_RUN_IDS['eval-rl'] == 'hyak-eval-rl'
    assert set(P.SINGLETON_STAGES) == {'train-uniform', 'train-rl'}
    assert set(P.DEBUG_STAGES) == {'check-env', 'probe'}
    assert P.SBATCH_FILES['train-rl'] == P.SBATCH_FILES['train-uniform'] == '07_train.sbatch'
    assert P.SBATCH_FILES['eval-rl'] == P.SBATCH_FILES['eval-uniform'] == '09_eval_score.sbatch'
    assert P.STAGE_PROGRESS['train-rl'] == {'completed_field': 'step', 'total': 20000, 'unit': 'steps'}
    assert P.stage_suffix('train-uniform') == 'TRAIN_UNIFORM'
    assert len(C.STAGE_OVERRIDE_KEYS) == len(C.STAGE_OVERRIDE_FLAGS) * len(P.STAGES)


def test_python39_compatible(repo_root, tmp_config):
    """hyak.common and the registry tool must run under the login node's system Python (3.9)."""
    candidates = [c for c in ('/usr/bin/python3',) if Path(c).exists()]
    if not candidates:
        pytest.skip('no system python3')
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(repo_root)}
    for python in candidates:
        version = subprocess.run([python, '-c', 'import sys; print(sys.version_info[:2] >= (3, 9))'],
                                 capture_output=True, text=True)
        if version.stdout.strip() != 'True':
            continue
        for args in (['-m', 'hyak.common.config', '--check'],
                     ['-m', 'hyak.tools.register_dashboard', '--dry-run',
                      '--registry', str(repo_root / 'plans/run-dashboard.json')],
                     ['-c', 'import hyak.common.status, hyak.common.slurm, hyak.common.io, hyak.common.paths']):
            result = subprocess.run([python, *args], capture_output=True, text=True, env=env, cwd=repo_root)
            assert result.returncode == 0, (python, args, result.stderr)


# --- the suite's own isolation (conftest) ---------------------------------------------------------------------
def test_conftest_points_hyak_config_at_an_absent_file(repo_root):
    """tests-v1: code that forgets to configure itself must not read (and act on) the checkout's real
    hyak/config.env, which on the cluster is the user's configuration."""
    configured = Path(os.environ['HYAK_CONFIG'])
    assert not configured.exists() and configured != repo_root / 'hyak' / 'config.env'
    assert C.config_path() == configured
    with pytest.raises(C.ConfigError, match='not found'):
        C.load_config()


def test_conftest_isolates_the_hugging_face_state():
    """tests-12: no test may reach the hub or write into the user's cache."""
    hf_home = Path(os.environ['HF_HOME'])
    assert os.environ['HF_HUB_OFFLINE'] == '1' or os.environ.get('HYAK_TEST_NETWORK') == '1'
    assert hf_home.is_dir() and hf_home != Path.home() / '.cache' / 'huggingface'
    assert Path(os.environ['HF_HUB_CACHE']).parent == hf_home          # per-test cache, in a temporary directory
    constants = pytest.importorskip('huggingface_hub.constants')
    assert Path.home() / '.cache' not in Path(constants.HF_HOME).parents    # fixed at import: a temporary directory
    if os.environ.get('HYAK_TEST_NETWORK') != '1':
        assert constants.HF_HUB_OFFLINE is True


def test_conftest_writes_no_bytecode():
    assert sys.dont_write_bytecode and os.environ['PYTHONDONTWRITEBYTECODE'] == '1'


def test_scratch_path_skips_with_a_reason_when_the_asset_is_absent(request, monkeypatch, tmp_path):
    """tests-6: laptop-only assets are absent on the cluster; their tests skip cleanly and say why."""
    monkeypatch.setenv('HYAK_TEST_SCRATCH', str(tmp_path))
    (tmp_path / 'smollm2').mkdir()
    find = request.getfixturevalue('scratch_path')
    assert find('smollm2') == tmp_path / 'smollm2'
    with pytest.raises(pytest.skip.Exception) as skipped:
        find('mini-corpus')
    assert 'mini-corpus' in str(skipped.value) and 'HYAK_TEST_SCRATCH' in str(skipped.value)
