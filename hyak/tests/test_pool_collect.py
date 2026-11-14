"""hyak.pool.collect on synthetic manifests: refusal rules, write-once index, pinned consumer proof."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import hyak.pool as pool
from hyak.pool.collect import index_text, verify_manifests
from hyak.tests.test_pool_support import make_test_plan, tokenizer_bytes

REPO = Path(__file__).resolve().parents[2]


def identity(plan, plan_path, **changes):
    value = {'plan_sha256': plan['plan_sha256'],
             'plan_file_sha256': hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
             'production_plan': False, 'worker_sha256': pool.WORKER_SHA256, 'rayon_threads': 4,
             'runtime_wheel_sha256': 'a' * 64, 'upstream_revision': pool.UPSTREAM_REVISION,
             'patches': dict(pool.PATCH_SHA256), 'recipe_sha256': 'b' * 64,
             'corpus': {b['name']: {'train_bytes': 10, 'train_sha256': None} for b in plan['recipe']['buckets']}}
    value.update(changes)
    return value


def write_pool(pool_dir: Path, plan: dict, ident: dict, indices=None):
    library = {}
    for config in plan['configs']:
        if indices is not None and config['index'] not in indices:
            continue
        key = (config['algorithm'], config['pretokenizer'], config['vocab_size'])
        if key not in library:
            library[key] = tokenizer_bytes(*key)
        raw = library[key]
        folder = pool_dir / 'tokenizers' / f'{config["index"]:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'tokenizer.json').write_bytes(raw)
        stats = {b['name']: {'quota_bytes': config['bucket_bytes'][b['name']], 'used_bytes': config['bucket_bytes'][b['name']],
                             'documents': 1, 'documents_sha256': '0' * 64} for b in plan['recipe']['buckets']}
        manifest = {'status': 'complete', 'config': config, 'source_stats': stats,
                    'vocab_size_actual': config['vocab_size'], 'train_seconds': 1.0 + config['index'],
                    'elapsed_seconds': 2.0, 'cpu_seconds': 3.0, 'peak_rss_gib': 0.5,
                    'estimated_process_cost_usd': 0.001, 'total_used_bytes': plan['total_bytes_per_tokenizer'] - 3,
                    'validation': {}, 'tokenizer_bytes': len(raw), 'tokenizer_sha256': hashlib.sha256(raw).hexdigest(),
                    'provenance': {**ident, 'host': 'n1', 'attempt': 1}}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


@pytest.fixture(scope='module')
def big(tmp_path_factory):
    """A consistent 1,000-config test plan (vocabulary 300 everywhere) and a complete synthetic pool."""
    base = tmp_path_factory.mktemp('collect')
    plan_path = base / 'plan.json'
    plan = make_test_plan(plan_path, count=1000, total_bytes=23_000, vocab=300)
    return base, plan_path, plan


def collect(tmp_config, plan_path, pool_dir, *extra):
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.collect', '--plan', str(plan_path), '--allow-test-plan',
                             '--pool-dir', str(pool_dir), '--monitor-dir', str(tmp_config.base / 'monitor'), *extra],
                            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    return result.returncode, result.stdout, result.stderr


def test_index_text_is_the_original_format():
    plan = {'plan_sha256': 'p'}
    manifests = {1: {'b': 2, 'a': [1]}, 0: {'z': 'é'}}
    # scripts/run_tokenizer_pool_farm.py:194-196 verbatim:
    original = json.dumps({"plan_sha256": plan["plan_sha256"], "completed": len(manifests),
                           "tokenizers": [manifests[i] for i in sorted(manifests)]}, indent=2) + "\n"
    assert index_text(plan, manifests) == original


def test_complete_pool_is_written_once_and_loads_with_the_pinned_classes(big, tmp_config):
    base, plan_path, plan = big
    pool_dir = tmp_config.base / 'pool'
    write_pool(pool_dir, plan, identity(plan, plan_path))
    code, out, err = collect(tmp_config, plan_path, pool_dir, '--check-only')
    assert code == 0, err
    assert not (pool_dir / 'full-results.json').exists() and not (pool_dir / 'plan.json').exists()
    code, out, err = collect(tmp_config, plan_path, pool_dir)
    assert code == 0, err
    target = pool_dir / 'full-results.json'
    data = json.loads(target.read_text())
    assert data['completed'] == 1000 and data['plan_sha256'] == plan['plan_sha256']
    assert [m['config']['index'] for m in data['tokenizers']] == list(range(1000))
    assert (pool_dir / 'plan.json').read_bytes() == plan_path.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o444
    report = json.loads((tmp_config.base / 'monitor' / 'collect-report.json').read_text())
    consumers = report['consumers']
    assert consumers['loaded'] == 1000 and consumers['feature_shape'] == [1000, 18]
    assert consumers['index_sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not list(pool_dir.glob('.collect-proof-*')) and not list(pool_dir.glob('.full-results.json.*'))
    before = (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes())
    code, out, err = collect(tmp_config, plan_path, pool_dir)
    assert code == 0, err
    assert 'never rewritten' in out and consumers['index_sha256'] in out
    assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == before
    # A manifest edited after the index was written: refused loudly, index untouched.
    path = pool_dir / 'tokenizers' / '0007' / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['train_seconds'] += 1
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    code, out, err = collect(tmp_config, plan_path, pool_dir)
    assert code == 4 and 'differs' in err
    assert target.read_bytes() == before[2]


def test_incomplete_pool_is_refused(big, tmp_config):
    base, plan_path, plan = big
    pool_dir = tmp_config.base / 'pool'
    write_pool(pool_dir, plan, identity(plan, plan_path), indices=set(range(999)))
    code, out, err = collect(tmp_config, plan_path, pool_dir)
    assert code == 75 and 'only 999 of 1000' in err and '0999: no manifest.json' in err   # not complete yet
    assert not (pool_dir / 'full-results.json').exists()
    code, out, err = collect(tmp_config, plan_path, pool_dir, '--check-only')
    assert code == 75
    manifest = pool_dir / 'tokenizers' / '0005' / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['vocab_size_actual'] -= 1
    manifest.write_text(json.dumps(data, indent=2) + '\n')
    code, out, err = collect(tmp_config, plan_path, pool_dir, '--check-only')
    assert code == 4 and '0005' in err                                          # invalid, not just incomplete


@pytest.mark.parametrize('damage', ['vocab', 'slack', 'hash', 'identity', 'config', 'status'])
def test_inconsistent_manifests_are_refused(big, tmp_path, damage):
    base, plan_path, plan = big
    pool_dir = tmp_path / 'pool'
    write_pool(pool_dir, plan, identity(plan, plan_path), indices={0, 1, 2, 3})
    small = {**plan, 'configs': plan['configs'][:4]}
    folder = pool_dir / 'tokenizers' / '0002'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if damage == 'vocab':
        manifest['vocab_size_actual'] -= 1
    elif damage == 'slack':
        manifest['total_used_bytes'] = plan['total_bytes_per_tokenizer'] - 70
    elif damage == 'hash':
        (folder / 'tokenizer.json').write_bytes((folder / 'tokenizer.json').read_bytes().replace(b'tok1"', b'tok9"', 1))
    elif damage == 'identity':
        manifest['provenance']['runtime_wheel_sha256'] = 'c' * 64
    elif damage == 'config':
        manifest['config'] = {**manifest['config'], 'seed': manifest['config']['seed'] + 1}
    else:
        manifest['status'] = 'failed'
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifests, problems, summary = verify_manifests(small, pool_dir / 'tokenizers', require_production_identity=False)
    assert set(problems) == {2} and set(manifests) == {0, 1, 3}
    if damage == 'identity':
        assert 'runtime_wheel_sha256' in problems[2][0]


def test_production_identity_is_required_for_the_production_plan(big, tmp_path):
    base, plan_path, plan = big
    pool_dir = tmp_path / 'pool'
    write_pool(pool_dir, plan, identity(plan, plan_path), indices={0})
    small = {**plan, 'configs': plan['configs'][:1]}
    _, problems, _ = verify_manifests(small, pool_dir / 'tokenizers', require_production_identity=True)
    assert 'production worker' in problems[0][0]


def test_a_test_plan_with_fewer_than_1000_tokenizers_never_gets_an_index(tmp_config):
    plan_path = tmp_config.base / 'small-plan.json'
    plan = make_test_plan(plan_path, count=3, vocab=300)
    pool_dir = tmp_config.base / 'small-pool'
    write_pool(pool_dir, plan, identity(plan, plan_path))
    code, out, err = collect(tmp_config, plan_path, pool_dir, '--check-only')
    assert code == 0, err
    code, out, err = collect(tmp_config, plan_path, pool_dir)
    assert code == 4 and 'exactly 1000' in err
    assert not (pool_dir / 'full-results.json').exists()


def test_existing_different_plan_json_is_refused(big, tmp_config):
    base, plan_path, plan = big
    pool_dir = tmp_config.base / 'pool'
    write_pool(pool_dir, plan, identity(plan, plan_path))
    (pool_dir / 'plan.json').write_text('{}')
    code, out, err = collect(tmp_config, plan_path, pool_dir)
    assert code == 2 and 'byte copy' in err
    assert not (pool_dir / 'full-results.json').exists()


def test_policy_features_of_the_real_plan_match_the_original_run():
    spec = importlib.util.spec_from_file_location('policy_for_test', REPO / pool.SETTINGS_POLICY_REL)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    plan, _ = pool.load_plan(REPO / pool.PLAN_REL)
    configs = plan['configs']
    assert policy.settings_features(configs).shape == pool.FEATURE_SHAPE
    bandit = policy.SettingsTokenizerBandit(configs)
    assert bandit.settings['feature_sha256'] == pool.FEATURE_SHA256
    assert bandit.sample() == pool.FIRST_POLICY_ACTION


def test_production_collect_refuses_without_manifests(tmp_config):
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.collect'], cwd=str(REPO), env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 75 and 'only 0 of 1000' in result.stderr
    assert not tmp_config.paths.pool_results.exists()


def test_publish_once_never_replaces(tmp_path):
    from hyak.pool.collect import _publish_once
    target = tmp_path / 'full-results.json'
    assert _publish_once(target, 'abc\n') is True
    assert target.read_text() == 'abc\n' and target.stat().st_mode & 0o777 == 0o444
    assert _publish_once(target, 'abc\n') is False             # identical: another job won the race
    with pytest.raises(pool.PoolError, match='different content'):
        _publish_once(target, 'xyz\n')
    assert target.read_text() == 'abc\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['full-results.json']


@pytest.mark.parametrize('signals', [1, 2])
def test_collect_and_the_forwarded_stop_signal(big, tmp_config, monkeypatch, capsys, signals):
    """pool-7: the first SIGTERM is noted and collect finishes atomically; a second one stops it with 75 and
    nothing published (no index, no proof directory).  Stale proof directories of a killed run are removed."""
    from hyak.pool import collect as module
    base, plan_path, plan = big
    pool_dir = tmp_config.base / f'pool-{signals}'
    write_pool(pool_dir, plan, identity(plan, plan_path))
    stale = pool_dir / '.collect-proof-old'
    stale.mkdir()
    old = time.time() - 3600
    os.utime(stale, (old, old))

    def prove(folder, code_dir, policy_path, *, production):   # the signal arrives while the consumers load the pool
        for _ in range(signals):
            os.kill(os.getpid(), signal.SIGTERM)
        return {'index_sha256': pool.file_sha256(Path(folder) / 'full-results.json')}

    monkeypatch.setattr(module, 'prove_consumers', prove)
    previous = signal.getsignal(signal.SIGTERM)
    code = module.main(['--plan', str(plan_path), '--allow-test-plan', '--pool-dir', str(pool_dir),
                        '--monitor-dir', str(tmp_config.base / 'monitor')])
    err = capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == previous         # handlers restored
    assert not list(pool_dir.glob('.collect-proof-*'))
    if signals == 1:
        assert code == 0, err
        assert 'SIGTERM received: finishing' in err and (pool_dir / 'full-results.json').exists()
    else:
        assert code == 75, err
        assert 'nothing was published' in err and not (pool_dir / 'full-results.json').exists()
