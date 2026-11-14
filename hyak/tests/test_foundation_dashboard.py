"""Dashboard integration: register_dashboard on the real registry, StatusWriter through the real
progress_json adapter, the learning API for hyak-rl-training, and the one-line learning.js gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.request import urlopen

import pytest

from hyak.common.paths import DASHBOARD_RUN_IDS, monitor_rel
from hyak.common.status import StatusWriter
from hyak.tools import register_dashboard as R
from monitoring.state import Reader, registry, snapshot

#: monitoring/web/learning.js as committed at 3de64a5 (Qingyun's gate lists his two RL runs); the port's working
#: copy adds 'hyak-rl-training' and 'hyak-raw-bpb-training' to that list and nothing else.
ORIGINAL_LEARNING_JS_SHA256 = 'dad17d32693beaefaf0fe1ca09464c6cd37676b8f19bffaf9266dcb02ab853fc'
ORIGINAL_GATE = ("    get('learning').hidden = !['anytok-pool-settings-rl-training', 'anytok-pool-regret-rl-training']"
                 ".includes(runId);")
NEW_GATE = ("    get('learning').hidden = !['anytok-pool-settings-rl-training', 'anytok-pool-regret-rl-training', "
            "'hyak-rl-training', 'hyak-raw-bpb-training'].includes(runId);")
MANAGED = list(DASHBOARD_RUN_IDS.values())


@pytest.fixture
def fake_root(tmp_path, repo_root, tmp_config):
    """A dashboard root: copy of the real registry + plan; monitor dirs at the standard repo-relative place."""
    root = tmp_path / 'dash'
    (root / 'plans').mkdir(parents=True)
    # The checkout's registry already holds the hyak entries once any job has run register_dashboard on it
    # (every job script does): start from the registry without them, in the tracked file's own format.
    data = json.loads((repo_root / 'plans/run-dashboard.json').read_text(encoding='utf-8'))
    data['runs'] = [r for r in data['runs'] if r.get('id') not in MANAGED]
    (root / 'plans/run-dashboard.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n',
                                                   encoding='utf-8')
    shutil.copy2(repo_root / 'plans/tokenizer-pool-1000-v1.json', root / 'plans/tokenizer-pool-1000-v1.json')
    return root


def mdir(root, stage):
    return root / monitor_rel(stage)


# --- register_dashboard -------------------------------------------------------------------------------
def test_register_preserves_existing_entries_byte_for_byte(fake_root, repo_root, tmp_config):
    path = fake_root / 'plans/run-dashboard.json'
    original = path.read_text(encoding='utf-8')
    original_runs = json.loads(original)['runs']
    assert len(original_runs) >= 36 and not {r['id'] for r in original_runs} & set(MANAGED)
    entries = R.build_entries(tmp_config.cfg)
    assert R.register(path, entries) is True
    new = path.read_text(encoding='utf-8')
    runs = json.loads(new)['runs']
    # the hyak entries come AFTER every existing one: the dashboard's default view (the first entry) is unchanged
    assert [r['id'] for r in runs[len(original_runs):]] == MANAGED
    assert runs[:len(original_runs)] == original_runs and runs[0] == original_runs[0]
    # text-level: the original array content is a prefix of the new one, unchanged
    _, _, elements = R.scan_runs(original)
    last = elements[-1][1]
    assert new.startswith(original[:last])
    assert new.endswith(original[last:])
    for start, end, _ in elements:
        assert original[start:end] in new
    # canonical formatting is kept, so the file equals json.dumps of its content
    assert new == json.dumps(json.loads(new), indent=2, ensure_ascii=False) + '\n'
    ids = [r['id'] for r in registry(fake_root, path)]
    assert len(ids) == len(set(ids)) == len(original_runs) + len(MANAGED)
    # idempotent: second run changes nothing and does not rewrite the file
    stamp = path.stat().st_mtime_ns
    assert R.register(path, entries) is False
    assert path.read_text(encoding='utf-8') == new and path.stat().st_mtime_ns == stamp
    assert R.register(path, entries, check=True) is False


def test_register_updates_stale_entries_and_keeps_foreign_hyak_ids(fake_root, tmp_config, make_tmp_config):
    path = fake_root / 'plans/run-dashboard.json'
    data = json.loads(path.read_text())
    data['runs'].insert(3, {'id': 'hyak-rl-training', 'name': 'old', 'adapter': 'progress_json',
                            'platform': 'x', 'status': 'old.json'})
    data['runs'].insert(5, {'id': 'hyak-custom-thing', 'name': 'mine', 'adapter': 'progress_json',
                            'platform': 'x', 'status': 'mine.json'})
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    entries = R.build_entries(make_tmp_config(HYAK_PLATFORM_POOL='klone · CPU').cfg)
    assert R.register(path, entries)
    runs = json.loads(path.read_text())['runs']
    ids = [r['id'] for r in runs]
    assert ids.count('hyak-rl-training') == 1 and ids[-len(MANAGED):] == MANAGED
    assert next(r for r in runs if r['id'] == 'hyak-rl-training')['name'] == \
        'AnyTok · Joint RL 20k · straight BPB reward · entropy 0 · Hyak'
    assert 'hyak-custom-thing' in ids[:-len(MANAGED)] and ids.index('hyak-custom-thing') == 4  # kept in place
    assert 'old' not in [r['name'] for r in runs]
    # registering again is a no-op even though entries had to be moved once
    assert not R.register(path, entries)
    pool = next(r for r in runs if r['id'] == 'hyak-tokenizer-pool')
    assert pool['platform'] == 'klone · CPU'


def test_register_rejects_invalid_registries(tmp_path, tmp_config):
    entries = R.build_entries(tmp_config.cfg)
    for text in ('[]', '{"other": 1}', '{"runs": {}}', '{"runs": [], "runs": []}', '{"runs": [] } x'):
        path = tmp_path / 'r.json'
        path.write_text(text)
        with pytest.raises(ValueError):
            R.register(path, entries)
        assert path.read_text() == text
    path.write_text('{"runs": [{"id": "x", "name": "x", "adapter": "progress_json", "platform": "p", '
                    '"status": "s", "profiles": ["general"]}]}')
    with pytest.raises(ValueError, match='Disabled'):
        R.register(path, entries)


def test_entries_shape(tmp_config):
    entries = {e['id']: e for e in R.build_entries(tmp_config.cfg)}
    assert list(entries) == MANAGED
    for e in entries.values():
        # monitoring-2: complete only when the stage says so (training at 20,000 steps is still 'auditing')
        assert e['completion_stage'] == 'complete'
        assert e['adapter'] == 'progress_json' and e['profiles'] == [] and e['app_names'] == []
        assert e['platform'] == 'Hyak Tillicum · H200'
        assert e['status'].startswith('modal_results/hyak_tillicum/') and not Path(e['status']).is_absolute()
        assert all(not Path(p).is_absolute() for p in e['logs'])
    rl = entries['hyak-rl-training']
    assert (rl['completed_field'], rl['total'], rl['unit']) == ('step', 20000, 'steps')
    # D8: like Qingyun's anytok-pool-regret-rl-training entry, the uniform sanity scores (original cuts) are not
    # overlaid on the RL run's (corrected paired cuts): a baseline glob that matches nothing, baseline_required false
    assert rl['learning'] == {'logs': 'modal_results/hyak_tillicum/train-rl/*-train.log',
                              'baseline_logs': 'modal_results/hyak_tillicum/train-rl/no-old-protocol-baseline-*.log',
                              'plan': 'plans/tokenizer-pool-1000-v1.json',
                              'snapshots': 'modal_results/hyak_tillicum/train-rl/policy-monitor.json',
                              'protocol_note': R.RL_PROTOCOL_NOTE, 'baseline_required': False}
    assert 'straight' in rl['name'] and 'entropy 0' in rl['name'] and 'not overlaid' in R.RL_PROTOCOL_NOTE
    regret = next(r for r in json.loads((Path(__file__).resolve().parents[2] / 'plans/run-dashboard.json')
                                        .read_text(encoding='utf-8'))['runs']
                  if r['id'] == 'anytok-pool-regret-rl-training')
    assert set(regret['learning']) == set(rl['learning']) and regret['learning']['baseline_required'] is False
    assert 'learning' not in entries['hyak-uniform-training']
    assert entries['hyak-tokenizer-pool']['total'] == 1000 and entries['hyak-corpus']['total'] == 23
    for name in ('hyak-eval-uniform', 'hyak-eval-rl', 'hyak-eval-prepare', 'hyak-memory-probe'):
        assert entries[name]['total_field'] == 'total' and 'total' not in entries[name]
    for name in ('hyak-uniform-training', 'hyak-rl-training', 'hyak-tokenizer-pool', 'hyak-corpus'):
        assert 'total_field' not in entries[name]


def test_cli_modes(fake_root, tmp_config):
    path = fake_root / 'plans/run-dashboard.json'
    before = path.read_bytes()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    run = lambda *a: subprocess.run([sys.executable, '-m', 'hyak.tools.register_dashboard', '--registry',
                                     str(path), *a], capture_output=True, text=True, env=env)
    dry = run('--dry-run')
    assert dry.returncode == 0 and [e['id'] for e in json.loads(dry.stdout)] == MANAGED
    assert 'would change' in dry.stderr and path.read_bytes() == before
    assert run('--check').returncode == 4 and path.read_bytes() == before   # 4 = a check failed, never 1
    assert run().returncode == 0 and path.read_bytes() != before
    assert run('--check').returncode == 0
    assert 'unchanged' in run().stdout
    broken = dict(env, HYAK_CONFIG=str(fake_root / 'missing.env'))
    fail = subprocess.run([sys.executable, '-m', 'hyak.tools.register_dashboard', '--registry', str(path)],
                          capture_output=True, text=True, env=broken)
    assert fail.returncode == 2 and 'not found' in fail.stderr


# --- StatusWriter through the real adapter ---------------------------------------------------------------
@pytest.fixture
def registered(fake_root, tmp_config):
    path = fake_root / 'plans/run-dashboard.json'
    R.register(path, R.build_entries(tmp_config.cfg))
    runs = {r['id']: r for r in registry(fake_root, path)}
    return fake_root, runs


def test_training_status_through_adapter(registered, monkeypatch):
    root, runs = registered
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'tillicum')
    monkeypatch.setenv('SLURM_JOB_PARTITION', 'gpu-h200')
    monkeypatch.setattr('hyak.common.status.slurm.is_job_alive', lambda job, timeout=20.0: True)
    w = StatusWriter('train-uniform', 20000, 'steps', 'step', platform='Hyak Tillicum · H200',
                     monitor_dir=mdir(root, 'train-uniform'))
    w.update(1234, stage='training', eta_seconds=3600.5, elapsed_seconds=600,
             workers=[w.worker_row(task='Step 1,234 / 20,000', state='Training', eta_seconds=3600)])
    data = snapshot(Reader(root), runs['hyak-uniform-training'])
    assert (data['completed'], data['total'], data['stage'], data['elapsed']) == (1234, 20000, 'training', 600)
    assert data['eta'] == '≈1h 0m · reported by run'
    (worker,) = data['workers']
    assert worker['id'] == '4242' and worker['task'] == 'Step 1,234 / 20,000' and worker['state'] == 'Training'
    assert worker['profile'] == 'tillicum/gpu-h200' and worker['eta'] == '≈1h 0m'
    assert [p['name'] for p in data['profiles']] == ['tillicum/gpu-h200'] and data['profiles'][0]['active'] == 1
    w.update(1300, state='failed', error='CUDA out of memory')
    data = snapshot(Reader(root), runs['hyak-uniform-training'])
    assert data['eta'] == 'Failed' and data['workers'] == [] and data['completed'] == 1300
    w.update(20000, state='complete')
    data = snapshot(Reader(root), runs['hyak-uniform-training'])
    assert data['stage'] == 'complete' and data['eta'] == 'Complete'


def test_a_counter_at_its_total_is_not_complete_until_the_stage_says_so(registered):
    """monitoring-2: auditing / paused / failed at 20,000 steps, 1,000 tokenizers or 23 buckets."""
    root, runs = registered
    train = StatusWriter('train-uniform', 20000, 'steps', 'step', platform='P', monitor_dir=mdir(root, 'train-uniform'),
                         job='9')
    pool = StatusWriter('pool', 1000, 'tokenizers', platform='P', monitor_dir=mdir(root, 'pool'), job='9')
    corpus = StatusWriter('corpus', 23, 'buckets', platform='P', monitor_dir=mdir(root, 'corpus'), job='9')

    def shown(run_id):
        data = snapshot(Reader(root), runs[run_id])
        return data['stage'], data['eta']
    train.update(9000, stage='training')
    assert shown('hyak-uniform-training')[0] == 'training'
    train.update(20000, stage='auditing')                    # the monitor's stage until final_audit.json is valid
    assert shown('hyak-uniform-training')[0] == 'auditing'
    train.update(20000, stage='paused', state='paused')      # the job ended cleanly before the audit finished
    assert shown('hyak-uniform-training') == ('paused', 'Awaiting measurements')
    train.update(20000, stage='failed', state='failed', error='final audit: checksum mismatch')
    assert shown('hyak-uniform-training') == ('failed', 'Failed')
    pool.update(1000, stage='collect refused', state='failed', error='failed indices')
    assert shown('hyak-tokenizer-pool') == ('collect refused', 'Failed')
    corpus.update(23, stage='failed', state='failed', error='bucket failed')
    assert shown('hyak-corpus') == ('failed', 'Failed')
    train.update(20000, state='complete')
    assert shown('hyak-uniform-training') == ('complete', 'Complete')


def test_pool_worker_rows_show_their_task_text(registered):
    """monitoring-7: a row with an ``index`` key switches the UI to a pool layout that needs other keys."""
    root, runs = registered
    pool = StatusWriter('pool', 1000, 'tokenizers', platform='P', monitor_dir=mdir(root, 'pool'), job='9')
    row = pool.worker_row(task='0413 bpe/regex 96k: training', state='Training', suffix='s0', index=413, attempt=1)
    pool.update(3, stage='generating', workers=[row])
    (worker,) = snapshot(Reader(root), runs['hyak-tokenizer-pool'])['workers']
    assert 'index' not in worker and worker['pool_index'] == 413
    assert worker['task'] == '0413 bpe/regex 96k: training' and worker['state'] == 'Training'


def test_pool_and_eval_status_through_adapter(registered):
    root, runs = registered
    pool = StatusWriter('pool', 1000, 'tokenizers', platform='P', monitor_dir=mdir(root, 'pool'), job='9')
    pool.update(999, stage='generating')
    assert snapshot(Reader(root), runs['hyak-tokenizer-pool'])['stage'] == 'generating'
    pool.update(1000, state='complete')
    data = snapshot(Reader(root), runs['hyak-tokenizer-pool'])
    assert (data['completed'], data['total'], data['stage'], data['unit']) == (1000, 1000, 'complete', 'tokenizers')
    ev = StatusWriter('eval-rl', 1000, 'tokenizers', platform='P', monitor_dir=mdir(root, 'eval-rl'), job='9')
    ev.update(1000, stage='packing')        # completion_stage: not complete until the stage says so
    data = snapshot(Reader(root), runs['hyak-eval-rl'])
    assert (data['completed'], data['total'], data['stage']) == (1000, 1000, 'packing')
    ev.update(1000, state='complete')
    assert snapshot(Reader(root), runs['hyak-eval-rl'])['stage'] == 'complete'
    probe = StatusWriter('probe', None, 'measurements', platform='P', monitor_dir=mdir(root, 'probe'), job='9')
    probe.update(2, total=5, stage='measuring')
    data = snapshot(Reader(root), runs['hyak-memory-probe'])
    assert (data['completed'], data['total']) == (2, 5)
    probe.update(3, state='stopped', error='TERM')
    assert snapshot(Reader(root), runs['hyak-memory-probe'])['eta'] == 'Stopped'


def test_dashboard_monitor_serves_hyak_entries_with_learning(registered, monkeypatch):
    root, _ = registered
    ui = _load_dashboard()
    rl_dir, uni_dir = mdir(root, 'train-rl'), mdir(root, 'train-uniform')
    rl_dir.mkdir(parents=True)
    uni_dir.mkdir(parents=True)
    initial = [{'bytes': 1000, 'examples': 92, 'truncated_forms': 3, 'generated_bpb': 1.5, 'canonical_bpb': 0.9},
               {'bytes': 900, 'examples': 92, 'truncated_forms': 0, 'generated_bpb': 1.7, 'canonical_bpb': 0.95}]
    rl_lines = [{'stage': 'initial_eval', 'results': initial}]
    for step in (1, 2, 3):
        rl_lines.append({'stage': 'train', 'step': step, 'tokenizer_config_index': 652 if step == 1 else step,
                         'policy_effective_tokenizers': 1000.0 - step, 'policy_max_probability': 0.001,
                         'policy_unique_tokenizers': step, 'policy_entropy': 6.9})
    rl_lines.append({'stage': 'eval', 'step': 3, 'results': [{**r, 'generated_bpb': r['generated_bpb'] - 0.1}
                                                            for r in initial]})
    (rl_dir / '4242-train.log').write_text(''.join(json.dumps(x) + '\n' for x in rl_lines))
    (uni_dir / '4100-train.log').write_text(json.dumps({'stage': 'initial_eval', 'results': initial}) + '\n'
                                            + json.dumps({'stage': 'eval', 'step': 2, 'results': initial}) + '\n')
    StatusWriter('train-rl', 20000, 'steps', 'step', platform='P', monitor_dir=rl_dir, job='4242').update(3)
    monitor = ui.Monitor(root, root / 'plans/run-dashboard.json', offline=True)
    assert monitor.state()['id'] == json.loads((root / 'plans/run-dashboard.json').read_text())['runs'][0]['id']
    assert monitor.state()['id'] not in MANAGED                   # registering does not change the default view
    assert monitor.state('hyak-rl-training')['completed'] == 3
    learning = monitor.learning('hyak-rl-training')
    assert learning['latest_step'] == 3 and learning['pool_size'] == 1000
    assert [p['step'] for p in learning['curves']['rl']] == [0, 3]
    assert learning['curves']['uniform'] == []          # D8: the uniform sanity scores are not overlaid
    assert learning['protocol_note'] == R.RL_PROTOCOL_NOTE
    assert learning['curves']['rl'][1]['tokenizers'][0]['gap'] == pytest.approx(1.4 - 0.9)
    assert learning['latest_policy']['policy_unique_tokenizers'] == 3
    assert sum(learning['recent_exposure']['algorithm'].values()) == pytest.approx(1.0)
    assert learning['warnings'] == []
    with pytest.raises(KeyError):
        monitor.learning('hyak-uniform-training')
    server = ui.make_server(monitor, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/learning?run=hyak-rl-training') as response:
            assert response.status == 200 and json.load(response)['latest_step'] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _load_dashboard():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('run_dashboard_hyak_test', root / 'scripts/run_dashboard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- learning.js gate (DESIGN 5.K) ------------------------------------------------------------------------------
def test_learning_js_gate_is_the_only_change(repo_root):
    text = (repo_root / 'monitoring/web/learning.js').read_text(encoding='utf-8')
    assert text.count(NEW_GATE + '\n') == 1 and ORIGINAL_GATE not in text
    restored = text.replace(NEW_GATE + '\n', ORIGINAL_GATE + '\n')
    assert hashlib.sha256(restored.encode('utf-8')).hexdigest() == ORIGINAL_LEARNING_JS_SHA256
