"""hyak.train.monitor and hyak.tools.policy_monitor, checked through the dashboard's own readers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from hyak.common import status as status_module
from hyak.common.io import read_json
from hyak.common.paths import Paths, monitor_rel
from hyak.tools import policy_monitor as PM
from hyak.tools import register_dashboard as REG
from hyak.train import monitor as M
from hyak.train import retention as R
from hyak.tests.test_train_retention import make_checkpoint
from monitoring.learning import LearningReader
from monitoring.state import Reader, registry, snapshot


@pytest.fixture
def dash(tmp_path, repo_root, tmp_config, monkeypatch):
    """A dashboard root (registry copy with the hyak entries) whose monitor dir the Monitor writes."""
    root = tmp_path / 'dash'
    (root / 'plans').mkdir(parents=True)
    shutil.copy2(repo_root / 'plans/run-dashboard.json', root / 'plans/run-dashboard.json')
    shutil.copy2(repo_root / 'plans/tokenizer-pool-1000-v1.json', root / 'plans/tokenizer-pool-1000-v1.json')
    REG.register(root / 'plans/run-dashboard.json', REG.build_entries(tmp_config.cfg))
    runs = {r['id']: r for r in registry(root, root / 'plans/run-dashboard.json')}
    monitor_root = root / 'modal_results/hyak_tillicum'
    monkeypatch.setenv('HYAK_MONITOR_ROOT', str(monitor_root))
    paths = Paths.from_config(tmp_config.cfg, monitor_root=monitor_root)
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'tillicum')
    monkeypatch.setenv('SLURM_JOB_PARTITION', 'gpu-h200')
    monkeypatch.setattr('hyak.common.status.slurm.is_job_alive', lambda job, timeout=20.0: True)
    return root, runs, paths, tmp_config.cfg


def sample_lines(scratch_dir):
    path = scratch_dir / 'public-zett' / 'events.jsonl'
    if not path.is_file():
        pytest.skip(f'{path} (public ZeTT-run events) not available')
    return path.read_text().splitlines()


def test_real_trainer_log_through_the_dashboard(dash, scratch_dir):
    root, runs, paths, cfg = dash
    lines = sample_lines(scratch_dir)
    run_dir = paths.run_dir('uniform')
    run_dir.mkdir(parents=True)
    log = run_dir / 'events.jsonl'
    log.write_text('\n'.join(lines[:96]) + '\n' + lines[96][:20])      # through paused_at 6120 + a partial line
    clock = [1000.0]
    monitor = M.Monitor('uniform', paths, cfg, clock=lambda: clock[0])
    monitor.writer.started_at = 400.0
    status = monitor.refresh(force=True)
    train = [json.loads(l) for l in lines[:96] if json.loads(l)['stage'] == 'train']
    last_per_step = {r['step']: r['seconds'] for r in train}           # a replayed step keeps its last record
    assert len(last_per_step) < len(train)                             # the real log has resume duplicates
    import statistics
    expected_median = statistics.median([last_per_step[s] for s in sorted(last_per_step)[-100:]])
    assert status['step'] == 6100 and status['stage'] == 'segment_paused' and status['state'] == 'running'
    assert status['seconds_per_step'] == pytest.approx(expected_median)
    assert status['eta_seconds'] == pytest.approx(round((20000 - 6100) * status['seconds_per_step'], 1))
    assert status['elapsed_seconds'] == 600.0 and status['peak_gib'] == train[-1]['peak_gib']
    assert status['last_eval']['step'] == 6120 and len(status['initial_eval']) == 2
    data = snapshot(Reader(root), runs['hyak-uniform-training'])
    assert (data['completed'], data['total'], data['stage']) == (6100, 20000, 'segment_paused')
    assert data['eta'].endswith('· reported by run')
    (worker,) = data['workers']
    assert worker['id'] == '4242' and worker['task'] == 'Step 6,100 / 20,000' and worker['profile'] == 'tillicum/gpu-h200'
    # nothing new: status.json is rewritten (fresh at_utc, elapsed) but no event line is appended
    events_before = len((paths.monitor_dir('train-uniform') / 'events.jsonl').read_text().splitlines())
    clock[0] = 1015.0
    again = monitor.refresh()
    assert again['step'] == 6100 and again['elapsed_seconds'] == 615.0
    assert read_json(paths.monitor_dir('train-uniform') / 'status.json')['elapsed_seconds'] == 615.0
    assert len((paths.monitor_dir('train-uniform') / 'events.jsonl').read_text().splitlines()) == events_before
    # the rest of the real log, including steps 1600-1800 replayed after a resume
    with open(log, 'a') as stream:
        stream.write(lines[96][20:] + '\n' + '\n'.join(lines[97:]) + '\n')
    status = monitor.refresh()
    assert status['step'] == 20000 and status['stage'] == 'auditing' and status['state'] == 'running'
    assert monitor.log.seconds[1600] == [json.loads(l) for l in lines if json.loads(l).get('step') == 1600
                                         and json.loads(l)['stage'] == 'train'][-1]['seconds']
    assert status['eta_seconds'] == 0
    names = [e.get('event') for e in map(json.loads, (paths.monitor_dir('train-uniform') / 'events.jsonl').open())]
    assert names.count('trainer_paused_at') == 8 and names.count('trainer_initial_eval') == 1
    assert names.count('trainer_eval') == 47 and all('at_utc' in json.loads(l) for l in
                                                     (paths.monitor_dir('train-uniform') / 'events.jsonl').open())
    # completion needs the newest full restart at 20,000 and a valid grid audit
    for step in R.grid_steps():
        make_checkpoint(run_dir, step, 'uniform', R.FULL if step % 2000 == 0 else R.WEIGHTS_ONLY)
    (run_dir / R.FINAL_AUDIT).write_text(json.dumps({'checkpoint_count': 40, 'final_step': 20000}))
    status = monitor.refresh()
    assert (status['state'], status['stage'], status['workers']) == ('complete', 'complete', [])
    data = snapshot(Reader(root), runs['hyak-uniform-training'])
    assert data['stage'] == 'complete' and data['eta'] == 'Complete'
    final = M.finalize(monitor, 0)
    assert final['state'] == 'complete' and final['step'] == 20000


def test_final_refresh_process_of_the_same_job_keeps_the_elapsed_time(dash, monkeypatch):
    """07_train/11_smoke_train stop the background monitor and run ``monitor --once`` in a NEW process of the
    same job; without $SLURM_JOB_START_TIME its clock starts then, so it must keep what was counted."""
    root, runs, paths, cfg = dash
    monkeypatch.delenv('SLURM_JOB_START_TIME', raising=False)
    run_dir = paths.run_dir('uniform')
    run_dir.mkdir(parents=True)
    (run_dir / 'events.jsonl').write_text(json.dumps({'stage': 'train', 'step': 1, 'seconds': 3.0}) + '\n')
    background = M.Monitor('uniform', paths, cfg, clock=lambda: 1600.0)
    background.writer.started_at = 1000.0
    assert background.refresh(force=True)['elapsed_seconds'] == 600.0
    final = M.Monitor('uniform', paths, cfg, clock=lambda: 1700.0)      # the job's --once, started at 1690
    final.writer.started_at = 1690.0
    assert final.refresh(force=True)['elapsed_seconds'] == 610.0
    assert status_module.finalize('train-uniform', 0, monitor_dir=paths.monitor_dir('train-uniform'))[
        'elapsed_seconds'] == 610.0
    # with $SLURM_JOB_START_TIME every monitor process of a job measures from it (no double counting)
    monkeypatch.setenv('SLURM_JOB_START_TIME', '1000')
    rl_dir = paths.run_dir('rl')
    rl_dir.mkdir(parents=True)
    (rl_dir / 'events.jsonl').write_text(json.dumps({'stage': 'train', 'step': 1, 'seconds': 3.0}) + '\n')
    assert M.Monitor('rl', paths, cfg, clock=lambda: 1600.0).refresh(force=True)['elapsed_seconds'] == 600.0
    assert M.Monitor('rl', paths, cfg, clock=lambda: 1700.0).refresh(force=True)['elapsed_seconds'] == 700.0


def test_finalize_and_cumulative_elapsed(dash, monkeypatch):
    root, runs, paths, cfg = dash
    run_dir = paths.run_dir('rl')
    run_dir.mkdir(parents=True)
    (run_dir / 'events.jsonl').write_text(''.join(json.dumps({'stage': 'train', 'step': s, 'seconds': 3.0,
                                                              'train_bpb': 1.2, 'policy_entropy': 6.9,
                                                              'policy_effective_tokenizers': 990.0,
                                                              'policy_max_probability': 0.0011,
                                                              'policy_unique_tokenizers': s}) + '\n'
                                                  for s in range(1, 31)))
    monitor = M.Monitor('rl', paths, cfg, clock=lambda: 1300.0)
    monitor.writer.started_at = 1000.0
    status = monitor.refresh(force=True)
    assert status['step'] == 30 and status['eta_seconds'] == (20000 - 30) * 3.0
    assert status['policy_effective_tokenizers'] == 990.0 and status['train_bpb'] == 1.2
    assert snapshot(Reader(root), runs['hyak-rl-training'])['workers'][0]['task'] == 'RL step 30 / 20,000'
    final = M.finalize(monitor, 0)
    assert final['state'] == 'paused' and final['step'] == 30 and final['elapsed_seconds'] == 300.0
    # the next job carries the elapsed time on (hyak.common.status start, then a new monitor)
    monkeypatch.setenv('SLURM_JOB_ID', '4243')
    status_module.start('train-rl', monitor_dir=paths.monitor_dir('train-rl'))
    second = M.Monitor('rl', paths, cfg, clock=lambda: 2100.0)
    second.writer.started_at = 2000.0
    assert second.refresh(force=True)['elapsed_seconds'] == 400.0
    log = paths.monitor_dir('train-rl') / 'slurm-4243.out'
    log.write_text('Traceback ...\ntorch.OutOfMemoryError: CUDA out of memory\n')
    failed = M.finalize(second, 1, log)
    assert failed['state'] == 'failed' and 'CUDA out of memory' in failed['error']
    assert snapshot(Reader(root), runs['hyak-rl-training'])['eta'] == 'Failed'
    stopped = M.finalize(M.Monitor('rl', paths, cfg), 143)
    assert stopped['state'] == 'stopped'


def test_rl_policy_refresh_after_each_grid_checkpoint(dash):
    root, runs, paths, cfg = dash
    run_dir = paths.run_dir('rl')
    run_dir.mkdir(parents=True)
    calls = []
    monitor = M.Monitor('rl', paths, cfg, refresh_policy=lambda: calls.append(1))
    monitor.refresh(force=True)
    assert len(calls) == 1                                    # once at start
    with open(run_dir / 'events.jsonl', 'a') as stream:
        for row in ({'stage': 'checkpoint', 'step': 0}, {'stage': 'checkpoint', 'step': 777},
                    {'stage': 'checkpoint', 'step': 500}):
            stream.write(json.dumps(row) + '\n')
    monitor.refresh()
    assert len(calls) == 2
    monitor.refresh()
    assert len(calls) == 2
    with open(run_dir / 'events.jsonl', 'a') as stream:
        stream.write(json.dumps({'stage': 'checkpoint', 'step': 1000}) + '\n')
    monitor.refresh()
    assert len(calls) == 3


def test_policy_refresher_failure_is_recorded(dash, tmp_path):
    """monitoring-5: a collector exit status other than 0 raises, so policy_monitor_failed is written."""
    root, runs, paths, cfg = dash
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/sh\necho "collecting"\necho "policy monitor: HYAK_X is not set (configuration)"\n'
                    'exit "${FAKE_EXIT:-2}"\n')
    fake.chmod(0o755)
    refresh = M.policy_refresher(paths, python=str(fake))
    with pytest.raises(RuntimeError, match=r'exited 2: policy monitor: HYAK_X is not set'):
        refresh()
    log = paths.monitor_dir('train-rl') / 'policy-monitor.log'
    assert 'HYAK_X is not set' in log.read_text()
    paths.run_dir('rl').mkdir(parents=True)
    monitor = M.Monitor('rl', paths, cfg, refresh_policy=refresh)
    monitor.refresh(force=True)                               # the status is still written
    events = [json.loads(line) for line in (paths.monitor_dir('train-rl') / 'events.jsonl').read_text().splitlines()]
    failed = [e for e in events if e.get('event') == 'policy_monitor_failed']
    assert len(failed) == 1 and 'exited 2' in failed[0]['error'] and 'HYAK_X' in failed[0]['error']
    assert (paths.monitor_dir('train-rl') / 'status.json').is_file()


def test_policy_refresher_success_raises_nothing(dash, tmp_path, monkeypatch):
    root, runs, paths, cfg = dash
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/sh\necho "policy panel written"\nexit 0\n')
    fake.chmod(0o755)
    assert M.policy_refresher(paths, python=str(fake))() is None


# --- policy monitor ---------------------------------------------------------------------------------------
def write_policy_checkpoint(run_dir: Path, step: int, *, seed: int = 0):
    import torch
    folder = run_dir / R.step_name(step)
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed + step)
    counts = torch.zeros(1000, dtype=torch.int64)
    picks = torch.randint(0, 1000, (step,), generator=generator)
    counts.index_add_(0, picks, torch.ones(step, dtype=torch.int64))
    torch.save({'network': {}, 'logits': torch.randn(1000, dtype=torch.float64, generator=generator) * 0.01,
                'features': torch.zeros(1000, 18, dtype=torch.float64), 'counts': counts, 'updates': step,
                'settings': {'kind': 'shared_settings_mlp', 'size': 1000}}, folder / 'policy.pt')
    raw = (folder / 'policy.pt').read_bytes()
    (folder / 'manifest.json').write_text(json.dumps({'step': step, 'kind': R.WEIGHTS_ONLY, 'files': {
        'policy.pt': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}}))


def test_policy_monitor_snapshots_and_learning_panel(dash, repo_root):
    root, runs, paths, cfg = dash
    run_dir = paths.run_dir('rl')
    for step in (500, 1000):
        write_policy_checkpoint(run_dir, step)
    (run_dir / 'step-000000').mkdir()
    output = paths.monitor_dir('train-rl') / 'policy-monitor.json'
    plan = repo_root / 'plans/tokenizer-pool-1000-v1.json'
    expected_plan = read_json(plan)['plan_sha256']
    saved = PM.collect(run_dir, output, plan, expected_plan_sha256=expected_plan, profile='tillicum/gpu-h200')
    assert saved['error'] is None and [r['step'] for r in saved['snapshots']] == [500, 1000]
    assert saved['snapshots'][0]['plan_sha256'] == expected_plan and 'groups' in saved['snapshots'][0]
    # a corrupt new checkpoint keeps the previous snapshots and records the error
    write_policy_checkpoint(run_dir, 1500)
    (run_dir / 'step-001500' / 'policy.pt').write_bytes(b'corrupt')
    saved = PM.collect(run_dir, output, plan, expected_plan_sha256=expected_plan)
    assert 'size does not match' in saved['error'] or 'SHA-256' in saved['error']
    assert [r['step'] for r in saved['snapshots']] == [500, 1000]
    # a replayed step (new policy.pt hash) is read again
    shutil.rmtree(run_dir / 'step-001500')
    shutil.rmtree(run_dir / 'step-001000')
    write_policy_checkpoint(run_dir, 1000, seed=7)
    before = {r['step']: r['policy_sha256'] for r in saved['snapshots']}
    saved = PM.collect(run_dir, output, plan, expected_plan_sha256=expected_plan)
    after = {r['step']: r['policy_sha256'] for r in saved['snapshots']}
    assert saved['error'] is None and after[500] == before[500] and after[1000] != before[1000]
    # the learning panel reads the snapshots, the RL train log and the uniform baseline log
    entry = runs['hyak-rl-training']['learning']
    initial = {'stage': 'initial_eval', 'results': [
        {'generated_bpb': 3.0, 'canonical_bpb': 1.06, 'examples': 92, 'bytes': 10452, 'truncated_forms': 596},
        {'generated_bpb': 3.1, 'canonical_bpb': 1.05, 'examples': 92, 'bytes': 10610, 'truncated_forms': 88}]}
    rows = [initial] + [{'stage': 'train', 'step': s, 'tokenizer_config_index': s, 'policy_entropy': 6.9,
                         'policy_effective_tokenizers': 999.0, 'policy_max_probability': 0.001,
                         'policy_unique_tokenizers': s} for s in range(1, 501)]
    rows.append({'stage': 'eval', 'step': 500, 'results': initial['results']})
    (paths.monitor_dir('train-rl') / '4242-train.log').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    paths.monitor_dir('train-uniform').mkdir(parents=True, exist_ok=True)
    (paths.monitor_dir('train-uniform') / '4241-train.log').write_text(
        json.dumps(initial) + '\n' + json.dumps({'stage': 'eval', 'step': 500, 'results': initial['results']}) + '\n')
    data = LearningReader(root).read(entry)
    assert data['warnings'] == [] and data['latest_step'] == 500
    # D8: the uniform run's sanity scores (original cuts) are not overlaid on the RL run's (corrected cuts)
    assert data['curves']['uniform'] == [] and [p['step'] for p in data['curves']['rl']] == [0, 500]
    assert data['protocol_note'] == REG.RL_PROTOCOL_NOTE and entry['baseline_required'] is False
    # even when the uniform initial scores differ (they do: other cuts), no warning and nothing hidden
    (paths.monitor_dir('train-uniform') / '4241-train.log').write_text(json.dumps(
        {'stage': 'initial_eval', 'results': [{**r, 'bytes': r['bytes'] - 37} for r in initial['results']]}) + '\n')
    data = LearningReader(root).read(entry)
    assert data['warnings'] == [] and [p['step'] for p in data['curves']['rl']] == [0, 500]
    assert [s['step'] for s in data['snapshots']] == [500, 1000] and data['snapshot_error'] is None
    assert entry['snapshots'] == f'{monitor_rel("train-rl")}/policy-monitor.json'


def test_policy_monitor_cli_defaults(dash, repo_root, capsys):
    root, runs, paths, cfg = dash
    write_policy_checkpoint(paths.run_dir('rl'), 500)
    paths.pool_dir.mkdir(parents=True)
    shutil.copy2(repo_root / 'plans/tokenizer-pool-1000-v1.json', paths.pool_plan)
    assert PM.main(['--once']) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['snapshots'] == 1
    assert read_json(paths.monitor_dir('train-rl') / 'policy-monitor.json')['snapshots'][0]['step'] == 500


# --- review fix round (owner B) -----------------------------------------------------------------------------------
def _trainer_rows(run_dir: Path, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'events.jsonl').open('a') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def _milestone_counts(paths, stage='train-uniform'):
    counts = {}
    for line in (paths.monitor_dir(stage) / 'events.jsonl').read_text().splitlines():
        name = json.loads(line).get('event', '')
        if name.startswith('trainer_'):
            counts[name] = counts.get(name, 0) + 1
    return counts


def test_milestones_are_reported_once_across_processes_and_jobs(dash, monkeypatch):
    """monitoring-1: a new monitor process (two per job) does not replay earlier milestones."""
    _, _, paths, cfg = dash
    run_dir = paths.run_dir('uniform')
    _trainer_rows(run_dir, [{'stage': 'initial_eval', 'results': []}, {'stage': 'train', 'step': 1, 'seconds': 2.0},
                            {'stage': 'eval', 'step': 1, 'results': []}, {'stage': 'checkpoint', 'step': 1},
                            {'stage': 'paused_at', 'step': 1}])
    for job in ('5001', '5002', '5003'):
        monkeypatch.setenv('SLURM_JOB_ID', job)
        status_module.start('train-uniform', monitor_dir=paths.monitor_dir('train-uniform'))
        M.Monitor('uniform', paths, cfg).refresh(force=True)          # background monitor
        if job == '5002':
            _trainer_rows(run_dir, [{'stage': 'train', 'step': 2, 'seconds': 2.0},
                                    {'stage': 'eval', 'step': 2, 'results': []},
                                    {'stage': 'checkpoint', 'step': 2}, {'stage': 'paused_at', 'step': 2}])
        M.Monitor('uniform', paths, cfg).refresh(force=True)          # the job's final --once
        status_module.finalize('train-uniform', 0, monitor_dir=paths.monitor_dir('train-uniform'))
    assert _milestone_counts(paths) == {'trainer_initial_eval': 1, 'trainer_eval': 2, 'trainer_checkpoint': 2,
                                        'trainer_paused_at': 2}
    # a replaced trainer log (new inode) is reported from its start
    (run_dir / 'events.jsonl').unlink()
    _trainer_rows(run_dir, [{'stage': 'initial_eval', 'results': []}])
    M.Monitor('uniform', paths, cfg).refresh(force=True)
    assert _milestone_counts(paths)['trainer_initial_eval'] == 2


def test_non_finite_trainer_numbers_never_stop_status_updates(dash):
    """monitoring-6: NaN/Infinity in a trainer row become null; the rest of the poll is reported."""
    _, _, paths, cfg = dash
    run_dir = paths.run_dir('uniform')
    run_dir.mkdir(parents=True)
    (run_dir / 'events.jsonl').write_text(
        '{"stage": "train", "step": 500, "seconds": 2.0, "peak_gib": NaN}\n'
        '{"stage": "eval", "step": 500, "results": [{"generated_bpb": NaN, "canonical_bpb": Infinity}]}\n'
        '{"stage": "checkpoint", "step": 500}\n')
    status = M.Monitor('uniform', paths, cfg).refresh(force=True)
    assert status['step'] == 500 and status['peak_gib'] is None
    assert status['last_eval']['results'][0]['generated_bpb'] is None
    text = (paths.monitor_dir('train-uniform') / 'status.json').read_text()
    json.loads(text, parse_constant=lambda name: pytest.fail(f'non-strict JSON constant {name}'))
    events = [json.loads(l) for l in (paths.monitor_dir('train-uniform') / 'events.jsonl').read_text().splitlines()]
    evals = [e for e in events if e.get('event') == 'trainer_eval']
    assert evals[0]['non_finite'] is True and [e for e in events if e.get('event') == 'trainer_checkpoint']
    assert M.Monitor('uniform', paths, cfg).refresh(force=True)['step'] == 500     # a new process starts cleanly


def test_follow_survives_refresh_errors(dash, monkeypatch):
    """monitoring-4: a failing first refresh is logged and retried; follow never raises."""
    _, _, paths, cfg = dash
    monitor = M.Monitor('uniform', paths, cfg)
    calls = []
    real = monitor.refresh

    def flaky(force=False):
        calls.append(force)
        if len(calls) == 1:
            raise PermissionError(13, 'Permission denied', 'events.jsonl')
        return real(force=force)

    monitor.refresh = flaky
    monkeypatch.setattr(M, '_alive', lambda pid: len(calls) < 3)
    M.follow(monitor, interval=0.01, parent_pid=12345)
    assert len(calls) >= 3
    assert (paths.monitor_dir('train-uniform') / 'status.json').is_file()
    names = [json.loads(l).get('event') for l in
             (paths.monitor_dir('train-uniform') / 'events.jsonl').read_text().splitlines()]
    assert names.count('monitor_refresh_failed') == 1 and 'monitor_refresh_recovered' in names
    with pytest.raises(PermissionError):                     # --once still reports the failure (exit 1)
        calls.clear()
        M.follow(monitor, once=True)


def test_rotated_status_restarts_the_elapsed_time(dash):
    """monitoring-8 (monitor side): status.json moved away by run_segment's rotation resets elapsed."""
    _, _, paths, cfg = dash
    stage_dir = paths.monitor_dir('train-rl')
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / 'status.json').write_text(json.dumps({'elapsed_seconds': 86400.0, 'elapsed_job': 'old'}))
    clock = [1000.0]
    monitor = M.Monitor('rl', paths, cfg, clock=lambda: clock[0])
    monitor.writer.started_at = 900.0
    assert monitor.refresh(force=True)['elapsed_seconds'] == 86500.0
    previous = stage_dir / 'previous-T'
    previous.mkdir()
    (stage_dir / 'status.json').rename(previous / 'status.json')
    assert monitor.refresh(force=True)['elapsed_seconds'] == 100.0


def test_smoke_stage_ends_complete_after_smoke_verified(dash):
    """e2e leftover: the smoke-train status ends 'complete', not 'paused', after a verified smoke."""
    _, _, paths, cfg = dash
    smoke = paths.smoke_dir('rl')
    _trainer_rows(smoke, [{'stage': 'train', 'step': 3, 'seconds': 1.0}, {'stage': 'paused_at', 'step': 3}])
    monitor = M.Monitor('rl', paths, cfg, smoke=True)
    assert monitor.refresh(force=True)['state'] == 'running'
    (smoke / M.SMOKE_VERIFIED).write_text('{}')
    status = monitor.refresh(force=True)
    assert (status['state'], status['stage'], status['step']) == ('complete', 'complete', 3)
    final = status_module.finalize('smoke-train', 0, monitor_dir=paths.monitor_dir('smoke-train'))
    assert final['state'] == 'complete'


def test_rl_paired_metrics_reach_the_status_and_the_milestones(dash):
    """D8: the straight-BPB-reward trainer's paired metrics are passed through to status.json and milestone events."""
    _, _, paths, cfg = dash
    run_dir = paths.run_dir('rl')
    paired = {'train_bpb': 1.3, 'policy_reward_bpb': 1.3, 'paired_canonical_bpb': 1.05, 'paired_gap_bpb': 0.25,
              'paired_valid_examples': 128, 'paired_maximum_positions': 126}
    _trainer_rows(run_dir, [{'stage': 'initial_eval', 'results': []},
                            {'stage': 'checkpoint', 'step': 0},
                            {'stage': 'train', 'step': 1, 'seconds': 2.0, **paired},
                            {'stage': 'train', 'step': 2, 'seconds': 2.0, **paired, 'paired_gap_bpb': 0.5,
                             'train_bpb': 1.55, 'policy_reward_bpb': 1.55},
                            {'stage': 'eval', 'step': 2, 'results': []}, {'stage': 'checkpoint', 'step': 2},
                            {'stage': 'paused_at', 'step': 2}])
    status = M.Monitor('rl', paths, cfg).refresh(force=True)
    assert (status['policy_reward_bpb'], status['paired_gap_bpb'], status['paired_canonical_bpb']) == (1.55, 0.5, 1.05)
    assert (status['paired_valid_examples'], status['paired_maximum_positions']) == (128, 126)
    events = [json.loads(line) for line in (paths.monitor_dir('train-rl') / 'events.jsonl').read_text().splitlines()]
    milestones = {(e['event'], e.get('step')): e for e in events if e.get('event', '').startswith('trainer_')}
    assert 'paired' not in milestones[('trainer_initial_eval', None)]
    assert 'paired' not in milestones[('trainer_checkpoint', 0)]
    for name in ('trainer_eval', 'trainer_checkpoint', 'trainer_paused_at'):
        assert milestones[(name, 2)]['paired'] == {'train_step': 2, **paired, 'paired_gap_bpb': 0.5,
                                                   'train_bpb': 1.55, 'policy_reward_bpb': 1.55}
    # a later poll carries the newest train row of earlier polls forward
    _trainer_rows(run_dir, [{'stage': 'checkpoint', 'step': 3}])
    M.Monitor('rl', paths, cfg).refresh(force=True)
    events = [json.loads(line) for line in (paths.monitor_dir('train-rl') / 'events.jsonl').read_text().splitlines()]
    late = [e for e in events if e.get('event') == 'trainer_checkpoint' and e.get('step') == 3]
    assert len(late) == 1 and late[0]['paired']['train_step'] == 2
    # the uniform run's milestones carry no paired block
    _trainer_rows(paths.run_dir('uniform'), [{'stage': 'train', 'step': 1, 'seconds': 2.0},
                                             {'stage': 'checkpoint', 'step': 1}])
    M.Monitor('uniform', paths, cfg).refresh(force=True)
    events = [json.loads(line) for line in (paths.monitor_dir('train-uniform') / 'events.jsonl').read_text().splitlines()]
    assert all('paired' not in e for e in events if e.get('event') == 'trainer_checkpoint')
