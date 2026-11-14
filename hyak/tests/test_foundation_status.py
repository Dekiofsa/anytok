"""hyak.common.status: StatusWriter shape, worker merging, start/finalize, CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hyak.common import slurm
from hyak.common import status as S
from hyak.common.io import read_json


@pytest.fixture
def mon(tmp_path):
    return tmp_path / 'mon'


def writer(mon, stage='pool', total=1000, unit='tokenizers', field='completed', job='111', **kw):
    return S.StatusWriter(stage, total, unit, field, platform='Test · CPU', monitor_dir=mon / stage, job=job, **kw)


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_update_shape(mon, monkeypatch):
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'tillicum')
    monkeypatch.setenv('SLURM_JOB_PARTITION', 'gpu-h200')
    w = writer(mon)
    row = w.worker_row(task='tokenizer 0413', state='Training', eta_seconds=60)
    status = w.update(12, stage='generating', eta_seconds=3600.5, elapsed_seconds=10, workers=[row], per_family={'bpe': 3})
    on_disk = read_json(mon / 'pool/status.json')
    assert on_disk == status
    assert on_disk['completed'] == 12 and on_disk['total'] == 1000 and on_disk['stage'] == 'generating'
    assert on_disk['state'] == 'running' and on_disk['at_utc'].endswith('Z')
    assert on_disk['per_family'] == {'bpe': 3} and on_disk['completed_field'] == 'completed'
    (worker,) = on_disk['workers']
    assert worker['id'] == '111' and worker['profile'] == 'tillicum/gpu-h200' and worker['platform'] == 'Test · CPU'
    assert worker['task'] == 'tokenizer 0413' and worker['state'] == 'Training' and worker['eta_seconds'] == 60
    assert worker['started_at'].endswith('Z') and isinstance(worker['elapsed_seconds'], float)
    # pool-11: the events line is the status without its (growing) worker rows
    assert events(mon / 'pool/events.jsonl') == [{k: v for k, v in on_disk.items() if k != 'workers'}]
    w.update(13, append=False)
    assert len(events(mon / 'pool/events.jsonl')) == 1


def test_step_field_and_defaults(mon):
    w = writer(mon, stage='train-rl', total=20000, unit='steps', field='step')
    status = w.update(5)
    assert status['step'] == 5 and 'completed' not in status and status['stage'] == 'running'
    assert status['workers'][0]['task'] == 'running'
    done = w.update(20000, state='complete')
    assert done['stage'] == 'complete' and done['workers'] == []
    assert not (mon / 'train-rl/workers').exists() or not list((mon / 'train-rl/workers').iterdir())


@pytest.mark.parametrize('kwargs,needle', [
    ({'completed': float('nan')}, 'finite'), ({'completed': -1}, '>= 0'), ({'completed': True}, 'finite'),
    ({'completed': 1, 'state': 'done'}, 'state must be'), ({'completed': 1, 'state': 'complete', 'stage': 'x'}, 'complete'),
    ({'completed': float('inf')}, 'finite'), ({'completed': 1, 'total': float('nan')}, 'finite'),
    ({'completed': 1, 'eta_seconds': 'soon'}, 'number'), ({'completed': 1, 'workers': [{'no': 'id'}]}, 'id'),
    ({'completed': 1, 'at_utc': 'x'}, 'reserved'), ({'completed': 1, 'obj': object()}, 'not valid JSON'),
])
def test_update_rejects_bad_values(mon, kwargs, needle):
    w = writer(mon)
    with pytest.raises(ValueError, match=needle):
        w.update(**kwargs)
    assert not (mon / 'pool/status.json').exists()


def test_non_finite_numbers_are_written_as_null_with_the_raw_string(mon):
    """monitoring-6: one NaN from a trainer or scorer must never stop a status or event write."""
    w = writer(mon)
    row = w.worker_row(task='t', eta_seconds=float('inf'))
    status = w.update(1, eta_seconds=float('nan'), elapsed_seconds=float('-inf'), loss=float('nan'),
                      results=[{'bpb': float('inf'), 'ok': 1.5}, float('nan')], workers=[row])
    on_disk = json.loads((mon / 'pool/status.json').read_text(), parse_constant=lambda c: pytest.fail(c))
    assert on_disk == status
    assert (status['eta_seconds'], status['eta_seconds_raw']) == (None, 'nan')
    assert (status['elapsed_seconds'], status['elapsed_seconds_raw']) == (None, '-inf')
    assert (status['loss'], status['loss_raw']) == (None, 'nan')
    assert status['results'] == [{'bpb': None, 'bpb_raw': 'inf', 'ok': 1.5}, None]
    assert (status['workers'][0]['eta_seconds'], status['workers'][0]['eta_seconds_raw']) == (None, 'inf')
    assert (mon / 'pool/events.jsonl').read_text().count('NaN') == 0
    # the next update works as before (a poisoned status never blocks the following writes)
    assert w.update(2, loss=0.5)['loss'] == 0.5
    ev = w.event({'event': 'eval', 'generated_bpb': float('nan'), 'step': 500})
    assert ev['generated_bpb'] is None and ev['generated_bpb_raw'] == 'nan' and ev['step'] == 500
    last = events(mon / 'pool/events.jsonl')[-1]
    assert last == ev
    assert w.event({'a': [float('inf')], 'b': {'c': float('-inf')}})['b'] == {'c': None, 'c_raw': '-inf'}


def test_constructor_validation(mon):
    with pytest.raises(ValueError):
        writer(mon, stage='nope')
    with pytest.raises(ValueError):
        writer(mon, field='a.b')
    with pytest.raises(ValueError):
        writer(mon, field='workers')
    with pytest.raises(ValueError):
        writer(mon, unit='')


def test_platform_from_config(make_tmp_config):
    t = make_tmp_config(HYAK_PLATFORM_POOL='klone · CPU')
    assert S.StatusWriter('pool', 1000, 'tokenizers').platform == 'klone · CPU'
    assert S.StatusWriter('corpus', 23, 'buckets', cfg=t.cfg).platform == 'Hyak Tillicum · H200'
    # default monitor dir follows HYAK_MONITOR_ROOT (never the repository in tests)
    assert S.StatusWriter('corpus', 23, 'buckets').monitor_dir == t.base / 'monitor' / 'corpus'


def test_total_update_via_extra(mon):
    w = writer(mon, stage='probe', total=None, unit='measurements')
    assert w.update(0)['total'] is None
    assert w.update(1, total=4)['total'] == 4 and w.total == 4


def test_two_jobs_merge_rows_and_dead_jobs_vanish(mon, monkeypatch):
    alive = {'111': True, '222': True}
    monkeypatch.setattr(S.slurm, 'is_job_alive', lambda job, timeout=20.0: alive.get(job))
    a, b = writer(mon, job='111'), writer(mon, job='222')
    a.update(1, workers=[a.worker_row(task='A', suffix=0), a.worker_row(task='A', suffix=1)])
    status = b.update(2, workers=[b.worker_row(task='B')])
    assert [r['id'] for r in status['workers']] == ['111.0', '111.1', '222']
    assert [r['id'] for r in a.update(3)['workers']] == ['111', '222']
    alive['222'] = False
    a._alive.clear()
    assert [r['id'] for r in a.update(4)['workers']] == ['111']
    assert sorted(p.name.split('__')[0] for p in (mon / 'pool/workers').iterdir()) == ['111']
    final = a.update(5, state='paused')
    assert final['workers'] == [] and list((mon / 'pool/workers').iterdir()) == []


def fake_listing(monkeypatch, jobs, clock=None):
    """slurm._run replaced by a counter that answers the listing ``jobs`` ({id: STATE}); per-job asks are
    answered 'Invalid job id' (finished and forgotten) and counted separately."""
    calls = {'listing': 0, 'single': []}

    def run(command, timeout=20.0):
        if '-u' in command:
            calls['listing'] += 1
            return 0, ''.join(f'{job}|{state}|anytok-pool\n' for job, state in jobs.items()), ''
        calls['single'].append(command[command.index('-j') + 1])
        return 1, '', 'slurm_load_jobs error: Invalid job id specified'
    monkeypatch.setattr(slurm, '_run', run)
    if clock is not None:
        monkeypatch.setattr(slurm, '_clock', lambda: clock[0])
        monkeypatch.setattr(S.time, 'time', lambda: clock[0])
    return calls


def test_many_jobs_cost_one_squeue_listing_per_minute(mon, monkeypatch):
    """pool-2 / monitoring-9: N jobs used to make N x (N-1) squeue calls per minute."""
    clock = [1_000_000.0]
    jobs = {str(6000 + i): 'RUNNING' for i in range(20)}
    calls = fake_listing(monkeypatch, jobs, clock)
    writers = [writer(mon, job=job) for job in jobs]
    for w in writers:
        w.update(1)                                           # every job has a worker file
    first = writers[0]
    calls.update(listing=0, single=[])
    for cycle in range(3):
        clock[0] += 61
        rows = first.other_rows()
        assert len(rows) == 19
        first.update(cycle)
    assert calls['listing'] == 3 and calls['single'] == []    # one listing per refresh, no per-job calls
    # all 20 writers of one process share the same cached listing
    calls.update(listing=0)
    clock[0] += 61
    for w in writers:
        assert len(w.other_rows()) == 19
    assert calls['listing'] == 1 and calls['single'] == []


def test_finished_job_is_found_by_one_single_query_and_forgotten(mon, monkeypatch):
    clock = [1_000_000.0]
    calls = fake_listing(monkeypatch, {'222': 'COMPLETED', '444': 'RUNNING'}, clock)
    watcher = writer(mon, job='111')
    others = {job: writer(mon, job=job) for job in ('222', '333', '444')}
    for w in others.values():
        w._store_own_rows([w.worker_row(task='x')])             # worker files without the merge side effects
    assert [r['id'] for r in watcher.other_rows()] == ['444']   # 222 listed as COMPLETED, 333 'Invalid job id'
    assert calls['single'] == ['333'] and calls['listing'] == 1
    assert not others['222'].worker_file.exists() and not others['333'].worker_file.exists()
    clock[0] += 61
    assert [r['id'] for r in watcher.other_rows()] == ['444']
    assert calls['single'] == ['333'] and calls['listing'] == 2  # a False verdict is final: asked once


def test_unusable_squeue_never_means_finished(mon, monkeypatch):
    """pool-3: no listing and empty single answers (a filter hid the job) leave the rows alone."""
    monkeypatch.setattr(slurm, '_run', lambda command, timeout=20.0: (0, '', ''))
    old, new = writer(mon, job='333'), writer(mon, job='444')
    old.update(1)
    assert [r['id'] for r in new.update(2)['workers']] == ['333', '444']
    assert old.worker_file.exists()
    monkeypatch.setattr(slurm, '_run', lambda command, timeout=20.0: None)
    slurm.reset_job_listing()
    new._alive.clear()
    assert [r['id'] for r in new.update(3)['workers']] == ['333', '444']
    assert old.worker_file.exists()


def test_start_row_of_a_dead_process_does_not_come_back(mon, monkeypatch):
    """monitoring-11 / init-10: ``status start`` leaves a 'Starting' row in its own worker file; once the stage
    writes its terminal status the row must not be merged back in until the job's exit trap."""
    monkeypatch.setenv('SLURM_JOB_ID', '8001')
    monkeypatch.setattr(S.slurm, 'is_job_alive', lambda job, timeout=20.0: True)  # the job itself is alive
    d = mon / 'eval-rl'
    code = ('from hyak.common import status as S; import sys; '
            f'S.start("eval-rl", task="scoring", monitor_dir={str(d)!r}, job="8001")')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          env={**os.environ, 'SLURM_JOB_ID': '8001', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert done.returncode == 0, done.stderr
    stage = S.StatusWriter('eval-rl', 8, 'steps', platform='P', monitor_dir=d, job='8001')
    running = stage.update(1, stage='scoring', workers=[stage.worker_row(task='scoring tokenizers')])
    assert [(r['id'], r['task']) for r in running['workers']] == [('8001', 'scoring tokenizers')]
    complete = stage.update(8, state='complete')
    assert complete['workers'] == [] and read_json(d / 'status.json')['workers'] == []
    assert len(list((d / 'workers').glob('*.json'))) == 0       # the dead process's file was deleted


def test_unusable_job_listing_keeps_the_pid_rule_for_local_jobs(mon):
    ghost = writer(mon, job='local-1')
    ghost.update(1)
    assert [r['id'] for r in writer(mon, job='local-2').other_rows()] == ['local-1']  # same pid: alive


def test_unknown_liveness_uses_staleness(mon, monkeypatch):
    monkeypatch.setattr(S.slurm, 'is_job_alive', lambda job, timeout=20.0: None)
    old = writer(mon, job='333')
    old.update(1)
    new = writer(mon, job='444', stale_seconds=3600)
    assert [r['id'] for r in new.update(2)['workers']] == ['333', '444']
    os.utime(old.worker_file, (1, 1))
    assert [r['id'] for r in new.update(3)['workers']] == ['444']
    assert old.worker_file.exists()  # unknown liveness never deletes another job's file


def test_local_rows_use_pid_liveness(mon):
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    ghost = writer(mon, job=f'local-{dead.pid}')
    ghost.pid = dead.pid
    ghost.worker_file = ghost.workers_dir / f'local-{dead.pid}__{ghost.host}__{dead.pid}.json'
    ghost.update(1)
    me = writer(mon, job=f'local-{os.getpid()}')
    assert [r['id'] for r in me.update(2)['workers']] == [f'local-{os.getpid()}']
    assert not ghost.worker_file.exists()


def test_io_errors_are_reported_not_raised(mon, monkeypatch, capsys):
    w = writer(mon)
    monkeypatch.setattr(S, 'atomic_write_json', lambda *a, **k: (_ for _ in ()).throw(OSError('stale NFS handle')))
    status = w.update(1)
    assert status['completed'] == 1 and w.write_errors >= 1
    assert 'stale NFS handle' in capsys.readouterr().err
    strict = writer(mon, strict=True)
    with pytest.raises(OSError):
        strict.update(1)


def test_event(mon):
    w = writer(mon)
    row = w.event({'event': 'claimed', 'index': 413})
    assert row['hyak_stage'] == 'pool' and row['job_id'] == '111' and row['index'] == 413
    assert events(mon / 'pool/events.jsonl') == [row]
    assert w.event({'x': float('nan')})['x'] is None
    with pytest.raises(ValueError):
        w.event({'x': object()})
    with pytest.raises(ValueError):
        w.event([1])


def test_finalize_semantics(mon, monkeypatch, tmp_path):
    monkeypatch.setattr(S.slurm, 'is_job_alive', lambda job, timeout=20.0: True)
    d = mon / 'pool'
    w = writer(mon, job='111')
    w.update(40, stage='generating', eta_seconds=100, extra_info='kept')
    # exit 0 while still running and no other job: paused, progress kept, extra fields kept
    s = S.finalize('pool', 0, monitor_dir=d, job='111')
    assert (s['state'], s['stage'], s['completed'], s['total'], s['workers']) == ('paused', 'paused', 40, 1000, [])
    assert s['extra_info'] == 'kept' and s['exit_code'] == 0 and s['eta_seconds'] is None
    # another job alive: its row stays and the state stays running
    w.update(41, stage='generating')
    other = writer(mon, job='222')
    other.update(41, stage='generating')
    s = S.finalize('pool', 0, monitor_dir=d, job='111')
    assert s['state'] == 'running' and [r['id'] for r in s['workers']] == ['222']
    # non-zero exit: failed with the log tail; signals: stopped
    log = tmp_path / 'job.log'
    log.write_text('step 1\nTraceback (most recent call last):\nRuntimeError: boom\n')
    s = S.finalize('pool', 1, monitor_dir=d, job='222', log_path=log)
    assert s['state'] == 'failed' and s['stage'] == 'failed' and 'exited with code 1' in s['error']
    assert 'RuntimeError: boom' in s['error']
    s = S.finalize('pool', 143, monitor_dir=d, job='222')
    assert s['state'] == 'stopped' and 'SIGTERM' in s['error']
    # a complete stage stays complete after start and exit 0
    w.update(1000, state='complete')
    S.start('pool', monitor_dir=d, job='999')
    s = S.finalize('pool', 0, monitor_dir=d, job='999')
    assert (s['state'], s['stage'], s['completed']) == ('complete', 'complete', 1000)
    kinds = [e.get('event') for e in events(d / 'events.jsonl')]
    assert kinds.count('job_end') == 5 and 'job_start' in kinds


def test_finalize_keeps_a_failure_the_stage_wrote_itself(mon):
    """monitoring-11: the probe's own verdict and label survive the job's exit-trap finalize."""
    d = mon / 'probe'
    w = writer(mon, stage='probe', total=None, unit='measurements', job='77')
    w.update(2, stage='does not fit', state='failed', total=2,
             error='DOES NOT FIT: chunk 16384 at 128000 rows peaks at 141.2 GiB')
    s = S.finalize('probe', 4, monitor_dir=d, job='77', message='memory probe: a check failed')
    assert s['state'] == 'failed' and s['stage'] == 'does not fit' and s['exit_code'] == 4
    assert s['error'] == 'DOES NOT FIT: chunk 16384 at 128000 rows peaks at 141.2 GiB | memory probe: a check failed'
    # a failure written by an EARLIER job is replaced as before
    s = S.finalize('probe', 1, monitor_dir=d, job='78', message='crashed')
    assert s['stage'] == 'failed' and s['error'] == 'crashed'


def test_finalize_error_keeps_the_end_of_long_log_lines(mon, tmp_path):
    """slurm-13: hyak_die puts the remedy at the end of a long line."""
    log = tmp_path / 'job.log'
    long_line = 'ERROR: ' + 'x' * 1000 + ' Remedy: on the login node run: bash /gpfs/setup_train_env.sh'
    log.write_text('short line\n' + long_line + '\n')
    s = S.finalize('corpus', 1, monitor_dir=mon / 'corpus', platform='P', log_path=log)
    assert 'short line' in s['error'] and s['error'].endswith('bash /gpfs/setup_train_env.sh')
    assert s['error'].count('x') < 800 and ' ... ' in s['error'] and s['error'].count('ERROR: ') == 1
    exact = 'y' * S.LINE_KEEP
    assert S._clip(exact) == exact and len(S._clip(exact + 'z')) < len(exact) + 6


def test_index_extra_is_renamed_so_the_dashboard_shows_the_task(mon):
    """monitoring-7: the unchanged UI uses its pool layout for any row with ``index`` and hides ``task``."""
    row = writer(mon).worker_row(task='0413 bpe/regex 96k: training', index=413, state='Training')
    assert 'index' not in row and row['pool_index'] == 413 and row['task'].startswith('0413')


def test_finalize_exit_0_message_is_stored_as_note(mon):
    """slurm-13 (b): job.sh passes JOB_NOTE for a 'not now' job; shown as a note, never as an error."""
    d = mon / 'train-rl'
    s = S.finalize('train-rl', 0, monitor_dir=d, platform='P', job='31',
                   message='not now: the uniform bootstrap has not frozen input/pairing.json yet')
    assert s['note'].startswith('not now: the uniform bootstrap') and s['error'] is None and s['state'] == 'paused'
    assert read_json(d / 'status.json')['note'] == s['note']
    # the next job does not inherit it
    assert 'note' not in S.start('train-rl', monitor_dir=d, platform='P', job='32')
    assert 'note' not in S.finalize('train-rl', 0, monitor_dir=d, platform='P', job='32')
    # a non-zero exit keeps using the message as the error, not as a note
    s = S.finalize('train-rl', 2, monitor_dir=d, platform='P', job='33', message='bad config')
    assert 'note' not in s and s['error'].startswith('bad config')


def test_finalize_without_status_uses_stage_defaults(mon):
    s = S.finalize('train-uniform', 0, monitor_dir=mon / 'train-uniform', platform='P')
    assert s['step'] == 0 and s['total'] == 20000 and s['unit'] == 'steps' and s['state'] == 'paused'


def test_finalize_redacts_secrets(mon, monkeypatch, tmp_path):
    monkeypatch.setenv('HF_TOKEN', 'hf_abcdefghijk')
    log = tmp_path / 'log'
    log.write_text('401 for token hf_abcdefghijk\n')
    s = S.finalize('corpus', 1, monitor_dir=mon / 'corpus', platform='P', log_path=log)
    assert 'hf_abcdefghijk' not in json.dumps(s) and '***' in s['error']


def test_cli(mon, monkeypatch):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'SLURM_JOB_ID': '555'}
    env.pop('HYAK_CONFIG', None)
    d = str(mon / 'eval-rl')

    def run(*args):
        return subprocess.run([sys.executable, '-m', 'hyak.common.status', *args, '--stage', 'eval-rl',
                               '--monitor-dir', d], capture_output=True, text=True, env=env)
    start = run('start', '--task', 'warming up')
    assert start.returncode == 0, start.stderr  # works without any config file (platform fallback)
    status = read_json(Path(d) / 'status.json')
    assert status['state'] == 'running' and status['workers'][0]['task'] == 'warming up'
    assert run('event', '--name', 'claimed', '--set', 'index=7', '--set', 'note=hello').returncode == 0
    assert events(Path(d) / 'events.jsonl')[-1]['index'] == 7
    final = run('finalize', '--exit-code', '2', '--message', 'scorer crashed')
    assert final.returncode == 0
    status = json.loads(run('show').stdout)
    assert status['state'] == 'failed' and status['error'] == 'scorer crashed' and status['workers'] == []
    bad = run('event', '--name', 'x', '--set', 'novalue')
    assert bad.returncode == 2 and 'KEY=VALUE' in bad.stderr           # usage error, never 1
    nan = run('event', '--name', 'eval', '--set', 'generated_bpb=NaN')
    assert nan.returncode == 0 and events(Path(d) / 'events.jsonl')[-1]['generated_bpb'] is None
