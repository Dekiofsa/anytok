"""hyak.tools.reconcile (dead jobs), hyak.tools.policy_monitor (lock and exit status) and the registry splice."""
from __future__ import annotations

import errno
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hyak.common import slurm
from hyak.common.io import read_json
from hyak.common.paths import default_monitor_dir
from hyak.common.status import StatusWriter
from hyak.tools import policy_monitor as PM
from hyak.tools import reconcile as RC
from hyak.tools import register_dashboard as R


# --- reconcile ---------------------------------------------------------------------------------------------
def running_pool(job='501', completed=7, **extra):
    """A pool status of a job that then died without running its exit trap (status and worker file stay)."""
    w = StatusWriter('pool', 1000, 'tokenizers', platform='P', job=job)
    row = w.worker_row(task='0413 bpe: training', state='Training', suffix='s0')
    w.update(completed, stage='generating', eta_seconds=3600, workers=[row], per_family={'bpe': 3}, **extra)
    return w


def status_of(stage='pool'):
    return read_json(default_monitor_dir(stage) / 'status.json')


def table_of(**jobs):
    return {job: (state, name) for job, (state, name) in jobs.items()}


@pytest.fixture
def liveness(monkeypatch):
    """slurm.is_job_alive answers from a dict (a missing job: 'Invalid job id' = False)."""
    answers = {}
    monkeypatch.setattr(slurm, 'is_job_alive', lambda job, timeout=20.0: answers.get(job, False))
    return answers


def test_reconcile_marks_a_dead_job_stopped_and_carries_everything_else(liveness):
    running_pool(completed=7, note='kept')
    result = RC.reconcile('pool', table={})
    assert result['action'] == 'marked stopped' and result['jobs'] == {'501': False}
    s = status_of()
    assert (s['state'], s['stage'], s['workers'], s['completed'], s['total']) == ('stopped', 'stopped', [], 7, 1000)
    assert s['note'] == 'kept' and s['per_family'] == {'bpe': 3} and s['hyak_stage'] == 'pool'
    assert '501' in s['error'] and 'ended without recording their end' in s['error']
    events = [json.loads(x) for x in (default_monitor_dir('pool') / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'reconciled'
    assert not list((default_monitor_dir('pool') / 'workers').glob('*.json'))
    # a second run finds nothing to do
    assert RC.reconcile('pool', table={}) is None


def test_reconcile_dry_run_writes_nothing(liveness):
    running_pool()
    before = (default_monitor_dir('pool') / 'status.json').read_bytes()
    assert RC.reconcile('pool', dry_run=True, table={})['action'] == 'would mark stopped'
    assert (default_monitor_dir('pool') / 'status.json').read_bytes() == before


def test_reconcile_leaves_a_stage_alone_when_any_job_may_be_alive(liveness):
    running_pool(job='501')
    other = StatusWriter('pool', 1000, 'tokenizers', platform='P', job='502')
    other._store_own_rows([other.worker_row(task='x')])        # a second job's worker file
    liveness.update({'501': False, '502': True})
    assert RC.reconcile('pool', table={})['action'] == 'left alone'
    assert status_of()['state'] == 'running'
    liveness['502'] = None                                      # squeue cannot say: never "finished"
    assert RC.reconcile('pool', table={})['action'] == 'left alone'
    liveness['502'] = False
    assert RC.reconcile('pool', table={})['action'] == 'marked stopped'
    assert '501, 502' in status_of()['error']


def test_reconcile_uses_the_listing_and_leaves_a_stage_with_a_queued_job(liveness):
    running_pool()
    # listed as finished: decisive without a single-job query
    liveness['501'] = True
    assert RC.reconcile('pool', dry_run=True, table={'501': ('TIMEOUT', 'anytok-pool')})['action'] == 'would mark stopped'
    # a job of the stage (same name) is pending: it will correct the status itself
    table = {'501': ('TIMEOUT', 'anytok-pool'), '600': ('PENDING', 'anytok-pool')}
    out = RC.reconcile('pool', table=table)
    assert out['action'] == 'left alone' and '600' in out['reason'] and status_of()['state'] == 'running'
    # another stage's job does not matter
    assert RC.reconcile('pool', table={'501': ('TIMEOUT', 'anytok-pool'), '601': ('RUNNING', 'anytok-eval-rl')},
                        dry_run=True)['action'] == 'would mark stopped'


def test_reconcile_never_touches_other_states_or_unreadable_statuses(liveness):
    w = running_pool()
    w.update(7, state='paused')
    assert RC.reconcile('pool', table={}) is None
    w.update(1000, state='complete')
    assert RC.reconcile('pool', table={}) is None
    assert RC.reconcile('probe', table={}) is None             # no status at all
    path = default_monitor_dir('corpus') / 'status.json'
    path.parent.mkdir(parents=True)
    path.write_text('{not json')
    assert RC.reconcile('corpus', table={}) is None and path.read_text() == '{not json'


def test_reconcile_without_job_ids_or_squeue_changes_nothing(liveness):
    w = StatusWriter('pool', 1000, 'tokenizers', platform='P')     # a local run: job id local-<pid>
    w.update(3)
    assert RC.reconcile('pool', table={})['reason'] == 'the status names no Slurm job'
    running_pool()
    assert RC.reconcile('pool', table=None)['reason'] == 'squeue could not be asked'
    assert status_of()['state'] == 'running'


def test_reconcile_skips_a_status_that_moved_while_it_was_judged(monkeypatch):
    running_pool()

    def alive(job, timeout=20.0):
        running_pool(completed=8)                                # the stage (or a new job) wrote meanwhile
        return False
    monkeypatch.setattr(slurm, 'is_job_alive', alive)
    out = RC.reconcile('pool', table={})
    assert out['action'] == 'left alone' and 'changed meanwhile' in out['reason']
    assert status_of()['state'] == 'running' and status_of()['completed'] == 8


def test_reconcile_main_exit_codes(monkeypatch, capsys, liveness):
    assert RC.main([]) == 0 and 'nothing to do' in capsys.readouterr().out    # asks squeue only when needed
    running_pool()
    monkeypatch.setattr(slurm, 'job_table', lambda **kw: None)
    assert RC.main(['--stage', 'pool']) == 2
    assert 'squeue could not be asked' in capsys.readouterr().err and status_of()['state'] == 'running'
    monkeypatch.setattr(slurm, 'job_table', lambda **kw: {})
    assert RC.main(['--stage', 'pool', '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['action'] == 'would mark stopped'
    assert RC.main(['--stage', 'pool']) == 0 and status_of()['state'] == 'stopped'
    with pytest.raises(SystemExit) as err:
        RC.main(['--stage', 'nope'])
    assert err.value.code == 2


def test_reconcile_runs_under_the_login_node_python(repo_root, tmp_path):
    python = Path('/usr/bin/python3')
    if not python.exists():
        pytest.skip('no system python3')
    done = subprocess.run([str(python), '-B', '-m', 'hyak.tools.reconcile', '--dry-run'], capture_output=True,
                          text=True, cwd=repo_root, env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(repo_root),
                                                         'HYAK_MONITOR_ROOT': str(tmp_path / 'm'),
                                                         'HYAK_CONFIG': str(tmp_path / 'absent.env')})
    assert done.returncode == 0 and 'nothing to do' in done.stdout, done.stderr


# --- policy collector ------------------------------------------------------------------------------------------
@pytest.fixture
def collector(tmp_path, monkeypatch, repo_root):
    """main() arguments for a collector whose collect() is a stub that records its calls."""
    calls = []

    def fake_collect(run_dir, output, plan, **kw):
        calls.append(output)
        Path(output).write_text('{"snapshots": [], "error": null}\n')
        return {}
    monkeypatch.setattr(PM, 'collect', fake_collect)
    args = ['--once', '--run-dir', str(tmp_path / 'rl'), '--plan', str(repo_root / 'plans/tokenizer-pool-1000-v1.json'),
            '--output', str(tmp_path / 'out' / 'policy-monitor.json')]
    return args, calls, tmp_path / 'out' / 'policy-monitor.json'


def test_collector_continues_when_locking_is_unavailable(collector, monkeypatch, capsys):
    """monitoring-5: ENOLCK / ENOTSUP from flock (GPFS, NFS) must not end the collector."""
    args, calls, output = collector

    def no_locks(fd, operation):
        raise OSError(errno.ENOLCK, 'No locks available')
    monkeypatch.setattr(PM.fcntl, 'flock', no_locks)
    assert PM.main(args) == 0
    assert calls == [output] and output.is_file()
    warning = json.loads(capsys.readouterr().out.splitlines()[0])
    assert warning['stage'] == 'policy_metadata' and 'no lock' in warning['warning'] and 'at_utc' in warning


def test_collector_skips_when_another_collector_holds_the_lock(collector, capsys):
    args, calls, output = collector
    import fcntl
    output.parent.mkdir(parents=True)
    with open(output.with_name(output.name + '.lock'), 'a') as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        assert PM.main(args) == 0
    assert calls == [] and 'another collector holds the lock' in capsys.readouterr().out
    assert PM.main(args) == 0 and calls == [output]            # released: it runs


def test_collector_exit_status_separates_usage_errors_from_crashes(collector, monkeypatch, tmp_path, capsys):
    args, calls, output = collector
    # no --run-dir: the config is needed, and the conftest points HYAK_CONFIG at a file that does not exist
    assert PM.main(['--once', '--plan', str(tmp_path / 'plan.json'), '--output', str(output)]) == 2
    err = capsys.readouterr().err
    assert 'not found' in err and 'Remedy' in err                 # configuration error: exit 2 with a remedy
    monkeypatch.setattr(PM, 'collect', lambda *a, **k: (_ for _ in ()).throw(ImportError('no torch')))
    with pytest.raises(ImportError):                           # a bug is not swallowed: traceback, exit 1
        PM.main(args)


def test_collector_records_snapshot_errors_in_the_file(tmp_path, repo_root):
    pytest.importorskip('torch')
    output = tmp_path / 'policy-monitor.json'
    run = tmp_path / 'rl' / 'step-000500'
    run.mkdir(parents=True)
    (run / 'manifest.json').write_text('{"step": 500, "files": {}}')   # malformed: no policy.pt entry
    saved = PM.collect(tmp_path / 'rl', output, repo_root / 'plans/tokenizer-pool-1000-v1.json')
    assert saved['snapshots'] == [] and 'policy.pt' in saved['error']
    assert json.loads(output.read_text())['error'] == saved['error']


# --- registry splice -------------------------------------------------------------------------------------------
def test_register_appends_after_the_last_entry_without_touching_earlier_text(tmp_path, tmp_config):
    entries = R.build_entries(tmp_config.cfg)
    text = '{\n  "note": "x",\n  "runs": [\n    {"id": "a",   "name": "odd   spacing"},\n    {"id": "b"}\n  ],\n  "tail": 1\n}\n'
    path = tmp_path / 'r.json'
    path.write_text(text)
    # the minimal registry is not a valid dashboard registry, so only the text level is checked here
    new = R.render(text, entries)
    cut = text.index('{"id": "b"}') + len('{"id": "b"}')
    assert new.startswith(text[:cut]) and new.endswith('\n  ],\n  "tail": 1\n}\n')
    assert [r['id'] for r in json.loads(new)['runs']] == ['a', 'b'] + [e['id'] for e in entries]
    assert R.render(new, entries) == new                       # idempotent
    # an empty runs array
    empty = R.render('{"runs": []}', entries)
    assert [r['id'] for r in json.loads(empty)['runs']] == [e['id'] for e in entries]


def test_register_check_exits_4_not_1(tmp_path, repo_root):
    path = tmp_path / 'r.json'
    path.write_text((repo_root / 'plans/run-dashboard.json').read_text())
    done = subprocess.run([sys.executable, '-B', '-m', 'hyak.tools.register_dashboard', '--registry', str(path),
                           '--check', '--config', str(tmp_path / 'absent.env')],
                          capture_output=True, text=True, cwd=repo_root)
    assert done.returncode == 2 and 'not found' in done.stderr          # usage / configuration error
