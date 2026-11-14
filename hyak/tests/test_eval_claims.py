"""hyak.eval.Claims / create_once / parse_indices: atomic work claims shared by several Slurm jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import pytest

from hyak.eval import Claims, EvalError, create_once, parse_indices


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    return process.pid


def test_acquire_contend_release(tmp_path):
    a = Claims(tmp_path, 3600, job='111', host='node1', pid=os.getpid(), alive=lambda job: True)
    b = Claims(tmp_path, 3600, job='222', host='node2', pid=os.getpid(), alive=lambda job: True)
    assert a.try_acquire('0007', task='features')
    owner = json.loads((tmp_path / '0007' / 'owner.json').read_text())
    assert owner['job_id'] == '111' and owner['host'] == 'node1' and owner['task'] == 'features'
    assert not b.try_acquire('0007')            # owner job alive and fresh: no takeover
    assert a.live_names() == ['0007']
    a.release('0007')
    assert not (tmp_path / '0007').exists()
    assert b.try_acquire('0007') and b.is_mine('0007')
    with pytest.raises(EvalError, match='no longer ours'):
        a.release('0007')


def test_takeover_when_the_owner_job_is_gone(tmp_path):
    a = Claims(tmp_path, 3600, job='111', host='node1', pid=1, alive=lambda job: True)
    assert a.try_acquire('0001')
    b = Claims(tmp_path, 3600, job='222', host='node2', pid=2, alive=lambda job: job != '111')
    assert b.try_acquire('0001')
    assert b.takeovers[0]['reason'] == 'owner job 111 is no longer alive (squeue)'
    assert b.takeovers[0]['previous']['job_id'] == '111'
    assert not [p for p in tmp_path.iterdir() if p.name.startswith('.')]   # stale copy removed
    with pytest.raises(EvalError):
        a.release('0001')


def test_takeover_when_the_owner_pid_on_this_host_is_gone(tmp_path):
    a = Claims(tmp_path, 3600, job='local-1', host='here', pid=dead_pid())
    assert a.try_acquire('0002')
    b = Claims(tmp_path, 3600, job='local-2', host='here', pid=os.getpid())
    assert b.try_acquire('0002') and 'is gone' in b.takeovers[0]['reason']


def test_unknown_liveness_waits_for_the_timeout(tmp_path):
    now = [1000.0]
    a = Claims(tmp_path, 60, job='111', host='node1', pid=1, alive=lambda job: None, clock=lambda: now[0])
    assert a.try_acquire('0003')
    b = Claims(tmp_path, 60, job='222', host='node2', pid=2, alive=lambda job: None, clock=lambda: now[0])
    assert not b.try_acquire('0003')
    now[0] += 61
    assert b.try_acquire('0003') and 'old' in b.takeovers[0]['reason']


def test_alive_but_stuck_owner_is_taken_over_after_the_timeout(tmp_path):
    now = [0.0]
    a = Claims(tmp_path, 100, job='111', host='node1', pid=1, alive=lambda job: True, clock=lambda: now[0])
    assert a.try_acquire('0004')
    b = Claims(tmp_path, 100, job='222', host='node2', pid=2, alive=lambda job: True, clock=lambda: now[0])
    now[0] = 50
    assert not b.try_acquire('0004')
    now[0] = 101
    assert b.try_acquire('0004')


def test_claim_directory_without_owner_file_gets_a_grace_period(tmp_path):
    (tmp_path / '0005').mkdir()
    b = Claims(tmp_path, 3600, job='222', host='node2', pid=2, alive=lambda job: True)
    assert not b.try_acquire('0005')            # just created by someone else
    old = (tmp_path / '0005').stat().st_mtime - 700
    os.utime(tmp_path / '0005', (old, old))
    assert b.try_acquire('0005')


def test_only_one_of_many_threads_wins(tmp_path):
    winners = []
    barrier = threading.Barrier(16)

    def attempt(k):
        claims = Claims(tmp_path, 3600, job=f'{100 + k}', host=f'node{k}', pid=k + 10, alive=lambda job: True)
        barrier.wait()
        if claims.try_acquire('0009'):
            winners.append(k)

    threads = [threading.Thread(target=attempt, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1


def test_create_once_never_replaces(tmp_path):
    target = tmp_path / 'a' / 'plan.json'
    assert create_once(target, {'x': 1}) is True
    assert create_once(target, {'x': 2}) is False
    assert json.loads(target.read_text()) == {'x': 1}
    assert [p.name for p in target.parent.iterdir()] == ['plan.json']   # no temporary files left behind
    results = []
    barrier = threading.Barrier(8)

    def race(k):
        barrier.wait()
        results.append(create_once(tmp_path / 'b.json', {'k': k}, indent=None))

    threads = [threading.Thread(target=race, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1


def test_parse_indices():
    assert parse_indices(None, range(5)) == [0, 1, 2, 3, 4]
    assert parse_indices(['0-2', '4'], range(5)) == [0, 1, 2, 4]
    assert parse_indices(['3,1'], [1, 3, 5]) == [3, 1]
    with pytest.raises(EvalError, match='unique'):
        parse_indices(['1', '1'], range(5))
    with pytest.raises(EvalError, match='belong to the plan'):
        parse_indices(['7'], range(5))


# --- review eval-4: takeover race; review eval-5: pid test only within the same job -----------------------
def _stale_claim(root):
    old = Claims(root, 3600, job='999', host='gone-node', pid=1, alive=lambda job: True)
    assert old.try_acquire('0042')


@pytest.mark.parametrize('takers', [2, 8])
def test_concurrent_takeover_of_a_stale_claim_has_exactly_one_winner(tmp_path, takers):
    import random
    import time
    for trial in range(100):
        root = tmp_path / f'trial{trial}'
        _stale_claim(root)
        rng = random.Random(trial)
        delays = [rng.uniform(0, 0.02) for _ in range(takers)]
        winners, errors = [], []
        barrier = threading.Barrier(takers)

        def liveness(job, rng=rng):
            time.sleep(rng.uniform(0.002, 0.02))   # like an squeue call
            return job != '999'

        def attempt(k):
            claims = Claims(root, 3600, job=str(2000 + k), host=f'node{k}', pid=os.getpid(), alive=liveness)
            barrier.wait()
            time.sleep(delays[k])
            try:
                if claims.try_acquire('0042'):
                    winners.append(claims)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=attempt, args=(k,)) for k in range(takers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [] and len(winners) == 1, (trial, errors, len(winners))
        assert winners[0].is_mine('0042')
        winners[0].release('0042')                 # its claim was never renamed away
        assert not [p for p in root.iterdir() if p.name.startswith('.')]


def test_a_taker_that_decided_on_an_old_owner_backs_off(tmp_path):
    """Forced interleaving: B judges the claim stale, A completes a takeover meanwhile, then B continues."""
    _stale_claim(tmp_path)
    a = Claims(tmp_path, 3600, job='2001', host='nodeA', pid=os.getpid(), alive=lambda job: job != '999')

    def b_liveness(job):
        assert a.try_acquire('0042')               # A's whole takeover happens inside B's squeue call
        return False if job == '999' else True

    b = Claims(tmp_path, 3600, job='2002', host='nodeB', pid=os.getpid(), alive=b_liveness)
    assert b.try_acquire('0042') is False
    assert b.takeovers == [] and a.is_mine('0042')
    a.release('0042')


def test_a_lost_claim_is_a_warning_after_the_work_is_written(tmp_path):
    a = Claims(tmp_path, 3600, job='111', host='node1', pid=1, alive=lambda job: True)
    assert a.try_acquire('0001')
    b = Claims(tmp_path, 3600, job='222', host='node2', pid=2, alive=lambda job: job != '111')
    assert b.try_acquire('0001')
    assert a.release_after_work('0001') is False and a.lost == ['0001']
    assert b.release_after_work('0001') is True and b.lost == []


def test_pid_test_only_for_the_same_job_or_local_processes(tmp_path):
    """Another Slurm job's pid may live in another PID namespace: judge it by squeue and the timeout only."""
    gone = dead_pid()
    other = Claims(tmp_path, 3600, job='5001', host='g3071', pid=gone, alive=lambda job: True)
    assert other.try_acquire('0001')
    me = Claims(tmp_path, 3600, job='5002', host='g3071', pid=os.getpid(), alive=lambda job: True)
    assert me.stale_reason('0001') is None and not me.try_acquire('0001')
    # The same job (e.g. an earlier step of this job on this node): the pid test applies.
    same = Claims(tmp_path, 3600, job='5001', host='g3071', pid=os.getpid(), alive=lambda job: True)
    assert 'is gone' in same.stale_reason('0001')
    # A non-Slurm process on this host: the pid test applies (squeue knows nothing about it).
    local = Claims(tmp_path / 'l', 3600, job='local-1', host='here', pid=gone)
    assert local.try_acquire('0002')
    assert 'is gone' in Claims(tmp_path / 'l', 3600, job='5003', host='here', pid=1).stale_reason('0002')


def test_a_takeover_lock_left_by_a_killed_process_expires(tmp_path):
    import time
    _stale_claim(tmp_path)
    (tmp_path / '.takeover-0042').mkdir()
    b = Claims(tmp_path, 3600, job='2002', host='nodeB', pid=os.getpid(), alive=lambda job: job != '999')
    assert b.try_acquire('0042') is False          # another taker seems to be at work
    old = time.time() - Claims.TAKEOVER_LOCK_SECONDS - 5
    os.utime(tmp_path / '.takeover-0042', (old, old))
    assert b.try_acquire('0042') is False          # the expired lock is removed ...
    assert b.try_acquire('0042') is True           # ... and the next attempt takes over
    assert b.live_names() == ['0042']
