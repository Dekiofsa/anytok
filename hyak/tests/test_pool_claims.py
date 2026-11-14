"""Claim protocol of hyak.pool.worker: atomic mkdir, ownership, takeover rules, concurrency."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hyak.common.io import atomic_write_json
from hyak.pool.worker import HEARTBEAT_FILE, OWNER_FILE, ClaimStore

REPO = Path(__file__).resolve().parents[2]


def store(root, **kwargs):
    kwargs.setdefault('alive', lambda job: True)
    kwargs.setdefault('job', '1000')
    return ClaimStore(root, **kwargs)


def fake_owner(root, index, *, job='5555', host='other-host', pid=12345, started=None, beat=None, token='t' * 32):
    folder = Path(root) / f'{index:04d}'
    folder.mkdir(parents=True)
    started = time.time() if started is None else started
    atomic_write_json(folder / OWNER_FILE, {'token': token, 'index': index, 'job_id': job, 'host': host, 'pid': pid,
                                            'started_unix': started, 'worker_id': job or f'local-{pid}'})
    atomic_write_json(folder / HEARTBEAT_FILE, {'token': token, 'at_unix': started if beat is None else beat})
    return folder


def dead_pid():
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    return process.pid


def test_claim_is_exclusive_and_release_frees_it(tmp_path):
    a = store(tmp_path / 'claims')
    b = store(tmp_path / 'claims', job='2000')
    claim = a.try_claim(7, slot=1)
    assert claim is not None and claim.path.name == '0007'
    owner = json.loads((claim.path / OWNER_FILE).read_text())
    assert owner['job_id'] == '1000' and owner['slot'] == 1 and owner['token'] == claim.token
    assert b.try_claim(7) is None            # job 1000 is alive -> never taken over
    assert a.held(claim)
    a.heartbeat(claim)
    assert a.release(claim)
    assert not (tmp_path / 'claims' / '0007').exists()
    assert b.try_claim(7) is not None


def test_release_by_non_owner_keeps_the_claim(tmp_path):
    a = store(tmp_path)
    claim = a.try_claim(1)
    other = type(claim)(claim.index, claim.path, 'x' * 32, {})
    assert not a.release(other)
    assert claim.path.exists() and a.held(claim)


def test_live_job_is_never_taken_over_even_without_heartbeat(tmp_path):
    fake_owner(tmp_path, 3, beat=time.time() - 100 * 3600)
    assert store(tmp_path, alive=lambda job: True).try_claim(3) is None


def test_dead_job_is_taken_over(tmp_path):
    fake_owner(tmp_path, 3, job='5555')
    events = []
    s = store(tmp_path, alive=lambda job: False if job == '5555' else True,
              on_event=lambda name, **f: events.append((name, f)))
    claim = s.try_claim(3)
    assert claim is not None and s.held(claim)
    assert events and events[0][0] == 'claim_takeover' and 'no longer known' in events[0][1]['reason']
    assert not list(tmp_path.glob('.stale-*')) and not list(tmp_path.glob('*.break'))


def test_unknown_job_state_uses_the_heartbeat_timeout(tmp_path):
    fake_owner(tmp_path, 4, job='5555', beat=time.time() - 3600)
    s = store(tmp_path, alive=lambda job: None, timeout_seconds=2 * 3600)
    assert s.try_claim(4) is None
    later = store(tmp_path, alive=lambda job: None, timeout_seconds=2 * 3600, clock=lambda: time.time() + 2 * 3600)
    assert later.try_claim(4) is not None


def test_fresh_heartbeat_keeps_an_old_claim(tmp_path):
    fake_owner(tmp_path, 4, job='5555', started=time.time() - 30 * 3600, beat=time.time())
    assert store(tmp_path, alive=lambda job: None, timeout_seconds=6 * 3600).try_claim(4) is None


def test_ownerless_claim_has_a_grace_period(tmp_path):
    (tmp_path / '0005').mkdir()
    assert store(tmp_path).try_claim(5) is None
    assert store(tmp_path, clock=lambda: time.time() + 3600).try_claim(5) is not None


def test_local_owner_on_this_host(tmp_path):
    s = store(tmp_path, job=None, alive=lambda job: None)
    fake_owner(tmp_path, 6, job=None, host=s.host, pid=dead_pid())
    assert s.try_claim(6) is not None          # pid gone -> provably dead
    sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        fake_owner(tmp_path, 8, job=None, host=s.host, pid=sleeper.pid, token='u' * 32)
        assert s.try_claim(8) is None          # pid alive
    finally:
        sleeper.kill()
        sleeper.wait()


def test_same_job_dead_process_is_taken_over(tmp_path):
    s = store(tmp_path, job='777', alive=lambda job: True)
    fake_owner(tmp_path, 9, job='777', host=s.host, pid=dead_pid())
    assert s.try_claim(9) is not None


def test_break_lock_blocks_a_second_breaker(tmp_path):
    fake_owner(tmp_path, 2, job='5555')
    (tmp_path / '0002.break').mkdir()
    assert store(tmp_path, alive=lambda job: False if job == '5555' else True).try_claim(2) is None
    assert (tmp_path / '0002' / OWNER_FILE).exists()


def test_stale_break_lock_is_cleared(tmp_path):
    fake_owner(tmp_path, 2, job='5555')
    lock = tmp_path / '0002.break'
    lock.mkdir()
    old = time.time() - 3600
    os.utime(lock, (old, old))
    s = store(tmp_path, alive=lambda job: False if job == '5555' else True)
    assert s.try_claim(2) is None               # this call clears the stale lock
    assert not lock.exists()
    assert s.try_claim(2) is not None


CLAIMER = r'''
import json, os, sys, time
from pathlib import Path
from hyak.pool.worker import ClaimStore
root, go, count, dead = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
claims = ClaimStore(root, job=f'9{os.getpid()}', alive=lambda job: False if job == dead else True)
while not go.exists():
    time.sleep(0.001)
won = [i for i in range(count) if claims.try_claim(i) is not None]
print(json.dumps(won))
'''


@pytest.mark.parametrize('stale_first', [False, True])
def test_concurrent_processes_never_share_an_index(tmp_path, stale_first):
    root = tmp_path / 'claims'
    root.mkdir()
    count = 40
    if stale_first:
        for index in range(0, count, 2):
            fake_owner(root, index, job='deadjob', token=f'{index:032d}')
    script = tmp_path / 'claimer.py'
    script.write_text(CLAIMER)
    go = tmp_path / 'go'
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    processes = [subprocess.Popen([sys.executable, str(script), str(root), str(go), str(count), 'deadjob'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
                 for _ in range(8)]
    time.sleep(1.5)
    go.touch()
    won = []
    for process in processes:
        out, err = process.communicate(timeout=60)
        assert process.returncode == 0, err
        won.extend(json.loads(out))
    assert sorted(won) == list(range(count)), 'every index claimed exactly once'
    for index in range(count):
        owner = json.loads((root / f'{index:04d}' / OWNER_FILE).read_text())
        assert owner['job_id'] != 'deadjob'
    assert not list(root.glob('.stale-*')) and not list(root.glob('*.break'))


def test_requeued_job_takes_over_its_earlier_claims_and_others_do_not(tmp_path):
    """pool-4: a requeued job keeps its job id; claims of its earlier incarnation are stale for it only."""
    fake_owner(tmp_path, 3, job='4242', host='g3001', started=time.time() - 3 * 3600)
    fake_owner(tmp_path, 4, job='4242', host='g3001', started=time.time() - 30 * 3600, token='v' * 32)
    alive = lambda job: True  # noqa: E731 -- squeue lists job 4242 (it runs again)
    other = store(tmp_path, job='5000', alive=alive)
    assert other.try_claim(3) is None and other.try_claim(4) is None        # never another live job's claim
    again_same_node = store(tmp_path, job='4242', alive=alive, host='g3001', restart_count=0, pid=os.getpid())
    reason = again_same_node.stale_reason(again_same_node.inspect(3))
    assert reason and 'exited' in reason                                    # same incarnation, other host check
    requeued = store(tmp_path, job='4242', alive=alive, host='g3007', restart_count=1)
    assert 'incarnation 0' in requeued.stale_reason(requeued.inspect(3))
    assert requeued.try_claim(3) is not None and requeued.try_claim(4) is not None
    owner = json.loads((tmp_path / '0003' / OWNER_FILE).read_text())
    assert owner['restart_count'] == 1 and owner['host'] == 'g3007'
    assert other.try_claim(3) is None                                       # the new claims are live again


def test_restart_count_comes_from_slurm(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_RESTART_COUNT', '2')
    assert ClaimStore(tmp_path, job='1', alive=lambda job: True).restart_count == 2
    monkeypatch.delenv('SLURM_RESTART_COUNT')
    assert ClaimStore(tmp_path, job='1', alive=lambda job: True).restart_count == 0
