"""hyak.common.io and hyak.common.slurm."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time

import pytest

from hyak.common import io, slurm
from monitoring.state import stamp


# --- io -------------------------------------------------------------------------------------------------
def test_sha256_and_record(tmp_path):
    data = os.urandom(3 * 1024 * 1024 + 17)
    path = tmp_path / 'blob'
    path.write_bytes(data)
    assert io.sha256_file(path, chunk=1024 * 1024) == hashlib.sha256(data).hexdigest()
    assert io.file_record(path) == {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    assert io.sha256_bytes(b'abc') == hashlib.sha256(b'abc').hexdigest()


def test_atomic_write_leaves_no_temporaries_and_keeps_mode(tmp_path):
    target = tmp_path / 'sub' / 'status.json'
    io.atomic_write_json(target, {'b': 1, 'a': 'é'})
    assert target.read_text(encoding='utf-8') == '{\n  "a": "é",\n  "b": 1\n}\n'
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o666 & ~io._UMASK
    os.chmod(target, 0o640)
    io.atomic_write_text(target, 'x')
    assert stat.S_IMODE(target.stat().st_mode) == 0o640 and target.read_text() == 'x'
    assert sorted(p.name for p in target.parent.iterdir()) == ['status.json']


def test_atomic_write_rejects_nan_without_touching_file(tmp_path):
    target = tmp_path / 's.json'
    io.atomic_write_json(target, {'ok': 1})
    with pytest.raises(ValueError):
        io.atomic_write_json(target, {'bad': float('nan')})
    assert json.loads(target.read_text()) == {'ok': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['s.json']


def test_atomic_write_failure_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / 'f.txt'
    target.write_text('old')

    def boom(*_):
        raise OSError('disk full')
    monkeypatch.setattr(io.os, 'replace', boom)
    with pytest.raises(OSError):
        io.atomic_write_text(target, 'new')
    assert target.read_text() == 'old' and [p.name for p in tmp_path.iterdir()] == ['f.txt']


def test_concurrent_readers_never_see_partial_json(tmp_path):
    target = tmp_path / 'status.json'
    io.atomic_write_json(target, {'n': 0, 'pad': 'x' * 100000})
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                json.loads(target.read_text())
            except ValueError as exc:  # pragma: no cover - would be the bug
                errors.append(exc)
    thread = threading.Thread(target=reader)
    thread.start()
    for n in range(200):
        io.atomic_write_json(target, {'n': n, 'pad': 'x' * 100000})
    stop.set()
    thread.join()
    assert not errors


def test_append_jsonl(tmp_path):
    path = tmp_path / 'e' / 'events.jsonl'
    source = {'stage': 'x', 'n': 1}
    row = io.append_jsonl(path, source)
    assert 'at_utc' not in source and list(row)[0] == 'at_utc'
    io.append_jsonl(path, {'at_utc': '2026-01-01T00:00:00Z', 'stage': 'y'})
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1]) == {'at_utc': '2026-01-01T00:00:00Z', 'stage': 'y'}
    assert stamp(json.loads(lines[0])['at_utc']) == pytest.approx(time.time(), abs=5)
    with pytest.raises(ValueError):
        io.append_jsonl(path, {'x': float('inf')})
    with pytest.raises(TypeError):
        io.append_jsonl(path, ['not', 'a', 'dict'])
    assert len(path.read_text().splitlines()) == 2


def test_utc_now_format():
    value = io.utc_now()
    assert value.endswith('Z') and len(value) == len('2026-11-13T20:00:00.123Z')
    assert stamp(value) == pytest.approx(time.time(), abs=5)


def test_read_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_json(tmp_path / 'missing.json')
    assert io.read_json(tmp_path / 'missing.json', None) is None
    bad = tmp_path / 'bad.json'
    bad.write_text('{"x":')
    with pytest.raises(ValueError, match='bad.json'):
        io.read_json(bad)


def test_atomic_copy(tmp_path):
    source = tmp_path / 'a.bin'
    source.write_bytes(b'hello' * 1000)
    record = io.atomic_copy(source, tmp_path / 'out' / 'b.bin', chunk=7)
    assert record == {'bytes': 5000, 'sha256': hashlib.sha256(b'hello' * 1000).hexdigest()}
    assert (tmp_path / 'out' / 'b.bin').read_bytes() == source.read_bytes()


def test_tail_lines(tmp_path):
    path = tmp_path / 'log'
    path.write_text(''.join(f'line {i}\n' for i in range(10000)))
    assert io.tail_lines(path, 3) == ['line 9997', 'line 9998', 'line 9999']
    assert io.tail_lines(tmp_path / 'missing') == []


# --- slurm ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('text,seconds', [
    ('90', 5400), ('0', 0), ('45:30', 2730), ('24:00:00', 86400), ('01:00:00', 3600),
    ('1-0', 86400), ('2-03', 183600), ('1-02:03', 93780), ('1-02:03:04', 93784), ('00:59', 59),
    (' 12:00:00 ', 43200),
])
def test_parse_time_limit(text, seconds):
    value = slurm.parse_time_limit(text)
    assert value == seconds and isinstance(value, int)


def test_parse_time_limit_special():
    assert slurm.parse_time_limit('UNLIMITED') == math.inf
    assert slurm.parse_time_limit('infinite') == math.inf
    for bad in ('NOT_SET', 'INVALID', '', '1:2:3:4', '-5', '1-', 'abc', '1.5'):
        with pytest.raises(ValueError):
            slurm.parse_time_limit(bad)


def test_signal_seconds():
    assert slurm.signal_seconds(45, 86400) == 2700
    assert slurm.signal_seconds(45, 3600) == 900          # capped at a quarter of the limit
    assert slurm.signal_seconds(2000, 7 * 86400) == 65535  # Slurm's maximum
    assert slurm.signal_seconds(0, 86400) == 0
    assert slurm.signal_seconds(45, math.inf) == 2700


def fake_squeue(tmp_path, monkeypatch, body: str):
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    script = bindir / 'squeue'
    script.write_text('#!/bin/sh\n' + body + '\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')
    return script


def test_job_identity(monkeypatch):
    assert slurm.job_id() is None and slurm.worker_id() == f'local-{os.getpid()}'
    assert slurm.profile_label() == 'local'
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'tillicum')
    monkeypatch.setenv('SLURM_JOB_PARTITION', 'gpu-h200')
    monkeypatch.setenv('SLURMD_NODENAME', 'g3001')
    monkeypatch.setenv('SLURM_JOB_START_TIME', '1763000000')
    assert slurm.worker_id() == '4242' and slurm.profile_label() == 'tillicum/gpu-h200'
    assert slurm.node_name() == 'g3001' and slurm.job_start_time() == 1763000000.0


def test_seconds_remaining_from_end_time(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_END_TIME', '1000100')
    assert slurm.seconds_remaining(now=1000000) == 100
    assert slurm.seconds_remaining(now=2000000) == 0
    monkeypatch.setenv('SLURM_JOB_END_TIME', 'garbage')
    assert slurm.seconds_remaining(default=7) == 7


@pytest.mark.parametrize('output,expected', [
    ('1-02:03:04', 93784.0), ('45:10', 2710.0), ('23:59:59', 86399.0), ('UNLIMITED', math.inf),
    ('NOT_SET', None), ('', None), ('12:00\n13:00', None),
])
def test_seconds_remaining_from_squeue(tmp_path, monkeypatch, output, expected):
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    fake_squeue(tmp_path, monkeypatch, f'[ "$2 $3 $4 $5 $6" = "-j 77 -o %L " ] || exit 9\nprintf "%b\\n" "{output}"')
    assert slurm.seconds_remaining(default=None) == expected


def test_seconds_remaining_without_slurm(monkeypatch, tmp_path):
    assert slurm.seconds_remaining(default=123) == 123
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    monkeypatch.setenv('PATH', str(tmp_path))  # no squeue at all
    assert slurm.seconds_remaining(default=5) == 5


@pytest.mark.parametrize('body,expected', [
    ('echo RUNNING', True), ('echo PENDING', True), ('echo COMPLETING', True), ('echo SUSPENDED', True),
    ('echo COMPLETED', False), ('echo TIMEOUT', False), ('echo CANCELLED', False), ('echo OUT_OF_MEMORY', False),
    ('true', None), ('echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1', False),
    ('echo "Socket timed out on send/recv operation" >&2; exit 1', None), ('echo WEIRD_STATE', None),
    ('sleep 5', None),
])
def test_is_job_alive(tmp_path, monkeypatch, body, expected):
    fake_squeue(tmp_path, monkeypatch, body)
    assert slurm.is_job_alive('123', timeout=1.0) is expected


def test_squeue_is_asked_for_all_states_without_the_callers_filters(tmp_path, monkeypatch):
    """pool-3: a running job hidden by SQUEUE_STATES / SQUEUE_PARTITION prints nothing with exit 0; that must
    neither be read as "finished" nor reach squeue at all."""
    fake_squeue(tmp_path, monkeypatch,
                'case " $* " in *" -t all "*) ;; *) exit 9;; esac\n'
                '[ -z "${SQUEUE_STATES}${SQUEUE_PARTITION}${SQUEUE_USERS}" ] || exit 0\n'
                'echo RUNNING')
    monkeypatch.setenv('SQUEUE_STATES', 'PENDING')
    monkeypatch.setenv('SQUEUE_PARTITION', 'cpu')
    monkeypatch.setenv('SQUEUE_USERS', 'nobody')
    assert slurm.is_job_alive('123') is True
    slurm.reset_job_listing()
    fake_squeue(tmp_path, monkeypatch, '[ -z "${SQUEUE_STATES}" ] || exit 0\necho "123|RUNNING|anytok-pool"')
    assert slurm.job_states() == {'123': 'RUNNING'}
    # even if the filter did get through, an empty answer is unknown, never "finished"
    fake_squeue(tmp_path, monkeypatch, 'exit 0')
    assert slurm.is_job_alive('123') is None


def test_job_listing_is_one_cached_call_per_minute(tmp_path, monkeypatch):
    counter = tmp_path / 'calls'
    fake_squeue(tmp_path, monkeypatch,
                f'echo x >> {counter}\nprintf "%s\\n" "11|RUNNING|anytok-pool" "12|COMPLETED|anytok-pool" '
                '"13_4|PENDING|my job | with bar"')
    now = [1000.0]
    monkeypatch.setattr(slurm, '_clock', lambda: now[0])
    slurm.reset_job_listing()
    expected = {'11': ('RUNNING', 'anytok-pool'), '12': ('COMPLETED', 'anytok-pool'),
                '13_4': ('PENDING', 'my job | with bar')}
    assert slurm.job_table() == expected
    assert slurm.job_states() == {'11': 'RUNNING', '12': 'COMPLETED', '13_4': 'PENDING'}
    now[0] += 59
    assert slurm.job_table() == expected
    assert len(counter.read_text().split()) == 1
    now[0] += 2                      # the minute is over
    slurm.job_table()
    assert len(counter.read_text().split()) == 2
    assert [slurm.state_alive(s) for s in ('RUNNING', 'COMPLETED', 'WEIRD', 'pending ')] == [True, False, None, True]


@pytest.mark.parametrize('body', ['exit 1', 'echo "garbage line"', 'sleep 5'])
def test_job_listing_failure_is_none_and_cached(tmp_path, monkeypatch, body):
    counter = tmp_path / 'calls'
    fake_squeue(tmp_path, monkeypatch, f'echo x >> {counter}\n{body}')
    monkeypatch.setattr(slurm, '_clock', lambda: 5.0)
    assert slurm.job_table(timeout=1.0) is None and slurm.job_states() is None
    assert len(counter.read_text().split()) == 1          # a broken squeue is not hammered either
    slurm.reset_job_listing()
    monkeypatch.setenv('PATH', str(tmp_path / 'nothing'))
    assert slurm.job_table() is None                      # no squeue command at all


def test_listing_with_no_jobs_is_an_empty_table_not_a_failure(tmp_path, monkeypatch):
    fake_squeue(tmp_path, monkeypatch, 'exit 0')
    assert slurm.job_table() == {}


def test_is_job_alive_unknown_ids_and_missing_command(monkeypatch, tmp_path):
    assert slurm.is_job_alive('local-55') is None
    assert slurm.is_job_alive(None) is None
    assert slurm.is_job_alive('1; rm -rf /') is None
    monkeypatch.setenv('PATH', str(tmp_path))
    assert slurm.is_job_alive('123') is None
    assert slurm.is_job_alive('123_4') is None


def test_sbatch_resources(make_tmp_config):
    cfg = make_tmp_config(HYAK_PARTITION_POOL='gpu-h200-mig', HYAK_TIME_POOL='12:00:00', HYAK_ACCOUNT_POOL='grp',
                          HYAK_QOS_PROBE='normal', HYAK_GRES_CORPUS='none').cfg
    train = slurm.sbatch_resources(cfg, 'train-uniform')
    assert train == {'partition': 'gpu-h200', 'qos': 'normal', 'account': None, 'gres': 'gpu:1', 'cpus': '8',
                     'mem': cfg['HYAK_MEM'] or None, 'time': '24:00:00', 'time_seconds': 86400, 'margin_minutes': 45,
                     'signal_seconds': 2700, 'job_name': 'anytok-train-uniform', 'singleton': True,
                     'debug': False, 'sbatch_file': '07_train.sbatch', 'run': 'uniform'}
    pool = slurm.sbatch_resources(cfg, 'pool')
    assert (pool['partition'], pool['time'], pool['account'], pool['singleton']) == ('gpu-h200-mig', '12:00:00', 'grp', False)
    debug_pool = slurm.sbatch_resources(cfg, 'pool', debug=True)
    assert (debug_pool['qos'], debug_pool['time'], debug_pool['signal_seconds']) == ('debug', '01:00:00', 900)
    probe = slurm.sbatch_resources(cfg, 'probe')
    assert (probe['qos'], probe['time']) == ('normal', '01:00:00')  # per-stage QoS beats the debug default
    assert slurm.sbatch_resources(cfg, 'probe', debug=True)['qos'] == 'debug'
    assert slurm.sbatch_resources(cfg, 'corpus')['gres'] is None
    assert slurm.sbatch_resources(cfg, 'eval-rl')['run'] == 'rl'
