"""hyak.corpus.build: per-bucket build/verify/retry/lock logic and the child-process orchestration."""
from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zlib

import pytest

from hyak.corpus import build, sources
from hyak.tests.test_corpus_fixtures import freeze_gzip_time, synthetic_records

TINY = (60000, 15000)


def _fake_stream(kind='fineweb', count=3000, fail_after=None, failures=None, exc=None):
    """A records() replacement; optionally raises ``exc`` after ``fail_after`` records for the first ``failures`` calls."""
    state = {'calls': 0}

    def records(spec, split):
        state['calls'] += 1
        call = state['calls']
        for i, record in enumerate(synthetic_records(spec['kind'], count, seed=zlib.crc32(split.encode()) % 97, prefix=split)):
            if fail_after is not None and i == fail_after and call <= (failures or 0):
                raise exc
            yield copy.deepcopy(record)

    records.state = state
    return records


def _build(tmp_path, monkeypatch, bucket='papers-en', size='full', **kwargs):
    monkeypatch.setattr(sources, 'records', kwargs.pop('records', None) or _fake_stream())
    return build.run_bucket(tmp_path, size, bucket, test_scale=TINY, sleep=lambda s: None, **kwargs)


def test_parse_test_scale_and_selection():
    assert build.parse_test_scale('200000:50000') == (200000, 50000)
    for bad in ('200000', '0:5', 'a:b', '1:2:3'):
        with pytest.raises(ValueError):
            build.parse_test_scale(bad)
    tasks = build.selected_tasks(None, None)
    assert [spec['name'] for _, spec in tasks] == list(sources.bucket_specs())
    tasks = build.selected_tasks(['extra'], ['qa-en'], (10, 20))
    assert tasks == [('extra', {'name': 'qa-en', 'kind': 'qa', 'train_target': 10, 'eval_target': 20})]
    with pytest.raises(ValueError, match='not in sizes'):
        build.selected_tasks(['extra'], ['web-en'])


def test_run_bucket_builds_verifies_and_skips(tmp_path, monkeypatch):
    code, result = _build(tmp_path, monkeypatch)
    assert code == build.EXIT_OK and result['status'] == 'built', result
    directory = tmp_path / 'full-v1' / 'papers-en'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['test_scale'] == build.test_scale_marker(TINY)
    assert manifest['spec'] == dict(sources.plan_for_size('full')[-1], train_target=TINY[0], eval_target=TINY[1])
    assert manifest['hyak']['port_of_sha256'] == sources.ORIGINAL_SOURCE_SHA256
    assert manifest['hyak']['zlib']['version'] and 'packages' in manifest['hyak']
    stored = json.loads((tmp_path / '.hyak' / 'results' / 'full__papers-en.json').read_text())
    assert stored['status'] == 'built' and stored['summary']['train']['documents'] == manifest['counts']['train_docs']
    assert not (tmp_path / '.hyak' / 'progress' / 'full__papers-en.json').exists()
    assert not (tmp_path / '.hyak' / 'locks' / 'full__papers-en').exists()
    code, result = _build(tmp_path, monkeypatch)
    assert code == build.EXIT_OK and result['status'] == 'verified-existing'


def test_existing_bucket_that_fails_verification_is_never_rebuilt(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    train = tmp_path / 'full-v1' / 'papers-en' / 'train.jsonl.gz'
    data = bytearray(train.read_bytes())
    data[-12] ^= 0xFF
    train.write_bytes(bytes(data))
    stream = _fake_stream()
    code, result = _build(tmp_path, monkeypatch, records=stream)
    assert code == build.EXIT_FAILED and result['status'] == 'verify-failed'
    assert 'Checksum mismatch' in result['error'] and 'delete' in result['error']
    assert stream.state['calls'] == 0


def test_production_run_rejects_test_scale_bucket(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    monkeypatch.setattr(sources, 'records', _fake_stream())
    code, result = build.run_bucket(tmp_path, 'full', 'papers-en', sleep=lambda s: None)
    assert code == build.EXIT_FAILED and result['status'] == 'verify-failed'
    assert 'test-scale bucket' in result['error'] or 'spec is' in result['error']


def test_transient_failure_restarts_from_zero_with_identical_content(tmp_path, monkeypatch):
    freeze_gzip_time(monkeypatch)
    clean_root, retry_root = tmp_path / 'clean', tmp_path / 'retry'
    _build(clean_root, monkeypatch)
    stream = _fake_stream(fail_after=10, failures=2, exc=ConnectionResetError('connection reset by peer'))
    code, result = _build(retry_root, monkeypatch, records=stream, attempts=3)
    assert code == build.EXIT_OK and result['attempts'] == 3 and len(result['errors']) == 2
    for name in ('train.jsonl.gz', 'eval.jsonl.gz'):
        assert ((retry_root / 'full-v1/papers-en' / name).read_bytes()
                == (clean_root / 'full-v1/papers-en' / name).read_bytes())
    assert sorted(p.name for p in (retry_root / 'full-v1/papers-en').iterdir()) == [
        'eval.jsonl.gz', 'manifest.json', 'train.jsonl.gz']


def test_permanent_failures_are_not_retried(tmp_path, monkeypatch):
    stream = _fake_stream(count=20)  # too short: "Source exhausted before target"
    code, result = _build(tmp_path, monkeypatch, records=stream, attempts=3)
    assert code == build.EXIT_FAILED and result['attempts'] == 1 and 'Source exhausted' in result['error']
    assert not any((tmp_path / 'full-v1' / 'papers-en').glob('*.tmp'))
    stream = _fake_stream(fail_after=5, failures=9, exc=sources.SourceUnavailable('HTTP 404 for u'))
    code, result = _build(tmp_path / 'b', monkeypatch, records=stream, attempts=3)
    assert code == build.EXIT_FAILED and result['attempts'] == 1


def test_transient_classification():
    import zlib
    assert build.is_transient(ConnectionResetError())
    assert build.is_transient(EOFError())
    assert build.is_transient(zlib.error())
    assert build.is_transient(json.JSONDecodeError('x', 'y', 0))
    assert not build.is_transient(sources.SourceUnavailable('gone'))
    assert not build.is_transient(RuntimeError('Source exhausted before target for x'))
    assert not build.is_transient(UnicodeEncodeError('utf-8', 'x', 0, 1, 'surrogate'))
    assert not build.is_transient(KeyError('text'))
    wrapped = OSError('wrapper')
    wrapped.__cause__ = sources.SourceUnavailable('gone')
    assert not build.is_transient(wrapped)

    class FakeHub(Exception):
        pass

    FakeHub.__module__ = 'huggingface_hub.errors'
    assert build.is_transient(FakeHub())
    # corpus-9 (a): disk full, quota, read-only and permission errors are permanent
    import errno
    for number in (errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES):
        assert not build.is_transient(OSError(number, os.strerror(number)))
    assert build.is_transient(OSError(errno.ECONNRESET, 'reset'))


def test_failed_progress_write_does_not_abort_the_build(tmp_path, monkeypatch):
    """corpus-9 (b): the dashboard progress file is best effort."""
    real = build.atomic_write_json

    def flaky(path, obj, *args, **kwargs):
        if '/progress/' in str(path):
            raise OSError(28, 'No space left on device')
        return real(path, obj, *args, **kwargs)

    monkeypatch.setattr(build, 'atomic_write_json', flaky)
    code, result = _build(tmp_path, monkeypatch)
    assert code == build.EXIT_OK and result['status'] == 'built' and result['attempts'] == 1


def test_validated_packages_are_the_data_pins():
    """corpus-2: one source of truth: hyak.code.pinned.DATA_PINS (what setup_data_env.sh installs)."""
    from hyak.code import pinned
    pins = {build.normalise_package_name(name): version for name, version in pinned.DATA_PINS}
    assert build.VALIDATED_PACKAGES == pins and len(pins) == len(pinned.DATA_PINS)
    assert set(build.VALIDATED_PACKAGES) == set(build.RECORDED_PACKAGES)     # every pin is checked and recorded
    assert build.normalise_package_name('HuggingFace_Hub') == 'huggingface-hub'
    assert build.normalise_package_name('hf.xet') == 'hf-xet'


def test_validated_package_set(monkeypatch, tmp_config, capsys):
    """corpus-2: a production build refuses another package set unless a reason is given and recorded."""
    assert build.package_problems(dict(build.VALIDATED_PACKAGES)) == []
    wrong = dict(build.VALIDATED_PACKAGES, datasets='5.3.0', urllib3=None)
    problems = build.package_problems(wrong)
    assert problems == ['datasets 5.3.0 (validated: 5.2.0)', 'urllib3 not installed (validated: 2.8.0)']
    monkeypatch.setattr(build, '_check_data_env', lambda: None)
    monkeypatch.setattr(build, 'package_versions', lambda: wrong)
    assert build.main(['--buckets', 'web-en']) == build.EXIT_USAGE
    err = capsys.readouterr().err
    assert 'validated package set' in err and 'datasets 5.3.0' in err and '--allow-unvalidated-env' in err
    # verify-only never refuses (no download)
    builder = build.Builder(tmp_config.paths.corpus_root, [], workers=1, unvalidated_env_reason='approved by the user')
    assert builder.command('full', 'web-en')[-2:] == ['--allow-unvalidated-env', 'approved by the user']


def test_unvalidated_reason_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    code, _ = _build(tmp_path, monkeypatch, env_note={'packages_validated': False, 'unvalidated_env_reason': 'why'})
    manifest = json.loads((tmp_path / 'full-v1' / 'papers-en' / 'manifest.json').read_text())
    assert code == build.EXIT_OK and manifest['hyak']['packages_validated'] is False
    assert manifest['hyak']['unvalidated_env_reason'] == 'why'


def test_rebuild_of_a_lost_bucket_is_byte_identical(tmp_path, monkeypatch):
    """Recorded gzip header times are reused: a deleted bucket comes back with the same SHA-256."""
    monkeypatch.setattr(sources, 'records', _fake_stream())
    import shutil
    assert build.run_bucket(tmp_path, 'full', 'papers-en', test_scale=TINY)[0] == build.EXIT_OK
    directory = tmp_path / 'full-v1' / 'papers-en'
    first = json.loads((directory / 'manifest.json').read_text())
    times = first['hyak']['gzip_mtime']
    assert build.gzip_header_mtime(directory / 'train.jsonl.gz') == times['train.jsonl.gz']
    shutil.rmtree(directory)
    time.sleep(1.1)  # a new build time would differ
    assert build.run_bucket(tmp_path, 'full', 'papers-en', test_scale=TINY)[0] == build.EXIT_OK
    again = json.loads((directory / 'manifest.json').read_text())
    assert again['files'] == first['files'] and again['hyak']['gzip_mtime']['source'].endswith('papers-en.json')
    # from an input/manifest.json elsewhere (the record that outlives a purged corpus root)
    other = tmp_path / 'other'
    record = {'kind': 'anytok_hyak_inputs', 'files': {}, 'provenance': {'corpus': {'gzip_mtime': {
        'papers-en': {'train.jsonl.gz': times['train.jsonl.gz'], 'eval.jsonl.gz': times['eval.jsonl.gz']}}}}}
    (tmp_path / 'inputs.json').write_text(json.dumps(record))
    assert build.run_bucket(other, 'full', 'papers-en', test_scale=TINY, time_sources=[tmp_path / 'inputs.json'])[0] == 0
    assert json.loads((other / 'full-v1' / 'papers-en' / 'manifest.json').read_text())['files'] == first['files']


def test_fixed_mtime_writes_the_gzip_open_bytes(tmp_path, monkeypatch):
    freeze_gzip_time(monkeypatch, 1_792_695_906.0)
    with gzip.open(tmp_path / 'a.jsonl.gz.tmp', 'wb') as handle:
        handle.write(b'{"id": "x", "text": "y"}\n' * 50)
    shim = build._FixedMtimeGzip({'b.jsonl.gz': 1_792_695_906})
    with shim.open(tmp_path / 'b.jsonl.gz.tmp', 'wb') as handle:
        handle.write(b'{"id": "x", "text": "y"}\n' * 50)
    a, b = (tmp_path / 'a.jsonl.gz.tmp').read_bytes(), (tmp_path / 'b.jsonl.gz.tmp').read_bytes()
    name = len('a.jsonl.gz.tmp') + 1
    assert a[:10] == b[:10]                       # magic, flags, header time, xfl, os
    assert a[10 + name:] == b[10 + name:]         # the deflate stream and trailer (only the stored names differ)
    assert shim.GzipFile is gzip.GzipFile


def test_planned_gzip_times(tmp_path):
    production = build.planned_gzip_mtimes(tmp_path, 'full', 'web-en', now=1_800_000_000)
    assert production['times'] == {'train.jsonl.gz': 1_800_000_000, 'eval.jsonl.gz': 1792695906}
    scaled = build.planned_gzip_mtimes(tmp_path, 'full', 'web-en', test_scale=TINY, now=1_800_000_000)
    assert scaled['times'] == {'train.jsonl.gz': 1_800_000_000, 'eval.jsonl.gz': 1_800_000_000}


def test_probe_fails_fast_with_exit_2(monkeypatch):
    seen = []

    def probe_one(spec):
        seen.append((sources.OPEN_ATTEMPTS, sources.OPEN_BACKOFF_SECONDS))
        raise ConnectionError('no route to host')

    monkeypatch.setattr(sources, 'probe_one', probe_one)
    assert build.run_probe(['specialty']) == build.EXIT_USAGE
    assert seen == [(2, 5.0), (2, 5.0)] and sources.OPEN_ATTEMPTS == 5


def test_lock_blocks_a_second_owner_and_takes_over_dead_ones(tmp_path):
    first = build.BucketLock(tmp_path, 'full', 'web-en')
    assert first.acquire() is None
    second = build.BucketLock(tmp_path, 'full', 'web-en')
    owner = second.acquire()
    assert owner['pid'] == os.getpid()
    first.release()
    assert second.acquire() is None
    second.release()
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    stale = build.BucketLock(tmp_path, 'full', 'web-en')
    stale.owner['pid'] = dead.pid
    stale.owner['job_id'] = f'local-{dead.pid}'
    assert stale.acquire() is None  # writes owner.json with the dead pid
    stale.held = False
    third = build.BucketLock(tmp_path, 'full', 'web-en')
    assert third.acquire() is None
    third.release()


def test_locked_bucket_is_skipped(tmp_path, monkeypatch, capsys):
    lock = build.BucketLock(tmp_path, 'full', 'papers-en')
    assert lock.acquire() is None
    code, result = _build(tmp_path, monkeypatch)
    assert code == build.EXIT_LOCKED and result['status'] == 'locked'
    # corpus-5: the message names the lock directory and the remedy
    assert result['lock'] == str(lock.path)
    out = capsys.readouterr().out
    assert str(lock.path) in out and f'rm -r {lock.path}' in out
    lock.release()


RACE = r'''
import json, os, sys, time
from pathlib import Path
from hyak.corpus import build
root, name = Path(sys.argv[1]), sys.argv[2]
flags = root / 'flags'
original = build.BucketLock._stale
def stale(self, owner):
    result = original(self, owner)
    if owner and owner.get('pid') == 999999:          # both judge the dead owner, then race
        (flags / f'{name}.judged').write_text(str(result))
        other = 'second' if name == 'first' else 'first'
        deadline = time.time() + 10
        while not (flags / f'{other}.judged').exists() and time.time() < deadline:
            time.sleep(0.01)
        if name == 'second':
            time.sleep(0.7)
    return result
build.BucketLock._stale = stale
lock = build.BucketLock(root, 'full', 'web-en')
owner = lock.acquire()
time.sleep(1.5)
current = json.loads((lock.path / 'owner.json').read_text())['pid']
print(json.dumps({'name': name, 'held': lock.held, 'owner_now': current, 'pid': os.getpid()}), flush=True)
'''


def test_stale_lock_race_has_exactly_one_winner(tmp_path, repo_root):
    """corpus-4: two processes that both judged the same owner stale; only one may hold the lock."""
    from hyak.common import slurm
    root = tmp_path / 'corpus'
    (root / 'flags').mkdir(parents=True)
    dead = build.BucketLock(root, 'full', 'web-en')
    dead.owner.update(job_id='local-999999', host=slurm.node_name(), pid=999999)
    assert dead.acquire() is None
    dead.held = False
    dead._beat_stop.set()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(repo_root))
    procs = [subprocess.Popen([sys.executable, '-c', RACE, str(root), name], env=env, cwd=str(repo_root),
                              stdout=subprocess.PIPE, text=True) for name in ('first', 'second')]
    rows = [json.loads(p.communicate(timeout=60)[0]) for p in procs]
    holders = [r for r in rows if r['held']]
    assert len(holders) == 1, rows
    assert all(r['owner_now'] == holders[0]['pid'] for r in rows)
    assert sorted(p.name for p in (root / '.hyak' / 'locks').iterdir()) == ['full__web-en']  # nothing renamed


def test_release_by_a_former_owner_leaves_the_new_lock(tmp_path):
    first = build.BucketLock(tmp_path, 'full', 'web-en')
    assert first.acquire() is None
    second = build.BucketLock(tmp_path, 'full', 'web-en')
    second._take()                      # as if second had taken the lock over (first looked dead)
    first.release()
    assert (first.path / 'owner.json').is_file()
    assert json.loads((first.path / 'owner.json').read_text())['started_unix'] == second.owner['started_unix']
    second.release()
    assert not first.path.exists()


def test_lock_liveness_rules(tmp_path, monkeypatch):
    """corpus-5: same host -> pid decides for every job id; other host without Slurm answer -> heartbeat age."""
    from hyak.common import slurm
    lock = build.BucketLock(tmp_path, 'full', 'web-en')
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    monkeypatch.setattr(slurm, 'is_job_alive', lambda job, timeout=20.0: True)
    # a killed worker of a still running job on this host
    assert lock._stale({'job_id': '4242', 'host': slurm.node_name(), 'pid': dead.pid, 'started_unix': time.time()})
    assert not lock._stale({'job_id': '4242', 'host': slurm.node_name(), 'pid': os.getpid(), 'started_unix': 0})
    # a login-node build on the other login node: liveness unknown, heartbeat decides
    monkeypatch.setattr(slurm, 'is_job_alive', lambda job, timeout=20.0: None)
    lock.path.mkdir(parents=True)
    other = {'job_id': 'local-123', 'host': 'tillicum-login02', 'pid': 123, 'started_unix': time.time() - 3600}
    (lock.path / 'owner.json').write_text(json.dumps(other))
    assert not lock._stale(other)                                   # fresh heartbeat (file just written)
    old = time.time() - build.LOCK_STALE_SECONDS - 60
    os.utime(lock.path / 'owner.json', (old, old))
    assert lock._stale(other)
    assert lock.acquire() is None and lock.took_over == other       # taken over without renaming the directory
    lock.release()


def test_lock_heartbeat_refreshes_the_owner_file(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'LOCK_HEARTBEAT_SECONDS', 0.05)
    lock = build.BucketLock(tmp_path, 'full', 'web-en')
    assert lock.acquire() is None
    old = time.time() - 3600
    os.utime(lock.path / 'owner.json', (old, old))
    time.sleep(0.3)
    assert time.time() - (lock.path / 'owner.json').stat().st_mtime < 5
    lock.release()


# --- verify_bucket invariants ------------------------------------------------------------------------
def _rewrite(path, transform):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    rows = transform(rows)
    with gzip.open(path, 'wb') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False).encode() + b'\n')
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][path.name] = sources._sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


@pytest.mark.parametrize('case,message', [
    ('swap_sides', 'wrong side of the held-out rule'),
    ('key_order', 'in that order'),
    ('extra_doc', 'count mismatch'),
    ('short_doc', 'outside 100'),
])
def test_verify_bucket_invariants(tmp_path, monkeypatch, case, message):
    monkeypatch.setattr(sources, 'records', _fake_stream())
    spec = dict(sources.bucket_specs()['web-en'][1], train_target=6000, eval_target=1500)
    sources.prepare_bucket(tmp_path, spec)
    directory = tmp_path / 'web-en'
    assert build.verify_bucket(directory, spec)['train']['documents'] > 0
    train, held = directory / 'train.jsonl.gz', directory / 'eval.jsonl.gz'
    if case == 'swap_sides':
        with gzip.open(held, 'rt') as handle:
            first_eval = json.loads(handle.readline())
        _rewrite(train, lambda rows: [first_eval] + rows)
    elif case == 'key_order':
        _rewrite(train, lambda rows: [{'text': r['text'], 'id': r['id']} for r in rows])
    elif case == 'extra_doc':
        _rewrite(train, lambda rows: rows + [rows[-1]])
    else:
        _rewrite(train, lambda rows: rows[:-1] + [{'id': rows[-1]['id'], 'text': 'tiny'}])
    with pytest.raises(build.VerificationError, match=message):
        build.verify_bucket(directory, spec)


def test_verify_bucket_stopping_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, 'records', _fake_stream())
    spec = dict(sources.bucket_specs()['web-en'][1], train_target=6000, eval_target=1500)
    sources.prepare_bucket(tmp_path, spec)
    directory = tmp_path / 'web-en'

    extra_id = next(f'extra-train-doc-{i}' for i in range(100) if not sources.held_out(f'extra-train-doc-{i}'))

    def add_one_more(rows):
        return rows + [{'id': extra_id, 'text': 'q' * 400}]

    manifest = _rewrite(directory / 'train.jsonl.gz', add_one_more)
    manifest['counts']['train_docs'] += 1
    manifest['counts']['train_bytes'] += 400
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with pytest.raises(build.VerificationError, match='stopping rule'):
        build.verify_bucket(directory, spec)


# --- orchestration (children replaced by tiny scripts) ---------------------------------------------------
class FakeWriter:
    def __init__(self):
        self.updates, self.events = [], []

    def worker_row(self, **kwargs):
        return {'id': kwargs.get('suffix'), **kwargs}

    def update(self, completed, **kwargs):
        self.updates.append(dict(kwargs, completed=completed))
        return self.updates[-1]

    def event(self, obj):
        self.events.append(obj)


CHILD = r'''
import json, os, sys, time
from pathlib import Path
root, size, bucket, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
if mode == 'sleep':
    time.sleep(60)
code = 0 if mode == 'ok' else 1
status = 'built' if mode == 'ok' else 'failed'
out = root / '.hyak' / 'results' / f'{size}__{bucket}.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'status': status, 'pid': os.getpid(), 'started_at': 'x', 'seconds': 0.1,
                           'error': None if code == 0 else 'boom'}))
sys.exit(code)
'''


class ScriptBuilder(build.Builder):
    def __init__(self, *args, modes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = modes or {}

    def command(self, size, bucket):
        return [sys.executable, '-c', CHILD, str(self.corpus_root), size, bucket, self.modes.get(bucket, 'ok')]


def test_builder_runs_children_and_reports(tmp_path):
    tasks = build.selected_tasks(['extra'], None, TINY)
    writer = FakeWriter()
    builder = ScriptBuilder(tmp_path, tasks, workers=2, test_scale=TINY, writer=writer, poll_seconds=0.05,
                            modes={'qa-en': 'fail'})
    assert builder.tasks[0][1]['name'] == 'math-en'  # equal targets keep build order
    code = builder.run()
    assert code == build.EXIT_FAILED
    assert builder.outcome['math-en']['status'] == 'built' and builder.outcome['qa-en']['status'] == 'failed'
    final = writer.updates[-1]
    assert final['state'] == 'failed' and 'qa-en' in final['error']
    assert set(final['buckets']) == set(sources.bucket_specs()) and final['buckets']['math-en']['state'] == 'built'
    assert [e['bucket'] for e in writer.events if e['event'] == 'bucket_end']
    logs = sorted(p.name for p in (tmp_path / '.hyak' / 'logs').iterdir())
    assert logs == ['extra__books-en.log', 'extra__math-en.log', 'extra__qa-en.log']


def test_builder_paused_when_not_all_23_complete(tmp_path):
    tasks = build.selected_tasks(['specialty'], None, TINY)
    writer = FakeWriter()
    code = ScriptBuilder(tmp_path, tasks, workers=1, test_scale=TINY, writer=writer, poll_seconds=0.05).run()
    assert code == build.EXIT_OK and writer.updates[-1]['state'] == 'paused'


def test_complete_manifests_but_not_started_is_paused(tmp_path, monkeypatch):
    """corpus-6: buckets that were never started (or verified) are not reported complete."""
    from hyak.tests.test_corpus_fixtures import write_pool
    write_pool(tmp_path)
    assert build.Builder(tmp_path, [], workers=1).write_summaries() == ['full', 'extra', 'specialty']
    tasks = build.selected_tasks(None, None)
    writer = FakeWriter()
    builder = ScriptBuilder(tmp_path, tasks, workers=2, verify_only=True, writer=writer, poll_seconds=0.05,
                            min_remaining_seconds=3600)
    monkeypatch.setattr(builder, 'time_left', lambda: 300.0)   # below even the short verify-only guard
    assert builder.run() == build.EXIT_OK
    assert writer.updates[-1]['state'] == 'paused' and 'nothing was verified' in writer.updates[-1]['error']
    # a verify-only run with 30 minutes left starts its children (guard min(60 min, 10 min))
    writer = FakeWriter()
    builder = ScriptBuilder(tmp_path, tasks, workers=4, verify_only=True, writer=writer, poll_seconds=0.05,
                            min_remaining_seconds=3600)
    monkeypatch.setattr(builder, 'time_left', lambda: 1800.0)
    builder.run()
    assert not builder.not_started and len(builder.outcome) == 23


def test_builder_does_not_start_buckets_without_time(tmp_path, monkeypatch):
    tasks = build.selected_tasks(['specialty'], None, TINY)
    writer = FakeWriter()
    builder = ScriptBuilder(tmp_path, tasks, workers=1, test_scale=TINY, writer=writer, poll_seconds=0.05,
                            min_remaining_seconds=3600)
    monkeypatch.setattr(builder, 'time_left', lambda: 1800.0)
    assert builder.run() == build.EXIT_OK
    assert builder.not_started == ['medical-en', 'legal-en'] and not builder.outcome
    assert 'time limit' in writer.updates[-1]['error']


def test_builder_stops_children_on_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'TERM_GRACE_SECONDS', 5.0)
    tasks = build.selected_tasks(['specialty'], None, TINY)
    writer = FakeWriter()
    builder = ScriptBuilder(tmp_path, tasks, workers=2, test_scale=TINY, writer=writer, poll_seconds=0.05,
                            modes={'medical-en': 'sleep', 'legal-en': 'sleep'})
    timer = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    start = time.time()
    try:
        code = builder.run()
    finally:
        timer.cancel()
    assert code == build.EXIT_TERMINATED and time.time() - start < 30
    assert writer.updates[-1]['state'] == 'stopped'
    assert {r['status'] for r in builder.outcome.values()} <= {'terminated', 'failed'}


def test_summaries_written_for_complete_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, 'records', _fake_stream())
    for name in ('medical-en', 'legal-en'):
        assert build.run_bucket(tmp_path, 'specialty', name, test_scale=TINY)[0] == build.EXIT_OK
    builder = build.Builder(tmp_path, [], workers=1, test_scale=TINY)
    assert builder.write_summaries() == ['specialty']
    summary = json.loads((tmp_path / 'specialty-v1' / 'summary.json').read_text())
    assert summary['buckets'] == ['medical-en', 'legal-en'] and summary['size'] == 'specialty'
    assert sources.verify_corpus(tmp_path, 'specialty')['verified_buckets'] == 2
    verifier = build.Builder(tmp_path, [], workers=1, test_scale=TINY, verify_only=True)
    assert verifier.check_summaries() == (['specialty'], [])
    (tmp_path / 'specialty-v1' / 'summary.json').write_text('{}')
    assert verifier.check_summaries()[1]


# --- command line -----------------------------------------------------------------------------------------
def _cli(args, env_extra, cwd):
    env = dict(os.environ, PYTHONPATH=str(cwd), PYTHONDONTWRITEBYTECODE='1', **env_extra)
    return subprocess.run([sys.executable, '-m', 'hyak.corpus.build', *args], capture_output=True, text=True,
                          env=env, cwd=str(cwd), timeout=120)


def test_cli_refuses_test_scale_in_production_root(tmp_config, repo_root):
    env = {'HYAK_CONFIG': str(tmp_config.config_file), 'HYAK_MONITOR_ROOT': str(tmp_config.base / 'monitor')}
    result = _cli(['--test-scale', '100:100'], env, repo_root)
    assert result.returncode == build.EXIT_USAGE and 'separate --corpus-root' in result.stderr
    result = _cli(['--test-scale', '100:100', '--corpus-root', str(tmp_config.paths.corpus_root / 'mini')], env, repo_root)
    assert result.returncode == build.EXIT_USAGE
    result = _cli(['--buckets', 'nope'], env, repo_root)
    assert result.returncode == build.EXIT_USAGE and 'nope' in result.stderr
    result = _cli(['--worker', 'full', 'web-en'], env, repo_root)
    assert result.returncode == 2 and '--corpus-root' in result.stderr


def test_cli_verify_only_on_a_tiny_corpus(tmp_config, repo_root, monkeypatch):
    """End to end through real child processes (verify-only needs no network)."""
    monkeypatch.setattr(sources, 'records', _fake_stream())
    root = tmp_config.base / 'mini'
    for name in ('medical-en', 'legal-en'):
        build.run_bucket(root, 'specialty', name, test_scale=TINY)
    assert build.Builder(root, [], workers=1, test_scale=TINY).write_summaries() == ['specialty']
    env = {'HYAK_CONFIG': str(tmp_config.config_file), 'HYAK_MONITOR_ROOT': str(tmp_config.base / 'monitor')}
    result = _cli(['--verify-only', '--sizes', 'specialty', '--corpus-root', str(root), '--test-scale', '60000:15000',
                   '--workers', '2'], env, repo_root)
    assert result.returncode == build.EXIT_OK, result.stdout + result.stderr
    assert 'specialty/medical-en: verified-existing' in result.stdout
    status = json.loads((tmp_config.base / 'monitor' / 'corpus' / 'status.json').read_text())
    assert status['state'] == 'paused' and status['completed'] == 2 and status['total'] == 23
    assert status['buckets']['legal-en']['state'] == 'verified-existing'
    events = [json.loads(line) for line in (tmp_config.base / 'monitor' / 'corpus' / 'events.jsonl').read_text().splitlines()]
    assert any(e.get('event') == 'build_start' for e in events)
    result = _cli(['--verify-only', '--buckets', 'web-en', '--corpus-root', str(root), '--test-scale', '60000:15000'],
                  env, repo_root)
    assert result.returncode == build.EXIT_FAILED and 'missing' in result.stdout
