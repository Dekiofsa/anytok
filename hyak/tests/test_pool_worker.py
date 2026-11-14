"""hyak.pool.worker end to end with a fake scripts/tokenizer_pool.py (real processes, signals, exit codes)."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from hyak.pool import IDENTITY_KEYS, load_plan, manifest_problems
from hyak.tests.test_pool_support import make_corpus, make_test_plan, write_fake_trainer, write_tokenizer_library

REPO = Path(__file__).resolve().parents[2]


class Pool:
    """A test pool: plan, corpus, fake trainer, and a runner for ``python -m hyak.pool.worker``."""

    def __init__(self, tmp_config, count=4, vocab=300, **plan_kwargs):
        base = tmp_config.base
        self.cfg = tmp_config
        self.plan_path = base / 'test-plan.json'
        self.plan = make_test_plan(self.plan_path, count=count, vocab=vocab, **plan_kwargs)
        self.recipe = make_corpus(base / 'corpus', self.plan)
        self.pool_dir = base / 'testpool'
        self.monitor = base / 'monitor' / 'pool'
        self.fake = write_fake_trainer(base)
        self.library = write_tokenizer_library(base / 'library', self.plan)
        self.log = base / 'fake.log'
        self.state = base / 'fake-state'

    def args(self, *extra, workers=2):
        return ['--plan', str(self.plan_path), '--allow-test-plan', '--pool-dir', str(self.pool_dir),
                '--recipe', str(self.recipe), '--worker-script', str(self.fake), '--no-wheel-check',
                '--monitor-dir', str(self.monitor), '--workers', str(workers), '--threads', '4',
                '--poll-seconds', '0.05', '--heartbeat-seconds', '0.2', '--status-seconds', '0.3',
                '--stagger-seconds', '0', *extra]

    def env(self, **extra):
        env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1',
               'FAKE_TOKENIZERS': str(self.library), 'FAKE_LOG': str(self.log), 'FAKE_STATE': str(self.state)}
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def start(self, *extra, workers=2, **env):
        return subprocess.Popen([sys.executable, '-m', 'hyak.pool.worker', *self.args(*extra, workers=workers)],
                                cwd=str(REPO), env=self.env(**env), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)

    def run(self, *extra, workers=2, timeout=120, **env):
        process = self.start(*extra, workers=workers, **env)
        out, err = process.communicate(timeout=timeout)
        return process.returncode, out, err

    def invocations(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def manifest(self, index):
        return json.loads((self.pool_dir / 'tokenizers' / f'{index:04d}' / 'manifest.json').read_text())

    def failure(self, index):
        return json.loads((self.pool_dir / 'tokenizers' / f'{index:04d}' / 'failure.json').read_text())

    def status(self):
        return json.loads((self.monitor / 'status.json').read_text())

    def events(self):
        return [json.loads(line) for line in (self.monitor / 'events.jsonl').read_text().splitlines()]


@pytest.fixture
def pool(tmp_config):
    return Pool(tmp_config)


def test_happy_path_generates_every_index_once_with_provenance(pool):
    code, out, err = pool.run()
    assert code == 0, err
    calls = pool.invocations()
    assert sorted(c['index'] for c in calls) == list(range(4))
    assert all(c['rayon'] == '4' and c['parallelism'] == 'true' for c in calls)
    plan, _ = load_plan(pool.plan_path, allow_test_plan=True)
    identities = set()
    for index in range(4):
        manifest = pool.manifest(index)
        folder = pool.pool_dir / 'tokenizers' / f'{index:04d}'
        assert manifest_problems(manifest, plan['configs'][index], plan, folder / 'tokenizer.json') == []
        assert sorted(p.name for p in folder.iterdir()) == ['manifest.json', 'tokenizer.json']
        provenance = manifest['provenance']
        assert provenance['plan_sha256'] == plan['plan_sha256'] and provenance['rayon_threads'] == 4
        assert provenance['production_plan'] is False and provenance['attempt'] == 1
        assert provenance['unigram_patches'] and provenance['wrapper'] == 'hyak.pool.worker'
        identities.add(json.dumps({k: provenance[k] for k in IDENTITY_KEYS}, sort_keys=True))
        # Manifest text format of the original wrapper: json.dumps(indent=2) + newline.
        text = (folder / 'manifest.json').read_text()
        assert text == json.dumps(json.loads(text), indent=2) + '\n'
    assert len(identities) == 1
    assert [p.name for p in (pool.pool_dir / 'claims').iterdir()] == []
    status = pool.status()
    assert status['state'] == 'complete' and status['stage'] == 'complete' and status['completed'] == 4
    assert status['total'] == 4 and status['workers'] == []
    events = pool.events()
    names = Counter(e.get('event') for e in events)
    assert names['tokenizer_complete'] == 4 and names['tokenizer_start'] == 4
    # status.json is rewritten, never appended (pool-11): no status lines, no workers rows in events.jsonl;
    # a small 'progress' event only when the counts change
    assert not [e for e in events if isinstance(e.get('workers'), list)]   # pool_job_start's workers is a count
    assert not [e for e in events if 'event' not in e]
    progress = [e for e in events if e.get('event') == 'progress']
    assert [e['completed'] for e in progress] == sorted({e['completed'] for e in progress})
    assert progress[-1]['completed'] == 4 and progress[-1]['total'] == 4
    assert progress[-1]['failed_indices'] == [] and 'pilots_complete' in progress[-1]
    # A second run finds nothing to do and starts no child (and no new progress: the counts did not change
    # within that job, it writes one progress event for its own start).
    code, _, err = pool.run()
    assert code == 0, err
    assert len(pool.invocations()) == 4
    again = [e for e in pool.events() if e.get('event') == 'progress'][len(progress):]
    assert len(again) == 1 and again[0]['completed'] == 4


def test_pilots_are_claimed_first(tmp_config):
    pool = Pool(tmp_config, count=6)
    pilots = pool.plan['pilot_indices']
    code, _, err = pool.run(workers=1)
    assert code == 0, err
    order = [c['index'] for c in pool.invocations()]
    assert order[:len(pilots)] == pilots
    assert sorted(order) == list(range(6))


def test_pilots_only_and_explicit_indices(tmp_config):
    pool = Pool(tmp_config, count=6)
    code, _, err = pool.run('--pilots-only', workers=1)
    assert code == 0, err
    assert sorted(c['index'] for c in pool.invocations()) == sorted(pool.plan['pilot_indices'])
    before = len(pool.invocations())
    code, _, err = pool.run('--indices', '5', '4', workers=1)
    assert code == 0, err
    expected = [i for i in (5, 4) if i not in pool.plan['pilot_indices']]
    assert [c['index'] for c in pool.invocations()][before:] == expected
    assert (pool.pool_dir / 'tokenizers' / '0005' / 'manifest.json').exists()


def test_failures_are_recorded_bounded_and_never_substituted(pool):
    modes = json.dumps({'1': 'fail'})
    code, out, err = pool.run('--max-attempts', '2', FAKE_MODES=modes)
    assert code == 3
    failure = pool.failure(1)
    assert failure['attempts'] == 1 and failure['config'] == pool.plan['configs'][1]
    assert 'exit status 1' in failure['error'] and 'simulated failure' in failure['error']
    assert not (pool.pool_dir / 'tokenizers' / '0001' / 'manifest.json').exists()
    assert [p.name for p in (pool.pool_dir / 'claims').iterdir()] == []
    assert pool.status()['failed_indices'] == [1]
    code, _, _ = pool.run('--max-attempts', '2', FAKE_MODES=modes)
    assert code == 3 and pool.failure(1)['attempts'] == 2 and len(pool.failure(1)['history']) == 2
    before = len(pool.invocations())
    code, _, _ = pool.run('--max-attempts', '2', FAKE_MODES=modes)
    assert code == 3 and len(pool.invocations()) == before       # exhausted: not retried
    assert pool.status()['exhausted_indices'] == [1]
    code, _, _ = pool.run('--max-attempts', '2', '--retry-failed')  # now succeeds
    assert code == 0
    assert pool.manifest(1)['provenance']['previous_failed_attempts'] == 2
    assert not (pool.pool_dir / 'tokenizers' / '0001' / 'failure.json').exists()
    counts = Counter(c['index'] for c in pool.invocations())
    assert counts[1] == 3 and all(counts[i] == 1 for i in (0, 2, 3))


def test_invalid_child_output_is_a_failure(pool):
    code, _, _ = pool.run(FAKE_MODES=json.dumps({'0': 'bad_vocab', '2': 'bad_slack'}))
    assert code == 3
    assert 'vocab_size_actual' in pool.failure(0)['error']
    assert 'total_used_bytes' in pool.failure(2)['error']
    assert not (pool.pool_dir / 'tokenizers' / '0000' / 'manifest.json').exists()


def test_completed_manifest_wins_and_invalid_manifest_blocks(pool):
    code, _, err = pool.run()
    assert code == 0, err
    tokenizer = pool.pool_dir / 'tokenizers' / '0002' / 'tokenizer.json'
    tokenizer.write_bytes(tokenizer.read_bytes() + b' ')
    manifest = pool.pool_dir / 'tokenizers' / '0003' / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['provenance']['recipe_sha256'] = '0' * 64
    manifest.write_text(json.dumps(data, indent=2) + '\n')
    before = len(pool.invocations())
    code, out, err = pool.run('--report')
    assert code == 0
    report = json.loads(out)
    assert set(report['blocked']) == {'2', '3'} or set(report['blocked']) == {2, 3}
    code, _, _ = pool.run()
    assert code == 3                                              # needs the user
    assert len(pool.invocations()) == before                      # nothing retrained
    assert tokenizer.exists() and manifest.exists()               # nothing deleted
    assert pool.status()['blocked_indices'] == [2, 3]


def test_no_new_work_when_job_time_is_short(pool):
    code, out, err = pool.run(SLURM_JOB_END_TIME=int(time.time()) + 600)
    assert code == 0, err
    assert pool.invocations() == []
    assert any(e.get('event') == 'slot_idle' and 'job time left' in e.get('reason', '') for e in pool.events())


def _one_family(config):
    config.update(algorithm='wordpiece', pretokenizer='regex', noise_per_million=None)


def test_family_estimate_defers_work_that_cannot_fit(tmp_config):
    pool = Pool(tmp_config, count=4, mutate=_one_family)
    code, _, err = pool.run('--indices', '0', workers=1)
    assert code == 0, err
    # The fake records elapsed_seconds 0.3, so the family needs 0.375 s; this job has less than 0.1 s beyond the margin.
    code, _, err = pool.run('--min-remaining-minutes', '0', '--kill-margin-seconds', '3599.9', workers=1,
                            SLURM_JOB_END_TIME=int(time.time()) + 3600)
    assert code == 0, err
    assert [c['index'] for c in pool.invocations()] == [0]
    assert any('deferred' in e.get('reason', '') for e in pool.events() if e.get('event') == 'slot_idle')
    code, _, err = pool.run(workers=1)
    assert code == 0, err
    assert sorted(c['index'] for c in pool.invocations()) == [0, 1, 2, 3]


def test_timeout_is_a_failure(pool):
    code, _, _ = pool.run('--indices', '0', '--tokenizer-timeout-hours', '0.0003', workers=1,
                          FAKE_MODES=json.dumps({'0': 'sleep30'}))
    assert code == 3
    assert 'timed out' in pool.failure(0)['error']


def test_sigterm_twice_interrupts_without_counting_a_failure(pool):
    process = pool.start('--indices', '0', '1', workers=2, FAKE_MODE='sleep60')
    deadline = time.time() + 30
    while time.time() < deadline and len(pool.invocations()) < 2:
        time.sleep(0.1)
    assert len(pool.invocations()) == 2
    process.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    assert process.poll() is None                 # first TERM: running tokenizers continue
    process.send_signal(signal.SIGTERM)
    out, err = process.communicate(timeout=60)
    assert process.returncode == 0, err
    for index in (0, 1):
        folder = pool.pool_dir / 'tokenizers' / f'{index:04d}'
        assert not (folder / 'failure.json').exists() and not (folder / 'manifest.json').exists()
    assert [p.name for p in (pool.pool_dir / 'claims').iterdir()] == []
    assert sum(e.get('event') == 'tokenizer_interrupted' for e in pool.events()) == 2
    for call in pool.invocations():                # the children are gone
        with pytest.raises(ProcessLookupError):
            os.kill(call['pid'], 0)


def test_first_sigterm_lets_running_tokenizers_finish(pool):
    process = pool.start('--indices', '0', '1', '2', workers=1, FAKE_DELAY='1.5')
    deadline = time.time() + 30
    while time.time() < deadline and not pool.invocations():
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    out, err = process.communicate(timeout=60)
    assert process.returncode == 0, err
    assert [c['index'] for c in pool.invocations()] == [0]          # explicit order; no new start after TERM
    finished = [i for i in range(4) if (pool.pool_dir / 'tokenizers' / f'{i:04d}' / 'manifest.json').exists()]
    assert finished == [0]                                           # the running one completed


def test_concurrent_jobs_train_each_index_exactly_once(tmp_config):
    pool = Pool(tmp_config, count=12)
    processes = [pool.start(workers=2, FAKE_DELAY='0.3', SLURM_JOB_ID='', SLURM_CLUSTER_NAME='test') for _ in range(3)]
    for process in processes:
        out, err = process.communicate(timeout=180)
        assert process.returncode == 0, err
    counts = Counter(c['index'] for c in pool.invocations())
    assert sorted(counts) == list(range(12)) and set(counts.values()) == {1}
    assert pool.status()['completed'] == 12 and pool.status()['state'] == 'complete'


def test_production_guards(tmp_config):
    pool = Pool(tmp_config)
    env = pool.env()

    def run(*args):
        result = subprocess.run([sys.executable, '-m', 'hyak.pool.worker', *args], cwd=str(REPO), env=env,
                                capture_output=True, text=True, timeout=60)
        return result.returncode, result.stderr

    code, err = run('--plan', str(pool.plan_path), '--pool-dir', str(pool.pool_dir))
    assert code == 2 and 'not the saved production plan' in err
    code, err = run('--plan', str(pool.plan_path), '--allow-test-plan')
    assert code == 2 and 'explicit --pool-dir' in err
    code, err = run('--plan', str(pool.plan_path), '--allow-test-plan', '--pool-dir', str(tmp_config.paths.pool_dir))
    assert code == 2 and 'production pool' in err
    code, err = run('--worker-script', str(pool.fake))
    assert code == 2 and 'only allowed with --allow-test-plan' in err
    code, err = run('--threads', '3')
    assert code == 2 and ('RAYON_NUM_THREADS=4' in err or 'pool-recipe' in err or 'recipe' in err)
    code, err = run()
    assert code == 2 and 'stage-inputs' in err          # production plan, no input/pool-recipe.json yet


def test_production_rejects_a_thread_change(tmp_config):
    pool = Pool(tmp_config)
    # A production-plan recipe that matches: copy the real plan recipe with rewritten paths.
    plan, _ = load_plan(REPO / 'plans/tokenizer-pool-1000-v1.json')
    recipe = json.loads(json.dumps(plan['recipe']))
    for bucket in recipe['buckets']:
        folder = tmp_config.base / 'prod-corpus' / Path(*Path(bucket['source_path']).parts[-2:])
        folder.mkdir(parents=True)
        (folder / 'train.jsonl.gz').write_bytes(b'')
        bucket['source_path'] = str(folder)
    recipe_path = tmp_config.paths.pool_recipe
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe))
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.worker', '--threads', '3'], cwd=str(REPO),
                            env=pool.env(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and 'RAYON_NUM_THREADS=4' in result.stderr
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.worker'], cwd=str(REPO),
                            env=pool.env(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and 'provenance.json' in result.stderr      # no wheel built yet
    # --report works without a wheel (login node) and writes nothing.
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.worker', '--report'], cwd=str(REPO),
                            env=pool.env(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['counts']['missing'] == 1000 and report['production_plan'] is True
    assert report['identity_checked'] is False and 'provenance.json' in report['identity_note']
    assert not tmp_config.paths.pool_dir.exists() and not (tmp_config.base / 'monitor' / 'pool').exists()
    # A swapped bucket path is refused.
    a, b = recipe['buckets'][0]['source_path'], recipe['buckets'][1]['source_path']
    recipe['buckets'][0]['source_path'], recipe['buckets'][1]['source_path'] = b, a
    recipe_path.write_text(json.dumps(recipe))
    result = subprocess.run([sys.executable, '-m', 'hyak.pool.worker'], cwd=str(REPO),
                            env=pool.env(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and 'does not end with' in result.stderr


def test_family_filter(tmp_config):
    pool = Pool(tmp_config, count=8)
    wanted = f"{pool.plan['configs'][0]['algorithm']}/{pool.plan['configs'][0]['pretokenizer']}"
    code, _, err = pool.run('--families', wanted, workers=1)
    assert code == 0, err
    trained = sorted(c['index'] for c in pool.invocations())
    assert trained == [c['index'] for c in pool.plan['configs'] if f"{c['algorithm']}/{c['pretokenizer']}" == wanted]
    code, _, err = pool.run('--families', 'bpe/whitespace', workers=1)
    assert code == 2 and 'Unknown families' in err


# --- review fixes ---------------------------------------------------------------------------------------------
def test_same_corpus_under_another_path_keeps_the_pool_identity(tmp_config):
    """pool-5: the identity has no absolute paths; a byte-identical corpus elsewhere continues the pool."""
    import shutil
    pool = Pool(tmp_config, count=6)
    code, _, err = pool.run('--indices', '0', '1', '2')
    assert code == 0, err
    moved = pool.cfg.base / 'elsewhere' / 'corpus'
    shutil.copytree(pool.cfg.base / 'corpus', moved)
    recipe = json.loads(pool.recipe.read_text())
    for bucket in recipe['buckets']:
        bucket['source_path'] = bucket['source_path'].replace(str(pool.cfg.base / 'corpus'), str(moved))
    pool.recipe = moved / 'pool-recipe.json'
    pool.recipe.write_text(json.dumps(recipe, indent=2) + '\n')
    code, _, err = pool.run()
    assert code == 0, err
    status = pool.status()
    assert status['completed'] == 6 and status['blocked_indices'] == []
    assert len({pool.manifest(i)['provenance']['recipe_sha256'] for i in range(6)}) == 1


def test_owner_that_died_with_its_job_counts_as_an_attempt(tmp_config):
    """pool-6: a tokenizer that takes its whole job down is recorded and exhausted after --max-attempts."""
    pool = Pool(tmp_config, count=4)
    modes = json.dumps({'1': 'poison'})
    codes = [pool.run('--indices', '1', '0', '2', '3', '--max-attempts', '3', workers=1, FAKE_MODES=modes)[0]
             for _ in range(3)]
    assert codes == [-signal.SIGKILL] * 3
    code, _, err = pool.run('--indices', '1', '0', '2', '3', '--max-attempts', '3', workers=1, FAKE_MODES=modes)
    failure = pool.failure(1)
    assert failure['attempts'] == 3 and {h['kind'] for h in failure['history']} == {'owner_died'}
    assert 'ended while training' in failure['error']
    assert code == 3, err                                       # exhausted: the user decides
    assert Counter(c['index'] for c in pool.invocations())[1] == 3   # no fourth start
    assert pool.status()['exhausted_indices'] == [1]
    assert all((pool.pool_dir / 'tokenizers' / f'{i:04d}' / 'manifest.json').exists() for i in (0, 2, 3))
    code, _, err = pool.run('--retry-failed', '--max-attempts', '3', workers=1)   # the user's decision clears it
    assert code == 0, err


def test_brought_in_pool_is_never_retrained_or_replaced(pool):
    """pool-v1: tokenizer.json files without this worker's manifests, and an existing index, stop pool jobs."""
    folder = pool.pool_dir / 'tokenizers'
    originals = {}
    for config in pool.plan['configs']:
        path = folder / f'{config["index"]:04d}' / 'tokenizer.json'
        path.parent.mkdir(parents=True)
        raw = (pool.library / f"{config['algorithm']}-{config['pretokenizer']}-{config['vocab_size']}.json").read_bytes()
        path.write_bytes(raw + b' ')
        originals[path] = path.read_bytes()
    code, out, err = pool.run()
    assert code == 3 and pool.invocations() == []                 # every index blocked, nothing trained
    assert set(pool.status()['blocked_indices']) == {0, 1, 2, 3}
    (pool.pool_dir / 'full-results.json').write_text('{"plan_sha256": "x", "completed": 4, "tokenizers": []}\n')
    code, out, err = pool.run()
    assert code == 2 and 'brought-in pool' in err and pool.invocations() == []
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not (pool.pool_dir / 'claims').exists() or not any((pool.pool_dir / 'claims').iterdir())


def test_publish_moves_aside_only_a_failure_record(pool):
    from hyak.pool.worker import PoolState
    folder = pool.pool_dir / 'tokenizers' / '0002'
    folder.mkdir(parents=True)
    (folder / 'notes.txt').write_text('a file this worker did not write')
    code, _, _ = pool.run('--indices', '2', workers=1)
    assert code == 3 and (folder / 'notes.txt').exists() and not (folder / 'manifest.json').exists()
    assert pool.invocations() == [] and pool.status()['blocked_indices'] == [2]
    state = PoolState(pool.plan, pool.pool_dir / 'tokenizers', None)
    assert state.classify(2)[0] == 'blocked' and 'unexpected files' in state.classify(2)[1]
    (folder / 'notes.txt').unlink()
    (folder / 'failure.json').write_text(json.dumps({'config': pool.plan['configs'][2], 'error': 'x', 'attempts': 1}))
    assert state.classify(2) == ('failed', 1)                     # a failure record of this worker is fine
    code, _, err = pool.run('--indices', '2', workers=1)
    assert code == 0, err
    assert (folder / 'manifest.json').exists() and not (folder / 'failure.json').exists()  # moved aside, replaced


def test_complete_pool_exits_0_although_an_attempt_failed_here(pool):
    """pool-7 (a): job A fails index 1 once, job B repairs it, job A finishes the pool: exit 0, status complete."""
    modes = json.dumps({'1': 'fail_once'})
    a = pool.start('--indices', '1', '0', '2', '3', workers=1, FAKE_MODES=modes, FAKE_DELAY='2.0')
    time.sleep(1.5)
    b = pool.start('--indices', '1', workers=1, FAKE_MODES=modes, FAKE_DELAY='0.2')
    _, err_b = b.communicate(timeout=120)
    _, err_a = a.communicate(timeout=120)
    assert b.returncode == 0, err_b
    assert a.returncode == 0, err_a
    assert pool.status()['state'] == 'complete' and pool.status()['completed'] == 4


def test_no_start_and_kill_loop_at_the_end_of_the_job(pool):
    """pool-9 (c): nothing is started when the remaining time is within the kill margin."""
    code, _, err = pool.run('--min-remaining-minutes', '0', '--kill-margin-seconds', '3', workers=2,
                            FAKE_MODE='sleep30', SLURM_JOB_END_TIME=int(time.time()) + 5, timeout=60)
    assert code == 0, err
    interrupted = [e for e in pool.events() if e.get('event') == 'tokenizer_interrupted']
    assert len(pool.invocations()) <= 2 and len(interrupted) <= 2
    assert any('deferred' in e.get('reason', '') for e in pool.events() if e.get('event') == 'slot_idle')


def test_sigkill_failure_names_the_memory_limit(pool):
    """pool-9 (d)."""
    code, _, _ = pool.run('--indices', '0', workers=1, FAKE_MODES=json.dumps({'0': 'sigkill'}))
    assert code == 3
    error = pool.failure(0)['error']
    assert 'killed by signal 9' in error and "SIGKILL: usually the job's memory limit" in error and 'sacct' in error


def test_index_presets():
    from hyak.pool import BINDING_INDICES, PILOT_INDICES, parse_indices
    assert parse_indices(['pilots+binding'], 1000) == list(PILOT_INDICES) + list(BINDING_INDICES)
    assert parse_indices(['binding'], 1000) == [64, 100, 632, 687, 728, 821, 914]
    assert parse_indices(['pilots', '64', 'binding'], 1000) == list(PILOT_INDICES) + list(BINDING_INDICES)


def test_refresh_status_writes_outside_the_worker_lock():
    """pool-2: slot threads are not blocked by status I/O; stale snapshots are not written; progress on change."""
    import threading
    from types import SimpleNamespace
    from hyak.pool import worker as W

    worker = W.PoolWorker.__new__(W.PoolWorker)          # no setup: only what refresh_status reads
    worker.lock = threading.RLock()
    worker.status_lock = threading.Lock()
    worker._status_seq = worker._status_written_seq = 0
    worker._last_status, worker._last_progress = {}, None
    counts = {'complete': 1}
    worker._scan = lambda: (dict(counts), [], [], [])
    worker.state = SimpleNamespace(family_stats=lambda: {}, classify=lambda index: ('complete', None), summaries={})
    worker.pilots, worker.slot_rows, worker.count, worker.started = [0], {}, 2, time.time()
    worker.plan, worker.production = {'plan_sha256': 'p', 'configs': []}, False
    worker.args = SimpleNamespace(max_attempts=3)
    events = []
    worker._event = lambda name, **fields: events.append((name, fields))

    def lock_is_free():
        got = []
        thread = threading.Thread(target=lambda: got.append(worker.lock.acquire(timeout=2) and (worker.lock.release() or True)))
        thread.start()
        thread.join()
        return got == [True]

    writes = []

    class Status:
        def other_rows(self):
            assert lock_is_free()
            return []

        def update(self, completed, append=True, **fields):
            assert lock_is_free() and append is False
            writes.append(completed)
            return {'completed': completed, **fields}

    worker.status = Status()
    assert worker.refresh_status()['completed'] == 1
    worker.refresh_status()
    assert writes == [1, 1] and [name for name, _ in events] == ['progress']      # unchanged counts: no event
    counts['complete'] = 2
    assert worker.refresh_status()['state'] == 'complete'
    assert [f['completed'] for _, f in events] == [1, 2] and events[-1][1]['total'] == 2
    # a snapshot older than the last written one is skipped; its caller gets the newest status
    worker._status_written_seq = worker._status_seq + 5
    newest = worker._last_status
    assert worker.refresh_status() is newest and newest['state'] == 'complete'
    assert writes == [1, 1, 2]
