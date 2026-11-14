"""The sbatch scripts of every stage, run end to end with bash against a fake environment.

Nothing here needs Slurm, conda, a GPU or the real tools: ``$ANYTOK_ROOT/env/{train,data}/bin/python``
is a fake interpreter (``FAKE_DRIVER`` below) that logs every call and plays a scripted scenario per
module, a fake ``conda`` activates it, and ``hyak.common.status`` calls (and ``python -c`` / stdin
scripts) are passed to the real interpreter, so status.json / events.jsonl are the real ones (in the
per-test HYAK_MONITOR_ROOT).  ``hyak.tools.register_dashboard`` is always intercepted: the real
plans/run-dashboard.json must never change (checked after every test).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

from hyak.common.paths import SBATCH_FILES, STAGES

BASH = shutil.which('bash')
pytestmark = pytest.mark.skipif(BASH is None, reason='bash not available')
REPO = Path(__file__).resolve().parents[2]
SLURM_DIR = REPO / 'hyak' / 'slurm'
REGISTRY = REPO / 'plans' / 'run-dashboard.json'
SBATCH = sorted(SLURM_DIR.glob('*.sbatch'))
PROLOGUE_START = '# --- common job prologue'
PROLOGUE_END = '# --- end of common job prologue ---'

FAKE_DRIVER = r'''
import json, os, signal, sys, time
real = os.environ['FAKE_REAL_PYTHON']
args = sys.argv[1:]
keep = ('HYAK_STAGE', 'HYAK_RUN', 'HYAK_ENV', 'CONDA_PREFIX', 'PYTHONPATH', 'TOKENIZERS_PARALLELISM',
        'HF_HOME', 'PYTHONDONTWRITEBYTECODE', 'HYAK_POOL_ARGS', 'HF_HUB_OFFLINE', 'LD_LIBRARY_PATH', 'PYTHONNOUSERSITE',
        'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES',
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'PYTHONOPTIMIZE', 'PYTORCH_CUDA_ALLOC_CONF')
record = {'argv': args, 'pid': os.getpid(), 'interpreter': os.environ.get('FAKE_WHICH'),
          'env': {k: os.environ.get(k, '<unset>') for k in keep}}


def log(entry):
    with open(os.environ['FAKE_LOG'], 'a') as handle:
        handle.write(json.dumps(entry) + '\n')


if not args or args[0] in ('-c', '-') or args[:2] == ['-m', 'hyak.common.status']:
    record['kind'] = 'passthrough'
    log(record)
    os.execv(real, [real] + args)
if args[:2] == ['-m', 'hyak.tools.register_dashboard']:
    record['kind'] = 'register'
    log(record)
    if os.environ.get('FAKE_REGISTER_HANG'):      # a registry lock whose holder never lets go
        time.sleep(float(os.environ['FAKE_REGISTER_HANG']))
    sys.exit(0)
module = args[1] if args[0] == '-m' else args[0]
scenario = json.load(open(os.environ['FAKE_SCENARIO'])) if os.environ.get('FAKE_SCENARIO') else {}
key = next((f'{module} {a}' for a in args[2:] if f'{module} {a}' in scenario), module)
steps = scenario.get(key, [{}])
counter = os.path.join(os.environ['FAKE_STATE'], key.replace('/', '_').replace(' ', '_'))
n = int(open(counter).read()) if os.path.exists(counter) else 0
open(counter, 'w').write(str(n + 1))
step = steps[min(n, len(steps) - 1)]
record.update(kind='tool', key=key, call=n)
log(record)
for path, text in step.get('write', {}).items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(text)
print(f'FAKE-STDOUT {key} {n}', flush=True)
print(f'FAKE-STDERR {key} {n}', file=sys.stderr, flush=True)
if step.get('stdout'):
    print(step['stdout'], flush=True)
if step.get('wait_term'):
    got = []
    signal.signal(signal.SIGTERM, lambda *_: got.append(1))
    if step.get('ready'):
        open(step['ready'], 'w').write('ready')
    deadline = time.time() + step.get('max_wait', 30)
    while not got and time.time() < deadline:
        time.sleep(0.05)
    log({'kind': 'exit', 'key': key, 'terminated': bool(got), 'pid': os.getpid()})
    sys.exit(step.get('rc_on_term', 0) if got else step.get('rc', 0))
if step.get('ready'):
    open(step['ready'], 'w').write('ready')
time.sleep(step.get('sleep', 0))
sys.exit(step.get('rc', 0))
'''

FAKE_CONDA = ('#!/bin/sh\n[ "$1 $2" = "shell.bash hook" ] || exit 3\n'
              "cat <<'EOF'\nconda() { if [ \"$1\" = activate ]; then export CONDA_PREFIX=\"$2\"; "
              'export PATH="$2/bin:$PATH"; else command conda "$@"; fi; }\nEOF\n')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def registry_untouched():
    before = sha(REGISTRY)
    yield
    assert sha(REGISTRY) == before, 'plans/run-dashboard.json was modified by a test'


class Job:
    """A fake cluster environment in which one sbatch script runs."""

    def __init__(self, tmp_path, make_tmp_config, **config):
        self.t = make_tmp_config(HYAK_MODULES='', **config)
        self.tmp = tmp_path
        self.root = self.t.paths.root
        self.monitor = tmp_path / 'monitor'
        self.log = tmp_path / 'calls.jsonl'
        self.state = tmp_path / 'fake-state'
        self.state.mkdir()
        driver = tmp_path / 'fake_driver.py'
        driver.write_text(FAKE_DRIVER)
        for which in ('train', 'data'):
            prefix = self.root / 'env' / which
            (prefix / 'bin').mkdir(parents=True)
            (prefix / 'lib').mkdir()
            python = prefix / 'bin' / 'python'
            python.write_text(f'#!/bin/bash\nexport FAKE_WHICH={which}\n'
                              f'exec "$FAKE_REAL_PYTHON" {shlex.quote(str(driver))} "$@"\n')
            python.chmod(0o755)
        bindir = tmp_path / 'fakebin'
        bindir.mkdir()
        (bindir / 'conda').write_text(FAKE_CONDA)
        (bindir / 'conda').chmod(0o755)
        self.env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_REPO': str(REPO),
                    'FAKE_REAL_PYTHON': sys.executable, 'FAKE_LOG': str(self.log), 'FAKE_STATE': str(self.state),
                    'HYAK_PYTHON': str(self.root / 'env' / 'train' / 'bin' / 'python'),
                    'SLURM_JOB_ID': '4242', 'SLURM_CPUS_PER_TASK': '8'}
        for key in [k for k in self.env if re.fullmatch(r'HYAK_\w+_ARGS|HYAK_EVAL_\w+|HYAK_CORPUS_PROBE|HYAK_GCC_MODULE', k)]:
            del self.env[key]

    def state_reset(self):
        """Forget how often each fake tool was called (a scenario list restarts at its first step)."""
        for path in self.state.iterdir():
            path.unlink()

    def scenario(self, data: dict):
        path = self.tmp / 'scenario.json'
        path.write_text(json.dumps(data))
        self.env['FAKE_SCENARIO'] = str(path)

    def slurm_log(self, stage):
        return self.monitor / stage / 'slurm-4242.out'

    def start(self, stage, **env):
        script = SLURM_DIR / SBATCH_FILES[stage]
        log = self.slurm_log(stage)
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log, 'a')
        return subprocess.Popen([BASH, str(script)], env={**self.env, 'HYAK_STAGE': stage, **env}, cwd=str(REPO),
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    def run(self, stage, timeout=120, **env):
        proc = self.start(stage, **env)
        try:
            proc.wait(timeout=timeout)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        return proc.returncode, self.slurm_log(stage).read_text()

    def calls(self, kind='tool'):
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [r for r in rows if kind is None or r['kind'] == kind]

    def tools(self):
        """argv (after -m) of every tool call, in order."""
        return [c['argv'][1:] if c['argv'][0] == '-m' else c['argv'] for c in self.calls()]

    def status(self, stage):
        return json.loads((self.monitor / stage / 'status.json').read_text())

    def events(self, stage):
        path = self.monitor / stage / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def job(tmp_path, make_tmp_config):
    return Job(tmp_path, make_tmp_config)


# --- static checks ---------------------------------------------------------------------------------------
def test_one_script_per_table_entry():
    assert sorted(p.name for p in SBATCH) == sorted(set(SBATCH_FILES.values()))
    for stage in STAGES:
        assert (SLURM_DIR / SBATCH_FILES[stage]).is_file(), stage


@pytest.mark.parametrize('path', SBATCH, ids=lambda p: p.name)
def test_syntax_and_conventions(path):
    for bash in {BASH, '/bin/bash'}:
        if bash and Path(bash).exists():
            out = subprocess.run([bash, '-n', str(path)], capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
    text = path.read_text()
    assert text.startswith('#!/bin/bash\n')
    assert not re.search(r'^#SBATCH', text, re.M), 'resources come from submit.sh only'
    assert '$0' not in text and '${0' not in text, 'never derive paths from $0 in a batch job'
    assert 'set -euo pipefail' in text and 'source "$HYAK_REPO/hyak/slurm/lib.sh"' in text
    assert 'source "$HYAK_REPO/hyak/slurm/job.sh"' in text
    assert 'hyak_register_dashboard' in text and 'job_begin ' in text
    assert os.access(path, os.X_OK)


@pytest.mark.parametrize('name', ['lib.sh', 'job.sh', 'stop.sh', 'submit.sh'])
def test_shared_scripts_have_valid_syntax(name):
    for bash in {BASH, '/bin/bash'}:
        if bash and Path(bash).exists():
            out = subprocess.run([bash, '-n', str(SLURM_DIR / name)], capture_output=True, text=True)
            assert out.returncode == 0, (bash, out.stderr)


def test_prologue_identical_everywhere_and_the_life_cycle_lives_in_job_sh():
    blocks = []
    for path in SBATCH:
        text = path.read_text()
        start, end = text.index(PROLOGUE_START), text.index(PROLOGUE_END)
        blocks.append(text[start:end])
        # the functions are defined once, in job.sh (slurm-14): no script carries its own copy
        for function in ('job_begin()', 'job_on_exit()', 'job_finalize_status()', 'job_step()', 'job_stopping()',
                         'job_extra_args()'):
            assert function not in text, (path.name, function)
    assert len(set(blocks)) == 1 and len(blocks[0].splitlines()) <= 12
    job = (SLURM_DIR / 'job.sh').read_text()
    for function in ('job_begin()', 'job_on_exit()', 'job_finalize_status()', 'job_step()', 'job_step_foreground()',
                     'job_stopping()', 'job_extra_args()', 'job_explain()', 'job_not_now()'):
        assert function in job, function


def test_stage_names_in_scripts_match_table():
    for path in SBATCH:
        (line,) = [l for l in path.read_text().splitlines() if l.startswith('job_begin ')]
        assert sorted(line.split()[1:]) == sorted(s for s, f in SBATCH_FILES.items() if f == path.name), path.name


# --- check-env, probe, init, stage-inputs, replay-check ------------------------------------------------------
def test_check_env_success(job):
    rc, out = job.run('check-env')
    assert rc == 0, out
    assert job.tools() == [['hyak.env.check_env']]
    (call,) = job.calls()
    assert call['interpreter'] == 'train' and call['env']['HYAK_STAGE'] == 'check-env'
    assert call['env']['TOKENIZERS_PARALLELISM'] == '<unset>' and call['env']['PYTHONPATH'] == str(REPO)
    assert call['env']['PYTHONDONTWRITEBYTECODE'] == '1'
    assert len(job.calls('register')) == 1
    status = job.status('check-env')
    assert status['state'] == 'paused' and status['workers'] == []  # the fake tool wrote no state itself
    assert [e.get('event') for e in job.events('check-env')][-1] == 'job_end'


def test_check_env_failure_and_extra_args(job):
    job.scenario({'hyak.env.check_env': [{'rc': 4}]})            # 4 = a check failed
    rc, out = job.run('check-env', HYAK_CHECK_ENV_ARGS='--no-hash-model --skip network')
    assert rc == 4, out
    assert job.tools() == [['hyak.env.check_env', '--no-hash-model', '--skip', 'network']]
    status = job.status('check-env')
    assert status['state'] == 'failed' and 'remedy' in status['error'] and 'FAKE-STDERR' in status['error']
    assert 'at least one check failed' in status['error']


def test_wrong_stage_is_refused(job):
    proc = subprocess.run([BASH, str(SLURM_DIR / '00_check_env.sbatch')], env={**job.env, 'HYAK_STAGE': 'probe'},
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1 and "not HYAK_STAGE='probe'" in proc.stderr
    proc = subprocess.run([BASH, str(SLURM_DIR / '00_check_env.sbatch')],
                          env={k: v for k, v in job.env.items() if k != 'HYAK_REPO'},
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and 'submit with hyak/slurm/submit.sh' in proc.stderr
    assert job.calls() == []


def test_missing_environment_fails_with_remedy(job):
    shutil.rmtree(job.root / 'env' / 'train')
    job.env['HYAK_PYTHON'] = str(job.root / 'env' / 'data' / 'bin' / 'python')
    rc, out = job.run('init')
    assert rc == 1 and 'setup_train_env.sh' in out
    status = job.status('init')
    assert status['state'] == 'failed' and 'setup_train_env.sh' in status['error']


@pytest.mark.parametrize('rc,state,needle', [
    (0, 'paused', None), (4, 'failed', 'do not start training'), (2, 'failed', 'setup or unexpected error'),
    (1, 'failed', 'crashed: uncaught exception, traceback in the job log')])
def test_probe(job, rc, state, needle):
    job.scenario({'hyak.memory.probe': [{'rc': rc}]})
    code, out = job.run('probe', HYAK_PROBE_ARGS='--chunk-rows 16384 8192')
    assert code == rc, out
    assert job.tools() == [['hyak.memory.probe', '--also-pinned', '--chunk-rows', '16384', '8192']]
    status = job.status('probe')
    assert status['state'] == state
    if needle:
        assert needle in status['error']


def test_init_and_stage_inputs(job):
    rc, out = job.run('init')
    assert rc == 0, out
    rc, out = job.run('stage-inputs', HYAK_STAGE_INPUTS_ARGS='--force')
    assert rc == 0, out
    assert job.tools() == [['hyak.model.init_hypernet', '--status'],
                           ['hyak.corpus.stage_inputs', '--force'], ['hyak.corpus.stage_inputs', '--check']]
    assert all(c['interpreter'] == 'train' for c in job.calls())


@pytest.mark.parametrize('rc,needle', [(2, 'usage, configuration, input or environment error'),
                                       (4, 'a check failed (for example the calibration against original_in_scaler.npz)'),
                                       (1, 'crashed: uncaught exception')])
def test_init_failures_have_their_own_text(job, rc, needle):
    job.scenario({'hyak.model.init_hypernet': [{'rc': rc}]})
    code, out = job.run('init')
    assert code == rc and job.tools() == [['hyak.model.init_hypernet', '--status']]
    status = job.status('init')
    assert status['state'] == 'failed' and needle in status['error']
    if rc != 1:
        assert 'INIT FAILED' in status['error']


def test_stage_inputs_refusal_skips_check(job):
    job.scenario({'hyak.corpus.stage_inputs': [{'rc': 4}]})
    rc, out = job.run('stage-inputs')
    assert rc == 4 and job.tools() == [['hyak.corpus.stage_inputs']]
    assert 'refused or found an inconsistency (also for --check)' in job.status('stage-inputs')['error']


def test_stage_inputs_check_failure_is_an_inconsistency(job):
    job.scenario({'hyak.corpus.stage_inputs --check': [{'rc': 4}]})
    rc, out = job.run('stage-inputs')
    assert rc == 4 and job.tools() == [['hyak.corpus.stage_inputs'], ['hyak.corpus.stage_inputs', '--check']]
    assert 'inconsistency' in job.status('stage-inputs')['error']


def test_stage_inputs_scorecard_stop_is_a_decision(job):
    job.scenario({'hyak.corpus.stage_inputs': [{'rc': 3}]})
    rc, out = job.run('stage-inputs')
    error = job.status('stage-inputs')['error']
    assert rc == 3 and job.tools() == [['hyak.corpus.stage_inputs']]
    assert 'no exact corpus scorecard' in error and 'scorecard.txt' in error and 'README section 10' in error
    assert "needs the user's approval" in error and '--accept-corpus-scorecard "<reason>"' in error
    assert 'HYAK_STAGE_INPUTS_ARGS' in error


def test_stage_inputs_missing_model_says_what_to_run(job):
    job.scenario({'hyak.corpus.stage_inputs': [{'rc': 2}]})
    rc, out = job.run('stage-inputs')
    assert rc == 2 and 'run python -m hyak.model.fetch_model' in job.status('stage-inputs')['error']


def test_replay_check(job):
    """The example replay runs with replay_check's own default (2,000 updates): the job passes no --updates."""
    rc, out = job.run('replay-check', HYAK_REPLAY_CHECK_ARGS='--strict')
    assert rc == 0, out
    assert job.tools() == [['hyak.train.replay_check', '--schedule-only'],
                           ['hyak.train.replay_check', '--strict']]
    job.log.unlink()
    rc, out = job.run('replay-check')
    assert rc == 0 and job.tools() == [['hyak.train.replay_check', '--schedule-only'], ['hyak.train.replay_check']]
    assert '--updates' not in ' '.join(' '.join(t) for t in job.tools())
    job.log.unlink()
    rc, out = job.run('replay-check', HYAK_REPLAY_CHECK_ARGS='--updates 500')       # the user may still choose a length
    assert rc == 0 and job.tools()[-1] == ['hyak.train.replay_check', '--updates', '500']


def test_replay_check_failures(job):
    # a schedule or RNG-digest mismatch (--schedule-only exits 4) stops before the example replay
    job.scenario({'hyak.train.replay_check --schedule-only': [{'rc': 4}]})
    rc, out = job.run('replay-check')
    assert rc == 4 and job.tools() == [['hyak.train.replay_check', '--schedule-only']]
    error = job.status('replay-check')['error']
    assert 'RNG schedule' in error and 'digest' in error and job.status('replay-check')['state'] == 'failed'
    # the example replay: 4 = a check failed (the trainer would raise, or --strict mismatch), 2 = inputs
    for code, needle in ((4, 'a check failed'), (2, 'missing or wrong staged code, corpus, pool, model'),
                         (1, 'crashed: uncaught exception')):
        job.log.unlink()
        job.scenario({'hyak.train.replay_check': [{'rc': 0}, {'rc': code}]})
        rc, out = job.run('replay-check')
        assert rc == code and len(job.tools()) == 2, out
        assert needle in job.status('replay-check')['error']
        job.state_reset()


# --- corpus, wheel, pool -------------------------------------------------------------------------------------
def test_corpus_probe_build_verify(job):
    rc, out = job.run('corpus', HYAK_CORPUS_ARGS='--workers 4')
    assert rc == 0, out
    assert job.tools() == [['hyak.corpus.build', '--probe'], ['hyak.corpus.build', '--workers', '4'],
                           ['hyak.corpus.verify']]
    assert all(c['interpreter'] == 'data' for c in job.calls())
    assert job.calls()[0]['env']['HF_HOME'] == str(job.root / 'hf-cache')
    status = job.status('corpus')
    assert status['workers'] == [] and status['total'] == 23


NOT_EXACT = 'corpus scorecard is not exact: do not submit stage-inputs or the pool (README section 10)'


def test_corpus_probe_failure_stops(job):
    """The probe's failure status is 2 (hyak.corpus.build returns EXIT_USAGE for it)."""
    job.scenario({'hyak.corpus.build --probe': [{'rc': 2}]})
    rc, out = job.run('corpus')
    assert rc == 2 and job.tools() == [['hyak.corpus.build', '--probe']]
    assert 'probe.json' in job.status('corpus')['error']
    job.log.unlink()
    job.scenario({'hyak.corpus.build --probe': [{'rc': 1}]})
    rc, out = job.run('corpus')
    assert rc == 1 and 'crashed: uncaught exception' in job.status('corpus')['error']


def test_corpus_build_exit_codes_are_mapped(job):
    for code, needle in ((2, 'usage, configuration or environment error'), (4, 'a corpus bucket failed'),
                         (1, 'crashed: uncaught exception')):
        job.log.unlink(missing_ok=True)
        job.scenario({'hyak.corpus.build': [{'rc': code}]})
        rc, out = job.run('corpus', HYAK_CORPUS_PROBE='0')
        assert rc == code and job.tools() == [['hyak.corpus.build']], out
        assert needle in job.status('corpus')['error'] and job.status('corpus')['state'] == 'failed'


def write_scorecard(job, verdict):
    path = job.root / 'corpus' / 'scorecard.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'Corpus fidelity scorecard  2026-11-14T00:00:00Z\ncorpus root: x\nverdict: {verdict}\nexpected: y\n')


def test_corpus_warns_when_the_scorecard_is_not_exact_and_keeps_the_exit_status(job):
    exact = ('content-identical to every recorded anchor (train bytes and training content of all 23 buckets, the original '
             'corpus manifest, regenerated recipe, all held-out documents, hashes and excerpts)')
    write_scorecard(job, exact)
    rc, out = job.run('corpus', HYAK_CORPUS_PROBE='0')
    assert rc == 0 and NOT_EXACT not in out, out
    for verdict in ('differs from the original: web-en 1/3', 'incomplete: 20/23 buckets built',
                    'not checked completely: web-en skipped (--skip-train-files); rerun python -m hyak.corpus.verify without it'):
        write_scorecard(job, verdict)
        rc, out = job.run('corpus', HYAK_CORPUS_PROBE='0')
        assert rc == 0 and NOT_EXACT in out, (verdict, out)             # a warning only: the job's status is unchanged
        assert job.status('corpus')['state'] != 'failed'
    (job.root / 'corpus' / 'scorecard.txt').unlink()                    # no scorecard at all is not exact either
    rc, out = job.run('corpus', HYAK_CORPUS_PROBE='0')
    assert rc == 0 and NOT_EXACT in out
    job.scenario({'hyak.corpus.verify': [{'rc': 2}]})                   # verify failing: warning too, still exit 0
    write_scorecard(job, exact)
    rc, out = job.run('corpus', HYAK_CORPUS_PROBE='0')
    assert rc == 0 and NOT_EXACT in out and 'exited 2' in out


def test_corpus_without_probe_and_failed_bucket(job):
    job.scenario({'hyak.corpus.build': [{'rc': 4}]})
    rc, out = job.run('corpus', HYAK_CORPUS_PROBE='0')
    assert rc == 4 and job.tools() == [['hyak.corpus.build']]
    assert 'a corpus bucket failed' in job.status('corpus')['error']


def wheel_repo(tmp_path, stub_rc):
    """A copy of the checkout whose hyak/pool/build_wheel.sh is a stub exiting with ``stub_rc``."""
    repo = tmp_path / 'repo'
    for rel in ('hyak', 'monitoring'):
        shutil.copytree(REPO / rel, repo / rel, ignore=shutil.ignore_patterns('__pycache__', 'tests'))
    stub = repo / 'hyak/pool/build_wheel.sh'
    stub.write_text(f'#!/bin/bash\necho "stub build_wheel.sh $*"\nexit {stub_rc}\n')
    return repo


@pytest.mark.parametrize('stub_rc', [0, 7])
def test_wheel(tmp_path, make_tmp_config, stub_rc):
    repo = wheel_repo(tmp_path, stub_rc)
    job = Job(tmp_path, make_tmp_config)
    job.env['HYAK_REPO'] = str(repo)
    proc = subprocess.run([BASH, str(repo / 'hyak/slurm/03_wheel.sbatch')],
                          env={**job.env, 'HYAK_STAGE': 'wheel', 'HYAK_WHEEL_ARGS': '--jobs 8'},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == stub_rc, proc.stderr
    assert 'stub build_wheel.sh build install --jobs 8' in proc.stdout
    status = job.status('wheel')
    if stub_rc == 0:
        assert status['state'] == 'complete' and status['stage'] == 'complete' and status['completed'] == 2
    else:
        assert status['state'] == 'failed' and 'fetch phase' in status['error']


def test_wheel_loads_the_gcc_module_when_hyak_gcc_module_is_set(tmp_path, make_tmp_config):
    """HYAK_GCC_MODULE is the user's choice: it is loaded even when a compiler is on PATH (a fake Lmod via LMOD_PKG)."""
    lmod = tmp_path / 'lmod'
    (lmod / 'init').mkdir(parents=True)
    calls = tmp_path / 'module-calls.txt'
    (lmod / 'init' / 'bash').write_text(f'module() {{ echo "$*" >> {shlex.quote(str(calls))}; }}\n')
    repo = wheel_repo(tmp_path, 0)
    job = Job(tmp_path, make_tmp_config)
    job.env['HYAK_REPO'] = str(repo)
    proc = subprocess.run([BASH, str(repo / 'hyak/slurm/03_wheel.sbatch')],
                          env={**job.env, 'HYAK_STAGE': 'wheel', 'HYAK_GCC_MODULE': 'gcc/12.1.0', 'LMOD_PKG': str(lmod)},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert calls.read_text().split() == ['load', 'gcc/12.1.0']
    assert 'C and C++ compiler module gcc/12.1.0' in proc.stdout + proc.stderr
    calls.unlink()
    proc = subprocess.run([BASH, str(repo / 'hyak/slurm/03_wheel.sbatch')],            # not set, compilers present: no module
                          env={**job.env, 'HYAK_STAGE': 'wheel', 'LMOD_PKG': str(lmod)}, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0 and not calls.exists()


def pool_status_complete(job):
    return {str(job.monitor / 'pool' / 'status.json'): json.dumps({'state': 'complete', 'stage': 'complete',
                                                                   'completed': 1000, 'total': 1000})}


def test_pool_pilot_job_never_collects(job):
    rc, out = job.run('pool', HYAK_POOL_ARGS='--pilots-only')
    assert rc == 0, out
    assert job.tools() == [['hyak.pool.worker', '--workers', '2', '--threads', '4', '--pilots-only']]
    (call,) = job.calls()
    assert call['interpreter'] == 'data' and call['env']['TOKENIZERS_PARALLELISM'] == 'true'
    assert call['env']['HYAK_POOL_ARGS'] == '--pilots-only'


def test_pool_incomplete_does_not_collect(job):
    """The manifests decide (collect --check-only), not the shared status: 75 = not complete yet."""
    job.scenario({'hyak.pool.collect --check-only': [{'rc': 75}]})
    rc, out = job.run('pool')
    assert rc == 0, out
    assert job.tools() == [['hyak.pool.worker', '--workers', '2', '--threads', '4'], ['hyak.pool.collect', '--check-only']]
    assert [c['interpreter'] for c in job.calls()] == ['data', 'train']
    assert 'pool not complete yet' in out and job.status('pool')['state'] != 'failed'


def test_pool_complete_collects_in_train_env(job):
    job.scenario({'hyak.pool.worker': [{'rc': 0}]})
    rc, out = job.run('pool')
    assert rc == 0, out
    assert job.tools() == [['hyak.pool.worker', '--workers', '2', '--threads', '4'],
                           ['hyak.pool.collect', '--check-only'], ['hyak.pool.collect'],
                           ['hyak.pool.scorecard', '--workers', '8']]
    assert [c['interpreter'] for c in job.calls()] == ['data', 'train', 'train', 'train']
    # a later job does not recompute a scorecard that is newer than the index AND covers everything ...
    index = job.root / 'input/pool/full-results.json'
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text('{}')
    report = job.root / 'eval/reports/pool-scorecard.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    future = time.time() + 5
    report.write_text(json.dumps({'planned': 1000, 'indices': 1000, 'levels': [1, 2, 3]}))
    os.utime(report, (future, future))
    job.log.unlink()
    rc, out = job.run('pool')
    assert rc == 0, out
    assert [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect', 'hyak.pool.collect']
    assert 'not recomputed' in out
    # ... but a newer report of a pilot subset or of one level is recomputed (pool-7c)
    for partial in ({'planned': 1000, 'indices': 12, 'levels': [1, 2, 3]},
                    {'planned': 1000, 'indices': 1000, 'levels': [1]}, {}):
        report.write_text(json.dumps(partial))
        os.utime(report, (future, future))
        job.log.unlink()
        rc, out = job.run('pool')
        assert rc == 0 and 'hyak.pool.scorecard' in [t[0] for t in job.tools()], (partial, out)


def test_pool_failure_in_this_job_does_not_fail_a_complete_pool(job):
    """One tokenizer failed here (worker exit 3) and another job retried it: the pool is complete (pool-7a)."""
    job.scenario({'hyak.pool.worker': [{'rc': 3, 'write': pool_status_complete(job)}]})
    rc, out = job.run('pool')
    assert rc == 0, out
    assert [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect', 'hyak.pool.collect',
                                           'hyak.pool.scorecard']
    status = job.status('pool')
    assert status['state'] == 'complete', 'the EXIT trap must not write "failed" over a complete pool'


def test_pool_worker_failure_with_an_incomplete_pool_stays_a_decision(job):
    job.scenario({'hyak.pool.worker': [{'rc': 3}], 'hyak.pool.collect --check-only': [{'rc': 75}]})
    rc, out = job.run('pool')
    assert rc == 3 and [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect']
    assert '--report' in job.status('pool')['error']
    # a worker that ended well while the pool has failed or blocked indices is not this job's failure
    job.log.unlink()
    job.scenario({'hyak.pool.collect --check-only': [{'rc': 75}]})
    rc, out = job.run('pool')
    assert rc == 0, out


def test_pool_collect_refusal_fails_the_job(job):
    """collect's own codes: 4 = a manifest is invalid or an existing index differs (a failure, with the refusal text)."""
    job.scenario({'hyak.pool.collect': [{'rc': 4}], 'hyak.pool.collect --check-only': [{'rc': 0}]})
    rc, out = job.run('pool')
    assert rc == 4 and [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect', 'hyak.pool.collect']
    status = job.status('pool')
    assert status['state'] == 'failed' and 'refused the pool' in status['error'] and 'invalid or inconsistent' in status['error']
    job.log.unlink()
    job.scenario({'hyak.pool.collect --check-only': [{'rc': 2}]})
    rc, out = job.run('pool')
    assert rc == 2 and 'invalid inputs' in job.status('pool')['error']
    job.log.unlink()
    job.scenario({'hyak.pool.collect --check-only': [{'rc': 4}]})        # --check-only: a manifest is invalid
    rc, out = job.run('pool')
    assert rc == 4 and [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect']
    assert 'invalid or inconsistent' in job.status('pool')['error'] and job.status('pool')['state'] == 'failed'


def test_pool_collect_75_is_not_now_not_a_failure(job):
    """75 from plain collect (not complete after all, or stopped by a second TERM): the job ends 0 and says what to do."""
    job.scenario({'hyak.pool.collect': [{'rc': 75}], 'hyak.pool.collect --check-only': [{'rc': 0}]})
    rc, out = job.run('pool')
    assert rc == 0, out
    assert [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect', 'hyak.pool.collect']   # no scorecard
    status = job.status('pool')
    assert status['state'] != 'failed' and not status.get('error')
    assert 'submit one more pool job' in out
    (event,) = [e for e in job.events('pool') if e.get('event') == 'not_now']
    assert event['reason'] == 'collect_not_now' and 'submit one more pool job' in event['note']
    assert not (job.monitor / 'pool' / 'HALTED.json').exists()


def test_pool_scorecard_failure_is_reported_with_its_own_cause(job):
    job.scenario({'hyak.pool.scorecard': [{'rc': 1}]})
    rc, out = job.run('pool')
    assert rc == 1
    error = job.status('pool')['error']
    assert 'the pool index was written' in error and 'crashed: uncaught exception' in error


def test_pool_attention_exit(job):
    job.scenario({'hyak.pool.worker': [{'rc': 3}], 'hyak.pool.collect --check-only': [{'rc': 75}]})
    rc, out = job.run('pool')
    assert rc == 3
    assert '--report' in job.status('pool')['error']


@pytest.mark.parametrize('rc,needle', [(2, 'brought-in pool'), (1, 'crashed'), (137, 'killed')])
def test_pool_worker_errors_stop_before_collect(job, rc, needle):
    job.scenario({'hyak.pool.worker': [{'rc': rc}]})
    code, out = job.run('pool')
    assert code == rc and len(job.tools()) == 1
    assert needle in job.status('pool')['error']


def test_the_early_warning_signal_is_not_forwarded_to_collect(job):
    """collect and scorecard take minutes and write once: they finish, and still run after the signal (pool-7b)."""
    ready = job.tmp / 'collect-ready'
    job.scenario({'hyak.pool.collect': [{'sleep': 1.5, 'ready': str(ready)}]})
    rc, out = run_with_term_file(job, 'pool', ready)
    assert rc == 0, out
    assert [t[0] for t in job.tools()] == ['hyak.pool.worker', 'hyak.pool.collect', 'hyak.pool.collect',
                                           'hyak.pool.scorecard']
    assert 'forwarding SIGTERM' not in out and 'received SIGTERM (no child running)' in out
    assert not job.calls('exit')


# --- training -------------------------------------------------------------------------------------------------
def should_start(newest, reason, start=True):
    value = 'null' if newest is None else str(newest)
    return {'rc': 0 if start else 75,                  # 75 = "not now": the job ends with 0 and says why
            'stdout': f'{{"newest_full_step": {value}, "reason": "{reason}", "run": "uniform", "start": {str(start).lower()}}}'}


MONITOR = {'hyak.train.monitor': [{'wait_term': True, 'rc_on_term': 0}], 'hyak.train.monitor --once': [{'rc': 0}]}


def train_calls(job):
    return [t for t in job.tools() if t[0] != 'hyak.train.monitor']


def test_train_uniform_bootstrap_then_resume(job):
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [
        should_start(None, 'fresh'), should_start(1, 'resume'), should_start(1234, 'insufficient_time', False)]})
    rc, out = job.run('train-uniform')
    assert rc == 0, out
    segment = ['hyak.train.run_segment', '--run', 'uniform']
    assert train_calls(job) == [segment + ['--should-start'], segment, segment + ['--should-start'], segment,
                                segment + ['--should-start']]
    monitors = [c for c in job.calls() if c['argv'][1] == 'hyak.train.monitor']
    assert monitors[0]['argv'][2:5] == ['--run', 'uniform', '--parent-pid']
    assert monitors[-1]['argv'][2:] == ['--run', 'uniform', '--once']
    exits = job.calls('exit')
    assert exits and exits[0]['terminated'], 'the background monitor must be stopped with SIGTERM'
    log = (job.monitor / 'train-uniform' / '4242-train.log').read_text()
    assert log.count('FAKE-STDOUT hyak.train.run_segment') == 2 and 'FAKE-STDERR' not in log
    assert 'should-start' not in log and 'FAKE-STDERR hyak.train.run_segment' in out
    status = job.status('train-uniform')
    assert status['state'] == 'paused' and status['job_id'] == '4242'
    assert not (job.monitor / 'train-uniform' / 'HALTED.json').exists()


def test_train_passes_the_recorded_chunk_change_reason_to_segments_only(job):
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [
        should_start(2000, 'resume'), should_start(2000, 'insufficient_time', False)]})
    reason = 'out of memory at update 13,500 with 16384 rows'
    rc, out = job.run('train-uniform', HYAK_ACCEPT_CHUNK_CHANGE=reason)
    assert rc == 0, out
    segment = ['hyak.train.run_segment', '--run', 'uniform']
    assert train_calls(job) == [segment + ['--should-start'], segment + ['--accept-chunk-change', reason],
                                segment + ['--should-start']]
    assert 'HYAK_ACCEPT_CHUNK_CHANGE is set' in out


def test_train_no_progress_is_an_error_and_halts(job):
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(4000, 'resume')]})
    rc, out = job.run('train-rl')
    assert rc == 4, out                      # 4 = a check failed (1 is reserved for crashes)
    assert [t for t in train_calls(job) if '--should-start' not in t] == [['hyak.train.run_segment', '--run', 'rl']]
    halted = json.loads((job.monitor / 'train-rl' / 'HALTED.json').read_text())
    assert halted['exit_code'] == 4 and halted['run'] == 'rl' and 'no checkpoint progress' in halted['reason']
    status = job.status('train-rl')
    assert status['state'] == 'failed' and 'chain halted' in status['error']
    # the next job of the chain refuses to start and runs nothing
    job.log.unlink()
    rc, out = job.run('train-rl')
    assert rc == 3 and job.tools() == [] and 'halted by' in out
    assert 'halted by' in job.status('train-rl')['error']


def test_train_pending_audit_is_not_an_error(job):
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(20000, 'audit_pending')]})
    rc, out = job.run('train-uniform')
    assert rc == 0, out
    assert 'final audit is still pending' in out
    assert not (job.monitor / 'train-uniform' / 'HALTED.json').exists()


def test_train_complete_run_starts_nothing(job):
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(20000, 'complete', False)]})
    rc, out = job.run('train-uniform')
    assert rc == 0, out
    assert train_calls(job) == [['hyak.train.run_segment', '--run', 'uniform', '--should-start']]


@pytest.mark.parametrize('segment_rc,should_rc,needle', [
    (1, 0, 'crashed: uncaught exception'), (2, 0, 'configuration, input or environment'), (4, 0, 'a check failed'),
    (0, 2, 'configuration, input or environment'), (0, 1, 'crashed: uncaught exception'), (0, 3, 'needs your decision')])
def test_train_failures(job, segment_rc, should_rc, needle):
    steps = [should_start(None, 'fresh')] if should_rc == 0 else [{'rc': should_rc, 'stdout': ''}]
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': steps,
                  'hyak.train.run_segment': [{'rc': segment_rc}]})
    rc, out = job.run('train-uniform')
    expected = segment_rc or should_rc
    assert rc == expected, out
    assert (job.monitor / 'train-uniform' / 'HALTED.json').exists()      # every status but 0/130/143 halts the chain
    status = job.status('train-uniform')
    assert status['state'] == 'failed' and needle in status['error']


@pytest.mark.parametrize('code', [137, 139])
def test_a_killed_trainer_is_a_failure_that_halts_the_chain(job, code):
    """Exit 137 (signal 9, typically the memory limit) used to be recorded as "stopped" (monitoring-v1)."""
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(None, 'fresh')],
                  'hyak.train.run_segment': [{'rc': code}]})
    rc, out = job.run('train-uniform')
    assert rc == code, out
    halted = json.loads((job.monitor / 'train-uniform' / 'HALTED.json').read_text())
    assert halted['exit_code'] == code and 'killed' in halted['reason']
    status = job.status('train-uniform')
    assert status['state'] == 'failed' and status['stage'] == 'failed'
    assert 'killed' in status['error'] and 'chain halted' in status['error']


def test_rl_before_the_uniform_pairing_is_not_now_not_a_failure(job):
    """exit 75 of --should-start: the job ends with 0, writes no HALTED.json, and the status says why."""
    pending = {'rc': 75, 'stdout': '{"newest_full_step": null, "reason": "uniform_bootstrap_pending", "run": "rl", "start": false}'}
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [pending]})
    for _ in range(3):                                   # a whole chain of RL jobs submitted too early: none is spent
        rc, out = job.run('train-rl')
        assert rc == 0, out
        assert not (job.monitor / 'train-rl' / 'HALTED.json').exists()
    assert [t for t in train_calls(job) if '--should-start' not in t] == []
    status = job.status('train-rl')
    assert status['state'] == 'paused' and not status.get('error')
    events = [e for e in job.events('train-rl') if e.get('event') == 'not_now']
    assert len(events) == 3 and events[0]['reason'] == 'uniform_bootstrap_pending'
    assert 'uniform_bootstrap_pending' in events[0]['note'] and 'not now' in out


def test_train_stops_on_term_and_finalizes(job):
    ready = job.tmp / 'segment-ready'
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(None, 'fresh')],
                  'hyak.train.run_segment': [{'wait_term': True, 'rc_on_term': 0, 'ready': str(ready)}]})
    proc = job.start('train-uniform')
    try:
        deadline = time.time() + 30
        while not ready.exists():
            assert time.time() < deadline and proc.poll() is None, job.slurm_log('train-uniform').read_text()
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGTERM)           # Slurm's --signal=B:TERM reaches only the batch shell
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    out = job.slurm_log('train-uniform').read_text()
    assert proc.returncode == 0, out
    segment_exits = [e for e in job.calls('exit') if e['key'] == 'hyak.train.run_segment']
    assert segment_exits == [{**segment_exits[0], 'terminated': True}]
    assert len([t for t in train_calls(job) if '--should-start' in t]) == 1, 'no new segment after the signal'
    assert job.status('train-uniform')['state'] == 'paused'
    assert not (job.monitor / 'train-uniform' / 'HALTED.json').exists()


def test_train_run_mismatch(job):
    rc, out = job.run('train-uniform', HYAK_RUN='rl')
    assert rc == 1 and 'contradicts' in out and job.tools() == []


# --- smoke-train ------------------------------------------------------------------------------------------------
def test_smoke_train(job):
    job.scenario({'hyak.train.monitor': [{'wait_term': True}], 'hyak.train.monitor --once': [{'rc': 0}]})
    rc, out = job.run('smoke-train', HYAK_RUN='rl')
    assert rc == 0, out
    assert job.tools()[0][:4] == ['hyak.train.monitor', '--run', 'rl', '--smoke']
    assert ['hyak.train.run_segment', '--run', 'rl', '--smoke'] in job.tools()
    assert job.tools()[-1] == ['hyak.train.monitor', '--run', 'rl', '--smoke', '--once']
    assert 'FAKE-STDOUT hyak.train.run_segment' in (job.monitor / 'smoke-train' / '4242-train.log').read_text()


def test_smoke_train_75_is_not_now_not_a_failure(job):
    """run_segment --smoke exit 75 (RL smoke before the pair is frozen, or the smoke lock held): the job ends 0."""
    job.scenario({**MONITOR, 'hyak.train.run_segment': [{'rc': 75}]})
    rc, out = job.run('smoke-train', HYAK_RUN='rl')
    assert rc == 0, out
    status = job.status('smoke-train')
    assert status['state'] != 'failed' and not status.get('error')
    assert not (job.monitor / 'smoke-train' / 'HALTED.json').exists()
    (event,) = [e for e in job.events('smoke-train') if e.get('event') == 'not_now']
    assert event['reason'] == 'smoke_not_now' and 'frozen uniform pair' in event['note'] and 'smoke lock' in event['note']
    assert 'not now' in out and 'smoke_verified' not in out


def test_smoke_train_needs_run(job):
    rc, out = job.run('smoke-train')
    assert rc == 1 and 'uniform or rl' in out and job.tools() == []


# --- evaluation ---------------------------------------------------------------------------------------------------
def test_eval_prepare_order(job):
    rc, out = job.run('eval-prepare')
    assert rc == 0, out
    assert job.tools() == [['hyak.eval.rebuild_panel', '--panel', 'all'], ['hyak.eval.prepare_features'],
                           ['hyak.eval.validate_prefixes', '--panel', 'pool'],
                           ['hyak.eval.validate_prefixes', '--panel', 'natural']]
    assert {c['env']['TOKENIZERS_PARALLELISM'] for c in job.calls()} == {'false'}


@pytest.mark.parametrize('rc,needle', [(4, 'did not rebuild exactly'), (2, 'usage or input error'), (1, 'crashed')])
def test_eval_prepare_rebuild_panel_failure_stops_the_job(job, rc, needle):
    """rebuild_panel exits 4 when a stored panel does not rebuild exactly (2 is a usage or input error)."""
    job.scenario({'hyak.eval.rebuild_panel': [{'rc': rc}]})
    code, out = job.run('eval-prepare')
    assert code == rc and job.tools() == [['hyak.eval.rebuild_panel', '--panel', 'all']]
    status = job.status('eval-prepare')
    assert status['state'] == 'failed' and needle in status['error']
    if rc == 4:
        assert 'rebuild-report.json' in status['error']


def test_eval_prepare_protocol_stop_of_the_pool_panel_still_validates_the_natural_panel(job):
    """The natural panel does not depend on the pool panel (eval-2): the job goes on, then ends with the stop's status."""
    job.scenario({'hyak.eval.validate_prefixes pool': [{'rc': 3}]})
    rc, out = job.run('eval-prepare')
    assert rc == 3, out
    assert job.tools()[-2:] == [['hyak.eval.validate_prefixes', '--panel', 'pool'],
                                ['hyak.eval.validate_prefixes', '--panel', 'natural']]
    error = job.status('eval-prepare')['error']
    assert 'decision is needed' in error and 'panels/pool/validation-report.json' in error
    assert 'panels/natural/validation-report.json' not in error        # the natural panel validated fine
    assert job.status('eval-prepare')['state'] == 'failed'


def test_eval_prepare_both_panels_stopping_names_both(job):
    job.scenario({'hyak.eval.validate_prefixes': [{'rc': 3}]})
    rc, out = job.run('eval-prepare')
    assert rc == 3 and len([t for t in job.tools() if t[0] == 'hyak.eval.validate_prefixes']) == 2
    error = job.status('eval-prepare')['error']
    assert 'panels/pool/validation-report.json' in error and 'panels/natural/validation-report.json' in error


def test_eval_prepare_other_failures_stop_at_once_and_extra_args_reach_both_panels(job):
    job.scenario({'hyak.eval.validate_prefixes pool': [{'rc': 1}]})
    rc, out = job.run('eval-prepare', HYAK_EVAL_PREPARE_ARGS='--workers 2 --recheck')
    assert rc == 1 and job.tools()[-1] == ['hyak.eval.validate_prefixes', '--panel', 'pool', '--workers', '2', '--recheck']
    assert 'crashed: uncaught exception' in job.status('eval-prepare')['error']
    job.log.unlink()
    job.scenario({})
    rc, out = job.run('eval-prepare', HYAK_EVAL_PREPARE_ARGS='--workers 2')
    assert rc == 0, out
    assert job.tools()[-2:] == [['hyak.eval.validate_prefixes', '--panel', 'pool', '--workers', '2'],
                                ['hyak.eval.validate_prefixes', '--panel', 'natural', '--workers', '2']]


def test_eval_score_full(job):
    job.scenario({'hyak.eval.pack': [{'rc': 75}, {'rc': 0}]})          # 75 = not complete yet: not a failure
    rc, out = job.run('eval-rl')
    assert rc == 0, out
    assert 'not complete yet' in out and job.status('eval-rl')['state'] != 'failed'
    assert job.tools() == [['hyak.eval.score', '--run', 'rl', '--panel', 'pool'],
                           ['hyak.eval.score', '--run', 'rl', '--panel', 'natural'],
                           ['hyak.eval.pack', '--panel', 'pool'], ['hyak.eval.pack', '--panel', 'natural']]
    assert job.calls()[0]['env']['HYAK_RUN'] == 'rl'


def test_eval_score_75_continues_with_the_next_panel_and_packing_then_ends_as_not_now(job):
    """score exit 75 = this panel's work list is not complete (other live jobs hold claims): not an error."""
    job.scenario({'hyak.eval.score pool': [{'rc': 75}], 'hyak.eval.score natural': [{'rc': 0}]})
    rc, out = job.run('eval-rl')
    assert rc == 0, out
    assert job.tools() == [['hyak.eval.score', '--run', 'rl', '--panel', 'pool'],
                           ['hyak.eval.score', '--run', 'rl', '--panel', 'natural'],
                           ['hyak.eval.pack', '--panel', 'pool'], ['hyak.eval.pack', '--panel', 'natural']]
    status = job.status('eval-rl')
    assert status['state'] == 'paused' and not status.get('error'), status        # the stage is NOT complete
    assert not (job.monitor / 'eval-rl' / 'HALTED.json').exists()
    (event,) = [e for e in job.events('eval-rl') if e.get('event') == 'not_now']
    assert event['reason'] == 'eval_incomplete' and 'panel(s) pool of rl' in event['note']
    assert 'natural' not in event['note'].split('not completely')[0]


def test_eval_score_75_on_every_panel_names_both_and_still_packs(job):
    job.scenario({'hyak.eval.score': [{'rc': 75}]})
    rc, out = job.run('eval-uniform')
    assert rc == 0, out
    assert [t[0] for t in job.tools()] == ['hyak.eval.score', 'hyak.eval.score', 'hyak.eval.pack', 'hyak.eval.pack']
    (event,) = [e for e in job.events('eval-uniform') if e.get('event') == 'not_now']
    assert 'panel(s) pool natural of uniform' in event['note']


def test_eval_score_75_in_a_pilot_is_not_now_too(job):
    job.scenario({'hyak.eval.score': [{'rc': 75}]})
    rc, out = job.run('eval-uniform', HYAK_EVAL_INDICES='0 5')
    assert rc == 0 and [t[0] for t in job.tools()] == ['hyak.eval.score'], out
    assert job.status('eval-uniform')['state'] == 'paused'


def test_eval_score_extra_arguments_reach_every_score_call(job):
    rc, out = job.run('eval-uniform', HYAK_EVAL_SCORE_ARGS='--checkpoint /archive/step-020000')
    assert rc == 0, out
    scores = [t for t in job.tools() if t[0] == 'hyak.eval.score']
    assert scores == [['hyak.eval.score', '--run', 'uniform', '--panel', panel, '--checkpoint', '/archive/step-020000']
                      for panel in ('pool', 'natural')]
    job.log.unlink()
    rc, out = job.run('eval-uniform', HYAK_EVAL_INDICES='0 5', HYAK_EVAL_SCORE_ARGS='--checkpoint /archive/step-020000')
    assert rc == 0 and job.tools() == [['hyak.eval.score', '--run', 'uniform', '--panel', 'pool', '--indices', '0', '5',
                                        '--checkpoint', '/archive/step-020000']]


def test_eval_score_pilot(job):
    rc, out = job.run('eval-uniform', HYAK_EVAL_INDICES='0 616 822 192')
    assert rc == 0, out
    assert job.tools() == [['hyak.eval.score', '--run', 'uniform', '--panel', 'pool', '--indices',
                            '0', '616', '822', '192']]


def test_eval_score_failures(job):
    job.scenario({'hyak.eval.pack': [{'rc': 2}]})
    rc, out = job.run('eval-uniform', HYAK_EVAL_PANELS='natural')
    assert rc == 2 and [t[0] for t in job.tools()] == ['hyak.eval.score', 'hyak.eval.pack']
    # a crash of pack is a failure, never "not complete yet" (slurm-1: exit 1 used to mean both)
    job.log.unlink()
    job.scenario({'hyak.eval.pack': [{'rc': 1}]})
    rc, out = job.run('eval-uniform', HYAK_EVAL_PANELS='natural')
    assert rc == 1 and job.status('eval-uniform')['state'] == 'failed'
    assert 'crashed: uncaught exception' in job.status('eval-uniform')['error']
    job.log.unlink()
    job.scenario({'hyak.eval.score': [{'rc': 137}]})
    rc, out = job.run('eval-uniform', HYAK_EVAL_PANELS='natural')
    assert rc == 137 and 'killed' in job.status('eval-uniform')['error'] and job.status('eval-uniform')['state'] == 'failed'
    job.log.unlink()
    job.scenario({'hyak.eval.score': [{'rc': 2}]})                       # an input or configuration error stops at once
    rc, out = job.run('eval-uniform')
    assert rc == 2 and [t[0] for t in job.tools()] == ['hyak.eval.score']
    assert 'input or configuration error' in job.status('eval-uniform')['error']
    rc, out = job.run('eval-uniform', HYAK_EVAL_PANELS='pool bogus')
    assert rc == 1 and "only 'pool' and 'natural'" in out


def test_stale_stop_flag_in_the_environment_is_ignored(job):
    rc, out = job.run('init', HYAK_TERM_RECEIVED='1')
    assert rc == 0, out
    assert job.tools() == [['hyak.model.init_hypernet', '--status']]


def run_with_term(job, stage, **env):
    """Start ``stage``; send SIGTERM to the batch shell once the scripted step wrote its ready file."""
    return run_with_term_file(job, stage, job.tmp / 'step-ready', **env)


def run_with_term_file(job, stage, ready, **env):
    """Start ``stage``; send SIGTERM to the batch shell once ``ready`` exists."""
    proc = job.start(stage, **env)
    try:
        deadline = time.time() + 30
        while not ready.exists():
            assert time.time() < deadline and proc.poll() is None, job.slurm_log(stage).read_text()
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    return proc.returncode, job.slurm_log(stage).read_text()


@pytest.mark.parametrize('stage,key,graceful_rc,expected_tools', [
    ('eval-prepare', 'hyak.eval.prepare_features', 0,
     [['hyak.eval.rebuild_panel', '--panel', 'all'], ['hyak.eval.prepare_features']]),
    ('eval-uniform', 'hyak.eval.score', 0, [['hyak.eval.score', '--run', 'uniform', '--panel', 'pool']]),
    ('corpus', 'hyak.corpus.build', 143, [['hyak.corpus.build', '--probe'], ['hyak.corpus.build']]),
])
def test_stop_signal_between_steps(job, stage, key, graceful_rc, expected_tools):
    """Slurm's TERM is forwarded to the running program; no further program starts afterwards."""
    job.scenario({key: [{'wait_term': True, 'rc_on_term': graceful_rc, 'ready': str(job.tmp / 'step-ready')}],
                  'hyak.corpus.build --probe': [{'rc': 0}]})
    rc, out = run_with_term(job, stage)
    assert rc == graceful_rc, out
    assert job.tools() == expected_tools
    assert [e['terminated'] for e in job.calls('exit')] == [True]
    assert job.status(stage)['state'] == ('paused' if graceful_rc == 0 else 'stopped')


# --- the exit-code contract in every job script (slurm-1, monitoring-v1) --------------------------------------------
#: stage -> (scenario key of the program whose status decides the job, extra scenario entries, env)
CONTRACT_STAGES = {
    'check-env': ('hyak.env.check_env', {}, {}),
    'probe': ('hyak.memory.probe', {}, {}),
    'init': ('hyak.model.init_hypernet', {}, {}),
    'stage-inputs': ('hyak.corpus.stage_inputs', {}, {}),
    'replay-check': ('hyak.train.replay_check', {}, {}),
    'corpus': ('hyak.corpus.build', {}, {'HYAK_CORPUS_PROBE': '0'}),
    'eval-prepare': ('hyak.eval.rebuild_panel', {}, {}),
    'eval-uniform': ('hyak.eval.score', {}, {}),
    'pool': ('hyak.pool.worker', {}, {}),
}


@pytest.mark.parametrize('stage', sorted(CONTRACT_STAGES))
def test_exit_1_is_a_crash_and_137_is_killed_in_every_stage(job, stage):
    key, extra, env = CONTRACT_STAGES[stage]
    for code, state, needle in ((1, 'failed', 'crashed: uncaught exception, traceback in the job log'),
                                (137, 'failed', 'killed')):
        job.scenario({key: [{'rc': code}], **extra})
        rc, out = job.run(stage, **env)
        assert rc == code, out
        status = job.status(stage)
        assert status['state'] == state and needle in status['error'], (stage, code, status['error'])
        job.log.unlink()


def test_smoke_train_maps_the_contract_too(job):
    for code, needle in ((1, 'crashed'), (2, 'configuration, input or environment error'), (3, 'needs your decision'),
                         (4, 'a pre-flight check failed'), (137, 'killed')):
        job.scenario({**MONITOR, 'hyak.train.run_segment': [{'rc': code}]})
        rc, out = job.run('smoke-train', HYAK_RUN='uniform')
        assert rc == code, out
        assert needle in job.status('smoke-train')['error'] and job.status('smoke-train')['state'] == 'failed'
        job.log.unlink()


@pytest.mark.parametrize('stage,key', [('init', 'hyak.model.init_hypernet'), ('check-env', 'hyak.env.check_env')])
def test_a_stop_that_was_asked_for_is_still_stopped_not_failed(job, stage, key):
    """143 stays "stopped" (an asked-for stop); only 137 and the other 128+n are failures."""
    job.scenario({key: [{'rc': 143}]})
    rc, out = job.run(stage)
    assert rc == 143 and job.status(stage)['state'] == 'stopped'


# --- the job environment (train-lifecycle-13, slurm-11) -----------------------------------------------------------------
def test_login_environment_is_sanitised_and_recorded(job, tmp_path):
    rc, out = job.run('init', PYTHONPATH='/junk', PYTHONHOME='/home/x', PYTHONSTARTUP='/s.py', PYTHONUSERBASE='/u',
                      LD_PRELOAD='/lib/preload.so', CUDA_VISIBLE_DEVICES='3', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1')
    assert rc == 0, out
    (call,) = [c for c in job.calls() if c['kind'] == 'tool']
    env = call['env']
    assert env['PYTHONPATH'] == str(REPO)                      # hyak_activate's, never the login shell's
    assert env['PYTHONHOME'] == env['PYTHONSTARTUP'] == env['PYTHONUSERBASE'] == env['LD_PRELOAD'] == '<unset>'
    assert env['PYTHONNOUSERSITE'] == '1'
    assert env['CUDA_VISIBLE_DEVICES'] == '3'                  # Slurm's: never touched
    (event,) = [e for e in job.events('init') if e.get('event') == 'environment_sanitized']
    assert event['removed'] == 'PYTHONPATH,PYTHONHOME,PYTHONSTARTUP,PYTHONUSERBASE,LD_PRELOAD'
    assert 'removed from the login environment' in out and '/junk' not in out and '/home/x' not in out   # names only


def test_library_path_holds_only_the_libpython_link_and_never_accumulates(job):
    for which in ('train', 'data'):
        (job.root / 'env' / which / 'lib' / 'libpython3.12.so.1.0').write_bytes(b'')
    stale = f'{job.root}/env/data/lib:{job.root}/env/train/hyak-libpython:/usr/lib64:/sw/cuda-compat'
    rc, out = job.run('pool', LD_LIBRARY_PATH=stale, HYAK_POOL_ARGS='--pilots-only')
    assert rc == 0, out
    (call,) = job.calls()
    assert call['env']['LD_LIBRARY_PATH'] == f'{job.root}/env/data/hyak-libpython:/usr/lib64:/sw/cuda-compat'
    link = job.root / 'env' / 'data' / 'hyak-libpython' / 'libpython3.12.so.1.0'
    assert link.is_symlink() and os.readlink(link) == '../lib/libpython3.12.so.1.0'
    assert [p.name for p in link.parent.iterdir()] == ['libpython3.12.so.1.0']      # nothing else of conda's lib
    # the pool job switches from the data to the train environment: the paths are replaced, not stacked
    job.log.unlink()
    rc, out = job.run('pool', LD_LIBRARY_PATH=stale)
    assert rc == 0, out
    after = [c['env']['LD_LIBRARY_PATH'] for c in job.calls()]
    assert after[0] == f'{job.root}/env/data/hyak-libpython:/usr/lib64:/sw/cuda-compat'
    assert after[1:] == [f'{job.root}/env/train/hyak-libpython:/usr/lib64:/sw/cuda-compat'] * (len(after) - 1)


OFFLINE_STAGES = ['train-uniform', 'train-rl', 'smoke-train', 'eval-prepare', 'eval-uniform', 'eval-rl', 'probe', 'init',
                  'replay-check']
ONLINE_STAGES = ['corpus', 'pool', 'wheel', 'stage-inputs', 'check-env']


@pytest.mark.parametrize('stage', OFFLINE_STAGES + ['corpus', 'pool', 'check-env', 'stage-inputs'])
def test_hub_is_offline_exactly_for_the_stages_that_read_only_staged_files(job, stage):
    if stage != 'corpus':                       # (conftest exports HF_HUB_OFFLINE=1; corpus must clear it, the others leave it)
        job.env.pop('HF_HUB_OFFLINE', None)
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(20000, 'complete', False)]})
    env = {'HYAK_RUN': 'uniform'} if stage == 'smoke-train' else {}
    if stage == 'pool':
        env['HYAK_POOL_ARGS'] = '--pilots-only'
    job.run(stage, **env)
    call = next(c for c in job.calls() if c['kind'] == 'tool' and c['argv'][1] != 'hyak.train.monitor')
    assert call['env']['HF_HUB_OFFLINE'] == ('1' if stage in OFFLINE_STAGES else '<unset>'), stage


@pytest.mark.parametrize('stage', ['train-uniform', 'train-rl', 'smoke-train', 'init', 'eval-rl'])
def test_the_training_stages_never_see_the_tf32_override_or_pythonoptimize(job, stage):
    """The pinned trainers are validated without TF32 override and with asserts on; PYTORCH_CUDA_ALLOC_CONF stays."""
    job.scenario({**MONITOR, 'hyak.train.run_segment --should-start': [should_start(20000, 'complete', False)]})
    env = {'HYAK_RUN': 'uniform'} if stage == 'smoke-train' else {}
    job.run(stage, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1', PYTHONOPTIMIZE='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', **env)
    call = next(c for c in job.calls() if c['kind'] == 'tool')
    expected = ('<unset>', '<unset>') if stage in ('train-uniform', 'train-rl', 'smoke-train') else ('1', '2')
    assert (call['env']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'], call['env']['PYTHONOPTIMIZE']) == expected, stage
    assert call['env']['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'


# --- the registry lock must not hold a job in its prologue (slurm-4) ----------------------------------------------------
def test_a_registry_lock_that_never_returns_is_given_up_with_a_warning(job):
    start = time.time()
    rc, out = job.run('init', FAKE_REGISTER_HANG='60', HYAK_REGISTER_TIMEOUT_SECONDS='2')
    assert rc == 0, out
    assert time.time() - start < 40, 'the job waited for the stuck registration'
    assert 'dashboard registration did not finish within 2 s' in out
    assert job.tools() == [['hyak.model.init_hypernet', '--status']]        # the job went on


# --- paths that conda would rewrite (slurm-3) -------------------------------------------------------------------------
@pytest.mark.parametrize('suffix', ['/', '//', '/./', '/sub/..'])
def test_an_unnormalised_storage_path_is_refused_before_any_job_starts(job, suffix):
    rc, out = job.run('init', ANYTOK_ROOT=f'{job.root}{suffix}')
    assert rc == 1 and 'normalised path' in out and job.tools() == [] and 'hyak_activate' not in out


# --- a TERM that arrives between two programs reaches the next one (slurm-v2) ---------------------------------------------
def test_a_signal_recorded_while_no_child_ran_is_delivered_to_the_next_child(tmp_path):
    child = tmp_path / 'child.sh'
    child.write_text('#!/bin/bash\nsleep 20 &\nwait $!\nexit 99\n')            # no trap: the signal's default action ends it
    child.chmod(0o755)
    script = (f'set -euo pipefail\nsource {shlex.quote(str(SLURM_DIR / "lib.sh"))}\nhyak_trap_term\n'
              f'kill -TERM $$                                  # lands while no child runs (a few ms window in job_step)\n'
              f'rc=0; hyak_run_forwarding {shlex.quote(str(child))} || rc=$?\necho "rc=$rc"\n')
    start = time.time()
    out = subprocess.run([BASH, '-c', script], capture_output=True, text=True, timeout=60,
                         env={**os.environ, 'HYAK_REPO': str(REPO)})
    # before the fix the flag was reset when the child started, the TERM was lost, and the child ran its 20 s (rc=99)
    assert time.time() - start < 15, out.stderr
    assert 'rc=143' in out.stdout, (out.stdout, out.stderr)
    assert 'arrived before child' in out.stderr


# --- stop.sh ------------------------------------------------------------------------------------------------------------
@pytest.fixture
def fake_queue(tmp_path):
    bindir = tmp_path / 'qbin'
    bindir.mkdir()
    log = tmp_path / 'scancel.log'
    (bindir / 'squeue').write_text('#!/bin/bash\nprintf "%s" "$FAKE_SQUEUE"\n')
    (bindir / 'scancel').write_text(f'#!/bin/bash\necho "scancel $*" >> {shlex.quote(str(log))}\n')
    for tool in ('squeue', 'scancel'):
        (bindir / tool).chmod(0o755)
    return {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}'}, log


def stop(args, env, queue):
    return subprocess.run([BASH, str(SLURM_DIR / 'stop.sh'), *args], env={**env, 'FAKE_SQUEUE': queue},
                          capture_output=True, text=True, timeout=60)


def test_stop_cancels_the_queue_first_then_terms_the_running_batch_shell(fake_queue):
    env, log = fake_queue
    out = stop(['train-rl'], env, '101 RUNNING\n102 PENDING\n103 PENDING\n104 COMPLETING\n')
    assert out.returncode == 0, out.stderr
    assert log.read_text().splitlines() == ['scancel 102', 'scancel 103', 'scancel --batch --signal=TERM 101']
    assert 'queued: 102 103; running: 101' in out.stdout and '104(COMPLETING) (not touched)' in out.stdout
    assert 'latest-slurm.out' in out.stdout


def test_stop_dry_run_and_empty_queue_change_nothing(fake_queue):
    env, log = fake_queue
    out = stop(['train-uniform', '--dry-run'], env, '7 RUNNING\n8 PENDING\n')
    assert out.returncode == 0 and not log.exists() and 'dry run: scancel --batch --signal=TERM 7' in out.stdout
    out = stop(['pool'], env, '')
    assert out.returncode == 0 and 'No job named anytok-pool' in out.stdout and not log.exists()


def test_stop_refuses_unknown_stages_and_options(fake_queue):
    env, log = fake_queue
    assert stop(['train'], env, '').returncode == 1
    out = stop(['pool', '--now'], env, '')
    assert out.returncode == 1 and 'Unknown option' in out.stderr and not log.exists()
    assert stop([], env, '').returncode == 2
    assert stop(['--help'], env, '').stdout.startswith('hyak/slurm/stop.sh')
