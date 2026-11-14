"""The raw-BPB handoff stages (README section 0) end to end with bash, against fakes.

* ``12_smoke_rawbpb.sbatch`` and ``13_train_rawbpb.sbatch`` run under bash with a fake ``conda`` and a fake train
  interpreter.  Every Python call goes to the real interpreter (so ``hyak.handoff.run`` and ``hyak.common.status``
  are the real ones) except Qingyun's launcher ``scripts/run_raw_bpb_training.py``, which is replaced by
  ``FAKE_LAUNCHER``: a stand-in that writes what the real trainer writes (run_config.json, events.jsonl,
  step-NNNNNN with manifest.json and a torch state.pt, the launcher's status.json), refuses what the trainer
  refuses, and saves a full restart on SIGTERM.  ``register_dashboard`` and ``policy_monitor`` are intercepted.
* ``hyak.handoff.run`` prepare / verify directly; ``hyak.handoff.fetch`` and ``fetch_inputs.sh`` with a small fake
  bundle (1,000 tiny tokenizers, 23 corpus files) built here; the registry entries; submit.sh dry runs; the
  environment pins against docs/raw-bpb-handoff/requirements.txt.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import types

import pytest

from hyak.handoff import common as C
from hyak.handoff import fetch as F
from hyak.handoff import run as R

BASH = shutil.which('bash')
REPO = Path(__file__).resolve().parents[2]
SLURM_DIR = REPO / 'hyak' / 'slurm'
SUBMIT = SLURM_DIR / 'submit.sh'
REGISTRY = REPO / 'plans' / 'run-dashboard.json'
needs_bash = pytest.mark.skipif(BASH is None, reason='bash not available')


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def registry_untouched():
    before = sha(REGISTRY)
    yield
    assert sha(REGISTRY) == before, 'plans/run-dashboard.json was modified by a test'


# --- a fake trainer output ----------------------------------------------------------------------------------
RUN_CONFIG = {**C.EXPECTED_CONFIG, 'policy': dict(C.EXPECTED_POLICY), 'main_updates': 190000}


def train_row(step, **override):
    train = 1.0 + step * 1e-4
    canonical = 0.8
    row = {'stage': 'train', 'step': step, 'train_bpb': train, 'paired_canonical_bpb': canonical,
           'paired_gap_bpb': train - canonical, 'policy_reward_bpb': train, 'paired_valid_examples': 128,
           'paired_maximum_positions': 128, 'policy_updates': step, 'reward_objective': 'raw_bpb',
           'seconds': 0.01, 'peak_gib': 1.0, 'tokenizer_config_index': step % 1000}
    row.update(override)
    return row


def save_checkpoint(output: Path, step: int, full: bool, counts_off=0):
    import torch
    stage = output / f'.writing-{step}'
    stage.mkdir()
    (stage / 'lm.pt').write_bytes(b'lm' * 8)
    (stage / 'policy.pt').write_bytes(b'policy')
    (stage / 'hypernet').mkdir()
    (stage / 'hypernet' / 'model.safetensors').write_bytes(b'hn')
    if full:
        (stage / 'optimizer.pt').write_bytes(b'opt' * 4)
        counts = torch.zeros(1000, dtype=torch.long)
        counts[0] = step + counts_off
        torch.save({'step': step, 'bandit': {'updates': step, 'counts': counts}}, stage / 'state.pt')
    files = {p.relative_to(stage).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage / 'manifest.json').write_text(json.dumps({'step': step, 'files': files, 'kind':
                                                      'full_restart_checkpoint' if full else 'weights_only_checkpoint'}))
    os.replace(stage, output / f'step-{step:06d}')


def make_run(output: Path, steps: int, *, full_at=None, weights_after=()):
    """A trainer output that stopped at ``steps`` with a full restart at ``full_at`` (default ``steps``)."""
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_config.json').write_text(json.dumps(RUN_CONFIG))
    rows = [{'stage': 'initial_eval', 'results': []}] + [train_row(s) for s in range(1, steps + 1)]
    (output / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    save_checkpoint(output, 0, True)
    save_checkpoint(output, steps if full_at is None else full_at, True)
    for step in weights_after:
        save_checkpoint(output, step, False)
    (output / 'status.json').write_text(json.dumps({'stage': 'stopped', 'step': steps}))


FAKE_LAUNCHER = r'''
import argparse, json, os, signal, sys, time
from pathlib import Path
sys.path.insert(0, os.environ['FAKE_TEST_DIR'])
import test_handoff_rawbpb as T

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--stop-at-update', type=int, default=20000)
args = parser.parse_args()
state = Path(os.environ['FAKE_STATE'])
counter = state / 'launches'
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
plan = json.loads(os.environ.get('FAKE_LAUNCH', '[{}]'))
step_plan = plan[min(n, len(plan) - 1)]
with open(os.environ['FAKE_LOG'], 'a') as log:
    log.write(json.dumps({'kind': 'launcher', 'argv': sys.argv[1:], 'pid': os.getpid(),
                          'env': {k: os.environ.get(k) for k in ('ANYTOK_RUN_PLATFORM', 'ANYTOK_RUN_PROFILE',
                                  'TOKENIZERS_PARALLELISM', 'PYTORCH_ALLOC_CONF', 'HF_HUB_OFFLINE')}}) + '\n')
out = args.output
print(json.dumps({'stage': 'training_command', 'command': ['fake']}), flush=True)
if step_plan.get('rc_at_start') is not None:
    print('Traceback: fake crash at start', flush=True)
    sys.exit(step_plan['rc_at_start'])
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

def record(row):
    print(json.dumps(row), flush=True)
    with (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')

def status(stage, step):
    (out / 'status.json').write_text(json.dumps({'stage': stage, 'step': step, 'total': args.stop_at_update,
                                                  'workers': []}))

if args.resume:
    if not out.is_dir():
        print('error: --resume requires an existing run directory', flush=True); sys.exit(2)
    restarts = T.C.full_restarts(out)
    if not restarts:
        print('ValueError: No complete checkpoint to resume', flush=True); sys.exit(1)
    start = restarts[-1][0]
else:
    if out.exists() and any(out.iterdir()):
        print('error: Output directory must be empty unless --resume is set', flush=True); sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps(T.RUN_CONFIG))
    record({'stage': 'initial_eval', 'results': []})
    T.save_checkpoint(out, 0, True)
    start = 0
stop_at = min(args.stop_at_update, step_plan.get('run_to', args.stop_at_update))
if step_plan.get('ready'):
    Path(step_plan['ready']).write_text('ready')
step = start
for step in range(start + 1, stop_at + 1):
    override = {}
    if step_plan.get('corrupt') == 'reward':
        override['policy_reward_bpb'] = 0.5
    if step_plan.get('corrupt') == 'gap':
        override['paired_gap_bpb'] = 0.123
    record(T.train_row(step, **override))
    status('train', step)
    final = step == stop_at and not step_plan.get('wait_term')
    if step % 500 == 0 or final:
        path = out / f'step-{step:06d}'
        if path.exists():
            print(f'FileExistsError: {path}', flush=True); sys.exit(1)
        T.save_checkpoint(out, step, step % 2000 == 0 or final, counts_off=step_plan.get('counts_off', 0))
if step_plan.get('wait_term'):
    deadline = time.time() + 60
    while not stop and time.time() < deadline:
        time.sleep(0.05)
    with open(os.environ['FAKE_LOG'], 'a') as log:
        log.write(json.dumps({'kind': 'launcher_term', 'terminated': bool(stop)}) + '\n')
    record({'stage': 'stop_requested', 'signal': 15})
    step += 1
    record(T.train_row(step))
    T.save_checkpoint(out, step, True)
rc = step_plan.get('rc', 0)
record({'stage': 'paused_at', 'step': step})
status('failed' if rc else ('complete' if step >= args.stop_at_update else 'stopped'), step)
sys.exit(rc)
'''

FAKE_DRIVER = r'''
import json, os, signal, sys, time
real = os.environ['FAKE_REAL_PYTHON']
args = sys.argv[1:]
def log(entry):
    with open(os.environ['FAKE_LOG'], 'a') as handle:
        handle.write(json.dumps(entry) + '\n')
words = [a for a in args if a != '-u']
if words and words[0].endswith('run_raw_bpb_training.py'):
    os.execv(real, [real, os.environ['FAKE_LAUNCHER'], *words[1:]])
if args[:2] == ['-m', 'hyak.tools.register_dashboard']:
    log({'kind': 'register'})
    sys.exit(0)
if args[:2] == ['-m', 'hyak.tools.policy_monitor']:
    log({'kind': 'policy_monitor', 'argv': args})
    if '--follow' in args:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        while True:
            time.sleep(0.2)
    sys.exit(0)
os.execv(real, [real] + args)
'''

FAKE_CONDA = ('#!/bin/sh\n[ "$1 $2" = "shell.bash hook" ] || exit 3\n'
              "cat <<'EOF'\nconda() { if [ \"$1\" = activate ]; then export CONDA_PREFIX=\"$2\"; "
              'export PATH="$2/bin:$PATH"; else command conda "$@"; fi; }\nEOF\n')


class RawJob:
    def __init__(self, tmp_path, make_tmp_config):
        self.t = make_tmp_config(HYAK_MODULES='')
        self.tmp = tmp_path
        self.root, self.runs = self.t.paths.root, self.t.paths.runs
        self.monitor = tmp_path / 'monitor'
        self.log = tmp_path / 'calls.jsonl'
        self.state = tmp_path / 'fake-state'
        self.state.mkdir()
        (tmp_path / 'fake_driver.py').write_text(FAKE_DRIVER)
        (tmp_path / 'fake_launcher.py').write_text(FAKE_LAUNCHER)
        prefix = self.root / 'env' / 'train'
        (prefix / 'bin').mkdir(parents=True)
        (prefix / 'lib').mkdir()
        python = prefix / 'bin' / 'python'
        python.write_text(f'#!/bin/bash\nexec "$FAKE_REAL_PYTHON" {shlex.quote(str(tmp_path / "fake_driver.py"))} "$@"\n')
        python.chmod(0o755)
        bindir = tmp_path / 'fakebin'
        bindir.mkdir()
        (bindir / 'conda').write_text(FAKE_CONDA)
        (bindir / 'conda').chmod(0o755)
        self.env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_REPO': str(REPO),
                    'FAKE_REAL_PYTHON': sys.executable, 'FAKE_LOG': str(self.log), 'FAKE_STATE': str(self.state),
                    'FAKE_LAUNCHER': str(tmp_path / 'fake_launcher.py'),
                    'FAKE_TEST_DIR': str(Path(__file__).resolve().parent),
                    'HYAK_PYTHON': str(python), 'SLURM_JOB_ID': '4242', 'PYTHONDONTWRITEBYTECODE': '1'}
        self.output = self.runs / 'raw-bpb-20k'
        self.smoke = self.runs / 'raw-bpb-smoke'

    def launch_plan(self, steps):
        self.env['FAKE_LAUNCH'] = json.dumps(steps)
        counter = self.state / 'launches'
        if counter.exists():
            counter.unlink()

    def start(self, stage, **env):
        script = SLURM_DIR / ('13_train_rawbpb.sbatch' if stage == 'train-rawbpb' else '12_smoke_rawbpb.sbatch')
        out = self.monitor / stage / 'slurm-4242.out'
        out.parent.mkdir(parents=True, exist_ok=True)
        handle = open(out, 'a')
        return subprocess.Popen([BASH, str(script)], env={**self.env, 'HYAK_STAGE': stage, **env}, cwd=str(REPO),
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    def run(self, stage, timeout=300, **env):
        proc = self.start(stage, **env)
        try:
            proc.wait(timeout=timeout)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        return proc.returncode, (self.monitor / stage / 'slurm-4242.out').read_text()

    def calls(self, kind):
        if not self.log.exists():
            return []
        return [r for r in map(json.loads, self.log.read_text().splitlines()) if r['kind'] == kind]

    def status(self, stage):
        return json.loads((self.monitor / stage / 'status.json').read_text())

    def events(self, stage):
        path = self.monitor / stage / 'events.jsonl'
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def halted(self):
        return self.monitor / 'train-rawbpb' / 'HALTED.json'


@pytest.fixture
def raw(tmp_path, make_tmp_config):
    return RawJob(tmp_path, make_tmp_config)


# --- train-rawbpb ---------------------------------------------------------------------------------------------
@needs_bash
@pytest.mark.slow
def test_train_fresh_then_resume_then_complete(raw):
    raw.launch_plan([{'run_to': 2300}, {}])
    rc, out = raw.run('train-rawbpb')
    assert rc == 0, out
    (call,) = raw.calls('launcher')
    assert '--resume' not in call['argv'] and call['argv'][call['argv'].index('--output') + 1] == str(raw.output)
    assert call['argv'][call['argv'].index('--inputs') + 1] == str(raw.root / 'input')
    assert call['env'] == {'ANYTOK_RUN_PLATFORM': 'Hyak Tillicum · H200', 'ANYTOK_RUN_PROFILE': 'local',
                           'TOKENIZERS_PARALLELISM': 'false', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
                           'HF_HUB_OFFLINE': '1'}
    assert (raw.monitor / 'train-rawbpb' / 'run').resolve() == raw.output.resolve()
    assert json.loads((raw.monitor / 'train-rawbpb' / 'run' / 'status.json').read_text())['step'] == 2300
    status = raw.status('train-rawbpb')
    assert status['state'] == 'paused' and status['step'] == 2300
    assert any(e.get('event') == 'rawbpb_verified' for e in raw.events('train-rawbpb'))
    assert [c['argv'][:3] for c in raw.calls('policy_monitor')] == [['-m', 'hyak.tools.policy_monitor', '--follow'],
                                                                     ['-m', 'hyak.tools.policy_monitor', '--once']]
    log = (raw.monitor / 'train-rawbpb' / '4242-train.log').read_text()
    assert '"stage": "rawbpb_verified"' in log and '"action": "fresh"' in log
    # job 2: resumes from the full restart at 2,300 and reaches 20,000
    rc, out = raw.run('train-rawbpb')
    assert rc == 0, out
    second = raw.calls('launcher')[1]
    assert '--resume' in second['argv']
    assert raw.status('train-rawbpb')['state'] == 'complete' and raw.status('train-rawbpb')['step'] == 20000
    assert C.full_restarts(raw.output)[-1][0] == 20000
    # job 3: complete, nothing launched
    rc, out = raw.run('train-rawbpb')
    assert rc == 0, out
    assert len(raw.calls('launcher')) == 2
    status = raw.status('train-rawbpb')
    assert status['state'] == 'complete' and 'complete' in status.get('note', '')
    assert not raw.halted().exists()


@needs_bash
def test_train_refuses_partial_state_and_halts(raw):
    raw.output.mkdir(parents=True)
    (raw.output / 'run_config.json').write_text(json.dumps(RUN_CONFIG))
    (raw.output / 'events.jsonl').write_text('{"stage": "initial_eval"}\n')
    rc, out = raw.run('train-rawbpb')
    assert rc == 2, out
    assert raw.calls('launcher') == []
    assert sorted(p.name for p in raw.output.iterdir()) == ['.run.lock', 'events.jsonl', 'run_config.json']
    assert f'mv {raw.output} {raw.output}.partial-' in out
    assert json.loads(raw.halted().read_text())['exit_code'] == 2
    assert raw.status('train-rawbpb')['state'] == 'failed'
    rc, out = raw.run('train-rawbpb')           # the chain is halted
    assert rc == 3 and raw.calls('launcher') == []


@needs_bash
def test_train_resume_moves_newer_checkpoints_aside(raw):
    make_run(raw.output, 2600, full_at=2000, weights_after=(2500,))
    (raw.output / '.writing-abc').mkdir()
    raw.launch_plan([{'run_to': 2600}])
    rc, out = raw.run('train-rawbpb')
    assert rc == 0, out
    assert '--resume' in raw.calls('launcher')[0]['argv']
    (moved,) = list((raw.output / 'interrupted').iterdir())
    assert sorted(p.name for p in moved.iterdir()) == ['.writing-abc', 'step-002500']
    assert (raw.output / 'step-002500').is_dir()       # written again by the resumed run, no FileExistsError


@needs_bash
def test_train_term_reaches_the_launcher(raw):
    ready = raw.tmp / 'ready'
    raw.launch_plan([{'run_to': 1200, 'wait_term': True, 'ready': str(ready)}])
    proc = raw.start('train-rawbpb')
    deadline = time.time() + 120
    while not ready.exists() and time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)
    assert ready.exists()
    os.kill(proc.pid, signal.SIGTERM)             # Slurm's --signal=B:TERM reaches only the batch shell
    try:
        rc = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    out = (raw.monitor / 'train-rawbpb' / 'slurm-4242.out').read_text()
    assert rc == 0, out
    assert raw.calls('launcher_term') == [{'kind': 'launcher_term', 'terminated': True}]
    assert 'forwarding SIGTERM' in out
    assert C.full_restarts(raw.output)[-1][0] == 1201
    assert raw.status('train-rawbpb')['state'] == 'paused' and not raw.halted().exists()


@needs_bash
@pytest.mark.parametrize('launch,code,text', [
    ({'rc_at_start': 1}, 1, 'crashed'),
    ({'rc_at_start': 247}, 137, 'killed'),
    ({'run_to': 600, 'corrupt': 'reward'}, 4, 'receipt failed'),
])
def test_train_failures_halt_the_chain(raw, launch, code, text):
    raw.launch_plan([launch])
    rc, out = raw.run('train-rawbpb')
    assert rc == code, out
    halt = json.loads(raw.halted().read_text())
    assert halt['exit_code'] == code and text in halt['reason']
    status = raw.status('train-rawbpb')
    assert status['state'] == 'failed' and text in status['error']


@needs_bash
def test_train_refuses_another_protocol(raw):
    make_run(raw.output, 2000)
    config = json.loads((raw.output / 'run_config.json').read_text())
    config['reward_objective'] = 'regret'
    (raw.output / 'run_config.json').write_text(json.dumps(config))
    rc, out = raw.run('train-rawbpb')
    assert rc == 3, out
    assert raw.calls('launcher') == [] and 'reward_objective' in out


# --- smoke-rawbpb ---------------------------------------------------------------------------------------------
@needs_bash
def test_smoke_passes_with_good_metrics(raw):
    rc, out = raw.run('smoke-rawbpb')
    assert rc == 0, out
    calls = raw.calls('launcher')
    assert [c['argv'][c['argv'].index('--stop-at-update') + 1] for c in calls] == ['2', '3']
    assert '--resume' not in calls[0]['argv'] and '--resume' in calls[1]['argv']
    assert all(c['argv'][c['argv'].index('--output') + 1] == str(raw.smoke) for c in calls)
    assert raw.status('smoke-rawbpb')['state'] == 'complete' and raw.status('smoke-rawbpb')['step'] == 3
    log = (raw.monitor / 'smoke-rawbpb' / '4242-train.log').read_text()
    verified = [json.loads(l) for l in log.splitlines() if '"rawbpb_verified"' in l]
    assert [v['steps'] for v in verified] == [[1, 2], [1, 3]]
    assert verified[-1]['resume_counters'] == {'2': {'model': 2, 'policy': 2, 'selections': 2},
                                               '3': {'model': 3, 'policy': 3, 'selections': 3}}


@needs_bash
@pytest.mark.parametrize('launch', [{'corrupt': 'gap'}, {'corrupt': 'reward'}, {'counts_off': 1}])
def test_smoke_fails_with_broken_metrics(raw, launch):
    raw.launch_plan([launch])
    rc, out = raw.run('smoke-rawbpb')
    assert rc == 4, out
    assert 'CHECK FAILED' in out and raw.status('smoke-rawbpb')['state'] == 'failed'
    assert len(raw.calls('launcher')) == 1


@needs_bash
def test_smoke_refuses_an_existing_output(raw):
    raw.smoke.mkdir(parents=True)
    (raw.smoke / 'run_config.json').write_text('{}')
    rc, out = raw.run('smoke-rawbpb')
    assert rc == 2, out
    assert raw.calls('launcher') == [] and f'rm -rf {raw.smoke}' in out
    assert (raw.smoke / 'run_config.json').exists()


# --- hyak.handoff.run directly ----------------------------------------------------------------------------------
def test_prepare_decisions(tmp_path):
    out = tmp_path / 'run'
    assert R.prepare(out)[0] == 0 and R.prepare(out)[1]['action'] == 'fresh'
    out.mkdir()
    assert R.prepare(out)[1]['action'] == 'fresh'           # an empty directory is a fresh start
    assert not (out / '.run.lock').exists()                 # and stays empty (the trainer refuses anything else)
    make_run(out, 4000)
    code, record = R.prepare(out)
    assert (code, record['action'], record['newest_full_step']) == (0, 'resume', 4000)
    assert R.prepare(out, total=4000)[:1] == (75,) and R.prepare(out, total=4000)[1]['action'] == 'complete'
    assert R.prepare(out, smoke=True)[0] == 2
    (out / 'step-004000' / 'optimizer.pt').write_bytes(b'x')        # size differs from the manifest
    code, record = R.prepare(out)
    assert code == 3 and record['reason'] == 'newest_restart_incomplete' and 'resumes from step 0' in record['remedy']


def test_prepare_lock_held(tmp_path):
    import fcntl
    out = tmp_path / 'run'
    make_run(out, 2000)
    with open(out / '.run.lock', 'a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = ('import sys; from hyak.handoff import run; '
                  f'sys.exit(run.prepare(run.Path({str(out)!r}))[0])')
        proc = subprocess.run([sys.executable, '-c', script], cwd=str(REPO), env={**os.environ, 'PYTHONPATH': str(REPO)})
        assert proc.returncode == 75


def test_verify_checks_each_receipt_field(tmp_path):
    out = tmp_path / 'run'
    make_run(out, 3)
    (out / 'status.json').write_text(json.dumps({'stage': 'complete', 'step': 3}))
    save_checkpoint(out, 2, True)
    code, record = R.verify(out, smoke=True, expect_step=3)
    assert code == 0, record
    for field, value, needle in [('policy_reward_bpb', 1.5, 'raw BPB'), ('paired_valid_examples', 127, '128'),
                                 ('paired_maximum_positions', 129, '> 128'), ('policy_updates', 9, 'policy_updates'),
                                 ('reward_objective', 'regret', 'reward_objective'),
                                 ('paired_gap_bpb', 0.2 + 1e-11, 'paired_gap_bpb')]:
        broken = [train_row(s) for s in (1, 2, 3)]
        broken[1][field] = value
        (out / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in broken))
        code, record = R.verify(out, smoke=True, expect_step=3)
        assert code == 4 and any(needle in p for p in record['problems']), (field, record)
    (out / 'events.jsonl').write_text(''.join(json.dumps(train_row(s)) + '\n' for s in (1, 2, 3)))
    config = dict(RUN_CONFIG, policy={**C.EXPECTED_POLICY, 'entropy_weight': 0.01})
    (out / 'run_config.json').write_text(json.dumps(config))
    code, record = R.verify(out, smoke=True, expect_step=3)
    assert code == 4 and any('entropy_weight' in p for p in record['problems'])


def test_verify_progress_rule(tmp_path):
    out = tmp_path / 'run'
    make_run(out, 2000)
    assert R.verify(out, before_step=2000)[0] == 4
    code, record = R.verify(out, before_step=2000, stop_requested=True)
    assert code == 0 and 'warning' in record
    assert R.verify(out, before_step=0)[0] == 0


def test_requirements_match_the_setup_script():
    from hyak.code import pinned
    pins = C.parse_requirements(REPO / C.DOCS_REL / C.REQUIREMENTS_NAME)
    assert [f'{n}=={v}' for n, v in pins] == [f'{n}=={v}' for n, v in pinned.TRAIN_PINS + pinned.TRAIN_HANDOFF_PINS]
    text = (REPO / 'hyak/env/setup_train_env.sh').read_text()
    script = re.search(r'^PINS=\(([^)]*)\)', text, re.M).group(1).split()
    script += re.search(r'^HANDOFF_PINS=\(([^)]*)\)', text, re.M).group(1).split()
    assert script == [f'{n}=={v}' for n, v in pins]
    assert C.requirement_problems(pins, {'torch': '2.15.0+cu132', **{n: v for n, v in pins if n != 'torch'}}) == []
    problems = C.requirement_problems(pins, {n: v for n, v in pins if n != 'huggingface_hub'} | {'numpy': '2.3.4'})
    assert len(problems) == 2 and any('huggingface_hub' in p for p in problems)


# --- fetch_inputs ---------------------------------------------------------------------------------------------
def build_bundle(base: Path) -> tuple:
    """A small bundle in the real layout: (archive, reference_dir).  1,000 tokenizers, 23 buckets."""
    tree = base / 'bundle-tree'
    files = {}
    rows = []
    for i in range(1000):
        data = json.dumps({'tokenizer': i}).encode()
        files[f'input/pool/tokenizers/{i:04d}/tokenizer.json'] = data
        rows.append({'config': {'index': i}, 'tokenizer_bytes': len(data),
                     'tokenizer_sha256': hashlib.sha256(data).hexdigest()})
    files['input/pool/full-results.json'] = json.dumps({'completed': 1000, 'tokenizers': rows}).encode()
    for b in range(23):
        files[f'input/corpus/bucket{b:02d}/train.jsonl.gz'] = f'corpus {b}'.encode()
    files['input/recipe.json'] = b'{}'
    for name, data in files.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    entries = {name: {'bytes': len(d), 'sha256': hashlib.sha256(d).hexdigest()} for name, d in sorted(files.items())}
    total = sum(e['bytes'] for e in entries.values())
    manifest = {'corpus_buckets': 23, 'tokenizers': 1000, 'total_bytes': total, 'files': entries}
    (tree / C.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    archive = base / 'source' / 'anytok-training-inputs-v1.tar.gz'
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(tree / 'input', arcname='input')
        tar.add(tree / C.MANIFEST_NAME, arcname=C.MANIFEST_NAME)
    digest, size = sha(archive), archive.stat().st_size
    (archive.parent / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n')
    receipt = {'archive': archive.name, 'bytes': size, 'input_bytes': total, 'input_files': len(files),
               'manifest': C.MANIFEST_NAME, 'sha256': digest, 'tokenizers': 1000}
    (archive.parent / C.RECEIPT_NAME).write_text(json.dumps(receipt))
    ref = base / 'reference'
    ref.mkdir()
    (ref / C.RECEIPT_NAME).write_text(json.dumps(receipt))
    (ref / C.REFERENCE_NAME).write_text(json.dumps({
        'archive_url': f'https://huggingface.co/datasets/x/y/resolve/abc/{archive.name}', 'bytes': size,
        'input_files': len(files), 'repo_id': 'x/y', 'revision': 'abc', 'sha256': digest}))
    shutil.copyfile(tree / C.MANIFEST_NAME, ref / C.MANIFEST_NAME)
    shutil.copyfile(REPO / C.DOCS_REL / C.REQUIREMENTS_NAME, ref / C.REQUIREMENTS_NAME)
    return archive, ref


@pytest.fixture
def bundle(tmp_path, tmp_config):
    archive, ref = build_bundle(tmp_path / 'b')
    return tmp_config, archive, ref


def fetch(*args):
    return F.main([*map(str, args)])


def test_fetch_extracts_verifies_and_is_a_noop_afterwards(bundle, capsys):
    t, archive, ref = bundle
    assert fetch('--from-dir', archive.parent, '--reference-dir', ref) == 0
    out = capsys.readouterr().out
    assert 'OK:' in out and 'verified_files' not in out
    assert (t.paths.input / 'pool' / 'full-results.json').is_file() and t.paths.bundle_manifest.is_file()
    receipt = json.loads(t.paths.handoff_receipt.read_text())
    pin = json.loads((ref / C.RECEIPT_NAME).read_text())
    assert receipt['verification'] == {'verified_files': 1025, 'tokenizers': 1000, 'corpus_buckets': 23,
                                       'input_bytes': pin['input_bytes']}
    assert (t.paths.handoff_dir / archive.name).is_file()          # kept by default
    assert not (t.paths.root / '.handoff-extract').exists()
    before = t.paths.handoff_receipt.stat().st_mtime_ns
    assert fetch('--from-dir', '/nonexistent', '--reference-dir', ref, '--remove-archive') == 0
    out = capsys.readouterr().out
    assert 'nothing to do' in out and t.paths.handoff_receipt.stat().st_mtime_ns == before
    assert not (t.paths.handoff_dir / archive.name).exists()       # --remove-archive on a verified tree


@pytest.mark.parametrize('damage', ['size', 'sha', 'sha_file', 'receipt'])
def test_fetch_refuses_a_wrong_download(bundle, capsys, damage):
    t, archive, ref = bundle
    reference = json.loads((ref / C.REFERENCE_NAME).read_text())
    if damage == 'size':
        reference['bytes'] += 1
    elif damage == 'sha':
        reference['sha256'] = '0' * 64
    if damage in ('size', 'sha'):
        receipt = json.loads((ref / C.RECEIPT_NAME).read_text())
        receipt.update(bytes=reference['bytes'], sha256=reference['sha256'])
        (ref / C.RECEIPT_NAME).write_text(json.dumps(receipt))
        (ref / C.REFERENCE_NAME).write_text(json.dumps(reference))
    elif damage == 'sha_file':
        (archive.parent / (archive.name + '.sha256')).write_text(f'{"1" * 64}  {archive.name}\n')
    else:
        receipt = json.loads((archive.parent / C.RECEIPT_NAME).read_text())
        receipt['input_files'] += 1
        (archive.parent / C.RECEIPT_NAME).write_text(json.dumps(receipt))
    assert fetch('--from-dir', archive.parent, '--reference-dir', ref) == 4
    assert not t.paths.input.exists() and not t.paths.handoff_receipt.exists()
    assert 'remedy' in capsys.readouterr().out


def test_fetch_refuses_an_unverified_tree_unless_forced(bundle, capsys):
    t, archive, ref = bundle
    (t.paths.input / 'model').mkdir(parents=True)
    (t.paths.input / 'model' / 'config.json').write_text('{}')
    assert fetch('--from-dir', archive.parent, '--reference-dir', ref) == 2
    assert 'not the verified bundle' in capsys.readouterr().out and (t.paths.input / 'model').is_dir()
    assert fetch('--from-dir', archive.parent, '--reference-dir', ref, '--force') == 0
    (aside,) = [p for p in t.paths.root.iterdir() if p.name.startswith('input.replaced-')]
    assert (aside / 'model' / 'config.json').is_file()               # moved, never deleted
    assert t.paths.handoff_receipt.is_file()


def test_fetch_rejects_members_outside_input(bundle, tmp_path, capsys):
    t, archive, ref = bundle
    bad = tmp_path / 'bad' / archive.name
    bad.parent.mkdir()
    with tarfile.open(archive) as src, tarfile.open(bad, 'w:gz') as dst:
        for member in src.getmembers():
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        info = tarfile.TarInfo('evil.txt')
        info.size = 3
        dst.addfile(info, io.BytesIO(b'bad'))
    for name in (archive.name + '.sha256', C.RECEIPT_NAME):
        shutil.copyfile(archive.parent / name, bad.parent / name)
    digest, size = sha(bad), bad.stat().st_size
    (bad.parent / (bad.name + '.sha256')).write_text(f'{digest}  {bad.name}\n')
    for path in (ref / C.RECEIPT_NAME, bad.parent / C.RECEIPT_NAME, ref / C.REFERENCE_NAME):
        data = json.loads(path.read_text())
        data.update(sha256=digest, bytes=size)
        path.write_text(json.dumps(data))
    assert fetch('--from-dir', bad.parent, '--reference-dir', ref) == 4
    assert 'unexpected member' in capsys.readouterr().out and not t.paths.input.exists()
    assert not (t.paths.root / 'evil.txt').exists()


def test_fetch_downloads_with_huggingface_hub_and_no_token(bundle, monkeypatch, capsys):
    t, archive, ref = bundle
    calls = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        shutil.copyfile(archive.parent / kwargs['filename'], Path(kwargs['local_dir']) / kwargs['filename'])
        return str(Path(kwargs['local_dir']) / kwargs['filename'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=hf_hub_download))
    assert fetch('--reference-dir', ref, '--download-only') == 0
    assert [c['filename'] for c in calls] == [archive.name, archive.name + '.sha256', C.RECEIPT_NAME]
    assert all(c['token'] is False and c['revision'] == 'abc' and c['repo_type'] == 'dataset' for c in calls)
    assert not t.paths.input.exists()
    calls.clear()
    assert fetch('--reference-dir', ref) == 0                   # the complete archive is not downloaded again
    assert [c['filename'] for c in calls] == [archive.name + '.sha256', C.RECEIPT_NAME]
    assert t.paths.handoff_receipt.is_file()


@needs_bash
def test_fetch_inputs_sh_dry_run_and_help(bundle, tmp_path):
    t, archive, ref = bundle
    env = {**os.environ, 'HYAK_PYTHON': sys.executable}
    script = REPO / 'hyak/handoff/fetch_inputs.sh'
    out = subprocess.run([BASH, str(script), '--help'], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'debug job' in out.stdout and '--remove-archive' in out.stdout
    out = subprocess.run([BASH, str(script), '--dry-run'], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert 'c2421b0c046967a7eb05fa628546c5058f69d035' in out.stdout and '9,314,144,730 bytes' in out.stdout
    assert not t.paths.handoff_dir.exists() and not t.paths.input.exists()
    out = subprocess.run([BASH, str(script), '--from-dir', str(archive.parent), '--reference-dir', str(ref)],
                         env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stdout + out.stderr
    assert t.paths.handoff_receipt.is_file()


def test_real_pin_is_self_consistent():
    ref = C.load_reference(REPO / C.DOCS_REL)
    assert ref['archive'] == 'anytok-training-inputs-v1.tar.gz'
    assert ref['reference']['bytes'] == 9_314_144_730 and ref['reference']['input_files'] == 1062
    manifest = json.loads(ref['manifest_path'].read_text())
    assert len(manifest['files']) == 1062 and manifest['tokenizers'] == 1000 and manifest['corpus_buckets'] == 23
    assert manifest['total_bytes'] == ref['receipt']['input_bytes'] == 13_445_041_321
    # the bundle's plan is the repository's plan (the dashboard's learning panel reads that one)
    assert manifest['files']['input/pool/plan.json']['sha256'] == sha(REPO / 'plans/tokenizer-pool-1000-v1.json')


# --- registry and submit.sh -------------------------------------------------------------------------------------
def test_registry_entries(tmp_config):
    from hyak.tools import register_dashboard as D
    entries = {e['id']: e for e in D.build_entries(tmp_config.cfg)}
    train, smoke = entries['hyak-raw-bpb-training'], entries['hyak-raw-bpb-smoke']
    assert list(entries)[:2] == ['hyak-raw-bpb-training', 'hyak-raw-bpb-smoke']
    assert train['status'] == 'modal_results/hyak_tillicum/train-rawbpb/run/status.json'
    assert (train['adapter'], train['completed_field'], train['total'], train['unit']) == ('progress_json', 'step', 20000, 'steps')
    assert train['logs'][0] == 'modal_results/hyak_tillicum/train-rawbpb/run/events.jsonl'
    assert train['category'] == smoke['category'] == 'Training'
    assert train['platform'] == 'Hyak Tillicum · H200' and train['completion_stage'] == 'complete'
    assert train['learning']['logs'] == 'modal_results/hyak_tillicum/train-rawbpb/*-train.log'
    assert train['learning']['snapshots'] == 'modal_results/hyak_tillicum/train-rawbpb/policy-monitor.json'
    assert train['learning']['baseline_required'] is False and 'not overlaid' in train['learning']['protocol_note']
    assert (smoke['status'], smoke['total']) == ('modal_results/hyak_tillicum/smoke-rawbpb/run/status.json', 3)
    assert all(not Path(p).is_absolute() for e in (train, smoke) for p in e['logs'])


def test_registry_entry_reads_the_launcher_status_through_the_link(tmp_path, tmp_config):
    from hyak.tools import register_dashboard as D
    from monitoring.state import Reader, snapshot
    entry = next(e for e in D.build_entries(tmp_config.cfg) if e['id'] == 'hyak-raw-bpb-training')
    root = tmp_path / 'dash'
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'status.json').write_text(json.dumps({'stage': 'train', 'step': 1234, 'total': 20000, 'workers': [
        {'id': '9', 'profile': 'tillicum/gpu-h200', 'platform': 'Hyak Tillicum · H200', 'state': 'train'}]}))
    link = root / 'modal_results/hyak_tillicum/train-rawbpb/run'
    link.parent.mkdir(parents=True)
    link.symlink_to(output)
    view = snapshot(Reader(root), entry)
    assert (view['completed'], view['total'], view['stage']) == (1234, 20000, 'train')
    assert view['workers'][0]['profile'] == 'tillicum/gpu-h200'


@pytest.fixture
def fake_slurm(tmp_path, make_tmp_config):
    t = make_tmp_config()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'sbatch.log'
    (bindir / 'sbatch').write_text(f'#!/bin/bash\necho "$@" >> {shlex.quote(str(log))}\necho 777\n')
    (bindir / 'squeue').write_text('#!/bin/sh\nexit 0\n')
    for tool in ('sbatch', 'squeue'):
        (bindir / tool).chmod(0o755)
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_PYTHON': sys.executable}
    return env, log, t


def submit(env, *args):
    return subprocess.run([BASH, str(SUBMIT), *args], env=env, capture_output=True, text=True, timeout=60)


@needs_bash
def test_submit_dry_runs_of_the_new_stages(fake_slurm):
    env, log, t = fake_slurm
    out = submit(env, 'smoke-rawbpb', '--dry-run')
    assert out.returncode == 0, out.stderr
    line = next(l for l in out.stdout.splitlines() if l.startswith('sbatch '))
    assert '--qos=debug' in line and '--time=01:00:00' in line and '--signal=B:TERM@600' in line
    assert 'singleton' not in line and line.endswith('12_smoke_rawbpb.sbatch ')
    assert 'inputs-verified.json has not been written yet' in out.stderr
    out = submit(env, 'train-rawbpb', '--count', '3', '--dry-run')
    assert out.returncode == 0, out.stderr
    line = next(l for l in out.stdout.splitlines() if l.startswith('sbatch '))
    assert '--qos=normal' in line and '--time=24:00:00' in line and '--signal=B:TERM@1200' in line
    assert '--dependency=singleton' in line and '--job-name=anytok-train-rawbpb' in line
    assert line.endswith('13_train_rawbpb.sbatch ')
    assert submit(env, 'smoke-rawbpb', '--count', '2', '--dry-run').returncode == 1
    assert submit(env, 'train-rawbpb', '--env', 'HYAK_TIME_TRAIN_RAWBPB=00:30:00', '--dry-run').returncode == 1
    assert not log.exists()


@needs_bash
def test_submit_refuses_without_the_verified_bundle(fake_slurm):
    env, log, t = fake_slurm
    for stage in ('smoke-rawbpb', 'train-rawbpb'):
        out = submit(env, stage)
        assert out.returncode == 1 and 'fetch_inputs.sh' in out.stderr
    assert not log.exists()
    t.paths.handoff_dir.mkdir(parents=True)
    t.paths.handoff_receipt.write_text('{}')
    out = submit(env, 'train-rawbpb', '--count', '3')
    assert out.returncode == 0, out.stderr
    assert len(log.read_text().splitlines()) == 3


def test_output_config_validation(make_tmp_config, tmp_path):
    from hyak.common.config import ConfigError
    t = make_tmp_config()
    assert C.outputs(t.cfg) == (t.paths.runs / 'raw-bpb-20k', t.paths.runs / 'raw-bpb-smoke')
    t = make_tmp_config(HYAK_RAWBPB_OUTPUT=str(tmp_path / 'proj' / 'run'))
    assert C.outputs(t.cfg)[0] == tmp_path / 'proj' / 'run'
    for bad in ('relative/dir', str(tmp_path / 'x') + '/', str(REPO / 'out')):
        with pytest.raises(ConfigError):
            make_tmp_config(HYAK_RAWBPB_OUTPUT=bad)
    with pytest.raises(ConfigError, match='different'):
        make_tmp_config(HYAK_RAWBPB_OUTPUT=str(tmp_path / 'same'), HYAK_RAWBPB_SMOKE_OUTPUT=str(tmp_path / 'same'))


# --- check_env --handoff ----------------------------------------------------------------------------------------
def test_check_env_handoff_mode(tmp_config, capsys):
    from hyak.env import check_env
    code = check_env.main(['--handoff', '--no-status', '--only',
                           'handoff_requirements,handoff_inputs,code_stage,model_files,trainer_import_rl'])
    out = capsys.readouterr().out
    assert '[PASS] handoff_requirements: 7 packages equal docs/raw-bpb-handoff/requirements.txt' in out
    assert '[FAIL] handoff_inputs' in out and 'fetch_inputs.sh' in out      # no bundle in this temporary root
    for name in ('code_stage', 'model_files', 'trainer_import_rl'):
        assert f'[SKIP] {name}: not used by the raw-BPB handoff run' in out
    assert code == 4
    code = check_env.main(['--no-status', '--only', 'handoff_requirements'])
    assert code == 0 and 'only with --handoff' in capsys.readouterr().out
