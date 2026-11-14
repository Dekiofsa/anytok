"""hyak/slurm/lib.sh and hyak/slurm/submit.sh (run with the local bash; bash 3.2 compatible)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

from hyak.common import config as C
from hyak.common.paths import SBATCH_FILES, STAGE_RUN, STAGES
from hyak.common.slurm import parse_time_limit, sbatch_resources

BASH = shutil.which('bash')
pytestmark = pytest.mark.skipif(BASH is None, reason='bash not available')
REPO = Path(__file__).resolve().parents[2]
LIB = REPO / 'hyak/slurm/lib.sh'
SUBMIT = REPO / 'hyak/slurm/submit.sh'


def run_bash(script, env=None, timeout=30, cwd=None):
    return subprocess.run([BASH, '-c', script], env=env if env is not None else dict(os.environ),
                          capture_output=True, text=True, timeout=timeout, cwd=cwd)


def lib_script(body):
    return f'set -euo pipefail\nsource {shlex.quote(str(LIB))}\n{body}\n'


def test_syntax():
    for path in (LIB, SUBMIT):
        result = subprocess.run([BASH, '-n', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert os.access(SUBMIT, os.X_OK)


def test_tables_match_python():
    out = run_bash(lib_script('echo "$HYAK_STAGES"; echo "$HYAK_STAGE_OVERRIDE_FLAGS"; echo "$HYAK_MONITOR_REL"'))
    stages, flags, rel = out.stdout.splitlines()
    assert stages.split() == list(STAGES)
    assert tuple(flags.split()) == C.STAGE_OVERRIDE_FLAGS
    assert rel == 'modal_results/hyak_tillicum'


def test_repo_from_lib_location():
    env = {k: v for k, v in os.environ.items() if k != 'HYAK_REPO'}
    out = run_bash(lib_script('echo "$HYAK_REPO"'), env=env)
    assert Path(out.stdout.strip()) == REPO.resolve()


# --- configuration parity with hyak.common.config ------------------------------------------------
def bash_config(env):
    return run_bash(lib_script('hyak_load_config\nhyak_print_config'), env=env)


@pytest.mark.parametrize('key', ['ANYTOK_ROOT', 'ANYTOK_RUNS'])
@pytest.mark.parametrize('where', ['', '/runs', '/hyak/x'])
def test_storage_inside_the_checkout_is_refused_by_both_validators(tmp_path, key, where):
    """readme-v2: environments, corpus and checkpoints must not land in the git working tree."""
    other = {'ANYTOK_ROOT': '/gpfs/projects/grp/anytok-data', 'ANYTOK_RUNS': '/gpfs/scrubbed/grp/runs'}
    other[key] = str(REPO.resolve()) + where
    file = tmp_path / 'c.env'
    file.write_text(''.join(f'{k}={v}\n' for k, v in other.items()))
    env = {**os.environ, 'HYAK_CONFIG': str(file), 'HYAK_REPO': str(REPO.resolve())}
    out = bash_config(env)
    assert out.returncode == 1 and 'inside the repository checkout' in out.stderr, out.stderr
    with pytest.raises(C.ConfigError, match='inside the repository checkout'):
        C.load_config(environ=env)


def python_config(env):
    cfg = C.load_config(environ=env)
    return '\n'.join(f'{k}={v}' for k, v in sorted(C.redacted(cfg).items())) + '\n'


CONFIG_TEXT = '''# test config
ANYTOK_ROOT=/gpfs/projects/grp/anytok   # comment
export ANYTOK_RUNS="/gpfs/scrubbed/grp/runs"
HYAK_PLATFORM='Hyak Tillicum · H200 (test)'
HYAK_ACCOUNT=
HYAK_PARTITION_POOL=gpu-h200-mig
HYAK_TIME_POOL="12:00:00"
HYAK_PLATFORM_CORPUS="klone · CPU"
HF_TOKEN=hf_example_token_value
HYAK_MODULES="conda gcc/13.4.0"
'''


@pytest.mark.parametrize('env_extra', [
    {},
    {'HYAK_CPUS': '4', 'HYAK_MEM': '', 'HYAK_GRES_CORPUS': 'gpu:2', 'HYAK_TIME_POOL': '06:00:00'},
    {'ANYTOK_ROOT': '/elsewhere/root', 'HF_TOKEN': ''},
])
def test_config_parity(tmp_path, env_extra):
    file = tmp_path / 'c.env'
    file.write_text(CONFIG_TEXT, encoding='utf-8')
    env = {**os.environ, 'HYAK_CONFIG': str(file), **env_extra}
    out = bash_config(env)
    assert out.returncode == 0, out.stderr
    assert out.stdout == python_config(env)
    assert 'hf_example_token_value' not in out.stdout


def test_config_exported_or_not(tmp_config):
    out = run_bash(lib_script('hyak_load_config\nsh -c \'printf "%s|%s" "$ANYTOK_ROOT" "$HYAK_CONFIG_FILE"\''))
    assert out.stdout == f'{tmp_config.paths.root}|{tmp_config.config_file}'
    out = run_bash(lib_script('hyak_load_config --no-export\necho "$ANYTOK_ROOT"\n'
                              'sh -c \'printf "child=%s" "${ANYTOK_ROOT-unset}"\''))
    assert out.stdout == f'{tmp_config.paths.root}\nchild=unset'


@pytest.mark.parametrize('text,needle', [
    ('HYAK_CPUS=8\n', 'still contain CHANGE_ME'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_PARTITON=x\n', 'unknown key'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_CPUS = 8\n', 'expected KEY=value'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_MEM=1 60G\n', 'must be unquoted'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nANYTOK_RUNS=/c\n', 'duplicate key'),
    ('ANYTOK_ROOT=/a\r\nANYTOK_RUNS=/b\n', 'carriage return'),
    ('ANYTOK_ROOT=rel\nANYTOK_RUNS=/b\n', 'absolute path'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_TIME=soon\n', 'HYAK_TIME'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_TIME_POOL=0\n', 'HYAK_TIME_POOL'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_SEGMENT_MARGIN_MINUTES=x\n', 'HYAK_SEGMENT_MARGIN_MINUTES'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_CPUS_POOL=0\n', 'HYAK_CPUS_POOL'),
    ('ANYTOK_ROOT=/a/\nANYTOK_RUNS=/b\n', 'normalised'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b/\n', 'normalised'),
    ('ANYTOK_ROOT=/a//b\nANYTOK_RUNS=/b\n', 'normalised'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=//b\n', 'normalised'),
    ('ANYTOK_ROOT=/a/./b\nANYTOK_RUNS=/b\n', 'normalised'),
    ('ANYTOK_ROOT=/a/../b\nANYTOK_RUNS=/b\n', 'normalised'),
    ('ANYTOK_ROOT=/a/b/..\nANYTOK_RUNS=/b\n', 'normalised'),
    ('ANYTOK_ROOT=/a\nANYTOK_RUNS=/b\nHYAK_HN_CHUNK_ROWS=4095\n', 'HYAK_HN_CHUNK_ROWS'),
])
def test_config_errors_agree(tmp_path, text, needle):
    file = tmp_path / 'c.env'
    file.write_bytes(text.encode())
    env = {**os.environ, 'HYAK_CONFIG': str(file)}
    out = bash_config(env)
    assert out.returncode == 1 and needle in out.stderr, out.stderr
    with pytest.raises(C.ConfigError) as err:
        C.load_config(environ=env)
    assert needle in str(err.value)


def test_missing_config_file(tmp_path):
    out = bash_config({**os.environ, 'HYAK_CONFIG': str(tmp_path / 'nope.env')})
    assert out.returncode == 1 and 'not found. Remedy: cp' in out.stderr


@pytest.mark.parametrize('text', ['90', '45:30', '24:00:00', '1-0', '2-03', '1-02:03', '1-02:03:04', '00:59',
                                  'UNLIMITED', '08:09:07'])
def test_parse_time_parity(text):
    out = run_bash(lib_script(f'hyak_parse_time {shlex.quote(text)}'))
    expected = parse_time_limit(text)
    assert out.stdout.strip() == ('inf' if expected == float('inf') else str(expected))


@pytest.mark.parametrize('text', ['NOT_SET', '1:2:3:4', 'abc', '', '1.5'])
def test_parse_time_invalid(text):
    out = run_bash(lib_script(f'if hyak_parse_time {shlex.quote(text)}; then echo ok; else echo bad; fi'))
    assert out.stdout.strip() == 'bad'


def test_stage_value_and_suffix(make_tmp_config):
    make_tmp_config(HYAK_PLATFORM_POOL='klone · CPU')
    out = run_bash(lib_script('hyak_load_config\nhyak_stage_value HYAK_PLATFORM pool\n'
                              'hyak_stage_value HYAK_PLATFORM corpus\nhyak_stage_suffix eval-prepare'))
    assert out.stdout.splitlines() == ['klone · CPU', 'Hyak Tillicum · H200', 'EVAL_PREPARE']
    bad = run_bash(lib_script('hyak_stage_suffix train'))
    assert bad.returncode == 1 and "Unknown stage 'train'" in bad.stderr


# --- monitor dirs, modules, activation, status helpers ---------------------------------------------------
def test_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('HYAK_MONITOR_ROOT', str(tmp_path / 'mon'))
    env = {**os.environ, 'SLURM_JOB_ID': '77'}
    out = run_bash(lib_script('hyak_monitor_dir pool; d=$(hyak_log_dir pool); echo "$d"'), env=env)
    assert out.stdout.splitlines() == [str(tmp_path / 'mon/pool')] * 2
    link = tmp_path / 'mon/pool/latest-slurm.out'
    assert link.is_symlink() and os.readlink(link) == 'slurm-77.out'
    env.pop('HYAK_MONITOR_ROOT')
    out = run_bash(lib_script('hyak_monitor_dir train-rl'), env=env)
    assert out.stdout.strip() == f'{REPO.resolve()}/modal_results/hyak_tillicum/train-rl'
    assert run_bash(lib_script('hyak_log_dir nope'), env=env).returncode == 1


def _fake_env(tmp_path, root: Path, which='train'):
    prefix = root / 'env' / which
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'lib').mkdir()
    (prefix / 'bin' / 'python').symlink_to(sys.executable)
    bindir = tmp_path / 'fakebin'
    bindir.mkdir()
    conda = bindir / 'conda'
    conda.write_text('#!/bin/sh\n[ "$1 $2" = "shell.bash hook" ] || exit 3\n'
                     "cat <<'EOF'\nconda() { if [ \"$1\" = activate ]; then export CONDA_PREFIX=\"$2\"; "
                     'export PATH="$2/bin:$PATH"; else return 1; fi; }\nEOF\n')
    conda.chmod(0o755)
    return prefix, bindir


@pytest.mark.parametrize('stage,parallelism', [('train-uniform', '<unset>'), ('eval-rl', 'false'),
                                                ('pool', 'true'), ('init', '<unset>')])
def test_activate(tmp_path, make_tmp_config, stage, parallelism):
    t = make_tmp_config(HYAK_MODULES='')
    prefix, bindir = _fake_env(tmp_path, t.paths.root)
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_STAGE': stage,
           'TOKENIZERS_PARALLELISM': 'stale', 'LD_LIBRARY_PATH': '/old/lib', 'PYTHONPATH': '/junk'}
    out = run_bash(lib_script('hyak_load_config\nhyak_activate train\n'
                              'printf "%s\\n" "$CONDA_PREFIX" "$(command -v python)" "$PYTHONPATH" "$LD_LIBRARY_PATH" '
                              '"$HF_HOME" "$PYTHONDONTWRITEBYTECODE$PYTHONUNBUFFERED$PYTHONNOUSERSITE" "$HYAK_ENV" '
                              '"${TOKENIZERS_PARALLELISM-<unset>}"'), env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [str(prefix), f'{prefix}/bin/python', str(REPO.resolve()),
                                       f'{prefix}/hyak-libpython:/old/lib', str(t.paths.root / 'hf-cache'), '111', 'train',
                                       parallelism]


@pytest.mark.parametrize('stage,kept', [('train-uniform', False), ('train-rl', False), ('smoke-train', False),
                                        ('eval-rl', True), ('probe', True), ('init', True), ('pool', True)])
def test_activate_unsets_the_numerics_overrides_for_the_training_stages_only(tmp_path, make_tmp_config, stage, kept):
    """TORCH_ALLOW_TF32_CUBLAS_OVERRIDE and PYTHONOPTIMIZE would change what the pinned trainer computes and checks."""
    t = make_tmp_config(HYAK_MODULES='')
    prefix, bindir = _fake_env(tmp_path, t.paths.root)
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_STAGE': stage,
           'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1', 'PYTHONOPTIMIZE': '2',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    out = run_bash(lib_script('hyak_load_config\nhyak_activate train\n'
                              'printf "%s\\n" "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-<unset>}" "${PYTHONOPTIMIZE-<unset>}" '
                              '"${PYTORCH_CUDA_ALLOC_CONF-<unset>}"'), env=env)
    assert out.returncode == 0, out.stderr
    tf32, optimize, alloc = out.stdout.splitlines()
    assert alloc == 'expandable_segments:True'                  # the allocator setting is never touched
    assert (tf32, optimize) == (('1', '2') if kept else ('<unset>', '<unset>'))
    if not kept:
        assert 'unsetting TORCH_ALLOW_TF32_CUBLAS_OVERRIDE' in out.stderr and 'unsetting PYTHONOPTIMIZE' in out.stderr


def test_activate_failures(tmp_path, make_tmp_config):
    t = make_tmp_config(HYAK_MODULES='')
    out = run_bash(lib_script('hyak_load_config\nhyak_activate data'))
    assert out.returncode == 1 and 'setup_data_env.sh' in out.stderr
    _fake_env(tmp_path, t.paths.root, 'data')
    env = {**os.environ, 'PATH': '/usr/bin:/bin'}
    out = run_bash(lib_script('hyak_load_config\nhyak_activate data'), env=env)
    assert out.returncode == 1 and 'conda not found' in out.stderr
    assert run_bash(lib_script('hyak_load_config\nhyak_activate base')).returncode == 1


@pytest.mark.parametrize('present,gcc_module,expected', [
    (('cc', 'c++'), None, None), (('gcc', 'g++'), None, None), (('cc', 'g++'), None, None),
    (('cc',), None, 'gcc/13.4.0'), (('c++',), None, 'gcc/13.4.0'), ((), None, 'gcc/13.4.0'), (('gcc',), None, 'gcc/13.4.0'),
    (('g++',), None, 'gcc/13.4.0'), (('cc', 'c++'), 'gcc/12.1.0', 'gcc/12.1.0'), ((), 'gcc/12.1.0', 'gcc/12.1.0')])
def test_gcc_module_is_wanted_when_set_or_when_a_c_or_cxx_compiler_is_missing(tmp_path, present, gcc_module, expected):
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    for name in present:
        (stubs / name).write_text('#!/bin/sh\nexit 0\n')
        (stubs / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != 'HYAK_GCC_MODULE'}
    if gcc_module:
        env['HYAK_GCC_MODULE'] = gcc_module
    # PATH is narrowed after lib.sh is sourced: `command -v` is a builtin, so only the stubs can be found
    out = run_bash(lib_script(f'PATH={shlex.quote(str(stubs))}\nrc=0\nm=$(hyak_gcc_module) || rc=$?\necho "$rc:$m"'), env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ('1:' if expected is None else f'0:{expected}')


def test_modules(tmp_path, make_tmp_config):
    make_tmp_config(HYAK_MODULES='conda gcc/13.4.0')
    log = tmp_path / 'modules.log'
    body = (f'module() {{ printf "%s\\n" "$*" >> {shlex.quote(str(log))}; [ "$2" != broken ]; }}\n'
            'hyak_load_config\nhyak_modules\necho done')
    out = run_bash(lib_script(body))
    assert out.returncode == 0 and out.stdout.strip() == 'done', out.stderr
    assert log.read_text().splitlines() == ['load conda', 'load gcc/13.4.0']
    out = run_bash(lib_script(body.replace('hyak_modules', 'HYAK_MODULES=broken; hyak_modules')))
    assert out.returncode == 1 and 'module load broken failed' in out.stderr


def test_status_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv('HYAK_MONITOR_ROOT', str(tmp_path / 'mon'))
    env = {**os.environ, 'HYAK_PYTHON': sys.executable, 'SLURM_JOB_ID': '31'}
    log = tmp_path / 'job.log'
    log.write_text('ValueError: bad input\n')
    out = run_bash(lib_script(f'hyak_status_start corpus "building"\nhyak_status_event corpus claimed bucket=web-en n=3\n'
                              f'hyak_finalize_status corpus 1 {shlex.quote(str(log))}\necho survived'), env=env)
    assert out.returncode == 0 and out.stdout.strip() == 'survived', out.stderr
    status = json.loads((tmp_path / 'mon/corpus/status.json').read_text())
    assert status['state'] == 'failed' and 'ValueError: bad input' in status['error'] and status['total'] == 23
    rows = [json.loads(x) for x in (tmp_path / 'mon/corpus/events.jsonl').read_text().splitlines()]
    assert [r.get('event') for r in rows] == ['job_start', 'claimed', 'job_end']
    assert rows[1]['bucket'] == 'web-en' and rows[1]['n'] == 3
    broken = {**env, 'HYAK_PYTHON': '/nonexistent/python'}
    out = run_bash(lib_script('hyak_finalize_status corpus 0\necho survived'), env=broken)
    assert out.returncode == 0 and 'survived' in out.stdout and 'WARNING' in out.stderr


# --- hyak_run_forwarding ---------------------------------------------------------------------------------------
CHILD = '''#!/bin/bash
trap 'echo "TERM $(date +%s)" >> "$MARK"; sleep 1; exit ${CHILD_EXIT:-7}' TERM
echo ready > "$READY"
n=0
while [ "$n" -lt 300 ]; do sleep 0.1; n=$((n + 1)); done
exit 99
'''


def forwarding(tmp_path, *, signals=(), child=CHILD, child_exit=None, pre='', post=''):
    script = tmp_path / 'child.sh'
    script.write_text(child)
    script.chmod(0o755)
    ready, mark, result = tmp_path / 'ready', tmp_path / 'mark', tmp_path / 'result'
    for f in (ready, mark, result):
        if f.exists():
            f.unlink()
    harness = lib_script(f'{pre}\nrc=0\nhyak_run_forwarding {shlex.quote(str(script))} || rc=$?\n'
                         f'echo "rc=$rc term=$HYAK_TERM_RECEIVED traps=[$(trap -p TERM)]" > {shlex.quote(str(result))}\n{post}')
    env = {**os.environ, 'READY': str(ready), 'MARK': str(mark)}
    if child_exit is not None:
        env['CHILD_EXIT'] = str(child_exit)
    proc = subprocess.Popen([BASH, '-c', harness], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        deadline = time.time() + 10
        while not ready.exists():
            assert time.time() < deadline and proc.poll() is None, 'child never started'
            time.sleep(0.02)
        time.sleep(0.2)
        for sig, delay in signals:
            os.kill(proc.pid, sig)
            time.sleep(delay)
        out, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    marks = mark.read_text().splitlines() if mark.exists() else []
    return result.read_text().strip(), marks, out, err


def test_forwarding_term_while_waiting(tmp_path):
    result, marks, _, err = forwarding(tmp_path, signals=[(signal.SIGTERM, 0)])
    assert result == 'rc=7 term=1 traps=[]', err
    assert len(marks) == 1 and 'forwarding SIGTERM' in err


def test_forwarding_repeated_term_and_exit_code(tmp_path):
    result, marks, _, err = forwarding(tmp_path, signals=[(signal.SIGTERM, 0.3), (signal.SIGTERM, 0.1)],
                                       child_exit=0)
    assert result == 'rc=0 term=1 traps=[]', err   # a clean stop after TERM is exit 0, not 143
    assert len(marks) >= 1


def test_forwarding_int_becomes_term(tmp_path):
    result, marks, _, err = forwarding(tmp_path, signals=[(signal.SIGINT, 0)], child_exit=5)
    assert result == 'rc=5 term=1 traps=[]', err
    assert len(marks) == 1


def test_forwarding_without_signal(tmp_path):
    child = '#!/bin/bash\necho ready > "$READY"\nsleep 0.5\nexit 3\n'
    result, marks, _, _ = forwarding(tmp_path, child=child)
    assert result == 'rc=3 term=0 traps=[]' and marks == []


def test_forwarding_child_killed_by_untrapped_term(tmp_path):
    child = '#!/bin/bash\necho ready > "$READY"\nexec sleep 30\n'
    result, _, _, err = forwarding(tmp_path, child=child, signals=[(signal.SIGTERM, 0)])
    assert result == 'rc=143 term=1 traps=[]', err


def test_forwarding_restores_persistent_trap_and_flag_loop(tmp_path):
    child = '#!/bin/bash\necho ready > "$READY"\nsleep 0.3\nexit 0\n'
    result, _, _, err = forwarding(tmp_path, child=child, pre='hyak_trap_term')
    assert result == "rc=0 term=0 traps=[trap -- '_hyak_on_signal TERM' SIGTERM]", err


def test_forwarding_requires_command():
    out = run_bash(lib_script('hyak_run_forwarding'))
    assert out.returncode == 1 and 'no command given' in out.stderr


# --- submit.sh -----------------------------------------------------------------------------------------------
def parse_dry_run(stdout):
    (line,) = [l for l in stdout.splitlines() if l.startswith('sbatch ')]
    args = shlex.split(line)
    flags = {}
    for arg in args[1:-1]:
        key, sep, value = arg[2:].partition('=')
        flags[key] = value if sep else True
    return flags, args[-1]


def submit(args, env, script=SUBMIT, timeout=60):
    return subprocess.run([BASH, str(script), *args], env=env, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def sub_env(make_tmp_config, tmp_path):
    def make(**overrides):
        t = make_tmp_config(**overrides)
        bindir = tmp_path / 'bin'
        bindir.mkdir(exist_ok=True)
        marker = tmp_path / 'sbatch-called'
        for tool in ('sbatch', 'squeue'):
            path = bindir / tool
            path.write_text(f'#!/bin/sh\necho "{tool} $*" >> {shlex.quote(str(marker))}\necho 1\n')
            path.chmod(0o755)
        env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}'}
        return env, t, marker
    return make


def expected_flags(cfg, stage, debug, repo, monitor_root, config_file):
    r = sbatch_resources(cfg, stage, debug)
    flags = {'parsable': True}
    for key, flag in (('partition', 'partition'), ('qos', 'qos'), ('account', 'account'), ('gres', 'gres'),
                      ('cpus', 'cpus-per-task'), ('mem', 'mem')):
        if r[key] is not None:
            flags[flag] = r[key]
    flags.update({'time': r['time'], 'job-name': f'anytok-{stage}',
                  'output': f'{monitor_root}/{stage}/slurm-%j.out', 'chdir': str(repo)})
    if r['signal_seconds']:
        flags['signal'] = f'B:TERM@{r["signal_seconds"]}'
    if r['singleton']:
        flags['dependency'] = 'singleton'
    flags['export'] = 'ALL'      # the job's HYAK_* variables are exported by submit.sh itself (no --get-user-env)
    return flags


OVERRIDES = dict(HYAK_ACCOUNT='grp', HYAK_PARTITION_POOL='gpu-h200-mig', HYAK_TIME_POOL='12:00:00',
                 HYAK_GRES_POOL='gpu:1', HYAK_ACCOUNT_POOL='none', HYAK_CPUS_POOL='1', HYAK_MEM_POOL='30G',
                 HYAK_QOS_CORPUS='long', HYAK_TIME_CORPUS='3-00:00:00', HYAK_GRES_WHEEL='none',
                 HYAK_SEGMENT_MARGIN_MINUTES_TRAIN_RL='60', HYAK_QOS_PROBE='normal')


@pytest.mark.parametrize('overrides', [{}, OVERRIDES], ids=['defaults', 'overrides'])
@pytest.mark.parametrize('debug', [False, True], ids=['normal', 'debug'])
def test_dry_run_every_stage(sub_env, overrides, debug):
    env, t, marker = sub_env(**overrides)
    for stage in STAGES:
        out = submit([stage, '--dry-run'] + (['--debug'] if debug else []), env)
        if debug and stage in ('train-uniform', 'train-rl'):    # a training job needs margin + 20 minutes
            assert out.returncode != 0 and 'Remedy' in out.stderr, (stage, out.stderr)
            continue
        assert out.returncode == 0, (stage, out.stderr)
        flags, file = parse_dry_run(out.stdout)
        assert flags == expected_flags(t.cfg, stage, debug, REPO.resolve(), t.base / 'monitor', t.config_file), stage
        assert file == str(REPO.resolve() / 'hyak/slurm' / SBATCH_FILES[stage])
        assert ('dependency' in flags) == (stage in ('train-uniform', 'train-rl'))
        if debug:
            assert (flags['qos'], flags['time']) == ('debug', '01:00:00')
    assert not marker.exists(), 'dry run must not call sbatch or squeue'
    assert not (t.base / 'monitor').exists(), 'dry run must not create monitor directories'


def test_dry_run_explicit_values(sub_env):
    env, t, _ = sub_env(**OVERRIDES)
    flags, _ = parse_dry_run(submit(['train-uniform', '--dry-run'], env).stdout)
    assert flags['partition'] == 'gpu-h200' and flags['qos'] == 'normal' and flags['gres'] == 'gpu:1'
    assert flags['cpus-per-task'] == '8' and flags.get('mem') == (t.cfg['HYAK_MEM'] or None) \
        and flags['time'] == '24:00:00'
    assert flags['account'] == 'grp' and flags['signal'] == 'B:TERM@2700' and flags['dependency'] == 'singleton'
    assert flags['job-name'] == 'anytok-train-uniform' and flags['export'] == 'ALL'
    assert 'HYAK_RUN=uniform' in submit(['train-uniform', '--dry-run'], env).stdout
    flags, _ = parse_dry_run(submit(['pool', '--dry-run'], env).stdout)
    assert (flags['partition'], flags['time'], flags['cpus-per-task'], flags['mem']) == ('gpu-h200-mig', '12:00:00', '1', '30G')
    assert 'account' not in flags and 'dependency' not in flags
    assert 'HYAK_RUN' not in submit(['pool', '--dry-run'], env).stdout
    flags, _ = parse_dry_run(submit(['pool', '--dry-run', '--debug'], env).stdout)
    assert (flags['qos'], flags['time'], flags['signal']) == ('debug', '01:00:00', 'B:TERM@900')
    flags, _ = parse_dry_run(submit(['wheel', '--dry-run'], env).stdout)
    assert 'gres' not in flags
    out = submit(['train-rl', '--dry-run'], env)
    flags, _ = parse_dry_run(out.stdout)
    assert flags['signal'] == 'B:TERM@3600' and 'HYAK_RUN=rl' in out.stdout
    flags, _ = parse_dry_run(submit(['probe', '--dry-run'], env).stdout)
    assert (flags['qos'], flags['time']) == ('normal', '01:00:00')
    flags, _ = parse_dry_run(submit(['corpus', '--dry-run'], {**env, 'HYAK_TIME_CORPUS': '2-00:00:00'}).stdout)
    assert (flags['qos'], flags['time'], flags['signal']) == ('long', '2-00:00:00', 'B:TERM@2700')
    env_defaults, _, _ = sub_env()
    flags, _ = parse_dry_run(submit(['check-env', '--dry-run'], env_defaults).stdout)
    assert 'account' not in flags and (flags['qos'], flags['time']) == ('debug', '01:00:00')


def test_dry_run_options(sub_env):
    env, _, _ = sub_env()
    out = submit(['train-rl', '--dry-run', '--after', '11:12', '--count', '4', '--env', 'X_Y=1',
                  '--', '--mail-type=END'], env)
    flags, _ = parse_dry_run(out.stdout)
    assert flags['dependency'] == 'singleton,afterok:11:12' and flags['mail-type'] == 'END'
    assert flags['kill-on-invalid-dep'] == 'yes' and flags['export'] == 'ALL' and ' X_Y=1' in out.stdout
    flags, _ = parse_dry_run(submit(['init', '--dry-run', '--after', '5'], env).stdout)
    assert flags['dependency'] == 'afterok:5'
    assert 'HYAK_RUN=rl' in submit(['smoke-train', '--dry-run', '--run', 'rl'], env).stdout
    flags, _ = parse_dry_run(submit(['pool', '--dry-run'], {**env, 'HYAK_SEGMENT_MARGIN_MINUTES': '0'}).stdout)
    assert 'signal' not in flags
    refused = submit(['train-uniform', '--dry-run'], {**env, 'HYAK_SEGMENT_MARGIN_MINUTES': '0'})
    assert refused.returncode != 0 and 'HYAK_SEGMENT_MARGIN_MINUTES' in refused.stderr   # training needs a signal
    for bad, needle in [(['init', '--dry-run', '--count', '2'], 'one-shot'),
                        (['eval-score', '--dry-run'], 'eval-uniform or eval-rl'),
                        (['train', '--dry-run'], 'train-uniform or train-rl'),
                        (['bogus', '--dry-run'], 'Unknown stage'),
                        (['pool', '--dry-run', '--count', '0'], 'positive integer'),
                        (['train-rl', '--dry-run', '--run', 'uniform'], 'contradicts'),
                        (['pool', '--dry-run', '--', '--dependency=afterok:1'], '--after'),
                        (['pool', '--dry-run', '--', '--job-name=x'], 'set by submit.sh'),
                        (['pool', '--dry-run', '--env', 'HYAK_STAGE=x'], 'set by submit.sh'),
                        (['pool', '--dry-run', '--after', 'abc'], 'JOBID'),
                        (['pool', '--frobnicate'], 'Unknown option')]:
        out = submit(bad, env)
        assert out.returncode != 0 and needle in out.stderr, (bad, out.stderr)
    assert submit(['--help'], env).stdout.startswith('Usage:')


def test_dry_run_rejects_bad_config(sub_env, tmp_path):
    env, t, _ = sub_env()
    (tmp_path / 'bad.env').write_text('ANYTOK_ROOT=/gpfs/projects/CHANGE_ME/x\nANYTOK_RUNS=/r\n')
    out = submit(['pool', '--dry-run'], {**env, 'HYAK_CONFIG': str(tmp_path / 'bad.env')})
    assert out.returncode == 1 and 'CHANGE_ME' in out.stderr
    (tmp_path / 'bad2.env').write_text('ANYTOK_ROOT=/a\nANYTOK_RUNS=/r\nHYAK_MEM=160GB\n')
    out = submit(['pool', '--dry-run'], {**env, 'HYAK_CONFIG': str(tmp_path / 'bad2.env'), 'HYAK_PYTHON': sys.executable})
    assert out.returncode == 1 and 'Configuration check failed' in out.stderr


def test_a_crash_of_the_config_checker_is_not_reported_as_a_configuration_error(sub_env, tmp_path):
    """hyak.common.config --check exits 2 for a bad configuration; any other status means the checker itself broke."""
    env, t, _ = sub_env()
    fake = tmp_path / 'fakepython'
    fake.write_text('#!/bin/sh\ncase "$1" in -c) exit 0 ;; *) echo "Traceback: boom" >&2; exit 1 ;; esac\n')
    fake.chmod(0o755)
    out = submit(['pool', '--dry-run'], {**env, 'HYAK_PYTHON': str(fake)})
    assert out.returncode == 1 and 'Configuration check failed' in out.stderr
    assert 'exited 1; 2 is a configuration error, so the checker itself broke' in out.stderr and 'boom' in out.stderr
    (tmp_path / 'bad3.env').write_text('ANYTOK_ROOT=/a\nANYTOK_RUNS=/r\nHYAK_MEM=160GB\n')
    out = submit(['pool', '--dry-run'], {**env, 'HYAK_CONFIG': str(tmp_path / 'bad3.env'), 'HYAK_PYTHON': sys.executable})
    assert out.returncode == 1 and 'Configuration check failed:' in out.stderr and 'checker itself' not in out.stderr


def fake_repo(tmp_path, with_sbatch=True):
    root = tmp_path / 'fakerepo'
    for rel in ('hyak/__init__.py', 'hyak/config.env.example', 'hyak/slurm/lib.sh', 'hyak/slurm/submit.sh',
                *[f'hyak/common/{p.name}' for p in (REPO / 'hyak/common').glob('*.py')]):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    if with_sbatch:
        for name in set(SBATCH_FILES.values()):
            (root / 'hyak/slurm' / name).write_text('#!/bin/bash\necho fake\n')
    return root.resolve()


def fake_tools(tmp_path, squeue_output=''):
    bindir = tmp_path / 'tools'
    bindir.mkdir(exist_ok=True)
    log = tmp_path / 'sbatch.log'
    (bindir / 'sbatch').write_text(
        '#!/bin/bash\n{ for a in "$@"; do printf "ARG %s\\n" "$a"; done\n'
        'printf "ENV %s|%s|%s|%s\\n" "${HYAK_STAGE-unset}" "${HYAK_RUN-unset}" "${HYAK_REPO-unset}" "${X_Y-unset}"\n'
        f'echo END; }} >> {shlex.quote(str(log))}\n'
        f'echo "$((1000 + $(grep -c ^END {shlex.quote(str(log))})));tillicum"\n')
    (bindir / 'squeue').write_text(f'#!/bin/sh\nprintf "%s" {shlex.quote(squeue_output)}\n')
    for tool in ('sbatch', 'squeue'):
        (bindir / tool).chmod(0o755)
    return bindir, log


def calls(log):
    out, current = [], None
    for line in log.read_text().splitlines():
        if line == 'END':
            out.append(current)
            current = None
        else:
            current = current or {'args': [], 'env': None}
            if line.startswith('ARG '):
                current['args'].append(line[4:])
            else:
                current['env'] = line[4:]
    return out


def test_submission_with_fake_sbatch(tmp_path, tmp_config):
    repo = fake_repo(tmp_path)
    bindir, log = fake_tools(tmp_path)
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_PYTHON': sys.executable,
           'HYAK_RUN': 'stale', 'HYAK_STAGE': 'stale', 'X_Y': 'from-user'}
    out = submit(['train-uniform', '--count', '3'], env, script=repo / 'hyak/slurm/submit.sh')
    assert out.returncode == 0, out.stderr
    assert [l for l in out.stdout.splitlines() if l.startswith('submitted')] == [
        f'submitted anytok-train-uniform job {1000 + n} ({n}/3)' for n in (1, 2, 3)]
    made = calls(log)
    assert len(made) == 3 and made[0]['args'] == made[1]['args'] == made[2]['args']
    assert '--dependency=singleton' in made[0]['args'] and made[0]['args'][-1] == str(repo / 'hyak/slurm/07_train.sbatch')
    assert made[0]['env'] == f'train-uniform|uniform|{repo}|from-user'
    assert (tmp_config.base / 'monitor/train-uniform').is_dir()
    log.unlink()
    out = submit(['pool', '--env', 'X_Y=override'], env, script=repo / 'hyak/slurm/submit.sh')
    assert out.returncode == 0, out.stderr
    (made,) = calls(log)
    assert made['env'] == f'pool|unset|{repo}|override'     # a stale HYAK_RUN never reaches the job


def test_submission_refuses_duplicates_and_missing_files(tmp_path, tmp_config):
    repo = fake_repo(tmp_path)
    bindir, log = fake_tools(tmp_path, squeue_output='12345\n')
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_PYTHON': sys.executable}
    script = repo / 'hyak/slurm/submit.sh'
    out = submit(['init'], env, script=script)
    assert out.returncode == 1 and 'already queued or running (12345)' in out.stderr and not log.exists()
    out = submit(['init', '--allow-duplicate'], env, script=script)
    assert out.returncode == 0 and len(calls(log)) == 1
    out = submit(['pool', '--count', '2'], env, script=script)   # multi-job stages ignore the queue
    assert out.returncode == 0 and len(calls(log)) == 3
    bare = fake_repo(tmp_path / 'bare', with_sbatch=False)
    out = submit(['pool'], env, script=bare / 'hyak/slurm/submit.sh')
    assert out.returncode == 1 and 'does not exist' in out.stderr
    out = submit(['pool', '--dry-run'], env, script=bare / 'hyak/slurm/submit.sh')
    assert out.returncode == 0 and 'does not exist yet (dry run continues)' in out.stderr


def test_segment_loop_pattern_stops_after_term(tmp_path):
    """The training-job pattern: persistent flag trap, loop of forwarded children, finaliser still runs."""
    script = tmp_path / 'child.sh'
    script.write_text(CHILD)
    script.chmod(0o755)
    ready, mark, result = tmp_path / 'ready', tmp_path / 'mark', tmp_path / 'result'
    res = shlex.quote(str(result))
    harness = lib_script(f'''hyak_trap_term
i=0
while [ "$HYAK_TERM_RECEIVED" = 0 ] && [ "$i" -lt 5 ]; do
  rc=0; hyak_run_forwarding {shlex.quote(str(script))} || rc=$?
  echo "segment $i rc=$rc" >> {res}
  i=$((i + 1))
done
rc=0; hyak_run_forwarding /bin/sh -c 'exit 0' || rc=$?
echo "final rc=$rc term=$HYAK_TERM_RECEIVED" >> {res}''')
    env = {**os.environ, 'READY': str(ready), 'MARK': str(mark), 'CHILD_EXIT': '0'}
    proc = subprocess.Popen([BASH, '-c', harness], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        deadline = time.time() + 10
        while not ready.exists():
            assert time.time() < deadline and proc.poll() is None
            time.sleep(0.02)
        time.sleep(0.2)
        os.kill(proc.pid, signal.SIGTERM)
        _, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode == 0, err
    assert result.read_text().splitlines() == ['segment 0 rc=0', 'final rc=0 term=1']
