"""Consistency sweep of the port (DESIGN.md 5.I).

* Every ``python -m hyak.*`` entry point named in the run guide, the job scripts, the shell scripts
  or a remedy message in the code answers ``--help`` with this interpreter, and every ``--flag``
  that the run guide, the job scripts and the shell scripts pass to it exists.
* Every ``hyak/slurm/submit.sh`` command in the run guide uses a real stage and real options, and
  the variables it passes with ``--env`` exist (their flags, for the extra-argument variables).
* Every ``bash hyak/...sh`` command in the run guide names an existing script and its flags.
* ``bash -n`` on every shell file; ``submit.sh --dry-run`` finds a job script for every stage;
  ``--env`` values with spaces reach the job.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

from hyak.common.config import STAGE_OVERRIDE_KEYS, parse_env_file
from hyak.common.paths import SBATCH_FILES, STAGES

BASH = shutil.which('bash')
REPO = Path(__file__).resolve().parents[2]
README = REPO / 'hyak' / 'README.md'
SUBMIT = REPO / 'hyak' / 'slurm' / 'submit.sh'
SBATCH = sorted((REPO / 'hyak' / 'slurm').glob('*.sbatch'))
# Filter on the path RELATIVE to the repository: a checkout under a directory that is itself called "tests"
# (or "__pycache__") must not empty these lists.
SHELL_FILES = sorted(p for p in (REPO / 'hyak').rglob('*.sh') if '__pycache__' not in p.relative_to(REPO).parts)
PY_SOURCES = sorted(p for p in (REPO / 'hyak').rglob('*.py')
                    if 'tests' not in p.relative_to(REPO).parts and '__pycache__' not in p.relative_to(REPO).parts)
#: Bytecode files that exist when the module is collected (a test run elsewhere, an earlier import by this very
#: process, or the user's own work may have left some): only a file that is NEW afterwards is this test's doing.
PYC_AT_COLLECTION = frozenset(p for p in (REPO / 'hyak').rglob('*.pyc'))
MODULE_RE = re.compile(r'python3? -m (hyak(?:\.[a-z_]+)+)')
FLAG_RE = re.compile(r'(?<![\w-])(--[a-z][a-z0-9-]*)')
STOP_CHARS = '`|;&)'
SUBMIT_OPTIONS = {'--count', '--debug', '--dry-run', '--run', '--after', '--env', '--allow-duplicate', '--help'}
SUBMIT_VALUE_OPTIONS = {'--count', '--run', '--after', '--env'}

#: Extra-argument variables of the job scripts -> the tool that receives their words.
EXTRA_ARG_TOOLS = {
    'HYAK_CHECK_ENV_ARGS': 'hyak.env.check_env',
    'HYAK_PROBE_ARGS': 'hyak.memory.probe',
    'HYAK_CORPUS_ARGS': 'hyak.corpus.build',
    'HYAK_POOL_ARGS': 'hyak.pool.worker',
    'HYAK_STAGE_INPUTS_ARGS': 'hyak.corpus.stage_inputs',
    'HYAK_INIT_ARGS': 'hyak.model.init_hypernet',
    'HYAK_REPLAY_CHECK_ARGS': 'hyak.train.replay_check',
    'HYAK_WHEEL_ARGS': 'hyak/pool/build_wheel.sh',
    'HYAK_EVAL_PREPARE_ARGS': 'hyak.eval.validate_prefixes',
    'HYAK_EVAL_SCORE_ARGS': 'hyak.eval.score',
}
OTHER_JOB_VARIABLES = {'HYAK_CORPUS_PROBE', 'HYAK_GCC_MODULE', 'HYAK_EVAL_INDICES', 'HYAK_EVAL_PANELS',
                       'HYAK_ACCEPT_CHUNK_CHANGE'}
CONFIG_KEYS = set(parse_env_file(REPO / 'hyak' / 'config.env.example')) | set(STAGE_OVERRIDE_KEYS)


def documents():
    """(name, text) of everything that names commands for the user or runs them."""
    docs = [(README.relative_to(REPO).as_posix(), README.read_text())]
    docs += [(p.relative_to(REPO).as_posix(), p.read_text()) for p in SBATCH + SHELL_FILES]
    return docs


def segments(text):
    """Yield (module, rest-of-command) for every ``python -m hyak.*`` mention, one per mention."""
    for line in text.splitlines():
        matches = list(MODULE_RE.finditer(line))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            rest = line[match.end():end]
            cut = min([rest.index(c) for c in STOP_CHARS if c in rest] + [len(rest)])
            yield match.group(1), rest[:cut]


def module_mentions(include_sources=False):
    mentions = {}
    for name, text in documents():
        for module, rest in segments(text):
            mentions.setdefault(module, []).append((name, rest))
    if include_sources:
        for path in PY_SOURCES:
            for module, rest in segments(path.read_text()):
                mentions.setdefault(module, []).append((path.relative_to(REPO).as_posix(), None))
    return mentions


@lru_cache(maxsize=None)
def help_text(module, subcommand=None):
    argv = [sys.executable, '-m', module] + ([subcommand] if subcommand else []) + ['--help']
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    out = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, f'{" ".join(argv[1:])} failed: {out.stderr[-2000:]}'
    assert 'usage:' in out.stdout, out.stdout[:500]
    return out.stdout


def flags_of(module, rest):
    """Flags in ``rest`` and the help text that must list them (subcommand-aware)."""
    words = rest.split()
    sub = None
    if words and re.fullmatch(r'[a-z][a-z_-]*', words[0]):
        if module == 'hyak.common.status' and words[0] in ('start', 'event', 'finalize', 'show'):
            sub = words[0]
    return FLAG_RE.findall(rest), help_text(module, sub)


ALL_MODULES = sorted(module_mentions(include_sources=True))


def test_mentions_found():
    mentions = module_mentions()
    for module in ('hyak.env.check_env', 'hyak.memory.probe', 'hyak.corpus.build', 'hyak.pool.worker',
                   'hyak.train.run_segment', 'hyak.train.monitor', 'hyak.eval.score', 'hyak.eval.pack'):
        assert module in mentions, module
    assert len(ALL_MODULES) >= 25


@pytest.mark.parametrize('module', ALL_MODULES)
def test_help(module):
    help_text(module)


@pytest.mark.parametrize('module', sorted(module_mentions()))
def test_flags_exist(module):
    for where, rest in module_mentions()[module]:
        flags, text = flags_of(module, rest)
        for flag in flags:
            assert re.search(rf'(?<![\w-]){re.escape(flag)}(?![\w-])', text), \
                f'{where}: python -m {module} has no option {flag} (used in: {rest.strip()!r})'


def test_help_writes_no_bytecode():
    for module in ALL_MODULES:
        help_text(module)
    new = sorted(str(p.relative_to(REPO)) for p in (REPO / 'hyak').rglob('*.pyc') if p not in PYC_AT_COLLECTION)
    assert not new, f'python -m ... --help must not write bytecode, but wrote {new[:5]}'


# --- submit.sh and shell commands in the run guide -------------------------------------------------------------
def readme_commands(prefix):
    """Command lines of the run guide that start with ``prefix`` (inside code spans or code blocks)."""
    found = []
    for line in README.read_text().splitlines():
        for match in re.finditer(re.escape(prefix) + r'([^`\n]*)', line):
            found.append(match.group(1))
    return found


def check_tool_words(variable, words, where):
    tool = EXTRA_ARG_TOOLS[variable]
    flags = [w.split('=', 1)[0] for w in words if w.startswith('--')]
    if tool.endswith('.sh'):
        text = (REPO / tool).read_text()
    else:
        text = help_text(tool)
    for flag in flags:
        assert re.search(rf'(?<![\w-]){re.escape(flag)}(?![\w-])', text), f'{where}: {variable} passes {flag}, unknown to {tool}'


def test_readme_submit_commands():
    commands = readme_commands('hyak/slurm/submit.sh ')
    assert len(commands) >= 20
    seen_stages = set()
    for rest in commands:
        try:
            words = shlex.split(rest)
        except ValueError:
            words = rest.split()
        if not words or words[0].startswith('<'):
            continue                                   # the synopsis line
        stage = words[0]
        assert stage in STAGES, f'unknown stage in README: {rest!r}'
        seen_stages.add(stage)
        i = 1
        while i < len(words):
            word = words[i]
            if word == '--':
                break
            if word.startswith('--'):
                option = word.split('=', 1)[0]
                assert option in SUBMIT_OPTIONS, f'submit.sh has no option {option}: {rest!r}'
                if option in SUBMIT_VALUE_OPTIONS and '=' not in word:
                    value = words[i + 1] if i + 1 < len(words) else ''
                    if option == '--env' and not value.startswith(('VAR', 'KEY')):
                        name, _, val = value.partition('=')
                        assert name in EXTRA_ARG_TOOLS or name in OTHER_JOB_VARIABLES or name in CONFIG_KEYS, \
                            f'--env {name}: not a job variable or config key ({rest!r})'
                        if name in EXTRA_ARG_TOOLS:
                            check_tool_words(name, val.split(), 'README')
                    i += 1
            i += 1
    assert set(STAGES) <= seen_stages, set(STAGES) - seen_stages


def test_readme_shell_commands():
    commands = readme_commands('bash hyak/')
    assert commands
    for rest in commands:
        words = rest.split()
        script = REPO / 'hyak' / words[0]
        assert script.is_file(), f'README names a missing script hyak/{words[0]}'
        text = script.read_text()
        for word in words[1:]:
            if word.startswith('#'):
                break
            if word.startswith('--'):
                assert word.split('=', 1)[0] in text, f'hyak/{words[0]} has no option {word}'
            elif re.fullmatch(r'[a-z]+', word):
                assert re.search(rf'\b{word}\b', text), f'hyak/{words[0]} has no phase {word}'


def test_job_variables_documented_and_used():
    readme = README.read_text()
    scripts = '\n'.join(p.read_text() for p in SBATCH)
    for name in list(EXTRA_ARG_TOOLS) + sorted(OTHER_JOB_VARIABLES):
        assert f'`{name}`' in readme, f'{name} is not documented in the README table'
        assert name in scripts, f'{name} is documented but no job script reads it'
    for name in set(re.findall(r'\b(HYAK_[A-Z_]+_ARGS)\b', scripts)):
        assert name in EXTRA_ARG_TOOLS, f'{name} is read by a job script but not covered here'
    for name in set(re.findall(r'job_extra_args (HYAK_[A-Z_]+)', scripts)):
        assert name in EXTRA_ARG_TOOLS


def test_job_scripts_pass_extra_args_to_the_right_tool():
    for path in SBATCH:
        text = path.read_text().replace('\\\n', ' ')     # join continued lines
        for name in re.findall(r'job_extra_args (HYAK_[A-Z_]+)', text):
            tool = EXTRA_ARG_TOOLS[name]
            after = text[text.index(f'job_extra_args {name}'):]
            call = next(l for l in after.splitlines() if 'JOB_ARGS[@]' in l)
            assert tool in call, f'{path.name}: {name} should feed {tool}: {call}'


def test_readme_paths_exist():
    for rel in sorted(set(re.findall(r'(hyak/[A-Za-z0-9_./-]+\.(?:sh|sbatch|py|example))', README.read_text()))):
        assert (REPO / rel).is_file(), f'README names a missing file {rel}'
    # repository-relative names only: not the tail of a longer path (hyak/variants/<variant>/scripts/... or a
    # staged tree's own scripts/train_joint_smollm_input.py inside a longer path)
    for rel in sorted(set(re.findall(r'(?<![A-Za-z0-9_./-])((?:reports|plans|scripts|monitoring)/[A-Za-z0-9_./-]+'
                                     r'\.(?:json|csv|py|js|gz))', README.read_text()))):
        assert (REPO / rel).is_file(), f'README names a missing file {rel}'


# --- shell syntax and submit.sh ---------------------------------------------------------------------------------
@pytest.mark.skipif(BASH is None, reason='bash not available')
@pytest.mark.parametrize('path', SHELL_FILES + SBATCH, ids=lambda p: p.relative_to(REPO).as_posix())
def test_bash_syntax(path):
    for bash in {BASH, '/bin/bash'}:
        if bash and Path(bash).exists():
            out = subprocess.run([bash, '-n', str(path)], capture_output=True, text=True)
            assert out.returncode == 0, f'{bash}: {out.stderr}'


@pytest.fixture
def fake_slurm(tmp_path, make_tmp_config):
    t = make_tmp_config()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'sbatch.log'
    (bindir / 'sbatch').write_text(
        '#!/bin/bash\n{ printf "ARGS"; for a in "$@"; do printf " [%s]" "$a"; done; printf "\\n";'
        ' printf "POOL_ARGS [%s]\\n" "${HYAK_POOL_ARGS-unset}"; }' f' >> {shlex.quote(str(log))}\necho 777\n')
    (bindir / 'squeue').write_text('#!/bin/sh\nexit 0\n')
    for tool in ('sbatch', 'squeue'):
        (bindir / tool).chmod(0o755)
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HYAK_PYTHON': sys.executable}
    return env, log, t


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_dry_run_every_stage_finds_its_script(fake_slurm):
    env, log, t = fake_slurm
    for stage in STAGES:
        args = [stage, '--dry-run'] + (['--run', 'uniform'] if stage == 'smoke-train' else [])
        out = subprocess.run([BASH, str(SUBMIT), *args], env=env, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, (stage, out.stderr)
        assert 'does not exist' not in out.stderr, (stage, out.stderr)
        (line,) = [l for l in out.stdout.splitlines() if l.startswith('sbatch ')]
        script = Path(shlex.split(line)[-1])
        assert script == REPO.resolve() / 'hyak' / 'slurm' / SBATCH_FILES[stage] and script.is_file()
    assert not log.exists()


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_env_value_with_spaces_reaches_the_job(fake_slurm):
    env, log, t = fake_slurm
    out = subprocess.run([BASH, str(SUBMIT), 'pool', '--env', 'HYAK_POOL_ARGS=--families unigram/no-regex --workers 1'],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    text = log.read_text()
    assert 'POOL_ARGS [--families unigram/no-regex --workers 1]' in text
    # plain --export=ALL: no KEY=VALUE list (it would enable --get-user-env and put the values on the command line)
    assert '[--export=ALL]' in text and 'HYAK_POOL_ARGS=--families' not in text.splitlines()[0]
    assert text.splitlines()[0].endswith(f'[{REPO.resolve() / "hyak/slurm/04_pool.sbatch"}]')


def test_stage_table_documented():
    readme = README.read_text()
    for stage, script in SBATCH_FILES.items():
        assert f'hyak/slurm/{script}' in readme, script
        assert stage in readme, stage
    data = json.loads((REPO / 'plans' / 'run-dashboard.json').read_text())
    assert isinstance(data.get('runs'), list)   # the registry is read, never written, by this test


def test_pins_agree_across_owners():
    """init_hypernet, chunked_hypernet and code/pinned.py pin the same hf_hypernet bytes and packages."""
    from hyak.code import pinned
    from hyak.memory import chunked_hypernet
    init = pytest.importorskip('hyak.model.init_hypernet')
    for name, digest in init.PINNED_SOURCES.items():
        assert pinned.PINNED_FILES[f'runtime/anytok_pool_20k/{name}'][0] == digest, name
    for name, digest in chunked_hypernet.PINNED_SOURCES.items():
        assert pinned.PINNED_FILES[f'runtime/anytok_pool_20k/hf_hypernet/{name}'][0] == digest, name
    assert set(init.PINNED_PACKAGES.items()) <= set(pinned.TRAIN_PINS)


# --- submit.sh: what a submission may and may not do (slurm-2, 3, 6, 7, 8, 10; train-lifecycle-3, 12) ----------------
def dry(args, env, **extra):
    return subprocess.run([BASH, str(SUBMIT), *args, '--dry-run'], env={**env, **extra}, capture_output=True, text=True,
                          timeout=60)


def own_warnings(stderr):
    """The warnings of submit.sh itself (the tests' HYAK_MONITOR_ROOT always adds one more)."""
    return '\n'.join(l for l in stderr.splitlines() if 'WARNING' in l and 'HYAK_MONITOR_ROOT is set' not in l)


def sbatch_flags(stdout):
    (line,) = [l for l in stdout.splitlines() if l.startswith('sbatch ')]
    return shlex.split(line)[1:-1]


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_export_is_plain_all_and_env_values_stay_off_the_command_line(fake_slurm):
    env, log, t = fake_slurm
    out = dry(['train-uniform', '--env', 'HYAK_POOL_ARGS=--a,b c'], env)
    assert out.returncode == 0, out.stderr
    flags = sbatch_flags(out.stdout)
    assert '--export=ALL' in flags and not [f for f in flags if f.startswith('--export=ALL,')]
    assert not [f for f in flags if 'HYAK_POOL_ARGS' in f]                  # commas and spaces are fine: nothing is listed
    assert '# job environment' in out.stdout and 'HYAK_POOL_ARGS=--a,b c' in out.stdout
    # a real submission: the job's variables reach sbatch through its environment
    real = subprocess.run([BASH, str(SUBMIT), 'train-uniform', '--env', 'HYAK_POOL_ARGS=--a,b c'], env=env,
                          capture_output=True, text=True, timeout=60)
    assert real.returncode == 0, real.stderr
    assert 'POOL_ARGS [--a,b c]' in log.read_text() and '[--export=ALL]' in log.read_text()


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_secrets_are_refused_in_env_and_never_echoed(fake_slurm, tmp_path):
    env, log, t = fake_slurm
    for name in ('HF_TOKEN', 'MY_API_KEY', 'DB_PASSWORD'):
        out = dry(['pool', '--env', f'{name}=hf_supersecret123'], env)
        assert out.returncode == 1 and 'secrets are not passed' in out.stderr and 'hf_supersecret123' not in out.stdout + out.stderr
    out = dry(['pool', '--env', 'not a pair hf_supersecret123'], env)
    assert out.returncode == 1 and 'hf_supersecret123' not in out.stdout + out.stderr
    # a malformed config line may hold a token: only file and line number are shown
    bad = tmp_path / 'bad.env'
    bad.write_text(f'ANYTOK_ROOT={t.paths.root}\nANYTOK_RUNS={t.paths.runs}\nHF_TOKEN hf_supersecret123\n')
    out = dry(['pool'], env, HYAK_CONFIG=str(bad))
    assert out.returncode == 1 and 'bad.env:3: expected KEY=value' in out.stderr
    assert 'hf_supersecret123' not in out.stdout + out.stderr


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_a_config_key_given_with_env_decides_the_resources_and_the_signal(fake_slurm):
    env, log, t = fake_slurm
    out = dry(['smoke-train', '--run', 'uniform', '--env', 'HYAK_TIME_SMOKE_TRAIN=00:40:00',
               '--env', 'HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_TRAIN=5'], env)
    assert out.returncode == 0, out.stderr
    flags = sbatch_flags(out.stdout)
    assert '--time=00:40:00' in flags and '--signal=B:TERM@300' in flags
    out = dry(['pool', '--env', 'HYAK_CPUS_POOL=4', '--env', 'HYAK_MEM_POOL=64G'], env)
    flags = sbatch_flags(out.stdout)
    assert '--cpus-per-task=4' in flags and '--mem=64G' in flags


@pytest.mark.skipif(BASH is None, reason='bash not available')
@pytest.mark.parametrize('flag', ['--time=00:40:00', '-t', '--signal=B:TERM@60', '-p', '--partition=x', '--qos=debug',
                                  '--gres=gpu:2', '--mem=100G', '--mem-per-cpu=1G', '-c4', '--cpus-per-task=2',
                                  '--account=x', '--kill-on-invalid-dep=no'])
def test_resource_flags_after_the_double_dash_are_refused(fake_slurm, flag):
    env, log, t = fake_slurm
    out = dry(['smoke-train', '--run', 'uniform', '--', flag], env)
    assert out.returncode == 1 and (flag in out.stderr) and not log.exists()
    if not flag.startswith('--kill'):
        assert 'HYAK_<FLAG>_<STAGE>' in out.stderr
    ok = dry(['pool', '--', '--mail-type=END'], env)
    assert ok.returncode == 0 and '--mail-type=END' in sbatch_flags(ok.stdout)


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_after_adds_kill_on_invalid_dep(fake_slurm):
    env, log, t = fake_slurm
    flags = sbatch_flags(dry(['init', '--after', '5:6'], env).stdout)
    assert '--kill-on-invalid-dep=yes' in flags and '--dependency=afterok:5:6' in flags
    assert not [f for f in sbatch_flags(dry(['init'], env).stdout) if f.startswith('--kill')]
    flags = sbatch_flags(dry(['train-uniform', '--after', '5'], env).stdout)
    assert '--dependency=singleton,afterok:5' in flags and '--kill-on-invalid-dep=yes' in flags


@pytest.mark.skipif(BASH is None, reason='bash not available')
@pytest.mark.parametrize('margin', ['08', '09', '045'])
def test_a_margin_with_a_leading_zero_is_decimal(fake_slurm, margin):
    """bash arithmetic read 08 as an invalid octal and silently dropped --signal (slurm-8)."""
    env, log, t = fake_slurm
    out = dry(['train-uniform'], env, HYAK_SEGMENT_MARGIN_MINUTES=margin)
    assert out.returncode == 0, out.stderr
    assert f'--signal=B:TERM@{int(margin) * 60}' in sbatch_flags(out.stdout)
    assert f'signal_seconds={int(margin) * 60}' in out.stdout


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_training_needs_a_limit_of_margin_plus_twenty_minutes_and_a_margin(fake_slurm):
    env, log, t = fake_slurm
    out = dry(['train-uniform', '--debug'], env)
    assert out.returncode == 1 and 'shorter than the margin (45 min) + 20 min = 65 min' in out.stderr
    assert '--debug is for the short stages' in out.stderr and 'HYAK_TIME_TRAIN_UNIFORM' in out.stderr
    out = dry(['train-rl'], env, HYAK_TIME_TRAIN_RL='01:00:00')
    assert out.returncode == 1 and '1:05:00' in out.stderr
    assert dry(['train-rl', '--env', 'HYAK_TIME_TRAIN_RL=01:00:00', '--env', 'HYAK_SEGMENT_MARGIN_MINUTES_TRAIN_RL=30'],
               env).returncode == 0
    out = dry(['train-uniform'], env, HYAK_SEGMENT_MARGIN_MINUTES='0')
    assert out.returncode == 1 and 'would not be asked to stop' in out.stderr
    out = dry(['smoke-train', '--run', 'uniform'], env, HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_TRAIN='0')
    assert out.returncode == 1
    # the short stages are not subject to it
    assert dry(['init', '--debug'], env).returncode == 0 and dry(['smoke-train', '--run', 'uniform', '--debug'], env).returncode == 0


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_rl_is_refused_until_the_uniform_pairing_exists(fake_slurm):
    env, log, t = fake_slurm
    pairing = t.paths.root / 'input' / 'pairing.json'
    out = subprocess.run([BASH, str(SUBMIT), 'train-rl', '--count', '4'], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 1 and 'pairing.json has not been written yet' in out.stderr and 'pairing_frozen' in out.stderr
    assert not log.exists()                                              # nothing was submitted
    out = subprocess.run([BASH, str(SUBMIT), 'smoke-train', '--run', 'rl'], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 1 and not log.exists()
    warned = dry(['train-rl'], env)                                      # a dry run only warns
    assert warned.returncode == 0 and 'refused until then' in own_warnings(warned.stderr)
    pairing.parent.mkdir(parents=True)
    pairing.write_text('{}')
    out = subprocess.run([BASH, str(SUBMIT), 'train-rl', '--count', '2'], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.count('submitted anytok-train-rl') == 2
    assert own_warnings(dry(['train-rl'], env).stderr) == ''
    assert dry(['train-uniform'], env).returncode == 0                  # the uniform run never needs it


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_default_resources_of_the_short_stages_and_no_memory_flag(fake_slurm):
    """Defaults of config.env.example: short limits for short stages, no --mem, training at 24 h / 45 min."""
    env, log, t = fake_slurm
    expected = {'init': '01:00:00', 'corpus': '06:00:00', 'wheel': '00:30:00', 'replay-check': '02:00:00',
                'eval-prepare': '01:00:00', 'smoke-train': '02:00:00', 'train-uniform': '24:00:00', 'train-rl': '24:00:00',
                'pool': '24:00:00', 'check-env': '01:00:00', 'probe': '01:00:00'}
    for stage, limit in expected.items():
        out = dry([stage] + (['--run', 'uniform'] if stage == 'smoke-train' else []), env)
        assert out.returncode == 0, (stage, out.stderr)
        flags = sbatch_flags(out.stdout)
        assert f'--time={limit}' in flags, (stage, flags)
        assert not [f for f in flags if f.startswith('--mem')], stage      # the cluster's per-GPU default applies
        assert '--cpus-per-task=8' in flags
    assert '--signal=B:TERM@600' in sbatch_flags(dry(['smoke-train', '--run', 'uniform'], env).stdout)
    assert '--signal=B:TERM@2700' in sbatch_flags(dry(['train-uniform'], env).stdout)
    assert '--mem=160G' in sbatch_flags(dry(['pool'], env, HYAK_MEM='160G').stdout)             # still an override


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_tillicum_warnings_know_mig_units_and_the_qos_limits(fake_slurm):
    env, log, t = fake_slurm
    def warnings(args, **extra):
        out = dry(args, env, **extra)
        assert out.returncode == 0, out.stderr
        return own_warnings(out.stderr)
    mig = warnings(['pool', '--env', 'HYAK_PARTITION_POOL=gpu-h200-mig'])
    assert '2 workers x 4 threads need 8 CPUs' in mig and 'MIG units' in mig and 'a MIG unit has 1 CPU' in mig
    assert 'a MIG unit has 30 GB' not in mig                              # no --mem by default
    assert 'a MIG unit has 30 GB' in warnings(['init', '--env', 'HYAK_PARTITION_INIT=gpu-h200-mig', '--env', 'HYAK_CPUS_INIT=1',
                                              '--env', 'HYAK_MEM_INIT=64G'])
    assert '195.3125 GB' in warnings(['init'], HYAK_MEM='201G') and 'WARNING' not in warnings(['init'], HYAK_MEM='160G')
    assert 'WARNING' not in warnings(['init'], HYAK_MEM='200000')                  # unit-aware: bare numbers are MiB
    assert '195.3125 GB' in warnings(['init'], HYAK_MEM='200001M')
    assert 'at most 8 CPUs per GPU' in warnings(['init'], HYAK_CPUS='9') and 'WARNING' not in warnings(['init', '--env', 'HYAK_GRES_INIT=gpu:2', '--env', 'HYAK_CPUS_INIT=16'])
    assert 'has no GPU' in warnings(['init'], HYAK_GRES='none')
    assert 'debug QoS allows 1 h' in warnings(['init', '--env', 'HYAK_QOS_INIT=debug', '--env', 'HYAK_TIME_INIT=02:00:00'])
    assert 'debug QoS allows 1 GPU' in warnings(['init', '--env', 'HYAK_QOS_INIT=debug', '--env', 'HYAK_TIME_INIT=00:30:00',
                                                 '--env', 'HYAK_GRES_INIT=gpu:2'])
    assert 'normal QoS allows 24 h' in warnings(['init', '--env', 'HYAK_TIME_INIT=2-00:00:00'])
    assert 'WARNING' not in warnings(['init'])
    # a misspelled per-stage variable is no longer ignored in silence
    assert 'HYAK_TIME_SMOKETRAIN is not a per-stage setting' in warnings(['init'], HYAK_TIME_SMOKETRAIN='00:40:00')


@pytest.mark.skipif(BASH is None, reason='bash not available')
def test_config_example_does_not_recommend_the_mig_partition():
    text = (REPO / 'hyak' / 'config.env.example').read_text()
    assert '#   HYAK_PARTITION_POOL=gpu-h200-mig' not in text and 'Do not use the MIG partition' in text
    assert 'HYAK_TIME_SMOKE_TRAIN=' in text and 'cu132' in text and 'too old for the default CUDA 13 wheel' not in text
