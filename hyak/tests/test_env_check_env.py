"""Tests of hyak.env.check_env (DESIGN.md 5.A).

The real checks need a Linux GPU node.  Here: every pure decision function, the report and
exit-code logic, the real filesystem probe, the CPU form of the GPU smoke step, and the
subprocess probes against (a) a C-compiled fake ``rust_utils`` extension that mimics
``rust_utils/src/lib.rs`` in Python, (b) the macOS build of the real crate if it exists in the
scratch directory (``rust-utils-macos/``), and (c) the committed Linux library (which must be
reported as the wrong platform on this Mac).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

from hyak.code import pinned, stage_code
from hyak.env import check_env as ce
from hyak.env.check_env import FAIL, PASS, SKIP, WARN, CheckResult


def _args(**overrides):
    values = dict(config=None, only=None, skip=None, no_gpu=True, no_network=True, rust_library=None,
                  io_test_mib=0, hash_model=False, probe_timeout=300.0, report=None, no_status=True)
    values.update(overrides)
    return argparse.Namespace(**values)


# --- pure helpers --------------------------------------------------------------------------------------
def test_version_parsers():
    assert ce.parse_version('580.95.05') == (580, 95, 5)
    assert ce.parse_version('13.2') == (13, 2)
    assert ce.parse_version('2.15.0+cu134') == (2, 15, 0)
    assert ce.parse_version(None) == ()
    assert ce.parse_glibc('glibc 2.34') == (2, 34)
    assert ce.parse_glibc('musl 1.2') == ()
    assert ce.torch_build_label('2.15.0+cu134') == 'cu134'
    assert ce.torch_build_label('2.15.0') == ''
    assert ce.expected_torch_label('') == ''
    assert ce.expected_torch_label('https://download.pytorch.org/whl/cu132/') == 'cu132'


NVIDIA_QUERY = 'NVIDIA H200, 143771, 580.95.05, 9.0\nNVIDIA H200, 143771, 580.95.05, 9.0\n'
NVIDIA_PLAIN = '''+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+'''


def test_nvidia_smi_parsers():
    rows = ce.parse_nvidia_query(NVIDIA_QUERY)
    assert rows[0] == {'name': 'NVIDIA H200', 'memory_total_mib': 143771, 'driver_version': '580.95.05',
                       'compute_cap': '9.0'}
    assert len(rows) == 2
    assert ce.parse_driver_cuda(NVIDIA_PLAIN) == '13.0'
    assert ce.parse_driver_cuda('no gpu') is None
    with pytest.raises(ValueError):
        ce.parse_nvidia_query('garbage')


@pytest.mark.parametrize('driver,driver_cuda,torch_cuda,status,needle', [
    ('595.45.04', '13.2', '13.2', PASS, 'runs torch CUDA 13.2'),
    ('620.10', '13.5', '13.2', PASS, 'runs'),
    ('580.95.05', '13.0', '13.2', WARN, 'minor version compatibility'),
    ('575.57.08', '12.9', '13.2', FAIL, 'too old'),
    ('550.54.15', '12.4', '12.8', WARN, 'minor version compatibility'),
    ('520.61.05', '12.0', '12.8', FAIL, 'too old'),
    ('595.45.04', '13.2', None, FAIL, 'CPU-only'),
    ('', None, '13.2', FAIL, 'cannot read'),
])
def test_cuda_driver_verdict(driver, driver_cuda, torch_cuda, status, needle):
    got, summary, remedy = ce.cuda_driver_verdict(driver, driver_cuda, torch_cuda)
    assert got == status and needle in summary
    assert (remedy is None) == (status == PASS)


def test_old_driver_remedy_explains_that_no_index_url_helps():
    _, _, remedy = ce.cuda_driver_verdict('570.124.06', '12.8', '13.2')
    assert 'no HYAK_TORCH_INDEX_URL fixes this' in remedy and 'cu132, cu134' in remedy
    assert '>= 580' in remedy and 'cuda-compat' in remedy and 'approval' in remedy
    assert ce.TORCH_CUDA_BUILDS == {'': '13.2', 'cu132': '13.2', 'cu134': '13.4'}


LDD = '''\tlinux-vdso.so.1 (0x00007ffc)
\tlibpython3.12.so.1.0 => not found
\tlibgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f)
\tlibc.so.6 => /lib64/libc.so.6 (0x00007f)
/lib64/libc.so.6: version `GLIBC_2.34' not found (required by ./librust_utils.so)
'''


def test_parse_ldd_and_remedies():
    facts = ce.parse_ldd(LDD)
    assert facts['not_found'] == ['libpython3.12.so.1.0']
    assert facts['version_errors'] and 'GLIBC_2.34' in facts['version_errors'][0]
    assert facts['libraries']['libc.so.6'] == '/lib64/libc.so.6'
    assert 'LD_LIBRARY_PATH' in ce.rust_failure_remedy('ImportError: x', facts)
    assert 'glibc >= 2.34' in ce.rust_failure_remedy('ImportError: x', {'version_errors': ['v'], 'not_found': []})
    assert 'stage the code' in ce.rust_failure_remedy("FileNotFoundError: Build ZeTT's Rust sampler first")
    assert 'another platform' in ce.rust_failure_remedy('dlopen(...) (slice is not valid mach-o file)')
    assert 'another platform' in ce.rust_failure_remedy('librust_utils.so: invalid ELF header')
    assert 'extension-module' in ce.rust_failure_remedy('probe crashed with signal 11')
    assert 'not a rust_utils extension' in ce.rust_failure_remedy(
        'ImportError: dynamic module does not define module export function (PyInit_rust_utils)')
    assert 'build_rust_utils.sh' in ce.rust_failure_remedy('something else')


def test_binary_info_of_the_committed_library_and_this_interpreter(repo_root):
    info = ce.binary_info(repo_root / pinned.RUST_LIBRARY)
    assert info == {'format': 'ELF', 'bits': 64, 'machine': 'x86_64'}
    host = ce.binary_info(Path(sys.executable).resolve())
    assert host['format'] == ce.host_binary_expectation()['format']
    assert ce.binary_info(repo_root / 'README.md')['format'] == 'unknown'


def test_package_problems_and_probe_output():
    pins = (('torch', '2.15.0'), ('numpy', '2.3.5'))
    assert ce.package_problems({'torch': '2.15.0+cu134', 'numpy': '2.3.5'}, pins) == []
    assert ce.package_problems({'torch': '2.14.1', 'numpy': None}, pins) == [
        'torch 2.14.1 installed, need 2.15.0', 'numpy is not installed (need 2.3.5)']
    out = 'noise\nHYAK_PROBE_RESULT {"ok": false}\nHYAK_PROBE_RESULT {"ok": true}\nHYAK_PROBE_RESULT {bad\n'
    assert ce.parse_probe_output(out) == {'ok': True}
    assert ce.parse_probe_output('nothing') is None


def test_check_result_rules_render_and_exit_code():
    with pytest.raises(ValueError):
        CheckResult('x', WARN, 'needs a remedy')
    with pytest.raises(ValueError):
        CheckResult('x', 'odd', 's')
    results = [CheckResult('a', PASS, 'fine'), CheckResult('b', WARN, 'hmm', 'do this'),
               CheckResult('c', SKIP, 'skipped')]
    assert ce.exit_code(results) == 0
    assert ce.render(results[1]) == '[WARN] b: hmm\n       remedy: do this'
    assert ce.render(results[0]) == '[PASS] a: fine'
    results.append(CheckResult('d', FAIL, 'broken', 'fix it'))
    assert ce.exit_code(results) == 4 == ce.EXIT_CHECK_FAILED      # the project's contract: 4 = a check failed
    assert ce.summary_line(results) == 'check_env FAILED: 1 passed, 1 warnings, 1 failed, 1 skipped'


def test_select_checks():
    assert ce.select_checks() == list(ce.CHECK_NAMES)
    assert ce.select_checks(['glibc,python']) == ['python', 'glibc']        # pipeline order kept
    assert 'network' not in ce.select_checks(None, ['network'])
    with pytest.raises(ValueError, match='unknown check'):
        ce.select_checks(['nope'])


def test_disk_verdict_thresholds():
    gb = 10 ** 9
    assert ce.disk_verdict({'runs': 2000 * gb, 'root': 500 * gb})[0] == PASS
    status, _, remedy = ce.disk_verdict({'runs': 800 * gb, 'root': 500 * gb})
    assert status == WARN and '1100 GB' in remedy
    assert ce.disk_verdict({'runs': 2000 * gb, 'root': 50 * gb})[0] == WARN
    assert ce.disk_verdict({'runs': 20 * gb, 'root': 500 * gb})[0] == FAIL


# --- runner, report, status ---------------------------------------------------------------------------
def test_run_checks_turns_crashes_into_failures_and_applies_skips(monkeypatch):
    def boom(ctx):
        raise RuntimeError('kaboom')
    checks = (('config', ce.check_config, ()), ('python', boom, ()), ('nvidia_smi', boom, ('gpu',)),
              ('network', boom, ('network',)), ('code_stage', boom, ('config',)))
    monkeypatch.setattr(ce, 'CHECKS', checks)
    ctx = ce.Context(args=_args(), config_error='ConfigError: missing')
    seen = []
    results = ce.run_checks(ctx, [c[0] for c in checks], on_result=lambda r, done: seen.append(r.name))
    by = {r.name: r for r in results}
    assert by['config'].status == FAIL and 'missing' in by['config'].summary
    assert by['python'].status == FAIL and 'kaboom' in by['python'].summary
    assert 'Traceback' in by['python'].details['traceback']
    assert by['nvidia_smi'].status == SKIP and by['network'].status == SKIP and by['code_stage'].status == SKIP
    assert seen == [c[0] for c in checks]


def test_filesystem_semantics_on_a_local_directory(tmp_path):
    facts = ce.filesystem_semantics(tmp_path, io_mib=8)
    for key in ce.REQUIRED_FS_FACTS + ('replace_dir_over_empty',):
        assert facts[key] is True, (key, facts)
    assert facts['flock_while_held'] == 'blocked' and facts['flock_after_release'] == 'acquired'
    assert facts['io_test_bytes'] == 8 * 2 ** 20 and facts['write_mb_s'] > 0
    assert list(tmp_path.iterdir()) == []                       # cleaned up
    assert ce.filesystem_verdict('fs', 'ANYTOK_RUNS', facts).status in (PASS, WARN)


def test_filesystem_verdict_failures():
    facts = {'base': '/x', 'flock_excludes': False, 'mkdir_exclusive': True, 'replace_file': True,
             'replace_dir': True, 'fsync': True}
    result = ce.filesystem_verdict('filesystem_runs', 'ANYTOK_RUNS', facts)
    assert result.status == FAIL and '.run.lock' in result.remedy
    slow = {**facts, 'flock_excludes': True, 'write_mb_s': 50.0, 'read_hash_mb_s': 60.0}
    assert ce.filesystem_verdict('filesystem_runs', 'ANYTOK_RUNS', slow).status == WARN


class _Resp:
    def __init__(self, status):
        self.status = status

    def read(self, n):
        return b''

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_network_check_is_warning_only():
    ok = ce.check_network(ce.Context(args=_args()), opener=lambda request, timeout: _Resp(200))
    assert ok.status == PASS and set(ok.details['endpoints']) == {n for n, _, _ in ce.NETWORK_ENDPOINTS}

    def blocked(request, timeout):
        raise OSError('Network is unreachable')
    result = ce.check_network(ce.Context(args=_args()), opener=blocked)
    assert result.status == WARN and 'login node' in result.remedy
    assert result.details['endpoints']['huggingface.co']['error'].startswith('OSError')


def test_gpu_smoke_step_on_cpu():
    torch = pytest.importorskip('torch')
    result = ce.gpu_smoke('cpu')
    assert result['matmul_finite'] and result['parameters_changed'] and result['parameters_finite']
    assert result['output_dtype'] == str(torch.bfloat16)


def test_packages_and_python_checks_in_the_pinned_test_venv(tmp_config):
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    packages = ce.check_packages(ctx)
    assert packages.status == PASS, packages
    assert packages.details['tokenizers_patched'] is False
    python = ce.check_python(ctx)
    assert python.status == WARN and 'not the train env' in python.summary   # the test venv is not env/train
    ctx.cfg = {**tmp_config.cfg, 'HYAK_TORCH_INDEX_URL': 'https://download.pytorch.org/whl/cu134'}
    mismatch = ce.check_packages(ctx)
    assert mismatch.status == WARN and '--recreate' in mismatch.remedy


def test_model_files_check(tmp_config, scratch_assets):
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    assert ce.check_model_files(ctx).status == WARN
    dest = tmp_config.paths.model_dir
    dest.mkdir(parents=True)
    shutil.copyfile(scratch_assets / 'full' / 'config.json', dest / 'config.json')
    result = ce.check_model_files(ctx)
    assert result.status == FAIL and 'model.safetensors: missing' in result.summary


def test_main_writes_report_and_status(tmp_config, capsys):
    code = ce.main(['--only', 'config,python,glibc,resources,filesystem_root', '--no-network', '--no-gpu'])
    out = capsys.readouterr().out
    assert code == 0, out
    monitor = tmp_config.paths.monitor_dir('check-env')
    report = json.loads((monitor / 'check_env_report.json').read_text())
    assert report['kind'] == 'hyak_check_env_report' and report['exit_code'] == 0
    assert [c['name'] for c in report['checks']] == ['config', 'python', 'glibc', 'filesystem_root', 'resources']
    assert report['warnings'] == ['python']
    status = json.loads((monitor / 'status.json').read_text())
    assert status['state'] == 'complete' and status['stage'] == 'complete' and status['completed'] == 5
    assert status['total'] == 5 and status['warnings'] == ['python']
    events = [json.loads(line) for line in (monitor / 'events.jsonl').read_text().splitlines()]
    assert events[0]['completed'] == 0 and events[-1]['state'] == 'complete'
    assert 'check_env OK' in out


def test_main_fails_with_exit_code_4_and_failed_status(tmp_config, capsys):
    code = ce.main(['--only', 'code_stage', '--no-network'])
    assert code == 4
    out = capsys.readouterr().out
    assert '[FAIL] code_stage' in out and 'remedy: python -m hyak.code.stage_code' in out
    status = json.loads((tmp_config.paths.monitor_dir('check-env') / 'status.json').read_text())
    assert status['state'] == 'failed' and status['failed'] == ['code_stage'] and 'code_stage' in status['error']


def test_main_without_config_reports_and_skips(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('HYAK_CONFIG', str(tmp_path / 'nothing.env'))
    code = ce.main(['--only', 'config,code_stage', '--no-status'])
    out = capsys.readouterr().out
    assert code == 4 and '[FAIL] config' in out and '[SKIP] code_stage' in out


def test_no_status_writes_only_the_requested_report(tmp_config, tmp_path):
    report = tmp_path / 'r.json'
    assert ce.main(['--only', 'resources', '--no-status', '--report', str(report)]) == 0
    assert json.loads(report.read_text())['checks'][0]['name'] == 'resources'
    assert not tmp_config.paths.monitor_dir('check-env').exists()


def test_list(capsys):
    assert ce.main(['--list']) == 0
    names = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert names == list(ce.CHECK_NAMES)


# --- probes against real and fake extension modules ----------------------------------------------------
#: Python model of rust_utils/src/lib.rs (TokenizerSampler.sample_tokenizer), compiled into a C
#: extension so the probe loads it through the trainer's exact loader (ExtensionFileLoader).
FAKE_RUST_UTILS = r'''
import math as _math, random as _random, re as _re
__version__ = "0.14.1-dev.0"
_SPLIT = _re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
_WS = ("Ġ", "Ċ", "ĉ")

class TokenizerSampler:
    def __init__(self):
        self._cache = []

    def sample_tokenizer(self, map, seed_size, max_length, stride=None, noise_std=None,
                         pop_prev=None, push_current=None):
        stride = 1 if stride is None else stride
        noise_std = 0.0 if noise_std is None else noise_std
        pop_prev = True if pop_prev is None else pop_prev
        push_current = True if push_current is None else push_current
        current = {}
        for sentence, n in map.items():
            for piece in _SPLIT.findall(" " + sentence):
                piece = piece.replace(" ", "Ġ").replace("\n", "Ċ").replace("\t", "ĉ")
                for i in range(0, len(piece), stride):
                    for k in range(1, max_length):
                        if i + k > len(piece):
                            break
                        token = piece[i:i + k]
                        current[token] = current.get(token, 0) + n * len(token.encode())
        previous = self._cache.pop() if (pop_prev and self._cache) else None
        self._cache.insert(0, current)
        out = []
        if pop_prev:
            merged = {}
            for entry in self._cache:
                for key, value in entry.items():
                    merged[key] = merged.get(key, 0) + value
            total = float(sum(merged.values()))
            from tokenizers import pre_tokenizers
            out = [(c, _math.log(min(merged.values()) / total)) for c in pre_tokenizers.ByteLevel.alphabet()]
            for c1 in _WS:
                for i in range(1, max_length):
                    for c2 in _WS:
                        out.append((c2 + c1 * i, 0.0))
            rng = _random.Random()
            scored = []
            for key, value in merged.items():
                noised = value / total + (rng.gauss(0.0, noise_std) if noise_std else 0.0)
                scored.append((key, _math.log(noised) if noised > 0 else -100000.0))
            scored.sort(key=lambda row: row[1], reverse=True)
            for key, score in scored:
                if len(key) == 1 or sum(ch in _WS for ch in key) >= 2:
                    continue
                out.append((key, score))
                if len(out) >= seed_size:
                    break
        if not push_current:
            self._cache.pop(0)
            if previous is not None:
                self._cache.append(previous)
        return out
'''

C_TEMPLATE = r'''
#define PY_SSIZE_T_CLEAN
#include <Python.h>
static const char *SOURCE = %s;
static struct PyModuleDef definition = {PyModuleDef_HEAD_INIT, "rust_utils", NULL, -1, NULL};
PyMODINIT_FUNC PyInit_rust_utils(void) {
    PyObject *module = PyModule_Create(&definition);
    if (module == NULL) return NULL;
    PyObject *globals = PyModule_GetDict(module);
    if (PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins()) < 0) return NULL;
    PyObject *result = PyRun_String(SOURCE, Py_file_input, globals, globals);
    if (result == NULL) { Py_DECREF(module); return NULL; }
    Py_DECREF(result);
    return module;
}
'''


@pytest.fixture(scope='module')
def fake_rust_library(tmp_path_factory):
    """``librust_utils.so``: a CPython extension exporting PyInit_rust_utils (needs a C compiler)."""
    compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    include = sysconfig.get_paths()['include']
    if compiler is None or not Path(include, 'Python.h').is_file():
        pytest.skip('no C compiler or Python headers to build the fake extension')
    work = tmp_path_factory.mktemp('fake-rust')
    (work / 'fake.c').write_text(C_TEMPLATE % json.dumps(FAKE_RUST_UTILS))
    target = work / 'librust_utils.so'
    flags = ['-shared', '-fPIC', '-O1']
    if sys.platform == 'darwin':
        flags += ['-undefined', 'dynamic_lookup']
    command = [compiler, *flags, f'-I{include}', str(work / 'fake.c'), '-o', str(target)]
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode != 0 and sys.platform == 'darwin' and 'tapi error' in out.stderr:
        # Local toolchain quirk on this Mac: the newest SDK's .tbd files confuse ld; use an older SDK.
        for sdk in sorted(Path('/Library/Developer/CommandLineTools/SDKs').glob('MacOSX1[0-9]*.sdk'), reverse=True):
            out = subprocess.run(command, capture_output=True, text=True, env={**os.environ, 'SDKROOT': str(sdk)})
            if out.returncode == 0:
                break
    if out.returncode != 0:
        pytest.skip(f'could not compile the fake extension: {out.stderr[-500:]}')
    return target


@pytest.fixture
def macos_rust_library(scratch_dir):
    """The real crate built on this Mac by hyak/env/build_rust_utils.sh (scratch asset), if present."""
    path = scratch_dir / 'rust-utils-macos' / 'librust_utils.so'
    if sys.platform != 'darwin' or not path.is_file():
        pytest.skip(f'no macOS build of the real rust_utils at {path}')
    return path


def _rust_ctx(repo_root, library, paths=None):
    return ce.Context(args=_args(rust_library=library), paths=paths, repo=repo_root)


def test_rust_probe_with_the_fake_extension(fake_rust_library, repo_root):
    result = ce.check_rust_library(_rust_ctx(repo_root, fake_rust_library))
    assert result.status == PASS, (result.summary, result.details.get('probe'), result.details.get('probe_stderr'))
    probe = result.details['probe']
    assert [s['step'] for s in probe['steps']] == ['loaded', 'raw_calls', 'prime_sampler', 'draw_tokenizer']
    assert probe['steps'][0]['loader'] == 'ExtensionFileLoader'
    assert probe['steps'][-1]['vocab'] == 32768 and probe['version'] == pinned.RUST_UTILS_VERSION


def test_rust_probe_with_the_real_crate_built_on_this_mac(macos_rust_library, repo_root):
    result = ce.check_rust_library(_rust_ctx(repo_root, macos_rust_library))
    assert result.status == PASS, (result.summary, result.details.get('probe'), result.details.get('probe_stderr'))
    assert result.details['probe']['steps'][-1]['vocab'] == 32768
    assert result.details['committed'] is False


def test_rust_probe_rejects_a_module_without_pyinit_rust_utils(repo_root, tmp_path):
    candidates = [p for p in Path(sysconfig.get_paths()['purelib']).rglob('*.so') if p.stat().st_size < 2_000_000]
    other = min(candidates, key=lambda p: p.stat().st_size, default=None)
    if other is None:
        pytest.skip('no compiled extension module to stand in')
    result = ce.check_rust_library(_rust_ctx(repo_root, other))
    assert result.status == FAIL and 'PyInit_rust_utils' in result.summary
    assert 'not a rust_utils extension' in result.remedy


@pytest.mark.skipif(sys.platform.startswith('linux') and os.uname().machine == 'x86_64',
                    reason='on Linux x86-64 the committed library is the right platform')
def test_committed_linux_library_is_reported_as_the_wrong_platform(repo_root):
    result = ce.check_rust_library(_rust_ctx(repo_root, repo_root / pinned.RUST_LIBRARY))
    assert result.status == FAIL and 'ELF x86_64 binary' in result.summary
    assert 'build_rust_utils.sh' in result.remedy


def test_rust_library_missing(repo_root, tmp_path):
    result = ce.check_rust_library(_rust_ctx(repo_root, tmp_path / 'absent.so'))
    assert result.status == FAIL and 'not found' in result.summary


def test_run_probe_reports_crashes_and_timeouts():
    crashed = ce.run_probe('import os, signal; os.kill(os.getpid(), signal.SIGSEGV)', {}, timeout=60)
    assert crashed['result'] is None and 'signal 11' in crashed['error']
    slow = ce.run_probe('import time; time.sleep(30)', {}, timeout=1)
    assert 'timed out' in slow['error']
    silent = ce.run_probe('print("hello")', {}, timeout=60)
    assert 'without a result' in silent['error']


@pytest.mark.slow
@pytest.mark.parametrize('library', ['fake', 'macos'])
def test_staged_trainers_import_from_their_trees(library, request, tmp_config, repo_root):
    """Stage both trees with a loadable library and import each pinned trainer the way the launchers did."""
    path = request.getfixturevalue('fake_rust_library' if library == 'fake' else 'macos_rust_library')
    stage_code.stage(tmp_config.paths, rust_library=str(path), log=lambda m: None)
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths, repo=repo_root)
    assert ce.check_code_stage(ctx).status == PASS
    for run, sha in (('uniform', pinned.UNIFORM_TRAINER_SHA256), ('rl', pinned.RL_TRAINER_SHA256)):
        result = ce._check_trainer_import(ctx, run)
        assert result.status == PASS, (result.summary, result.details.get('probe_stderr'))
        probe = result.details['probe']
        assert probe['trainer_sha256'] == sha
        assert probe['repo'] == str(tmp_config.paths.code_dir(run).resolve())
        assert ('hf_hypernet.settings_policy' in probe['modules']) == (run == 'rl')
        assert ('hf_hypernet.paired_reward' in probe['modules']) == (run == 'rl')
        assert probe['versions']['torch'] == '2.15.0'
    rust = ce.check_rust_library(ctx)            # the staged uniform tree, as on the cluster
    assert rust.status == PASS, rust.summary


def test_trainer_import_reports_the_wrong_platform_library(tmp_config, repo_root):
    if sys.platform.startswith('linux') and os.uname().machine == 'x86_64':
        pytest.skip('the committed library is loadable here')
    stage_code.stage(tmp_config.paths, log=lambda m: None)
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths, repo=repo_root)
    result = ce._check_trainer_import(ctx, 'uniform')
    assert result.status == FAIL and 'ImportError' in result.summary
    assert 'build_rust_utils.sh' in result.remedy


# --- cluster-only checks driven by fakes ----------------------------------------------------------------
class _Proc:
    def __init__(self, stdout='', stderr='', returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_nvidia_smi_check_with_fake_command(monkeypatch):
    monkeypatch.setattr(ce.shutil, 'which', lambda name: '/usr/bin/' + name)

    def run(command, timeout=60):
        return (_Proc(NVIDIA_QUERY) if '--query-gpu=name,memory.total,driver_version,compute_cap' in command
                else _Proc(NVIDIA_PLAIN)), None
    monkeypatch.setattr(ce, '_run', run)
    ctx = ce.Context(args=_args(no_gpu=False))
    result = ce.check_nvidia_smi(ctx)
    assert result.status == PASS and 'NVIDIA H200' in result.summary and 'CUDA 13.0' in result.summary
    assert ctx.facts['nvidia']['driver_cuda'] == '13.0'
    monkeypatch.setattr(ce, '_run', lambda command, timeout=60: (_Proc('NVIDIA A100-SXM4-80GB, 81920, 580.1, 8.0'), None))
    assert ce.check_nvidia_smi(ce.Context(args=_args())).status == WARN
    monkeypatch.setattr(ce, '_run', lambda command, timeout=60: (_Proc('', 'NVIDIA-SMI has failed', 9), None))
    assert ce.check_nvidia_smi(ce.Context(args=_args())).status == FAIL
    monkeypatch.setattr(ce.shutil, 'which', lambda name: None)
    assert ce.check_nvidia_smi(ce.Context(args=_args())).status == FAIL


def _fake_torch(available=True, cuda='13.2', bf16=True):
    import types
    props = types.SimpleNamespace(total_memory=150_000_000_000)
    cuda_mod = types.SimpleNamespace(
        is_available=lambda: available, device_count=lambda: 1, get_device_name=lambda i: 'NVIDIA H200',
        get_device_capability=lambda i: (9, 0), get_device_properties=lambda i: props,
        is_bf16_supported=lambda: bf16)
    backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(is_available=lambda: True, version=lambda: 92600))
    return types.SimpleNamespace(__version__='2.15.0', version=types.SimpleNamespace(cuda=cuda), cuda=cuda_mod,
                                 backends=backends)


@pytest.mark.parametrize('driver,driver_cuda,available,status', [
    ('595.45.04', '13.2', True, PASS),
    ('580.95.05', '13.0', True, WARN),
    ('575.57.08', '12.9', False, FAIL),
    ('575.57.08', '12.9', True, WARN),                  # CUDA works through cuda-compat: a warning, not a failure
])
def test_torch_cuda_check_with_fake_torch(monkeypatch, driver, driver_cuda, available, status):
    monkeypatch.setitem(sys.modules, 'torch', _fake_torch(available=available))
    ctx = ce.Context(args=_args(no_gpu=False))
    ctx.facts['nvidia'] = {'gpus': [{'driver_version': driver}], 'driver_cuda': driver_cuda}
    result = ce.check_torch_cuda(ctx)
    assert result.status == status, result.summary
    assert ctx.facts.get('cuda_ok', False) == available
    if status == FAIL and driver.startswith('575'):
        assert 'no HYAK_TORCH_INDEX_URL fixes this' in result.remedy
    if status == WARN and driver.startswith('575'):
        assert 'forward-compatibility' in result.summary and result.details['forward_compatibility']


def test_gpu_smoke_check_is_skipped_without_cuda():
    assert ce.check_gpu_smoke(ce.Context(args=_args())).status == SKIP


def test_torch_cuda_and_gpu_smoke_work_when_run_alone(monkeypatch):
    """--only torch_cuda / --only gpu_smoke: no earlier check left facts behind (env-3)."""
    monkeypatch.setitem(sys.modules, 'torch', _fake_torch(available=True))
    monkeypatch.setattr(ce.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(ce, '_run', lambda command, timeout=60: (
        _Proc(NVIDIA_QUERY.replace('580.95.05', '595.45.04')) if '--query-gpu=name,memory.total,driver_version,compute_cap'
        in command else _Proc(NVIDIA_PLAIN.replace('13.0', '13.2')), None))
    ctx = ce.Context(args=_args(no_gpu=False))                       # empty facts: torch_cuda asks nvidia-smi itself
    result = ce.check_torch_cuda(ctx)
    assert result.status == PASS and 'driver 595.45.04' in result.summary and ctx.facts['nvidia']['driver_cuda'] == '13.2'
    monkeypatch.setattr(ce, 'gpu_smoke', lambda device='cuda': {'loss': 1.0})
    assert ce.check_gpu_smoke(ce.Context(args=_args(no_gpu=False))).status == PASS    # decides availability itself
    monkeypatch.setitem(sys.modules, 'torch', _fake_torch(available=False))
    skipped = ce.check_gpu_smoke(ce.Context(args=_args(no_gpu=False)))
    assert skipped.status == SKIP and 'not available' in skipped.summary


def test_glibc_check_with_fake_confstr(monkeypatch, tmp_config):
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    monkeypatch.setattr(ce.os, 'confstr', lambda name: 'glibc 2.34', raising=False)
    assert ce.check_glibc(ctx).status == PASS
    monkeypatch.setattr(ce.os, 'confstr', lambda name: 'glibc 2.31', raising=False)
    result = ce.check_glibc(ctx)
    assert result.status == FAIL and 'committed librust_utils.so' in result.summary
    ctx.facts['manifest'] = {'rust_library': {'source': 'rebuilt'}}
    assert ce.check_glibc(ctx).status == PASS
    monkeypatch.setattr(ce.os, 'confstr', lambda name: 'glibc 2.17', raising=False)
    assert ce.check_glibc(ctx).status == FAIL


def test_runtime_env_on_linux(monkeypatch, tmp_config, tmp_path):
    prefix = tmp_path / 'env' / 'train'
    (prefix / 'lib').mkdir(parents=True)
    (prefix / 'lib' / 'libpython3.12.so.1.0').write_bytes(b'')
    link_dir = prefix / 'hyak-libpython'                    # what hyak_activate puts on LD_LIBRARY_PATH
    link_dir.mkdir()
    (link_dir / 'libpython3.12.so.1.0').symlink_to('../lib/libpython3.12.so.1.0')
    monkeypatch.setattr(ce.sys, 'platform', 'linux')
    monkeypatch.setattr(ce.sys, 'prefix', str(prefix))
    monkeypatch.setenv('LD_LIBRARY_PATH', f'{link_dir}:/usr/lib64')
    monkeypatch.setenv('HF_HOME', str(tmp_config.paths.hf_cache))
    import site
    monkeypatch.setattr(site, 'ENABLE_USER_SITE', False)
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    assert ce.check_runtime_env(ctx).status == PASS
    # conda's whole lib directory on the path (the old activation) shadows system libraries: a warning
    monkeypatch.setenv('LD_LIBRARY_PATH', f'{prefix}/lib:/usr/lib64')
    wide = ce.check_runtime_env(ctx)
    assert wide.status == WARN and 'shadow' in wide.summary and 'hyak-libpython' in wide.remedy
    monkeypatch.setenv('LD_LIBRARY_PATH', '/sw/cuda/12.9.1/lib64')
    (prefix / 'lib' / 'libpython3.12.so.1.0').unlink()
    result = ce.check_runtime_env(ctx)
    assert result.status == WARN
    for needle in ('is not on LD_LIBRARY_PATH', 'libpython3.12.so.1.0 is missing', 'CUDA toolkit directories'):
        assert needle in result.summary


def test_freeze_check(monkeypatch, tmp_config):
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    monkeypatch.setattr(ce, '_run', lambda command, timeout=60: (_Proc('torch==2.15.0\nnumpy==2.3.5\n'), None))
    assert ce.check_freeze(ctx).status == WARN                      # no freeze file yet
    freeze = tmp_config.paths.env_freeze('train')
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text('numpy==2.3.5\ntorch==2.15.0\n')
    assert ce.check_freeze(ctx).status == PASS
    freeze.write_text('numpy==2.3.4\ntorch==2.15.0\n')
    result = ce.check_freeze(ctx)
    assert result.status == WARN and result.details['added'] == ['numpy==2.3.5']
    assert result.details['removed'] == ['numpy==2.3.4']


def test_disk_and_resources_checks_run(tmp_config, monkeypatch):
    for name in ('SLURM_JOB_ID', 'SLURM_JOB_END_TIME'):
        monkeypatch.delenv(name, raising=False)      # outside a Slurm job there is no job end to report
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    disk = ce.check_disk_space(ctx)
    assert disk.status in (PASS, WARN, FAIL) and disk.details['runs']['path'] == str(tmp_config.paths.runs)
    assert '60 days' in disk.details['scrubbed_warning']
    assert ce.check_resources(ctx).status == PASS


def test_resources_reports_job_end_time_and_warns_without_it(tmp_config, monkeypatch):
    """Without SLURM_JOB_END_TIME (and no squeue answer) run_segment has no in-process stop timer: a warning only."""
    import time
    from hyak.common import slurm
    ctx = ce.Context(args=_args(), cfg=tmp_config.cfg, paths=tmp_config.paths)
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    monkeypatch.delenv('SLURM_JOB_END_TIME', raising=False)
    real_seconds_remaining = slurm.seconds_remaining
    monkeypatch.setattr(slurm, 'seconds_remaining', lambda *a, **k: None)
    result = ce.check_resources(ctx)
    assert result.status == WARN and result.name == 'resources'
    assert 'SLURM_JOB_END_TIME' in result.summary and '--signal' in result.summary and result.remedy
    assert result.details['SLURM_JOB_END_TIME'] is None and result.details['seconds_remaining'] is None
    assert ce.exit_code([result]) == 0                                  # warnings never fail the check
    monkeypatch.setattr(slurm, 'seconds_remaining', real_seconds_remaining)
    end = str(int(time.time()) + 3600)
    monkeypatch.setenv('SLURM_JOB_END_TIME', end)
    result = ce.check_resources(ctx)
    assert result.status == PASS and result.details['SLURM_JOB_END_TIME'] == end
    assert 3000 < result.details['seconds_remaining'] <= 3600 and 'job time left' in result.summary


# --- the model hash, directory fsync and the monitor directory (env-1, env-v1, env-12) -----------------------------
def _stand_in_model(tmp_config, scratch_assets, monkeypatch, *, flip=False):
    """A model directory whose model.safetensors is a small stand-in with its own pin (optionally one byte off)."""
    import hashlib
    from hyak.model import fetch_model
    weights = b'stand-in safetensors bytes ' * 200
    expected = {'config.json': fetch_model.expected_files()['config.json'],
                'tokenizer.json': fetch_model.expected_files()['tokenizer.json'],
                'model.safetensors': {'bytes': len(weights), 'sha256': hashlib.sha256(weights).hexdigest()}}
    monkeypatch.setattr(fetch_model, 'expected_files', lambda repo=None: dict(expected))
    dest = tmp_config.paths.model_dir
    dest.mkdir(parents=True)
    for name in ('config.json', 'tokenizer.json'):
        shutil.copyfile(scratch_assets / 'full' / name, dest / name)
    data = bytearray(weights)
    if flip:
        data[len(data) // 2] ^= 0x01                         # same size, one bit different
    (dest / 'model.safetensors').write_bytes(bytes(data))


def test_model_files_are_hashed_by_default(tmp_config, scratch_assets, monkeypatch, capsys):
    _stand_in_model(tmp_config, scratch_assets, monkeypatch, flip=True)
    code = ce.main(['--only', 'model_files', '--no-network', '--no-gpu', '--no-status'])
    out = capsys.readouterr().out
    assert code == 4 and '[FAIL] model_files' in out and 'sha256' in out       # the default hashes the model
    assert ce.main(['--only', 'model_files', '--no-hash-model', '--no-status']) == 0      # sizes only
    assert '(--no-hash-model)' in capsys.readouterr().out


def test_model_files_default_passes_for_the_real_bytes(tmp_config, scratch_assets, monkeypatch, capsys):
    _stand_in_model(tmp_config, scratch_assets, monkeypatch)
    assert ce.main(['--only', 'model_files', '--no-network', '--no-gpu', '--no-status']) == 0
    assert 'bytes and SHA-256' in capsys.readouterr().out


def _refuse_directory_fsync(monkeypatch):
    import errno
    import stat
    real = ce.os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, 'Invalid argument')
        return real(fd)
    monkeypatch.setattr(ce.os, 'fsync', fsync)


def test_a_refused_directory_fsync_is_only_a_warning(tmp_path, tmp_config, monkeypatch, capsys):
    _refuse_directory_fsync(monkeypatch)
    facts = ce.filesystem_semantics(tmp_path)
    assert facts['fsync_file'] is True and facts['fsync_dir'] is False and facts['fsync'] is False
    result = ce.filesystem_verdict('filesystem_runs', 'ANYTOK_RUNS', facts)
    assert result.status == WARN and 'directory fsync is not supported here' in result.summary
    # the program's exit status stays 0: the port's writers ignore that error (hyak.common.io.fsync_dir)
    assert ce.main(['--only', 'filesystem_root', '--no-network', '--no-gpu', '--no-status']) == 0
    assert '[WARN] filesystem_root' in capsys.readouterr().out
    # a failing file fsync is still fatal
    assert ce.filesystem_verdict('filesystem_runs', 'ANYTOK_RUNS', {**facts, 'fsync_file': False}).status == FAIL


def test_the_monitor_directory_is_probed_and_its_trouble_is_a_warning(tmp_config, capsys):
    assert ce.main(['--only', 'filesystem_monitor', '--no-network', '--no-gpu', '--no-status']) == 0
    assert '[PASS] filesystem_monitor' in capsys.readouterr().out
    facts = {'base': '/x', 'flock_excludes': False, 'mkdir_exclusive': False, 'replace_file': True,
             'replace_dir': True, 'fsync_file': True, 'fsync_dir': True}
    assert ce.filesystem_verdict('filesystem_monitor', 'the monitor directory', facts,
                                 ce.MONITOR_FS_FACTS).status == WARN         # a stuck lock is given up by the jobs
    assert ce.filesystem_verdict('filesystem_monitor', 'the monitor directory', {**facts, 'replace_file': False},
                                 ce.MONITOR_FS_FACTS).status == FAIL         # status files need atomic replace
    assert ce.filesystem_verdict('filesystem_runs', 'ANYTOK_RUNS', facts).status == FAIL
