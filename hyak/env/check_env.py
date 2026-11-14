"""Environment check for the Hyak port: run it on a GPU node before anything long (DESIGN.md 5.A, D10).

    python -m hyak.env.check_env                     # in 00_check_env.sbatch (debug QoS, one H200)
    python -m hyak.env.check_env --no-gpu            # login node: everything except the GPU checks
    python -m hyak.env.check_env --only rust_library --rust-library PATH --no-status
    python -m hyak.env.check_env --list

Every check is a small function returning a :class:`CheckResult` (``pass``/``warn``/``fail``/
``skip``) with a one-line summary and, when it is not a pass, a one-line remedy.  The program
prints one line per check, writes a JSON report (``check_env_report.json`` and
``check_env_report-<job>.json`` in ``modal_results/hyak_tillicum/check-env/``, plus
``$ANYTOK_ROOT/env/check_env_report.json``), updates the stage status files, and exits 4 if
any check failed (0 otherwise; warnings do not fail; 2 for a usage error, 1 only for a crash of
this program, with a traceback).  This is the project's exit-code contract (see hyak/slurm/job.sh).

What is checked (names for ``--only`` / ``--skip``):

* ``config``, ``python`` (3.12), ``packages`` (the six exact pins, stock ``tokenizers`` -- the
  patched pool wheel must not be in the train environment, ``pytest``), ``runtime_env``
  (LD_LIBRARY_PATH holding the libpython3.12 link for the Rust library, HF_HOME, user site, CUDA
  modules),
  ``glibc`` (>= 2.34 for the committed ``librust_utils.so``);
* ``nvidia_smi`` (GPU name, memory, driver, the driver's CUDA version), ``torch_cuda``
  (``torch.version.cuda``, ``torch.cuda.is_available()``, driver versus CUDA runtime rule; a driver
  below the minimum is a warning, not a failure, when CUDA initialises anyway, which is what the
  forward-compatibility libraries ``cuda-compat-13-2`` do), ``gpu_smoke`` (bf16 matmul,
  bf16-autocast Linear + LayerNorm, joint 0.1 clip, fused AdamW step; it tests CUDA itself, so
  ``--only gpu_smoke`` works);
* ``code_stage`` (``hyak.code.stage_code`` verification), ``rust_library`` (imports the staged
  ``librust_utils.so`` exactly as the trainer does -- ``train_joint_smollm_input.py:22-30`` -- then
  runs the trainer's own ``prime_sampler`` and ``draw_tokenizer`` on synthetic text),
  ``trainer_import_uniform`` / ``trainer_import_rl`` (imports each staged trainer module the way
  the Modal launchers did and checks where every module came from);
* ``filesystem_runs`` / ``filesystem_root`` (``fcntl.flock`` exclusion between two processes as
  the trainer's ``.run.lock`` needs, ``mkdir`` claims, ``os.replace`` of files and of the trainer's
  ``.writing-*`` directory, file fsync, write/read throughput; a directory fsync that the file system
  refuses is a warning only, every writer of the port ignores that error), ``filesystem_monitor``
  (the same probe on the checkout's ``modal_results`` directory, where the status files and the
  dashboard registry lock live; trouble there is a warning, jobs time out a stuck lock),
  ``disk_space``, ``model_files`` (bytes and SHA-256 of the 3.4 GB model by default; ``--no-hash-model``
  compares bytes only),
  ``network`` (HTTPS to huggingface.co and the package indexes; warnings only), ``freeze``
  (``pip freeze`` against ``env/train.freeze.txt``), ``resources`` (CPUs and memory visible; inside a Slurm
job also SLURM_JOB_END_TIME and the seconds left, with a warning when neither is known: the training
wrapper would then have no in-process stop timer and rely on Slurm's --signal alone).

The torch rule (verified 2026-11-13 on pypi.org and download.pytorch.org): ``torch==2.15.0``
exists for Linux x86-64 only as CUDA 13 builds -- PyPI default CUDA 13.2 (what the Modal images
installed), ``https://download.pytorch.org/whl/cu132`` and ``.../cu134``; there is no CUDA 12
build.  CUDA 13 needs an NVIDIA driver >= 580 (CUDA release notes, minor version compatibility
table; driver branch R595 is native for 13.2).  A driver below 580 cannot run any torch 2.15.0
CUDA build: no ``HYAK_TORCH_INDEX_URL`` helps; see the remedy text of ``torch_cuda``.

Runs on Python >= 3.9 (so a wrong interpreter is reported, not a syntax error); the GPU and
probe checks need the train environment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Optional

PASS, WARN, FAIL, SKIP = 'pass', 'warn', 'fail', 'skip'
STATUSES = (PASS, WARN, FAIL, SKIP)
STAGE = 'check-env'
PROBE_MARKER = 'HYAK_PROBE_RESULT '

#: Minimum NVIDIA driver for each CUDA major version under minor version compatibility
#: (CUDA Toolkit release notes, "CUDA Toolkit and Minimum Required Driver Version for CUDA
#: Minor Version Compatibility": 12.x >= 525.60.13 (12.0 GA), 13.x >= 580).
MIN_DRIVER_FOR_CUDA_MAJOR = {12: (525, 60, 13), 13: (580, 0, 0)}
#: torch==2.15.0 CUDA builds that exist for cp312 Linux x86-64 (pypi.org JSON API and
#: download.pytorch.org/whl/torch/, checked 2026-11-13): local label -> CUDA runtime.
TORCH_CUDA_BUILDS = {'': '13.2', 'cu132': '13.2', 'cu134': '13.4'}
GLIBC_FOR_COMMITTED_LIBRARY = (2, 34)
GLIBC_FOR_TORCH = (2, 28)   # manylinux_2_28 wheels
#: Free space thresholds (GB = 1e9 bytes).  Two runs at the original retention need about
#: 2 x (493 GB grid + 23.5 GB transient restart) (NOTES/modal-launchers.md 3.3).
RUNS_FREE_WARN_GB = 1100
RUNS_FREE_FAIL_GB = 30       # less than one full restart (23.5 GB) plus slack
ROOT_FREE_WARN_GB = 150
NETWORK_ENDPOINTS = (
    ('huggingface.co', 'https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-1.7B/revision/'
                       'effd688a12921b4cc83e3312b6feb579f70f9c71', True),
    ('pypi.org', 'https://pypi.org/simple/pip/', False),
    ('index.crates.io', 'https://index.crates.io/config.json', False),
    ('github.com', 'https://github.com/huggingface/tokenizers', False),
)


@dataclass
class CheckResult:
    """Outcome of one check."""

    name: str
    status: str
    summary: str
    remedy: Optional[str] = None
    details: dict = field(default_factory=dict)
    seconds: float = 0.0

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f'unknown check status {self.status!r}')
        if self.status in (WARN, FAIL) and not (isinstance(self.remedy, str) and self.remedy):
            raise ValueError(f'check {self.name}: a {self.status} needs a one-line remedy string')
        if self.status in (PASS, SKIP) and self.remedy is not None:
            raise ValueError(f'check {self.name}: a {self.status} carries no remedy (details go in details=)')
        if not isinstance(self.details, dict):
            raise ValueError(f'check {self.name}: details must be a dict')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Context:
    """Shared state: parsed arguments, config, paths and facts found by earlier checks."""

    args: argparse.Namespace
    cfg: Optional[dict] = None
    paths: object = None
    repo: Optional[Path] = None
    config_error: Optional[str] = None
    facts: dict = field(default_factory=dict)


# =============================================================================================
# Pure helpers (unit-tested on any machine)
# =============================================================================================
def parse_version(text) -> tuple:
    """Leading numeric components of a version string: ``'580.95.05'`` -> ``(580, 95, 5)``; ``()`` if none."""
    match = re.match(r'\s*(\d+(?:\.\d+)*)', str(text or ''))
    return tuple(int(part) for part in match.group(1).split('.')) if match else ()


def parse_glibc(confstr_value) -> tuple:
    """``os.confstr('CS_GNU_LIBC_VERSION')`` (``'glibc 2.34'``) -> ``(2, 34)``; ``()`` if not glibc."""
    match = re.fullmatch(r'\s*glibc\s+(\d+(?:\.\d+)*)\s*', str(confstr_value or ''))
    return parse_version(match.group(1)) if match else ()


def parse_nvidia_query(text) -> list:
    """Rows of ``nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits``."""
    rows = []
    for line in str(text).splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            raise ValueError(f'unexpected nvidia-smi line: {line!r}')
        row = {'name': parts[0], 'memory_total_mib': None, 'driver_version': parts[2],
               'compute_cap': parts[3] if len(parts) > 3 else None}
        try:
            row['memory_total_mib'] = int(float(parts[1]))
        except ValueError:
            pass
        rows.append(row)
    return rows


def parse_driver_cuda(text) -> Optional[str]:
    """The ``CUDA Version: X.Y`` of plain ``nvidia-smi`` output (the newest CUDA the driver supports)."""
    match = re.search(r'CUDA Version\s*:\s*(\d+\.\d+)', str(text))
    return match.group(1) if match else None


def torch_build_label(version) -> str:
    """Local version label of a torch version: ``'2.15.0+cu134'`` -> ``'cu134'``; PyPI's ``'2.15.0'`` -> ``''``."""
    version = str(version or '')
    return version.split('+', 1)[1] if '+' in version else ''


def expected_torch_label(index_url) -> str:
    """Local label ``setup_train_env.sh`` installs for ``HYAK_TORCH_INDEX_URL`` (empty URL = PyPI = ``''``)."""
    url = str(index_url or '').strip().rstrip('/')
    return url.rsplit('/', 1)[-1] if url else ''


def cuda_driver_verdict(driver_version, driver_cuda, torch_cuda):
    """``(status, summary, remedy)`` for an NVIDIA driver against torch's CUDA runtime.

    Rule (CUDA minor version compatibility): the driver must be at least the minimum of the
    runtime's major version (13.x: >= 580, 12.x: >= 525.60.13).  A driver whose own CUDA
    version is older than the runtime but of the same major runs under minor version
    compatibility (warning; the GPU smoke test is the proof).
    """
    if not torch_cuda:
        return (FAIL, 'torch is a CPU-only build (torch.version.cuda is None)',
                'reinstall the train environment: bash hyak/env/setup_train_env.sh --recreate '
                '(leave HYAK_TORCH_INDEX_URL empty for the PyPI CUDA 13.2 build)')
    runtime = parse_version(torch_cuda)
    driver = parse_version(driver_version)
    if not driver:
        return (FAIL, f'cannot read the NVIDIA driver version ({driver_version!r})',
                'run on a GPU node (submit hyak/slurm/submit.sh check-env) and check nvidia-smi')
    major = runtime[0] if runtime else None
    minimum = MIN_DRIVER_FOR_CUDA_MAJOR.get(major)
    if minimum is None:
        return (WARN, f'no driver rule known for CUDA {torch_cuda}; driver {driver_version}',
                'compare the driver with the CUDA release notes by hand')
    shown = '.'.join(str(p) for p in minimum)
    if driver < minimum:
        remedy = (f'CUDA {torch_cuda} needs an NVIDIA driver >= {shown}; this node has {driver_version}. ')
        if major == 13:
            remedy += ('torch==2.15.0 exists only as CUDA 13 builds (PyPI default 13.2; download.pytorch.org cu132, '
                       'cu134), so no HYAK_TORCH_INDEX_URL fixes this. Ask Hyak support for H200 nodes with driver '
                       '>= 580 or the CUDA forward-compatibility libraries (cuda-compat-13-2 on LD_LIBRARY_PATH; '
                       'Data Center GPUs only). Changing the torch version would be a material deviation that '
                       'needs the user\'s approval.')
        else:
            remedy += 'use a node with a newer driver.'
        return (FAIL, f'driver {driver_version} is too old for torch CUDA {torch_cuda} (minimum {shown})', remedy)
    if driver_cuda and parse_version(driver_cuda) < runtime[:2]:
        return (WARN, f'driver {driver_version} supports CUDA {driver_cuda} < torch runtime {torch_cuda}: running '
                      'under CUDA minor version compatibility',
                'nothing to change if gpu_smoke passes (torch ships sm_90 code for the H200, no PTX JIT needed); '
                'a driver of the native branch removes the caveat')
    return (PASS, f'driver {driver_version} (CUDA {driver_cuda or "?"}) runs torch CUDA {torch_cuda}', None)


def driver_below_minimum(driver_version, torch_cuda) -> bool:
    """True when the driver is older than the minimum of the CUDA major version of ``torch_cuda``."""
    runtime = parse_version(torch_cuda)
    driver = parse_version(driver_version)
    minimum = MIN_DRIVER_FOR_CUDA_MAJOR.get(runtime[0] if runtime else None)
    return bool(driver and minimum and driver < minimum)


def parse_ldd(text) -> dict:
    """``ldd`` output -> ``{"not_found": [...], "version_errors": [...], "libraries": {soname: path}}``."""
    not_found, version_errors, libraries = [], [], {}
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if 'version `' in line and 'not found' in line:
            version_errors.append(line)
            continue
        match = re.match(r'(\S+)\s+=>\s+not found', line)
        if match:
            not_found.append(match.group(1))
            continue
        match = re.match(r'(\S+)\s+=>\s+(\S+)', line)
        if match:
            libraries[match.group(1)] = match.group(2)
    return {'not_found': not_found, 'version_errors': version_errors, 'libraries': libraries}


def binary_info(path) -> dict:
    """``{"format": "ELF"|"Mach-O"|"unknown", "bits", "machine"}`` from a binary's header."""
    try:
        with open(path, 'rb') as stream:
            head = stream.read(64)
    except OSError as exc:
        return {'format': 'unreadable', 'error': str(exc)}
    if head[:4] == b'\x7fELF' and len(head) >= 20:
        order = 'little' if head[5] == 1 else 'big'
        machine = int.from_bytes(head[18:20], order)
        names = {0x3E: 'x86_64', 0xB7: 'aarch64', 0x03: 'i386', 0x28: 'arm'}
        return {'format': 'ELF', 'bits': 64 if head[4] == 2 else 32, 'machine': names.get(machine, hex(machine))}
    if head[:4] in (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe'):
        return {'format': 'Mach-O', 'bits': 64 if head[:4] in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf') else None,
                'machine': None}
    return {'format': 'unknown', 'bits': None, 'machine': None}


def host_binary_expectation() -> dict:
    """The binary format the running interpreter can load (``ELF``/``x86_64`` on Tillicum)."""
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    if sys.platform.startswith('linux'):
        return {'format': 'ELF', 'machine': machine}
    if sys.platform == 'darwin':
        return {'format': 'Mach-O', 'machine': None}
    return {'format': None, 'machine': None}


def package_problems(installed: dict, pins) -> list:
    """Problems of ``{name: version or None}`` against ``((name, version), ...)``; local labels are ignored."""
    problems = []
    for name, wanted in pins:
        have = installed.get(name)
        if have is None:
            problems.append(f'{name} is not installed (need {wanted})')
        elif str(have).split('+', 1)[0] != wanted:
            problems.append(f'{name} {have} installed, need {wanted}')
    return problems


def parse_probe_output(stdout) -> Optional[dict]:
    """The JSON object a probe printed after ``HYAK_PROBE_RESULT`` (last such line), or None."""
    found = None
    for line in str(stdout).splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                value = json.loads(line[len(PROBE_MARKER):])
            except ValueError:
                continue
            if isinstance(value, dict):
                found = value
    return found


def rust_failure_remedy(error_text, ldd=None) -> str:
    """Remedy for a failed import of librust_utils.so, from the error text and ``ldd`` facts."""
    text = str(error_text or '')
    ldd = ldd or {}
    missing = list(ldd.get('not_found') or [])
    if any('libpython' in name for name in missing) or 'libpython3' in text:
        return ('libpython3.12.so.1.0 is not found: run inside "hyak_activate train" (it puts '
                '$CONDA_PREFIX/hyak-libpython, a directory holding only the link to it, on LD_LIBRARY_PATH); if '
                '$ANYTOK_ROOT/env/train/lib has no libpython3.12.so.1.0, rebuild with '
                'bash hyak/env/build_rust_utils.sh (the extension-module build needs no libpython)')
    if ldd.get('version_errors') or 'GLIBC_' in text:
        return ('the committed library needs glibc >= 2.34; rebuild it on this cluster: '
                'bash hyak/env/build_rust_utils.sh')
    if 'Build ZeTT' in text or 'FileNotFoundError' in text:
        return 'stage the code first: python -m hyak.code.stage_code'
    if any(marker in text for marker in ('invalid ELF header', 'wrong ELF class', 'not valid mach-o',
                                         'cannot open shared object file: Exec format error')):
        return ('the library was built for another platform than this node; rebuild it here: '
                'bash hyak/env/build_rust_utils.sh')
    if missing:
        return f'shared libraries not found: {", ".join(missing)}; rebuild with bash hyak/env/build_rust_utils.sh'
    if 'crashed with signal' in text:
        return ('the library crashed the interpreter (a library linked against a second libpython can do this); '
                'rebuild the extension-module variant, which links no libpython: bash hyak/env/build_rust_utils.sh')
    if 'PyInit_rust_utils' in text:
        return 'the file is not a rust_utils extension module; rebuild with bash hyak/env/build_rust_utils.sh'
    return 'see the error above; rebuild with bash hyak/env/build_rust_utils.sh (fetch on the login node)'


#: Exit status when a check failed (the project's exit-code contract: 4 = a check failed; 1 is reserved
#: for a crash, 2 for a usage error).
EXIT_CHECK_FAILED = 4


def exit_code(results) -> int:
    """4 (a check failed) if any check failed, else 0."""
    return EXIT_CHECK_FAILED if any(r.status == FAIL for r in results) else 0


def render(result: CheckResult) -> str:
    """One printed line (two for warnings and failures: the remedy follows)."""
    line = f'[{result.status.upper():4}] {result.name}: {result.summary}'
    if result.remedy and result.status in (WARN, FAIL):
        line += f'\n       remedy: {result.remedy}'
    return line


def summary_line(results) -> str:
    counts = {status: sum(r.status == status for r in results) for status in STATUSES}
    verdict = 'FAILED' if counts[FAIL] else 'OK'
    return (f'check_env {verdict}: {counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed, '
            f'{counts[SKIP]} skipped')


# =============================================================================================
# Subprocess probes (run with the same interpreter, isolated: -I -B)
# =============================================================================================
#: Synthetic, deterministic text batches of the trainer's queue shape (QUEUE_BATCH_SIZE texts of
#: 2,048 characters; train_joint_smollm_input.py:478-482) for the sampler exercise.
_SYNTHETIC = r'''
def synthetic_batches(batches=2, texts=128, chars=2048, seed=20261113):
    import random
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(batches):
        batch = []
        for _ in range(texts):
            words = []
            length = 0
            while length < chars:
                word = "".join(rng.choice(letters) for _ in range(rng.randint(2, 11)))
                if rng.random() < 0.08:
                    word += rng.choice(".,;:!?")
                if rng.random() < 0.05:
                    word = str(rng.randint(0, 99999))
                words.append(word)
                length += len(word) + 1
            batch.append(" ".join(words)[:chars])
        out.append(batch)
    return out
'''

RUST_PROBE = _SYNTHETIC + r'''
import importlib.util, json, sys, time, traceback
from pathlib import Path

args = json.loads(sys.argv[1])
result = {"ok": False, "steps": []}
START = time.time()

def step(name, **info):
    result["steps"].append({"step": name, "seconds": round(time.time() - START, 3), **info})

try:
    # --- runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py:22-30, verbatim, except that REPO
    # --- is the staged code root passed in instead of Path(__file__).resolve().parents[1].
    REPO = Path(args["code_root"]).resolve()
    sys.path.insert(0, str(REPO))
    rust_library = REPO / "rust_utils" / "target" / "release" / "librust_utils.so"
    if not rust_library.is_file():
        raise FileNotFoundError(f"Build ZeTT's Rust sampler first: {rust_library}")
    rust_spec = importlib.util.spec_from_file_location("rust_utils", rust_library)
    rust_utils = importlib.util.module_from_spec(rust_spec)
    sys.modules["rust_utils"] = rust_utils
    rust_spec.loader.exec_module(rust_utils)
    # --- end of the trainer's loader block.
    result["version"] = getattr(rust_utils, "__version__", None)
    step("loaded", loader=type(rust_spec.loader).__name__, file=str(rust_library))
    # Raw call shapes of rust_utils/src/lib.rs:40-62 (map, seed_size, max_length, stride, noise_std,
    # pop_prev, push_current): pop_prev=False only fills the queue and returns [].
    raw = rust_utils.TokenizerSampler()
    primed = raw.sample_tokenizer({"hello world": 1}, 30000, 16, 4, 0.0, False, True)
    if primed != []:
        raise AssertionError(f"priming call returned {len(primed)} rows, expected []")
    drawn = raw.sample_tokenizer({"hello again world": 1}, 1000, 16, 4, 0.0, True, True)
    if not drawn or not all(isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str)
                            and isinstance(r[1], float) for r in drawn):
        raise AssertionError("drawing call did not return a list of (str, float)")
    step("raw_calls", rows=len(drawn))
    # The trainer's fresh-start use (train_joint_smollm_input.py:474-497) through the pinned helpers.
    import numpy as np
    import hf_hypernet.joint_training as jt
    if jt.rust_utils is not rust_utils:
        raise AssertionError("hf_hypernet.joint_training did not get the registered rust_utils module")
    batches = synthetic_batches()
    sampler = rust_utils.TokenizerSampler()
    jt.prime_sampler(sampler, batches)
    eval_sampler = rust_utils.TokenizerSampler()
    jt.prime_sampler(eval_sampler, batches)
    step("prime_sampler", batches=len(batches))
    tokenizer, vocab, noise = jt.draw_tokenizer(eval_sampler, batches[0], np.random.default_rng(42), "<|endoftext|>")
    if len(vocab) != jt.VOCAB_SIZE or tuple(vocab[0]) != ("<|endoftext|>", 0.0):
        raise AssertionError(f"draw_tokenizer gave {len(vocab)} rows starting {vocab[:1]}")
    sample = batches[0][0][:300]
    ids = tokenizer.encode(sample, add_special_tokens=False).ids
    if tokenizer.decode(ids) != sample:
        raise AssertionError("the drawn tokenizer does not round-trip text")
    step("draw_tokenizer", vocab=len(vocab), noise=noise, sample_tokens=len(ids))
    result["ok"] = True
except BaseException as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-6000:]
result["seconds"] = round(time.time() - START, 3)
print("HYAK_PROBE_RESULT " + json.dumps(result), flush=True)
'''

TRAINER_PROBE = r'''
import hashlib, json, sys, time, traceback
from pathlib import Path

args = json.loads(sys.argv[1])
result = {"ok": False}
START = time.time()
try:
    code_root = Path(args["code_root"]).resolve()
    # As the launchers did: scripts/modal_anytok_pool_20k.py:267-268, scripts/modal_anytok_pool_rl_20k.py:190-191.
    sys.path.insert(0, str(code_root))
    import scripts.train_joint_smollm_input as trainer
    trainer_file = Path(trainer.__file__).resolve()
    result["trainer_file"] = str(trainer_file)
    result["trainer_sha256"] = hashlib.sha256(trainer_file.read_bytes()).hexdigest()
    result["repo"] = str(trainer.REPO)
    if Path(trainer.REPO) != code_root:
        raise AssertionError(f"trainer.REPO is {trainer.REPO}, expected the staged tree {code_root}")
    if result["trainer_sha256"] != args["trainer_sha256"]:
        raise AssertionError(f"imported trainer sha256 {result['trainer_sha256']} is not the pinned {args['trainer_sha256']}")
    for name in ("main", "save_checkpoint", "record", "ZettHypernet", "rust_utils"):
        if not hasattr(trainer, name):
            raise AssertionError(f"trainer module has no attribute {name}")
    rust = sys.modules.get("rust_utils")
    result["rust_utils_file"] = getattr(rust, "__file__", None)
    result["rust_utils_version"] = getattr(rust, "__version__", None)
    if rust is None or Path(rust.__file__).resolve() != code_root / "rust_utils/target/release/librust_utils.so":
        raise AssertionError(f"rust_utils was loaded from {getattr(rust, '__file__', None)}")
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name == "hf_hypernet" or name.startswith("hf_hypernet.") or name == "scripts.train_joint_smollm_input":
            location = getattr(module, "__file__", None)
            modules[name] = location
            if location and not Path(location).resolve().is_relative_to(code_root):
                raise AssertionError(f"{name} was imported from {location}, outside {code_root}")
    result["modules"] = modules
    if args["run"] == "rl" and "hf_hypernet.settings_policy" not in sys.modules:
        raise AssertionError("the RL trainer did not import hf_hypernet.settings_policy")
    if args["run"] == "rl" and "hf_hypernet.paired_reward" not in sys.modules:
        raise AssertionError("the RL trainer did not import hf_hypernet.paired_reward")
    import importlib.metadata as md
    result["versions"] = {p: md.version(p) for p in ("torch", "transformers", "tokenizers", "numpy",
                                                     "safetensors", "accelerate")}
    result["ok"] = True
except BaseException as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-6000:]
result["seconds"] = round(time.time() - START, 3)
print("HYAK_PROBE_RESULT " + json.dumps(result), flush=True)
'''

_FLOCK_CHILD = r'''
import fcntl, sys
handle = open(sys.argv[1], "a+")
try:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("blocked")
except OSError as exc:
    print("error " + repr(exc))
else:
    print("acquired")
'''


def run_probe(script, args: dict, *, timeout: float, python=None) -> dict:
    """Run a probe script in a child interpreter; ``{"result": dict|None, "returncode", "stdout", "stderr", "error"}``."""
    command = [python or sys.executable, '-I', '-B', '-c', script, json.dumps(args)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=timeout,
                              cwd=tempfile.gettempdir(), check=False)
    except subprocess.TimeoutExpired as exc:
        return {'result': None, 'returncode': None, 'stdout': _tail(exc.stdout), 'stderr': _tail(exc.stderr),
                'error': f'probe timed out after {timeout:.0f} s'}
    except OSError as exc:
        return {'result': None, 'returncode': None, 'stdout': '', 'stderr': '', 'error': f'cannot run probe: {exc}'}
    result = parse_probe_output(proc.stdout)
    error = None
    if result is None:
        error = (f'probe crashed with signal {-proc.returncode}' if proc.returncode < 0
                 else f'probe exited {proc.returncode} without a result')
    return {'result': result, 'returncode': proc.returncode, 'stdout': _tail(proc.stdout),
            'stderr': _tail(proc.stderr), 'error': error}


def _tail(text, limit=4000):
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    return text[-limit:]


def _run(command, timeout=60):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    return proc, None


# =============================================================================================
# Checks
# =============================================================================================
def check_config(ctx: Context) -> CheckResult:
    if ctx.cfg is None:
        return CheckResult('config', FAIL, f'configuration not loaded: {ctx.config_error}',
                           'cp hyak/config.env.example hyak/config.env and set ANYTOK_ROOT and ANYTOK_RUNS')
    from hyak.common.config import redacted
    details = {k: v for k, v in redacted(ctx.cfg).items() if not k.startswith('HYAK_') or v}
    return CheckResult('config', PASS, f'ANYTOK_ROOT={ctx.cfg["ANYTOK_ROOT"]} ANYTOK_RUNS={ctx.cfg["ANYTOK_RUNS"]}',
                       details=details)


def check_python(ctx: Context) -> CheckResult:
    from hyak.code.pinned import PYTHON_VERSION
    details = {'version': sys.version, 'executable': sys.executable, 'prefix': sys.prefix}
    wanted = '.'.join(map(str, PYTHON_VERSION))
    if tuple(sys.version_info[:2]) != PYTHON_VERSION:
        return CheckResult('python', FAIL, f'Python {platform.python_version()}, need {wanted}.x',
                           'run inside the train environment (hyak_activate train) or create it with '
                           'bash hyak/env/setup_train_env.sh', details)
    expected = str(ctx.paths.env_train) if ctx.paths is not None else None
    if expected and not _same_path(sys.prefix, expected):
        return CheckResult('python', WARN, f'Python {platform.python_version()} at {sys.prefix}, not the train env '
                           f'{expected}', 'activate the train environment: hyak_activate train', details)
    return CheckResult('python', PASS, f'Python {platform.python_version()} ({sys.executable})', details=details)


def _same_path(a, b) -> bool:
    """Equal after resolving symlinks (GPFS mount points are often links); empty never matches."""
    if not a or not b:
        return False
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def installed_versions(names) -> dict:
    import importlib.metadata as md
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def _quiet_stdout(function):
    """Call ``function`` with file descriptor 1 sent to /dev/null (stock tokenizers prints
    "Ignored unknown kwargs option ..." from Rust, bypassing sys.stdout)."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        return function()
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def tokenizers_is_patched() -> Optional[bool]:
    """The probe of scripts/tokenizer_pool.py:154-157: does BpeTrainer keep ``noise_per_million``?"""
    from tokenizers import trainers

    def probe():
        trainer = trainers.BpeTrainer(vocab_size=300, noise_per_million=12345, noise_seed=1,
                                      random_from_all=True, show_progress=False)
        return trainer.__getstate__()
    state = _quiet_stdout(probe)
    return json.loads(state)['BpeTrainer'].get('noise_per_million') == 12345


def check_packages(ctx: Context) -> CheckResult:
    from hyak.code.pinned import TRAIN_PINS, TRAIN_TEST_PINS
    names = [n for n, _ in TRAIN_PINS + TRAIN_TEST_PINS] + ['huggingface-hub', 'pip']
    installed = installed_versions(names)
    details = {'installed': installed}
    problems = package_problems(installed, TRAIN_PINS)
    if problems:
        return CheckResult('packages', FAIL, '; '.join(problems),
                           'install the exact pins: bash hyak/env/setup_train_env.sh', details)
    label = torch_build_label(installed['torch'])
    wanted_label = expected_torch_label((ctx.cfg or {}).get('HYAK_TORCH_INDEX_URL', ''))
    details['torch_build_label'] = label
    details['torch_cuda_runtime_expected'] = TORCH_CUDA_BUILDS.get(label)
    try:
        patched = tokenizers_is_patched()
    except Exception as exc:  # noqa: BLE001 - a broken tokenizers import is a failure, reported as such
        return CheckResult('packages', FAIL, f'tokenizers probe failed: {type(exc).__name__}: {exc}',
                           'reinstall: bash hyak/env/setup_train_env.sh --recreate', details)
    details['tokenizers_patched'] = patched
    if patched:
        return CheckResult('packages', FAIL, 'the patched tokenizers wheel (noise trainer) is installed in this '
                           'environment; training and scoring use stock tokenizers==0.22.2',
                           'the patched wheel belongs only to the data env: bash hyak/env/setup_train_env.sh --recreate',
                           details)
    if ctx.cfg is not None and label != wanted_label:
        return CheckResult('packages', WARN, f'torch build "{label or "pypi"}" installed but HYAK_TORCH_INDEX_URL asks '
                           f'for "{wanted_label or "pypi"}"', 'bash hyak/env/setup_train_env.sh --recreate', details)
    missing_tools = package_problems(installed, TRAIN_TEST_PINS)
    if missing_tools:
        return CheckResult('packages', WARN, f'six pins OK; {"; ".join(missing_tools)}',
                           'bash hyak/env/setup_train_env.sh (installs pytest for the test suite)', details)
    return CheckResult('packages', PASS, 'six pins OK (torch ' + installed['torch'] + '), stock tokenizers',
                       details=details)


#: Directory inside the conda prefix that holds only the link to libpython (what hyak_activate puts on
#: LD_LIBRARY_PATH instead of conda's whole lib directory).
LIBPYTHON_DIRNAME = 'hyak-libpython'


def check_runtime_env(ctx: Context) -> CheckResult:
    import site
    prefix = Path(sys.prefix)
    lib = prefix / 'lib'
    link_dir = prefix / LIBPYTHON_DIRNAME
    libpython = lib / 'libpython3.12.so.1.0'
    ld_path = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    details = {'LD_LIBRARY_PATH': ld_path, 'libpython': str(libpython), 'libpython_exists': libpython.is_file(),
               'libpython_link_dir': str(link_dir),
               'HF_HOME': os.environ.get('HF_HOME'), 'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM'),
               'PYTHONNOUSERSITE': os.environ.get('PYTHONNOUSERSITE'), 'user_site_enabled': bool(site.ENABLE_USER_SITE),
               'HYAK_STAGE': os.environ.get('HYAK_STAGE'), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}
    warnings, remedies = [], []
    if sys.platform.startswith('linux'):
        has_link = any(_same_path(link_dir, p) for p in ld_path) and (link_dir / libpython.name).exists()
        wide = any(_same_path(lib, p) for p in ld_path)
        if not has_link and not wide:
            warnings.append(f'{link_dir} (the libpython link) is not on LD_LIBRARY_PATH')
            remedies.append('run inside "hyak_activate train" (lib.sh), which creates and exports it')
        elif wide and not has_link:
            warnings.append(f'conda\'s whole {lib} is on LD_LIBRARY_PATH: its libz/libstdc++ shadow the system\'s '
                            'for every program of the job')
            remedies.append(f'use {link_dir}, which holds only the libpython link (hyak_activate does)')
        if not libpython.is_file():
            warnings.append(f'{libpython} is missing (the committed librust_utils.so links against it)')
            remedies.append('rebuild the library without that dependency: bash hyak/env/build_rust_utils.sh')
    cuda_dirs = [p for p in ld_path if re.search(r'/cuda[-/]|/cuda$|cudatoolkit', p, re.I)]
    if cuda_dirs:
        details['cuda_dirs_on_ld_library_path'] = cuda_dirs
        warnings.append(f'CUDA toolkit directories are on LD_LIBRARY_PATH ({", ".join(cuda_dirs)})')
        remedies.append('do not load cuda modules for training: torch bundles its own CUDA 13 libraries, a '
                        'module\'s libraries may shadow them (remove cuda from HYAK_MODULES)')
    if ctx.paths is not None and not _same_path(os.environ.get('HF_HOME', ''), ctx.paths.hf_cache):
        warnings.append(f'HF_HOME is {os.environ.get("HF_HOME")!r}, not {ctx.paths.hf_cache}')
        remedies.append('hyak_activate exports HF_HOME=$ANYTOK_ROOT/hf-cache (home quota is 10 GB)')
    if site.ENABLE_USER_SITE:
        warnings.append('user site-packages are enabled (~/.local packages could shadow the pins)')
        remedies.append('export PYTHONNOUSERSITE=1 (hyak_activate does)')
    if warnings:
        return CheckResult('runtime_env', WARN, '; '.join(warnings), '; '.join(remedies), details)
    return CheckResult('runtime_env', PASS, f'LD_LIBRARY_PATH has {link_dir}, libpython3.12 present, HF_HOME set',
                       details=details)


def _staged_library_source(ctx: Context) -> Optional[str]:
    manifest = ctx.facts.get('manifest')
    if manifest is None and ctx.paths is not None and ctx.paths.code_manifest.is_file():
        try:
            from hyak.common.io import read_json
            manifest = read_json(ctx.paths.code_manifest)
        except (OSError, ValueError):
            manifest = None
    return (manifest or {}).get('rust_library', {}).get('source')


def check_glibc(ctx: Context) -> CheckResult:
    try:
        raw = os.confstr('CS_GNU_LIBC_VERSION')
    except (ValueError, OSError, AttributeError):
        raw = None
    version = parse_glibc(raw)
    details = {'confstr': raw, 'version': '.'.join(map(str, version)) or None}
    if not version:
        return CheckResult('glibc', SKIP, f'not a glibc system ({platform.system()})', details=details)
    shown = details['version']
    if version < GLIBC_FOR_TORCH:
        return CheckResult('glibc', FAIL, f'glibc {shown} < 2.28 (torch manylinux_2_28 wheels)',
                           'use the Tillicum nodes (Rocky Linux 9, glibc 2.34)', details)
    source = _staged_library_source(ctx)
    if version < GLIBC_FOR_COMMITTED_LIBRARY and source != 'rebuilt':
        return CheckResult('glibc', FAIL, f'glibc {shown} < 2.34 needed by the committed librust_utils.so',
                           'rebuild it on this cluster: bash hyak/env/build_rust_utils.sh', details)
    return CheckResult('glibc', PASS, f'glibc {shown}', details=details)


def collect_nvidia(ctx: Context):
    """``(details, failure)``: the nvidia-smi facts, asked once and kept in ``ctx.facts['nvidia']``.

    ``details`` is None and ``failure`` is ``(summary, remedy)`` when nvidia-smi is missing or fails.
    """
    if ctx.facts.get('nvidia') is not None:
        return ctx.facts['nvidia'], None
    if shutil.which('nvidia-smi') is None:
        return None, ('nvidia-smi not found', 'run on a GPU node: hyak/slurm/submit.sh check-env')
    proc, error = _run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
                        '--format=csv,noheader,nounits'], timeout=60)
    if proc is None or proc.returncode != 0:
        text = error or (proc.stderr or proc.stdout)
        return None, (f'nvidia-smi failed: {str(text).strip()[-300:]}',
                      'run on a GPU node with a GPU allocated (--gres=gpu:1)')
    rows = parse_nvidia_query(proc.stdout)
    plain, _ = _run(['nvidia-smi'], timeout=60)
    plain_text = plain.stdout if plain is not None else ''
    details = {'gpus': rows, 'driver_cuda': parse_driver_cuda(plain_text), 'nvidia_smi': plain_text[-6000:]}
    ctx.facts['nvidia'] = details
    return details, None


def check_nvidia_smi(ctx: Context) -> CheckResult:
    details, failure = collect_nvidia(ctx)
    if failure is not None:
        return CheckResult('nvidia_smi', FAIL, failure[0], failure[1])
    rows = details['gpus']
    if not rows:
        return CheckResult('nvidia_smi', FAIL, 'nvidia-smi lists no GPU', 'request a GPU: --gres=gpu:1', details)
    first = rows[0]
    summary = (f'{len(rows)} GPU(s): {first["name"]}, {first["memory_total_mib"]} MiB, driver '
               f'{first["driver_version"]} (CUDA {details["driver_cuda"] or "?"})')
    if 'H200' not in first['name']:
        return CheckResult('nvidia_smi', WARN, summary + ' -- not an H200',
                           'the port is sized for H200 (141 GB); check HYAK_PARTITION', details)
    return CheckResult('nvidia_smi', PASS, summary, details=details)


def check_torch_cuda(ctx: Context) -> CheckResult:
    import torch
    details = {'torch': torch.__version__, 'cuda': torch.version.cuda,
               'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
               'available': torch.cuda.is_available()}
    nvidia, _ = collect_nvidia(ctx)              # asks nvidia-smi itself when --only torch_cuda
    nvidia = nvidia or {}
    gpus = nvidia.get('gpus') or [{}]
    driver_version = gpus[0].get('driver_version')
    status, summary, remedy = cuda_driver_verdict(driver_version, nvidia.get('driver_cuda'), torch.version.cuda)
    details['driver_verdict'] = {'status': status, 'summary': summary}
    if not details['available']:
        return CheckResult('torch_cuda', FAIL, f'torch.cuda.is_available() is False (torch {torch.__version__}, '
                           f'CUDA {torch.version.cuda}); {summary}',
                           remedy or 'check the driver (nvidia-smi) and that a GPU is allocated to the job', details)
    details.update({'device_count': torch.cuda.device_count(), 'name': torch.cuda.get_device_name(0),
                    'capability': list(torch.cuda.get_device_capability(0)),
                    'total_memory_gib': round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
                    'bf16_supported': bool(torch.cuda.is_bf16_supported())})
    ctx.facts['cuda_ok'] = True
    line = (f'torch {torch.__version__} CUDA {torch.version.cuda} on {details["name"]} '
            f'({details["total_memory_gib"]} GiB, sm_{"".join(map(str, details["capability"]))}); {summary}')
    if status == FAIL and torch.version.cuda and (not parse_version(driver_version)
                                                  or driver_below_minimum(driver_version, torch.version.cuda)):
        # CUDA initialised although the kernel driver is below the minimum (or its version is unreadable): this is
        # what the forward-compatibility libraries (cuda-compat) do, and nvidia-smi keeps showing the kernel
        # driver's version.  Working CUDA is the fact that counts; gpu_smoke is the proof.
        details['forward_compatibility'] = True
        line = (f'torch {torch.__version__} CUDA {torch.version.cuda} on {details["name"]} '
                f'({details["total_memory_gib"]} GiB, sm_{"".join(map(str, details["capability"]))}); CUDA '
                f'initialised although the node driver ({driver_version or "unknown"}) is below the minimum for CUDA '
                f'{torch.version.cuda}: forward-compatibility libraries (cuda-compat) are in use')
        return CheckResult('torch_cuda', WARN, line,
                           'nothing to change if gpu_smoke passes; keep the cuda-compat directory on LD_LIBRARY_PATH in '
                           'every job, and ask for nodes with driver >= 580 to remove the caveat', details)
    if status == FAIL:
        return CheckResult('torch_cuda', FAIL, line, remedy, details)
    if status == WARN:
        return CheckResult('torch_cuda', WARN, line, remedy, details)
    if not details['bf16_supported']:
        return CheckResult('torch_cuda', FAIL, line + '; bf16 not supported', 'the trainer needs bf16 autocast (H200)',
                           details)
    return CheckResult('torch_cuda', PASS, line, details=details)


def gpu_smoke(device: str = 'cuda') -> dict:
    """bf16 matmul (scripts/modal_anytok_pool_20k.py:241-249) and one training-shaped step: bf16 autocast
    Linear + LayerNorm + GELU forward, float32 loss, backward, joint 0.1 gradient clip, fused AdamW step
    (the trainer's switches).  Raises AssertionError on a non-finite or unchanged result."""
    import torch
    generator = torch.Generator(device='cpu').manual_seed(0)
    left = torch.randn(256, 256, generator=generator).to(device=device, dtype=torch.bfloat16)
    product = left @ left
    matmul_finite = bool(torch.isfinite(product).all().item())
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.LayerNorm(128), torch.nn.GELU(),
                                torch.nn.Linear(128, 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, fused=True)
    inputs = torch.randn(32, 64, generator=generator).to(device)
    targets = torch.randn(32, 8, generator=generator).to(device)
    before = [p.detach().clone() for p in model.parameters()]
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        output = model(inputs)
    loss = torch.nn.functional.mse_loss(output.float(), targets)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()
    changed = all(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))
    finite = all(bool(torch.isfinite(p).all().item()) for p in model.parameters())
    result = {'device': str(device), 'matmul_finite': matmul_finite, 'output_dtype': str(output.dtype),
              'loss': float(loss.item()), 'grad_norm_before_clip': float(norm.item()),
              'parameters_changed': changed, 'parameters_finite': finite}
    if torch.device(device).type == 'cuda':
        result['max_memory_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    if not (matmul_finite and finite and changed and output.dtype == torch.bfloat16 and result['loss'] == result['loss']):
        raise AssertionError(f'GPU smoke test gave an invalid result: {result}')
    return result


def check_gpu_smoke(ctx: Context) -> CheckResult:
    cuda_ok = ctx.facts.get('cuda_ok')
    if cuda_ok is None:                          # torch_cuda did not run (--only gpu_smoke): ask torch itself
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            return CheckResult('gpu_smoke', SKIP, f'skipped: torch is not importable ({type(exc).__name__}: {exc})')
    if not cuda_ok:
        return CheckResult('gpu_smoke', SKIP, 'skipped: CUDA is not available (torch.cuda.is_available() is False; '
                                              'see torch_cuda)')
    try:
        details = gpu_smoke('cuda')
    except Exception as exc:  # noqa: BLE001
        return CheckResult('gpu_smoke', FAIL, f'{type(exc).__name__}: {exc}',
                           'see torch_cuda/nvidia_smi; a driver/runtime mismatch shows up here first',
                           {'traceback': traceback.format_exc()[-4000:]})
    return CheckResult('gpu_smoke', PASS, f'bf16 matmul finite, fused AdamW step OK (loss {details["loss"]:.4f})',
                       details=details)


def check_code_stage(ctx: Context) -> CheckResult:
    from hyak.code import stage_code
    problems = stage_code.verify_stage(ctx.paths, repo=ctx.repo)
    details = {'code_root': str(ctx.paths.code_root), 'problems': problems}
    if problems:
        return CheckResult('code_stage', FAIL, f'{len(problems)} problem(s): {problems[0]}',
                           'python -m hyak.code.stage_code (add --force only if no run has started)', details)
    manifest = stage_code.load_manifest(ctx.paths)
    ctx.facts['manifest'] = manifest
    details.update({'identity_sha256': manifest['identity_sha256'], 'rust_library': manifest['rust_library']['source'],
                    'chunked_hypernet_sha256': manifest['chunked_hypernet']['sha256'],
                    'commit': (manifest.get('repository') or {}).get('commit')})
    return CheckResult('code_stage', PASS, f'uniform and rl trees verified (identity {manifest["identity_sha256"][:16]}..., '
                       f'{manifest["rust_library"]["source"]} librust_utils.so)', details=details)


def _candidate_code_root(ctx: Context, library: Path) -> Path:
    """A throw-away code root with the pinned hf_hypernet and ``library`` at the trainer's relative path."""
    from hyak.code import pinned
    base = None
    if ctx.paths is not None and ctx.paths.build_root.is_dir():
        base = str(ctx.paths.build_root)
    root = Path(tempfile.mkdtemp(prefix='check-rust-', dir=base))
    for rel, source in pinned.tree_layout('uniform').items():
        if rel == pinned.RUST_LIBRARY_REL:
            continue
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ctx.repo / source, target)
    target = root / pinned.RUST_LIBRARY_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, target)
    return root


def check_rust_library(ctx: Context) -> CheckResult:
    from hyak.code import pinned
    candidate = getattr(ctx.args, 'rust_library', None)
    temporary = None
    if candidate:
        library = Path(candidate)
        if not library.is_file():
            return CheckResult('rust_library', FAIL, f'{library} not found', 'pass the path of a built librust_utils.so')
        temporary = _candidate_code_root(ctx, library)
        code_root = temporary
    else:
        if ctx.paths is None:
            return CheckResult('rust_library', SKIP, 'skipped: no configuration')
        code_root = ctx.paths.code_dir('uniform')
        library = code_root / pinned.RUST_LIBRARY_REL
        if not library.is_file():
            return CheckResult('rust_library', FAIL, f'{library} not found', 'python -m hyak.code.stage_code')
    try:
        details = {'library': str(library), 'code_root': str(code_root), 'binary': binary_info(library),
                   'expected_binary': host_binary_expectation()}
        from hyak.common.io import sha256_file
        details['sha256'] = sha256_file(library)
        details['committed'] = details['sha256'] == pinned.RUST_LIBRARY_SHA256
        expected = details['expected_binary']
        binary = details['binary']
        if expected['format'] and (binary.get('format') != expected['format']
                                   or (expected['machine'] and binary.get('machine') != expected['machine'])):
            return CheckResult('rust_library', FAIL, f'{library} is a {binary.get("format")} {binary.get("machine") or ""} '
                               f'binary; this host needs {expected["format"]} {expected["machine"] or ""}'.strip(),
                               'rebuild it on this cluster: bash hyak/env/build_rust_utils.sh', details)
        if shutil.which('ldd'):
            proc, error = _run(['ldd', str(library)], timeout=60)
            details['ldd'] = parse_ldd(((proc.stdout or '') + (proc.stderr or '')) if proc else '')
            if error:
                details['ldd_error'] = error
        probe = run_probe(RUST_PROBE, {'code_root': str(code_root)}, timeout=ctx.args.probe_timeout)
        details['probe'] = probe['result']
        details['probe_stderr'] = probe['stderr']
        result = probe['result'] or {}
        if not result.get('ok'):
            error = result.get('error') or probe['error'] or 'unknown error'
            details['probe_stdout'] = probe['stdout']
            return CheckResult('rust_library', FAIL, f'import/exercise failed: {error}',
                               rust_failure_remedy(error + ' ' + probe['stderr'], details.get('ldd')), details)
        version = result.get('version')
        draw = next((s for s in result.get('steps', []) if s.get('step') == 'draw_tokenizer'), {})
        summary = (f'loaded as the trainer does (rust_utils {version}); prime_sampler + draw_tokenizer gave a '
                   f'{draw.get("vocab")}-piece Unigram tokenizer in {result.get("seconds")} s')
        if version != pinned.RUST_UTILS_VERSION:
            return CheckResult('rust_library', WARN, summary + f' -- version {version!r}, expected '
                               f'{pinned.RUST_UTILS_VERSION!r}', 'rebuild from runtime/anytok_pool_20k/rust_utils '
                               'with bash hyak/env/build_rust_utils.sh', details)
        return CheckResult('rust_library', PASS, summary, details=details)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _check_trainer_import(ctx: Context, run: str) -> CheckResult:
    from hyak.code import pinned
    name = f'trainer_import_{run}'
    code_root = ctx.paths.code_dir(run)
    if not (code_root / pinned.TRAINER_REL).is_file():
        return CheckResult(name, FAIL, f'{code_root / pinned.TRAINER_REL} not found', 'python -m hyak.code.stage_code')
    sha = pinned.UNIFORM_TRAINER_SHA256 if run == 'uniform' else pinned.RL_TRAINER_SHA256
    probe = run_probe(TRAINER_PROBE, {'code_root': str(code_root), 'trainer_sha256': sha, 'run': run},
                      timeout=ctx.args.probe_timeout)
    result = probe['result'] or {}
    details = {'code_root': str(code_root), 'probe': probe['result'], 'probe_stderr': probe['stderr']}
    if not result.get('ok'):
        error = result.get('error') or probe['error'] or 'unknown error'
        details['probe_stdout'] = probe['stdout']
        return CheckResult(name, FAIL, f'import of the staged {run} trainer failed: {error}',
                           rust_failure_remedy(error + ' ' + probe['stderr']) if 'rust' in error.lower()
                           or 'librust_utils' in error else 'see the error; restage with python -m hyak.code.stage_code '
                           'and check the train environment', details)
    return CheckResult(name, PASS, f'scripts.train_joint_smollm_input imported from {code_root} '
                       f'(sha256 {result["trainer_sha256"][:16]}..., REPO is the staged tree, '
                       f'{len(result.get("modules", {}))} modules local) in {result.get("seconds")} s', details=details)


def check_trainer_import_uniform(ctx: Context) -> CheckResult:
    return _check_trainer_import(ctx, 'uniform')


def check_trainer_import_rl(ctx: Context) -> CheckResult:
    return _check_trainer_import(ctx, 'rl')


def filesystem_semantics(base: Path, *, io_mib: int = 0, python=None) -> dict:
    """Exercise the filesystem operations the port relies on inside ``base``; returns facts.

    Keys: ``flock_excludes`` (a second process cannot take LOCK_EX|LOCK_NB while the first holds
    it, and can after release -- the trainer's ``.run.lock``, train_joint_smollm_input.py:352-356),
    ``mkdir_exclusive`` (pool/eval claims), ``replace_file`` (atomic status writes),
    ``replace_dir`` (the trainer's ``.writing-*`` -> ``step-NNNNNN``, lines 214-230),
    ``replace_dir_over_empty``, ``fsync_file`` and ``fsync_dir`` (``fsync`` = both), and with
    ``io_mib`` > 0 ``write_mb_s`` / ``read_mb_s``.
    Every failing step records an ``error_<step>`` entry instead of raising.
    """
    import fcntl
    facts = {'base': str(base)}
    work = Path(tempfile.mkdtemp(prefix=f'.hyak-check-env-{socket.gethostname()}-', dir=str(base)))
    try:
        lock_path = work / '.run.lock'
        try:
            handle = lock_path.open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def child():
                proc = subprocess.run([python or sys.executable, '-I', '-c', _FLOCK_CHILD, str(lock_path)],
                                      capture_output=True, text=True, timeout=60, check=False)
                return proc.stdout.strip() or proc.stderr.strip()[-200:]
            while_held = child()
            fcntl.flock(handle, fcntl.LOCK_UN)
            after_release = child()
            handle.close()
            facts['flock_while_held'] = while_held
            facts['flock_after_release'] = after_release
            facts['flock_excludes'] = while_held == 'blocked' and after_release == 'acquired'
        except Exception as exc:  # noqa: BLE001
            facts['flock_excludes'] = False
            facts['error_flock'] = f'{type(exc).__name__}: {exc}'
        try:
            claim = work / 'claims' / '0007'
            claim.parent.mkdir()
            os.mkdir(claim)
            try:
                os.mkdir(claim)
                facts['mkdir_exclusive'] = False
            except FileExistsError:
                facts['mkdir_exclusive'] = True
        except Exception as exc:  # noqa: BLE001
            facts['mkdir_exclusive'] = False
            facts['error_mkdir'] = f'{type(exc).__name__}: {exc}'
        try:
            target = work / 'status.json'
            for value in ('first', 'second'):
                temporary = work / f'.status.json.{value}.tmp'
                temporary.write_text(value)
                os.replace(temporary, target)
            facts['replace_file'] = target.read_text() == 'second' and not list(work.glob('.status.json.*'))
        except Exception as exc:  # noqa: BLE001
            facts['replace_file'] = False
            facts['error_replace_file'] = f'{type(exc).__name__}: {exc}'
        try:
            final = work / 'step-000001'
            with tempfile.TemporaryDirectory(prefix='.writing-', dir=str(work)) as temporary:
                stage = Path(temporary)
                (stage / 'hypernet').mkdir()
                (stage / 'hypernet' / 'config.json').write_text('{}')
                (stage / 'state.pt').write_bytes(b'x' * 1024)
                os.replace(stage, final)
            facts['replace_dir'] = (final / 'hypernet' / 'config.json').is_file() and not list(work.glob('.writing-*'))
        except Exception as exc:  # noqa: BLE001
            facts['replace_dir'] = False
            facts['error_replace_dir'] = f'{type(exc).__name__}: {exc}'
        try:
            source, empty = work / 'dir-src', work / 'dir-empty'
            source.mkdir()
            (source / 'f').write_text('x')
            empty.mkdir()
            os.replace(source, empty)
            facts['replace_dir_over_empty'] = (empty / 'f').is_file() and not source.exists()
        except Exception as exc:  # noqa: BLE001
            facts['replace_dir_over_empty'] = False
            facts['error_replace_dir_over_empty'] = f'{type(exc).__name__}: {exc}'
        try:
            with open(work / 'fsync.bin', 'wb') as stream:
                stream.write(b'y' * 4096)
                stream.flush()
                os.fsync(stream.fileno())
            facts['fsync_file'] = True
        except Exception as exc:  # noqa: BLE001
            facts['fsync_file'] = False
            facts['error_fsync'] = f'{type(exc).__name__}: {exc}'
        try:
            descriptor = os.open(str(work), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            facts['fsync_dir'] = True
        except Exception as exc:  # noqa: BLE001
            facts['fsync_dir'] = False
            facts['error_fsync_dir'] = f'{type(exc).__name__}: {exc}'
        facts['fsync'] = bool(facts['fsync_file'] and facts['fsync_dir'])
        if io_mib > 0:
            try:
                import hashlib
                block = os.urandom(8 * 2 ** 20)
                path = work / 'throughput.bin'
                start = time.monotonic()
                with open(path, 'wb') as stream:
                    for _ in range(max(1, io_mib // 8)):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                written = path.stat().st_size
                facts['write_mb_s'] = round(written / 1e6 / max(time.monotonic() - start, 1e-6), 1)
                start = time.monotonic()
                digest = hashlib.sha256()
                with open(path, 'rb') as stream:
                    while True:
                        chunk = stream.read(8 * 2 ** 20)
                        if not chunk:
                            break
                        digest.update(chunk)
                facts['read_hash_mb_s'] = round(written / 1e6 / max(time.monotonic() - start, 1e-6), 1)
                facts['io_test_bytes'] = written
            except Exception as exc:  # noqa: BLE001
                facts['error_io'] = f'{type(exc).__name__}: {exc}'
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return facts


#: Facts that must hold on ANYTOK_RUNS and ANYTOK_ROOT.  A directory fsync is not among them: every
#: writer of the port goes through hyak.common.io.fsync_dir, which ignores its error, and the pinned
#: trainers never call fsync at all.
REQUIRED_FS_FACTS = ('flock_excludes', 'mkdir_exclusive', 'replace_file', 'replace_dir', 'fsync_file')
#: Facts that matter on the checkout's monitor directory: status files are replaced atomically.  The
#: registry lock is taken with a time-out and the claims live on ANYTOK_ROOT, so those are warnings there.
MONITOR_FS_FACTS = ('replace_file', 'fsync_file')


def _fact(facts: dict, key: str):
    """``facts[key]``; ``fsync_file`` falls back to the older combined ``fsync`` key."""
    if key == 'fsync_file' and key not in facts:
        return facts.get('fsync')
    return facts.get(key)


def filesystem_verdict(name: str, label: str, facts: dict, required=REQUIRED_FS_FACTS) -> CheckResult:
    """Turn :func:`filesystem_semantics` facts into a check result.

    A missing ``required`` fact fails; any other missing fact (flock, claims, rename of a directory, a
    directory fsync) is a warning, so ``required`` decides how much the location matters.
    """
    all_facts = ('flock_excludes', 'mkdir_exclusive', 'replace_file', 'replace_dir', 'fsync_file', 'fsync_dir')
    failed = [key for key in required if not _fact(facts, key)]
    soft = [key for key in all_facts if key not in required and key in facts and not _fact(facts, key)]
    if failed:
        remedy = []
        if 'flock_excludes' in failed:
            remedy.append('fcntl.flock does not exclude a second process here, so the trainer\'s .run.lock cannot '
                          'stop two jobs on one run directory: put ANYTOK_RUNS on a filesystem with working locks '
                          '(project or scrubbed GPFS) and never submit train jobs without submit.sh (singleton)')
        if set(failed) - {'flock_excludes'}:
            remedy.append('atomic mkdir/rename/fsync failed: choose another storage location for ' + label)
        return CheckResult(name, FAIL, f'{label} {facts.get("base")}: failed {", ".join(failed)}', '; '.join(remedy),
                           facts)
    if soft:
        notes = []
        if 'fsync_dir' in soft:
            notes.append('directory fsync is not supported here; renames stay atomic and the port ignores that error')
        if set(soft) - {'fsync_dir'}:
            notes.append(f'{", ".join(sorted(set(soft) - {"fsync_dir"}))} failed here (not needed on this location '
                         'to run; a stuck lock is given up by the jobs after 60 s)')
        return CheckResult(name, WARN, f'{label} {facts.get("base")}: ' + '; '.join(notes),
                           'nothing to change unless training or status files misbehave', facts)
    summary = f'{label} {facts.get("base")}: flock, mkdir claims, atomic file and directory rename, fsync OK'
    if 'write_mb_s' in facts:
        summary += f'; write {facts["write_mb_s"]} MB/s, read+hash {facts["read_hash_mb_s"]} MB/s'
        if facts['write_mb_s'] < 100:
            return CheckResult(name, WARN, summary, 'slow storage: a 23.5 GB restart takes '
                               f'{23500 / max(facts["write_mb_s"], 1):.0f} s; keep HYAK_SEGMENT_MARGIN_MINUTES generous',
                               facts)
    return CheckResult(name, PASS, summary, details=facts)


def _filesystem_check(ctx: Context, name: str, label: str, base: Path, io_mib: int,
                      required=REQUIRED_FS_FACTS) -> CheckResult:
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        status = FAIL if required is REQUIRED_FS_FACTS else WARN
        return CheckResult(name, status, f'cannot create {base}: {exc}', f'fix {label} in hyak/config.env'
                           if status == FAIL else 'check the permissions of the checkout\'s modal_results directory')
    facts = filesystem_semantics(base, io_mib=io_mib)
    return filesystem_verdict(name, label, facts, required)


def check_filesystem_runs(ctx: Context) -> CheckResult:
    return _filesystem_check(ctx, 'filesystem_runs', 'ANYTOK_RUNS', ctx.paths.runs, ctx.args.io_test_mib)


def check_filesystem_root(ctx: Context) -> CheckResult:
    return _filesystem_check(ctx, 'filesystem_root', 'ANYTOK_ROOT', ctx.paths.root, 0)


def check_filesystem_monitor(ctx: Context) -> CheckResult:
    """The checkout's monitor directory (status files, events, the dashboard registry lock, Slurm logs)."""
    return _filesystem_check(ctx, 'filesystem_monitor', 'the monitor directory', ctx.paths.monitor_dir(STAGE), 0,
                             MONITOR_FS_FACTS)


def disk_verdict(free: dict) -> tuple:
    """``(status, summary, remedy)`` for ``{"runs": free bytes, "root": free bytes}``."""
    runs_gb, root_gb = free['runs'] / 1e9, free['root'] / 1e9
    summary = f'free: ANYTOK_RUNS {runs_gb:,.0f} GB, ANYTOK_ROOT {root_gb:,.0f} GB (filesystem view, not your quota)'
    if runs_gb < RUNS_FREE_FAIL_GB:
        return (FAIL, summary, f'less than {RUNS_FREE_FAIL_GB} GB on ANYTOK_RUNS: the first full restart '
                               '(23.5 GB) would not fit; free space or move ANYTOK_RUNS')
    problems = []
    if runs_gb < RUNS_FREE_WARN_GB:
        problems.append(f'two runs at the original retention need about {RUNS_FREE_WARN_GB} GB on ANYTOK_RUNS')
    if root_gb < ROOT_FREE_WARN_GB:
        problems.append(f'corpus, environments, pool and features need about {ROOT_FREE_WARN_GB} GB on ANYTOK_ROOT')
    if problems:
        return (WARN, summary, '; '.join(problems) + ' (check your group quota with hyakstorage)')
    return (PASS, summary, None)


def check_disk_space(ctx: Context) -> CheckResult:
    free, details = {}, {}
    for key, path in (('runs', ctx.paths.runs), ('root', ctx.paths.root)):
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        free[key] = usage.free
        details[key] = {'path': str(path), 'measured_at': str(probe), 'total': usage.total, 'free': usage.free}
    if shutil.which('hyakstorage'):
        proc, error = _run(['hyakstorage'], timeout=60)
        details['hyakstorage'] = (proc.stdout[-4000:] if proc is not None else error)
    status, summary, remedy = disk_verdict(free)
    details['scrubbed_warning'] = ('files on /gpfs/scrubbed untouched for 60 days are deleted; copy the final '
                                   'checkpoints to project storage')
    return CheckResult('disk_space', status, summary, remedy, details)


def check_model_files(ctx: Context) -> CheckResult:
    from hyak.model import fetch_model
    dest = ctx.paths.model_dir
    if not dest.exists():
        return CheckResult('model_files', WARN, f'{dest} does not exist yet',
                           'fetch it on the login node: python -m hyak.model.fetch_model')
    problems = fetch_model.verify_model_dir(dest, check_hash=ctx.args.hash_model)
    how = 'bytes and SHA-256' if ctx.args.hash_model else 'bytes only (--no-hash-model); hash them with --hash-model'
    if problems:
        return CheckResult('model_files', FAIL, '; '.join(problems), 'python -m hyak.model.fetch_model (after removing '
                           'the wrong files it names)', {'problems': problems})
    return CheckResult('model_files', PASS, f'{dest}: three files match plans/zett-modal-20k-inputs.json ({how})')


def probe_url(url, timeout=15.0, opener=None) -> dict:
    """GET a URL (first KB only); ``{"ok", "status", "seconds", "error"}``.  Never raises."""
    import urllib.error
    import urllib.request
    opener = opener or urllib.request.urlopen
    start = time.monotonic()
    request = urllib.request.Request(url, headers={'User-Agent': 'hyak-check-env/1'})
    try:
        with opener(request, timeout=timeout) as response:
            response.read(1024)
            status = getattr(response, 'status', None) or response.getcode()
        return {'ok': 200 <= int(status) < 400, 'status': int(status), 'seconds': round(time.monotonic() - start, 2),
                'error': None}
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'status': exc.code, 'seconds': round(time.monotonic() - start, 2), 'error': str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {'ok': False, 'status': None, 'seconds': round(time.monotonic() - start, 2),
                'error': f'{type(exc).__name__}: {exc}'}


def check_network(ctx: Context, opener=None) -> CheckResult:
    results = {name: {'url': url, **probe_url(url, opener=opener)} for name, url, _ in NETWORK_ENDPOINTS}
    failed = [name for name, _, _ in NETWORK_ENDPOINTS if not results[name]['ok']]
    summary = ', '.join(f'{name} {"ok" if r["ok"] else "unreachable"}' for name, r in results.items())
    if failed:
        return CheckResult('network', WARN, summary,
                           'run the download steps on the login node (setup_*_env.sh, fetch_model, '
                           'build_rust_utils.sh fetch, build_wheel.sh fetch, corpus build if huggingface.co is '
                           'unreachable from compute nodes)', {'endpoints': results})
    return CheckResult('network', PASS, summary, details={'endpoints': results})


def check_freeze(ctx: Context) -> CheckResult:
    freeze_path = ctx.paths.env_freeze('train')
    proc, error = _run([sys.executable, '-m', 'pip', 'freeze', '--all'], timeout=180)
    if proc is None or proc.returncode != 0:
        return CheckResult('freeze', WARN, f'pip freeze failed: {error or proc.stderr[-300:]}',
                           'rerun bash hyak/env/setup_train_env.sh')
    now = sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())
    if not freeze_path.is_file():
        return CheckResult('freeze', WARN, f'{freeze_path} not found', 'bash hyak/env/setup_train_env.sh writes it',
                           {'freeze': now})
    recorded = sorted(line.strip() for line in freeze_path.read_text().splitlines() if line.strip())
    added = sorted(set(now) - set(recorded))
    removed = sorted(set(recorded) - set(now))
    details = {'freeze_file': str(freeze_path), 'added': added[:50], 'removed': removed[:50], 'packages': len(now)}
    if added or removed:
        return CheckResult('freeze', WARN, f'environment differs from {freeze_path.name}: +{len(added)} -{len(removed)} '
                           f'({", ".join((added + removed)[:4])})',
                           'the environment changed after setup; one frozen environment should serve both runs '
                           '(rerun setup_train_env.sh to record it, and say so in the run notes)', details)
    return CheckResult('freeze', PASS, f'{len(now)} packages equal {freeze_path}', details=details)


def check_resources(ctx: Context) -> CheckResult:
    details = {'cpu_count': os.cpu_count(), 'host': socket.gethostname(), 'platform': platform.platform()}
    if hasattr(os, 'sched_getaffinity'):
        details['cpus_usable'] = len(os.sched_getaffinity(0))
    for key in ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE',
                'SLURM_JOB_GPUS', 'SLURM_CLUSTER_NAME', 'SLURMD_NODENAME'):
        if os.environ.get(key):
            details[key] = os.environ[key]
    for candidate in ('/sys/fs/cgroup/memory.max',):
        try:
            details['cgroup_memory_max'] = Path(candidate).read_text().strip()
        except OSError:
            pass
    try:
        meminfo = Path('/proc/meminfo').read_text()
        match = re.search(r'MemTotal:\s+(\d+) kB', meminfo)
        if match:
            details['mem_total_gib'] = round(int(match.group(1)) / 2 ** 20, 1)
    except OSError:
        pass
    summary = f'{details.get("cpus_usable", details["cpu_count"])} CPUs usable'
    if details.get('SLURM_MEM_PER_NODE'):
        summary += f', job memory {details["SLURM_MEM_PER_NODE"]} MB'
    # The training wrapper's in-process stop timer reads the job's end from SLURM_JOB_END_TIME (else squeue);
    # without either it has no timer and relies on Slurm's --signal=B:TERM alone.  Warn only: it is no failure.
    details['SLURM_JOB_END_TIME'] = os.environ.get('SLURM_JOB_END_TIME') or None
    if os.environ.get('SLURM_JOB_ID'):
        from hyak.common import slurm
        remaining = slurm.seconds_remaining()
        details['seconds_remaining'] = ('unlimited' if remaining == float('inf') else
                                        None if remaining is None else round(remaining, 1))
        if remaining is None:
            return CheckResult(
                'resources', WARN,
                summary + '; the job end is unknown (SLURM_JOB_END_TIME is not set and squeue gave no time left): '
                          'a training segment would have no in-process stop timer and rely on Slurm\'s --signal alone',
                'run it inside the Slurm job that submit.sh started (it sets --signal and the time limit); if '
                'SLURM_JOB_END_TIME stays unset on this cluster, keep a margin that --signal can honour', details)
        summary += f', {details["seconds_remaining"]} s of job time left'
    return CheckResult('resources', PASS, summary, details=details)


#: name -> (function, requirements); requirements: "config" (needs a loaded config), "gpu", "network".
CHECKS = (
    ('config', check_config, ()),
    ('python', check_python, ()),
    ('packages', check_packages, ()),
    ('runtime_env', check_runtime_env, ()),
    ('glibc', check_glibc, ()),
    ('nvidia_smi', check_nvidia_smi, ('gpu',)),
    ('torch_cuda', check_torch_cuda, ('gpu',)),
    ('gpu_smoke', check_gpu_smoke, ('gpu',)),
    ('code_stage', check_code_stage, ('config',)),
    ('rust_library', check_rust_library, ()),
    ('trainer_import_uniform', check_trainer_import_uniform, ('config',)),
    ('trainer_import_rl', check_trainer_import_rl, ('config',)),
    ('filesystem_runs', check_filesystem_runs, ('config',)),
    ('filesystem_root', check_filesystem_root, ('config',)),
    ('filesystem_monitor', check_filesystem_monitor, ('config',)),
    ('disk_space', check_disk_space, ('config',)),
    ('model_files', check_model_files, ('config',)),
    ('network', check_network, ('network',)),
    ('freeze', check_freeze, ('config',)),
    ('resources', check_resources, ()),
)
CHECK_NAMES = tuple(name for name, _, _ in CHECKS)


def select_checks(only=None, skip=None) -> list:
    """Check names to run, in order; unknown names raise ValueError."""
    def expand(values):
        names = [n.strip() for value in values or () for n in str(value).split(',') if n.strip()]
        unknown = sorted(set(names) - set(CHECK_NAMES))
        if unknown:
            raise ValueError(f'unknown check(s) {unknown}; known: {", ".join(CHECK_NAMES)}')
        return names
    chosen = expand(only) or list(CHECK_NAMES)
    skipped = set(expand(skip))
    return [name for name in CHECK_NAMES if name in chosen and name not in skipped]


def run_checks(ctx: Context, names, on_result: Optional[Callable] = None) -> list:
    """Run the named checks in order; an unexpected exception becomes a ``fail``."""
    functions = {name: (function, needs) for name, function, needs in CHECKS}
    results = []
    for name in names:
        function, needs = functions[name]
        start = time.monotonic()
        if 'config' in needs and ctx.paths is None:
            result = CheckResult(name, SKIP, 'skipped: configuration not loaded (see config)')
        elif 'gpu' in needs and ctx.args.no_gpu:
            result = CheckResult(name, SKIP, 'skipped (--no-gpu)')
        elif 'network' in needs and ctx.args.no_network:
            result = CheckResult(name, SKIP, 'skipped (--no-network)')
        else:
            try:
                result = function(ctx)
            except Exception as exc:  # noqa: BLE001 - every unexpected error is a failed check
                result = CheckResult(name, FAIL, f'check crashed: {type(exc).__name__}: {exc}',
                                     'see the traceback in the report', {'traceback': traceback.format_exc()[-6000:]})
        result.seconds = round(time.monotonic() - start, 3)
        results.append(result)
        if on_result is not None:
            on_result(result, results)
    return results


def build_report(ctx: Context, results) -> dict:
    from hyak.common import slurm
    from hyak.common.io import utc_now
    return {
        'kind': 'hyak_check_env_report', 'schema': 1, 'at_utc': utc_now(), 'host': socket.gethostname(),
        'job_id': slurm.job_id(), 'python': sys.executable, 'argv': sys.argv[1:],
        'exit_code': exit_code(results), 'summary': summary_line(results),
        'failed': [r.name for r in results if r.status == FAIL],
        'warnings': [r.name for r in results if r.status == WARN],
        'checks': [r.to_dict() for r in results],
    }


def write_reports(ctx: Context, report: dict) -> list:
    """Write the report next to the stage status files and into ``$ANYTOK_ROOT/env`` (not with
    ``--no-status``) and to ``--report``; returns the paths written."""
    from hyak.common import slurm
    from hyak.common.io import atomic_write_json
    from hyak.common.paths import default_monitor_dir
    targets = []
    if not getattr(ctx.args, 'no_status', False):
        monitor = ctx.paths.monitor_dir(STAGE) if ctx.paths is not None else default_monitor_dir(STAGE)
        targets.append(monitor / 'check_env_report.json')
        targets.append(monitor / f'check_env_report-{slurm.worker_id()}.json')
        if ctx.paths is not None and ctx.paths.env_root.is_dir():
            targets.append(ctx.paths.env_root / 'check_env_report.json')
    if getattr(ctx.args, 'report', None):
        targets.append(Path(ctx.args.report))
    written = []
    for path in targets:
        try:
            atomic_write_json(path, report)
            written.append(str(path))
        except (OSError, ValueError) as exc:
            print(f'[check_env] WARNING: could not write {path}: {exc}', file=sys.stderr)
    return written


def _load_context(args) -> Context:
    ctx = Context(args=args)
    try:
        from hyak.common.config import load_config, repo_root
        from hyak.common.paths import Paths
        ctx.repo = repo_root()
        ctx.cfg = load_config(args.config)
        ctx.paths = Paths.from_config(ctx.cfg)
    except Exception as exc:  # noqa: BLE001 - reported by the config check
        ctx.cfg, ctx.paths = None, None
        ctx.config_error = f'{type(exc).__name__}: {exc}'
        if ctx.repo is None:
            ctx.repo = Path(__file__).resolve().parents[2]
    return ctx


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.env.check_env', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--list', action='store_true', help='list the checks and exit')
    parser.add_argument('--only', action='append', metavar='NAME[,NAME]', help='run only these checks')
    parser.add_argument('--skip', action='append', metavar='NAME[,NAME]', help='skip these checks')
    parser.add_argument('--no-gpu', action='store_true', help='skip nvidia_smi, torch_cuda, gpu_smoke (login node)')
    parser.add_argument('--no-network', action='store_true', help='skip the network check')
    parser.add_argument('--rust-library', type=Path, help='check this librust_utils.so instead of the staged one')
    parser.add_argument('--io-test-mib', type=int, default=1024,
                        help='MiB written and read back on ANYTOK_RUNS to measure throughput (0 = off)')
    parser.add_argument('--hash-model', action=argparse.BooleanOptionalAction, default=True,
                        help='hash the 3.4 GB model files, not just compare their sizes (default; takes seconds; '
                             '--no-hash-model skips it)')
    parser.add_argument('--probe-timeout', type=float, default=900.0, help='seconds per probe subprocess')
    parser.add_argument('--report', type=Path, help='also write the JSON report here')
    parser.add_argument('--no-status', action='store_true',
                        help='write no status files and no default reports (only --report)')
    args = parser.parse_args(argv)
    if args.list:
        for name, _, needs in CHECKS:
            print(f'{name}{"  (" + ", ".join(needs) + ")" if needs else ""}')
        return 0
    try:
        names = select_checks(args.only, args.skip)
    except ValueError as exc:
        parser.error(str(exc))
    ctx = _load_context(args)
    writer = None
    if not args.no_status:
        try:
            from hyak.common.config import stage_value
            from hyak.common.status import StatusWriter
            platform_text = stage_value(ctx.cfg, 'HYAK_PLATFORM', STAGE) if ctx.cfg else 'Hyak'
            monitor = ctx.paths.monitor_dir(STAGE) if ctx.paths is not None else None
            writer = StatusWriter(STAGE, len(names), 'checks', platform=platform_text, monitor_dir=monitor)
            writer.update(0, stage='running', workers=[writer.worker_row(task='starting checks', state='Checking')])
        except Exception as exc:  # noqa: BLE001 - status is best effort
            print(f'[check_env] WARNING: status files unavailable: {exc}', file=sys.stderr)
            writer = None

    def on_result(result, done):
        print(render(result), flush=True)
        if writer is not None:
            try:
                writer.update(len(done), stage='running',
                              workers=[writer.worker_row(task=f'check {result.name}: {result.status}', state='Checking')])
            except Exception:  # noqa: BLE001
                pass

    print(f'[check_env] {len(names)} checks on {socket.gethostname()} with {sys.executable}', flush=True)
    results = run_checks(ctx, names, on_result)
    report = build_report(ctx, results)
    written = write_reports(ctx, report)
    print(summary_line(results), flush=True)
    for path in written:
        print(f'[check_env] report: {path}', flush=True)
    code = exit_code(results)
    if writer is not None:
        try:
            if code:
                failed = [r for r in results if r.status == FAIL]
                writer.update(len(results), stage='failed', state='failed',
                              error='; '.join(f'{r.name}: {r.summary}' for r in failed)[:2000],
                              failed=[r.name for r in failed], warnings=report['warnings'])
            else:
                writer.update(len(results), stage='complete', state='complete', warnings=report['warnings'])
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == '__main__':
    sys.exit(main())
