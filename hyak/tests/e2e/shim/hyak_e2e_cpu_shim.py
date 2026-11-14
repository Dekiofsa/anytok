"""TEST-ONLY CUDA-to-CPU shim for running the pinned trainers on a CPU-only machine (hyak e2e harness).

Never imported by production code.  It is activated only by ``sitecustomize.py`` in this
directory when this directory is on ``PYTHONPATH`` *and* ``HYAK_E2E_CPU_SHIM=1``.

What it does, all in the child process only:

1. When ``torch`` is first imported, make the CUDA calls of the pinned trainers work on CPU:
   ``torch.cuda.is_available() -> True``, ``device_count() -> 1``, ``get_device_name`` names the
   shim, ``manual_seed_all`` / ``reset_peak_memory_stats`` are no-ops, ``max_memory_allocated``
   reports the process peak RSS, ``get_rng_state_all`` returns one dummy 16-byte state (the
   shape of one GPU's Philox state) and ``set_rng_state_all`` checks the length only.
   Device arguments ``"cuda"``, ``"cuda:N"``, ``torch.device("cuda")`` are rewritten to CPU in
   ``Tensor.to``, ``Module.to`` and the factory functions; ``Tensor.cuda`` / ``Module.cuda`` are
   identity.  ``torch.autocast("cuda", ...)`` becomes ``torch.autocast("cpu", ...)`` (same dtype),
   so training and the trainer's evaluation (which uses ``source.device.type``, i.e. CPU) both
   run under CPU bf16 autocast.  ``HYAK_E2E_AUTOCAST=off`` disables autocast instead.
   ``HYAK_E2E_TORCH_THREADS=N`` replaces the trainer's ``torch.set_num_threads(1)`` by N.
   ``HYAK_E2E_DROPOUT=0`` forces every ``F.dropout`` probability to 0 (used only for the
   chunked-forward agreement check; the trainer's dropout is otherwise untouched).
2. ``python -m hyak.train.run_segment`` (also the ``--smoke`` children) is run through its
   module ``main()`` with the wrapper's explicit TEST parameters injected into ``run_segment``:
   ``init_bytes`` (``HYAK_E2E_INIT_BYTES``), ``plan_file_sha256`` (``HYAK_E2E_PLAN_SHA256``),
   ``model_expected`` (the tiny model's hash table, ``HYAK_E2E_MODEL_HASHES``) and ``allow_test_scale``
   (``HYAK_E2E_ALLOW_TEST_SCALE=1``: the miniature corpus is a test-scale stage),
   because the miniature world has a tiny initial hypernetwork and a 1,000-entry TEST plan.
3. ``python -m hyak.eval.{prepare_features,validate_prefixes,score}`` are run through ``main()`` with the
   explicit TEST parameters ``pool_reference_plan`` (``HYAK_E2E_POOL_PLAN``) and, for the scorer,
   ``model_hashes`` (``HYAK_E2E_MODEL_HASHES``); the scorer also gets ``--checkpoint`` from
   ``HYAK_E2E_EVAL_CHECKPOINT_<RUN>`` when its caller (the real 09_eval_score.sbatch) passes none.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import resource
import sys

SHIM_DEVICE_NAME = 'CPU via hyak e2e test shim (not a GPU)'
_PATCHED = {'torch': False, 'runpy': False}


# --- torch ------------------------------------------------------------------------------------------------
def _is_cuda(value) -> bool:
    import torch
    if isinstance(value, torch.device):
        return value.type == 'cuda'
    if isinstance(value, str):
        return value == 'cuda' or value.startswith('cuda:')
    return False


def _cpu(value):
    import torch
    return torch.device('cpu') if _is_cuda(value) else value


def _rewrite(args, kwargs, positional_device=True):
    if 'device' in kwargs:
        kwargs['device'] = _cpu(kwargs['device'])
    if positional_device and args and _is_cuda(args[0]):
        args = (_cpu(args[0]),) + tuple(args[1:])
    return args, kwargs


def patch_torch(torch) -> None:
    if _PATCHED['torch']:
        return
    _PATCHED['torch'] = True
    cuda = torch.cuda

    cuda.is_available = lambda: True
    cuda.device_count = lambda: 1
    cuda.current_device = lambda: 0
    cuda.get_device_name = lambda device=None: SHIM_DEVICE_NAME
    cuda.is_bf16_supported = lambda *a, **k: True
    cuda.manual_seed_all = lambda seed: None
    cuda.manual_seed = lambda seed: None
    cuda.synchronize = lambda device=None: None
    cuda.empty_cache = lambda: None
    cuda.reset_peak_memory_stats = lambda device=None: None

    def max_memory_allocated(device=None):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == 'darwin' else peak * 1024   # macOS reports bytes, Linux KiB

    cuda.max_memory_allocated = max_memory_allocated
    cuda.max_memory_reserved = max_memory_allocated
    cuda.memory_allocated = lambda device=None: 0
    cuda.get_rng_state_all = lambda: [torch.zeros(16, dtype=torch.uint8)]

    def set_rng_state_all(states):
        if len(states) != 1:
            raise RuntimeError(f'hyak e2e shim: expected one CUDA RNG state, got {len(states)}')

    cuda.set_rng_state_all = set_rng_state_all

    tensor_to = torch.Tensor.to

    @functools.wraps(tensor_to)
    def to(self, *args, **kwargs):
        args, kwargs = _rewrite(args, kwargs)
        return tensor_to(self, *args, **kwargs)

    torch.Tensor.to = to
    torch.Tensor.cuda = lambda self, *a, **k: self
    module_to = torch.nn.Module.to

    @functools.wraps(module_to)
    def module_to_cpu(self, *args, **kwargs):
        args, kwargs = _rewrite(args, kwargs)
        return module_to(self, *args, **kwargs)

    torch.nn.Module.to = module_to_cpu
    torch.nn.Module.cuda = lambda self, *a, **k: self

    for name in ('tensor', 'as_tensor', 'ones', 'zeros', 'full', 'empty', 'arange', 'rand', 'randn', 'randint',
                 'randperm', 'linspace', 'eye', 'ones_like', 'zeros_like', 'full_like', 'empty_like'):
        original = getattr(torch, name)

        def factory(*args, __original=original, **kwargs):
            if 'device' in kwargs:
                kwargs['device'] = _cpu(kwargs['device'])
            return __original(*args, **kwargs)

        functools.update_wrapper(factory, original)
        setattr(torch, name, factory)

    mode = os.environ.get('HYAK_E2E_AUTOCAST', 'bf16')
    base_autocast = torch.amp.autocast

    class CpuAutocast(base_autocast):
        """``torch.autocast("cuda", ...)`` of the pinned trainer, run as CPU autocast."""

        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            if device_type == 'cuda':
                device_type = 'cpu'
            if mode == 'off':
                enabled = False
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    torch.autocast = CpuAutocast   # torch.amp.autocast (used by torch.utils.checkpoint) is left alone

    threads = os.environ.get('HYAK_E2E_TORCH_THREADS')
    if threads:
        original_threads = torch.set_num_threads
        torch.set_num_threads = lambda _n, __set=original_threads: __set(int(threads))

    if os.environ.get('HYAK_E2E_DROPOUT') == '0':
        import torch.nn.functional as F
        original_dropout = F.dropout

        def no_dropout(input, p=0.5, training=True, inplace=False):
            return original_dropout(input, 0.0, training, inplace)

        F.dropout = no_dropout
        torch.nn.Dropout.forward = lambda self, input: input


class _TorchFinder(importlib.abc.MetaPathFinder):
    """Patch ``torch`` right after its first import (this sitecustomize stays cheap for other processes)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != 'torch' or _PATCHED['torch']:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
            patch_torch(module)

        loader.exec_module = exec_module
        return spec


# --- run_segment test parameters -------------------------------------------------------------------------
def inject_run_segment(module) -> None:
    """Wrap ``module.run_segment`` so it receives the explicit TEST parameters of the miniature world."""
    extra = {}
    if os.environ.get('HYAK_E2E_INIT_BYTES'):
        extra['init_bytes'] = int(os.environ['HYAK_E2E_INIT_BYTES'])
    if os.environ.get('HYAK_E2E_PLAN_SHA256'):
        extra['plan_file_sha256'] = os.environ['HYAK_E2E_PLAN_SHA256']
    if os.environ.get('HYAK_E2E_MODEL_HASHES'):
        import json
        from pathlib import Path
        extra['model_expected'] = json.loads(Path(os.environ['HYAK_E2E_MODEL_HASHES']).read_text())
    if os.environ.get('HYAK_E2E_ALLOW_TEST_SCALE') == '1':
        extra['allow_test_scale'] = True
    original = module.run_segment
    if getattr(original, 'hyak_e2e_wrapped', False):
        return

    @functools.wraps(original)
    def run_segment(*args, **kwargs):
        for key, value in extra.items():
            kwargs.setdefault(key, value)
        return original(*args, **kwargs)

    run_segment.hyak_e2e_wrapped = True
    module.run_segment = run_segment


# --- evaluation entry points: test parameters ---------------------------------------------------------------
#: Entry points whose ``main()`` takes the explicit TEST parameters of a miniature world.
EVAL_MODULES = ('hyak.eval.prepare_features', 'hyak.eval.validate_prefixes', 'hyak.eval.score')


def eval_test_parameters(mod_name: str) -> dict:
    """``pool_reference_plan`` = the world's TEST plan (``HYAK_E2E_POOL_PLAN``; the byte-copy check stays on) and,
    for the scorer, ``model_hashes`` = the tiny model's {bytes, sha256} table (``HYAK_E2E_MODEL_HASHES``)."""
    import json
    from pathlib import Path
    params = {}
    if os.environ.get('HYAK_E2E_POOL_PLAN'):
        params['pool_reference_plan'] = Path(os.environ['HYAK_E2E_POOL_PLAN'])
    if mod_name == 'hyak.eval.score' and os.environ.get('HYAK_E2E_MODEL_HASHES'):
        params['model_hashes'] = json.loads(Path(os.environ['HYAK_E2E_MODEL_HASHES']).read_text())
    return params


def eval_argv(mod_name: str, argv: list) -> list:
    """The scorer's ``--checkpoint`` from ``HYAK_E2E_EVAL_CHECKPOINT_<RUN>`` when the caller passed none.

    The production batch script always scores ``<run>/step-020000``; a miniature run never gets there, so
    the harness names the checkpoint (an existing CLI option) for the real 09_eval_score.sbatch.  That
    checkpoint is not the final step, which hyak.eval.score refuses (exit 2) without
    ``--allow-non-final-checkpoint``, so the flag is added with it.
    """
    if mod_name != 'hyak.eval.score' or '--checkpoint' in argv or '--run' not in argv:
        return argv
    run = argv[argv.index('--run') + 1]
    checkpoint = os.environ.get(f'HYAK_E2E_EVAL_CHECKPOINT_{run.upper()}')
    if not checkpoint:
        return argv
    extra = ['--checkpoint', checkpoint]
    if '--allow-non-final-checkpoint' not in argv:
        extra.append('--allow-non-final-checkpoint')
    return argv + extra


def patch_runpy() -> None:
    """``python -m hyak.train.run_segment`` / ``hyak.eval.*``: import the module normally, inject, run ``main()``."""
    if _PATCHED['runpy']:
        return
    _PATCHED['runpy'] = True
    import runpy
    original = runpy._run_module_as_main

    def run_module_as_main(mod_name, alter_argv=True):
        if mod_name not in ('hyak.train.run_segment',) + EVAL_MODULES:
            return original(mod_name, alter_argv)
        import importlib
        module = importlib.import_module(mod_name)
        if alter_argv:
            sys.argv[0] = module.__file__
        if mod_name in EVAL_MODULES:
            sys.exit(module.main(eval_argv(mod_name, sys.argv[1:]), **eval_test_parameters(mod_name)))
        inject_run_segment(module)
        sys.exit(module.main())

    runpy._run_module_as_main = run_module_as_main


def install() -> None:
    if 'torch' in sys.modules:
        patch_torch(sys.modules['torch'])
    else:
        sys.meta_path.insert(0, _TorchFinder())
    patch_runpy()
