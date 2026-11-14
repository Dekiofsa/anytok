"""Build the randomly initialized input hypernetwork that both runs load as ``--warmup-checkpoint``.

DESIGN.md decision D1 and section 5.B.  Usage (train env, CPU is enough, about 4.6 GB RAM, ~20 s)::

    python -m hyak.model.init_hypernet [--model-dir DIR] [--code-dir DIR] [--out DIR]
        [--seed 42] [--prior-bias-std 0.02] [--calibrate-after-bias] [--no-rescaler-init]
        [--allow-unpinned-model] [--force] [--threads N] [--status] [--config FILE]
    python -m hyak.model.init_hypernet --verify [--out DIR] [--config FILE]

Defaults come from ``hyak/config.env``: ``--model-dir`` = ``$ANYTOK_ROOT/input/model``,
``--code-dir`` = ``$ANYTOK_ROOT/code/uniform`` (the staged pinned tree), ``--out`` =
``$ANYTOK_ROOT/input/hypernet_init``.  For local tests ``--code-dir`` may point at
``runtime/anytok_pool_20k`` (the ``hf_hypernet`` modules used here need no Rust library).
``--verify --out DIR`` needs no config file.

Recipe (D1, default "calibrate first", option (b)), in this order:

1. ``torch.manual_seed(seed)``; ``hn = build_input_hypernet(49152, 2048)`` with every other
   argument at its default (``runtime/anytok_pool_20k/hf_hypernet/identity_warmup.py:10-46``).
2. ``initialize_rescalers(hn, table)`` with the library's zero prior bias -- exactly the
   original pre-warmup construction -- on the float32 upcast of SmolLM2's
   ``model.embed_tokens.weight`` (stored BF16) and the default 2,048 anchors
   (``identity_warmup.py:49-57``, ``modeling_hypernet.py:225-244``).  The calibration check
   runs on this zero-bias network.  With seed 42 the computed ``in_scaler`` and ``scaler`` must
   then match the public checkpoint (``original_in_scaler.npz``) within 1e-5 (else INIT FAILED,
   exit 4), and the four frozen rescaler tensors are COPIED from that fixture, so they are
   bit-identical to the original runs' on any machine.
3. ``hn.model.target_priors_projection.bias ~ N(0, prior_bias_std**2)`` drawn with
   ``torch.nn.init.normal_`` from a separate ``torch.Generator`` seeded ``seed + 1000``, so no
   other tensor depends on this option.  Reason: with an exactly zero bias, zero priors (BPE and
   WordPiece, 660 of 1,000 pool tokenizers) feed an all-zero vector into the embedding LayerNorm
   (eps 1e-12) and its gradient dominates the trainer's joint 0.1 clip (DESIGN.md 1.2).
4. ``hn.save_pretrained(dir, safe_serialization=True)`` BEFORE any byte-fallback expansion:
   the trainer expands the fallback rows itself on its first start (``tokenizer_inputs.py:21-39``).

Options: ``--calibrate-after-bias`` (option (a)) draws the bias first and calibrates afterwards,
so the anchors are calibrated with the drawn bias but the frozen output ``scaler`` differs from
the original's for all 20,000 updates; ``--prior-bias-std 0`` (option (c)) keeps the library zero
bias, i.e. exactly the original pre-warmup state, with the degenerate zero-prior gradient (a
tokenizer-family-dependent weighting of the updates; not recommended for uniform versus RL);
``--no-rescaler-init`` leaves identity rescalers (not recommended).  The option, the seed and the
fixture digest are recorded in ``init_manifest.json`` (``recipe``) and in the stage status.

Before the directory appears (it is built in a sibling temporary directory and renamed into
place), this tool verifies: the pinned package versions and ``hf_hypernet`` source hashes; the
model files against ``plans/zett-modal-20k-inputs.json`` (a mismatch stops the production build
unless ``--allow-unpinned-model``); the construction-time initialization (normal(0, 0.02)
weights, zero biases, unit LayerNorms, identity rescalers: what transformers 5.15.1 does and what
the port was validated with); the config fields the trainer relies on; 138 float32 tensors,
344,182,784 parameters, ``model.safetensors`` exactly 1,376,746,800 bytes; every tensor finite;
per-tensor SHA-256 of the file and of a ``ZettHypernet.from_pretrained(local_files_only=True)``
reload equal to the in-memory model; the reloaded model computes the same anchor outputs (up to
float rounding: rtol 1e-3, atol 1e-4; the maximum and relative differences are recorded);
``in_scaler.{w,b}`` against the original public checkpoint; a zero-prior gradient diagnostic;
and an emulation of what the trainer does next
(``runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py:421-446`` and its first
checkpoint save at L218): 21 byte fallbacks, ``config.json`` byte-identical to the config of
every original trained checkpoint (SHA-256 ``22cb4f86...``, 2,382 bytes) and
``model.safetensors`` exactly 1,376,918,848 bytes, then a resume-style reload that must not
expand again.  Everything is recorded in ``init_manifest.json``.

The calibration check (``checks.calibration``) covers only the 2,048 anchor rows (token ids
0-2047 as one-piece forms, zero priors, one batch); on realistic vocabularies every option
starts with an offset that the first few hundred updates remove.

An existing directory is never overwritten without ``--force``; an existing valid directory
built with the same recipe and inputs is accepted as is (exit 0).  ``--force`` is refused
while ``$ANYTOK_RUNS/uniform`` or ``$ANYTOK_RUNS/rl`` holds anything (both runs depend on it).

Exit codes (project contract): 0 built or already built (``--verify``: VERIFIED); 2 usage,
configuration, input or environment error (INIT FAILED with a remedy); 4 a check failed (INIT
FAILED; ``--verify``: NOT VERIFIED); 1 an unexpected error (traceback); 128 + signal when stopped.

Determinism: two builds with the same seed, package versions, machine and thread count are
bit-identical (tested at small size).  With the default recipe the frozen rescalers come from the
fixture, so they are identical on every machine; other tensors come from torch's CPU generator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from hyak.common.io import atomic_write_json, file_record, fsync_dir, read_json, sha256_bytes, utc_now

# --- Fixed facts of this experiment ---------------------------------------------------------------

#: Canonical SmolLM2-1.7B vocabulary (len(tokenizer), with added tokens) and embedding width.
SOURCE_VOCAB = 49152
EMBED_WIDTH = 2048
#: Input-table key in SmolLM2's model.safetensors (the LM loads it with dtype=float32, T:424-427).
TABLE_KEY = 'model.embed_tokens.weight'
SPECIAL_TOKEN = '<|endoftext|>'   # T:48
SPECIAL_ID = 0                    # T:49
MAX_PIECES = 10                   # T:446 asserts hn_surface_maxlen == 10

#: Recorded size of the original warmup model.safetensors (plans/zett-modal-20k-inputs.json:223-225).
EXPECTED_MODEL_BYTES = 1_376_746_800
#: Size of every original trained hypernet/model.safetensors (22 fallback rows), from
#: plans/uniform-trajectory-inventory.json and reports/.../launch-verification.json.
EXPECTED_TRAINED_MODEL_BYTES = 1_376_918_848
#: SHA-256 of every original trained checkpoint's hypernet/config.json (2,382 bytes), uniform
#: and RL (plans/uniform-trajectory-inventory.json:18-20); equal to the public step-500 config.
ORIGINAL_TRAINED_CONFIG_SHA256 = '22cb4f86771ec9ca181ff9b32017981707d08f7848f6923f91028c708aca2c8d'
ORIGINAL_TRAINED_CONFIG_BYTES = 2382
#: The original warmup directory (plans/zett-modal-20k-inputs.json:219-226); informative only,
#: a random start cannot reproduce it.
ORIGINAL_WARMUP_CONFIG_SHA256 = '864a608717ca78bb73b059caa5bf47cb4ceb00a3a94cc4e406a57cd8601f4cb8'
ORIGINAL_WARMUP_MODEL_SHA256 = 'a41fb71cadc201ae6bbdfa5baf5abb225f58da1b7a8f4b9fc70b7575f2b0647b'
EXPECTED_TENSORS = 138
EXPECTED_PARAMETERS = 344_182_784
EXPECTED_FROZEN_PARAMETERS = 8_192
#: SmolLM2's tokenizer lacks 21 of the 256 ByteLevel symbols (tokenizer_inputs.py:12-18).
EXPECTED_FALLBACK_SYMBOLS = 21

DEFAULT_SEED = 42
PRIOR_BIAS_SEED_OFFSET = 1000
DIAGNOSTIC_SEED_OFFSET = 2000
DEFAULT_PRIOR_BIAS_STD = 0.02
DEFAULT_ANCHORS = 2048          # identity_warmup.py:50 default
IN_SCALER_TOLERANCE = 1e-6
PRIOR_BIAS_NAME = 'model.target_priors_projection.bias'
RESCALER_NAMES = ('in_scaler.w', 'in_scaler.b', 'scaler.w', 'scaler.b')
#: The degenerate case: the prior bias gradient alone exceeds all other gradients together.
DEGENERATE_RATIO = 1.0

#: Exact versions of the train env (DESIGN.md 5.A); the construction-time init depends on them.
PINNED_PACKAGES = {'torch': '2.15.0', 'transformers': '5.15.1', 'tokenizers': '0.22.2',
                   'numpy': '2.3.5', 'safetensors': '0.8.0'}
PINNED_PYTHON = (3, 12)

#: SHA-256 of the pinned hf_hypernet files this tool imports (runtime/anytok_pool_20k; equal in
#: the staged code/uniform tree).  NOTES/hypernetwork.md section 2.4.
PINNED_SOURCES = {
    'hf_hypernet/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'hf_hypernet/configuration_hypernet.py': 'e986d140ad63311b63425a3cac6e4af2410ed180c184847aca80e0ab6af554de',
    'hf_hypernet/hyper_roberta.py': '5c385150184fd0a55fb45a289a4eed2b6b7756880849f080c4f55b25355b8512',
    'hf_hypernet/hyper_t5.py': 'c40548e12cb2f2074ead8ce7876cf40c1ed77b057e735c0b6f9a8027b3e50eb2',
    'hf_hypernet/identity_warmup.py': '1c5cca36d2a2b3cdfeb7e2d3fbb8efdc300c69fd6f0bee47e2876d49b8ff5666',
    'hf_hypernet/modeling_hypernet.py': '9314094415250b093040f27948434f949811913d5a82b77c5676681361031541',
    'hf_hypernet/tokenizer_inputs.py': '70e99d9edb56fead0597a2d593dc992a2ad3b85d688c09bfe953bcc7d42077c8',
}
_MODULES = ('hf_hypernet', 'hf_hypernet.configuration_hypernet', 'hf_hypernet.hyper_t5',
            'hf_hypernet.hyper_roberta', 'hf_hypernet.modeling_hypernet', 'hf_hypernet.identity_warmup',
            'hf_hypernet.tokenizer_inputs')

#: The three files of the model directory (the original held exactly these, NOTES/hypernetwork.md 3.7).
MODEL_FILES = ('config.json', 'model.safetensors', 'tokenizer.json')
#: Files of a finished init directory.
OUTPUT_FILES = ('config.json', 'model.safetensors')
MANIFEST_NAME = 'init_manifest.json'
MANIFEST_FORMAT = 'hyak-hypernet-init-v1'

#: Reference rescaler of the original pipeline (built from the public checkpoint; see load_reference).
REFERENCE_NPZ = Path(__file__).with_name('original_in_scaler.npz')
REFERENCE_REPOSITORY = 'qingy2024/anytok-zett-smollm2-20k'
REFERENCE_MODEL_FILE = 'checkpoints/step-000500/hypernet/model.safetensors'
REFERENCE_CONFIG_FILE = 'checkpoints/step-000500/hypernet/config.json'
REFERENCE_STEP = 500
REFERENCE_W_SHA256 = 'd861123f58f33ef41efc21c912c2515b6f7732d89de7c1af7f9daca09d794430'
REFERENCE_B_SHA256 = 'c48aa64734fd24957694563de3bc1777dbe905dd16ec09be941dad1b62ee0de6'
REFERENCE_SCALER_W_SHA256 = '10b6a05b92426bec607a85e9b0bb26e2fa757dad95b0e547417f62f095e15f37'
REFERENCE_SCALER_B_SHA256 = '4864671f0f9bd845b7a085cd79f9573bc6d9bf574282c14cdd5de8ac0e7a7b91'
#: The original warmup's starting point is reproduced by the construction with this seed and a zero
#: prior bias (its frozen scaler matches to ~5e-7, measured for this port); informative check only.
ORIGINAL_CONSTRUCTION_SEED = 42
ORIGINAL_CONSTRUCTION_TOLERANCE = 1e-5
#: Calibration order: the default calibrates the zero-bias construction, then draws the prior bias
#: (option (b)); ``--calibrate-after-bias`` draws first and calibrates the result (option (a)).
CALIBRATE_BEFORE_BIAS = 'before_bias'
CALIBRATE_AFTER_BIAS = 'after_bias'
#: The line ``save_pretrained`` writes for the field the original warmup config did not have yet.
FALLBACK_CONFIG_LINE = b'  "hn_byte_fallback_symbols": null,\n'
#: Reload check: anchor outputs of the reloaded model (weights at another alignment) may differ by rounding.
RELOAD_RTOL, RELOAD_ATOL = 1e-3, 1e-4
#: Temporary build directories of another node older than this are leftovers of a hard kill.
STALE_SECONDS = 6 * 3600

#: Progress steps reported to status.json (stage ``init``) with ``--status``.
STEPS = ('check inputs', 'build', 'save', 'verify files', 'reload', 'gradient diagnostic',
         'trainer check', 'publish')


class InitError(RuntimeError):
    """A check failed; the message says what to do (exit code 4)."""


class InitUsageError(InitError):
    """Usage, configuration, input or environment error; the message names the remedy (exit code 2)."""


class Terminated(BaseException):
    """SIGTERM/SIGINT arrived; raised so temporary directories are removed before exiting."""


@dataclass(frozen=True)
class Profile:
    """Sizes and reference checks.  ``PRODUCTION`` is the only profile the command line uses.

    Small test profiles (other widths) keep every structural check but cannot be compared with
    the original checkpoint (``reference=False``) and have no fixed byte counts.
    """

    source_vocab: int = SOURCE_VOCAB
    width: int = EMBED_WIDTH
    expected_model_bytes: int | None = EXPECTED_MODEL_BYTES
    expected_trained_model_bytes: int | None = EXPECTED_TRAINED_MODEL_BYTES
    expected_tensors: int | None = EXPECTED_TENSORS
    expected_parameters: int | None = EXPECTED_PARAMETERS
    expected_frozen: int | None = EXPECTED_FROZEN_PARAMETERS
    expected_fallback_symbols: int | None = EXPECTED_FALLBACK_SYMBOLS
    reference: bool = True
    diagnostic_vocab: int = 256
    diagnostic_one_piece: int = 64


PRODUCTION = Profile()


def _log(message: str) -> None:
    print(f'[init] {message}', flush=True)


# --- Environment and code ---------------------------------------------------------------------------

def _base_version(version: str) -> str:
    return str(version).split('+', 1)[0]


def check_environment(strict: bool = True) -> dict:
    """Python and package versions; with ``strict`` raise InitError unless they are the pins.

    A local build suffix of torch (for example ``2.15.0+cu132`` or ``2.15.0+cu134`` from
    ``HYAK_TORCH_INDEX_URL``; the plain PyPI wheel ``2.15.0`` is the CUDA 13.2 build) is accepted.
    """
    import safetensors
    import tokenizers
    import transformers
    found = {'torch': torch.__version__, 'transformers': transformers.__version__,
             'tokenizers': tokenizers.__version__, 'numpy': np.__version__,
             'safetensors': safetensors.__version__}
    try:
        import accelerate
        found['accelerate'] = accelerate.__version__
    except ImportError:
        found['accelerate'] = None
    problems = [f'{name} {found[name]} (expected {pin})' for name, pin in PINNED_PACKAGES.items()
                if _base_version(found[name]) != pin]
    if tuple(sys.version_info[:2]) != PINNED_PYTHON:
        problems.append(f'Python {platform.python_version()} (expected {PINNED_PYTHON[0]}.{PINNED_PYTHON[1]})')
    if problems and strict:
        raise InitUsageError('The construction-time initialization depends on the pinned packages, but this '
                        f'environment has {"; ".join(problems)}. Remedy: run inside the train env '
                        '(source hyak/slurm/lib.sh; hyak_activate train), see hyak/env/setup_train_env.sh')
    return {'python': platform.python_version(), 'packages': found, 'pins_ok': not problems,
            'platform': platform.platform(), 'machine': platform.machine(), 'host': socket.gethostname(),
            'torch_threads': torch.get_num_threads(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}


def verify_code_dir(code_dir) -> dict:
    """SHA-256 of the pinned ``hf_hypernet`` files under ``code_dir``; InitError on any difference."""
    code_dir = Path(os.path.abspath(code_dir))
    hashes, problems = {}, []
    for rel, expected in PINNED_SOURCES.items():
        path = code_dir / rel
        if not path.is_file():
            problems.append(f'{rel} is missing')
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[rel] = digest
        if digest != expected:
            problems.append(f'{rel} has SHA-256 {digest}, expected {expected}')
    if problems:
        raise InitUsageError(f'Code directory {code_dir} is not the pinned hf_hypernet snapshot: '
                        f'{"; ".join(problems)}. Remedy: re-stage it with python -m hyak.code.stage_code '
                        '(or pass --code-dir runtime/anytok_pool_20k of an unmodified checkout)')
    return {'code_dir': str(code_dir), 'files': hashes}


def import_hf_hypernet(code_dir) -> SimpleNamespace:
    """Import the pinned hypernetwork code from ``code_dir`` (after :func:`verify_code_dir`).

    Bytecode is never written (``sys.dont_write_bytecode``).  Raises InitError when
    ``hf_hypernet`` is already imported from another directory in this process.
    """
    info = verify_code_dir(code_dir)
    code_dir = Path(info['code_dir'])
    package = os.path.realpath(code_dir / 'hf_hypernet')
    loaded = sys.modules.get('hf_hypernet')
    if loaded is not None:
        origin = os.path.realpath(os.path.dirname(getattr(loaded, '__file__', '') or ''))
        if origin != package:
            raise InitUsageError(f'hf_hypernet is already imported from {origin}, not {package}; '
                            'run this tool in a fresh Python process')
    elif str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        modules = {name: importlib.import_module(name) for name in _MODULES}
    finally:
        sys.dont_write_bytecode = previous
    for name, module in modules.items():
        where = os.path.realpath(os.path.dirname(module.__file__))
        if where != package:
            raise InitUsageError(f'{name} was imported from {where}, expected {package}; '
                            'remove other copies of hf_hypernet from PYTHONPATH')
    identity, modeling, inputs = (modules['hf_hypernet.identity_warmup'], modules['hf_hypernet.modeling_hypernet'],
                                  modules['hf_hypernet.tokenizer_inputs'])
    return SimpleNamespace(
        build_input_hypernet=identity.build_input_hypernet,
        initialize_rescalers=identity.initialize_rescalers,
        freeze_rescalers=identity.freeze_rescalers,
        ZettHypernet=modeling.ZettHypernet,
        Rescaler=modeling.Rescaler,
        enable_byte_fallbacks=inputs.enable_byte_fallbacks,
        missing_byte_symbols=inputs.missing_byte_symbols,
        code=info,
    )


# --- Inputs -------------------------------------------------------------------------------------------

def _pinned_model_records(repo: Path | None = None) -> dict:
    """``files["model/<name>"]`` of plans/zett-modal-20k-inputs.json (empty if unreadable)."""
    if repo is None:
        from hyak.common.config import repo_root
        repo = repo_root()
    try:
        files = read_json(Path(repo) / 'plans' / 'zett-modal-20k-inputs.json')['files']
        return {name: files[f'model/{name}'] for name in MODEL_FILES if f'model/{name}' in files}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def tokenizer_info(tokenizer_json) -> dict:
    """Vocabulary size with added tokens and the end-of-text id of a ``tokenizer.json``."""
    from tokenizers import Tokenizer
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
    except Exception as exc:  # tokenizers raises its own exception type
        raise InitUsageError(f'Cannot parse {tokenizer_json}: {exc}. Remedy: python -m hyak.model.fetch_model') from exc
    return {'vocab_size_with_added': tokenizer.get_vocab_size(with_added_tokens=True),
            'vocab_size_without_added': tokenizer.get_vocab_size(with_added_tokens=False),
            'eos_id': tokenizer.token_to_id(SPECIAL_TOKEN)}


def inspect_model_dir(model_dir, profile: Profile = PRODUCTION, *, repo: Path | None = None,
                      hash_files: bool = True) -> dict:
    """Check the model directory (three files, config and tokenizer facts) and record its files.

    The SHA-256 values are compared with ``plans/zett-modal-20k-inputs.json`` (``matches_pinned``,
    and ``pinned_sha256`` on a mismatch); :func:`create` refuses a mismatch for the production
    profile unless ``allow_unpinned_model``.  The content this tool uses is also checked directly
    (table shape, vocabulary, rescaler regression, byte fallbacks).
    """
    model_dir = Path(os.path.abspath(model_dir))
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise InitUsageError(f'{model_dir} lacks {", ".join(missing)}. Remedy: python -m hyak.model.fetch_model '
                        '(it downloads the pinned SmolLM2-1.7B config.json, model.safetensors, tokenizer.json)')
    extra = sorted(p.name for p in model_dir.iterdir() if p.name not in MODEL_FILES)
    config = read_json(model_dir / 'config.json')
    problems = []
    for key, expected in (('vocab_size', profile.source_vocab), ('bos_token_id', SPECIAL_ID),
                          ('eos_token_id', SPECIAL_ID)):
        if config.get(key) != expected:
            problems.append(f'config.json {key} is {config.get(key)!r}, expected {expected}')
    if profile.width == EMBED_WIDTH and config.get('hidden_size') != EMBED_WIDTH:
        problems.append(f'config.json hidden_size is {config.get("hidden_size")!r}, expected {EMBED_WIDTH}')
    tok = tokenizer_info(model_dir / 'tokenizer.json')
    if tok['vocab_size_with_added'] != profile.source_vocab:
        problems.append(f'tokenizer.json has {tok["vocab_size_with_added"]} tokens including added tokens, '
                        f'expected {profile.source_vocab}')
    if tok['eos_id'] != SPECIAL_ID:
        problems.append(f'tokenizer.json maps {SPECIAL_TOKEN} to {tok["eos_id"]!r}, expected {SPECIAL_ID}')
    if problems:
        raise InitUsageError(f'{model_dir} is not the pinned SmolLM2-1.7B directory: {"; ".join(problems)}. '
                        'Remedy: python -m hyak.model.fetch_model')
    pinned = _pinned_model_records(repo)
    files = {}
    for name in MODEL_FILES:
        record = file_record(model_dir / name) if hash_files else {'bytes': (model_dir / name).stat().st_size}
        if name in pinned and hash_files:
            record['matches_pinned'] = record == {k: pinned[name][k] for k in ('bytes', 'sha256')}
            if not record['matches_pinned']:
                record['pinned_sha256'] = pinned[name].get('sha256')
        files[name] = record
    return {'model_dir': str(model_dir), 'files': files, 'extra_files': extra, 'tokenizer': tok,
            'config': {k: config.get(k) for k in ('vocab_size', 'hidden_size', 'bos_token_id', 'eos_token_id',
                                                   'tie_word_embeddings', 'torch_dtype')}}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy()).hexdigest()


def load_source_table(model_file, profile: Profile = PRODUCTION) -> tuple:
    """``(table, info)``: SmolLM2's input table as float32, as the trainer's ``source``.

    The trainer loads the LM with ``dtype=torch.float32`` (T:424-427) and uses
    ``lm.get_input_embeddings().weight.detach()`` (T:449), a copy of the tied
    ``model.embed_tokens.weight``; upcasting the stored BF16 values to float32 is exact.
    """
    from safetensors import safe_open
    model_file = Path(model_file)
    try:
        with safe_open(str(model_file), framework='pt') as handle:
            if TABLE_KEY not in set(handle.keys()):
                raise InitUsageError(f'{model_file} has no tensor {TABLE_KEY}. Remedy: python -m hyak.model.fetch_model')
            sliced = handle.get_slice(TABLE_KEY)
            shape, dtype = tuple(sliced.get_shape()), sliced.get_dtype()
            expected = (profile.source_vocab, profile.width)
            if shape != expected:
                raise InitUsageError(f'{TABLE_KEY} in {model_file} has shape {list(shape)}, expected {list(expected)}. '
                                     'Remedy: python -m hyak.model.fetch_model')
            if dtype not in ('BF16', 'F16', 'F32'):
                raise InitUsageError(f'{TABLE_KEY} in {model_file} has dtype {dtype}; expected BF16 (SmolLM2 stores '
                                     'BF16). Remedy: python -m hyak.model.fetch_model')
            raw = handle.get_tensor(TABLE_KEY)
            metadata = handle.metadata()
    except InitError:
        raise
    except Exception as exc:
        raise InitUsageError(f'Cannot read {TABLE_KEY} from {model_file}: {exc}. '
                             'Remedy: python -m hyak.model.fetch_model') from exc
    info = {'key': TABLE_KEY, 'shape': list(shape), 'stored_dtype': dtype, 'stored_sha256': _tensor_sha256(raw),
            'metadata': metadata}
    table = raw.to(torch.float32).contiguous()
    del raw
    if not bool(torch.isfinite(table).all()):
        raise InitUsageError(f'{TABLE_KEY} in {model_file} contains non-finite values. '
                             'Remedy: python -m hyak.model.fetch_model')
    info['float32_sha256'] = _tensor_sha256(table)
    if dtype != 'BF16':
        _log(f'WARNING: {TABLE_KEY} is stored as {dtype}, not BF16 as in the pinned SmolLM2 file')
    return table, info


# --- Reference of the original pipeline ------------------------------------------------------------

def _row(npy) -> np.ndarray:
    return np.ascontiguousarray(np.load(npy, allow_pickle=False), dtype='<f4').reshape(1, -1)


def write_reference_npz(w_npy, b_npy, config_json, out, *, scaler_w_npy, scaler_b_npy) -> dict:
    """Create ``original_in_scaler.npz`` from arrays and the config extracted from the public checkpoint.

    ``w_npy``/``b_npy``: float32 [2048] arrays of ``in_scaler.w``/``in_scaler.b`` of
    ``REFERENCE_MODEL_FILE`` in ``REFERENCE_REPOSITORY``; ``scaler_w_npy``/``scaler_b_npy``: its
    ``scaler.w``/``scaler.b``; ``config_json``: that checkpoint's ``hypernet/config.json``.  Used
    once to build the committed fixture; kept for provenance.
    """
    text = Path(config_json).read_bytes().decode('utf-8')
    arrays = {
        'w': _row(w_npy), 'b': _row(b_npy), 'scaler_w': _row(scaler_w_npy), 'scaler_b': _row(scaler_b_npy),
        'repository': np.array(REFERENCE_REPOSITORY), 'model_file': np.array(REFERENCE_MODEL_FILE),
        'tensor_w': np.array('in_scaler.w'), 'tensor_b': np.array('in_scaler.b'),
        'tensor_scaler_w': np.array('scaler.w'), 'tensor_scaler_b': np.array('scaler.b'),
        'step': np.array(REFERENCE_STEP, dtype=np.int64),
        'config_file': np.array(REFERENCE_CONFIG_FILE), 'config_json': np.array(text),
        'note': np.array('in_scaler.w/b, scaler.w/b and hypernet/config.json of the public step-500 checkpoint of '
                         'the original ZeTT-baseline run (same hypernetwork code and SmolLM2 table). The rescalers are '
                         'frozen in warmup and training, so they are the values computed when the original warmup '
                         'started; recomputing Rescaler.scale_to(table, target_stds=0.02, target_means=0) reproduces '
                         'in_scaler, and torch.manual_seed(42) + build_input_hypernet(49152, 2048) + '
                         'initialize_rescalers with the library-default zero prior bias reproduces scaler. The config '
                         'equals every original trained pool checkpoint config (SHA-256 22cb4f86...).'),
    }
    with open(out, 'wb') as stream:
        np.savez(stream, **arrays)
    return load_reference(out)


def load_reference(path=REFERENCE_NPZ) -> dict:
    """Load and self-check ``original_in_scaler.npz``; InitError if it is not the committed fixture."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
    except (OSError, ValueError) as exc:
        raise InitUsageError(f'Cannot read the reference rescaler {path}: {exc}. Remedy: restore it from git') from exc
    needed = ('w', 'b', 'scaler_w', 'scaler_b', 'repository', 'model_file', 'tensor_w', 'tensor_b',
              'tensor_scaler_w', 'tensor_scaler_b', 'step', 'config_file', 'config_json')
    missing = [key for key in needed if key not in values]
    if missing:
        raise InitUsageError(f'{path} lacks {missing}; restore it from git')
    problems = []
    for name, digest in (('w', REFERENCE_W_SHA256), ('b', REFERENCE_B_SHA256),
                         ('scaler_w', REFERENCE_SCALER_W_SHA256), ('scaler_b', REFERENCE_SCALER_B_SHA256)):
        array = values[name]
        if array.dtype != np.dtype('<f4') or array.shape != (1, EMBED_WIDTH):
            problems.append(f'{name} has dtype {array.dtype} shape {array.shape}')
        elif hashlib.sha256(array.tobytes()).hexdigest() != digest:
            problems.append(f'{name} values differ from the committed fixture')
    text = str(values['config_json'])
    if sha256_bytes(text.encode('utf-8')) != ORIGINAL_TRAINED_CONFIG_SHA256:
        problems.append('config_json is not the original trained-checkpoint config (SHA-256 22cb4f86...)')
    keys = ('repository', 'model_file', 'tensor_w', 'tensor_b', 'tensor_scaler_w', 'tensor_scaler_b', 'config_file')
    provenance = {key: str(values[key]) for key in keys}
    expected = {'repository': REFERENCE_REPOSITORY, 'model_file': REFERENCE_MODEL_FILE, 'tensor_w': 'in_scaler.w',
                'tensor_b': 'in_scaler.b', 'tensor_scaler_w': 'scaler.w', 'tensor_scaler_b': 'scaler.b',
                'config_file': REFERENCE_CONFIG_FILE}
    if provenance != expected or int(values['step']) != REFERENCE_STEP:
        problems.append(f'provenance {provenance} step {values["step"]} differs from {expected} step {REFERENCE_STEP}')
    if problems:
        raise InitUsageError(f'{path} is not the committed reference fixture: {"; ".join(problems)}. '
                             'Remedy: restore it from git')
    return {'w': values['w'], 'b': values['b'], 'scaler_w': values['scaler_w'], 'scaler_b': values['scaler_b'],
            'config_text': text, 'config': json.loads(text), 'step': REFERENCE_STEP,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **provenance}


# --- Construction ------------------------------------------------------------------------------------

def _kind(name: str) -> str:
    if name in RESCALER_NAMES:
        return 'rescaler'
    if 'LayerNorm.' in name or '.ln.' in name or 'output_norm.' in name:
        return 'norm'
    if name.endswith('.bias'):
        return 'bias'
    return 'normal'


def check_construction_init(hn) -> dict:
    """Assert the construction-time init the port was validated with (transformers 5.15.1, DESIGN.md 1.2).

    Linear/Embedding weights and ``inter_token_blocks`` ~ N(0, initializer_range); every bias
    exactly 0; LayerNorm weight 1 and bias 0; rescalers identity.  The std/mean bounds are
    8 standard errors (at least 0.001), so they cannot flake but catch PyTorch-default init
    (Embedding std 1, Linear std 1/sqrt(3 fan_in), non-zero biases).
    """
    std = float(hn.embed_init_range)
    problems, kinds = [], {}
    for name, tensor in hn.state_dict().items():
        kind = _kind(name)
        kinds[kind] = kinds.get(kind, 0) + 1
        values = tensor.detach().float()
        if kind == 'rescaler' or kind == 'norm':
            target = 1.0 if (name.endswith('.w') or name.endswith('.weight')) else 0.0
            if not bool((values == target).all()):
                problems.append(f'{name} is not exactly {target}')
        elif kind == 'bias':
            if not bool((values == 0).all()):
                problems.append(f'{name} is not exactly 0 (max |x| {float(values.abs().max()):.3g})')
        else:
            n = values.numel()
            sample_std = float(values.std()) if n > 1 else 0.0
            mean = float(values.mean())
            std_tol = max(0.001, 8 * std / math.sqrt(2 * n))
            mean_tol = max(0.001, 8 * std / math.sqrt(n))
            if abs(sample_std - std) > std_tol or abs(mean) > mean_tol:
                problems.append(f'{name} has mean {mean:.4g} std {sample_std:.4g}, expected N(0, {std}^2)')
    if problems:
        raise InitError('The freshly constructed hypernetwork does not have the initialization this port was '
                        f'validated with (transformers 5.15.1 _init_weights): {"; ".join(problems[:8])}'
                        f'{" ..." if len(problems) > 8 else ""}. Remedy: use the pinned train env '
                        '(python -m hyak.env.check_env)')
    return {'passed': True, 'initializer_range': std, 'tensors_by_kind': kinds,
            'rule': 'weights ~ N(0, initializer_range); biases 0; LayerNorm 1/0; rescalers identity'}


def build_hypernet(hf, table: torch.Tensor, *, seed: int = DEFAULT_SEED,
                   prior_bias_std: float = DEFAULT_PRIOR_BIAS_STD, rescaler_init: bool = True,
                   anchor_count: int = DEFAULT_ANCHORS, reference: dict | None = None,
                   calibration: str | None = CALIBRATE_BEFORE_BIAS) -> tuple:
    """Run the D1 recipe on ``table`` ([source vocab, width] float32). Returns ``(hn, info)``.

    ``calibration='before_bias'`` (default, option (b)): global seed -> ``build_input_hypernet(V, W)``
    -> construction check -> ``initialize_rescalers(hn, table)`` with the constructed (zero) prior
    bias -> calibration report of that zero-bias network (``info['calibration']``) -> with
    ``reference`` and seed 42: hard check of ``in_scaler`` and ``scaler`` against the public
    checkpoint (1e-5, else :class:`InitError`) and copy of the four fixture tensors -> prior bias
    from ``torch.Generator().manual_seed(seed + 1000)`` (skipped when the std is 0).

    ``calibration='after_bias'`` (option (a), ``--calibrate-after-bias``): the bias is drawn first and
    ``initialize_rescalers`` calibrates the result.  With ``reference`` and seed 42 the zero-bias
    calibration is also measured for the record (``info['original_construction']``, informative):
    it draws no random numbers and the final tensors are bit-identical to a build without it.

    ``rescaler_init=False`` leaves identity rescalers (``calibration`` must then be None or the default).
    """
    _check_recipe_args(seed, prior_bias_std, anchor_count)
    calibration = _check_calibration(calibration, prior_bias_std, rescaler_init)
    if table.dtype != torch.float32 or table.ndim != 2 or table.device.type != 'cpu':
        raise InitUsageError(f'table must be a CPU float32 matrix, got {table.dtype} {tuple(table.shape)} {table.device}')
    source_vocab, width = (int(n) for n in table.shape)
    torch.manual_seed(seed)                                       # D1: construction under the seed
    hn = hf.build_input_hypernet(source_vocab, width)             # identity_warmup.py:10-46, defaults
    construction = check_construction_init(hn)
    original = _original_construction_applicability(reference, seed, rescaler_init, width)
    info = {'construction_init': construction, 'calibration_order': calibration}
    bias = hn.model.target_priors_projection.bias

    def draw_bias():
        if prior_bias_std > 0:
            generator = torch.Generator(device='cpu').manual_seed(seed + PRIOR_BIAS_SEED_OFFSET)
            with torch.no_grad():
                torch.nn.init.normal_(bias, mean=0.0, std=prior_bias_std, generator=generator)

    if calibration == CALIBRATE_BEFORE_BIAS:
        hf.initialize_rescalers(hn, table, anchor_count)          # the original construction: zero prior bias
        info['calibration'] = calibration_report(anchor_outputs(hn, table, anchor_count), table,
                                                 scope='zero-bias network, before the prior-bias draw')
        if original.get('applicable'):
            computed = {name: getattr(getattr(hn, name.split('.')[0]), name.split('.')[1]).detach().clone()
                        for name in RESCALER_NAMES}
            info['in_scaler_regression'] = in_scaler_regression(hn, reference, tolerance=ORIGINAL_CONSTRUCTION_TOLERANCE)
            original.update(_compare_scaler((computed['scaler.w'], computed['scaler.b']),
                                            (computed['scaler.w'], computed['scaler.b']), reference))
            in_diff = max(info['in_scaler_regression']['max_abs_diff_w'], info['in_scaler_regression']['max_abs_diff_b'])
            original['matches'] = bool(original['matches'] and in_diff <= ORIGINAL_CONSTRUCTION_TOLERANCE)
            if not original['matches']:
                raise InitError(
                    'The calibrated rescalers do not reproduce the original construction within '
                    f'{ORIGINAL_CONSTRUCTION_TOLERANCE} (in_scaler {in_diff:.3g}, scaler w '
                    f'{original["zero_bias_max_abs_diff_w"]:.3g}, b {original["zero_bias_max_abs_diff_b"]:.3g}), so the '
                    'fixture values cannot be copied safely. Remedy: check the input table (python -m '
                    'hyak.model.fetch_model) and the train env (python -m hyak.env.check_env); to build anyway '
                    'without the original rescalers use --calibrate-after-bias, and say so in the report')
            _copy_fixture_rescalers(hn, reference)
            original.update(enforced=True, passed=True, copied_from_fixture=True, fixture_sha256=reference['sha256'],
                            final_scaler_max_abs_diff_w=0.0, final_scaler_max_abs_diff_b=0.0,
                            final_scaler_w_mean=float(np.float64(reference['scaler_w'].astype(np.float64).mean())),
                            final_scaler_b_std=float(reference['scaler_b'].astype(np.float64).std(ddof=1)),
                            meaning='the four frozen rescaler tensors were checked against the public checkpoint '
                                    'and copied from it: they equal the original runs\' rescalers bit for bit')
        draw_bias()
    else:
        zero_bias_scaler = None
        if original.get('applicable') and prior_bias_std > 0:
            hf.initialize_rescalers(hn, table, anchor_count)      # what the original computed: zero prior bias
            zero_bias_scaler = (hn.scaler.w.detach().clone(), hn.scaler.b.detach().clone())
            with torch.no_grad():                                 # back to the constructed identity
                hn.scaler.w.fill_(1.0)                            # (modeling_hypernet.py:18-19)
                hn.scaler.b.fill_(0.0)
        draw_bias()
        if rescaler_init:
            hf.initialize_rescalers(hn, table, anchor_count)      # identity_warmup.py:49-57
        if original.get('applicable'):
            if zero_bias_scaler is None:                          # std 0: the recipe's own calibration
                zero_bias_scaler = (hn.scaler.w.detach().clone(), hn.scaler.b.detach().clone())
            original.update(_compare_scaler(zero_bias_scaler, (hn.scaler.w.detach(), hn.scaler.b.detach()),
                                            reference))
    info.update({'original_construction': original,
                 'prior_bias': {'std_requested': prior_bias_std,
                                'generator_seed': seed + PRIOR_BIAS_SEED_OFFSET if prior_bias_std > 0 else None,
                                'method': ('torch.nn.init.normal_(bias, 0.0, std, generator=Generator().manual_seed('
                                           'seed + 1000))') if prior_bias_std > 0 else 'library default (zeros)',
                                'sample_std': float(bias.detach().std()), 'max_abs': float(bias.detach().abs().max())},
                 'rescaler_init': rescaler_init, 'anchor_count': anchor_count if rescaler_init else None})
    return hn, info


def _check_calibration(calibration, prior_bias_std, rescaler_init):
    """Validated calibration order (None when the rescalers are not calibrated)."""
    if calibration not in (CALIBRATE_BEFORE_BIAS, CALIBRATE_AFTER_BIAS, None):
        raise InitUsageError(f'calibration must be {CALIBRATE_BEFORE_BIAS!r} or {CALIBRATE_AFTER_BIAS!r}, '
                             f'got {calibration!r}')
    if not rescaler_init:
        if calibration == CALIBRATE_AFTER_BIAS:
            raise InitUsageError('--calibrate-after-bias has no meaning with --no-rescaler-init')
        return None
    if calibration is None:
        raise InitUsageError('calibration must be given when the rescalers are calibrated')
    if calibration == CALIBRATE_AFTER_BIAS and prior_bias_std == 0:
        raise InitUsageError('--calibrate-after-bias has no effect with --prior-bias-std 0 (there is no bias to '
                             'draw); drop one of the two')
    return calibration


def _copy_fixture_rescalers(hn, reference) -> None:
    """Set in_scaler.{w,b} and scaler.{w,b} to the public checkpoint's float32 values (bit for bit)."""
    with torch.no_grad():
        for name, key in (('in_scaler.w', 'w'), ('in_scaler.b', 'b'), ('scaler.w', 'scaler_w'),
                          ('scaler.b', 'scaler_b')):
            module, attribute = name.split('.')
            target = getattr(getattr(hn, module), attribute)
            value = torch.from_numpy(np.ascontiguousarray(reference[key], dtype='<f4'))
            if tuple(value.shape) != tuple(target.shape):
                raise InitError(f'{name} has shape {tuple(target.shape)}, the fixture {tuple(value.shape)}')
            target.copy_(value)


def _original_construction_applicability(reference, seed, rescaler_init, width) -> dict:
    if reference is None:
        return {'applicable': False, 'reason': 'no reference checkpoint for this profile'}
    if not rescaler_init:
        return {'applicable': False, 'reason': 'rescalers not calibrated (--no-rescaler-init)'}
    if seed != ORIGINAL_CONSTRUCTION_SEED:
        return {'applicable': False, 'reason': f'seed {seed} (the original warmup started from seed 42)'}
    if reference['scaler_w'].shape != (1, width):
        return {'applicable': False, 'reason': f'width {width} differs from the reference'}
    return {'applicable': True}


def _compare_scaler(zero_bias: tuple, final: tuple, reference: dict) -> dict:
    """Zero-bias calibration and the recipe's final scaler versus the original checkpoint's frozen scaler."""
    ref_w, ref_b = reference['scaler_w'].astype(np.float64), reference['scaler_b'].astype(np.float64)
    zero_w, zero_b = (t.double().numpy() for t in zero_bias)
    final_w, final_b = (t.double().numpy() for t in final)
    report = {
        'reference': f'{reference["repository"]}/{reference["model_file"]} scaler.w, scaler.b',
        'zero_bias_max_abs_diff_w': float(np.abs(zero_w - ref_w).max()),
        'zero_bias_max_abs_diff_b': float(np.abs(zero_b - ref_b).max()),
        'tolerance': ORIGINAL_CONSTRUCTION_TOLERANCE,
        'final_scaler_max_abs_diff_w': float(np.abs(final_w - ref_w).max()),
        'final_scaler_max_abs_diff_b': float(np.abs(final_b - ref_b).max()),
        'reference_scaler_w_mean': float(ref_w.mean()), 'final_scaler_w_mean': float(final_w.mean()),
        'reference_scaler_b_std': float(ref_b.std(ddof=1)), 'final_scaler_b_std': float(final_b.std(ddof=1)),
    }
    report['matches'] = max(report['zero_bias_max_abs_diff_w'], report['zero_bias_max_abs_diff_b']) \
        <= ORIGINAL_CONSTRUCTION_TOLERANCE
    report['meaning'] = ('matches=true: this construction (seed 42, zero prior bias, calibrated) reproduces the '
                         'starting point of the original warmup; the frozen scaler the run keeps is the final one, '
                         'which differs from the original when the prior bias is drawn before calibration (D1)')
    return report


def _check_recipe_args(seed, prior_bias_std, anchor_count) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 63 - 1 - 4 * DIAGNOSTIC_SEED_OFFSET:
        raise InitUsageError(f'seed must be an integer in 0..2^63-8001, got {seed!r}')
    if isinstance(prior_bias_std, bool) or not isinstance(prior_bias_std, (int, float)) \
            or not math.isfinite(prior_bias_std) or prior_bias_std < 0:
        raise InitUsageError(f'prior_bias_std must be a finite number >= 0, got {prior_bias_std!r}')
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count < 2:
        raise InitUsageError(f'anchor_count must be an integer >= 2, got {anchor_count!r}')


@torch.no_grad()
def anchor_outputs(hn, table: torch.Tensor, anchor_count: int = DEFAULT_ANCHORS) -> torch.Tensor:
    """The deterministic forward ``init_rescaler`` calibrates on: anchors as one-piece forms, zero priors.

    Mirrors ``identity_warmup.py:53-57`` and ``modeling_hypernet.py:237``.
    """
    count = min(anchor_count, table.shape[0])
    forms = torch.arange(count)[:, None]
    priors = torch.zeros(count)
    predicted, output_embeddings, _ = hn(forms, priors, table, None)
    if output_embeddings is not None:
        raise InitError('The hypernetwork returned output embeddings; expected an input-only hypernetwork')
    return predicted.detach().clone()


#: What the calibration check covers (it says nothing about realistic vocabularies).
CALIBRATION_SCOPE = ('the first anchor rows (token ids 0..anchors-1) as one-piece forms with zero priors, '
                     'evaluated as one batch')


def calibration_report(outputs: torch.Tensor, table: torch.Tensor, *, scope: str | None = None) -> dict:
    """How well the calibrated outputs of the anchors match the canonical anchors per dimension.

    ``Rescaler.scale_to`` (modeling_hypernet.py:24-32) makes each output dimension's mean equal the
    target mean and its std equal the target std (up to the 1e-8 in the denominator).
    """
    target = table[:outputs.shape[0]]
    mean_error = float((outputs.mean(0) - target.mean(0)).abs().max())
    ratio = outputs.std(0, correction=0) / target.std(0, correction=0)
    report = {'anchors': int(outputs.shape[0]), 'scope': CALIBRATION_SCOPE + (f'; {scope}' if scope else ''),
              'max_abs_mean_error': mean_error,
              'max_abs_std_ratio_error': float((ratio - 1).abs().max()),
              'output_row_norm_mean': float(outputs.norm(dim=1).mean()),
              'target_row_norm_mean': float(target.norm(dim=1).mean())}
    report['passed'] = report['max_abs_mean_error'] <= 1e-5 and report['max_abs_std_ratio_error'] <= 1e-3
    return report


def in_scaler_regression(hn, reference: dict, *, tolerance: float = IN_SCALER_TOLERANCE) -> dict:
    """Max |difference| of ``in_scaler.w``/``.b`` to the original checkpoint's values (float64 arithmetic)."""
    report = {'reference': {k: reference[k] for k in ('repository', 'model_file', 'tensor_w', 'tensor_b', 'step')},
              'tolerance': tolerance}
    for key, ref in (('w', reference['w']), ('b', reference['b'])):
        ours = getattr(hn.in_scaler, key).detach().double().numpy()
        if ours.shape != ref.shape:
            raise InitError(f'in_scaler.{key} has shape {ours.shape}, reference {ref.shape}')
        report[f'max_abs_diff_{key}'] = float(np.abs(ours - ref.astype(np.float64)).max())
    report['passed'] = max(report['max_abs_diff_w'], report['max_abs_diff_b']) <= tolerance
    return report


# --- Tensor reports -------------------------------------------------------------------------------

def _moments(tensor: torch.Tensor, chunk: int = 1 << 24) -> tuple:
    flat = tensor.detach().reshape(-1)
    n = flat.numel()
    total = squares = 0.0
    absmax = 0.0
    finite = True
    for start in range(0, n, chunk):
        part = flat[start:start + chunk].double()
        finite = finite and bool(torch.isfinite(part).all())
        total += float(part.sum())
        squares += float((part * part).sum())
        absmax = max(absmax, float(part.abs().max())) if part.numel() else absmax
    mean = total / n if n else 0.0
    var = (squares - n * mean * mean) / (n - 1) if n > 1 else 0.0
    return n, mean, math.sqrt(max(var, 0.0)), absmax, finite


def tensor_report(state_dict: dict) -> dict:
    """Per tensor: shape, dtype, numel, mean, std, absmax, finite, sha256 (of the raw bytes)."""
    rows = {}
    for name in sorted(state_dict):
        tensor = state_dict[name]
        n, mean, std, absmax, finite = _moments(tensor)
        rows[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype).replace('torch.', ''),
                      'numel': n, 'mean': mean, 'std': std, 'absmax': absmax, 'finite': finite,
                      'sha256': _tensor_sha256(tensor)}
    return rows


def parameter_counts(hn) -> dict:
    total = sum(p.numel() for p in hn.parameters())
    frozen = sum(p.numel() for p in hn.parameters() if not p.requires_grad)
    return {'total': total, 'trainable': total - frozen, 'frozen': frozen,
            'frozen_names': sorted(n for n, p in hn.named_parameters() if not p.requires_grad)}


# --- Configuration checks ----------------------------------------------------------------------------

def expected_init_config(source_vocab: int, width: int) -> dict:
    """Config values ``build_input_hypernet(source_vocab, width)`` must produce (identity_warmup.py:15-42).

    Includes every key the code reads (NOTES/hypernetwork.md 3.3) and the trainer's conditions
    (train_joint_smollm_input.py:440-446, tokenizer_inputs.py:24-38).
    """
    heads = max(1, min(width // 64, 32))
    return {
        'hn_model_type': 'roberta', 'hn_n_layers': 3, 'hn_hidden_size': width,
        'hn_intermediate_size': 2 * width, 'hn_num_attention_heads': heads, 'n_embd': width,
        'hn_surface_maxlen': MAX_PIECES, 'hn_embed_using_source_embeddings': True,
        'hn_embed_target_priors': True, 'hn_add_inter_token_attention': True,
        'hn_inter_token_attention_bias_by_priors': False, 'hn_inter_token_attention_bias_scaler': 1.0,
        'hn_n_inter_token_blocks': 16, 'hn_embed_lang_id': False, 'hn_language_adapter_bottleneck_dim': 0,
        'hn_rescale_embeddings': True, 'hn_predict_bias': False, 'hn_single_head': False,
        'hn_concat_last_hidden_state': False, 'hn_flax_compatible_roberta': False, 'n_langs': None,
        'original_vocab_size': source_vocab, 'pad_token_id': source_vocab, 'hn_n_extra_tokens': 1,
        'hn_byte_fallback_symbols': None, 'separate_out_embeddings': False,
        'hn_backbone_config': {
            'model_type': 'roberta', 'hidden_size': width, 'intermediate_size': 2 * width,
            'num_hidden_layers': 3, 'num_attention_heads': heads, 'vocab_size': source_vocab + 1,
            'max_position_embeddings': MAX_PIECES + 2, 'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1, 'layer_norm_eps': 1e-12, 'initializer_range': 0.02,
            'hidden_act': 'gelu', 'type_vocab_size': 2,
        },
    }


def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def config_differences(actual: dict, expected: dict, *, exact: bool = False, prefix: str = '') -> list:
    """Differences ``key: actual != expected`` (types compared strictly; ``exact`` also reports extra keys)."""
    out = []
    for key, want in expected.items():
        name = f'{prefix}{key}'
        if key not in actual:
            out.append(f'{name}: missing (expected {want!r})')
        elif isinstance(want, dict) and isinstance(actual[key], dict):
            out.extend(config_differences(actual[key], want, exact=exact, prefix=f'{name}.'))
        elif not _same(actual[key], want):
            out.append(f'{name}: {actual[key]!r} != expected {want!r}')
    if exact:
        out.extend(f'{prefix}{key}: unexpected key (value {actual[key]!r})' for key in actual if key not in expected)
    return out


def reference_init_config(reference: dict) -> dict:
    """The original trained config with the trainer's byte-fallback change undone (tokenizer_inputs.py:33-34)."""
    config = json.loads(reference['config_text'])
    config['hn_byte_fallback_symbols'] = None
    config['hn_n_extra_tokens'] = 1
    return config


def check_init_config(config: dict, profile: Profile, reference: dict | None) -> dict:
    """Trainer-required fields; with the production reference, every field of the original config."""
    problems = config_differences(config, expected_init_config(profile.source_vocab, profile.width))
    report = {'required_fields_ok': not problems}
    if reference is not None:
        diff = config_differences(config, reference_init_config(reference), exact=True)
        report['reference_differences'] = diff
        problems += [f'versus the original config: {d}' for d in diff]
    if problems:
        raise InitError(f'config.json is not what the trainer and the original runs expect: {"; ".join(problems)}')
    report['passed'] = True
    return report


# --- Files ------------------------------------------------------------------------------------------------

def predicted_safetensors_bytes(state_dict: dict) -> int:
    """Byte size of a safetensors file of float32 tensors: 8 + padded header + data (as in the notes)."""
    header = {'__metadata__': {'format': 'pt'}}
    offset = 0
    for name in sorted(state_dict):
        size = state_dict[name].numel() * state_dict[name].element_size()
        header[name] = {'dtype': 'F32', 'shape': list(state_dict[name].shape), 'data_offsets': [offset, offset + size]}
        offset += size
    text = json.dumps(header, separators=(',', ':')).encode()
    return 8 + len(text) + (-len(text) % 8) + offset


def check_saved_files(directory: Path, tensors: dict, profile: Profile, *, chunk: int = 8 << 20) -> dict:
    """The saved directory: exactly config.json + model.safetensors; header, size, and every tensor's bytes.

    ``tensors`` is :func:`tensor_report` of the in-memory model.  The file is parsed directly
    (8-byte little-endian header length, JSON header, contiguous data) and each tensor's bytes
    are streamed from disk and hashed, so the check sees exactly what is stored.
    """
    if sys.byteorder != 'little':
        raise InitError('This check assumes a little-endian host (safetensors stores little-endian data)')
    names = sorted(p.name for p in directory.iterdir())
    if names != sorted(OUTPUT_FILES):
        raise InitError(f'save_pretrained wrote {names}, expected {sorted(OUTPUT_FILES)}')
    model = directory / 'model.safetensors'
    size = model.stat().st_size
    problems = []
    with open(model, 'rb') as stream:
        header_len = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_len < min(size, 100 << 20):
            raise InitError(f'{model} has an invalid safetensors header length {header_len}')
        header = json.loads(stream.read(header_len))
        entries = {k: v for k, v in header.items() if k != '__metadata__'}
        if header.get('__metadata__') != {'format': 'pt'}:
            problems.append(f'metadata {header.get("__metadata__")} != {{"format": "pt"}}')
        if sorted(entries) != sorted(tensors):
            problems.append(f'tensor names differ: missing {sorted(set(tensors) - set(entries))}, '
                            f'extra {sorted(set(entries) - set(tensors))}')
        offset = 0
        for name, entry in sorted(entries.items(), key=lambda item: item[1]['data_offsets'][0]):
            start, end = entry['data_offsets']
            if start != offset:
                problems.append(f'{name} starts at {start}, expected {offset} (gap or overlap)')
            offset = end
            want = tensors.get(name)
            if want is None:
                continue
            if entry['dtype'] != 'F32' or entry['shape'] != want['shape']:
                problems.append(f'{name} is stored as {entry["dtype"]} {entry["shape"]}, expected F32 {want["shape"]}')
                continue
            stream.seek(8 + header_len + start)
            digest, remaining = hashlib.sha256(), end - start
            while remaining:
                block = stream.read(min(chunk, remaining))
                if not block:
                    break
                digest.update(block)
                remaining -= len(block)
            if remaining or digest.hexdigest() != want['sha256']:
                problems.append(f'{name}: bytes in the file differ from the in-memory tensor')
    data_bytes = offset
    if 8 + header_len + data_bytes != size:
        problems.append(f'file size {size} != 8 + header {header_len} + data {data_bytes}')
    if profile.expected_model_bytes is not None and size != profile.expected_model_bytes:
        problems.append(f'model.safetensors is {size} bytes, expected {profile.expected_model_bytes}')
    if profile.expected_tensors is not None and len(entries) != profile.expected_tensors:
        problems.append(f'{len(entries)} tensors, expected {profile.expected_tensors}')
    if problems:
        raise InitError(f'The saved hypernetwork in {directory} is wrong: {"; ".join(problems[:10])}')
    return {'passed': True, 'tensors': len(entries), 'model_bytes': size, 'header_bytes': header_len,
            'data_bytes': data_bytes}


# --- Reload, diagnostic, trainer emulation -----------------------------------------------------------

def reload_hypernet(hf, directory: Path):
    """``ZettHypernet.from_pretrained(directory, local_files_only=True)`` as the trainer does (T:436-439)."""
    hn, loading = hf.ZettHypernet.from_pretrained(str(directory), local_files_only=True, output_loading_info=True)
    bad = {k: v for k, v in loading.items() if v and k in ('missing_keys', 'unexpected_keys', 'mismatched_keys',
                                                             'error_msgs')}
    if bad:
        raise InitError(f'Reloading {directory} reported {bad}; the trainer would start from re-initialized tensors')
    return hn


def input_lexical_loss(generated, surface_forms, source, pad_id: int, eos_id: int) -> torch.Tensor:
    """Verbatim copy of ``runtime/anytok_pool_20k/hf_hypernet/joint_training.py:199-209``.

    Copied because ``joint_training`` imports the Rust sampler module at load time; the unit
    tests check that this copy equals the pinned function.
    """
    first_ids = surface_forms[:, 0]
    overlap = ((surface_forms[:, 1:] == pad_id).all(dim=1)
               & (first_ids < source.shape[0]) & (first_ids != eos_id))
    if not overlap.any():
        raise ValueError("No original tokens overlap the sampled vocabulary")
    target = source[first_ids[overlap]].float()
    predicted = generated[overlap].float()
    return (predicted - target).square().sum(-1).mean() / target.norm(dim=1).mean()


def diagnostic_batch(source_vocab: int, width: int, *, vocab: int, one_piece: int, seed: int) -> tuple:
    """Synthetic zero-prior vocabulary: ``one_piece`` single-piece rows, the rest 2-10 pieces, pad = source_vocab."""
    if not 0 < one_piece <= vocab:
        raise InitError(f'one_piece must be in 1..{vocab}, got {one_piece}')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randint(1, source_vocab, (vocab, MAX_PIECES), generator=generator)
    lengths = torch.cat([torch.ones(one_piece, dtype=torch.long),
                         torch.randint(2, MAX_PIECES + 1, (vocab - one_piece,), generator=generator)])
    forms = torch.where(torch.arange(MAX_PIECES)[None] < lengths[:, None], ids, torch.full_like(ids, source_vocab))
    priors = torch.zeros(vocab)
    upstream = torch.randn(vocab, width, generator=generator)
    return forms, priors, upstream


def prior_gradient_diagnostic(hn, table: torch.Tensor, *, seed: int, vocab: int = 256, one_piece: int = 64) -> dict:
    """Gradient norm of the prior bias versus all other parameters for a zero-prior vocabulary.

    One training-shaped backward on CPU in float32: dropout on (``deterministic=False``),
    activation checkpointing as in the trainer (T:442-443), loss
    ``(generated.float() * g).sum() + 0.5 * input_lexical_loss(...)`` like T:577 with a random
    upstream gradient ``g ~ N(0, 1)`` in place of the LM's.  ``ratio`` > 1 means the 2,048-element
    bias alone dominates the trainer's joint gradient clip (the degenerate case of DESIGN.md 1.2).
    The global RNG is restored afterwards.
    """
    source_vocab, width = (int(n) for n in table.shape)
    forms, priors, upstream = diagnostic_batch(source_vocab, width, vocab=vocab, one_piece=one_piece, seed=seed)
    pad_id = int(hn.config.pad_token_id)
    hn.model.activation_checkpointing = True
    hn.activation_checkpoint_input = True
    hn.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.enable_grad():
            generated, output_embeddings, _ = hn(forms, priors, table, None, deterministic=False)
            if output_embeddings is not None or not bool(torch.isfinite(generated).all()):
                raise InitError('The hypernetwork produced invalid input embeddings in the gradient diagnostic')
            lexical = input_lexical_loss(generated, forms, table, pad_id, SPECIAL_ID)
            loss = (generated.float() * upstream).sum() + 0.5 * lexical
            loss.backward()
    bias_sq = other_sq = 0.0
    finite = True
    for name, parameter in hn.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.double()
        finite = finite and bool(torch.isfinite(grad).all())
        value = float((grad * grad).sum())
        if name == PRIOR_BIAS_NAME:
            bias_sq += value
        else:
            other_sq += value
    hn.zero_grad(set_to_none=True)
    bias_norm, other_norm = math.sqrt(bias_sq), math.sqrt(other_sq)
    ratio = bias_norm / other_norm if other_norm > 0 else math.inf
    if not finite or not math.isfinite(ratio):
        raise InitError('Non-finite gradients in the zero-prior gradient diagnostic')
    return {'vocab': vocab, 'one_piece_rows': one_piece, 'priors': 'all zero', 'seed': seed,
            'dtype': 'float32 (CPU; the trainer uses bf16 autocast on CUDA)', 'dropout': True,
            'upstream_gradient': 'N(0, 1) per element', 'lexical_loss': float(lexical.detach()),
            'loss': float(loss.detach()), 'prior_bias_grad_norm': bias_norm, 'other_grad_norm': other_norm,
            'ratio': ratio, 'clip_share': bias_norm / math.sqrt(bias_sq + other_sq),
            'degenerate': ratio > DEGENERATE_RATIO, 'degenerate_threshold': DEGENERATE_RATIO}


def load_source_tokenizer(model_dir):
    """``AutoTokenizer.from_pretrained(model_dir, local_files_only=True)`` and the EOS check (T:421-423)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN) != SPECIAL_ID:
        raise InitError("This runner expects SmolLM2's end-of-text token at ID zero (T:422-423)")
    return tokenizer


def trainer_first_start(hf, hn, tokenizer, check_dir: Path, *, profile: Profile, reference: dict | None,
                        init_fallback_sha256: str) -> dict:
    """Emulate the trainer's fresh start on the reloaded directory and its first checkpoint save.

    ``enable_byte_fallbacks`` (T:440), ``freeze_rescalers`` (T:441), the checkpoint flags
    (T:442-443), the configuration checks (T:444-446), then ``hn.save_pretrained(stage /
    "hypernet", safe_serialization=True)`` (T:218).  Modifies ``hn`` (adds the fallback rows).
    """
    source_size = len(tokenizer)
    symbols = hf.missing_byte_symbols(tokenizer)
    fallback = hf.enable_byte_fallbacks(hn, tokenizer)                                  # T:440
    hf.freeze_rescalers(hn)                                                             # T:441
    hn.model.activation_checkpointing = True                                            # T:442
    hn.activation_checkpoint_input = True                                               # T:443
    if not hn.config.hn_add_inter_token_attention or not hn.config.hn_embed_target_priors:  # T:444-445
        raise InitError('Expected vocabulary attention and target priors (T:444-445)')
    if hn.config.hn_surface_maxlen != MAX_PIECES:                                       # T:446
        raise InitError(f'hn_surface_maxlen is {hn.config.hn_surface_maxlen}, the trainer asserts 10 (T:446)')
    problems = []
    rows = hn.fallback_embeddings.num_embeddings
    if rows != 1 + len(symbols) or hn.config.hn_n_extra_tokens != rows:
        problems.append(f'{rows} fallback rows, hn_n_extra_tokens {hn.config.hn_n_extra_tokens}, symbols {len(symbols)}')
    if profile.expected_fallback_symbols is not None and len(symbols) != profile.expected_fallback_symbols:
        problems.append(f'{len(symbols)} byte-fallback symbols, expected {profile.expected_fallback_symbols}')
    if sorted(fallback.values()) != list(range(source_size + 1, source_size + 1 + len(symbols))):
        problems.append('fallback ids are not source_size + 1 ... source_size + n')
    if _tensor_sha256(hn.fallback_embeddings.weight[:1]) != init_fallback_sha256:
        problems.append('the existing fallback (pad) row changed during the expansion')
    frozen = [n for n, p in hn.named_parameters() if not p.requires_grad]
    if sorted(frozen) != sorted(RESCALER_NAMES):
        problems.append(f'frozen parameters after freeze_rescalers are {frozen}, expected the four rescaler tensors')
    if problems:
        raise InitError(f'Trainer emulation failed: {"; ".join(problems)}')
    stage = check_dir / 'hypernet'
    hn.save_pretrained(str(stage), safe_serialization=True)                             # T:218
    config_bytes = (stage / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    model_bytes = (stage / 'model.safetensors').stat().st_size
    report = {'fallback_symbols': symbols, 'fallback_ids': [min(fallback.values()), max(fallback.values())],
              'fallback_rows': rows, 'hn_n_extra_tokens': hn.config.hn_n_extra_tokens,
              'trainable_parameters': sum(p.numel() for p in hn.parameters() if p.requires_grad),
              'checkpoint_config': {'bytes': len(config_bytes), 'sha256': sha256_bytes(config_bytes)},
              'checkpoint_model_bytes': model_bytes}
    if profile.expected_trained_model_bytes is not None and model_bytes != profile.expected_trained_model_bytes:
        problems.append(f'checkpoint model.safetensors is {model_bytes} bytes, the original trained checkpoints '
                        f'have {profile.expected_trained_model_bytes}')
    if reference is not None:
        diff = config_differences(config, reference['config'], exact=True)
        report['reference_differences'] = diff
        problems += [f'checkpoint config versus the original: {d}' for d in diff]
        report['config_identical_to_original'] = report['checkpoint_config']['sha256'] == ORIGINAL_TRAINED_CONFIG_SHA256
        if not diff and not report['config_identical_to_original']:
            problems.append(f'checkpoint config.json has the same fields as the original but SHA-256 '
                            f'{report["checkpoint_config"]["sha256"]} != {ORIGINAL_TRAINED_CONFIG_SHA256} '
                            '(serialization differs: check the transformers version)')
    if problems:
        raise InitError(f'Trainer emulation failed: {"; ".join(problems)}')
    report['passed'] = True
    return report


def trainer_resume(hf, check_dir: Path, tokenizer, expected_rows: int) -> dict:
    """Resume path: reload the emulated checkpoint (T:436-439 with ``last``) and enable fallbacks again.

    ``enable_byte_fallbacks`` must accept the recorded symbols without a second expansion
    (tokenizer_inputs.py:35-38).
    """
    hn = reload_hypernet(hf, check_dir / 'hypernet')
    before = _tensor_sha256(hn.fallback_embeddings.weight)
    hf.enable_byte_fallbacks(hn, tokenizer)
    rows = hn.fallback_embeddings.num_embeddings
    unchanged = _tensor_sha256(hn.fallback_embeddings.weight) == before
    del hn
    if rows != expected_rows or not unchanged:
        raise InitError(f'Resume emulation changed the fallback rows ({rows} rows, unchanged={unchanged})')
    return {'passed': True, 'fallback_rows': rows}


# --- Directory management -------------------------------------------------------------------------------

def _host() -> str:
    from hyak.common import slurm
    return slurm.node_name()


def _temp_name(out: Path, kind: str) -> Path:
    return out.parent / f'.{out.name}.{kind}--{_host()}--{os.getpid()}'


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def cleanup_stale(out: Path, *, now: float | None = None) -> list:
    """Remove temporary build directories of dead processes. Returns the removed paths.

    On this host a directory is removed when its pid is dead; a directory of another host (a job
    killed hard on another node) when it is older than ``STALE_SECONDS`` (a build takes minutes).
    """
    now = time.time() if now is None else now
    removed = []
    if not out.parent.is_dir():
        return removed
    for path in sorted(out.parent.glob(f'.{out.name}.*--*--*')):
        try:
            _, host, pid = path.name.rsplit('--', 2)
            pid = int(pid)
        except ValueError:
            continue
        kind = path.name[len(out.name) + 2:].split('--', 1)[0]
        if kind == 'replaced':
            raise InitUsageError(f'{path} is a previous initialization moved aside by an interrupted --force; '
                            f'inspect it, then move it back to {out} or delete it')
        if kind not in ('building', 'trainer-check'):
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if (host == _host() and not _pid_alive(pid)) or (host != _host() and age > STALE_SECONDS):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
        else:
            _log(f'WARNING: leaving {path} (owned by {host} pid {pid}, {age / 3600:.1f} h old, possibly still '
                 f'running; it is removed automatically after {STALE_SECONDS // 3600} h). When no init job is '
                 f'running, delete it by hand: rm -rf {path}')
    return removed


def _non_empty(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return True


def _rename(source: Path, destination: Path) -> None:
    """``os.rename`` (atomic within one filesystem; fails if a non-empty directory is in the way)."""
    os.rename(source, destination)


def _publish(build_dir: Path, out: Path, force: bool) -> None:
    """Move the finished build into place; with ``force`` the old directory is moved aside, then removed."""
    if os.path.lexists(out):
        if not force:
            raise InitUsageError(f'{out} appeared while building; refusing to overwrite it (use --force)')
        if os.path.islink(out) or not out.is_dir():
            raise InitUsageError(f'{out} exists and is not a plain directory; remove it by hand')
        replaced = out.parent / f'.{out.name}.replaced--{_host()}--{os.getpid()}'
        _rename(out, replaced)
        try:
            _rename(build_dir, out)
        except BaseException:
            _rename(replaced, out)
            raise
        shutil.rmtree(replaced)
    else:
        _rename(build_dir, out)
    fsync_dir(out.parent)


def verify_directory(out, *, profile: Profile = PRODUCTION, reference: dict | None = ...) -> dict:
    """Check an existing init directory against its manifest. Returns ``{'ok', 'problems', 'manifest'}``.

    Checks the file set, the recorded bytes and SHA-256 of both files, the manifest's own
    verification results, and the config fields (with the production reference).
    """
    out = Path(out)
    problems = []
    manifest = None
    if not out.is_dir():
        return {'ok': False, 'problems': [f'{out} is not a directory'], 'manifest': None}
    names = sorted(p.name for p in out.iterdir())
    if names != sorted(OUTPUT_FILES + (MANIFEST_NAME,)):
        problems.append(f'{out} holds {names}, expected {sorted(OUTPUT_FILES + (MANIFEST_NAME,))}')
    try:
        manifest = read_json(out / MANIFEST_NAME)
    except (OSError, ValueError) as exc:
        return {'ok': False, 'problems': problems + [f'cannot read {MANIFEST_NAME}: {exc}'], 'manifest': None}
    if manifest.get('format') != MANIFEST_FORMAT or manifest.get('complete') is not True:
        problems.append(f'{MANIFEST_NAME} is not a complete {MANIFEST_FORMAT} manifest')
    for name in OUTPUT_FILES:
        recorded = (manifest.get('outputs') or {}).get(name)
        path = out / name
        if not path.is_file() or recorded is None:
            problems.append(f'{name} missing or not recorded')
            continue
        if path.stat().st_size != recorded.get('bytes'):
            problems.append(f'{name} is {path.stat().st_size} bytes, manifest says {recorded.get("bytes")}')
        elif file_record(path)['sha256'] != recorded.get('sha256'):
            problems.append(f'{name} SHA-256 differs from the manifest (modified after it was built)')
    if profile.expected_model_bytes is not None and (out / 'model.safetensors').is_file() \
            and (out / 'model.safetensors').stat().st_size != profile.expected_model_bytes:
        problems.append(f'model.safetensors is not {profile.expected_model_bytes} bytes')
    failed = [name for name, check in (manifest.get('checks') or {}).items()
              if isinstance(check, dict) and check.get('passed') is False]
    if failed:
        problems.append(f'checks recorded as failed: {failed}')
    if (out / 'config.json').is_file():
        if reference is ...:
            reference = load_reference() if profile.reference else None
        try:
            check_init_config(read_json(out / 'config.json'), profile, reference)
        except (InitError, ValueError) as exc:
            problems.append(str(exc))
    return {'ok': not problems, 'problems': problems, 'manifest': manifest}


# --- The whole build ---------------------------------------------------------------------------------

def recipe_option(recipe: dict) -> str:
    """Short name of the starting point: (a), (b) or (c) of the run guide, or 'identity rescalers'."""
    if not recipe.get('rescaler_init'):
        return 'identity rescalers (--no-rescaler-init)'
    if recipe.get('prior_bias_std') == 0:
        return '(c) zero prior bias (--prior-bias-std 0)'
    if recipe.get('calibration') == CALIBRATE_AFTER_BIAS:
        return '(a) bias drawn, then calibrated (--calibrate-after-bias)'
    return '(b) calibrated with the zero bias, then bias drawn (default)'


def _recipe(seed, prior_bias_std, rescaler_init, anchor_count, profile, calibration=CALIBRATE_BEFORE_BIAS,
            reference=None) -> dict:
    calibration = _check_calibration(calibration, prior_bias_std, rescaler_init)
    copied = (calibration == CALIBRATE_BEFORE_BIAS
              and _original_construction_applicability(reference, seed, rescaler_init, profile.width).get('applicable'))
    bias = 'prior bias ~ N(0, std^2) from torch.Generator(seed + 1000) (std 0: library default)'
    if calibration == CALIBRATE_BEFORE_BIAS:
        order = ['initialize_rescalers(hn, float32 table, anchor_count) with the zero prior bias']
        if copied:
            order.append('check in_scaler and scaler against original_in_scaler.npz (1e-5), then copy the four '
                         'rescaler tensors from it')
        order.append(bias)
    elif calibration == CALIBRATE_AFTER_BIAS:
        order = [bias, 'initialize_rescalers(hn, float32 table, anchor_count) with the drawn prior bias']
    else:
        order = [bias, 'rescalers left at identity (--no-rescaler-init)']
    recipe = {'seed': seed, 'prior_bias_std': float(prior_bias_std),
              'prior_bias_seed': seed + PRIOR_BIAS_SEED_OFFSET if prior_bias_std > 0 else None,
              'rescaler_init': bool(rescaler_init), 'anchor_count': anchor_count if rescaler_init else None,
              'calibration': calibration,
              'rescalers_from_fixture': ({'file': REFERENCE_NPZ.name, 'sha256': reference['sha256']}
                                         if copied else None),
              'build_input_hypernet': {'source_vocab_size': profile.source_vocab, 'embedding_width': profile.width,
                                       'other_arguments': 'defaults'},
              'order': ['torch.manual_seed(seed)', 'build_input_hypernet(source_vocab_size, embedding_width)',
                        *order, 'save_pretrained(dir, safe_serialization=True) before byte fallbacks']}
    recipe['option'] = recipe_option(recipe)
    return recipe


def _same_build(manifest: dict, recipe: dict, inputs: dict, code: dict) -> list:
    reasons = []
    if manifest.get('recipe') != recipe:
        reasons.append(f'recipe {manifest.get("recipe")} != requested {recipe}')
    old_files = {k: v.get('sha256') for k, v in ((manifest.get('inputs') or {}).get('files') or {}).items()}
    new_files = {k: v.get('sha256') for k, v in inputs['files'].items()}
    if old_files != new_files:
        reasons.append('the model files differ from those it was built from')
    if (manifest.get('code') or {}).get('files') != code['files']:
        reasons.append('the hf_hypernet sources differ')
    return reasons


def create(model_dir, code_dir, out, *, seed: int = DEFAULT_SEED, prior_bias_std: float = DEFAULT_PRIOR_BIAS_STD,
           rescaler_init: bool = True, force: bool = False, profile: Profile = PRODUCTION,
           run_dirs=None, progress=None, strict_env: bool = True, repo: Path | None = None,
           calibrate_after_bias: bool = False, allow_unpinned_model: bool = False) -> dict:
    """Build, verify and publish the init directory ``out``. Returns the manifest (``already_built`` if reused).

    ``run_dirs``: the production run directories (``$ANYTOK_RUNS/{uniform,rl}``) when ``out`` is
    the configured ``input/hypernet_init``; ``None`` when unknown.  ``progress(index, name)`` is
    called before each step of ``STEPS``.  ``calibrate_after_bias``: option (a) instead of the
    default (b).  ``allow_unpinned_model``: accept model files that differ from
    ``plans/zett-modal-20k-inputs.json`` (production profile; for local tests only).
    """
    started = time.time()
    timings = {}
    out = Path(os.path.abspath(out))
    model_dir = Path(os.path.abspath(model_dir))
    _check_recipe_args(seed, prior_bias_std, DEFAULT_ANCHORS)
    calibration = _check_calibration(CALIBRATE_AFTER_BIAS if calibrate_after_bias else
                                     (CALIBRATE_BEFORE_BIAS if rescaler_init else None), prior_bias_std, rescaler_init)

    def step(name):
        timings[name] = round(time.time() - started, 2)
        if progress is not None:
            progress(STEPS.index(name), name)
        _log(f'step {STEPS.index(name) + 1}/{len(STEPS)}: {name}')

    step('check inputs')
    environment = check_environment(strict=strict_env)
    hf = import_hf_hypernet(code_dir)
    reference = load_reference() if profile.reference else None
    inputs = inspect_model_dir(model_dir, profile, repo=repo)
    for name, record in inputs['files'].items():
        if record.get('matches_pinned') is False:
            if profile.expected_model_bytes is not None and not allow_unpinned_model:
                raise InitUsageError(
                    f'{model_dir / name} has SHA-256 {record.get("sha256")} ({record.get("bytes")} bytes), but '
                    f'plans/zett-modal-20k-inputs.json pins {record.get("pinned_sha256")}: the trainer would load '
                    'this directory as the LM. Remedy: python -m hyak.model.fetch_model (or, for a local test '
                    'only, --allow-unpinned-model)')
            _log(f'WARNING: {model_dir / name} differs from plans/zett-modal-20k-inputs.json; the content used '
                 'here is checked directly, but the trainer will load this directory as the LM')
    inputs['allow_unpinned_model'] = bool(allow_unpinned_model)
    if inputs['extra_files']:
        _log(f'WARNING: {model_dir} holds {inputs["extra_files"]} besides {list(MODEL_FILES)}; the original '
             'model directory held exactly those three, and AutoTokenizer would also read the others')
    recipe = _recipe(seed, prior_bias_std, rescaler_init, DEFAULT_ANCHORS, profile, calibration, reference)
    busy_runs = [str(p) for p in (run_dirs or ()) if _non_empty(Path(p))]
    cleanup_stale(out)
    if os.path.lexists(out):
        check = verify_directory(out, profile=profile, reference=reference)
        reasons = check['problems'] or _same_build(check['manifest'], recipe, inputs, hf.code)
        if not reasons and not force:
            _log(f'{out} already holds a verified initialization with the same recipe and inputs; nothing to do')
            manifest = dict(check['manifest'])
            manifest['already_built'] = True
            return manifest
        if busy_runs:
            raise InitUsageError(f'{out} exists and the run directories {busy_runs} are not empty: both runs started from '
                            'this initialization, so it must not be replaced. To restart the experiment from a new '
                            'initialization, move the run directories away first')
        if not force:
            raise InitUsageError(f'{out} exists but cannot be reused ({"; ".join(reasons)}). Both runs depend on it; '
                            'pass --force to replace it (only before any run has started)')
        _log(f'--force: {out} will be replaced ({"; ".join(reasons) or "same recipe, rebuilt on request"})')
    elif busy_runs:
        if not force:
            raise InitUsageError(f'{out} does not exist but the run directories {busy_runs} are not empty: they were '
                            'started from an initialization at this path. A rebuilt one is only valid if it is '
                            'bit-identical (the training wrapper compares its hash). Pass --force to rebuild anyway')
        _log(f'WARNING: rebuilding {out} although {busy_runs} are not empty (--force)')
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    # the build, the trainer-check copy and (with --force) the replaced directory coexist briefly
    need = 3 * (profile.expected_model_bytes or 0) + (1 << 28)
    if free < need:
        raise InitUsageError(f'{out.parent} has {free / 1e9:.1f} GB free; building needs about {need / 1e9:.1f} GB')

    build_dir = _temp_name(out, 'building')
    check_dir = _temp_name(out, 'trainer-check')
    for path in (build_dir, check_dir):
        if os.path.lexists(path):
            shutil.rmtree(path)
    build_dir.mkdir()
    check_dir.mkdir()
    checks = {}
    try:
        step('build')
        table, table_info = load_source_table(model_dir / 'model.safetensors', profile)
        hn, build_info = build_hypernet(hf, table, seed=seed, prior_bias_std=prior_bias_std,
                                        rescaler_init=rescaler_init, reference=reference, calibration=calibration)
        checks['construction_init'] = build_info.pop('construction_init')
        original = build_info.pop('original_construction')
        zero_bias_calibration = build_info.pop('calibration', None)
        fixture_regression = build_info.pop('in_scaler_regression', None)
        checks['original_construction'] = original      # enforced with the default recipe, else informative
        if original.get('copied_from_fixture'):
            _log('calibration with the zero prior bias reproduces the public checkpoint (in_scaler and scaler within '
                 f'{ORIGINAL_CONSTRUCTION_TOLERANCE}); the four frozen rescaler tensors were copied from '
                 f'{REFERENCE_NPZ.name} (sha256 {reference["sha256"][:16]}...)')
        elif original.get('applicable'):
            diff = max(original['zero_bias_max_abs_diff_w'], original['zero_bias_max_abs_diff_b'])
            if original['matches']:
                _log(f'construction reproduces the original warmup starting point (zero-bias scaler within '
                     f'{diff:.2g} of the public checkpoint); frozen scaler.w mean here '
                     f'{original["final_scaler_w_mean"]:.4f}, original {original["reference_scaler_w_mean"]:.4f}')
            else:
                _log(f'NOTE: the zero-bias scaler differs from the public checkpoint by {diff:.3g}: this machine '
                     'does not reproduce the original construction bit for bit (informative only)')
        counts = parameter_counts(hn)
        problems = []
        if profile.expected_parameters is not None and counts['total'] != profile.expected_parameters:
            problems.append(f'{counts["total"]} parameters, expected {profile.expected_parameters}')
        if profile.expected_frozen is not None and counts['frozen'] != profile.expected_frozen:
            problems.append(f'{counts["frozen"]} frozen parameters, expected {profile.expected_frozen}')
        if sorted(counts['frozen_names']) != sorted(RESCALER_NAMES):
            problems.append(f'frozen parameters {counts["frozen_names"]}, expected the four rescaler tensors')
        if hn.config.hn_byte_fallback_symbols is not None or hn.fallback_embeddings.num_embeddings != 1:
            problems.append('byte fallbacks already present before saving')
        if problems:
            raise InitError(f'Unexpected hypernetwork structure: {"; ".join(problems)}')
        checks['parameters'] = {'passed': True, **{k: v for k, v in counts.items() if k != 'frozen_names'}}
        outputs = anchor_outputs(hn, table, DEFAULT_ANCHORS)   # also the reference for the reload check
        if calibration == CALIBRATE_BEFORE_BIAS:
            report = zero_bias_calibration
            checks['calibration'] = report
            if not report['passed']:
                raise InitError(f'Calibrated anchor outputs of the zero-bias network do not match the canonical '
                                f'anchors: {report}')
            if prior_bias_std > 0:         # informative: the final network's anchors after the bias draw
                after = calibration_report(outputs, table, scope='final network, after the prior-bias draw')
                after.pop('passed')
                checks['anchors_after_bias'] = after
        elif rescaler_init:
            report = calibration_report(outputs, table, scope='final network (bias drawn before calibration)')
            checks['calibration'] = report
            if not report['passed']:
                raise InitError(f'Calibrated anchor outputs do not match the canonical anchors: {report}')
        else:
            checks['calibration'] = {'passed': None, 'skipped': 'rescalers left at identity (--no-rescaler-init)'}
        if reference is not None:
            regression = fixture_regression or in_scaler_regression(hn, reference)
            if rescaler_init:
                checks['in_scaler_regression'] = regression
                if not regression['passed']:
                    raise InitError('in_scaler differs from the original checkpoint by more than '
                                    f'{regression["tolerance"]} (w {regression["max_abs_diff_w"]:.3g}, b '
                                    f'{regression["max_abs_diff_b"]:.3g}): the input table is not the pinned '
                                    'SmolLM2 table. Remedy: python -m hyak.model.fetch_model')
            else:
                regression['passed'] = None
                regression['skipped'] = 'identity rescalers (--no-rescaler-init)'
                checks['in_scaler_regression'] = regression
        state = hn.state_dict()
        tensors = tensor_report(state)
        bad = [name for name, row in tensors.items() if not row['finite']]
        if bad:
            raise InitError(f'Non-finite values in {bad}')
        checks['finite'] = {'passed': True, 'tensors': len(tensors)}
        digests = {name: row['sha256'] for name, row in tensors.items()}
        predicted = predicted_safetensors_bytes(state)
        init_fallback_sha = _tensor_sha256(hn.fallback_embeddings.weight[:1])
        del state, table          # lower the save peak (safetensors copies every tensor); reloaded below
        gc.collect()

        step('save')
        hn.save_pretrained(str(build_dir), safe_serialization=True)   # D1: before any byte-fallback expansion
        del hn
        gc.collect()

        step('verify files')
        saved = check_saved_files(build_dir, tensors, profile)
        saved['predicted_bytes'] = predicted   # NOTES/hypernetwork.md 3.2 size formula (informative)
        saved['matches_prediction'] = saved['model_bytes'] == predicted
        checks['files'] = saved
        config = read_json(build_dir / 'config.json')
        checks['config'] = check_init_config(config, profile, reference)

        step('reload')
        hn = reload_hypernet(hf, build_dir)
        reloaded = {name: _tensor_sha256(t) for name, t in hn.state_dict().items()}
        if reloaded != digests:
            differ = sorted(n for n in set(reloaded) | set(digests) if reloaded.get(n) != digests.get(n))
            raise InitError(f'ZettHypernet.from_pretrained does not reproduce the saved tensors: {differ[:10]}')
        table, table_again = load_source_table(model_dir / 'model.safetensors', profile)
        if table_again['float32_sha256'] != table_info['float32_sha256']:
            raise InitError(f'{model_dir / "model.safetensors"} changed while the initialization was being built')
        again = anchor_outputs(hn, table, DEFAULT_ANCHORS)
        max_diff = float((again - outputs).abs().max())
        relative = float((again - outputs).norm() / outputs.norm().clamp_min(1e-30))
        identical = bool(torch.equal(again, outputs))
        # The tensors are bit-identical (checked above); the reloaded weights sit at another memory
        # alignment, which a BLAS may round differently, so only a real difference fails here.
        if not torch.allclose(again, outputs, rtol=RELOAD_RTOL, atol=RELOAD_ATOL):
            raise InitError(f'The reloaded hypernetwork computes different anchor outputs (max |diff| {max_diff:.3g}, '
                            f'relative {relative:.3g}, tolerance rtol {RELOAD_RTOL} atol {RELOAD_ATOL})')
        if not identical:
            _log(f'NOTE: reloaded anchor outputs are not bit-identical (max |diff| {max_diff:.3g}, relative '
                 f'{relative:.3g}): rounding of the BLAS on differently aligned weights; the tensors are identical')
        checks['reload'] = {'passed': True, 'tensors_bit_identical': True, 'anchor_output_max_abs_diff': max_diff,
                            'anchor_output_relative_error': relative, 'anchor_outputs_bit_identical': identical,
                            'tolerance': {'rtol': RELOAD_RTOL, 'atol': RELOAD_ATOL}}
        del again, outputs

        step('gradient diagnostic')
        hf.freeze_rescalers(hn)
        diagnostic = prior_gradient_diagnostic(hn, table, seed=seed + DIAGNOSTIC_SEED_OFFSET,
                                               vocab=profile.diagnostic_vocab, one_piece=profile.diagnostic_one_piece)
        if diagnostic['degenerate']:
            _log(f'WARNING: zero-prior gradient ratio {diagnostic["ratio"]:.3g}: the prior bias gradient dominates '
                 'the joint clip on BPE/WordPiece updates (expected with --prior-bias-std 0)')
        del table
        gc.collect()

        step('trainer check')
        tokenizer = load_source_tokenizer(model_dir)
        trainer = trainer_first_start(hf, hn, tokenizer, check_dir, profile=profile, reference=reference,
                                      init_fallback_sha256=init_fallback_sha)
        del hn
        gc.collect()
        trainer['resume'] = trainer_resume(hf, check_dir, tokenizer, trainer['fallback_rows'])
        checks['trainer_emulation'] = trainer

        step('publish')
        outputs_record = {name: file_record(build_dir / name) for name in OUTPUT_FILES}
        for name in OUTPUT_FILES:
            os.chmod(build_dir / name, 0o444)
        manifest = {
            'format': MANIFEST_FORMAT, 'complete': True, 'created_at_utc': utc_now(), 'directory': str(out),
            'tool': {'module': 'hyak.model.init_hypernet', 'sha256': file_record(Path(__file__))['sha256']},
            'recipe': recipe, 'build': build_info, 'environment': environment, 'code': hf.code,
            'inputs': {**inputs, 'table': table_info},
            'outputs': outputs_record,
            'original_warmup': original_warmup_record(build_dir / 'config.json', outputs_record),
            'checks': checks, 'prior_gradient_diagnostic': diagnostic, 'tensors': tensors,
            'reference': ({k: reference[k] for k in ('repository', 'model_file', 'config_file', 'step', 'sha256')}
                          if reference is not None else None),
            'run_directories_checked': run_dirs is not None,
            'step_started_after_seconds': timings, 'seconds': None, 'peak_rss_bytes': None,
        }
        manifest['seconds'] = round(time.time() - started, 1)
        manifest['peak_rss_bytes'] = _peak_rss()
        atomic_write_json(build_dir / MANIFEST_NAME, manifest)
        os.chmod(build_dir / MANIFEST_NAME, 0o444)
        _publish(build_dir, out, force)
        return manifest
    finally:
        for path in (build_dir, check_dir):
            if os.path.lexists(path):
                shutil.rmtree(path, ignore_errors=True)


def original_warmup_record(config_path: Path, outputs_record: dict) -> dict:
    """How this directory relates to the original warmup checkpoint (informative, never a failure)."""
    config_bytes = Path(config_path).read_bytes()
    without_key = config_bytes.replace(FALLBACK_CONFIG_LINE, b'', 1)
    return {
        'config_sha256': ORIGINAL_WARMUP_CONFIG_SHA256, 'model_sha256': ORIGINAL_WARMUP_MODEL_SHA256,
        'model_bytes': EXPECTED_MODEL_BYTES,
        'config_sha256_equal': outputs_record['config.json']['sha256'] == ORIGINAL_WARMUP_CONFIG_SHA256,
        'config_equal_without_fallback_key': (without_key != config_bytes
                                              and sha256_bytes(without_key) == ORIGINAL_WARMUP_CONFIG_SHA256),
        'note': ('The configuration is the original warmup\'s: this config.json without the line '
                 '"hn_byte_fallback_symbols": null (a field the config class did not have yet; it defaults to '
                 'None) hashes to the original warmup config. Only the weights differ: the public checkpoint shows '
                 'that the original warmup ran identity_warmup_step with the library defaults for 10,000 steps '
                 '(peak learning rate 3e-4, linear ramp, zero priors, one-piece forms) from the seed-42 '
                 'construction; its batch size is unknown and the warmup weights are private, so a random start '
                 'cannot match them.'),
    }


def _peak_rss() -> int:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


# --- Command line -----------------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m hyak.model.init_hypernet', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, help='SmolLM2 directory (default $ANYTOK_ROOT/input/model)')
    parser.add_argument('--code-dir', type=Path,
                        help='tree containing hf_hypernet/ (default $ANYTOK_ROOT/code/uniform)')
    parser.add_argument('--out', type=Path, help='output directory (default $ANYTOK_ROOT/input/hypernet_init)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='construction seed (default 42)')
    parser.add_argument('--prior-bias-std', type=float, default=DEFAULT_PRIOR_BIAS_STD,
                        help='std of model.target_priors_projection.bias (default 0.02; 0 keeps the library zeros, '
                             'option (c), not recommended)')
    parser.add_argument('--calibrate-after-bias', action='store_true',
                        help='option (a): draw the prior bias first, then calibrate the rescalers (the frozen output '
                             'scaler then differs from the original runs\'); default (b) calibrates the zero-bias '
                             'construction, checks and copies the original rescalers, then draws the bias')
    parser.add_argument('--allow-unpinned-model', action='store_true',
                        help='accept model files that differ from plans/zett-modal-20k-inputs.json (local tests only)')
    parser.add_argument('--no-rescaler-init', action='store_true',
                        help='leave in_scaler/scaler at identity (not recommended; the original pipeline calibrated them)')
    parser.add_argument('--force', action='store_true',
                        help='replace an existing directory (refused while a run directory is not empty)')
    parser.add_argument('--threads', type=int,
                        help='torch CPU threads (default $SLURM_CPUS_PER_TASK, else the torch default)')
    parser.add_argument('--status', action='store_true',
                        help='write progress to the init stage status.json/events.jsonl (sbatch use)')
    parser.add_argument('--verify', action='store_true', help='only check an existing directory against its manifest')
    parser.add_argument('--config', type=Path, help='config file (default $HYAK_CONFIG or hyak/config.env)')
    return parser


def _resolve(args) -> tuple:
    """Paths from the arguments, filling defaults from the config. Returns (model, code, out, cfg, run_dirs)."""
    from hyak.common.config import ConfigError, load_config
    from hyak.common.paths import RUNS, Paths
    if args.verify:          # --verify reads only --out
        need_config = args.status or args.out is None
    else:
        need_config = args.status or None in (args.model_dir, args.code_dir, args.out)
    cfg = paths = None
    try:
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
    except (ConfigError, ValueError) as exc:
        if need_config:
            raise InitUsageError(f'{exc}') from exc
    out = args.out or paths.hypernet_init
    if args.verify:
        return args.model_dir, args.code_dir, Path(out), cfg, None
    model = args.model_dir or paths.model_dir
    code = args.code_dir or paths.code_dir('uniform')
    run_dirs = None
    if paths is not None and os.path.normpath(os.path.abspath(out)) == os.path.normpath(str(paths.hypernet_init)):
        run_dirs = [paths.run_dir(run) for run in RUNS]
    return Path(model), Path(code), Path(out), cfg, run_dirs


EXIT_OK, EXIT_CRASH, EXIT_USAGE, EXIT_CHECK_FAILED = 0, 1, 2, 4


def main(argv=None) -> int:
    """Command line; returns the exit code (module docstring: 0, 2 usage/input, 4 check failed, 1 crash, 128+signal)."""
    args = _parser().parse_args(argv)
    sys.dont_write_bytecode = True
    # Before the first transformers/huggingface_hub import, which reads the variable at import time
    # (in a process that imported them already it has no effect; every load is local_files_only anyway).
    previous_offline = os.environ.get('HF_HUB_OFFLINE')
    os.environ['HF_HUB_OFFLINE'] = '1'

    def terminate(signum, _frame):
        raise Terminated(signum)

    previous_handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.disable_progress_bar()    # keep the Slurm log readable (no \r progress bars)
        return _main(args)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if previous_offline is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = previous_offline


def _main(args) -> int:
    writer = None
    index = [0]
    try:
        model, code, out, cfg, run_dirs = _resolve(args)
        if args.verify:
            result = verify_directory(out)
            for problem in result['problems']:
                print(f'PROBLEM: {problem}', file=sys.stderr)
            print(f'{"VERIFIED" if result["ok"] else "NOT VERIFIED"}: {out}', flush=True)
            return EXIT_OK if result['ok'] else EXIT_CHECK_FAILED
        threads = args.threads
        if threads is None and os.environ.get('SLURM_CPUS_PER_TASK', '').isdigit():
            threads = int(os.environ['SLURM_CPUS_PER_TASK'])
        if threads is not None:
            if threads < 1:
                raise InitUsageError(f'--threads must be >= 1, got {threads}')
            torch.set_num_threads(threads)
        if args.status:
            from hyak.common.paths import STAGE_INIT
            from hyak.common.status import StatusWriter
            writer = StatusWriter(STAGE_INIT, total=len(STEPS), unit='steps', cfg=cfg)

        def progress(i, name):
            index[0] = i
            if writer is not None:
                writer.update(i, stage=name, state='running', workers=[writer.worker_row(task=f'init: {name}')])

        if args.no_rescaler_init:
            _log('WARNING: --no-rescaler-init leaves in_scaler/scaler at identity forever (they are frozen); '
                 'the original pipeline calibrated them')
        if args.calibrate_after_bias:
            _log('NOTE: --calibrate-after-bias (option (a)): the frozen output scaler will differ from the original '
                 'runs\' scaler for all 20,000 updates')
        if args.prior_bias_std == 0:
            _log('WARNING: --prior-bias-std 0 (option (c)): zero-prior updates have a degenerate gradient that '
                 'dominates the joint clip; not recommended for the uniform-versus-RL comparison')
        manifest = create(model, code, out, seed=args.seed, prior_bias_std=args.prior_bias_std,
                          rescaler_init=not args.no_rescaler_init, force=args.force, run_dirs=run_dirs,
                          progress=progress, calibrate_after_bias=args.calibrate_after_bias,
                          allow_unpinned_model=args.allow_unpinned_model)
        model_record = manifest['outputs']['model.safetensors']
        diag = manifest.get('prior_gradient_diagnostic') or {}
        recipe = manifest.get('recipe') or {}
        fixture = recipe.get('rescalers_from_fixture')
        if writer is not None:
            writer.remove_job_rows()     # the job script's 'start' row would otherwise linger until finalize
            writer.update(len(STEPS), stage='complete', state='complete', workers=[],
                          directory=str(out), model_sha256=model_record['sha256'],
                          already_built=bool(manifest.get('already_built')),
                          recipe={k: recipe.get(k) for k in ('option', 'seed', 'prior_bias_std', 'calibration',
                                                             'rescalers_from_fixture')})
        print(f'INIT OK: {out} model.safetensors {model_record["bytes"]} bytes sha256 {model_record["sha256"]} '
              f'zero-prior gradient ratio {diag.get("ratio", float("nan")):.3g}; recipe '
              f'{recipe.get("option", "unrecorded (older manifest)")}, seed {recipe.get("seed")}, prior-bias std '
              f'{recipe.get("prior_bias_std")}, calibration {recipe.get("calibration")}'
              + (f', rescalers copied from {fixture["file"]} sha256 {fixture["sha256"][:16]}...' if fixture else '')
              + f'{" (already built)" if manifest.get("already_built") else ""}', flush=True)
        return EXIT_OK
    except Terminated as exc:
        signum = int(exc.args[0]) if exc.args else int(signal.SIGTERM)
        message = f'interrupted by signal {signum}; temporary files removed, nothing published'
        print(f'INIT STOPPED: {message}', file=sys.stderr, flush=True)
        if writer is not None:
            writer.remove_job_rows()
            writer.update(index[0], stage='stopped', state='stopped', workers=[], error=message)
        return 128 + signum
    except Exception as exc:  # report every failure the same way, with the traceback for unexpected ones
        if isinstance(exc, InitUsageError):
            code = EXIT_USAGE
        elif isinstance(exc, InitError):
            code = EXIT_CHECK_FAILED
        elif isinstance(exc, OSError):
            code = EXIT_USAGE
            import traceback
            traceback.print_exc()
            exc = InitUsageError(f'{exc}. Remedy: check the path, its permissions and free space on the file system')
        else:
            code = EXIT_CRASH
            import traceback
            traceback.print_exc()
        print(f'INIT FAILED: {exc}', file=sys.stderr, flush=True)
        if writer is not None:
            try:
                writer.remove_job_rows()
            except OSError:
                pass
            writer.update(index[0], stage='failed', state='failed', workers=[], error=str(exc)[:2000],
                          exit_code=code)
        return code


if __name__ == '__main__':
    sys.exit(main())
