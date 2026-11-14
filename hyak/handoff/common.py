"""Shared facts of the raw-BPB handoff run (standard library only, except :func:`load_state`).

The pins are Qingyun's committed files in ``docs/raw-bpb-handoff/`` (the repository copy is authoritative):
``bundle-reference.json`` (Hugging Face revision, archive URL, bytes, SHA-256), ``bundle-receipt.json``,
``BUNDLE_MANIFEST.json`` (every file of the extracted ``input/`` tree) and ``requirements.txt``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

#: Qingyun's handoff directory, relative to the repository.
DOCS_REL = 'docs/raw-bpb-handoff'
REFERENCE_NAME = 'bundle-reference.json'
RECEIPT_NAME = 'bundle-receipt.json'
MANIFEST_NAME = 'BUNDLE_MANIFEST.json'
REQUIREMENTS_NAME = 'requirements.txt'
#: His launcher and his input verifier, used as they are.
LAUNCHER_REL = 'scripts/run_raw_bpb_training.py'
VERIFY_INPUTS_REL = 'scripts/verify_training_inputs.py'
#: The committed Rust sampler the launcher stages (docs/raw-bpb-handoff/README.md, "Environment").
RUST_LIBRARY_REL = 'runtime/anytok_pool_20k/rust_utils/target/release/librust_utils.so'

#: Updates of the run (the launcher's default --stop-at-update) and the smoke's two phases.
FINAL_STEP = 20000
SMOKE_PHASES = (2, 3)
TOKENIZERS = 1000
CORPUS_BUCKETS = 23
#: What the raw-BPB run's run_config.json must say (docs/raw-bpb-handoff/README.md, "What must stay the same").
EXPECTED_CONFIG = {
    'reward_objective': 'raw_bpb',
    'policy_reward': 'paired_pre_model_update_transferred_bpb',
    'memory_execution': 'optimized',
    'pool_size': TOKENIZERS,
    'batch_size': 128,
    'microbatch_size': 32,
    'sequence_length': 128,
    'seed': 42,
}
EXPECTED_POLICY = {'entropy_weight': 0.0, 'learning_rate': 0.001, 'baseline_decay': 0.99, 'seed': 43,
                   'size': TOKENIZERS, 'kind': 'shared_settings_mlp'}
#: |paired_gap_bpb - (train_bpb - paired_canonical_bpb)| tolerance of scripts/verify_raw_bpb_handoff.py.
GAP_TOLERANCE = 1e-12
PAIRED_EXAMPLES = 128
MAX_POSITIONS = 128

_STEP_RE = re.compile(r'step-(\d{6})')


def docs_dir(repo) -> Path:
    return Path(repo) / DOCS_REL


def load_reference(reference_dir) -> dict:
    """``{"reference", "receipt", "manifest_path", "requirements_path", "archive"}`` from a handoff directory."""
    reference_dir = Path(reference_dir)
    reference = json.loads((reference_dir / REFERENCE_NAME).read_text())
    receipt = json.loads((reference_dir / RECEIPT_NAME).read_text())
    archive = receipt.get('archive') or reference['archive_url'].rsplit('/', 1)[-1]
    if not reference['archive_url'].endswith('/' + archive):
        raise ValueError(f'{REFERENCE_NAME}: archive_url does not end with the receipt\'s archive {archive}')
    for key in ('sha256', 'bytes'):
        if receipt.get(key) != reference.get(key):
            raise ValueError(f'{REFERENCE_NAME} and {RECEIPT_NAME} disagree on {key}')
    return {'reference': reference, 'receipt': receipt, 'archive': archive,
            'manifest_path': reference_dir / MANIFEST_NAME, 'requirements_path': reference_dir / REQUIREMENTS_NAME}


def outputs(cfg: dict) -> tuple:
    """``(train_output, smoke_output)`` as configured (HYAK_RAWBPB_OUTPUT / HYAK_RAWBPB_SMOKE_OUTPUT), else the
    defaults ``$ANYTOK_RUNS/raw-bpb-20k`` and ``$ANYTOK_RUNS/raw-bpb-smoke``."""
    runs = Path(os.path.normpath(cfg['ANYTOK_RUNS']))
    train = cfg.get('HYAK_RAWBPB_OUTPUT') or str(runs / 'raw-bpb-20k')
    smoke = cfg.get('HYAK_RAWBPB_SMOKE_OUTPUT') or str(runs / 'raw-bpb-smoke')
    return Path(train), Path(smoke)


# --- requirements ---------------------------------------------------------------------------------------
def normalise_name(name: str) -> str:
    """PEP 503 name: ``huggingface_hub`` -> ``huggingface-hub``."""
    return re.sub(r'[-_.]+', '-', name).lower()


def parse_requirements(path) -> list:
    """``[(name, version), ...]`` of a requirements file of exact ``name==version`` lines."""
    pins = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9_.+!-]+)', line)
        if not match:
            raise ValueError(f'{path}:{number}: not an exact pin: {line!r}')
        pins.append((match.group(1), match.group(2)))
    return pins


def installed_versions(names) -> dict:
    import importlib.metadata as md
    found = {}
    for name in names:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            found[name] = None
    return found


def requirement_problems(pins, installed: dict) -> list:
    """Every pin that the installed set does not equal (a local build label such as ``+cu132`` is allowed:
    PEP 440 ``==2.15.0`` matches ``2.15.0+cu132``)."""
    problems = []
    for name, wanted in pins:
        have = installed.get(name)
        if have is None:
            problems.append(f'{name} is not installed (requirements.txt: {name}=={wanted})')
        elif str(have).split('+', 1)[0] != wanted:
            problems.append(f'{name} {have} is installed, requirements.txt pins {wanted}')
    return problems


# --- the trainer's output directory -------------------------------------------------------------------
def step_dirs(output) -> list:
    """``[(step, path), ...]`` of every ``step-NNNNNN`` directory, ascending."""
    found = []
    output = Path(output)
    if not output.is_dir():
        return found
    for path in output.iterdir():
        match = _STEP_RE.fullmatch(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def full_restarts(output) -> list:
    """``[(step, path), ...]`` of the directories the trainer can resume from: those holding ``state.pt``
    (``train_joint_smollm_input.py`` resumes from the newest of them)."""
    return [(step, path) for step, path in step_dirs(output) if (path / 'state.pt').is_file()]


def restart_problems(step: int, path) -> list:
    """Structural problems of a full restart (the trainer re-hashes every file when it resumes)."""
    path = Path(path)
    manifest_path = path / 'manifest.json'
    if not manifest_path.is_file():
        return [f'{path}: manifest.json is missing']
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        return [f'{manifest_path}: unreadable ({exc})']
    problems = []
    if manifest.get('step') != step:
        problems.append(f'{manifest_path}: step {manifest.get("step")} instead of {step}')
    if manifest.get('kind') != 'full_restart_checkpoint':
        problems.append(f'{manifest_path}: kind {manifest.get("kind")!r} is not full_restart_checkpoint')
    files = manifest.get('files') or {}
    for required in ('state.pt', 'optimizer.pt', 'lm.pt', 'policy.pt'):
        if required not in files:
            problems.append(f'{manifest_path}: {required} is not listed')
    for name, item in files.items():
        target = path / name
        if not target.is_file():
            problems.append(f'{target}: missing')
        elif target.stat().st_size != item.get('bytes'):
            problems.append(f'{target}: {target.stat().st_size} bytes, the manifest says {item.get("bytes")}')
    return problems


def output_is_empty(output) -> bool:
    output = Path(output)
    return not output.exists() or (output.is_dir() and not any(output.iterdir()))


def load_state(path) -> dict:
    """``state.pt`` on the CPU, as scripts/verify_raw_bpb_handoff.py reads it (``weights_only=True``)."""
    import torch
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def counters(state: dict) -> dict:
    """``{"model", "policy", "selections"}`` of a loaded ``state.pt``."""
    return {'model': int(state['step']), 'policy': int(state['bandit']['updates']),
            'selections': int(state['bandit']['counts'].sum())}
