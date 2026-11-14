"""Pinned identities of every repository file the port stages or builds from (DESIGN.md 5.A).

All values were computed with SHA-256 over the raw bytes, from the repository at commit
``c2b9c400066a3e0aa799881ab177996c618b02da`` (branch ``zett-modal-20k``), except the files of
Qingyun's joint regret RL20k snapshot ``runtime/anytok_pool_regret_rl_20k`` (pinned at
``3de64a55815ec5ebecb37f4e47b7addbe886cdde``) and the port's own RL trainer variant under
``hyak/variants/anytok_pool_straight_rl_20k``.  The uniform trainer, policy, ``joint_training.py``
and ``pool_sampling.py`` values equal the hashes the original runs recorded (``trainer_sha256`` in
``plans/uniform-trajectory-inventory.json`` and
``reports/rl-tokenizer-selection-20k/data/provenance/launch-verification.json``).

The staged tree layouts reproduce the Modal training images:

* uniform -- ``scripts/modal_anytok_pool_20k.py:37-50``: ``add_local_dir(SOURCE/"hf_hypernet")``,
  ``add_local_file(SOURCE/"scripts/train_joint_smollm_input.py")``,
  ``add_local_file(SOURCE/"rust_utils/target/release/librust_utils.so")`` with
  ``SOURCE = runtime/anytok_pool_20k`` (the original uniform run, unchanged);
* rl -- the image of the joint regret launcher (``scripts/modal_anytok_pool_rl_20k.py:56-67`` with
  ``ANYTOK_RL_EXPERIMENT=regret``): the same ``hf_hypernet`` directory and ``.so`` from
  ``BASE = runtime/anytok_pool_20k``, ``settings_policy.py`` from ``POLICY_SOURCE =
  runtime/anytok_pool_settings_rl_20k``, ``paired_reward.py`` from ``SOURCE =
  runtime/anytok_pool_regret_rl_20k`` -- and, at ``scripts/train_joint_smollm_input.py``, not the
  regret trainer but its straight-BPB-reward variant (the user's decision of 2026-11-14,
  ``hyak/variants/anytok_pool_straight_rl_20k/README.md``: the policy receives ``train_bpb``
  instead of the paired gap; nothing else differs).  A union: no path comes from two sources.

Command line (standard library only, Python >= 3.9)::

    python -m hyak.code.pinned --check [--group all|snapshot|variant|rust-sources|worker|reference]
    python -m hyak.code.pinned --layout uniform|rl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

#: Commit at which the constants below were computed (the regret snapshot: REGRET_PINNED_AT_COMMIT).
PINNED_AT_COMMIT = 'c2b9c400066a3e0aa799881ab177996c618b02da'
#: Commit at which the files of the joint regret RL20k snapshot were pinned.
REGRET_PINNED_AT_COMMIT = '3de64a55815ec5ebecb37f4e47b7addbe886cdde'

UNIFORM_SNAPSHOT = 'runtime/anytok_pool_20k'
#: The original shared-settings RL snapshot: source of settings_policy.py (its trainer is pinned, not staged).
RL_SNAPSHOT = 'runtime/anytok_pool_settings_rl_20k'
#: Qingyun's joint regret RL20k snapshot: source of paired_reward.py and base of the port's RL trainer.
REGRET_SNAPSHOT = 'runtime/anytok_pool_regret_rl_20k'
#: The port's RL trainer variant (straight transferred BPB as the policy reward).
VARIANT_DIR = 'hyak/variants/anytok_pool_straight_rl_20k'
#: The pinned runtime snapshots (never edited; every file is pinned below).
RUNTIME_SNAPSHOTS = (UNIFORM_SNAPSHOT, RL_SNAPSHOT, REGRET_SNAPSHOT)

#: Paths inside a staged tree (the trainer derives everything from its own location:
#: runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py:22-30).
TRAINER_REL = 'scripts/train_joint_smollm_input.py'
RUST_LIBRARY_REL = 'rust_utils/target/release/librust_utils.so'
POLICY_REL = 'hf_hypernet/settings_policy.py'
PAIRED_REWARD_REL = 'hf_hypernet/paired_reward.py'
#: Files only the RL tree has (besides its trainer).
RL_ONLY_RELS = (POLICY_REL, PAIRED_REWARD_REL)

#: The complete hf_hypernet package of the uniform snapshot (copied whole by add_local_dir).
HF_HYPERNET_FILES = (
    '__init__.py', 'configuration_hypernet.py', 'hyper_roberta.py', 'hyper_t5.py',
    'identity_warmup.py', 'joint_training.py', 'modeling_hypernet.py', 'pool_sampling.py',
    'smollm_setup.py', 'tokenizer_inputs.py',
)

#: Every file of the runtime snapshots (staged, or the base of a staged variant): repo-relative path ->
#: (sha256, bytes).  The two regret files were pinned at REGRET_PINNED_AT_COMMIT; the original RL trainer and
#: the regret trainer are not staged (the RL tree runs VARIANT_FILES).
SNAPSHOT_FILES = {
    'runtime/anytok_pool_20k/hf_hypernet/__init__.py': ('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 0),
    'runtime/anytok_pool_20k/hf_hypernet/configuration_hypernet.py': ('e986d140ad63311b63425a3cac6e4af2410ed180c184847aca80e0ab6af554de', 2842),
    'runtime/anytok_pool_20k/hf_hypernet/hyper_roberta.py': ('5c385150184fd0a55fb45a289a4eed2b6b7756880849f080c4f55b25355b8512', 9559),
    'runtime/anytok_pool_20k/hf_hypernet/hyper_t5.py': ('c40548e12cb2f2074ead8ce7876cf40c1ed77b057e735c0b6f9a8027b3e50eb2', 8525),
    'runtime/anytok_pool_20k/hf_hypernet/identity_warmup.py': ('1c5cca36d2a2b3cdfeb7e2d3fbb8efdc300c69fd6f0bee47e2876d49b8ff5666', 6928),
    'runtime/anytok_pool_20k/hf_hypernet/joint_training.py': ('7ce18a9d9843faac1a9438f6ad595588569c0512574b7a602e5b11b432bdaf5e', 9358),
    'runtime/anytok_pool_20k/hf_hypernet/modeling_hypernet.py': ('9314094415250b093040f27948434f949811913d5a82b77c5676681361031541', 15817),
    'runtime/anytok_pool_20k/hf_hypernet/pool_sampling.py': ('846446a0ddb8b833801bb74b7939452a3195760a98e9bab2a3dd2e73d57f1b57', 3037),
    'runtime/anytok_pool_20k/hf_hypernet/smollm_setup.py': ('17082b9e68320bb763ba6ca8f0772ec4fc49f84e23abf404b27d1a4b299eed06', 1155),
    'runtime/anytok_pool_20k/hf_hypernet/tokenizer_inputs.py': ('70e99d9edb56fead0597a2d593dc992a2ad3b85d688c09bfe953bcc7d42077c8', 4053),
    'runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py': ('c6406d263fb9cbc8638492d8c77327ce249e215eb74dc43e10335bc95a11ed52', 30982),
    'runtime/anytok_pool_20k/rust_utils/target/release/librust_utils.so': ('9b485a745b374afbe20d846eba6e63c47fa251ff56c133f40c20c0fa7000e2f4', 3810872),
    'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py': ('6dbfcbf8c6552ef98b56de08c09ae59bd35a2d7d1a75e7e295cfcf5346a90b8c', 8836),
    'runtime/anytok_pool_settings_rl_20k/scripts/train_joint_smollm_input.py': ('36057349fd398b363a84ce6d8f12438e49cb17b401e4afdaf9e93a6333a691af', 33885),
    'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py': ('406cfc58bf1c46cef0dc6785bbb313ca5797c0d35bf6cd786b3115a92e5523c1', 4116),
    'runtime/anytok_pool_regret_rl_20k/scripts/train_joint_smollm_input.py': ('76ff232402154653c5899826a4f63f8a4afadf5a443829f87919f8e75b31c6d4', 34752),
}

#: The port's own RL trainer: the regret trainer with the policy reward changed to the straight transferred BPB
#: (hyak/variants/anytok_pool_straight_rl_20k/README.md; straight_reward.patch is the exact diff).
VARIANT_FILES = {
    'hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py': ('8dd3b8570dd6545123ab1aee5d338a8957d72ece26e48a759196d2619f43ff51', 34723),
}

#: Every source a staged tree may copy from (tree_layout values other than the Rust library).
STAGED_SOURCE_FILES = {**SNAPSHOT_FILES, **VARIANT_FILES}

#: Sources of the fallback rebuild of librust_utils.so (hyak/env/build_rust_utils.sh).
RUST_SOURCE_FILES = {
    'runtime/anytok_pool_20k/rust_utils/Cargo.toml': ('6daee3d7173013d62e024a88d10618d2cd4bf1371076c8e3a56ac09fbabe5e68', 489),
    'runtime/anytok_pool_20k/rust_utils/Cargo.lock': ('4d56ffc8c5aba3983a053e38a7c5103c419c2741487cfefa0347c0eccb41d90f', 25085),
    'runtime/anytok_pool_20k/rust_utils/pyproject.toml': ('ef76e3af35e0f3dc2bb354bc698a77ac3b0745b34aef6c1fa7ca7340e38915fe', 391),
    'runtime/anytok_pool_20k/rust_utils/rust-toolchain': ('2b92ea252be0fbc26f70317cdaa7b6411ea634b50d55338cd8c495e4dbf25d1d', 7),
    'runtime/anytok_pool_20k/rust_utils/src/lib.rs': ('0fd38c2f2a54771adc7dc509c0acec000cbea62b79885b52ee8f22bdc75724c4', 8629),
}

#: The unchanged tokenizer-pool worker (run as a child process by hyak.pool.worker).
WORKER_FILES = {
    'scripts/tokenizer_pool.py': ('b194b104a6dc4843f52f939d5f7bf9826b05e7b007bd2c03aef12f04be588a83', 12531),
}

#: Reference data read by the port (their owners check them as well).
REFERENCE_FILES = {
    'plans/zett-modal-20k-inputs.json': ('53d2968d9a1e1a7917b7f05716bca77acfcf9b61a06a08d35fa2bb3b4bbefb8c', 8353),
    'data/balanced-full-v3.json': ('b3bc193333aa13d3a513f1c8978805835f837190cd0e5991742c55e9a0752cfe', 7836),
    'plans/tokenizer-pool-1000-v1.json': ('4bbd89c6d40993a036ec33072a8300acf9f50a222e8b258c9ec2e0b24fe43b8e', 1230775),
}

PINNED_FILES = {**SNAPSHOT_FILES, **VARIANT_FILES, **RUST_SOURCE_FILES, **WORKER_FILES, **REFERENCE_FILES}

GROUPS = {
    'snapshot': SNAPSHOT_FILES,
    'variant': VARIANT_FILES,
    'rust-sources': RUST_SOURCE_FILES,
    'worker': WORKER_FILES,
    'reference': REFERENCE_FILES,
    'all': PINNED_FILES,
}

# --- named values used across the port --------------------------------------------------
UNIFORM_TRAINER = f'{UNIFORM_SNAPSHOT}/{TRAINER_REL}'
#: The trainer of the original RL run (entropy 0.01, raw reward, original cuts): pinned, no longer staged.
ORIGINAL_RL_TRAINER = f'{RL_SNAPSHOT}/{TRAINER_REL}'
#: Qingyun's joint regret RL20k trainer: the base of the port's RL trainer (pinned, not staged).
REGRET_TRAINER = f'{REGRET_SNAPSHOT}/{TRAINER_REL}'
#: The RL trainer the port stages and runs (straight-BPB-reward variant of REGRET_TRAINER).
RL_TRAINER = f'{VARIANT_DIR}/{TRAINER_REL}'
SETTINGS_POLICY = f'{RL_SNAPSHOT}/{POLICY_REL}'
PAIRED_REWARD = f'{REGRET_SNAPSHOT}/{PAIRED_REWARD_REL}'
JOINT_TRAINING = f'{UNIFORM_SNAPSHOT}/hf_hypernet/joint_training.py'
POOL_SAMPLING = f'{UNIFORM_SNAPSHOT}/hf_hypernet/pool_sampling.py'
RUST_LIBRARY = f'{UNIFORM_SNAPSHOT}/{RUST_LIBRARY_REL}'
TOKENIZER_POOL = 'scripts/tokenizer_pool.py'
INPUTS_MANIFEST = 'plans/zett-modal-20k-inputs.json'

UNIFORM_TRAINER_SHA256 = SNAPSHOT_FILES[UNIFORM_TRAINER][0]
ORIGINAL_RL_TRAINER_SHA256 = SNAPSHOT_FILES[ORIGINAL_RL_TRAINER][0]
REGRET_TRAINER_SHA256 = SNAPSHOT_FILES[REGRET_TRAINER][0]
RL_TRAINER_SHA256 = VARIANT_FILES[RL_TRAINER][0]
SETTINGS_POLICY_SHA256 = SNAPSHOT_FILES[SETTINGS_POLICY][0]
PAIRED_REWARD_SHA256 = SNAPSHOT_FILES[PAIRED_REWARD][0]
JOINT_TRAINING_SHA256 = SNAPSHOT_FILES[JOINT_TRAINING][0]
POOL_SAMPLING_SHA256 = SNAPSHOT_FILES[POOL_SAMPLING][0]
RUST_LIBRARY_SHA256, RUST_LIBRARY_BYTES = SNAPSHOT_FILES[RUST_LIBRARY]
TOKENIZER_POOL_SHA256 = WORKER_FILES[TOKENIZER_POOL][0]
INPUTS_MANIFEST_SHA256 = REFERENCE_FILES[INPUTS_MANIFEST][0]

#: The RL trainer variant of this port, as recorded in code_manifest.json, hyak_segment.json and pairing.json.
RL_TRAINER_VARIANT = {
    'name': 'straight_bpb_reward',
    'path': RL_TRAINER, 'sha256': RL_TRAINER_SHA256,
    'base_path': REGRET_TRAINER, 'base_sha256': REGRET_TRAINER_SHA256, 'base_commit': REGRET_PINNED_AT_COMMIT,
    'patch': f'{VARIANT_DIR}/straight_reward.patch',
    'policy_reward': 'paired_pre_model_update_transferred_bpb',
    'decision': 'the user, 2026-11-14: Qingyun\'s joint regret RL20k with the straight BPB loss as the reward',
}
#: Policy settings the RL wrapper passes explicitly (the regret launcher's POLICY with entropy 0:
#: scripts/anytok_pool_rl_config.py, scripts/modal_anytok_pool_rl_20k.py:262-264).
RL_POLICY_ENTROPY = 0.0
RL_POLICY_LR = 0.001
RL_POLICY_BASELINE_DECAY = 0.99

#: rust_utils.__version__ of the committed library (rust_utils/Cargo.toml:3, src/lib.rs:254).
RUST_UTILS_VERSION = '0.14.1-dev.0'
#: rustc recorded in the committed library's .comment section (NOTES/env-monitoring-tests.md 3.2.4).
RUST_TOOLCHAIN = '1.96.0'

# --- environment pins (the Modal training image: scripts/modal_anytok_pool_20k.py:37-42) ------
PYTHON_VERSION = (3, 12)
#: The six exact pins of every training/scoring image, in the image's order.
TRAIN_PINS = (
    ('torch', '2.15.0'), ('transformers', '5.15.1'), ('tokenizers', '0.22.2'),
    ('numpy', '2.3.5'), ('safetensors', '0.8.0'), ('accelerate', '1.14.0'),
)
#: The one pin the raw-BPB handoff adds (docs/raw-bpb-handoff/requirements.txt = TRAIN_PINS + this; tested).
TRAIN_HANDOFF_PINS = (('huggingface_hub', '1.33.0'),)
#: Test runner of the train environment (the version the port's tests were written with).
TRAIN_TEST_PINS = (('pytest', '9.2.0'),)
#: The package set the corpus build was validated with: it reproduces every record of the original corpus,
#: including the exact size of all 46 original files.  The original image (scripts/modal_prepare_corpus.py:63-65)
#: asked for ranges and never recorded what it resolved, so a range is no more faithful than this pin.
#: hyak/env/setup_data_env.sh installs exactly this; hyak.corpus.build compares it before building.
DATA_PINS = (
    ('datasets', '5.2.0'), ('pyarrow', '26.0.1'), ('huggingface-hub', '2.4.0'), ('hf-xet', '1.8.0'),
    ('fsspec', '2026.9.0'), ('requests', '2.34.2'), ('urllib3', '2.8.0'),
)
#: Data environment requirements: the pinned set plus the build tool (hyak/env/setup_data_env.sh).
DATA_REQUIREMENTS = tuple(f'{name}=={version}' for name, version in DATA_PINS) + ('maturin>=1.4,<2.0',)


def tree_layout(run: str) -> dict:
    """``{path inside the staged tree: repo-relative source path}`` of the ``uniform`` or ``rl`` tree."""
    if run not in ('uniform', 'rl'):
        raise ValueError(f"run must be 'uniform' or 'rl', got {run!r}")
    layout = {f'hf_hypernet/{name}': f'{UNIFORM_SNAPSHOT}/hf_hypernet/{name}' for name in HF_HYPERNET_FILES}
    layout[RUST_LIBRARY_REL] = RUST_LIBRARY
    if run == 'uniform':
        layout[TRAINER_REL] = UNIFORM_TRAINER
    else:
        layout[POLICY_REL] = SETTINGS_POLICY
        layout[PAIRED_REWARD_REL] = PAIRED_REWARD
        layout[TRAINER_REL] = RL_TRAINER
    return dict(sorted(layout.items()))


def sha256_path(path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_files(repo, files=None) -> list:
    """Problems (empty list = all good) of repository files against their pins.

    ``files`` is a ``{path: (sha256, bytes)}`` mapping (default: every pinned file).
    Symlinks are rejected: the staged copies must be the pinned bytes, never a link target.
    """
    repo = Path(repo)
    files = PINNED_FILES if files is None else files
    problems = []
    for rel, (sha, size) in sorted(files.items()):
        path = repo / rel
        if path.is_symlink():
            problems.append(f'{rel}: is a symlink; the pinned file must be a regular file')
            continue
        if not path.is_file():
            problems.append(f'{rel}: missing')
            continue
        actual_size = path.stat().st_size
        if actual_size != size:
            problems.append(f'{rel}: {actual_size} bytes, pinned {size}')
            continue
        actual = sha256_path(path)
        if actual != sha:
            problems.append(f'{rel}: sha256 {actual}, pinned {sha}')
    return problems


def unexpected_snapshot_files(repo) -> list:
    """Files in the three runtime snapshot directories that are not pinned (``__pycache__`` is ignored).

    The snapshots must stay byte-identical to the commits above, so an extra file means
    the checkout is not that snapshot.
    """
    repo = Path(repo)
    pinned = set(SNAPSHOT_FILES) | set(RUST_SOURCE_FILES)
    extra = []
    for snapshot in RUNTIME_SNAPSHOTS:
        base = repo / snapshot
        if not base.is_dir():
            continue
        for directory, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != '__pycache__')
            for name in sorted(filenames):
                rel = Path(directory, name).relative_to(repo).as_posix()
                if rel not in pinned:
                    extra.append(rel)
    return extra


def remedy(problems) -> str:
    """One-line remedy for :func:`check_files` problems."""
    paths = sorted({p.split(':', 1)[0] for p in problems})
    variant = [rel for rel in paths if rel in VARIANT_FILES]
    others = [rel for rel in paths if rel not in VARIANT_FILES]
    parts = []
    if others:
        parts.append('restore the pinned files with: git checkout ' + REGRET_PINNED_AT_COMMIT[:12] + ' -- '
                     + ' '.join(others))
    if variant:
        parts.append('restore the RL trainer variant from the checkout that pinned it (it must equal '
                     f'{REGRET_TRAINER} with {VARIANT_DIR}/straight_reward.patch applied): ' + ' '.join(variant))
    return '; '.join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.code.pinned', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='verify repository files against their pins')
    group.add_argument('--layout', choices=('uniform', 'rl'), help='print a staged tree layout as JSON')
    parser.add_argument('--group', choices=sorted(GROUPS), default='all', help='files to check (default all)')
    parser.add_argument('--repo', type=Path, help='repository root (default: two levels above this file)')
    args = parser.parse_args(argv)
    if args.layout:
        print(json.dumps(tree_layout(args.layout), indent=2))
        return 0
    repo = args.repo or Path(__file__).resolve().parents[2]
    problems = check_files(repo, GROUPS[args.group])
    extra = unexpected_snapshot_files(repo) if args.group in ('all', 'snapshot', 'rust-sources') else []
    for line in problems:
        print(f'MISMATCH {line}', file=sys.stderr)
    for rel in extra:
        print(f'UNEXPECTED {rel}: not a pinned file of the snapshot', file=sys.stderr)
    if problems or extra:
        remedies = []
        if problems:
            remedies.append(remedy(problems))
        if extra:
            remedies.append('remove the unexpected files: ' + ' '.join(extra))
        print(f'ERROR: {len(problems) + len(extra)} pinned file problem(s) in {repo}. Remedy: '
              + '; '.join(remedies), file=sys.stderr)
        return 1
    print(f'OK {len(GROUPS[args.group])} pinned files ({args.group}) in {repo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
