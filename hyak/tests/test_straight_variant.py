"""The port's RL trainer is Qingyun's joint regret RL20k trainer with the straight-BPB reward, nothing else.

``hyak/variants/anytok_pool_straight_rl_20k/straight_reward.patch`` applied to the pinned regret trainer
must reproduce the variant byte for byte, and the two files must differ in exactly the two approved lines
(the policy update and the ``policy_reward`` text of run_config.json).
"""
from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from hyak.code import pinned

VARIANT = Path(pinned.VARIANT_DIR)
PATCH = VARIANT / 'straight_reward.patch'
#: The only lines that may differ (base -> variant).
APPROVED = [
    ('              "policy_reward": "paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb",',
     '              "policy_reward": "paired_pre_model_update_transferred_bpb",'),
    ('        policy_metrics = bandit.update(pool_index, reward_bpb)',
     '        policy_metrics = bandit.update(pool_index, train_bpb)'),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hunks(patch_text: str):
    """Parse a unified diff of one file into ``[(old_start, old_lines, new_lines)]`` (old_start 1-based)."""
    hunks, current = [], None
    for line in patch_text.splitlines(keepends=True):
        if line.startswith(('--- ', '+++ ')):
            continue
        header = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if header:
            current = (int(header.group(1)), [], [])
            hunks.append(current)
            continue
        if current is None or line.startswith('\\'):
            continue
        tag, body = line[0], line[1:]
        if tag in (' ', '-'):
            current[1].append(body)
        if tag in (' ', '+'):
            current[2].append(body)
    return hunks


def apply_unified_diff(base: bytes, patch_text: str) -> bytes:
    """Pure-Python application of a single-file unified diff (strict: every context line must match)."""
    lines = base.decode('utf-8').splitlines(keepends=True)
    out, position = [], 0
    for start, old, new in _hunks(patch_text):
        start -= 1
        assert start >= position, 'overlapping hunks'
        assert lines[start:start + len(old)] == old, f'hunk at line {start + 1} does not match the base'
        out += lines[position:start] + new
        position = start + len(old)
    out += lines[position:]
    return ''.join(out).encode('utf-8')


def test_base_and_variant_are_the_pinned_files(repo_root):
    assert pinned.check_files(repo_root, {pinned.REGRET_TRAINER: pinned.SNAPSHOT_FILES[pinned.REGRET_TRAINER],
                                          pinned.RL_TRAINER: pinned.VARIANT_FILES[pinned.RL_TRAINER]}) == []
    assert _sha(repo_root / pinned.REGRET_TRAINER) == pinned.RL_TRAINER_VARIANT['base_sha256']
    assert _sha(repo_root / pinned.RL_TRAINER) == pinned.RL_TRAINER_VARIANT['sha256']
    readme = (repo_root / VARIANT / 'README.md').read_text(encoding='utf-8')
    for value in (pinned.REGRET_TRAINER_SHA256, pinned.RL_TRAINER_SHA256, pinned.PAIRED_REWARD_SHA256,
                  pinned.REGRET_PINNED_AT_COMMIT, '2026-11-14'):
        assert value in readme


def test_patch_applied_with_difflib_parser_reproduces_the_variant(repo_root):
    base = (repo_root / pinned.REGRET_TRAINER).read_bytes()
    patched = apply_unified_diff(base, (repo_root / PATCH).read_text(encoding='utf-8'))
    assert patched == (repo_root / pinned.RL_TRAINER).read_bytes()
    assert hashlib.sha256(patched).hexdigest() == pinned.RL_TRAINER_SHA256


@pytest.mark.skipif(shutil.which('patch') is None, reason='patch(1) is not installed')
def test_patch_applied_with_patch1_reproduces_the_variant(repo_root, tmp_path):
    work = tmp_path / 'train_joint_smollm_input.py'
    shutil.copyfile(repo_root / pinned.REGRET_TRAINER, work)
    result = subprocess.run(['patch', '--quiet', '--forward', '--no-backup-if-mismatch', str(work),
                             str(repo_root / PATCH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert work.read_bytes() == (repo_root / pinned.RL_TRAINER).read_bytes()
    assert not list(tmp_path.glob('*.rej')) and not list(tmp_path.glob('*.orig'))


def test_the_patch_is_the_exact_diff_and_changes_only_the_approved_lines(repo_root):
    base = (repo_root / pinned.REGRET_TRAINER).read_text(encoding='utf-8').splitlines(keepends=True)
    variant = (repo_root / pinned.RL_TRAINER).read_text(encoding='utf-8').splitlines(keepends=True)
    assert len(base) == len(variant)
    changed = [(a.rstrip('\n'), b.rstrip('\n')) for a, b in zip(base, variant) if a != b]
    assert changed == APPROVED
    expected = ''.join(difflib.unified_diff(base, variant, fromfile=f'a/{pinned.REGRET_TRAINER}',
                                            tofile=f'b/{pinned.RL_TRAINER}', n=3))
    assert (repo_root / PATCH).read_text(encoding='utf-8') == expected
    removed = [line[1:].rstrip('\n') for line in expected.splitlines(keepends=True)
               if line.startswith('-') and not line.startswith('---')]
    added = [line[1:].rstrip('\n') for line in expected.splitlines(keepends=True)
             if line.startswith('+') and not line.startswith('+++')]
    assert removed == [a for a, _ in APPROVED] and added == [b for _, b in APPROVED]


def test_the_variant_keeps_the_paired_protocol(repo_root):
    text = (repo_root / pinned.RL_TRAINER).read_text(encoding='utf-8')
    # the canonical control forward, the paired checks and all paired logging stay
    assert text.count('paired = make_paired_prefix_example(raw[:512], target_tokenizer, source_tokenizer, '
                      'min_chars=32)') == 2                     # the first try and the retry of a rejected example
    for needle in ('canonical_nats = canonical_loss_nats(lm, canonical_examples, source,',
                   "raise ValueError('Canonical and native targets or continuation bytes differ')",
                   '"paired_gap_bpb": reward_bpb,', '"paired_canonical_bpb": canonical_bpb,',
                   '"paired_valid_examples": len(pairing_evidence),', '"paired_maximum_positions":',
                   '"paired_batch_sha256":', '"pairing_protocol": "nearest_valid_canonical_midpoint_boundary_joint_128_positions_v1",',
                   'parser.add_argument("--policy-entropy", type=float, default=0.0)',
                   'args.seed+1, args.policy_lr,'):
        assert text.count(needle) == 1, needle
    assert text.count('bandit.update(') == 1 and 'bandit.update(pool_index, train_bpb)' in text
