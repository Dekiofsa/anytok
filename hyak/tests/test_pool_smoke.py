"""Slow local smoke: the real patched wheel and the unchanged scripts/tokenizer_pool.py through hyak.pool.worker.

Needs ``<scratch>/venv-pool`` with the wheel installed by ``build_wheel.sh install`` and
``<scratch>/wheel-build/wheels/provenance.json`` (skipped otherwise).  Three tiny tokenizers
(noisy BPE/regex, WordPiece/line256, Unigram/no-regex) over a synthetic 23-bucket corpus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from hyak.pool import load_plan, tokenizer_json_problems
from hyak.pool.collect import verify_manifests
from hyak.tests.test_pool_support import make_corpus, make_test_plan

REPO = Path(__file__).resolve().parents[2]
FAMILIES = [('bpe', 'regex', 750000), ('wordpiece', 'line256', None), ('unigram', 'no-regex', None)]


@pytest.mark.slow
def test_real_wheel_and_worker_on_three_families(tmp_config, scratch_dir):
    python = scratch_dir / 'venv-pool' / 'bin' / 'python'
    provenance = scratch_dir / 'wheel-build' / 'wheels' / 'provenance.json'
    if not python.exists() or not provenance.exists():
        pytest.skip('local patched wheel not built/installed (run hyak/pool/build_wheel.sh fetch build install)')
    families = iter(FAMILIES)

    def mutate(config):
        algorithm, pretokenizer, noise = next(families)
        config.update(algorithm=algorithm, pretokenizer=pretokenizer, noise_per_million=noise)

    base = tmp_config.base
    plan_path = base / 'plan.json'
    plan = make_test_plan(plan_path, count=3, total_bytes=46_000, vocab=512, mutate=mutate)
    recipe = make_corpus(base / 'corpus', plan)
    pool_dir = base / 'smoke-pool'
    args = [str(python), '-m', 'hyak.pool.worker', '--plan', str(plan_path), '--allow-test-plan',
            '--pool-dir', str(pool_dir), '--recipe', str(recipe), '--wheel-provenance', str(provenance),
            '--monitor-dir', str(base / 'monitor'), '--workers', '2', '--threads', '4', '--poll-seconds', '0.1']
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run(args, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    loaded, _ = load_plan(plan_path, allow_test_plan=True)
    manifests, problems, summary = verify_manifests(loaded, pool_dir / 'tokenizers', require_production_identity=False)
    assert problems == {} and summary['verified'] == 3 and summary['identities'] == 1
    wheel = json.loads(provenance.read_text())['wheel']['sha256']
    for index, manifest in manifests.items():
        assert manifest['provenance']['runtime_wheel_sha256'] == wheel
        assert manifest['vocab_size_actual'] == 512 and len(manifest['validation']) >= 2
        raw = (pool_dir / 'tokenizers' / f'{index:04d}' / 'tokenizer.json').read_bytes()
        assert tokenizer_json_problems(raw, manifest['config'], 512) == []
    noise = json.loads((pool_dir / 'tokenizers' / '0000' / 'tokenizer.json').read_text())
    assert noise['model']['type'] == 'BPE'
    again = subprocess.run(args, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert again.returncode == 0 and '"tokenizer_start"' not in again.stdout
