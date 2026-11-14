"""Full-size build of the random-init hypernetwork through the real command line (slow, ~3.5 GB RAM).

Uses the scratch assets: ``full/{config.json,tokenizer.json}`` (pinned SmolLM2 files) and
``embed_only/model.safetensors`` (only ``model.embed_tokens.weight``, BF16, same key as the real
file, hence ``--allow-unpinned-model``).  The build runs in a subprocess, so this pytest process stays small; the 1.4 GB output is
deleted afterwards.  Skip with HYAK_TEST_SKIP_SLOW=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from hyak.model import init_hypernet as ih

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / 'runtime' / 'anytok_pool_20k'


def _run(*args, timeout=1800):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-m', 'hyak.model.init_hypernet', *map(str, args)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def fullsize_model(tmp_path, scratch_assets):
    sources = {'config.json': scratch_assets / 'full' / 'config.json',
               'tokenizer.json': scratch_assets / 'full' / 'tokenizer.json',
               'model.safetensors': scratch_assets / 'embed_only' / 'model.safetensors'}
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        pytest.skip(f'scratch assets missing: {missing}')
    model = tmp_path / 'model'
    model.mkdir()
    for name, source in sources.items():
        (model / name).symlink_to(source)
    yield model
    for path in tmp_path.iterdir():          # never leave 1.4 GB in pytest's temporary directories
        if path.is_dir() and not path.is_symlink() and path.name != 'model':
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.slow
def test_fullsize_build_through_the_cli(fullsize_model, tmp_path):
    out = tmp_path / 'input' / 'hypernet_init'
    refused = _run('--model-dir', fullsize_model, '--code-dir', CODE, '--out', out, timeout=600)
    assert refused.returncode == 2 and 'Remedy: python -m hyak.model.fetch_model' in refused.stderr   # init-4
    assert not out.exists()
    result = _run('--model-dir', fullsize_model, '--code-dir', CODE, '--out', out, '--allow-unpinned-model')
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert 'recipe (b) calibrated with the zero bias, then bias drawn (default), seed 42' in result.stdout
    assert 'INIT OK' in result.stdout and 'differs from plans/zett-modal-20k-inputs.json' in result.stdout
    assert sorted(p.name for p in out.iterdir()) == ['config.json', 'init_manifest.json', 'model.safetensors']
    assert not [p for p in out.parent.iterdir() if p.name.startswith('.')]

    # exact format of the original warmup directory
    assert (out / 'model.safetensors').stat().st_size == ih.EXPECTED_MODEL_BYTES == 1_376_746_800
    config = json.loads((out / 'config.json').read_text())
    assert config['original_vocab_size'] == config['pad_token_id'] == 49152
    assert config['hn_n_extra_tokens'] == 1 and config['hn_byte_fallback_symbols'] is None
    assert config['hn_surface_maxlen'] == 10 and config['hn_backbone_config']['num_attention_heads'] == 32

    # in_scaler equals the original checkpoint's, recomputed independently from the file
    from safetensors import safe_open
    reference = ih.load_reference()
    with safe_open(str(out / 'model.safetensors'), framework='np') as handle:
        w, b = handle.get_tensor('in_scaler.w'), handle.get_tensor('in_scaler.b')
        names = list(handle.keys())
        finite = all(np.isfinite(handle.get_tensor(n)).all() for n in names)
    assert len(names) == 138 and finite
    assert float(np.abs(w.astype(np.float64) - reference['w']).max()) <= 1e-6
    assert float(np.abs(b.astype(np.float64) - reference['b']).max()) <= 1e-6

    manifest = json.loads((out / 'init_manifest.json').read_text())
    checks = manifest['checks']
    assert checks['in_scaler_regression']['passed'] and checks['in_scaler_regression']['max_abs_diff_w'] <= 1e-6
    assert checks['parameters']['total'] == 344_182_784 and checks['parameters']['frozen'] == 8_192
    assert checks['files']['model_bytes'] == checks['files']['predicted_bytes'] == 1_376_746_800
    assert checks['config']['reference_differences'] == []
    assert checks['reload']['tensors_bit_identical'] and checks['reload']['passed']
    trainer = checks['trainer_emulation']
    assert trainer['passed'] and len(trainer['fallback_symbols']) == 21 and trainer['fallback_rows'] == 22
    assert trainer['config_identical_to_original'] is True
    assert trainer['checkpoint_config'] == {'bytes': 2382, 'sha256': ih.ORIGINAL_TRAINED_CONFIG_SHA256}
    assert trainer['checkpoint_model_bytes'] == 1_376_918_848
    assert trainer['resume']['passed']
    diagnostic = manifest['prior_gradient_diagnostic']
    assert diagnostic['degenerate'] is False and diagnostic['ratio'] < 1
    # default (b): seed 42 with the zero library bias reproduces the original's rescalers (checked to 1e-5),
    # then the four frozen tensors are copied from the public checkpoint: bit-identical to the original's
    original = checks['original_construction']
    assert original['applicable'] and original['matches'] and original['passed'], original
    assert original['copied_from_fixture'] and original['fixture_sha256'] == reference['sha256']
    assert max(original['zero_bias_max_abs_diff_w'], original['zero_bias_max_abs_diff_b']) <= 1e-5
    assert original['final_scaler_max_abs_diff_w'] == original['final_scaler_max_abs_diff_b'] == 0.0
    with safe_open(str(out / 'model.safetensors'), framework='np') as handle:
        for name, key in (('in_scaler.w', 'w'), ('in_scaler.b', 'b'), ('scaler.w', 'scaler_w'), ('scaler.b', 'scaler_b')):
            assert np.array_equal(handle.get_tensor(name), reference[key]), name
    assert checks['calibration']['passed'] and 'zero-bias network' in checks['calibration']['scope']
    assert 'passed' not in checks['anchors_after_bias']
    assert manifest['recipe']['seed'] == 42 and manifest['recipe']['prior_bias_std'] == 0.02
    assert manifest['recipe']['calibration'] == 'before_bias'
    assert manifest['recipe']['rescalers_from_fixture'] == {'file': 'original_in_scaler.npz', 'sha256': reference['sha256']}
    assert manifest['original_warmup']['config_equal_without_fallback_key'] is True     # init-9
    assert manifest['inputs']['allow_unpinned_model'] is True
    assert manifest['inputs']['table']['stored_dtype'] == 'BF16'
    assert manifest['inputs']['files']['tokenizer.json']['matches_pinned'] is True
    assert manifest['outputs']['model.safetensors']['bytes'] == 1_376_746_800
    bias = manifest['tensors'][ih.PRIOR_BIAS_NAME]
    assert 0.018 < bias['std'] < 0.022
    print(json.dumps({'model_sha256': manifest['outputs']['model.safetensors']['sha256'],
                      'in_scaler': {k: checks['in_scaler_regression'][k] for k in ('max_abs_diff_w', 'max_abs_diff_b')},
                      'scaler_w_mean': manifest['tensors']['scaler.w']['mean'],
                      'ratio': diagnostic['ratio'], 'seconds': manifest['seconds'],
                      'peak_rss_bytes': manifest['peak_rss_bytes']}))

    # verify and idempotent re-run
    verified = _run('--verify', '--out', out, timeout=600)
    assert verified.returncode == 0 and 'VERIFIED' in verified.stdout, verified.stderr
    again = _run('--model-dir', fullsize_model, '--code-dir', CODE, '--out', out, '--allow-unpinned-model', timeout=600)
    assert again.returncode == 0 and '(already built)' in again.stdout, again.stderr
    other = _run('--model-dir', fullsize_model, '--code-dir', CODE, '--out', out, '--seed', '7',
                 '--allow-unpinned-model', timeout=600)
    assert other.returncode == 2 and 'pass --force' in other.stderr
    shutil.rmtree(out, ignore_errors=True)

    # option (a), --calibrate-after-bias: calibrated with the drawn bias, so the frozen output scaler differs
    after_out = tmp_path / 'input' / 'hypernet_init_a'
    after = _run('--model-dir', fullsize_model, '--code-dir', CODE, '--out', after_out, '--allow-unpinned-model',
                 '--calibrate-after-bias')
    assert after.returncode == 0, after.stdout[-3000:] + after.stderr[-3000:]
    manifest_a = json.loads((after_out / 'init_manifest.json').read_text())
    original_a = manifest_a['checks']['original_construction']
    assert manifest_a['recipe']['calibration'] == 'after_bias' and manifest_a['recipe']['rescalers_from_fixture'] is None
    assert original_a['applicable'] and original_a['matches'] and 'passed' not in original_a
    assert original_a['final_scaler_w_mean'] > original_a['reference_scaler_w_mean'] + 0.02
    assert manifest_a['checks']['calibration']['passed']
    assert manifest_a['tensors']['in_scaler.w']['sha256'] == manifest['tensors']['in_scaler.w']['sha256'] or \
        manifest_a['checks']['in_scaler_regression']['max_abs_diff_w'] <= 1e-6
    differing = {n for n in manifest['tensors'] if manifest['tensors'][n]['sha256'] != manifest_a['tensors'][n]['sha256']}
    assert {'scaler.w', 'scaler.b'} <= differing <= {'scaler.w', 'scaler.b', 'in_scaler.w', 'in_scaler.b'}
