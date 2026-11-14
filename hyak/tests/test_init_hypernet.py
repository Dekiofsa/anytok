"""Small-size tests of hyak.model.init_hypernet (DESIGN.md D1, 5.B).

Hypernetworks here have width 64 (one attention head); the pinned hf_hypernet code is imported
from runtime/anytok_pool_20k (no Rust library needed).  The end-to-end tests use the real
SmolLM2 tokenizer from the scratch assets (49,152 tokens, 21 byte fallbacks) with a synthetic
[49152, 64] input table, and skip when the assets are absent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from hyak.model import init_hypernet as ih

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / 'runtime' / 'anytok_pool_20k'
SMALL = ih.Profile(width=64, expected_model_bytes=None, expected_trained_model_bytes=None,
                   expected_parameters=None, expected_frozen=4 * 64, reference=False)


@pytest.fixture(scope='module')
def hf():
    return ih.import_hf_hypernet(CODE)


def _table(vocab=300, width=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    # per-dimension scales and offsets differ, like a real embedding table
    scale = torch.rand(width, generator=generator) * 0.1 + 0.01
    shift = torch.randn(width, generator=generator) * 0.01
    return (torch.randn(vocab, width, generator=generator) * scale + shift).to(torch.bfloat16).float()


def _digests(hn):
    return {name: ih._tensor_sha256(t) for name, t in hn.state_dict().items()}


def _differing(a, b):
    return sorted(name for name in a if a[name] != b[name])


# --- code, environment, reference --------------------------------------------------------------------

def test_pinned_sources_match_repository_snapshot():
    info = ih.verify_code_dir(CODE)
    assert info['files'] == ih.PINNED_SOURCES
    for rel, digest in ih.PINNED_SOURCES.items():   # the RL tree reuses the uniform hf_hypernet
        assert hashlib.sha256((CODE / rel).read_bytes()).hexdigest() == digest


def test_verify_code_dir_detects_tampering_and_missing_files(tmp_path):
    copy = tmp_path / 'code'
    shutil.copytree(CODE / 'hf_hypernet', copy / 'hf_hypernet')
    ih.verify_code_dir(copy)
    target = copy / 'hf_hypernet' / 'hyper_roberta.py'
    target.write_bytes(target.read_bytes() + b'\n')
    with pytest.raises(ih.InitError, match='hyper_roberta.py has SHA-256'):
        ih.verify_code_dir(copy)
    (copy / 'hf_hypernet' / 'identity_warmup.py').unlink()
    with pytest.raises(ih.InitError, match='identity_warmup.py is missing'):
        ih.verify_code_dir(copy)


def test_import_refuses_a_second_location(hf, tmp_path):
    copy = tmp_path / 'code'
    shutil.copytree(CODE / 'hf_hypernet', copy / 'hf_hypernet')
    with pytest.raises(ih.InitError, match='already imported'):
        ih.import_hf_hypernet(copy)


def test_import_never_writes_bytecode_into_the_code_tree(tmp_path):
    copy = tmp_path / 'code'
    shutil.copytree(CODE / 'hf_hypernet', copy / 'hf_hypernet')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONDONTWRITEBYTECODE'}
    script = ('import sys; from hyak.model import init_hypernet as ih; hf = ih.import_hf_hypernet(sys.argv[1]); '
              'print(hf.ZettHypernet.__module__)')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(copy)], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'hf_hypernet.modeling_hypernet'
    assert not list(copy.rglob('__pycache__')), 'bytecode was written into the code tree'


def test_environment_pins(monkeypatch):
    info = ih.check_environment(strict=True)
    assert info['pins_ok'] and info['packages']['transformers'] == '5.15.1'
    monkeypatch.setattr(torch, '__version__', '2.15.0+cu132')
    assert ih.check_environment(strict=True)['pins_ok']
    monkeypatch.setattr(torch, '__version__', '2.14.1')
    with pytest.raises(ih.InitError, match='torch 2.14.1'):
        ih.check_environment(strict=True)
    assert ih.check_environment(strict=False)['pins_ok'] is False


def test_reference_fixture_provenance_and_original_config_hash():
    ref = ih.load_reference()
    assert (ref['repository'], ref['model_file'], ref['tensor_w'], ref['tensor_b'], ref['step']) == (
        'qingy2024/anytok-zett-smollm2-20k', 'checkpoints/step-000500/hypernet/model.safetensors',
        'in_scaler.w', 'in_scaler.b', 500)
    assert (ref['tensor_scaler_w'], ref['tensor_scaler_b']) == ('scaler.w', 'scaler.b')
    assert ref['w'].shape == ref['b'].shape == ref['scaler_w'].shape == ref['scaler_b'].shape == (1, 2048)
    assert ref['w'].dtype == ref['scaler_w'].dtype == np.float32
    assert abs(float(ref['w'].mean()) - 0.2975) < 1e-4          # DESIGN.md 1.2
    assert abs(float(ref['scaler_w'].mean()) - 0.182) < 1e-3    # DESIGN.md 1.2
    assert len(ref['config_text'].encode()) == ih.ORIGINAL_TRAINED_CONFIG_BYTES
    assert len(ref['config']['hn_byte_fallback_symbols']) == 21 and ref['config']['hn_n_extra_tokens'] == 22
    inventory = (REPO / 'plans' / 'uniform-trajectory-inventory.json').read_text()
    assert f'"sha256": "{ih.ORIGINAL_TRAINED_CONFIG_SHA256}"' in inventory
    inputs = json.loads((REPO / 'plans' / 'zett-modal-20k-inputs.json').read_text())['files']
    assert inputs['warmup/model.safetensors']['bytes'] == ih.EXPECTED_MODEL_BYTES
    assert inputs['warmup/config.json']['sha256'] == ih.ORIGINAL_WARMUP_CONFIG_SHA256


def test_reference_fixture_equals_the_extracted_public_arrays(scratch_dir):
    public = scratch_dir / 'public-zett'
    if not (public / 'in_scaler_w.npy').is_file():
        pytest.skip(f'public checkpoint extracts not found in {public}')
    ref = ih.load_reference()
    assert np.array_equal(ref['w'][0], np.load(public / 'in_scaler_w.npy'))
    assert np.array_equal(ref['b'][0], np.load(public / 'in_scaler_b.npy'))
    assert np.array_equal(ref['scaler_w'][0], np.load(public / 'scaler_w.npy'))
    assert np.array_equal(ref['scaler_b'][0], np.load(public / 'scaler_b.npy'))
    assert ref['config_text'].encode() == (public / 'hn_config_step500.json').read_bytes()


def test_reference_fixture_tampering_is_detected(tmp_path):
    with np.load(ih.REFERENCE_NPZ, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    values['w'] = values['w'].copy()
    values['w'][0, 7] += 1e-3
    bad = tmp_path / 'bad.npz'
    with open(bad, 'wb') as stream:
        np.savez(stream, **values)
    with pytest.raises(ih.InitError, match='w values differ'):
        ih.load_reference(bad)


def test_expected_config_equals_the_original_config_without_fallbacks():
    ref = ih.load_reference()
    expected = ih.expected_init_config(ih.SOURCE_VOCAB, ih.EMBED_WIDTH)
    assert ih.config_differences(ih.reference_init_config(ref), expected) == []
    # and the trainer-side values of the original are the expanded ones
    assert ih.config_differences(ref['config'], expected) == [
        "hn_n_extra_tokens: 22 != expected 1",
        f"hn_byte_fallback_symbols: {ref['config']['hn_byte_fallback_symbols']!r} != expected None"]


def test_config_differences_are_strict():
    assert ih.config_differences({'a': 1}, {'a': True}) == ['a: 1 != expected True']
    assert ih.config_differences({'a': 1.0}, {'a': 1}) == ['a: 1.0 != expected 1']
    assert ih.config_differences({'a': {'b': 2, 'c': 3}}, {'a': {'b': 2}}, exact=True) == [
        'a.c: unexpected key (value 3)']
    assert ih.config_differences({}, {'x': None}) == ['x: missing (expected None)']


def test_input_lexical_loss_copy_equals_the_pinned_function(hf, monkeypatch):
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    # rust_utils/ (without the registered .so) imports as an empty namespace package here
    from hf_hypernet.joint_training import input_lexical_loss as pinned
    generator = torch.Generator().manual_seed(3)
    source = torch.randn(50, 16, generator=generator)
    forms = torch.randint(0, 50, (40, 10), generator=generator)
    forms[:, 3:] = 50
    forms[:12, 1:] = 50
    forms[0, 0] = 0
    generated = torch.randn(40, 16, generator=generator)
    assert torch.equal(ih.input_lexical_loss(generated, forms, source, 50, 0), pinned(generated, forms, source, 50, 0))
    with pytest.raises(ValueError, match='No original tokens overlap'):
        ih.input_lexical_loss(generated, torch.full((4, 10), 50), source, 50, 0)


# --- the recipe ---------------------------------------------------------------------------------------

def test_recipe_order_matches_d1(hf):
    table = _table()
    # default, option (b): calibrate the zero-bias construction, then draw the bias
    hn, info = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02)
    torch.manual_seed(42)
    manual = hf.build_input_hypernet(300, 64)
    hf.initialize_rescalers(manual, table)
    with torch.no_grad():
        torch.nn.init.normal_(manual.model.target_priors_projection.bias, 0.0, 0.02,
                              generator=torch.Generator().manual_seed(42 + 1000))
    assert _digests(hn) == _digests(manual)
    assert info['prior_bias']['generator_seed'] == 1042 and info['rescaler_init'] is True
    assert info['construction_init']['passed'] and info['calibration_order'] == ih.CALIBRATE_BEFORE_BIAS
    assert info['calibration']['passed'] and 'zero-bias network' in info['calibration']['scope']
    # option (a), --calibrate-after-bias: draw the bias, then calibrate; only the output scaler differs
    after, after_info = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, calibration=ih.CALIBRATE_AFTER_BIAS)
    torch.manual_seed(42)
    manual = hf.build_input_hypernet(300, 64)
    with torch.no_grad():
        torch.nn.init.normal_(manual.model.target_priors_projection.bias, 0.0, 0.02,
                              generator=torch.Generator().manual_seed(42 + 1000))
    hf.initialize_rescalers(manual, table)
    assert _digests(after) == _digests(manual) and 'calibration' not in after_info
    assert _differing(_digests(hn), _digests(after)) == ['scaler.b', 'scaler.w']
    # saved before byte fallbacks
    assert hn.config.hn_byte_fallback_symbols is None and hn.fallback_embeddings.num_embeddings == 1


def test_prior_bias_option_changes_only_the_bias(hf):
    table = _table()
    states = {}
    for std in (0.0, 0.02, 0.05):
        torch.manual_seed(1234)
        hn, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=std)
        states[std] = (hn, _digests(hn), torch.get_rng_state())
    for std in (0.02, 0.05):
        # default (b): the frozen rescalers are those of the zero-bias construction whatever the std
        assert _differing(states[0.0][1], states[std][1]) == [ih.PRIOR_BIAS_NAME]
        assert torch.equal(states[0.0][2], states[std][2]), 'the separate generator touched the global RNG'
        after, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=std, calibration=ih.CALIBRATE_AFTER_BIAS)
        assert _differing(states[0.0][1], _digests(after)) == [ih.PRIOR_BIAS_NAME, 'scaler.b', 'scaler.w']
    zero = states[0.0][0].model.target_priors_projection.bias.detach()
    assert bool((zero == 0).all())
    small = states[0.02][0].model.target_priors_projection.bias.detach()
    large = states[0.05][0].model.target_priors_projection.bias.detach()
    assert torch.allclose(large, small * 2.5, rtol=1e-5, atol=0)          # same draw, scaled
    assert 0.01 < float(small.std()) < 0.03
    assert torch.equal(states[0.0][0].in_scaler.w, states[0.05][0].in_scaler.w)


def _fake_reference(zero_bias_hn, sha='f' * 64):
    """A stand-in for original_in_scaler.npz built from a zero-bias construction (small width)."""
    return {'w': zero_bias_hn.in_scaler.w.detach().numpy().copy(), 'b': zero_bias_hn.in_scaler.b.detach().numpy().copy(),
            'scaler_w': zero_bias_hn.scaler.w.detach().numpy().copy(),
            'scaler_b': zero_bias_hn.scaler.b.detach().numpy().copy(), 'repository': 'test',
            'model_file': 'zero-bias build', 'tensor_w': 'in_scaler.w', 'tensor_b': 'in_scaler.b', 'step': 0,
            'sha256': sha}


def test_original_construction_probe_is_exact_and_leaves_the_after_bias_recipe_unchanged(hf):
    table = _table()
    zero, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.0)
    fake = _fake_reference(zero)
    after = {'calibration': ih.CALIBRATE_AFTER_BIAS}
    plain, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, **after)
    probed, info = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, reference=fake, **after)
    assert _digests(probed) == _digests(plain), 'the probe changed the (a) result'
    original = info['original_construction']
    assert original['applicable'] and original['matches'] and 'passed' not in original   # informative in (a)
    assert original['zero_bias_max_abs_diff_w'] == 0.0 and original['zero_bias_max_abs_diff_b'] == 0.0
    assert original['final_scaler_max_abs_diff_w'] > 0       # (a) calibrates with the drawn bias
    assert ih.build_hypernet(hf, table, seed=43, reference=fake)[1]['original_construction'] == {
        'applicable': False, 'reason': 'seed 43 (the original warmup started from seed 42)'}
    assert not ih.build_hypernet(hf, table, rescaler_init=False, reference=fake)[1]['original_construction']['applicable']
    assert not ih.build_hypernet(hf, _table(width=32), reference=fake)[1]['original_construction']['applicable']
    assert not ih.build_hypernet(hf, table)[1]['original_construction']['applicable']


def test_default_recipe_checks_and_copies_the_original_rescalers(hf):
    """(b): the computed rescalers must match the fixture (1e-5); then the fixture values are copied bit for bit."""
    table = _table()
    zero, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.0)
    fake = _fake_reference(zero)
    nudged = {key: (value + np.float32(3e-6) if key in ('w', 'b', 'scaler_w', 'scaler_b') else value)
              for key, value in fake.items()}                       # within the tolerance
    hn, info = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, reference=nudged)
    original = info['original_construction']
    assert original['enforced'] and original['passed'] and original['copied_from_fixture']
    assert original['fixture_sha256'] == 'f' * 64 and 0 < original['zero_bias_max_abs_diff_w'] <= 1e-5
    for name, key in (('in_scaler', 'w'), ('in_scaler', 'b'), ('scaler', 'w'), ('scaler', 'b')):
        fixture_key = key if name == 'in_scaler' else f'scaler_{key}'
        assert np.array_equal(getattr(getattr(hn, name), key).detach().numpy(), nudged[fixture_key]), (name, key)
    assert info['in_scaler_regression']['passed'] and info['in_scaler_regression']['tolerance'] == 1e-5
    plain, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02)
    assert _differing(_digests(plain), _digests(hn)) == sorted(ih.RESCALER_NAMES)   # nothing else changes
    wrong = dict(fake, scaler_w=fake['scaler_w'] * np.float32(1.01))
    with pytest.raises(ih.InitError, match='do not reproduce the original construction'):
        ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, reference=wrong)
    wrong = dict(fake, w=fake['w'] + np.float32(1e-3))
    with pytest.raises(ih.InitError, match='do not reproduce the original construction'):
        ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, reference=wrong)
    # option (c), --prior-bias-std 0: the same checked, copied rescalers and a zero bias
    zero_c, info_c = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.0, reference=nudged)
    assert info_c['original_construction']['copied_from_fixture']
    assert bool((zero_c.model.target_priors_projection.bias == 0).all())


def test_no_rescaler_init_leaves_identity_rescalers(hf):
    table = _table()
    hn, info = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02, rescaler_init=False)
    ref, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02)
    for name in ih.RESCALER_NAMES:
        values = hn.state_dict()[name]
        assert bool((values == (1.0 if name.endswith('.w') else 0.0)).all()), name
    assert _differing(_digests(hn), _digests(ref)) == sorted(ih.RESCALER_NAMES)
    assert info['rescaler_init'] is False and info['anchor_count'] is None


def test_rescalers_follow_scale_to_and_calibrate_the_anchors(hf):
    table = _table()
    w, b = hf.Rescaler.scale_to(table, target_stds=torch.full_like(table[0], 0.02),
                                target_means=torch.zeros_like(table[0]))
    hn, info = ih.build_hypernet(hf, table, seed=42)
    assert torch.equal(hn.in_scaler.w, w) and torch.equal(hn.in_scaler.b, b)
    assert info['calibration']['passed'] and info['calibration']['anchors'] == 300
    # (b): the final network (bias drawn after calibration) is not calibrated on the anchors
    assert not ih.calibration_report(ih.anchor_outputs(hn, table), table)['passed']
    after, _ = ih.build_hypernet(hf, table, seed=42, calibration=ih.CALIBRATE_AFTER_BIAS)
    report = ih.calibration_report(ih.anchor_outputs(after, table), table)
    assert report['passed'] and report['anchors'] == 300, report
    assert 'one-piece forms with zero priors' in report['scope']


def test_determinism_same_seed_bit_identical_other_seed_differs(hf):
    table = _table()
    first, _ = ih.build_hypernet(hf, table, seed=42)
    second, _ = ih.build_hypernet(hf, table, seed=42)
    assert _digests(first) == _digests(second)
    other, _ = ih.build_hypernet(hf, table, seed=43)
    differing = _differing(_digests(first), _digests(other))
    assert 'model.layers.0.attention.query.weight' in differing and ih.PRIOR_BIAS_NAME in differing
    assert 'in_scaler.w' not in differing and 'in_scaler.b' not in differing   # a function of the table only


def test_recipe_argument_validation(hf):
    table = _table()
    for kwargs in ({'seed': -1}, {'seed': True}, {'prior_bias_std': -0.1}, {'prior_bias_std': float('nan')},
                   {'calibration': 'sideways'}, {'calibration': None},
                   {'calibration': ih.CALIBRATE_AFTER_BIAS, 'prior_bias_std': 0.0},
                   {'calibration': ih.CALIBRATE_AFTER_BIAS, 'rescaler_init': False}):
        with pytest.raises(ih.InitUsageError):
            ih.build_hypernet(hf, table, **kwargs)
    with pytest.raises(ih.InitError, match='CPU float32'):
        ih.build_hypernet(hf, table.double())


def test_construction_check_rejects_other_initializations(hf):
    torch.manual_seed(0)
    hn = hf.build_input_hypernet(300, 64)
    assert ih.check_construction_init(hn)['passed']
    hn.input_projection[0].reset_parameters()            # PyTorch default: uniform weight, non-zero bias
    with pytest.raises(ih.InitError, match='input_projection.0'):
        ih.check_construction_init(hn)
    torch.manual_seed(0)
    hn = hf.build_input_hypernet(300, 64)
    torch.nn.init.normal_(hn.model.position_embeddings.weight, std=1.0)   # nn.Embedding default
    with pytest.raises(ih.InitError, match='position_embeddings'):
        ih.check_construction_init(hn)


def test_gradient_diagnostic_flags_the_zero_bias_case(hf):
    table = _table(vocab=1000)
    zero, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.0)
    drawn, _ = ih.build_hypernet(hf, table, seed=42, prior_bias_std=0.02)
    hf.freeze_rescalers(zero)
    hf.freeze_rescalers(drawn)
    state = torch.get_rng_state()
    bad = ih.prior_gradient_diagnostic(zero, table, seed=7, vocab=96, one_piece=24)
    assert torch.equal(state, torch.get_rng_state())
    good = ih.prior_gradient_diagnostic(drawn, table, seed=7, vocab=96, one_piece=24)
    again = ih.prior_gradient_diagnostic(drawn, table, seed=7, vocab=96, one_piece=24)
    assert bad['degenerate'] and bad['ratio'] > 100, bad
    assert not good['degenerate'] and good['ratio'] < 0.5, good
    assert again == good
    assert all(p.grad is None for p in drawn.parameters())


def test_diagnostic_batch_shape():
    forms, priors, upstream = ih.diagnostic_batch(1000, 8, vocab=20, one_piece=5, seed=1)
    assert forms.shape == (20, 10) and priors.shape == (20,) and upstream.shape == (20, 8)
    assert bool((forms[:5, 1:] == 1000).all()) and bool((forms[:, 0] < 1000).all())
    assert bool((forms[5:, 1] < 1000).all()) and bool((forms[:, 0] >= 1).all())
    assert bool((priors == 0).all())


def test_check_saved_files_reads_the_bytes_on_disk(hf, tmp_path):
    hn, _ = ih.build_hypernet(hf, _table(), seed=1)
    report = ih.tensor_report(hn.state_dict())
    hn.save_pretrained(str(tmp_path / 'saved'), safe_serialization=True)
    profile = ih.Profile(source_vocab=300, width=64, expected_model_bytes=None)
    result = ih.check_saved_files(tmp_path / 'saved', report, profile)
    assert result['tensors'] == 138
    assert result['model_bytes'] == ih.predicted_safetensors_bytes(hn.state_dict())
    wrong = json.loads(json.dumps(report))
    wrong['scaler.w']['sha256'] = '0' * 64
    with pytest.raises(ih.InitError, match='scaler.w: bytes in the file differ'):
        ih.check_saved_files(tmp_path / 'saved', wrong, profile)
    wrong = json.loads(json.dumps(report))
    wrong['in_scaler.b']['shape'] = [64, 1]
    with pytest.raises(ih.InitError, match='in_scaler.b is stored as F32'):
        ih.check_saved_files(tmp_path / 'saved', wrong, profile)
    with pytest.raises(ih.InitError, match='expected 12345'):
        ih.check_saved_files(tmp_path / 'saved', report, ih.Profile(source_vocab=300, width=64,
                                                                     expected_model_bytes=12345))
    (tmp_path / 'saved' / 'extra.txt').write_text('x')
    with pytest.raises(ih.InitError, match='save_pretrained wrote'):
        ih.check_saved_files(tmp_path / 'saved', report, profile)


# --- inputs ---------------------------------------------------------------------------------------------

def _write_table(path, table, key=ih.TABLE_KEY):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: table.contiguous()}, str(path), metadata={'format': 'pt'})


def test_load_source_table_upcasts_exactly_and_checks_shape(tmp_path):
    table = torch.randn(49152, 4).to(torch.bfloat16)
    _write_table(tmp_path / 'a.safetensors', table)
    loaded, info = ih.load_source_table(tmp_path / 'a.safetensors', ih.Profile(width=4))
    assert loaded.dtype == torch.float32 and torch.equal(loaded, table.float())
    assert info['stored_dtype'] == 'BF16' and info['shape'] == [49152, 4]
    with pytest.raises(ih.InitError, match='shape'):
        ih.load_source_table(tmp_path / 'a.safetensors', ih.Profile(width=8))
    _write_table(tmp_path / 'b.safetensors', table, key='lm_head.weight')
    with pytest.raises(ih.InitError, match='no tensor model.embed_tokens.weight'):
        ih.load_source_table(tmp_path / 'b.safetensors', ih.Profile(width=4))
    bad = table.float()
    bad[3, 1] = float('inf')
    _write_table(tmp_path / 'c.safetensors', bad)
    with pytest.raises(ih.InitError, match='non-finite'):
        ih.load_source_table(tmp_path / 'c.safetensors', ih.Profile(width=4))


def _model_dir(base: Path, scratch_assets: Path, width: int = 64) -> Path:
    model = base / 'model'
    model.mkdir(parents=True)
    for name in ('config.json', 'tokenizer.json'):
        shutil.copyfile(scratch_assets / 'full' / name, model / name)
    _write_table(model / 'model.safetensors', _table(vocab=49152, width=width, seed=5).to(torch.bfloat16))
    return model


def test_inspect_model_dir(tmp_path, scratch_assets):
    model = _model_dir(tmp_path, scratch_assets)
    info = ih.inspect_model_dir(model, SMALL)
    assert info['tokenizer'] == {'vocab_size_with_added': 49152, 'vocab_size_without_added': 49152, 'eos_id': 0}
    assert info['files']['tokenizer.json']['matches_pinned'] is True
    assert info['files']['model.safetensors']['matches_pinned'] is False
    config = json.loads((model / 'config.json').read_text())
    config['vocab_size'] = 49153
    (model / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ih.InitError, match='vocab_size is 49153'):
        ih.inspect_model_dir(model, SMALL)
    (model / 'tokenizer.json').unlink()
    with pytest.raises(ih.InitError, match='fetch_model'):
        ih.inspect_model_dir(model, SMALL)


# --- the whole build at width 64 -------------------------------------------------------------------------

def test_create_end_to_end_small(tmp_path, scratch_assets):
    model = _model_dir(tmp_path, scratch_assets)
    out = tmp_path / 'input' / 'hypernet_init'
    seen = []
    manifest = ih.create(model, CODE, out, profile=SMALL, progress=lambda i, name: seen.append((i, name)))
    assert seen == list(enumerate(ih.STEPS))
    assert sorted(p.name for p in out.iterdir()) == ['config.json', 'init_manifest.json', 'model.safetensors']
    assert not [p for p in out.parent.iterdir() if p.name.startswith('.')], 'temporary directories left behind'
    for name in ('config.json', 'init_manifest.json', 'model.safetensors'):
        assert stat.S_IMODE((out / name).stat().st_mode) == 0o444
    assert json.loads((out / 'init_manifest.json').read_text()) == json.loads(json.dumps(manifest))
    checks = manifest['checks']
    for name in ('construction_init', 'parameters', 'calibration', 'finite', 'files', 'config', 'reload',
                 'trainer_emulation'):
        assert checks[name]['passed'] is True, name
    assert checks['reload']['tensors_bit_identical'] and checks['files']['tensors'] == 138
    assert checks['files']['matches_prediction']
    trainer = checks['trainer_emulation']
    assert len(trainer['fallback_symbols']) == 21 and trainer['fallback_ids'] == [49153, 49173]
    assert trainer['fallback_rows'] == 22 and trainer['resume']['passed']
    assert manifest['recipe']['seed'] == 42 and manifest['recipe']['prior_bias_seed'] == 1042
    assert manifest['recipe']['calibration'] == 'before_bias' and manifest['recipe']['option'].startswith('(b)')
    assert manifest['recipe']['rescalers_from_fixture'] is None          # no reference for the small profile
    assert checks['calibration']['passed'] and 'zero-bias network' in checks['calibration']['scope']
    assert 'passed' not in checks['anchors_after_bias'] and checks['anchors_after_bias']['max_abs_mean_error'] > 0
    assert checks['reload']['tolerance'] == {'rtol': 1e-3, 'atol': 1e-4}
    assert checks['reload']['anchor_output_relative_error'] == 0.0
    assert manifest['inputs']['allow_unpinned_model'] is False
    assert manifest['original_warmup']['config_equal_without_fallback_key'] is False   # width 64, not the original
    assert manifest['prior_gradient_diagnostic']['degenerate'] is False
    assert manifest['outputs']['model.safetensors'] == ih.file_record(out / 'model.safetensors')
    assert manifest['inputs']['table']['stored_dtype'] == 'BF16'
    assert manifest['code']['files'] == ih.PINNED_SOURCES
    config = json.loads((out / 'config.json').read_text())
    assert config['hn_byte_fallback_symbols'] is None and config['hn_n_extra_tokens'] == 1
    assert config['original_vocab_size'] == config['pad_token_id'] == 49152
    # the trainer's own loader accepts it
    hf = ih.import_hf_hypernet(CODE)
    hn = hf.ZettHypernet.from_pretrained(str(out), local_files_only=True)
    assert hn.config.hn_surface_maxlen == 10 and hn.fallback_embeddings.num_embeddings == 1
    assert ih.verify_directory(out, profile=SMALL, reference=None)['ok']

    # same recipe again: accepted as is
    again = ih.create(model, CODE, out, profile=SMALL)
    assert again['already_built'] is True
    # a different recipe is refused without --force
    with pytest.raises(ih.InitUsageError, match='cannot be reused.*recipe'):
        ih.create(model, CODE, out, profile=SMALL, seed=43)
    with pytest.raises(ih.InitUsageError, match='cannot be reused.*recipe'):
        ih.create(model, CODE, out, profile=SMALL, calibrate_after_bias=True)
    # --force is refused while a run directory holds anything
    runs = [tmp_path / 'runs' / 'uniform', tmp_path / 'runs' / 'rl']
    runs[0].mkdir(parents=True)
    (runs[0] / 'run_config.json').write_text('{}')
    with pytest.raises(ih.InitError, match='must not be replaced'):
        ih.create(model, CODE, out, profile=SMALL, seed=43, force=True, run_dirs=runs)
    shutil.rmtree(runs[0])
    # --force with empty run directories rebuilds; the whole pipeline is deterministic
    rebuilt = ih.create(model, CODE, out, profile=SMALL, force=True, run_dirs=runs)
    assert not rebuilt.get('already_built')
    assert rebuilt['outputs'] == manifest['outputs']
    assert rebuilt['tensors'] == manifest['tensors']


def test_create_with_non_default_options_records_them(tmp_path, scratch_assets, capsys):
    model = _model_dir(tmp_path, scratch_assets)
    zero = ih.create(model, CODE, tmp_path / 'zero', profile=SMALL, prior_bias_std=0.0)
    assert zero['prior_gradient_diagnostic']['degenerate'] is True
    assert 'zero-prior gradient ratio' in capsys.readouterr().out
    assert zero['recipe']['prior_bias_std'] == 0.0 and zero['recipe']['prior_bias_seed'] is None
    assert zero['tensors'][ih.PRIOR_BIAS_NAME]['absmax'] == 0.0
    assert zero['recipe']['option'].startswith('(c)') and 'anchors_after_bias' not in zero['checks']
    plain = ih.create(model, CODE, tmp_path / 'plain', profile=SMALL, rescaler_init=False)
    assert plain['checks']['calibration']['passed'] is None and plain['recipe']['anchor_count'] is None
    assert plain['recipe']['calibration'] is None
    assert plain['tensors']['in_scaler.w']['mean'] == 1.0 and plain['tensors']['scaler.b']['absmax'] == 0.0
    assert ih.verify_directory(tmp_path / 'plain', profile=SMALL, reference=None)['ok']
    after = ih.create(model, CODE, tmp_path / 'after', profile=SMALL, calibrate_after_bias=True)
    assert after['recipe']['calibration'] == 'after_bias' and after['recipe']['option'].startswith('(a)')
    assert after['checks']['calibration']['passed'] and 'anchors_after_bias' not in after['checks']
    default = ih.create(model, CODE, tmp_path / 'default', profile=SMALL)
    assert {n for n in after['tensors'] if after['tensors'][n]['sha256'] != default['tensors'][n]['sha256']} == \
        {'scaler.w', 'scaler.b'}
    with pytest.raises(ih.InitUsageError, match='no effect with --prior-bias-std 0'):
        ih.create(model, CODE, tmp_path / 'never', profile=SMALL, calibrate_after_bias=True, prior_bias_std=0.0)


def test_reload_check_tolerates_rounding_but_not_a_real_difference(tmp_path, scratch_assets, monkeypatch):
    """init-2: differently aligned reloaded weights may round differently; 1 % is a real difference."""
    model = _model_dir(tmp_path, scratch_assets)
    real_reload, real_anchor = ih.reload_hypernet, ih.anchor_outputs
    factor = {'value': None}

    def reload(hf, directory):
        hn = real_reload(hf, directory)
        hn._reloaded_for_test = True
        return hn

    def anchors(hn, table, anchor_count=ih.DEFAULT_ANCHORS):
        out = real_anchor(hn, table, anchor_count)
        if getattr(hn, '_reloaded_for_test', False):
            out = factor['value'](out)
        return out

    monkeypatch.setattr(ih, 'reload_hypernet', reload)
    monkeypatch.setattr(ih, 'anchor_outputs', anchors)
    noise = torch.randn(ih.DEFAULT_ANCHORS, 64, generator=torch.Generator().manual_seed(1)) * 1e-6
    factor['value'] = lambda out: out + noise[:out.shape[0]]
    manifest = ih.create(model, CODE, tmp_path / 'rounding', profile=SMALL)
    reload_check = manifest['checks']['reload']
    assert reload_check['passed'] and reload_check['anchor_outputs_bit_identical'] is False
    assert 0 < reload_check['anchor_output_relative_error'] < 1e-3
    factor['value'] = lambda out: out * 1.01
    with pytest.raises(ih.InitError, match='different anchor outputs'):
        ih.create(model, CODE, tmp_path / 'different', profile=SMALL)
    assert not (tmp_path / 'different').exists()


def test_production_build_refuses_an_unpinned_model_file(tmp_path, scratch_assets):
    """init-4: the production profile stops on model files that differ from the pins (unless allowed)."""
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('config.json', 'tokenizer.json'):
        shutil.copyfile(scratch_assets / 'full' / name, model / name)
    _write_table(model / 'model.safetensors', _table(vocab=49152, width=8, seed=5).to(torch.bfloat16))
    with pytest.raises(ih.InitUsageError, match='model.safetensors has SHA-256 .* pins .*fetch_model'):
        ih.create(model, CODE, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_create_refuses_tampered_or_foreign_directories(tmp_path, scratch_assets):
    model = _model_dir(tmp_path, scratch_assets)
    out = tmp_path / 'hypernet_init'
    ih.create(model, CODE, out, profile=SMALL)
    target = out / 'model.safetensors'
    target.chmod(0o644)
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    result = ih.verify_directory(out, profile=SMALL, reference=None)
    assert not result['ok'] and 'SHA-256 differs' in ' '.join(result['problems'])
    with pytest.raises(ih.InitError, match='cannot be reused'):
        ih.create(model, CODE, out, profile=SMALL)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'notes.txt').write_text('x')
    with pytest.raises(ih.InitError, match='cannot be reused'):
        ih.create(model, CODE, foreign, profile=SMALL)
    assert (foreign / 'notes.txt').read_text() == 'x'


def test_missing_directory_with_started_runs_needs_force(tmp_path, scratch_assets):
    model = _model_dir(tmp_path, scratch_assets)
    runs = [tmp_path / 'runs' / 'uniform']
    runs[0].mkdir(parents=True)
    (runs[0] / 'events.jsonl').write_text('')
    with pytest.raises(ih.InitError, match='Pass --force to rebuild anyway'):
        ih.create(model, CODE, tmp_path / 'hypernet_init', profile=SMALL, run_dirs=runs)
    assert not (tmp_path / 'hypernet_init').exists()


def test_interrupt_removes_temporary_directories(tmp_path, scratch_assets):
    model = _model_dir(tmp_path, scratch_assets)
    out = tmp_path / 'hypernet_init'

    def progress(index, name):
        if name == 'reload':
            raise ih.Terminated(15)

    with pytest.raises(ih.Terminated):
        ih.create(model, CODE, out, profile=SMALL, progress=progress)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['model']


def test_failed_force_replacement_restores_the_old_directory(tmp_path, scratch_assets, monkeypatch):
    model = _model_dir(tmp_path, scratch_assets)
    out = tmp_path / 'hypernet_init'
    first = ih.create(model, CODE, out, profile=SMALL)
    calls = []

    def flaky(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        if len(calls) == 2:
            raise OSError('simulated failure')
        os.rename(src, dst)

    monkeypatch.setattr(ih, '_rename', flaky)
    with pytest.raises(OSError, match='simulated'):
        ih.create(model, CODE, out, profile=SMALL, seed=7, force=True)
    assert [c[1].split('--')[0] for c in calls] == ['.hypernet_init.replaced', 'hypernet_init', 'hypernet_init']
    assert ih.file_record(out / 'model.safetensors') == first['outputs']['model.safetensors']
    assert not [p for p in tmp_path.iterdir() if p.name.startswith('.')]


def test_cleanup_stale(tmp_path):
    out = tmp_path / 'hypernet_init'
    host = ih._host()
    dead = subprocess.run([sys.executable, '-c', 'import os; print(os.getpid())'], capture_output=True, text=True)
    pid = int(dead.stdout)
    stale = tmp_path / f'.hypernet_init.building--{host}--{pid}'
    alive = tmp_path / f'.hypernet_init.trainer-check--{host}--{os.getpid()}'
    elsewhere = tmp_path / f'.hypernet_init.building--other-node--{pid}'
    killed = tmp_path / f'.hypernet_init.trainer-check--node-killed--{pid}'
    for path in (stale, alive, elsewhere, killed):
        path.mkdir()
    old = time.time() - 7 * 3600                    # init-6: a hard-killed job on another node, 7 h ago
    os.utime(killed, (old, old))
    assert ih.cleanup_stale(out) == [str(stale), str(killed)]
    assert alive.exists() and elsewhere.exists() and not stale.exists() and not killed.exists()
    (tmp_path / f'.hypernet_init.replaced--{host}--{pid}').mkdir()
    with pytest.raises(ih.InitUsageError, match='interrupted --force'):
        ih.cleanup_stale(out)


# --- command line ------------------------------------------------------------------------------------------

def test_cli_without_config_or_paths_explains_the_remedy(capsys):
    assert ih.main([]) == 2
    err = capsys.readouterr().err
    assert 'INIT FAILED' in err and 'config.env' in err


def test_cli_with_config_reports_missing_code_and_model_and_status(tmp_config, capsys):
    import signal
    offline_before, sigint_before = os.environ.get('HF_HUB_OFFLINE'), signal.getsignal(signal.SIGINT)
    assert ih.main(['--status']) == 2                  # default --code-dir: the (missing) staged tree
    err = capsys.readouterr().err
    assert 'stage_code' in err and str(tmp_config.paths.code_dir('uniform')) in err
    assert ih.main(['--status', '--code-dir', str(CODE)]) == 2
    err = capsys.readouterr().err
    assert 'fetch_model' in err and str(tmp_config.paths.model_dir) in err
    status = json.loads(tmp_config.paths.status_file('init').read_text())
    assert status['state'] == 'failed' and 'fetch_model' in status['error']
    assert status['hyak_stage'] == 'init' and status['total'] == len(ih.STEPS) and status['exit_code'] == 2
    assert os.environ.get('HF_HUB_OFFLINE') == offline_before, 'main() leaked HF_HUB_OFFLINE'
    assert signal.getsignal(signal.SIGINT) is sigint_before, 'main() leaked its signal handler'


def test_cli_verify_of_a_missing_directory(tmp_path, capsys):
    assert ih.main(['--verify', '--out', str(tmp_path / 'nothing'), '--model-dir', str(tmp_path),
                    '--code-dir', str(CODE)]) == 4
    assert 'NOT VERIFIED' in capsys.readouterr().out


def test_cli_verify_needs_no_config_file(tmp_path, monkeypatch, capsys):
    """init-5: --verify --out DIR reads only DIR."""
    monkeypatch.setenv('HYAK_CONFIG', str(tmp_path / 'missing.env'))
    assert ih.main(['--verify', '--out', str(tmp_path / 'nothing')]) == 4
    captured = capsys.readouterr()
    assert 'NOT VERIFIED' in captured.out and 'config.env' not in captured.err and 'missing.env' not in captured.err


def test_cli_sets_hf_hub_offline_before_importing_transformers(tmp_path):
    """init-8: huggingface_hub reads HF_HUB_OFFLINE at import time."""
    env = {k: v for k, v in os.environ.items() if k != 'HF_HUB_OFFLINE'}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    script = ('import sys; from hyak.model import init_hypernet as ih; '
              'assert "huggingface_hub" not in sys.modules; '
              'code = ih.main(["--verify", "--out", sys.argv[1]]); '
              'import huggingface_hub.constants as c; print(code, c.HF_HUB_OFFLINE)')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'nothing')], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ['4', 'True']


def test_terminal_status_removes_the_job_scripts_worker_rows(tmp_config, monkeypatch, capsys):
    """init-10: the final status must not keep the 'Starting' row of the job script."""
    from hyak.common.status import StatusWriter
    calls = []
    monkeypatch.setattr(StatusWriter, 'remove_job_rows', lambda self, job=None: calls.append(job) or 0)
    assert ih.main(['--status', '--code-dir', str(CODE)]) == 2
    assert calls == [None]


def test_original_warmup_config_is_the_init_config_without_the_fallback_key(tmp_path, scratch_dir):
    """init-9: removing one line of the saved config gives the original warmup config."""
    built = scratch_dir / 'init-test' / 'hypernet_init' / 'config.json'
    if not built.is_file():
        pytest.skip(f'{built} not found')
    record = ih.original_warmup_record(built, {'config.json': ih.file_record(built)})
    assert record['config_equal_without_fallback_key'] is True and record['config_sha256_equal'] is False
    assert 'identity_warmup_step' in record['note'] and 'batch size is unknown' in record['note']
    other = tmp_path / 'config.json'
    other.write_bytes(built.read_bytes().replace(b'"hn_n_layers": 3', b'"hn_n_layers": 4'))
    assert ih.original_warmup_record(other, {'config.json': ih.file_record(other)})[
        'config_equal_without_fallback_key'] is False


def test_cli_help_runs_in_a_fresh_process():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-m', 'hyak.model.init_hypernet', '--help'], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and '--prior-bias-std' in result.stdout and '--no-rescaler-init' in result.stdout
    assert '--calibrate-after-bias' in result.stdout and '--allow-unpinned-model' in result.stdout
