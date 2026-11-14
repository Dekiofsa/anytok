"""Deterministic equivalence of the chunked forward with the pinned forward (DESIGN.md 5.D tests 1, 4, 6).

1. float64, dropout off: outputs and ALL parameter gradients equal the pinned ``__call__`` to
   <= 1e-9 relative for V not divisible by the chunk size, chunk sizes {1, 7, V, 10 V}, zero and
   non-zero priors, padded rows and byte-fallback ids, the pinned path with and without its own
   activation-checkpoint flags, the chunked path with and without checkpointing.
4. The trainer's loss plumbing (T:546-577: detached float leaf, LM gradient injection, the real
   ``input_lexical_loss`` with weight 0.5) through the installed class-level patch.
6. bf16 autocast on CPU: chunked and pinned agree up to bf16 rounding (judged against the pinned
   path's own bf16-versus-float32 error, as the probe does).

Every check runs on the library initialisation and on models whose biases, LayerNorm weights and
rescalers are randomised (``build_hn(randomize=True)``), because the library initialisation (all
biases 0, norms 1) hides kernel differences of the two paths.
"""
from __future__ import annotations

import pytest
import torch

from hyak.tests.test_memory_common import (  # noqa: F401  (_no_leftover_patch is an autouse fixture)
    _no_leftover_patch, build_hn, chunked, grads_of, make_forms, make_priors, max_relative_errors, pinned,
    run_forward_backward,
)
from hyak.memory import chunked_hypernet as ch
from hf_hypernet.modeling_hypernet import ZettHypernet

V = 45            # not divisible by 7
SOURCE_VOCAB = 300
TOL64 = 1e-9


@pytest.fixture(scope='module', params=['library', 'randomized'])
def model64(request):
    hn, source = build_hn(SOURCE_VOCAB, width=64, layers=2, heads=4, dtype=torch.float64,
                          randomize=request.param == 'randomized')
    forms = make_forms(V, SOURCE_VOCAB)
    cotangent = torch.randn(V, 64, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    return hn, source, forms, cotangent


@pytest.fixture(scope='module')
def references(model64):
    """Pinned results for both prior kinds and both pinned checkpoint settings."""
    hn, source, forms, cotangent = model64
    refs = {}
    for kind in ('zero', 'unigram'):
        priors = make_priors(V, kind, dtype=torch.float64)
        for flags in (True, False):
            refs[kind, flags] = run_forward_backward(pinned(flags), hn, forms, priors, source, cotangent)
    return refs


def _assert_close(reference, result, tol, label):
    out_ref, second_ref, bias_ref, grads_ref = reference
    out, second, bias, grads = result
    assert second is None and second_ref is None
    assert bias.dtype == bias_ref.dtype and torch.equal(bias, bias_ref), f'{label}: bias output differs'
    assert out.dtype == out_ref.dtype and out.shape == out_ref.shape
    out_err = float((out - out_ref).abs().max()) / float(out_ref.abs().max())
    assert out_err <= tol, f'{label}: output relative error {out_err:.3e}'
    errors, mismatched = max_relative_errors(grads_ref, grads)
    assert not mismatched, f'{label}: gradient presence differs for {mismatched}'
    worst = max(errors, key=errors.get)
    assert errors[worst] <= tol, f'{label}: gradient of {worst} relative error {errors[worst]:.3e}'
    return out_err, errors[worst]


def test_pinned_references_are_consistent(references):
    # The pinned path's own checkpoint flags change nothing (sanity of the reference).
    for kind in ('zero', 'unigram'):
        _assert_close(references[kind, True], references[kind, False], TOL64, f'pinned flags {kind}')
    # Exactly the parameters the trainer never trains have no gradient.
    grads = references['unigram', True][3]
    none = sorted(name for name, grad in grads.items() if grad is None)
    assert none == ['in_scaler.b', 'in_scaler.w', 'model.lang_embedding.weight', 'model.word_embeddings.weight',
                    'scaler.b', 'scaler.w']
    assert grads['fallback_embeddings.weight'].abs().sum() > 0, 'fallback rows must receive gradient'


@pytest.mark.parametrize('rows', [1, 7, V, 10 * V])
@pytest.mark.parametrize('kind', ['zero', 'unigram'])
@pytest.mark.parametrize('use_checkpoint', [True, False])
def test_float64_outputs_and_all_gradients_equal_pinned(model64, references, rows, kind, use_checkpoint):
    hn, source, forms, cotangent = model64
    priors = make_priors(V, kind, dtype=torch.float64)
    result = run_forward_backward(chunked(rows, use_checkpoint), hn, forms, priors, source, cotangent)
    for flags in (True, False):
        _assert_close(references[kind, flags], result, TOL64,
                      f'rows={rows} priors={kind} checkpoint={use_checkpoint} pinned_flags={flags}')


@pytest.mark.parametrize('randomize', [False, True])
def test_float64_three_layers_and_float32_priors(randomize):
    # Trainer dtypes for the inputs (float32 priors), a third layer and a different width.
    hn, source = build_hn(SOURCE_VOCAB, width=48, layers=3, heads=3, seed=3, dtype=torch.float64, randomize=randomize)
    forms = make_forms(61, SOURCE_VOCAB, seed=9)
    priors = make_priors(61, 'unigram', dtype=torch.float32)
    cotangent = torch.randn(61, 48, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    reference = run_forward_backward(pinned(True), hn, forms, priors, source, cotangent)
    for rows in (8, 13, 60):
        _assert_close(reference, run_forward_backward(chunked(rows), hn, forms, priors, source, cotangent),
                      TOL64, f'3 layers rows={rows}')


def _trainer_plumbing(hn, forms, priors, source, *, lm_weight):
    """T:546-577 for one update, with a dummy LM loss in place of the four LM micro-batches."""
    from hf_hypernet.joint_training import input_lexical_loss
    hn.zero_grad(set_to_none=True)
    generated, output_embeddings, _ = hn(forms, priors, source, None, deterministic=True)
    assert output_embeddings is None and torch.isfinite(generated).all()
    leaf = generated.detach().float().requires_grad_(True)                        # T:551
    rows = torch.arange(0, forms.shape[0], 3)
    lm_loss = ((leaf[rows] @ lm_weight).tanh().square().sum(-1)).mean()           # stands in for T:552-567
    lm_loss.backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    lexical = input_lexical_loss(generated, forms, source, hn.config.pad_token_id, 0)  # T:570-573
    assert torch.isfinite(lexical)
    ((generated.float() * leaf.grad.detach()).sum() + 0.5 * lexical).backward()   # T:577
    return generated.detach().clone(), float(lexical.detach()), grads_of(hn)


@pytest.mark.parametrize('randomize', [False, True])
@pytest.mark.parametrize('kind', ['zero', 'unigram'])
def test_trainer_loss_plumbing_through_installed_patch(kind, randomize):
    hn, source = build_hn(SOURCE_VOCAB, width=64, layers=2, heads=4, seed=4, dtype=torch.float32, randomize=randomize)
    vocab = 157
    forms = make_forms(vocab, SOURCE_VOCAB, seed=11)
    priors = make_priors(vocab, kind)
    lm_weight = torch.randn(64, 32, generator=torch.Generator().manual_seed(6)) * 0.3
    reference = _trainer_plumbing(hn, forms, priors, source, lm_weight=lm_weight)
    assert not ch.is_installed(ZettHypernet)
    for rows in (16, 50, vocab):
        ch.install(ZettHypernet, rows, min_chunk_rows=1)   # tiny vocabulary: below the training floor
        try:
            assert ch.is_installed(ZettHypernet)
            result = _trainer_plumbing(hn, forms, priors, source, lm_weight=lm_weight)
        finally:
            ch.uninstall(ZettHypernet)
        out_err = float((result[0] - reference[0]).abs().max()) / float(reference[0].abs().max())
        assert out_err <= 2e-6, f'rows={rows}: generated relative error {out_err:.2e}'
        assert abs(result[1] - reference[1]) <= 2e-6 * abs(reference[1])
        errors, mismatched = max_relative_errors(reference[2], result[2])
        assert not mismatched
        worst = max(errors, key=errors.get)
        assert errors[worst] <= 2e-5, f'rows={rows}: gradient of {worst} relative error {errors[worst]:.2e}'


def _cpu_bf16_autocast_supported():
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = torch.nn.functional.linear(torch.ones(2, 4), torch.ones(3, 4))
        return out.dtype == torch.bfloat16
    except (RuntimeError, TypeError):
        return False


def _gradient_error(reference, result):
    """Global relative error over all gradients (None-ness must agree)."""
    diff = total = 0.0
    for name, ref in reference.items():
        got = result[name]
        assert (ref is None) == (got is None), name
        if ref is not None:
            diff += float((got.double() - ref.double()).norm()) ** 2
            total += float(ref.double().norm()) ** 2
    return (diff / total) ** 0.5


@pytest.mark.skipif(not _cpu_bf16_autocast_supported(), reason='CPU bf16 autocast not supported by this torch build')
@pytest.mark.parametrize('randomize', [False, True])
def test_bf16_autocast_agrees_up_to_bf16_rounding(randomize):
    hn, source = build_hn(SOURCE_VOCAB, width=64, layers=2, heads=4, seed=7, dtype=torch.float32, randomize=randomize)
    vocab = 157
    forms = make_forms(vocab, SOURCE_VOCAB, seed=12)
    priors = make_priors(vocab, 'unigram')
    cotangent = torch.randn(vocab, 64, generator=torch.Generator().manual_seed(9))

    def under_autocast(call):
        def wrapped(*args):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                return call(*args)
        return wrapped

    reference = run_forward_backward(under_autocast(pinned(True)), hn, forms, priors, source, cotangent)
    full = run_forward_backward(pinned(True), hn, forms, priors, source, cotangent)
    assert reference[0].dtype == torch.float32, 'the frozen fp32 output scaler makes the result float32'
    # the bf16 rounding floor: the pinned path itself, bf16 versus float32
    floor_out = float((reference[0] - full[0]).norm() / full[0].norm())
    floor_grad = _gradient_error(full[3], reference[3])
    assert 0 < floor_out < 5e-2 and 0 < floor_grad < 0.2
    for rows in (16, 50):
        result = run_forward_backward(under_autocast(chunked(rows)), hn, forms, priors, source, cotangent)
        assert result[0].dtype == reference[0].dtype
        assert torch.equal(result[2], reference[2])
        rel = float((result[0] - reference[0]).norm() / reference[0].norm())
        assert rel <= 3 * floor_out, f'rows={rows}: bf16 output error {rel:.3e} vs rounding floor {floor_out:.3e}'
        grad = _gradient_error(reference[3], result[3])
        assert grad <= 3 * floor_grad, f'rows={rows}: bf16 gradient error {grad:.3e} vs floor {floor_grad:.3e}'
