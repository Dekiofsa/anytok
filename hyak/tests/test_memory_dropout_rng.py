"""Dropout on: reproducibility and RNG continuity of the chunked forward (DESIGN.md 5.D test 2).

* With a fixed seed, ``use_checkpoint=True`` and ``use_checkpoint=False`` give bit-identical
  outputs and gradients (the recomputation sees the forward's dropout masks).
* The global RNG state after ``backward()`` equals the state right after the forward
  (checkpoint recomputation forks and restores the generator).
* Two consecutive training updates (forward, trainer loss, backward, clip, AdamW step) are
  reproducible end to end from a seed.
* The backward recomputes the chunk regions exactly once each, in reverse creation order.
"""
from __future__ import annotations

import copy

import pytest
import torch

from hyak.tests.test_memory_common import (  # noqa: F401
    _no_leftover_patch, build_hn, grads_of, make_forms, make_priors, pinned_call,
)
from hyak.memory import chunked_hypernet as ch
from hf_hypernet.modeling_hypernet import ZettHypernet

SOURCE_VOCAB = 300
V = 45


@pytest.fixture(scope='module', params=['library', 'randomized'])
def model32(request):
    hn, source = build_hn(SOURCE_VOCAB, width=64, layers=2, heads=4, seed=21, dtype=torch.float32,
                          randomize=request.param == 'randomized')
    forms = make_forms(V, SOURCE_VOCAB, seed=22)
    priors = make_priors(V, 'unigram')
    cotangent = torch.randn(V, 64, generator=torch.Generator().manual_seed(23))
    return hn, source, forms, priors, cotangent


def _dropout_run(hn, forms, priors, source, cotangent, rows, use_checkpoint, seed=1234):
    hn.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    before = torch.get_rng_state().clone()
    out, second, _ = ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=rows,
                                     use_checkpoint=use_checkpoint)
    after_forward = torch.get_rng_state().clone()
    (out * cotangent).sum().backward()
    after_backward = torch.get_rng_state().clone()
    return out.detach().clone(), grads_of(hn), before, after_forward, after_backward


@pytest.mark.parametrize('rows', [1, 7, 20, V])
def test_checkpoint_and_direct_chunk_loops_are_bit_identical(model32, rows):
    hn, source, forms, priors, cotangent = model32
    with_ckpt = _dropout_run(hn, forms, priors, source, cotangent, rows, True)
    direct = _dropout_run(hn, forms, priors, source, cotangent, rows, False)
    assert torch.equal(with_ckpt[0], direct[0]), 'outputs differ'
    for name, grad in with_ckpt[1].items():
        other = direct[1][name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert torch.equal(grad, other), f'gradient of {name} differs'
    # Dropout really is active: a deterministic forward gives something else.
    deterministic, _, _ = ch.chunked_call(hn, forms, priors, source, None, True, chunk_rows=rows)
    assert not torch.equal(deterministic.detach(), with_ckpt[0])


@pytest.mark.parametrize('use_checkpoint', [True, False])
def test_rng_after_backward_continues_where_the_forward_left_it(model32, use_checkpoint):
    hn, source, forms, priors, cotangent = model32
    _, _, before, after_forward, after_backward = _dropout_run(hn, forms, priors, source, cotangent, 7,
                                                               use_checkpoint)
    assert not torch.equal(before, after_forward), 'the forward must have drawn dropout masks'
    assert torch.equal(after_forward, after_backward), 'backward changed the global RNG state'
    # And the stream position after the forward equals that of an uncheckpointed forward.
    _, _, _, direct_after_forward, _ = _dropout_run(hn, forms, priors, source, cotangent, 7, False)
    assert torch.equal(after_forward, direct_after_forward)


def test_single_chunk_draws_exactly_the_pinned_dropout_stream(model32, request):
    hn, source, forms, priors, _ = model32
    torch.manual_seed(77)
    reference, _, _ = pinned_call()(hn, forms, priors, source, None, False)
    reference = reference.detach()
    if request.node.callspec.params['model32'] == 'library':
        torch.manual_seed(77)
        assert torch.equal(reference, ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=V)[0])
    torch.manual_seed(77)
    single = ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=V)[0].detach()
    # Same masks.  With non-zero biases the two paths feed the down-attention keys to a Linear as a
    # strided view (pinned) or a contiguous tensor (chunked), which may round differently in float32.
    assert float((single - reference).abs().max()) <= 1e-5 * float(reference.abs().max())
    torch.manual_seed(77)
    several = ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=7)[0].detach()
    assert float((several - reference).abs().max()) > 1e-2 * float(reference.abs().max()), \
        'several chunks draw a different (equally distributed) stream'


def _two_updates(hn, source, forms, priors, seed):
    """Two trainer-shaped updates (T:543-584 without the LM): patched hn(...), joint clip, AdamW."""
    from hf_hypernet.joint_training import input_lexical_loss
    torch.manual_seed(seed)
    params = [p for p in hn.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    norms = []
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        generated, _, _ = hn(forms, priors, source, None, deterministic=False)
        leaf = generated.detach().float().requires_grad_(True)
        (leaf[::2].tanh().sum()).backward()
        lexical = input_lexical_loss(generated, forms, source, hn.config.pad_token_id, 0)
        ((generated.float() * leaf.grad.detach()).sum() + 0.5 * lexical).backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(params, 0.1, foreach=False)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {name: p.detach().clone() for name, p in hn.named_parameters()}, norms, torch.get_rng_state()


def test_two_consecutive_updates_are_reproducible_from_a_seed(model32):
    hn0, source, forms, priors, _ = model32
    ch.install(ZettHypernet, 10, min_chunk_rows=1)   # tiny vocabulary: below the training floor
    try:
        first = _two_updates(copy.deepcopy(hn0), source, forms, priors, seed=99)
        second = _two_updates(copy.deepcopy(hn0), source, forms, priors, seed=99)
        other_seed = _two_updates(copy.deepcopy(hn0), source, forms, priors, seed=100)
    finally:
        ch.uninstall(ZettHypernet)
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    for name, value in first[0].items():
        assert torch.equal(value, second[0][name]), f'{name} differs between identical runs'
    assert first[1] != other_seed[1], 'a different seed must give different dropout masks'
    initial = dict(hn0.named_parameters())
    assert not torch.equal(first[0]['model.layers.0.output_dense.weight'],
                           initial['model.layers.0.output_dense.weight'].detach()), 'the updates must move weights'


def test_backward_recomputes_each_region_once_in_reverse_order(model32):
    hn, source, forms, priors, cotangent = model32
    rows = 20  # chunks of 20, 20, 5
    events = []
    with ch.observe(lambda region, layer, chunk, recompute: events.append((region, layer, chunk, recompute))):
        out, _, _ = ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=rows)
        forward = [e[:3] for e in events if not e[3]]
        (out * cotangent).sum().backward()
    recomputed = [e[:3] for e in events if e[3]]
    chunks = range(3)
    expected_forward = [('embed', None, c) for c in chunks]
    for layer in range(2):
        expected_forward += [('attention', layer, c) for c in chunks] + [('up_ffn', layer, c) for c in chunks]
    assert forward == expected_forward
    assert all(not e[3] for e in events[:len(forward)]), 'no recomputation may happen during the forward'
    assert recomputed == list(reversed(expected_forward)), recomputed


def test_no_recompute_and_pinned_result_without_grad(model32):
    hn, source, forms, priors, _ = model32
    ch.install(ZettHypernet, 7, min_chunk_rows=1)
    try:
        events = []
        with ch.observe(lambda *e: events.append(e)), torch.no_grad():
            out, _, _ = hn(forms, priors, source, None, deterministic=True)
        assert events == [], 'without gradients the pinned forward must run'
        with torch.no_grad():
            reference, _, _ = pinned_call()(hn, forms, priors, source, None, True)
        assert torch.equal(out, reference)
    finally:
        ch.uninstall(ZettHypernet)
