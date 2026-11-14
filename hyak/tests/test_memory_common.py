"""Shared helpers for the hyak.memory tests (plus sanity tests of the helpers themselves).

Small hypernetworks are built with the pinned ``build_input_hypernet`` from
``runtime/anytok_pool_20k`` (no Rust library needed) and calibrated with the pinned
``initialize_rescalers``, exactly as the random-init recipe (DESIGN.md D1) does.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PINNED_CODE = REPO / 'runtime' / 'anytok_pool_20k'
if str(PINNED_CODE) not in sys.path:
    sys.path.insert(0, str(PINNED_CODE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hf_hypernet.identity_warmup import build_input_hypernet, initialize_rescalers  # noqa: E402
from hf_hypernet.modeling_hypernet import ZettHypernet  # noqa: E402

from hyak.memory import chunked_hypernet as ch  # noqa: E402

#: Number of byte-fallback rows added like ``enable_byte_fallbacks`` does (row 0 = pad row).
FALLBACK_SYMBOLS = 3


def pinned_call():
    """The original (pinned) ``ZettHypernet.__call__``, whether or not a patch is installed."""
    return ch.original_call(ZettHypernet)


def build_hn(source_vocab=300, width=64, layers=2, heads=4, *, seed=0, dtype=torch.float64,
             fallback_symbols=FALLBACK_SYMBOLS, source_scale=0.1, randomize=False):
    """A small pinned hypernetwork in training mode plus its canonical source table.

    Returns ``(hn, source)``; ``hn`` has ``1 + fallback_symbols`` fallback rows (ids
    ``source_vocab + 1 ...``), calibrated rescalers and the trainer's checkpoint flags on.
    ``randomize=True`` then perturbs every bias, every LayerNorm weight and bias and the four
    rescaler tensors (the library initialisation leaves biases exactly 0 and norms exactly 1, which
    a trained model does not have and which hides kernel differences of the two paths under bf16).
    """
    torch.manual_seed(seed)
    hn = build_input_hypernet(source_vocab, width, layers=layers, attention_heads=heads)
    if fallback_symbols:
        hn.expand_fallback_embeddings(1 + fallback_symbols)  # tokenizer_inputs.py:33 style expansion
    generator = torch.Generator().manual_seed(seed + 17)
    source = torch.randn(source_vocab, width, generator=generator) * source_scale
    initialize_rescalers(hn, source)
    if randomize:
        randomize_parameters(hn, seed=seed + 29)
    hn = hn.to(dtype)
    hn.train()
    hn.model.activation_checkpointing = True
    hn.activation_checkpoint_input = True
    return hn, source.to(dtype)


def randomize_parameters(hn, *, seed):
    """Perturb biases, LayerNorm weights/biases and rescalers in place (see ``build_hn(randomize=True)``)."""
    generator = torch.Generator().manual_seed(seed)

    def noise(tensor, scale):
        return torch.randn(tensor.shape, generator=generator, dtype=torch.float32).to(tensor.dtype) * scale

    with torch.no_grad():
        for module in hn.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.mul_(1 + noise(module.weight, 0.1))
                module.bias.add_(noise(module.bias, 0.05))
            elif isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.add_(noise(module.bias, 0.05))
        for scaler in (hn.in_scaler, hn.scaler):
            scaler.w.mul_(1 + noise(scaler.w, 0.1))
            scaler.b.add_(noise(scaler.b, 0.01))


def make_forms(vocab, source_vocab, *, seed=1, fallback_symbols=FALLBACK_SYMBOLS, pieces=10):
    """``[vocab, pieces]`` int64 surface forms like build_surface_forms output.

    Row 0 is the end-of-text row ``[0, pad...]``; about a fifth of the rows are single
    canonical pieces (lexical-loss overlap rows); the rest have 2..pieces pieces with
    right padding (pad id = ``source_vocab``) and a sprinkling of byte-fallback ids.
    """
    generator = torch.Generator().manual_seed(seed)
    pad = source_vocab
    forms = torch.randint(1, source_vocab, (vocab, pieces), generator=generator)
    lengths = torch.randint(2, pieces + 1, (vocab,), generator=generator)
    single = torch.rand(vocab, generator=generator) < 0.2
    lengths[single] = 1
    forms[torch.arange(pieces)[None, :] >= lengths[:, None]] = pad
    if fallback_symbols:
        fallback = torch.rand(vocab, pieces, generator=generator) < 0.05
        fallback &= forms != pad
        fallback[single] = False
        ids = source_vocab + 1 + torch.randint(0, fallback_symbols, (vocab, pieces), generator=generator)
        forms[fallback] = ids[fallback]
    forms[0] = pad
    forms[0, 0] = 0
    return forms


def make_priors(vocab, kind, *, seed=2, dtype=torch.float32):
    """Zero priors (BPE / WordPiece pool members) or Unigram-like log-probabilities."""
    if kind == 'zero':
        return torch.zeros(vocab, dtype=dtype)
    if kind == 'unigram':
        generator = torch.Generator().manual_seed(seed)
        return (-torch.rand(vocab, generator=generator, dtype=torch.float64) * 12 - 2).to(dtype)
    raise ValueError(kind)


def grads_of(module):
    """``{name: grad clone or None}`` for every parameter."""
    return {name: (p.grad.detach().clone() if p.grad is not None else None)
            for name, p in module.named_parameters()}


def max_relative_errors(reference: dict, candidate: dict, floor_fraction=1e-6):
    """Per-tensor ``max|a-b| / max(max|ref|, floor_fraction * global max|ref|)``.

    The floor matters for gradients that are analytically zero (for example the key biases of
    every attention: softmax is invariant to the per-query shift they add), whose reference
    values are pure round-off.  Returns ``(errors dict, names whose None-ness differs)``.
    """
    mismatched = sorted(n for n in reference if (reference[n] is None) != (candidate.get(n) is None))
    scale = max((float(t.abs().max()) for t in reference.values() if t is not None and t.numel()), default=0.0)
    errors = {}
    for name, ref in reference.items():
        got = candidate.get(name)
        if ref is None or got is None:
            continue
        denom = max(float(ref.abs().max()), floor_fraction * scale, 1e-300)
        errors[name] = float((got.double() - ref.double()).abs().max()) / denom
    return errors, mismatched


def run_forward_backward(call, hn, forms, priors, source, cotangent, *, deterministic=True, seed=None):
    """Run ``call(hn, forms, priors, source, None, deterministic)``, backward ``(out * cot).sum()``.

    Returns ``(output clone, second output, third output, grads)``.
    """
    hn.zero_grad(set_to_none=True)
    if seed is not None:
        torch.manual_seed(seed)
    first, second, third = call(hn, forms, priors, source, None, deterministic)
    (first * cotangent).sum().backward()
    return first.detach().clone(), second, third.detach().clone(), grads_of(hn)


def chunked(rows, use_checkpoint=True):
    """A ``call`` for :func:`run_forward_backward` that uses ``chunked_call``."""
    def call(hn, forms, priors, source, lang_index, deterministic):
        return ch.chunked_call(hn, forms, priors, source, lang_index, deterministic, chunk_rows=rows,
                               use_checkpoint=use_checkpoint)
    return call


def pinned(hn_checkpoint_flags=True):
    """A ``call`` that uses the pinned ``__call__`` with or without its own checkpoint flags."""
    def call(hn, forms, priors, source, lang_index, deterministic):
        hn.model.activation_checkpointing = hn_checkpoint_flags
        hn.activation_checkpoint_input = hn_checkpoint_flags
        try:
            return pinned_call()(hn, forms, priors, source, lang_index, deterministic)
        finally:
            hn.model.activation_checkpointing = True
            hn.activation_checkpoint_input = True
    return call


@pytest.fixture(autouse=True)
def _no_leftover_patch():
    """No test may leave ZettHypernet patched (the patch is process-global)."""
    yield
    if ch.is_installed(ZettHypernet):
        ch.uninstall(ZettHypernet)
        pytest.fail('a test left the chunked patch installed on ZettHypernet')


# --- sanity tests of the helpers ----------------------------------------------------------------
def test_helper_forms_cover_padding_fallbacks_and_single_pieces():
    forms = make_forms(200, 300)
    pad = 300
    assert forms.dtype == torch.int64 and forms.shape == (200, 10)
    assert forms[0, 0] == 0 and bool((forms[0, 1:] == pad).all())
    assert bool((forms[:, 0] != pad).all()), 'every row needs at least one piece'
    single = (forms[:, 1:] == pad).all(dim=1)
    assert 10 < int(single.sum()) < 100
    assert bool(((forms > pad) & (forms <= pad + FALLBACK_SYMBOLS)).any()), 'fallback ids expected'
    assert bool((forms <= pad + FALLBACK_SYMBOLS).all())


def test_helper_randomize_perturbs_biases_norms_and_rescalers():
    plain, _ = build_hn(seed=5)
    noisy, _ = build_hn(seed=5, randomize=True)
    changed = {name for name, p in plain.named_parameters()
               if not torch.equal(p, dict(noisy.named_parameters())[name])}
    assert {'in_scaler.w', 'in_scaler.b', 'scaler.w', 'scaler.b', 'model.LayerNorm.weight',
            'model.layers.0.attention.query.bias'} <= changed
    assert 'model.layers.0.attention.query.weight' not in changed
    ch.check_supported(noisy)


def test_helper_builds_the_pinned_configuration():
    hn, source = build_hn()
    assert type(hn).__module__ == 'hf_hypernet.modeling_hypernet'
    assert Path(sys.modules[type(hn).__module__].__file__).resolve().parent == PINNED_CODE / 'hf_hypernet'
    assert hn.fallback_embeddings.num_embeddings == 1 + FALLBACK_SYMBOLS
    assert not torch.equal(hn.in_scaler.w, torch.ones_like(hn.in_scaler.w)), 'rescalers must be calibrated'
    assert hn.dtype == torch.float64 and source.dtype == torch.float64
    ch.check_supported(hn)
