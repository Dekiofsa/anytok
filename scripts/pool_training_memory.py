"""The GPU-only execution verified in the complete native-128k H200 smoke.

No CPU or disk offload, vocabulary truncation or whole-entry chunking.
All native matrix operations and parameter-gradient reductions keep their
original shapes. Call once after constructing/loading the hypernetwork.
"""
from types import MethodType

import torch

from chunked_hypernetwork import recompute
from local_memory_variants import apply_variant
from lossless_layer_norm import apply_chunked_norm, apply_chunked_source, native_mlp_layer


def apply_gpu_memory_fix(hypernetwork, chunk_rows=16384):
    if getattr(hypernetwork, '_pool_gpu_memory_fix', False):
        raise ValueError('GPU memory fix was already installed')
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    names = tuple(hypernetwork.state_dict())
    apply_variant(hypernetwork, 'split_norm_offload')
    for layer in hypernetwork.model.layers:
        original = MethodType(native_mlp_layer, layer)
        def wrapped(*values, original=original):
            return recompute(original, *values)
        layer.forward = wrapped
    apply_chunked_norm(hypernetwork, chunk_rows, stage_output_norm=False)
    apply_chunked_source(hypernetwork, 2048)
    if tuple(hypernetwork.state_dict()) != names:
        raise ValueError('Memory execution changed model parameter names')
    hypernetwork._pool_gpu_memory_fix = True


def backward_hypernetwork(generated, embedding_gradient, lexical, optimized):
    """Preserve both gradient contributions without a full-size product graph."""
    if optimized:
        torch.autograd.backward((generated, lexical),
                                (embedding_gradient, lexical.new_tensor(0.5)))
    else:
        ((generated.float() * embedding_gradient).sum() + 0.5 * lexical).backward()
