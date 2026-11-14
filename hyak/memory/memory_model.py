"""Peak-memory model of one training update (predictions only; DESIGN.md 5.C/5.D).

This module is deliberately separate from :mod:`hyak.memory.chunked_hypernet`: the SHA-256
of that file is part of every run's identity (``hyak_segment.json``, resume checks), so its
text must not change between the uniform bootstrap and update 20,000.  The prediction below
is documentation and may be corrected after the first probe without blocking a resume.
Nothing here changes what training computes.

Unit ``U(n) = n * 11 * W * 2`` bytes: one bf16 hidden tensor of ``n`` vocabulary rows
(W = 2048, 10 pieces + 1 prior position: 45,056 bytes per row; U(16,384) = 0.69 GiB,
U(128,000) = 5.37 GiB).

Fixed state ``F = 28.8 GiB`` once the AdamW moments exist (LM weights 6.75, LM gradients
6.00, hypernetwork weights 1.28 and gradients 0.91, AdamW moments 13.82 GiB; the very first
update peaks 13.82 GiB lower because ``optimizer.step()`` allocates the moments after it).

One update has two memory phases; the peak is the larger of the two:

1. LM micro-batches (T:552-567).  The hypernetwork forward has run; alive are the chunk
   boundaries, the down-attention graphs, the output-projection graph and ``generated``
   (about 8.06 U(V) for the chunked path), ``leaf.grad`` (0.18 U(V)), and one 32 x 128
   micro-batch through SmolLM2-1.7B: activations (about 11.6 to 13.5 GiB), the bf16 weight
   copies autocast keeps in the graph (3.0 GiB) and logits/cross-entropy tensors (about
   2 GiB), i.e. 17 to 21 GiB.  The hypernetwork gradients are not allocated yet
   (``zero_grad(set_to_none=True)``), so the fixed part is F - 0.91 = 27.9 GiB::

       LM phase ~ 27.9 GiB + 8.2 U(V) + 19 GiB (17 to 21)        (chunked)
       LM phase ~ 27.9 GiB + 6.0 U(V) + 19 GiB (17 to 21)        (pinned forward)

   ``HYAK_HN_CHUNK_ROWS`` does not change this phase.

2. Hypernetwork backward (T:577).  The V-proportional tensors above plus the recomputed
   graph of one checkpoint region at a time (the largest is stage (c), up attention +
   feed-forward of one chunk: 12.9 U(chunk) saved plus 2 to 4 U(chunk) of gradients)::

       chunked:  ~ 29.8 GiB + 7.8 U(V) + 16 U(r),   r = min(chunk_rows, V)
       pinned:   ~ 28.8 GiB + 31.6 U(V)             (31.6 = (72.2 - 28.8) GiB / U(32,768),
                                                     the measured original B300 peak)

Measured basis: the pinned anchor (72.21 GiB ``max_memory_allocated`` at V = 32,768, original
run); CPU float32 census of the chunked and pinned graphs (width 256); a real-width mps
census of the hypernetwork (7.35 U(V) + 14.7 U(chunk), scaled by the anchor to 8.0 / 16.1);
a census of a random LLaMA with SmolLM2's layer shape for the LM phase.  With 16,384-row
chunks the hypernetwork phase is already below the LM phase at every vocabulary size, so
smaller chunks do not lower the peak (and ``install`` refuses chunks below 4,096 rows).
These are predictions; ``python -m hyak.memory.probe`` measures whole updates on the GPU.
"""
from __future__ import annotations

GIB = 2 ** 30

#: Coefficients of the model above (real sizes: width 2048, 10 pieces + 1 prior position, bf16 autocast).
PREDICTION = {
    'width': 2048,
    'positions': 11,
    'unit_bytes_per_row': 11 * 2048 * 2,
    'fixed_state_gib': 28.8,
    'adamw_moments_gib': 13.82,
    'hypernetwork_gradients_gib': 0.91,
    # hypernetwork backward
    'chunked_overhead_gib': 1.0,
    'chunked_vocab_units': 7.8,
    'chunked_chunk_units': 16.0,
    'pinned_vocab_units': 31.6,
    'relative_uncertainty_of_activations': 0.10,
    # LM micro-batches
    'lm_chunked_vocab_units': 8.2,
    'lm_pinned_vocab_units': 6.0,
    'lm_microbatch_gib': 19.0,
    'lm_microbatch_low_gib': 17.0,
    'lm_microbatch_high_gib': 21.0,
    'anchor': 'pinned 72.2 GiB max_memory_allocated at V = 32,768 (original B300 run, '
              'qingy2024/anytok-zett-smollm2-20k run/events.jsonl)',
}

PHASE_HYPERNETWORK = 'hypernetwork backward'
PHASE_LM = 'LM micro-batches'


def _check(vocab, chunk_rows) -> tuple:
    if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab < 1:
        raise ValueError(f'vocab must be a positive integer, got {vocab!r}')
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 0:
        raise ValueError(f'chunk_rows must be a non-negative integer, got {chunk_rows!r}')
    return vocab, chunk_rows


def unit_gib(rows: int) -> float:
    """``U(rows)`` in GiB."""
    return rows * PREDICTION['unit_bytes_per_row'] / GIB


def predict_peak_gib(vocab: int, chunk_rows: int, *, first_update: bool = False) -> dict:
    """Predicted peak CUDA memory (GiB) of one training update on the real sizes.

    ``chunk_rows <= 0`` predicts the pinned forward.  ``first_update`` drops the AdamW moments.
    Returns ``{'gib', 'low_gib', 'high_gib', 'phase', 'hypernetwork_gib', 'lm_gib', 'mode',
    'fixed_gib', 'vocab_units', 'chunk_units', 'lm_vocab_units'}``; ``gib`` is the larger phase
    and ``phase`` names it.
    """
    vocab, rows = _check(vocab, chunk_rows)
    p = PREDICTION
    fixed = p['fixed_state_gib'] - (p['adamw_moments_gib'] if first_update else 0.0)
    lm_fixed = fixed - p['hypernetwork_gradients_gib']
    if rows <= 0:
        mode, vocab_units, chunk_units, lm_units = 'pinned', p['pinned_vocab_units'], 0.0, p['lm_pinned_vocab_units']
        hn_fixed = fixed
    else:
        mode = 'chunked'
        vocab_units, chunk_units, lm_units = p['chunked_vocab_units'], p['chunked_chunk_units'], p['lm_chunked_vocab_units']
        hn_fixed = fixed + p['chunked_overhead_gib']
    activations = (vocab_units * vocab + chunk_units * min(rows, vocab)) * unit_gib(1)
    spread = p['relative_uncertainty_of_activations'] * activations
    hn = (hn_fixed + activations - spread, hn_fixed + activations, hn_fixed + activations + spread)
    lm_vocab = lm_units * unit_gib(vocab)
    lm_spread = p['relative_uncertainty_of_activations'] * lm_vocab
    lm = (lm_fixed + lm_vocab - lm_spread + p['lm_microbatch_low_gib'],
          lm_fixed + lm_vocab + p['lm_microbatch_gib'],
          lm_fixed + lm_vocab + lm_spread + p['lm_microbatch_high_gib'])
    phase = PHASE_HYPERNETWORK if hn[1] >= lm[1] else PHASE_LM
    return {'mode': mode, 'phase': phase,
            'gib': round(max(hn[1], lm[1]), 1), 'low_gib': round(max(hn[0], lm[0]), 1),
            'high_gib': round(max(hn[2], lm[2]), 1),
            'hypernetwork_gib': round(hn[1], 1), 'lm_gib': round(lm[1], 1),
            'fixed_gib': round(fixed, 2), 'vocab_units': vocab_units, 'chunk_units': chunk_units,
            'lm_vocab_units': lm_units}


def prediction_table(vocabs=(32768, 64000, 96000, 128000), chunk_rows=(0, 32768, 16384, 8192)) -> list:
    """Rows ``{'vocab', 'chunk_rows', 'gib', 'phase', ...}`` for the run guide."""
    return [{'vocab': vocab, 'chunk_rows': rows, **predict_peak_gib(vocab, rows)}
            for vocab in vocabs for rows in chunk_rows]


def main(argv=None) -> int:
    """``python -m hyak.memory.memory_model``: print the prediction table (no GPU, no inputs)."""
    import argparse
    parser = argparse.ArgumentParser(prog='python -m hyak.memory.memory_model', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    for row in prediction_table():
        setting = 'pinned' if row['chunk_rows'] == 0 else f'chunk {row["chunk_rows"]}'
        print(f'V={row["vocab"]:>7,} {setting:>12}: {row["gib"]:6.1f} GiB ({row["low_gib"]}-{row["high_gib"]}; '
              f'{row["phase"]}; hypernetwork {row["hypernetwork_gib"]}, LM {row["lm_gib"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
