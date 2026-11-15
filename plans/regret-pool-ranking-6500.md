# Does the regret policy select the hardest training-pool tokenizers?

Freeze the LM, hypernetwork and selection probabilities from the same saved
step-6,500 checkpoint. Score all 1,000 original native tokenizers. The ongoing
20k training job continues independently.

The user explicitly chose to match the **current training reward**: fixed
training-text batches, variable continuations, maximum 128 total positions.
The older 576-passage / 32-scored-token frozen-policy panel is not substituted.

## Fixed inputs

Restore the saved corpus and Python sampling state at full checkpoint 6,000.
Replay only text consumption through 6,500, using the recorded selected actions
to reproduce any rejected examples. Validate every one of the 500 reconstructed
batch hashes against the actual training logs. Do not perform model updates.

Freeze the next 32 recipe-sampled batches of 128 raw text windows: 4,096 initial
examples with all 23 corpus buckets represented. Preserve 2,048-character source
reads and the original first-512-character window. Precompute a common ordered
replacement bank per bucket/batch for the original helper's rejection behavior;
keep its 128-rejection limit. No tokenizer or passage is silently dropped.

Use the first 16 batches for discovery and the remaining 16 for an independent
confirmation estimate. All 1,000 actions are scored on both sets. Draws follow
the original recipe proportions; no equal-domain reweighting is introduced.

## Exact scoring

- Native prefix, canonical continuation and output head, unchanged paired-cut
  helper, min 32 characters, BOS + prefix + variable continuation <=128 positions.
- Recompute the corresponding canonical loss for each action's own paired cuts.
  Cuts/targets can differ between actions, exactly as in training; targets and
  continuation bytes must match between each transferred/canonical pair.
- Original microbatch 32 and full 128-example update; sum token NLL within each
  update, divide by that update's continuation bytes and ln(2).
- Keep hypernetwork dropout active. Use a fixed independent CUDA seed for each
  batch, shared across actions. Generate a fresh embedding table for every batch;
  no deterministic-embedding proxy or dropout averaging shortcut.
- Freeze all LM/HN parameters and load no optimizer. Lexical loss is excluded
  because it is not part of the actual policy reward.
- Rank by the arithmetic mean of the 32 per-update regret rewards. Also report
  the two 16-batch rankings and paired confirmation uncertainty. These are finite
  estimates of a stochastic training reward, not exact population rankings.

## Compute and persistence

Four separate H200 workers on Tillicum account zlab, normal QoS, six-hour wall
limit per worker; no explicit CPU or memory reservations. Exactly 250 actions
per shard. Score higher-policy-probability actions first for useful progress;
every action is included and fixed seeds make order irrelevant to its scores.

Keep all corpus windows, weights and tokenizer files on cloud storage. The
existing shared input bundle/checkpoint is reused without copying large files
through the caller. Save every batch's scores and pairing hash atomically, so
an interrupted worker can resume without excluding or changing measurements.

The dashboard entry is `regret-pool-ranking-6500`, grouped under Evaluations.
Local metadata: modal_results/regret_pool_ranking/.
Cloud data/scores: the same directory under the Tillicum workspace.

## Report

Compare the saved policy with measured regret: dominant tokenizer's estimated
rank; probability on the hardest 1/10/50/100; policy-expected versus uniform
reward; remaining estimated oracle headroom; agreement between discovery and
confirmation; and confirmation difference between the discovery winner and
the dominant policy action. Report uncertainty before diagnosing exploration
failure. Distinguish this training-distribution check from held-out sanity or
OOD generalization.
