# Fresh 20k joint training with the hardness predictor

The user requested starting the real run on Tillicum and explicitly selected
**a fresh 20k run from the original initialization**. Start with the original
SmolLM2-1.7B LM and original identity-warmed HN, seed42, and a fresh settings
predictor, seed43. The previous regret run remains paused at8k.

The selector is the unchanged frozen-pilot predictor: 18→64→64→1 Tanh,
float64 CPU, Adam0.001, one single-observation MSE update on paired pre-update
transferred BPB minus canonical BPB. Every fourth update visits the next entry
of a shuffled full-pool permutation. Other updates choose the largest predicted
gap, with random ties. All1,000 tokenizers remain eligible. There is no entropy
bonus, warmup, replay buffer, or new tuned setting. Exactly five coverage visits
per tokenizer occur over20k updates.

Reuse the exact corrected paired scorer and optimized GPU memory execution of
the previous Tillicum run. Preserve the23-bucket corpus, fresh128-example batch,
microbatch32, variable canonical continuation, reversible cuts and128 total
positions including BOS. Full native vocabularies and priors remain intact.
Keep the same LM/HN AdamW groups, gradient clipping, .5 lexical term, canonical
input/output freezing and original190k-step LR schedule without rescaling.

The trainer loop is byte-identical to the existing raw/regret runner. Source
staging changes only the selector import and selector provenance/description.
The adapter exposes actual conditional probabilities at selection time and
stores regression predictions without inventing softmax logits. Monitoring
uses actual recent selection frequencies and labels them accordingly.

Use one Tillicum H200, accountzlab, normal QoS,24h wall limit and a five-minute
graceful-stop signal. Request no explicit CPU or RAM. Stage into a separate root
`/gpfs/scrubbed/qxli2/anytok-hardness-20k`, referencing the verified existing
environment and input files in cloud storage. Do not copy data through the local
caller or change the old run. Check all original input checksums before submit.

Before full training, run three real updates uninterrupted, then two plus an
independent-process resume through update three. Require matching scientific
metrics, paired hashes/counters and LM/HN/predictor weights. Abort before full
training on any mismatch or fit failure. Verify the first three actual fresh
training updates against preflight too. No proxy tokenizer or synthetic text.

Save weights every500 and full optimizer/corpus/RNG/selector restarts every2k,
at the final step, and on graceful stop. Verify and upload checkpoints directly
from Tillicum to the public model `cminst/anytok-hardness-predictor-20k` under
`checkpoints/step-NNNNNN`. Publish latest-restart metadata only after cloud hash
verification. A graceful walltime stop can submit the next dependent job;
training failures cannot. A new-run stop-request file prevents continuation.

Dashboard:`anytok-hardness-training`. Metadata/log mirror:
`modal_results/tillicum_hardness/`. Plot the identical two fixed sanity-tokenizer
scores every500 steps and real selection coverage/repetition. Keep the full
evaluation panels unchanged for later evaluation; no extra evaluations are
substituted for or silently removed from an agreed design.

## Launch and float32 verification

Initial job485560 stopped in preflight before the main run. All three training
metric dictionaries, paired batches, LM weights and predictor weights matched
exactly across uninterrupted versus resumed execution. One value in
`model.token_type_embeddings.weight` differed by9.313225746154785e-10, one
representable float32 step. Its relative difference versus that tensor's scale
was1.24e-8. No scoring, optimizer, data or training math was changed.

Job485615 performs an additional independent three-update GPU repeat. Input
and score checks remain exact; LM and predictor weights remain bitwise checks.
The HN weight comparison explicitly records and bounds every difference to at
most one adjacent float32 value. Shapes, names, dtypes and finite values must
match. Larger differences still abort. Original failed preflight artifacts are
preserved and hash-verified. The full fresh20k run starts only after these
checks pass. Exact preflight/launch receipts are mirrored to the local run state.

Atstep500 the payload upload and all cloud hash checks completed, but the
derived dashboard-summary import failed because the script directory rather
than the checkout root was on Python's import path. The model files were safe.
The active process was repaired with an import-path symlink and the summary was
rebuilt from the checksum-verified small policy checkpoint, without restarting
training. Future allocations explicitly add the root tosys.path, and verified
payloads can retry summary creation independently. A regression test covers
metadata repair without a duplicate upload and without duplicate snapshots.
