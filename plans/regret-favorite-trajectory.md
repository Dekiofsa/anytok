# Does training improve the policy's favorite tokenizer?

Evaluate original pool tokenizer #771 (WordPiece, no regex, native 127k vocab)
at every available checkpoint on the actual committed regret-training trajectory
through the user's paused step 8,000. Include saved handoff checkpoints as well
as the 500-step grid. Exclude the discarded Modal continuation after step 5,288;
use the resumed Tillicum checkpoints from 5,500 onward.

Run the existing local step-2,000 copy first to validate the full workload,
then prioritize the first/last snapshots (500 and 8,000), followed by every
remaining snapshot. Order changes no scores: text and seeds are fixed, and no
checkpoint is omitted.

The user explicitly requested local scoring. Reuse the previously approved exact
local input/tokenizer copy and download diagnostic model-weight copies as needed,
retaining cloud originals. Stage at most one additional checkpoint at a time;
remove only this diagnostic's newly downloaded weight copies after scoring.
Never download optimizer state for inference. Early checkpoints remain in their
persistent Modal Volumes and newer ones on Hugging Face. No new cloud GPU jobs.

Use the identical frozen 32 × 128 raw examples and replacement banks from the
step-6,500 all-pool ranking, with the exact paired helper, variable canonical
continuations and 128-total-position cap. All 23 training corpus buckets are
represented. Score every batch at every checkpoint. This is a training-distribution
diagnostic; the text is not claimed to be held out from later training.

Keep LM/hypernetwork weights frozen, training dropout active and the same
independent CUDA seed for each batch across checkpoints. Generate a fresh full
embedding table for every batch, matching the ranking/training protocol. Use
batch128/microbatch32 and the validated GPU-only memory execution. No tokenizer
truncation. Paired token targets, bytes, contexts, counts and hashes must match
across every checkpoint; canonical scores are recomputed for each saved LM.

Average the per-update byte-weighted rewards equally, as training does. Plot
absolute transferred/canonical BPB, paired regret, paired change from the first
saved checkpoint, and policy probability/recent selection frequency. Bootstrap
the same 32 batch indices for uncertainty. Plotting uses the established Manrope
font and repository colors. Keep individual batch scores, source hashes, exact
checkpoint lineage and CSV/JSON/PNG/PDF outputs. Cross-device bitwise equality
against the H200 ranking is not assumed; all trajectory points run on the same
local RTX PRO 6000 with pinned torch2.15.0.

Dashboard run: `regret-favorite-trajectory`, Evaluations. Local state:
`modal_results/regret_favorite_trajectory/`. Final report:
`reports/regret-favorite-trajectory/`.
