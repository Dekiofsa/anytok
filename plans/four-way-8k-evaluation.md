# Four training methods at step 8,000

Status: approved and launched on 2026-11-18 UTC. Training remains stopped.
Tillicum H200 evaluation array: 492775, one frozen checkpoint per task.
All four step-8,000 checkpoint weight hashes verified before submission.
All 576 passages passed all 62 encoders without changing any prefix or target.
Frozen machine-readable specification: `plans/four-way-8k-v1.json`.
Dashboard: `four-way-8k-evaluation`.

Compare the saved LM and hypernetwork together at exactly step 8,000:

| Label | Run |
| --- | --- |
| Uniform | Original uniform sampling from the 1,000-tokenizer pool |
| Raw-BPB RL | Original shared-settings REINFORCE run, entropy weight 0.01 |
| Regret RL | Paired BPB-gap reward, REINFORCE, entropy weight 0 |
| Hardness predictor | Paired BPB-gap predictor with every fourth choice covering the shuffled pool |

Training stays stopped. This proposal does not resume either paused run.

## Tokenizers and text

Reuse the exact 50 uniformly sampled IDs in
`plans/corrected-pool-check-v1.json`. These were drawn without replacement
with seed 20261117, before this comparison. No rerolling, exclusions based on
loss, or replacement of difficult encoders. The sample already includes old
worst #0635. It represents previously seen tokenizer files, evaluated on
held-out text; it is not a test of unseen tokenizer files.

Add all 11 saved held-out 48k BPE tokenizers: Base, Code50, Papers25, Math50,
QA25, Books25, Medical40, Legal40, Spanish35, Chinese35 and Japanese35.
Also include the existing base 64k tokenizer. Verify its exact file and pool
nonmembership before calling it held out. These 12 cases are a fixed panel,
not a random sample of unseen tokenizer families. In particular, the 11
specialty tokenizers all use BPE and line256 pretokenization.

Use the existing corrected broad-text passages from
`corrected-pool-check-v1/`: 288 original screen documents and 288 original
confirmation documents, each with 32 documents from each of nine domains:
English web, code, math, medical, legal, Chinese, Japanese, Arabic and Hindi.
These passages were not filtered by model weakness or context benefit.
Keep the two splits separate in the report and give their human-readable
labels as text sample A and text sample B. Explain their historical names
only in provenance. Do not substitute the selected 229-passage stress set.

All 62 tokenizers and all four models score every passage. Expected scope:
248 model/tokenizer pairs and 142,848 model/tokenizer/passage scores.

## One explicit scoring protocol

Every example is **BOS + unscored prefix + 32 scored canonical tokens**.
The whole input fits within 128 positions for both canonical and transferred
inputs. This is 32 scored tokens, not 128 scored tokens, and differs from
the variable continuation lengths used during joint training.

Keep the corrected cuts, continuation IDs and continuation bytes fixed.
Audit the common prefix against all 62 encoders before GPU scoring. If an
added held-out encoder needs a shorter prefix, trim only the prefix's left
edge, identically for every tokenizer and model; preserve valid canonical
boundaries, all 32 continuation tokens and all document IDs. Record original
and final prefix hashes and lengths. If no valid common prefix exists, stop
preparation and report the problem; do not discard examples or tokenizers.

Only native prefix embeddings come from the hypernetwork. BOS, continuation
input embeddings, continuation token IDs and output head remain canonical.
Load each checkpoint's actual trained LM backbone. Evaluate without dropout.
Generate embeddings for the complete native vocabulary with original priors.

Before comparing results, require identical source IDs, prefix text, target
IDs, target UTF-8 bytes and scored counts across all models and tokenizers.
Each split scores 9,216 continuation tokens per model/tokenizer pair. With
unchanged targets, continuation byte totals remain 27,039 and 26,668. Audit
canonical joint encoding across the cut, position bounds and loss masking.
Check the explicit causal loss against the model's built-in loss.

## Results and plots

For each model report canonical BPB, transferred BPB and their difference.
Compute BPB by summing loss and bytes within each domain, then weight the nine
domains equally. Preserve per-passage losses and domain scores. Show A and B
separately; do not let multilingual cases hide English-domain results.

Make four plots using the established plotting style:

1. All 62 tokenizers against all four models, with a shared color scale for
   BPB gaps and a clear divider between training-pool and held-out tokenizers.
2. The distribution across the random 50: mean, median, 90th percentile and
   the full individual points, so average improvements cannot hide a bad tail.
3. Changes relative to Uniform by text domain and tokenizer algorithm,
   pretokenizer, vocabulary size and merge noise. Keep small groups visibly
   labeled with their counts; these are associations, not isolated causes.
4. Actual transferred BPB alongside canonical BPB, separating changes in LM
   strength from changes in the cost of tokenizer transfer.

Use paired document-cluster bootstrap draws within domains and retain the
same draws across models. For an estimated full-pool mean, additionally
resample the random tokenizer IDs with finite-population correction. Keep
the 12 held-out cases separate from that population estimate. Intervals do
not cover variation from retraining with a different seed.

The key question is whether the predictor reduces average gaps and the bad
tail without increasing absolute BPB or moving failures into other domains.
This measures the resulting model's strengths and weaknesses. Fifty pool
tokenizers cannot establish the worst of all 1,000 or prove the selector
always finds the hardest choice.

## Checkpoint inventory and interpretation

Read-only inspection found both Regret RL and Hardness predictor full restart
manifests at step 8,000 on Tillicum. The stored Uniform inventory contains
step 8,000 on data-pipelines and the gathered gildmember volume. Raw-BPB's
completed-run metadata identifies the gathered checkpoint location as
neural-net-1411 / anytok-pool-rl-20k / settings-run. Live file verification for
Uniform and Raw-BPB remains to be done before execution.

Pin all four weight-file hashes, run configurations, tokenizer files, prepared
panel, evaluator source and runtime in the final machine-readable specification
before inference. Use cloud-to-cloud artifact transfers; only metadata and
results pass through the local caller. Reuse the existing scorer after auditing
the shared panel; do not reuse scores from incompatible old boundary protocols.

Equal step counts make this a useful comparison of actual runs, but not a
selector-only ablation: the original Uniform and Raw-BPB runs predate the
corrected training cuts, and the two REINFORCE runs have different entropy
settings. State those differences prominently when interpreting results.

Once the proposal is accepted, register one evaluation run in
`plans/run-dashboard.json`, including preparation and all 248 scoring tasks,
platform, logs and checkpoint provenance. Prepare and verify all cloud inputs
before requesting GPUs. Use the validated runtime and one common GPU type for
all four models; no explicit CPU or memory reservations. Confirm placement and
an observed timing estimate before committing a long allocation. Do not queue
training as part of this evaluation.
