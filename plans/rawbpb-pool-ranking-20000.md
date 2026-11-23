# Does the raw-BPB policy select the hardest training-pool tokenizers?

Repeat the step-6,500 full-pool ranking of Qingyun's regret run
(`plans/regret-pool-ranking-6500.md` and `reports/regret-pool-ranking-6500`
on zett-modal-20k) on the raw-BPB run (`reports/raw-bpb-rl-20k-tillicum/`).
Freeze the LM, hypernetwork and selection probabilities from the same saved
step-20,000 full restart. Score all 1,000 original native pool tokenizers on
32 fixed training batches with the exact training reward. No model or policy
is updated.

The user approved the ranking on 2026-11-20 as part of diagnosing the
raw-BPB run. It covers all 1,000 tokenizers on all 32 batches; no subset,
proxy or cheaper design is substituted.

## Question

Does the saved policy put its probability on the tokenizers that are hardest
for the frozen step-20,000 model? Report the favorite's rank among all 1,000,
the policy's mass on the hardest 1 / 10 / 50 / 100, the policy-expected
reward against uniform sampling, the headroom to the estimated hardest
tokenizer, and whether hardness found on the discovery batches holds on the
confirmation batches.

## The user's decisions

- **A fresh panel.** The text is the next 32 batches of 128 windows the run
  would have drawn after step 20,000, not the published step-6,500 panel.
  Both runs read the corpus sequentially with the same seed-42 bucket draws,
  so the raw-BPB run trained on the published panel's text at its updates
  6,501-6,532. The fresh panel is not all unseen either: wiki-ja and wiki-zh
  had wrapped once by step 20,000, so their windows (91 of the 4,096 items)
  are second readings. The report states this.
- **Both rankings.** Gap (transferred minus canonical BPB, which the
  step-6,500 ranking calls regret) stays the primary ranking, so ranks
  compare with that chart. Raw BPB (transferred BPB), this run's training
  reward, is reported beside it with the same comparisons.

## Fixed inputs

- **Checkpoint.** The run's `step-020000` full restart. The preparation
  checks `state.pt`, `policy.pt`, `lm.pt` and the hypernetwork files against
  the checkpoint manifest and requires the saved step and policy update
  counts to equal 20,000. `policy.json` holds the softmax of the saved
  logits, the pick counts and the hash of `policy.pt`.
- **Code.** The staged tree `.raw-bpb-code-d875965262f96b29` that the run
  trained with, checked file by file against its `source-manifest.json`
  (SHA-256 `c4714e6e...`, also the staged-code hash of the step-6,500
  ranking). The corpus reader, the paired-example helper and the tokenizer
  pool come from it.
- **Panel.** Restore the corpus stream and Python RNG saved at step 20,000
  and, with no replay, draw 32 recipe-sampled batches of 128 windows: 4,096
  initial examples covering all 23 corpus buckets, 2,048-character reads cut
  to their first 512 characters. Then draw one ordered 128-window
  replacement bank per bucket and batch. Batches 0-15 are for discovery and
  16-31 for confirmation; batch i uses dropout seed 2026111500 + i. Panel
  SHA-256 `87d13112de30d08394afc2985bf25d6aab5bd9e73b0a3bfc6951234681307c97`.
- **Cross-check.** Restore the step-18,000 full restart and replay the data
  stream of updates 18,001-20,000: corpus draws, Python RNG and each
  update's paired examples for its logged tokenizer, with no model updates.
  All 2,000 logged paired-batch hashes and rejected-example counts matched
  (7 rejected examples), and the replayed stream drew an identical panel.
- **Freezing.** `run-plan.json` records the scorer's SHA-256 and gives the
  input, checkpoint and code folders relative to the state folder, so no
  absolute path is frozen. `panel-manifest.json` is written last, only once
  its bytes match the hash in `run-plan.json`, and never over an existing
  one.

## Exact scoring

As in the step-6,500 ranking:

- Native prefix, canonical continuation and output head, the unchanged
  paired-cut helper, min 32 characters, BOS + prefix + variable
  continuation <= 128 positions.
- A canonical control recomputed for each action's own cuts. Targets and
  continuation bytes must match within each transferred/canonical pair.
- Microbatch 32 and the full 128-example update. Token NLL is summed over
  each update and divided by its continuation bytes and ln 2; the ranking
  metric is the arithmetic mean of the 32 per-update values.
- Hypernetwork dropout active with the batch's fixed CUDA seed, shared
  across actions, and a fresh embedding table for every batch.
- All LM and hypernetwork parameters frozen, no optimizer loaded, lexical
  loss excluded.
- A rejected item takes the next window of that bucket's replacement bank,
  within the original 128-rejection limit. No tokenizer or passage is
  dropped.
- Paired bootstrap over batches: 10,000 draws, seed 20261115.

## What differs from the step-6,500 ranking

- **Model.** Another run (raw-BPB reward, not regret) at another checkpoint
  (20,000, not 6,500), so absolute values do not compare.
- **Panel.** Fresh, drawn from the step-20,000 stream, as decided above.
- **Metrics.** Raw BPB is reported beside the gap ranking.
- **Scorer.** `scripts/score_rawbpb_pool_ranking.py` is his
  `score_training_pool_ranking.py` (5f431ea) with the state, input,
  checkpoint and code paths and the step as arguments that must match
  `run-plan.json`, the staged code tree instead of a new copy, step 20000,
  and a `--max-tokenizers` smoke stop. The scoring computation is
  unchanged.
- **Preparation.** `scripts/prepare_rawbpb_pool_ranking.py` is adapted from
  his `prepare_training_reward_panel.py` (5f431ea). It keeps his draw and
  replay code, but draws the panel directly from the stream saved at step
  20,000 and uses the replay from step 18,000 only as a cross-check. Its
  `--panel published` mode would copy his panel unchanged after a hash
  check.
- **Report.** `scripts/report_rawbpb_pool_ranking.py` ports his
  `report_training_pool_ranking.py` (5f431ea). It adds the raw-BPB
  comparisons, the comparison with the step-6,500 ranking (`--reference`),
  a 32-batch bootstrap of policy minus uniform, the hardest 10 / 50 / 100
  shared between the metrics and between the halves, pick counts, the
  bottom group's rank blocks and the gap Spearman outside that group. It
  also checks the step, policy and run plan against each other, refuses a
  reference from another step or pool, and withholds `manifest.json` while
  a report file holds an absolute path, an email address or the account
  name.
- **Account.** `stf` instead of `zlab`.

## Smoke check

After the preparation, job 503015 scored the first tokenizer of shard 0
(`--max-tokenizers 1`; stage `smoke_stop`, never `complete`). Shard 0 starts
with the highest policy probability, so this was the favorite #824: raw
0.9067 / canonical 0.8720 / gap 0.0347 BPB, against the training log's
0.9078 / 0.8717 / 0.0361 on #824's 396 updates in 19,501-20,000. Shard 0 of the array
reused this score.

## Compute

Tillicum, account `stf`, partition `gpu-h200`, one H200 per job, no explicit
CPU or memory reservations (each job got the default 4 CPUs and 120G).

| Job | Work | QoS, limit | Elapsed |
|---|---|---|---|
| 503015 | preparation, cross-check and smoke test | debug, 1 h | 0:20:25 |
| 503041_0..3 | scoring, 250 tokenizers per task | normal, 8 h | 3:41:19 / 3:47:44 / 3:43:47 / 3:49:25 |

The array was submitted after the preparation job ended. Tasks 0 and 1 ran
on 2026-11-20; tasks 2 and 3 waited for priority and ran on 2026-11-21. All
four completed 250 of 250 tokenizers. Total 15.38 GPU-hours, $13.84 at
$0.90 per GPU-hour (about $16 was approved).

`scripts/run_rawbpb_pool_ranking.py` watched the jobs from the local machine
over the existing SSH connection. It is monitor-only: it never submits,
cancels or changes anything on the cluster. When all four tasks completed it
copied the small result files home and verified their hashes.

## Persistence

- **Cloud data.** HF `Dekiofsa/RLtokenizers`, folder
  `raw-bpb-rl-20k/diagnostics/policy-ranking-step-020000/`, added
  2026-11-22 at revision `2533fc5660f63ac5e571661466acdcf6cf80ee35`:
  `all-scores.json.gz`, `training-panel.json.gz` (30,425,174 B),
  `panel-manifest.json`, `policy.json`, `run-plan.json`,
  `checkpoint-manifest.json`, `preparation-status.json`, and `records/`
  (logs, status, submissions and the as-run code). 23 files, none
  mismatched when checked after the upload. `report/`, a copy of the 13
  files of `reports/rawbpb-pool-ranking-20000/`, was added at revision
  `60670ac8cea5b53695ec43db8160c86856c2b70b`, also verified.
- **Cluster working folder.** The jobs ran in
  `/gpfs/projects/stf/<user>/anytok-runs/rawbpb-pool-ranking-20000/` on
  Tillicum. It was deleted on 2026-11-22, after its data was saved on
  Hugging Face.
- **Local metadata.** `modal_results/rawbpb_pool_ranking_20000/`
  (gitignored): the observer's `status.json`, `events.jsonl` and
  `controller.log`, and the copied results.
- **Report.** `reports/rawbpb-pool-ranking-20000/` and
  `plots/rawbpb_pool_ranking_20000.{png,pdf}`.
- **Dashboard.** Entry `rawbpb-pool-ranking-20000`, category Evaluations.

## Report

The report script reads the copied results (defaults
`--state modal_results/rawbpb_pool_ranking_20000` and `--step 20000`).
`--reference` is a folder holding the step-6,500 report's
`all-scores.json.gz`, `pool-index.json`, `policy.json` and
`panel-manifest.json`, filled from git for each NAME:

    git cat-file blob origin/zett-modal-20k:reports/regret-pool-ranking-6500/NAME > FOLDER/NAME

For each metric it reports the favorite's rank, the policy's mass on the
hardest 1 / 10 / 50 / 100, the policy-expected, uniform and maximum values
with the oracle headroom, policy minus uniform with its 32-batch bootstrap,
the discovery winner against the favorite on the 16 confirmation batches,
and rank stability between the halves. It then compares the gap ranking
with the step-6,500 ranking and lists the run's nine most-picked tokenizers.

The port reproduced the step-6,500 report from his inputs (CSV
byte-identical). An independent recompute on 2026-11-21 matched every
number; the additions made after that check changed no earlier value.
Tests: `tests/test_rawbpb_pool_ranking.py`, on synthetic scores.

Results, all in the report's README:

- The favorite #824 holds 97.89%. It ranks 633 by gap and 653 by raw BPB.
- The policy is no better than uniform sampling: gap +0.00056
  [-0.00022, +0.00134], raw BPB -0.00018 [-0.00103, +0.00065] (32-batch
  bootstrap 95% intervals).
- The hardest tokenizer by gap is #116 (Unigram, regex); by raw BPB, #127.

These are finite estimates of a stochastic training-distribution reward,
not exact population ranks or a held-out generalization test. Close ranks
are not decisive.
