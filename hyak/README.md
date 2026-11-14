# AnyTok on Hyak Tillicum: run guide

This directory is a port of the AnyTok joint-training experiment from Modal to the Slurm cluster
Hyak **Tillicum** (NVIDIA H200, 141 GB). It reruns two 20,000-update runs and evaluates both:

* the **uniform** run (a pool tokenizer drawn uniformly per update; the original uniform trainer), and
* the **RL** run (a shared-settings policy picks the tokenizer): since the user's decision of
  2026-11-14 it is the counterpart of Qingyun's joint regret RL20k with the **straight BPB loss as the
  policy reward** (corrected paired token boundaries, a canonical control forward, entropy 0; D8),

starting from pretrained SmolLM2-1.7B with a **randomly initialized hypernetwork and policy**, on a
corpus and a 1,000-tokenizer pool regenerated on the cluster from the repository's plans and public
Hugging Face data (or taken from the original owner, section 2.2). Then it runs the repository's
corrected evaluation for both final checkpoints (scope: section 1).

The pinned trainers under `runtime/` are not changed. Everything new lives under `hyak/`, including
the RL trainer variant (`hyak/variants/anytok_pool_straight_rl_20k`: Qingyun's regret trainer with
one line of the policy update and the reward label changed). Nothing here has run on Tillicum yet:
section 12 lists what was checked on a laptop and what only the cluster can show. Read sections 1
and 2 before you submit anything: section 2 holds the decisions that are expensive to reverse.

Contents

1. What is reproduced, the forced deviations (D1 to D7) and the RL protocol (D8)
2. Decisions before the expensive steps
3. Prerequisites
4. One-time setup (login node)
5. The pipeline at a glance
6. Step by step
7. Watching jobs and the dashboard
8. Chains: stop, resume, extend, repair
9. Storage budget, the 60-day purge and keeping the checkpoints
10. Fidelity reports and how to read them
11. Bringing original artifacts instead of regenerating them
12. What was verified locally, and what only the cluster can prove
13. Troubleshooting
14. Reference

All commands run from the repository root of your checkout on the cluster. Numbers marked
*estimate* were not measured on Tillicum. Billing on Tillicum is 0.90 USD per GPU-hour; the
first 100 GPU-hours are free. Every program of the port ends with the same exit-code contract
(table in section 14): 0 ok, 1 crash, 2 usage or setup problem, 3 a stop that needs your decision,
4 a check failed, 75 not now.

---

## 1. What is reproduced, the forced deviations and the RL protocol

The **uniform run is the original uniform run**, reproduced unchanged:

* The pinned uniform trainer (`runtime/anytok_pool_20k`), byte for byte, imported in-process exactly
  as the Modal launcher did, with the original example cuts.
* Its trainer arguments, seed 42, micro-batch 32, learning-rate schedule, optimizer, batch
  construction, the checkpoint grid (every 500 updates; full restarts at multiples of 2,000) and the
  final 40-checkpoint audit.
* The tokenizer and bucket schedule: the seed-42 schedule of all 20,000 pool draws is replayed and
  compared with the original run's table (section 6.8).

The **RL run follows Qingyun's new protocol with the straight reward** (D8;
`plans/anytok-pool-regret-rl-20k.md`), not the original RL run:

* Its trainer is `hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py`,
  Qingyun's joint regret RL20k trainer (`runtime/anytok_pool_regret_rl_20k`, pinned at commit
  `3de64a5`) with exactly two lines changed: the policy receives the paired, pre-update transferred
  BPB (`train_bpb`) instead of the paired gap, and `run_config.json` says so
  (`policy_reward: paired_pre_model_update_transferred_bpb`). `straight_reward.patch` there is the
  exact diff.
* Everything else is the regret run's: the corrected paired token boundaries for training batches
  and the sanity evaluation, a no-grad canonical forward per update on the pre-update LM (its RNG
  consumption restored; its result is now only logged, as `paired_gap_bpb`), the paired checks and
  logging, the staged code layout of the regret launcher (`settings_policy.py` of
  `runtime/anytok_pool_settings_rl_20k`, `paired_reward.py` of the regret snapshot), and its trainer
  arguments: `--policy-entropy 0 --policy-lr 0.001 --policy-baseline-decay 0.99` passed explicitly,
  policy seed 43, LM and hypernetwork seed 42, micro-batch 32, the same learning-rate schedule,
  optimizers, checkpoint grid and audit as the uniform run.
* The original RL trainer (`runtime/anytok_pool_settings_rl_20k`: entropy 0.01, raw training-BPB
  reward, original cuts) stays pinned and unchanged but is no longer run.

Both runs:

* The evaluation data: the stored 576 + 2,032 passages are rebuilt from stored token ids and
  hash-checked; all 1,000 pool tokenizers are scored with the original protocol, for both final
  checkpoints alike.

Not reproduced (and cannot be): the original warmup hypernetwork, the original sanity tokenizers,
and therefore the original weights and curves; the original RL run itself (D8). The original corpus
and pool are regenerated unless you obtain them (section 2.2).

### Scope of the evaluation

The repository holds eleven report families. The port reproduces two of them for both final
checkpoints, and packs them under the names `uniform20k-hyak` and `rl20k-straight-entropy0-hyak`:

* the **complete-pool table** (576 passages x 1,000 tokenizers, canonical and BOS-only controls),
  the counterpart of `reports/frozen-policy-search-20k`;
* the **natural panel** (2,032 passages x 38 tokenizers), the counterpart of
  `reports/natural-model-comparison-20k`.

Not rerun: the random-50 panel, the behaviour-transfer and trajectory reports, the passage and
explainer reports, and the policy search on the reward table (it can be run later on the new
table, section 6.15). The uniform checkpoint has no original complete-pool score to compare with.
The stored original `rl20k` numbers come from a different run (entropy 0.01, raw reward, original
cuts, warmup initialization): they are a reference, never a reproduction target (6.15).

### The forced deviations, and D8

D1 to D7 are forced by the cluster or by missing artifacts. D8 is not forced: it is the user's
decision of 2026-11-14, the explicit approval `AGENTS.md` asks for.

| | What changes | Why it is forced | Evidence |
|---|---|---|---|
| D1 | The hypernetwork starts from a random initialization built once by `python -m hyak.model.init_hypernet`: `build_input_hypernet(49152, 2048)` under seed 42; the rescalers calibrated on the SmolLM2 input table with the library's zero prior bias (`initialize_rescalers`, the original construction), checked against the public checkpoint (1e-5) and copied from it bit for bit; then the prior-projection bias is drawn from N(0, 0.02) with a separate generator (seed 1042); saved before byte-fallback expansion. Both runs load this one directory. Alternatives: section 2.1. | The original warmup checkpoint is private. The original author approved a random start. With the library's all-zero prior bias, BPE and WordPiece tokenizers (660 of 1,000) feed an exactly zero vector into a LayerNorm and its gradient swamps the joint 0.1 gradient clip. The original warmup training itself (see evidence) is not rerun. | Recomputing the input rescaler from the SmolLM2 table reproduces the public step-500 checkpoint to 6e-8 / 1e-8 and the output scaler to 4.9e-7, so the original did calibrate. The init directory's `config.json` without the line `"hn_byte_fallback_symbols": null,` hashes to the original warmup config (864a6087...). The public checkpoint shows that the original warmup ran `identity_warmup_step` with the library defaults (peak 3e-4, linear ramp, zero priors, one-piece forms) for 10,000 steps from this seed-42 construction; its batch size is unknown. With a zero bias the prior-bias gradient norm was 1.4e6 against about 9e2 for all other parameters together; the init tool's own diagnostic gives a ratio of 2,508.6 with bias std 0 and 0.115 with std 0.02. |
| D2 | During training the hypernetwork forward runs in vocabulary chunks of `HYAK_HN_CHUNK_ROWS` (default 16,384) rows, each chunk under activation checkpointing (a runtime patch; the pinned files are untouched). Evaluation uses the pinned forward. With dropout on, masks are drawn per chunk: same distribution, a different random stream from the pinned forward. Under bf16 autocast the chunked and pinned forwards are equal only up to bf16 rounding even with dropout off (float64 identity holds); the probe checks this against the pinned path's own bf16 rounding. | The unmodified trainer does not fit an H200 for large vocabularies, and the largest pool vocabularies have 128,000 rows. | The original ZeTT-baseline trainer peaked at 72.2 GiB at 32,768 rows on a B300; the linear model gives about 190 GiB at 127,000 rows, against 131 GiB (141 GB) on an H200. Predicted peaks (section 6.3) with 16,384-row chunks are 58 to 91 GiB, set by the LM micro-batches. The chunked forward equals the pinned one to 5e-16 (outputs) and 1.3e-15 (all gradients) in float64 tests; the probe repeats the check on the GPU. |
| D3 | One H200 per run; 24-hour Slurm jobs chained until update 20,000. The two runs may train at the same time once the uniform run has frozen its pair (D6). | Tillicum hardware and limits. | The originals ran on B300 at about 3.0 to 3.4 s per update including evaluation and checkpointing. |
| D4 | The corpus (23 buckets) and the pool (1,000 tokenizers) are regenerated on the cluster. | The originals are in private storage. | Fidelity is measured with the repository's own anchors: corpus scorecard (exact at full scale off-cluster), pool scorecard, replay check (section 10). |
| D5 | The two sanity tokenizers of the uniform run are new; the RL run receives the new pair. | The trainer draws them with an unseeded Rust random generator. | `rust_utils/src/lib.rs:193` uses `thread_rng()`. |
| D6 | The uniform run starts with a one-update bootstrap segment, which writes `run_config.json` and `heldout_tokenizers.json`; these are then frozen into `input/` and the RL run may start. The uniform run is never restarted from scratch afterwards. | The RL trainer reads the uniform run's config and sanity tokenizers; they cannot be recreated. | `scripts/modal_anytok_pool_rl_20k.py` copied the same two files. |
| D7 | Slurm jobs request 1 GPU, 8 CPUs, 24 h and no `--mem` flag (the job gets the cluster's per-GPU default memory, documented as 195.3125 GB). All in `hyak/config.env`. | `AGENTS.md` asks for no explicit CPU or memory reservations (written for Modal); Slurm needs a CPU count, and Tillicum caps 8 CPUs and about 200 GB per GPU. Memory is left to the cluster's default. | Short stages ask for shorter limits (section 5). |
| D8 | The RL run is the straight-reward counterpart of Qingyun's joint regret RL20k: his trainer (`runtime/anytok_pool_regret_rl_20k`) with `bandit.update(pool_index, train_bpb)` instead of the paired gap (`hyak/variants/anytok_pool_straight_rl_20k`), corrected paired boundaries for training and sanity examples, the no-grad canonical control forward of every update, entropy 0, policy learning rate 0.001, baseline decay 0.99, policy seed 43. The uniform run stays the original one (original trainer and cuts). Every RL segment and smoke is checked against the regret launcher's receipt plus the straight reward (6.11). | Not forced: the user's decision of 2026-11-14 ("the same as Qingyun's new joint regret RL20k, but with the straight BPB loss as the reward"). | `plans/anytok-pool-regret-rl-20k.md`; `hyak/variants/anytok_pool_straight_rl_20k/README.md` (provenance, hashes, the exact patch); `hyak/tests/test_straight_variant.py` (the patch reproduces the variant byte for byte; only the two approved lines differ). |

Other differences you should know about (smaller, all reported by the code's authors):

* **Corpus staging.** The trainer's corpus files (`input/corpus/<bucket>/train.jsonl.gz`) are links
  to the full regenerated pool files (13.6 GB of text); the original trainer read quota-sized
  copies (10 GB) made by a program that is not in the repository. They give the same text stream:
  the trainer reads each bucket from its first line and rewinds at the recipe quota, and the
  original copies held exactly the quota (the original `corpus/manifest.json`, rebuilt from the
  regenerated pools, hashes to the value pinned in `plans/zett-modal-20k-inputs.json`, and the
  reconstructed file sizes equal the pinned ones in all 23 buckets). The corpus scorecard checks
  both anchors (section 10).
* The corpus builder runs on Python 3.12 with seven pinned packages (`datasets` 5.2.0, `pyarrow`
  26.0.1, `huggingface-hub` 2.4.0, `hf-xet` 1.8.0, `fsspec` 2026.9.0, `requests` 2.34.2,
  `urllib3` 2.8.0): the set that reproduced every anchor of the original corpus at full scale (the
  original image used Python 3.11 and did not record its package versions). Training-file
  gzip header times are recorded and reused, so a rebuilt bucket is byte-identical.
* All 1,000 tokenizers are trained with one wheel carrying all three patches. Originally the first
  five pilots used an older base wheel. The Rust toolchain of the original wheel was not recorded;
  1.96.0 is used (the compiler of the committed `librust_utils.so`). The tokenizer worker runs on
  Python 3.12 (original 3.11); the wheel is an abi3 build and the worker uses only the standard
  library.
* Unigram training is not bit-reproducible even on one machine (measured: same token strings,
  different order and scores). Unigram members can never match the original file hash.
* `input/pool/full-results.json` stores timings, so its hash (`pool_index_sha256`) is new; both runs
  share one index. The corpus manifest hash recorded by the trainer is new for the same reason.
* Evaluation re-validates the stored prefixes against the regenerated pool with the repository's
  own trimming code and reports every trimmed row. A pool-panel row whose context was valid in the
  original panel and would now need a trim stops the evaluation (section 2.3).
* On a POSIX file system a hard kill can leave partial state that a Modal volume never showed. The
  training wrapper repairs it (stale `.writing-*`, saves newer than the restart moved to
  `<run>/interrupted/`, also for the uniform run) and states each repair in its events.
* A training chain halts after a failed job (section 8), like the Modal controllers, which never
  retried a failed segment automatically.

---

## 2. Decisions before the expensive steps

Five decisions and one caveat. The first four are cheap now and costly later; the fifth only needs a
look; the caveat (2.6) is about what the RL run can be compared with.

### 2.1 The initialization (D1): the default is option (b)

You do not need to decide anything unless you want (a) or (c). The default recipe is
**(b) calibrate first**: seed 42, `build_input_hypernet(49152, 2048)`, `initialize_rescalers` with
the library's zero prior bias (exactly the original pre-warmup construction), then the
prior-projection bias is drawn from N(0, 0.02) with a generator seeded 1042. After the calibration
the tool compares the computed `in_scaler` and output scaler with the public step-500 checkpoint
(`hyak/model/original_in_scaler.npz`) within 1e-5; if they differ it stops (`INIT FAILED`, exit 4);
if they agree it **copies** the four frozen rescaler tensors from that file, so they equal the
original runs' bit for bit on any machine. (Measured on the laptop: output scaler within 4.9e-7,
`in_scaler` within 6e-8.)

Why (b): it changes exactly one tensor of the original pre-warmup state, the 2,048-element bias,
which the original warmup also changed, and it starts both tokenizer families alike. Measured at
full size, the step-0 offset of one-piece rows (in embedding norms, against a canonical row norm of
3.17) and the lexical loss, for zero-prior (BPE and WordPiece) and Unigram vocabularies:

| Option | How | Frozen rescalers | Offset zero / Unigram | Lexical loss zero / Unigram | Zero-prior gradient |
|---|---|---|---|---|---|
| (a) | `--env HYAK_INIT_ARGS="--calibrate-after-bias"` | differ from the original for all 20,000 updates (`scaler.w` mean 0.2239 against 0.1821: +23% on average, 4-41% by dimension) | 3.5 / 9.5 | 10.3 / 34.6 | healthy (ratio 0.115) |
| (b) | default | identical to the original | 7.9 / 8.2 | 25.0 / 26.2 | healthy (ratio 0.115) |
| (c) | `--env HYAK_INIT_ARGS="--prior-bias-std 0"` | identical to the original | not measured at full size | not measured at full size (pool-weighted 11.4) | degenerate (ratio about 2,509; first update's gradient norm 2.08e6, the bias takes the whole 0.1 clip) |

Pool-weighted step-0 lexical loss: (a) 19.4, (b) 14.5, (c) 11.4; the floor is about 6.3. Notes:

* (a) starts the two families very differently (lexical loss 10 against 35) and its frozen output
  scaler differs from the original's for the whole run. In a width-128 miniature (a) and (b) were
  indistinguishable from update 300 to 5,000.
* (c) is the exact original pre-warmup state, but Unigram updates and zero-prior updates would be
  weighted very differently, and the RL policy changes the family mix, so (c) is a
  family-dependent confound for uniform versus RL. In the miniature it trailed (a) and (b) by about
  250, 450 and 530 updates at updates 1,000, 2,000 and 5,000. Not recommended.
* The calibration check in `init_manifest.json` (`checks.calibration`) covers only the 2,048 anchor
  rows (token ids 0-2047 as one-piece forms, zero priors, one batch). On a real vocabulary every
  option starts with an offset; it is expected to vanish within the first several hundred updates
  (the public run went from lexical loss 1.92 at update 1 to 0.42 at update 300). Only a cluster
  run confirms that.
* The choice is recorded in `input/hypernet_init/init_manifest.json` (`recipe.option`,
  `recipe.seed`, `recipe.calibration`, `recipe.rescalers_from_fixture`) and printed in the `INIT OK`
  line. Once a run directory exists the init directory cannot be replaced (`--force` is refused).
  A directory built before this recipe (no `recipe.calibration`) is not reused: rebuild it with
  `--force` while no run directory exists.

**Health indicator, whatever the option.** The ratio of the median `gradient_norm_before_clip` of
zero-prior (BPE and WordPiece) updates to that of Unigram updates over the last 200 updates should
be about 1 for (a) and (b). Both numbers are in the train log; the tokenizer algorithm comes from
`input/pool/plan.json`. From the repository root, with `ROOT` set as in section 4.1, after the first
jobs of a run have written their logs (`RUN_NAME=rl` for the RL run):

```bash
ROOT=$ROOT RUN_NAME=uniform $ROOT/env/train/bin/python - <<'EOF'
import glob, json, os, statistics
root, run = os.environ['ROOT'], os.environ.get('RUN_NAME', 'uniform')
algorithm = {c['index']: c['algorithm'] for c in json.load(open(f'{root}/input/pool/plan.json'))['configs']}
rows = {}
for path in glob.glob(f'modal_results/hyak_tillicum/train-{run}/*-train.log'):
    for line in open(path):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('stage') == 'train' and 'gradient_norm_before_clip' in row:
            rows[row['step']] = row
last = [rows[step] for step in sorted(rows)[-200:]]
zero = [r['gradient_norm_before_clip'] for r in last if algorithm[r['tokenizer_config_index']] != 'unigram']
unigram = [r['gradient_norm_before_clip'] for r in last if algorithm[r['tokenizer_config_index']] == 'unigram']
print(len(zero), len(unigram), statistics.median(zero) / statistics.median(unigram))
EOF
```

The printed values are the two sample sizes and the ratio.

### 2.2 Ask the pool owner for the original pool and corpus first

Regenerating the pool is the largest cost of the port: about 500 GPU-hours (about 450 USD), and the
regenerated pool differs from the original wherever training is not reproducible (all Unigram
members). The original owner has the real thing:

* the **pool**: `plan.json`, `full-results.json` and the 1,000 `tokenizers/NNNN/tokenizer.json`
  (about 4-5 GB);
* the **corpus** (about 5 GB): either the 23 corpus pools, or the 23 staged trainer files and
  `corpus/manifest.json` (section 11).

Getting them is free, more faithful, and saves about 500 GPU-hours, the pool wheel and the pilots.
The transfer must go directly between cloud storage and the cluster, never through your laptop
(`AGENTS.md`). Section 11 says where the files go and which checks run on them. If you cannot get
the pool, section 6.7 regenerates it, with a stop-or-go point after the first 19 tokenizers.

### 2.3 If prefix validation stops (exit 3)

`eval-prepare` re-validates the stored pool-panel prefixes against the pool in use with the
repository's own trimming loop. With a pool identical to the original it changes nothing. With a
regenerated pool, **one more native token on one of ten (tokenizer, passage) pairs** that sit
exactly at the 95-token cap, on passages whose context was valid in the original panel, stops the
evaluation of the complete-pool table with exit 3 (`validate_prefixes --panel pool`). Seven
tokenizers decide this: 64, 100, 632, 687, 728, 821 and 914 (the binding tokenizers). Section 6.7
trains them with the pilots, before the full pool, and gives a stop-or-go rule, so you learn about
the problem for a few tens of GPU-hours (*estimate*) instead of 500.

If it stops anyway, nothing is dropped automatically and no flag is a default. The report
`$ROOT/eval/panels/pool/validation-report.json` lists every trim (`valid_context_trims`: row id,
characters before and after, the forcing tokenizers). Targets and scored bytes never change. Two
options, **both need your explicit approval** (they change the frozen protocol, `AGENTS.md`):

1. Accept the listed trims and rerun with
   `hyak/slurm/submit.sh eval-prepare --env HYAK_EVAL_PREPARE_ARGS=--accept-valid-context-trims`.
   The trimmed rows are recorded in the panel plan, the validation report and every packed summary.
2. Bring the original pool (section 11) and validate with it: no trim.

Unsatisfiable rows (no shared lossless prefix of 16 canonical tokens), no convergence in five
rounds, the position cap and the coverage check have no option; they stay stops. The natural panel
is independent: it validates and scores even when the pool panel stopped (section 6.13).

### 2.4 Where the checkpoints live after the run

Both runs write to scrubbed storage, which deletes files that were not accessed for 60 days
(section 9). Decide now what survives: the two **final** checkpoints (`step-020000`, 23.5 GB each)
and the small run files go to project storage with the commands of section 9; the 40-checkpoint
**grids** (493 GB per run) either go to Kopah (S3) or Lolo (archive), the two long-term targets the
Hyak storage page names, or are allowed to expire. The scorer accepts an archived copy of the
final checkpoint (`HYAK_EVAL_SCORE_ARGS`, section 6.14), so scoring does not have to happen before
you archive.

### 2.5 The NVIDIA driver

The pinned `torch==2.15.0` exists only as CUDA 13 builds and needs an NVIDIA driver of **580 or
newer** on the compute nodes. `check-env` (section 6.1) reports the driver version; its
`torch_cuda` line is the test. With an older driver the options are the CUDA forward-compatibility
package (`cuda-compat` for CUDA 13.2, which Hyak support would have to provide) or a different
torch version; the latter is a material change that needs your approval. Run `check-env` before
you pay for anything else.

### 2.6 Comparisons with Qingyun's joint regret run are confounded (D8)

The Hyak RL run differs from Qingyun's joint regret RL20k (`plans/anytok-pool-regret-rl-20k.md`)
in its reward (straight transferred BPB here, the paired gap there), but not only there: his run
starts from the **private identity-warmed hypernetwork** (the original warmup) on the original pool
and corpus on a B300, while both Hyak runs start from the **random initialization of D1** on the
regenerated pool (D4) with the chunked hypernetwork forward (D2) on an H200. A difference between his
regret run and the Hyak RL run therefore cannot be attributed to the reward alone; the initialization
is the largest confound (section 2.1 measures how far apart the starts are). The comparisons that
share initialization, pool, corpus and hardware are the ones inside the port: Hyak RL against Hyak
uniform on the final evaluation (6.13 to 6.15), which scores both checkpoints on the same panels.
Their training-time sanity scores are not comparable with each other: the RL run uses the corrected
paired cuts, the uniform run the original cuts (6.11). Qingyun compares his regret run with the
original uniform run (warmup initialization, original cuts) in the same way.

---

## 3. Prerequisites

* A Tillicum account; `ssh <netid>@tillicum.hyak.uw.edu`. Login nodes are for editing, submitting
  and data transfer only; heavy processes there get killed. Every job needs at least one GPU
  (CPU-only jobs are not allowed), at most 8 CPUs and about 200 GB per GPU. QoS `normal` allows
  24 h, `debug` 1 h and one GPU, `interactive` 8 h.
* Storage: about 100 GB in project storage (`/gpfs/projects/<group>`, 1 TB, snapshots): about 50 GB
  for environments, corpus, pool, inputs and evaluation, plus 47 GB for the two archived final
  checkpoints; and about 1.1 TB in scrubbed storage (`/gpfs/scrubbed`, files not accessed for 60
  days are deleted). See section 9. Check your quotas with `hyakstorage` (`check-env` warns when
  less than 150 GB is free on the file system of `ANYTOK_ROOT`: a file-system view, not your quota).
* This checkout, including the `hyak/` directory and the one-line change in
  `monitoring/web/learning.js`, in project storage (not in your 10 GB home). Either push it to a
  branch you can clone on the cluster, or copy it:
  `rsync -a --exclude modal_results/ --exclude .git/ ./ <netid>@tillicum.hyak.uw.edu:/gpfs/projects/<group>/<you>/anytok/`
  (copying without `.git` loses the commit id that `stage_code` records; prefer a clone; if a copy
  loses the executable bit of `hyak/slurm/submit.sh`, `chmod +x hyak/slurm/*.sh`).
  The path must not contain a comma.
* Internet on the login node (pip, conda-forge, crates.io, GitHub, Hugging Face). Whether compute
  nodes can reach the internet is unknown; only the corpus build needs it in a job (section 13).

---

## 4. One-time setup (login node)

### 4.1 Configuration

```bash
cd /gpfs/projects/<group>/<you>/anytok          # the repository root
cp hyak/config.env.example hyak/config.env      # git-ignored
chmod 600 hyak/config.env                       # an HF_TOKEN would otherwise sit in group-readable storage
```

Edit `hyak/config.env`. Required:

* `ANYTOK_ROOT`: absolute path in project storage **outside the checkout**, e.g.
  `/gpfs/projects/<group>/anytok-data`. Environments, code, corpus, inputs and evaluation go here.
* `ANYTOK_RUNS`: absolute path in scrubbed storage, e.g. `/gpfs/scrubbed/<you>/anytok-runs`. The two
  run directories (about 0.5 TB each) go here. **Scrubbed storage is readable by other users by
  default:** create your directory yourself, with the mode you want, before the first job:
  `mkdir -p -m 700 /gpfs/scrubbed/<you>`.

Both paths must be written exactly as conda and Python would write them: absolute, no trailing
slash, no `//`, no `.` or `..` component (otherwise every job dies at environment activation; the
error names the right spelling). Neither may be equal to or inside the repository checkout
(`/x/anytok-data` next to a checkout `/x/anytok` is fine).

Optional: `HYAK_ACCOUNT` (empty = no `--account`), `HF_TOKEN` (all sources are public; a token only
raises rate limits; it is never printed and `submit.sh` refuses it in `--env`), per-stage
overrides such as `HYAK_TIME_POOL=12:00:00` or `HYAK_PARTITION_POOL=...` (the comments in the file
list them). Leave `HYAK_HN_CHUNK_ROWS` at 16384 (an integer of at least 4096) unless the probe says
otherwise, and do not change it after the uniform bootstrap (section 6.10 says what happens if you
must). `HYAK_MEM` is empty on purpose (no `--mem` flag: the cluster's per-GPU default applies). Do
not use the MIG partition (`gpu-h200-mig`) for the pool: a MIG unit has 1 CPU and 30 GB, the pool
needs 2 workers x 4 threads, and 8 units cost more than one GPU.

Check it, and see exactly what a job of each stage will request:

```bash
python3 -m hyak.common.config --check
python3 -m hyak.common.config --resources train-uniform
hyak/slurm/submit.sh train-uniform --dry-run
```

For the commands below, set two shell variables (do **not** `export ANYTOK_ROOT=...`: an
environment variable overrides `hyak/config.env` and `submit.sh` passes your environment to every
job):

```bash
ROOT=$(python3 -m hyak.common.config --get ANYTOK_ROOT)
RUNS=$(python3 -m hyak.common.config --get ANYTOK_RUNS)
export PYTHONDONTWRITEBYTECODE=1
```

### 4.2 Environments

```bash
bash hyak/env/setup_train_env.sh      # Python 3.12 + torch 2.15.0 and the other five pins + pytest
bash hyak/env/setup_data_env.sh       # the seven corpus packages, maturin, huggingface_hub + Rust 1.96.0
```

Both load the `conda` module themselves, write `$ROOT/env/<name>.freeze.txt`, never touch your
dotfiles, ignore pip configuration files and `PIP_*` variables, and are safe to re-run
(`--dry-run` shows what they would do). The train environment is about 3 GB to download and about
twice that installed, plus the pip cache. Good output ends with `done. Next: ...`. Facts about the
builds (checked 2026-11-13):

* `torch==2.15.0` for Linux x86-64 / Python 3.12 exists **only as CUDA 13 builds** (PyPI default
  CUDA 13.2, which the Modal images used; `download.pytorch.org/whl/cu132` and `cu134`). There is
  no CUDA 12 build.
* CUDA 13 needs an NVIDIA driver **>= 580** (section 2.5). Leave `HYAK_TORCH_INDEX_URL` empty:
  with a driver below 580 no torch 2.15.0 build runs, and neither `cu132` nor `cu134` helps.
* Never add a `cuda` module to `HYAK_MODULES`: the wheel bundles its CUDA libraries, and a module's
  libraries could shadow them.
* The data environment pins the seven validated corpus packages (section 1). After the
  patched tokenizers wheel is installed (section 6.6) `pip check` reports one conflict (the wheel
  requires `huggingface-hub<2.0` and is installed with `--no-deps`); `setup_data_env.sh` accepts
  it. The wheel job refreshes `env/data.freeze.txt`; `bash hyak/env/setup_data_env.sh --freeze-only`
  does it by hand. A changed freeze keeps the old record as `<name>.freeze.<UTC>.txt`.
* `accelerate` brings `psutil` into the train environment; that is harmless.

### 4.3 Code and model

```bash
$ROOT/env/train/bin/python -m hyak.code.stage_code       # copies of both trainer trees, hashed
$ROOT/env/train/bin/python -m hyak.model.fetch_model     # SmolLM2-1.7B at effd688a..., 3 files, 3.4 GB
```

`stage_code` writes read-only, hash-verified copies to `$ROOT/code/uniform` and `$ROOT/code/rl` and
`$ROOT/code/code_manifest.json`; its last line is JSON with `"stage": "code_staged"`. Run it once,
before any training, and never restage after a run directory exists (it refuses). Re-check any
time with `$ROOT/env/train/bin/python -m hyak.code.stage_code --verify`.

`fetch_model` checks every file's size and SHA-256 against `plans/zett-modal-20k-inputs.json`; its
last line is JSON with `"stage": "model_fetched"`. It is resumable; it never deletes files: wrong or
extra files stop it with an `rm` remedy. Re-check with `--verify-only`. Training re-hashes the three
model files at every segment.

### 4.4 Running one command in a job

Several steps below say "in a job": collect, scorecard, `retention verify` and `audit`, `pack`,
the Rust rebuild, the tests. They hash hundreds of GB or load all 1,000 tokenizers, which the login
node would kill, and every Tillicum job needs a GPU. For one command, take an interactive
allocation (this recipe is not verified on Tillicum; in particular, if `salloc` leaves you on the
login node (`hostname` says), start the shell on the node with `srun --pty bash -l`):

```bash
salloc --partition=gpu-h200 --qos=interactive --gres=gpu:1 --cpus-per-task=8 --time=02:00:00
cd /gpfs/projects/<group>/<you>/anytok
source hyak/slurm/lib.sh && hyak_load_config && HYAK_STAGE=replay-check hyak_activate train
python -m hyak.train.retention audit --run uniform      # or collect, scorecard, pack, pytest ...
exit
```

`lib.sh` is what the job scripts use: `hyak_load_config` reads `hyak/config.env`, `hyak_activate
train` (or `data`) loads the module, activates the environment and sets `PYTHONPATH`, `HF_HOME` and
the library path. `HYAK_STAGE` only selects per-stage environment details; `replay-check` gives
the offline settings of the train stages. Stop the allocation with `exit` as soon as the command
is done: it bills like any GPU job.

---

## 5. The pipeline at a glance

Default time limits are the ones in `hyak/config.env.example` (the short stages ask for 2-3 times
their estimate, which also schedules them sooner); `check-env` and `probe` use the debug QoS (1 h)
by default.

| Step | Where | Command | Time (limit) | GPU-hours |
|---|---|---|---|---|
| 6.1 environment check | debug job | `hyak/slurm/submit.sh check-env` | minutes (1 h) | < 1 |
| 6.2 random-init hypernetwork | debug job | `hyak/slurm/submit.sh init --debug` | under a minute of work (1 h) | < 1 |
| 6.3 memory probe | debug job | `hyak/slurm/submit.sh probe` | < 1 h (1 h) | < 1 |
| 6.4 corpus build + scorecard | job | `hyak/slurm/submit.sh corpus` | 26 min measured on a laptop; budget 1-2 h on the cluster (*estimate*) (6 h) | 0.3-2 (*estimate*) |
| 6.5 stage inputs | debug job | `hyak/slurm/submit.sh stage-inputs --debug` | minutes (1 h with `--debug`) | < 1 |
| 6.6 tokenizers wheel | login node + debug job | `bash hyak/pool/build_wheel.sh fetch`, then `hyak/slurm/submit.sh wheel --debug` | 2-5 min build (1 h with `--debug`, 30 min without) | < 1 |
| 6.7 pilots + binding tokenizers | 10 jobs | `hyak/slurm/submit.sh pool --count 10 --env HYAK_POOL_ARGS="--indices pilots+binding"` | 1-4 h (*estimate*) (24 h) | 10-40 (*estimate*) |
| 6.7 full pool (+ collect + scorecard) | N jobs | `hyak/slurm/submit.sh pool --count N` | about 25 h with 20 jobs; expect to resubmit (*estimate*) (24 h) | about 500, range 300-1,100 (*estimate*) |
| 6.8 replay check | job | `hyak/slurm/submit.sh replay-check` | about 1 h for 2,000 updates (*estimate*) (2 h) | 1-2 (*estimate*) |
| 6.9 smoke, uniform | job | `hyak/slurm/submit.sh smoke-train --run uniform` | not measured (2 h) | about 1 (*estimate*) |
| 6.10 uniform run | chain | `hyak/slurm/submit.sh train-uniform --count 4` | not measured on H200 (24 h each) | 30-50 or more (*estimate*) |
| 6.11 smoke, RL; RL run | job, chain | `hyak/slurm/submit.sh smoke-train --run rl`; `hyak/slurm/submit.sh train-rl --count 4` | as above, plus the canonical control forward (6.11) | 35-60 or more (*estimate*) |
| 6.13 evaluation preparation | job(s) | `hyak/slurm/submit.sh eval-prepare` | 10-20 min (*estimate*) (1 h) | < 1 |
| 6.14 scoring, both checkpoints | jobs | `hyak/slurm/submit.sh eval-uniform --count 2`, `... eval-rl --count 2` | — (24 h) | 1.5-4 per checkpoint, pool panel; 5-20 min natural panel (*estimate*) |

Total, derived from the estimates above: roughly 370 to 1,200 GPU-hours, centrally about 600, so
about 450 USD after the 100 free hours (range about 250 to 1,000 USD). Pool generation dominates;
its first 19 tokenizers (6.7) give the first real measurement. Steps 6.4 to 6.7 do not depend on
6.1 to 6.3, but run the probe (6.3) first: if it fails, nothing else is worth paying for.

**Ways to spend less on the pool** (everything else is fixed by the experiment's design):

| Route | Cost | Status |
|---|---|---|
| The original pool and corpus from their owner (section 2.2, section 11) | 0 GPU-hours | the best route: the original files, free; code paths tested only on synthetic data |
| Regenerate, two tokenizers per job (the default) | about 500 GPU-hours | the route of section 6.7 |
| Regenerate with more tokenizers per job (`HYAK_POOL_WORKERS`) | unmeasured saving | a timing-only knob: every tokenizer keeps `RAYON_NUM_THREADS=4`, so the files do not change; see 6.7 |
| A CPU cluster instead of GPU jobs | — | out of scope and not supported |

Order constraints: init needs the staged code and the model; the probe needs init; stage-inputs
needs the corpus (with an exact scorecard); the pool needs stage-inputs (it reads
`input/pool-recipe.json`) and the wheel; the replay check and training need the collected pool; the
RL run needs the uniform bootstrap; scoring needs the final checkpoints and eval-prepare.

Chain stages with `--after JOBID`, for example
`hyak/slurm/submit.sh stage-inputs --after 123456` starts stage-inputs only if job 123456
succeeded (a failed predecessor cancels it). If `sbatch` refuses a second debug job while one is
queued or running, wait for the first to finish: the debug QoS limits are not documented; on the
cluster, `sacctmgr show qos debug format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU` shows them.

---

## 6. Step by step

Every stage has one job script `hyak/slurm/NN_<stage>.sbatch` and is submitted with
`hyak/slurm/submit.sh <stage>`, which reads every Slurm resource from `hyak/config.env` and prints
the exact `sbatch` command (add `--dry-run` to see it without submitting; `sbatch --test-only` on
that command is a good first check of the Slurm flags). Each job writes
`modal_results/hyak_tillicum/<stage>/{status.json, events.jsonl, slurm-<jobid>.out, latest-slurm.out}`
in the checkout. When a job fails, its `status.json` holds the reason and the tail of the job log.
**The checkout is live code for queued jobs:** do not `git pull` or edit it while jobs are queued
or running.

### 6.1 Environment check

```bash
hyak/slurm/submit.sh check-env
```

Runs `python -m hyak.env.check_env` on one H200 (debug QoS). Good output in
`modal_results/hyak_tillicum/check-env/latest-slurm.out`: one `[PASS]`/`[WARN]`/`[FAIL]`/`[SKIP]`
line per check and

```
check_env OK: <n> passed, <w> warnings, 0 failed, <s> skipped
```

Every `[WARN]` and `[FAIL]` line is followed by a `remedy:` line. The report is
`modal_results/hyak_tillicum/check-env/check_env_report.json`; `python -m hyak.env.check_env
--list` names the checks. Things to look at even when it passes: the GPU name (H200), the driver
version (>= 580), `torch.version.cuda` (13.2), `torch_cuda` (a WARN there, with CUDA initialising
anyway, means forward-compatibility libraries are in use; `gpu_smoke` is the proof), the
`rust_library` and `trainer_import_*` checks, `filesystem_runs` and `filesystem_monitor` (flock and
rename on scrubbed storage and on the checkout), the model files (hashed by default), and the
`network` warning (if compute nodes cannot reach huggingface.co, see section 13). It exits 4 when a
check failed.

If `rust_library` or a trainer import fails, rebuild the Rust sampler. The fetch phase needs
internet (login node); the build compiles about 110 crates, so run build, check and install in an
interactive job on a compute node (section 4.4), with `--module gcc/13.4.0` when no C compiler is on
the path:

```bash
bash hyak/env/build_rust_utils.sh fetch                                 # login node
bash hyak/env/build_rust_utils.sh build check install --module gcc/13.4.0   # in an interactive job
hyak/slurm/submit.sh check-env
```

`install` puts the new library into both staged trees and the code manifest; it is refused once a
run directory is non-empty, so do this before any training.

### 6.2 Random-init hypernetwork (D1)

Section 2.1 describes the choice; the default needs no decision. Then:

```bash
hyak/slurm/submit.sh init --debug
```

CPU work in a GPU job (about 4.6 GB RAM, under a minute of CPU at full size; it needs about 4.4 GB
of free space next to `input/hypernet_init` for a short time). Good output:

```
INIT OK: <ROOT>/input/hypernet_init model.safetensors 1376746800 bytes sha256 <hex> zero-prior gradient ratio 0.115; recipe (b) calibrated with the zero bias, then bias drawn (default), seed 42, prior-bias std 0.02, calibration before_bias, rescalers copied from original_in_scaler.npz sha256 a47af5d8705ef0ea...
```

and a line `[init] calibration with the zero prior bias reproduces the public checkpoint (in_scaler and
scaler within 1e-05); the four frozen rescaler tensors were copied from original_in_scaler.npz`.
With the default this check is a gate: if the cluster does not reproduce the construction within
1e-5 the job stops with `INIT FAILED` (exit 4); then stop and report, or rebuild with
`--calibrate-after-bias` (section 2.1) and say so. The model files must match
`plans/zett-modal-20k-inputs.json`; otherwise `INIT FAILED` with exit 2 and the remedy
`python -m hyak.model.fetch_model`. Do not expect the SHA-256 to equal a laptop build; expect the
size 1,376,746,800 bytes. A line `[init] NOTE: reloaded anchor outputs are not bit-identical ...` is
harmless (the reload check tolerates rounding; the manifest records it). Re-running prints
`(already built)`. Check later with
`$ROOT/env/train/bin/python -m hyak.model.init_hypernet --verify --out $ROOT/input/hypernet_init`.
Leftover temporary directories `.hypernet_init.building--<host>--<pid>` of a job killed on another
node are removed automatically after 6 h.

### 6.3 Memory probe (D2): run it before anything expensive

```bash
hyak/slurm/submit.sh probe
```

Runs `python -m hyak.memory.probe --also-pinned` on one H200 (debug QoS; needs only the
environment, `input/model`, `input/hypernet_init` and `code/uniform`). Before importing anything it
checks the staged trainer files against the pins (remedy: `python -m hyak.code.stage_code
--verify`). It then checks on the GPU that the chunked forward gives the pinned outputs and
gradients (`[probe] EQUIVALENCE PASS`; as many chunks as the configured size gives at 128,000 rows,
including gradients with and without checkpointing with dropout on, which proves that recomputation
redraws the forward's dropout masks), then runs two real training-shaped updates per vocabulary
size (32,768; 64,000; 96,000; 128,000 rows) with the configured chunk size, and afterwards with the
pinned forward, which is information only: its out-of-memory or errors never fail the probe. Read
the lines

```
[probe] VERDICT V=128000 chunk_rows=16384: FITS (...)
[probe] SUMMARY ...
```

A `VERDICT` line shows the seconds of the first and of the last update, the **phase that set the
peak** (`LM micro-batches` or `hypernetwork backward`) and the predicted peak with its phase.
`FITS` means the H200 keeps at least 8 GiB free at the peak (`max_memory_reserved`). The pinned
forward is expected to run out of memory above roughly 84,000 rows (`DOES NOT FIT`, caught; that is
the reason for D2). Exit 0 only if every configured-chunk measurement fits and the equivalence
passed; exit 4 means `DOES NOT FIT`, a failed measurement or a failed equivalence (do not start
training); exit 2 is a setup problem (configuration, modified staged files, no CUDA, a configured
chunk below 4,096). Report: `modal_results/hyak_tillicum/probe/probe-report.json` (also
`$RUNS/probe/probe-report-<job>.json`).

Predicted peaks in GiB (`python -m hyak.memory.memory_model` prints the table). The peak is the
larger of the hypernetwork-backward phase and the LM micro-batch phase; the LM phase is 27.9 GiB +
8.2 U(V) + 17 to 21 GiB per 32 x 128 micro-batch, with U(V) = V x 45,056 bytes, and is uncertain by
about 10% of its vocabulary-proportional part plus that 17-21 GiB range:

| rows | pinned | chunk 32,768 | chunk 16,384 | chunk 8,192 |
|---|---|---|---|---|
| 32,768 | 72.2 | 62.5 | 58.2 (LM) | 58.2 (LM) |
| 64,000 | 113.7 | 72.7 | 68.9 (LM) | 68.9 (LM) |
| 96,000 | 156.1 | 83.2 | 79.9 (LM) | 79.9 (LM) |
| 128,000 | 198.5 | 93.7 | 90.9 (LM) | 90.9 (LM) |

With 16,384-row chunks the LM micro-batches already set the peak, so smaller chunks do not lower
it. If 16,384 does not fit **and the VERDICT says the hypernetwork backward set the peak**, try
smaller chunks, never below 4,096 (training refuses smaller values: more chunks sum the weight
gradients in bf16), for example
`hyak/slurm/submit.sh probe --env HYAK_PROBE_ARGS="--chunk-rows 8192 4096"`, and set the **largest
value that fits** as `HYAK_HN_CHUNK_ROWS` in `hyak/config.env` **before** the uniform bootstrap.
If the LM micro-batches set the peak, chunk size cannot help: stop and report. Never change the
setting afterwards without the recorded override of section 6.10: resume refuses a different value,
and both runs must use the same one. If the equivalence fails or nothing fits, stop and report the
numbers: no training should start.

The largest pool vocabulary has 128,000 rows (seven tokenizers); the uniform schedule first draws
one at update 261. The probe at 128,000 rows, not the bootstrap, is the memory test (section 6.10).

### 6.4 Corpus build and corpus scorecard (D4)

Skip this step if you obtained the original corpus (section 11).

```bash
hyak/slurm/submit.sh corpus
```

Data environment; the node needs outbound HTTPS to huggingface.co for the whole build. The job
first probes the first record of each of the ten sources at its pinned revision (exit 2 stops it;
see `modal_results/hyak_tillicum/corpus/probe.json`), then builds all 23 buckets to the original
pool targets (13.6 GB of raw training text plus 23 x 5 MB held-out text, about 5 GB of gzip) into
`$ROOT/corpus/<size>-v1/<bucket>/`, six buckets at a time, largest first; then writes the
fidelity scorecard. Built buckets are re-verified and skipped, so resubmitting is safe; a bucket
that fails verification is never rebuilt silently. A production build refuses (exit 2) any other
installed versions of the seven pinned packages unless `--allow-unvalidated-env "<reason>"` is
given (as `--env HYAK_CORPUS_ARGS="--allow-unvalidated-env reason-in-one-word"`; the reason is
recorded in every bucket manifest; this needs your approval). Status `paused` (time limit, or a bucket locked by another job): resubmit. Status
`failed` (exit 4): the error names the bucket and its log in `$ROOT/corpus/.hyak/logs/`.

* **Locks.** Each bucket has a lock directory `$ROOT/corpus/.hyak/locks/<size>__<bucket>` whose
  owner refreshes `owner.json` every 60 s. A lock is taken over when its owner process is gone on
  the same host, when Slurm no longer knows the owner job, or (when neither can be checked, for
  example a build on the other login node) after 15 minutes without a heartbeat. The `locked`
  message names the directory: if no corpus build runs anywhere (`squeue -u $USER`, and `ps -u
  $USER` on both login nodes), `rm -r` that directory and rerun. A job that could not start
  buckets because of the time guard ends `paused`, never `complete`. Disk-full, quota, read-only
  and permission errors are not retried.
* **Reproducible rebuilds.** Each bucket's gzip header times are recorded (manifest
  `hyak.gzip_mtime`, `$ROOT/corpus/.hyak/gzip-mtimes/`, and at staging in `$ROOT/input/manifest.json`).
  A lost or purged bucket is rebuilt with `$ROOT/env/data/bin/python -m hyak.corpus.build --buckets
  <bucket>` and comes back with the same SHA-256 that `input/corpus/manifest.json` and both runs
  recorded.
* **Estimates.** Measured: a full build on a laptop with 2 workers took 26 minutes of wall time
  (about 0.7 core-hours; 23 buckets in one attempt each; 4.67 GB of training gzip, 38.8 MB of
  held-out gzip, 2,921,003 training documents; largest bucket 5.3 minutes). On the cluster, with
  six workers, budget 1 to 2 hours if the network allows (*estimate*).

Never pass `--test-scale` (or `--allow-test-scale` to stage-inputs) in production: those flags exist
for laptop tests and mark the manifests as non-production; every training segment refuses such a
manifest.

**The corpus stop.** Read `$ROOT/corpus/scorecard.txt` (copy in
`modal_results/hyak_tillicum/corpus/scorecard.txt`). The expected verdict line, reproduced at full
scale off-cluster, is

```
verdict: content-identical to every recorded anchor (train bytes and training content of all 23 buckets, the original corpus manifest, regenerated recipe, all held-out documents, hashes and excerpts)
```

With any other verdict do **not** submit stage-inputs or the pool (the job warns, and
`stage-inputs` refuses with exit 3). Compare `$ROOT/env/data.freeze.txt` with the validated set,
delete the bucket directories whose row differs, resubmit `corpus` (about half an hour) and read
the scorecard again. If it still differs, report it: continuing needs your approval (`AGENTS.md`).
Section 10 explains how to read a partial match.

### 6.5 Stage the trainer inputs

```bash
hyak/slurm/submit.sh stage-inputs --debug
```

Runs `python -m hyak.corpus.stage_inputs`, then `--check`. It writes `input/recipe.json`,
`input/corpus/<bucket>/train.jsonl.gz` (symlinks into `$ROOT/corpus`), `input/corpus/manifest.json`
(exactly 23 keys), `input/eval/<bucket>.jsonl.gz`, `input/pool-recipe.json` (read by the pool
workers) and `input/manifest.json` (52 entries: it also registers the three `input/model` files,
which are verified against the pins, so training hashes them). It refuses buckets below their recipe
quota, a missing or wrong model (exit 2, remedy `python -m hyak.model.fetch_model`), a corpus root
inside `$RUNS`, and any change once a pool tokenizer or a run directory exists (unless `--force`
via `--env HYAK_STAGE_INPUTS_ARGS=--force`, which you should not need).

* **The corpus stop (exit 3).** It refuses unless `$ROOT/corpus/scorecard.json` is exact and was
  written for exactly the bucket manifests being staged (a rebuilt bucket makes the scorecard stale:
  run `$ROOT/env/data/bin/python -m hyak.corpus.verify` again, in a job). The only way past a non-exact scorecard is
  `hyak/slurm/submit.sh stage-inputs --env HYAK_STAGE_INPUTS_ARGS="--accept-corpus-scorecard reason-in-one-word"`
  (the words of the variable are split on blanks, so the reason must be one word; the same holds for
  `--allow-unvalidated-env`). It needs your approval, is recorded in `input/manifest.json`
  (`provenance.corpus.scorecard`) and is repeated as a warning by `--check`.
* **Durability.** The links point into `$ROOT/corpus`, which must stay unchanged until both runs
  and their evaluation are finished. Any corpus root other than `$ROOT/corpus` gets a loud warning,
  also in `--check`.
* Good output: the job log ends with `stage-inputs: ... staged and consistent` and the stage status
  is `complete`. Exit codes: 0 staged; 2 configuration or environment; 3 the corpus stop; 4 refused
  or inconsistent. If it fails, the log names the refused bucket or file: a missing or short bucket
  means the corpus is not finished (resubmit `corpus`); a change after pool tokenizers exist means
  the corpus was modified under the pool, which must not happen.

### 6.6 Patched tokenizers wheel

Skip this step with an original pool (section 11).

```bash
bash hyak/pool/build_wheel.sh fetch        # login node: clone at f383101a (tag v0.22.2), 3 patches, cargo fetch
hyak/slurm/submit.sh wheel --debug         # Rust tests + maturin build (offline) + install into env/data
```

The fetch phase needs `git` and internet. The build phase needs a C and a C++ compiler (`onig` is C,
`esaxx-rs` is C++, and `rustc` links with `cc`); the job loads `gcc/13.4.0` if no C compiler is on the
path (`--env HYAK_GCC_MODULE=...` to change), and when only `gcc`/`g++` exist the script chooses them.
A missing compiler stops the phase at once with `module load gcc (e.g. gcc/13.4.0)`. Good output ends
with `install: verified ...`; the stage status becomes `complete`; the wheel and
`$ROOT/wheels/provenance.json` (wheel and patch hashes, toolchain versions, test summary) are in
`$ROOT/wheels`. The wheel goes into the data environment only; the train environment keeps the
stock `tokenizers` (check-env fails if the patched one appears there). If `fetch` fails, a blob id
check or a patch did not apply (the message says which; do not edit the patches); if the job fails,
`$ROOT/wheels/logs/` holds the Rust test log, and the Slurm log the maturin or install error.

### 6.7 Tokenizer pool: pilots and binding tokenizers, full pool, collect, scorecard

Skip this step with an original pool (section 11).

**First: the pilots and the seven binding tokenizers, then a stop-or-go point.** The 12 pilot
indices cover the slow families and are the first measurement of per-tokenizer time and memory. The
seven binding tokenizers (64, 100, 632, 687, 728, 821, 914) have exactly 95 native tokens, the cap
of `scripts/pool_regime.py`, on passages that were valid in the original panel; none is a pilot.
If a regenerated one needs one more token, the complete-pool evaluation stops (section 2.3). Train
all 19 together:

```bash
hyak/slurm/submit.sh pool --count 10 --env HYAK_POOL_ARGS="--indices pilots+binding"
```

(presets: `pilots`, `binding`, `pilots+binding`). Ten jobs x two slots train the 19 at once (2
tokenizers x 4 Rayon threads per 8-CPU job, the original `RAYON_NUM_THREADS=4`). Wall time is the
slowest pilot, likely a no-regex BPE (indices 802 or 329), roughly 1-4 h (*estimate*). Then:

```bash
$ROOT/env/data/bin/python -m hyak.pool.worker --report                       # per-family median/max time and peak memory
$ROOT/env/train/bin/python -m hyak.pool.scorecard --indices pilots+binding   # levels 1 2 3 (--levels 1 needs no model)
```

The scorecard works on a partial pool and prints how many of the seven are identical to the
original at level 1 and how many protected rows are over the cap at level 3 (run it in a job if it
is slow, section 4.4).

* **Go** when the WordPiece pilots 86, 492 and 217 and the BPE pilots 413, 522, 262, 374, 802 and 329
  are identical to the original at level 1 (corpus prefix and wheel equal the original), and each
  binding tokenizer is identical at level 1 or shows 0 protected rows over the cap at level 3.
  (Unigram pilots 398, 663 and 79 may differ in ids and scores; they cannot match at level 1.)
* **Stop** when any protected row is over the cap: `hyak.eval.validate_prefixes --panel pool` would
  stop with exit 3 and no option continues, so the complete-pool table cannot be produced as
  specified (section 2.3). Also stop when no WordPiece or BPE pilot matches: the full pool would be a
  new draw. Either case needs your decision before you pay for the other 981 tokenizers.
* A binding tokenizer that is not identical but has no protected row over the cap may still stop the
  evaluation if a count moves; report it.

From the pilot report:

* If a family peaks above about 70 GiB, two workers could exceed the job memory. Give that family
  its own jobs with one worker: `--env HYAK_POOL_ARGS="--families unigram/no-regex --workers 1"`, and
  run the rest with `--families` listing the other families.
* If a pilot took more than about 4 h, raise the per-tokenizer timeout (default 6 h, the Modal
  timeout) with `--tokenizer-timeout-hours`.
* **More tokenizers per job** (`HYAK_POOL_WORKERS`) is a timing-only knob: every tokenizer keeps
  `RAYON_NUM_THREADS=4`, so files do not change. From `--report` read r = median_cpu_seconds /
  median_elapsed_seconds per family, use at most floor(8 / r) workers, and keep workers x
  max_peak_rss_gib x 1.3 below the job memory. The saving is unmeasured.

**Full pool** (resubmitting is always safe; claims keep jobs from training the same tokenizer):

```bash
hyak/slurm/submit.sh pool --count 20
```

*Estimate*: about 500 GPU-hours (range 300-1,100). A job stops claiming new tokenizers 150 minutes
before its end, so 20 jobs give at most about 430 job-hours: **expect to resubmit**. The pool is done
when `modal_results/hyak_tillicum/pool/status.json` shows 1000 of 1000 and
`$ROOT/input/pool/full-results.json` exists. Choose N from the pilot timings: N x 21 h should exceed
0.55 x the sum over families of (members x median hours) (two workers per job; 21 h is the 24 h
limit minus the 150 minutes). QoS `normal` allows 48 GPUs at once.

* **Status and failures.** `status.json` carries per-family timing. Exit 3 / status `failed` means
  an index failed in that job or failed/blocked indices exist: `--report` lists them. Bounded retries
  happen automatically (at most 3 attempts; `--retry-failed` allows one more); failed tokenizers are
  never replaced by another configuration. A tokenizer whose job was killed with it (for example the
  cluster killing the step for memory) is recorded as an `owner_died` attempt when its claim is taken
  over; a child killed by SIGKILL is reported with the remedy (`--families ... --workers 1`, or
  `HYAK_MEM_POOL`, and `sacct -j <job> -o JobID,MaxRSS,State`).
* **Stopping.** One TERM (also `hyak/slurm/stop.sh pool`) starts no new tokenizer; a second stops
  the running ones at once.
* **Collect and scorecard.** The job that finishes the last tokenizer then runs, in the train
  environment, `python -m hyak.pool.collect --check-only`, `python -m hyak.pool.collect` (verifies
  all 1,000 manifests and writes `input/pool/full-results.json` exactly once, read-only, plus
  `input/pool/plan.json`; it prints the index SHA-256, which both runs record) and
  `python -m hyak.pool.scorecard` (writes `$ROOT/eval/reports/pool-scorecard.{json,txt}`, a few
  minutes; skipped only when the existing report is newer than the index and covers all 1,000 indices
  at all three levels). Neither is stopped by Slurm's early-warning TERM.
* **Job exit status.** A job that finishes the last tokenizer ends 0 even if its own worker had exited
  3. A job whose worker ended normally also ends 0 when the pool is not complete yet (it ran out of
  time: submit more jobs), and ends 3 when its worker failed and the pool is still incomplete. If the
  status is `complete` and `full-results.json` is missing, submit one more pool job (collect exits 75
  when interrupted or incomplete). If you need to run collect by hand, do it in a job (section 4.4),
  not on the login node: it loads all 1,000 tokenizers.

### 6.8 Replay check (corpus and pool fidelity, before GPU training)

```bash
hyak/slurm/submit.sh replay-check
```

CPU work. First `--schedule-only` replays the seed-42 schedule of all 20,000 updates (pool index and
vocabulary size per update, and the final Python random state) against
`reports/rl-tokenizer-selection-20k/data/training/uniform.csv`; this must match 20,000 of 20,000
(it does on the laptop); a mismatch always fails the job (exit 4): a different pool index or
vocabulary size means a wrong plan or schedule (stop and report). Then the first **2,000 updates'**
examples are rebuilt from the staged corpus and pool and compared column by column: `scored_tokens`,
`rejected_examples` and `truncated_forms` depend only on corpus text, seed and tokenizers, so with a
faithful corpus and pool every column matches. 2,000 updates cover 860 of the 1,000 tokenizers and
the original rejections at updates 381, 619 and 1312 (about an hour on one CPU thread,
*estimate*); the tokenizers not drawn are covered by the pool scorecard. The first rows of the
original are (update, index, vocabulary, scored tokens, rejected, truncated) (1, 433, 127000, 8636,
0, 588), (2, 858, 59000, 8755, 0, 332), (3, 85, 80000, 8418, 0, 51). The whole stream can be
replayed with `--env HYAK_REPLAY_CHECK_ARGS="--updates 20000"` (and a longer limit, for example
`--env HYAK_TIME_REPLAY_CHECK=12:00:00`; not needed for the decision).

Reports: `modal_results/hyak_tillicum/replay-check/replay-check-*.json`. An example-column mismatch
is a fidelity result, not an error (exit 0; add `--env HYAK_REPLAY_CHECK_ARGS=--strict` to make it
exit 4): different `truncated_forms` means that update's tokenizer has other vocabulary strings than
the original; different `scored_tokens` with equal `truncated_forms` points at the corpus text, or
(for a Unigram tokenizer, whose scores differ) at its segmentation. An update at which the uniform
trainer would raise (`Too many non-roundtripping or oversized examples`) fails the job with exit 4
and status `failed`: the real run would crash there. Exit 2 means an input is missing or the staged
helper files differ from the pins (staged code, corpus, pool or model; remedy `python -m
hyak.code.stage_code --verify`).

### 6.9 Smoke test of the uniform run (required)

```bash
hyak/slurm/submit.sh smoke-train --run uniform
```

Not enforced by the scripts, but do it: it is the first time a pinned trainer runs on the cluster
(section 12). In `$RUNS/smoke/uniform` (wiped first): the bootstrap (one update), then a resume to
update 3 in a new process, with the production checkpoint and evaluation settings. The default
limit is 2 h with a 10-minute margin (`HYAK_TIME_SMOKE_TRAIN`,
`HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_TRAIN`). Exit 0 only when every check passes; look for
`smoke_verified` in `modal_results/hyak_tillicum/smoke-train/<jobid>-train.log`, and the status
`complete`. The first `train` lines there give the first H200 numbers: `seconds` per update and
`peak_gib`. `stop_timer_seconds` in the `segment_start` and `smoke_verified` records is the second
thing to read: `null` under Slurm means no end time was found, so only Slurm's signal could stop a
segment gracefully (and a warning was emitted). Delete `$RUNS/smoke/uniform` afterwards (two full
restarts, about 47 GB). If it fails, the Slurm log has the reason (an out-of-memory error, a
refused input, a failed check); fix it before the real run. Do not use `--debug` for smoke or
training jobs: a segment starts only when at least 20 minutes plus the margin remain, more than the
debug QoS allows (`submit.sh` refuses it). The RL smoke (6.11) runs the same sequence with the RL
variant trainer and additionally checks the paired-protocol receipt after each phase.

### 6.10 The uniform run

```bash
hyak/slurm/submit.sh train-uniform --count 4
```

Four 24 h jobs chained with `--dependency=singleton` (one job of the run at a time). Each job:

* refuses to start while `modal_results/hyak_tillicum/train-uniform/HALTED.json` exists (exit 3,
  section 8);
* starts the status monitor, then runs segments of `python -m hyak.train.run_segment --run uniform`
  while the run is incomplete and at least 20 minutes plus `HYAK_SEGMENT_MARGIN_MINUTES` (45) remain;
* the first segment of a fresh run is the one-update bootstrap (pool index 433, 127,000 rows); a
  resume segment follows in the same job. The bootstrap's update 1 runs before the AdamW moments
  exist and peaks about 14 GiB below steady state, so it is not the real memory test: the probe and
  the first 128,000-row tokenizer (update 261) are;
* each segment stops itself 50 minutes before the job's end (an in-process timer at the remaining
  time minus the margin minus 5 minutes; Slurm's `--signal=B:TERM` 45 minutes before the end is the
  backup): the trainer finishes the update, evaluates, writes a full restart and records
  `paused_at`.

The trainer's JSON lines go to `modal_results/hyak_tillicum/train-uniform/<jobid>-train.log`; the
wrapper's messages and any traceback to `slurm-<jobid>.out`. After the bootstrap the pair is frozen:
`input/uniform_run_config.json`, `input/uniform_heldout_tokenizers.json` and `input/pairing.json`
exist and `events.jsonl` has a `pairing_frozen` event. From then on the uniform run directory must
never be restarted from scratch (the wrapper refuses).

How many jobs: after the first job, read `seconds_per_step` in `status.json` (or the `seconds` field
of the `train` lines). Jobs needed is about 20,000 x seconds per update / (24 h - 50 min - start-up),
where start-up includes hashing the 23.5 GB restart and loading the models. Extra jobs are harmless:
a job whose run is complete exits 0 at once. Submit more at any time with the same command.

**What must not change until update 20,000.** The wrapper compares these with
`<run>/hyak_segment.json` and `input/pairing.json` at every segment and refuses to resume otherwise
(exit 3 or 4): `HYAK_HN_CHUNK_ROWS` and the file `hyak/memory/chunked_hypernet.py` in the checkout
(it is imported from the checkout, not from the staged code; its SHA-256 is part of the record; its
memory prediction now lives in `hyak/memory/memory_model.py`, which may be edited freely);
`ANYTOK_ROOT`, `ANYTOK_RUNS`; the staged code; `input/hypernet_init`; the corpus, the recipe and
the held-out files (all hashed through `input/manifest.json`); the pool index; and the model
directory, which must hold exactly `config.json`, `model.safetensors` and `tokenizer.json`, each
equal to its pin (hashed in full at every segment). **Do not update the checkout (`git pull`,
edits) while training jobs are queued or running.** A software-environment change between
segments (a different torch, say) is a loud warning and an `environment_changed` event, not a
refusal. A manifest marked test-scale is refused by every segment.

If the chunk setting must change mid-run (for example after an out-of-memory failure late in a
run), accept it on the record, once: stop the chain (section 8), edit `HYAK_HN_CHUNK_ROWS`, and
resubmit the chain with the reason as a job variable, for example
`hyak/slurm/submit.sh train-uniform --count 4 --env HYAK_ACCEPT_CHUNK_CHANGE="out of memory at update 13,500 with 16384 rows"`.
The job passes it to `python -m hyak.train.run_segment --run <run> --accept-chunk-change "<reason>"`;
segments whose setting is unchanged ignore it. The change and the reason go to
the events (`chunk_change_accepted`) and to `hyak_segment.json` (`chunk_changes`); for the RL run the
difference from the uniform run is recorded in `accepted_pairing_differences`. Later segments need
no flag. Any other mid-run difference (code, initial hypernetwork, model files, paths) is still
refused. A different chunk size also changes the random stream of dropout; say so with the results.

### 6.11 The RL run

Only after the uniform bootstrap froze the pair (6.10), that is, after the `pairing_frozen` event in
`modal_results/hyak_tillicum/train-uniform/events.jsonl` (`$ROOT/input/pairing.json` exists):

```bash
hyak/slurm/submit.sh smoke-train --run rl          # 2 updates, then a resume to 3; policy counters must equal 3
hyak/slurm/submit.sh train-rl --count 4
```

The RL run is the straight-BPB-reward counterpart of Qingyun's joint regret RL20k (D8). Its staged
tree (`$ROOT/code/rl`) is the regret launcher's image with the variant trainer in place of the
regret trainer (`$ROOT/code/rl/scripts/train_joint_smollm_input.py`), and every RL segment runs it with `--policy-entropy 0
--policy-lr 0.001 --policy-baseline-decay 0.99` (as the regret launcher does). The trainer itself
compares 21 settings with `input/uniform_run_config.json` (the uniform bootstrap's config) and stops
if any differs; these are unaffected by D8.

`submit.sh` refuses `train-rl` and `smoke-train --run rl` while `input/pairing.json` is missing. An
RL job that starts before the bootstrap froze the pair anyway ends with exit 0 (the segment
answered 75, reason `uniform_bootstrap_pending`) and status `paused`: no `HALTED.json`, nothing is
spent; wait for the event and resubmit. The RL run may train while the uniform run continues, on a second GPU.

**What each RL segment and each smoke phase validates.** After the trainer returns, besides the
checks of every segment (the newest full restart verified, its `state.pt` step, `bandit.updates ==
counts.sum() == step`, progress, the trainer's last event), the wrapper checks the newest full
restart like the regret launcher's receipt, plus the straight reward: in `run_config.json` the
policy has entropy weight 0 (and learning rate 0.001, baseline decay 0.99, seed 43) and
`policy_reward` is `paired_pre_model_update_transferred_bpb`; in the state's metrics
`paired_valid_examples == 128`, `paired_maximum_positions <= 128`, `|paired_gap_bpb - (train_bpb -
paired_canonical_bpb)| <= 1e-10` and `|policy_reward_bpb - train_bpb| <= 1e-12` (the policy received
the straight BPB). The receipt is the `paired_protocol_verified` event (in the smoke's
`smoke_verified` record as `paired_protocol`). At step 0 (the initial checkpoint, no update yet) only
`run_config.json` is checked. A failure is a check failure: exit 4, `HALTED.json`, the chain stops.
The smoke runs these checks after each of its two phases and again at step 3.

**The step-0 evaluation is not compared with the uniform run's (narrowed check).** The RL trainer
builds its sanity examples with the corrected paired cuts, the uniform trainer with the original
cuts, so the two `initial_eval.json` differ by design. After a fresh start the wrapper only checks
that the RL run's `heldout_tokenizers.json` equals the frozen `input/uniform_heldout_tokenizers.json`
(a difference is a warning); the event `initial_eval_pairing` records `initial_eval_equal`
(information only), `sanity_tokenizers_equal` and the reason (`check`). The dashboard's learning
panel shows the RL sanity curves without the uniform overlay (section 7). The RL job also refreshes
`modal_results/hyak_tillicum/train-rl/policy-monitor.json` after each 500-step checkpoint for the
dashboard's learning panel. Delete `$RUNS/smoke/rl` afterwards.

**Time.** Every RL update adds one no-grad forward of the LM over the 128 canonical sequences (the
control that the regret run needs for its reward and that this run keeps as a diagnostic). Expect
roughly 15-25% more time per update than the uniform run (*estimate*, not measured: a forward pass is
about a third of a forward and backward pass of the LM, and the hypernetwork forward and backward are
unchanged); read `seconds` in the first RL `train` lines and size the chain from it (6.10: the
formula), which may need a fifth 24 h job. Its peak memory is expected to stay below the training
forward's (no activations are kept), also an *estimate*. The RL run writes full restarts only
every 2,000 updates, so a hard kill can cost it up to 2,000 updates (section 8). If its stop timer
fires while a weights-only checkpoint (500, 1000 or 1500 past a multiple of 2,000) is being saved,
it waits for that save, at most 15 minutes; if a stop signal still lands inside such a save, the
segment ends `paused` at the newest full restart with a warning (exit 0), and the next segment
replays at most 1,500 updates.

### 6.12 The end of a run and the final audit

At update 20,000 the segment runs the 40-checkpoint grid audit and writes `<run>/final_audit.json`
(uniform: 40 checkpoints, 10 full; RL: 40 checkpoints, `bandit.updates == counts.sum() == 20000`),
and the status becomes `complete`. Hashing about 493 GB takes time; while the audit runs the status is
`auditing`; if the job ends first it is `paused` and the next job of the chain finishes it (submit
one more if none is queued); a failed audit is `failed`. The dashboard shows `complete` only when
the audit passed. A job of a complete run exits 0 at once. Then archive the finals (section 9).

```bash
$ROOT/env/train/bin/python -m hyak.train.retention summary --run uniform    # read-only listing
```

### 6.13 Evaluation preparation

Can run as soon as the pool is collected, long before training ends (do it early: it may stop for a
decision, section 2.3):

```bash
hyak/slurm/submit.sh eval-prepare
```

In order: rebuild both stored panels by decoding the stored token ids and check every hash
(`$ROOT/eval/panels/<panel>/rebuild-report.json`: 576/576 pool rows and 2,032/2,032 natural rows,
panel hashes `30ee5e0d...` and `5c8ad1c6...`; this passed on the laptop); build features for all
1,000 tokenizers; validate the prefixes of both panels for the pool in use with the repository's
trimming loop. Read `$ROOT/eval/panels/<panel>/validation-report.json`: `comparison.rows_trimmed` and
`comparison.trimmed[]` list every prefix that had to be shortened, and `rounds[].rows[].tokenizers`
the tokenizers that forced each trim (index: characters).

Exit 3 is a protocol stop. The pool panel validates first, then the natural panel (even when the
pool panel stopped: it does not depend on it); the job then ends with 3 and names each stopped panel.
For the pool panel an originally valid context that would be trimmed stops it with
`valid_context_trims` (one record per row) and `decision` in the report; the options and the
approval they need are in section 2.3. The extra-argument variable `HYAK_EVAL_PREPARE_ARGS` reaches
both `validate_prefixes` calls (for example `--workers 4`, `--recheck`, or the accepted-trims flag
after your approval; the flag has no effect on the natural panel). Nothing is dropped
automatically. A failed panel rebuild (exit 2) means the stored passages did not decode to their
hashes with the pinned SmolLM2 tokenizer: check `input/model` with `fetch_model --verify-only`; it
passed on the laptop with the same file.

### 6.14 Scoring both checkpoints

A four-tokenizer pilot first (pool panel only, no packing; index 192 is one of the seven tokenizers
with the largest vocabulary, 128,000 rows: 192, 235, 688, 790, 850, 971, 989):

```bash
hyak/slurm/submit.sh eval-rl --env HYAK_EVAL_INDICES="0 616 822 192"
```

Read `elapsed_seconds` and `peak_gib` in `$ROOT/eval/scores/rl/pool/NNNN.json`. Then:

```bash
hyak/slurm/submit.sh eval-uniform --count 2
hyak/slurm/submit.sh eval-rl --count 2
```

Each job scores `$RUNS/<run>/step-020000` (hash-checked against its manifest) on the pool panel
(1,000 tokenizers x 576 passages, canonical and BOS-only controls computed once), then the natural
panel (38 tokenizers x 2,032 passages), then packs. *Estimates*: 5-15 s per tokenizer, about
1.5-4 GPU-hours per checkpoint for the pool panel and 5-20 minutes for the natural panel; each job
start adds 3-5 minutes. Resubmitting is safe; work is shared by claims: after each pass a job waits
15 s and retries tokenizers claimed by other jobs, takes over a claim only when `squeue` no longer
lists its job or it is older than 60 minutes, and a claim lost anyway is a warning, not a failure.
Status: `modal_results/hyak_tillicum/eval-<run>/status.json` (1000/1000 `pool panel complete; natural
panel not complete yet`, then the natural panel; `complete` only when both panels are, for the same
checkpoint). To score the natural panel alone (for example after a pool-panel stop):
`hyak/slurm/submit.sh eval-rl --env HYAK_EVAL_PANELS=natural`.

**Scoring an archived copy.** The scoring plan identifies the checkpoint by the SHA-256 of its
training `manifest.json`, not by its path, so a copy of `step-020000` (with its `manifest.json`)
on project storage can be scored, or can continue an existing scores directory:
`hyak/slurm/submit.sh eval-rl --env HYAK_EVAL_SCORE_ARGS="--checkpoint /gpfs/projects/<group>/<you>/anytok-data/checkpoints/rl/step-020000"`.
Every result file records the `checkpoint_path` it came from. A checkpoint with another manifest is
refused (`Comparison plan changed`), and `--allow-non-final-checkpoint` (tests and diagnostics) is
the only way to score a step other than 20,000. A checkpoint whose files do not match its
`manifest.json` is refused (never score it); `--verify-only` (in a job) checks everything without
loading the model.

**What invalidates scores.** The panel plan records the SHA-256 of the pinned `hf_hypernet` files and
of the imported scripts (`scripts/natural_transfer.py`, `pool_regime.py`, and the others the
evaluation imports); the scoring plan records the SHA-256 of `hyak/eval/__init__.py` and
`hyak/eval/score.py`. `score` and `pack` refuse a plan whose hashes differ from the current files
and name them (restore the files with `git checkout`, or re-validate with `--force` and rescore).
Edits to `pack.py`, `prepare_features.py`, `validate_prefixes.py` or `rebuild_panel.py` do not
invalidate scores. Exit codes: `score` 0 when its work list is fully scored or it stopped on a
signal or the time margin, 75 when indices are still held by other live jobs after the bounded wait;
`pack` 0, or 75 while a run is not completely scored (the job ends 0).

### 6.15 Results and comparison

The scoring jobs run `python -m hyak.eval.pack` for each panel (exit 75 there only means the other
run is not complete yet). Results in `$ROOT/eval/reports/` (never in the repository's `reports/`):

* `pool/<run>/reward_table.json.gz` (the original reward-table format), `pool/<run>/summary.json`
  (nine-bucket balanced bits per byte, canonical and BOS-only controls, gaps, ranks, per split),
  `pool/<run>/rankings.csv`, and `pool/comparison.{json,csv}` (uniform versus RL, paired by
  tokenizer, with the original RL numbers as `reference_original`);
* `natural/results.json.gz` and `natural/summary.json` (five-domain balanced results, RL minus
  uniform paired effects, the original three-model reference; the pack requires the 38 indices and
  2,032 rows of the plan unless `--partial`, which marks the summary `partial: true`).

**Names.** The models are `uniform20k-hyak` (the uniform run) and `rl20k-straight-entropy0-hyak`
(the RL run of D8: straight transferred-BPB reward, entropy 0, corrected paired cuts, random
initialization). The natural panel's paired effects are named
`rl20k-straight-entropy0-hyak-minus-uniform20k-hyak`. The stored original `rl20k` (and the natural
panel's `zett20k` and `uniform20k`) appear only under `reference_original`, which carries
`relation` (`different_run` / `different_runs`), `reproduction_target: false` and a note: the
original `rl20k` used entropy 0.01, the raw training-BPB reward, the original example cuts and the
private warmup initialization, so it is a reference from a different run, never a target the Hyak
RL run should reproduce.

**Comparisons.** The paired comparison is Hyak RL against Hyak uniform (same initialization, pool,
corpus, hardware and evaluation panels). A comparison with Qingyun's joint regret run isolates the
reward only up to the initialization and the other confounds of section 2.6.

Every summary and `pool/comparison.json` carries a `panel` block (`panel_sha256`,
`stored_panel_sha256`, `equals_stored`, `trimmed_rows`, `trimmed_row_ids`, `trimmed_chars`,
`accepted_valid_context_trims`, `validation_rounds`, `pool_index_sha256`, `indices`), and `pack` logs
how many rows were trimmed relative to the stored panel: report them with the results.

The original numbers are a plausibility reference from different runs, never a target (different
protocol for RL (D8), pool, initialization and hardware). Original RL 20k on the complete pool: screen split canonical 0.96692, uniform-mean gap
+0.06690, worst tokenizer #0635 +0.10396; confirm split 0.96534, +0.08883, #0635 +0.13784. Natural
panel, confirm contexts, weak-pretokenization median gaps: ZeTT +0.41835, uniform +0.14240,
RL +0.13437; RL minus uniform -0.00809 [-0.01236, -0.00320]. The uniform checkpoint has no original
complete-pool counterpart.

To repack by hand (CPU, about a minute, imports torch; better in a job):
`$ROOT/env/train/bin/python -m hyak.eval.pack --panel pool`. `scripts/frozen_policy_search.py --cache
<reward_table.json.gz>` can load a new table (the policy search that is not rerun by the port), but
its `main()` rewrites `reports/` and `plots/` in the checkout and needs matplotlib (not in the train
environment): use a throwaway copy of the repository if you want it.

**Getting the results.** The packed results are small (a few MB per table); copy them and the
monitor directories to your laptop (never the checkpoints, `AGENTS.md`):
`rsync -a <netid>@tillicum.hyak.uw.edu:<ROOT>/eval/reports/ ./tillicum-results/eval-reports/` and
`rsync -a <netid>@tillicum.hyak.uw.edu:/gpfs/projects/<group>/<you>/anytok/modal_results/hyak_tillicum/ ./tillicum-results/monitor/`
(with `<ROOT>` written out). Cleaning up afterwards: section 9.

---

## 7. Watching jobs and the dashboard

```bash
squeue -u $USER                                                # job names are anytok-<stage>
tail -f modal_results/hyak_tillicum/<stage>/latest-slurm.out   # newest job of a stage
python3 -m hyak.common.status show --stage train-uniform        # current status.json
```

The dashboard: every job registers the `hyak-*` entries in `plans/run-dashboard.json` (a modified
tracked file in the cluster checkout; it holds only repository-relative paths and no cluster paths,
but jobs re-create it, so keep it out of your commits; `git checkout plans/run-dashboard.json`
discards it). The entries are appended after the existing ones, so the dashboard's default view is
unchanged: **pick a `hyak-*` entry in the run list.** Run the dashboard on the login node in a
terminal multiplexer and tunnel it:

```bash
# on your laptop: one SSH session that carries the tunnel and runs the dashboard on the same login node
ssh -L 1337:127.0.0.1:1337 <netid>@tillicum.hyak.uw.edu
cd /gpfs/projects/<group>/<you>/anytok
tmux new -s dashboard
ROOT=$(python3 -m hyak.common.config --get ANYTOK_ROOT)
$ROOT/env/train/bin/python scripts/run_dashboard.py --offline --host 127.0.0.1 --port 1337
```

Then open http://127.0.0.1:1337 on the laptop. The tunnel must end on the login node where the
dashboard runs (there are two login nodes: check `hostname` if you reconnect; use another port if
1337 is taken on a shared node). The dashboard only reads files; it has no job controls. Entries:
`hyak-uniform-training`, `hyak-rl-training` (named "Joint RL 20k · straight BPB reward · entropy 0";
with the learning panel, which, like Qingyun's regret entry, shows a protocol note and does not overlay
the uniform run's sanity scores: they use the original cuts, D8), `hyak-eval-uniform`,
`hyak-eval-rl`, `hyak-eval-prepare`, `hyak-tokenizer-pool`, `hyak-corpus`, `hyak-memory-probe`.
Stages without an entry (check-env, wheel, stage-inputs, init, replay-check, smoke-train) still
write `status.json` and `events.jsonl` in their monitor directory.

What to expect:

* With `--offline` the page header reads `OFFLINE · CACHED DATA` and the footer `Awaiting Modal
  metadata`, although the Hyak entries are read from files every 2 seconds. That is normal.
* The dashboard reads the registry once, at start: a dashboard started before the first job
  registered the entries shows none of them until it is restarted. Run
  `python3 -m hyak.tools.register_dashboard` once on the cluster before starting it.
* Clicking the profile row (`tillicum/gpu-h200`) empties the log pane (logs are filtered by a
  directory named like the profile); select a worker row to see that job's Slurm log.
* A stage shows `complete` only when the stage itself says so. At update 20,000 a run shows
  `auditing` while the final audit runs, `paused` if the job ended cleanly before it finished
  (resubmit; it only audits), `failed` after a failed audit; a pool with a refused collect and a
  corpus with a failed bucket show `failed` at 1,000/1,000 and 23/23.
* The Pool worker rows show the free-text task (`0413 bpe/regex 96k: training`).
* Status writers ask `squeue` once per process and minute for all of your jobs (a few calls per
  minute for N concurrent jobs), with all `SQUEUE_*` variables removed; an empty or failing answer is
  never read as "job finished".

**Stale rows after a hard kill.** A job killed hard (node failure, time limit without a graceful
stop, plain `scancel`) never writes its end, so its stage keeps saying `running` with a frozen ETA
until another job of the stage starts. On the login node (read-only `squeue`; not part of the
dashboard):

```bash
python3 -m hyak.tools.reconcile --dry-run          # say what would be marked; write nothing
python3 -m hyak.tools.reconcile [--stage STAGE ...] # mark dead jobs' stages "stopped"
```

(`hyak/tools/reconcile.py`; for example under `watch -n 300`.) For every stage whose `status.json`
says `running` or `waiting` it marks the stage `stopped` only when every job named there is
definitely finished, keeping progress and extra fields and recording a `reconciled` event. It leaves
a stage alone when any job is alive or unknown, when any job of yours named `anytok-<stage>` is
pending or running (that job corrects the status itself), or when the status changed meanwhile;
terminal states are never touched. One JSON line per examined stage; exit 0 (also when nothing is
running), exit 2 when `squeue` cannot be asked. `squeue` is always the truth.

---

## 8. Chains: stop, resume, extend, repair

* **Stop a run gracefully:** `hyak/slurm/stop.sh <stage>` (for example `hyak/slurm/stop.sh
  train-rl`; add `--dry-run` to see what it would do). It first cancels the stage's PENDING jobs
  (a singleton chain ignores exit states, so a queued successor would start the moment the running
  job is gone), then runs `scancel --batch --signal=TERM` on the RUNNING job. Only the batch shell is
  signalled; it forwards the TERM, the trainer finishes the update, evaluates, writes a full
  restart, and the job ends with exit 0 and status `paused` after some minutes (watch
  `latest-slurm.out`). Resubmit the stage to resume. A pool worker starts no new tokenizer on one
  TERM (a second TERM stops the running ones at once); a scoring job ends its current tokenizer. This
  was not yet tried on Tillicum (section 12). **Plain `scancel <jobid>` is a hard kill:** TERM to
  every process and KILL after the cluster's KillWait (`scontrol show config | grep -i killwait`,
  about 30 s), which is shorter than a 23.5 GB save plus its hashing, and it does not stop the queued
  successors. Command line only: nothing in the dashboard starts or stops a job.
* **Extend a chain:** submit more jobs with the same command
  (`hyak/slurm/submit.sh train-uniform --count 2`). Singleton jobs run one after another; each
  resumes from the newest verified full restart.
* **A failed training job halts its chain.** Any end except 0, 130 and 143 writes
  `modal_results/hyak_tillicum/train-<run>/HALTED.json` (job id, exit status, both logs, reason),
  and every later job of that chain exits 3 at once without touching the run. Exit 1 is a crash
  (`slurm-<jobid>.out` has the traceback), 2 a configuration or input error, 3 a protocol stop
  that needs your decision, 4 a failed check, 137 or another 128+n a killed trainer (status
  `failed`, usually the memory limit). Inspect the logs it names, fix the cause, delete
  `HALTED.json`, resubmit. Exit 75 ("not now": the RL run waits for the uniform bootstrap, or the
  run lock was still held) ends the job 0 and halts nothing. A segment that exits 0 without a newer
  full restart is treated as a failure (exit 4), as the Modal controllers did, except while only the
  final audit is pending.
* **After a hard kill** (node failure, time limit without a graceful stop, plain `scancel`) the next
  segment repairs what the kill left: stale `.writing-*` staging directories are removed;
  checkpoints newer than the newest full restart move to `<run>/interrupted/<utc>/`; a run whose
  first save never completed is cleared into `$RUNS/cleared/`. **Up to 2,000 updates are at risk in
  both runs.** The RL run writes full restarts only every 2,000 updates. The uniform run normally
  loses at most 500, but up to 2,000 when the kill hits the first grid save after a resume from an
  off-grid stop (every job boundary): the pinned trainer deletes the off-grid restart before that
  save completes. This is the original behaviour, kept on purpose; it is one more reason to use
  `stop.sh`.
* **A newest restart without `manifest.json`** (a kill between the trainer's rename and the manifest
  hook) is adopted automatically at the next segment start when its directory holds exactly a full
  restart's files (a warning and an `adopted_restart` event). Anything else stops with exit 3 and
  names the remedy; half-deleted remainders name `retention quarantine` only when an older full
  restart exists, otherwise move the RL run directory away and restart RL, or restore the uniform
  run.
* **The run lock.** A segment and every repair command take the run's lock. A segment waits up to 10
  minutes for it (events `waiting_for_run_lock` and `run_lock_acquired`), then exits 75. (This
  assumes that `flock` on GPFS excludes across nodes; `check-env` tests it on one node.)
* **Repair commands** (they take the run's lock, so not while a segment runs; `verify`, `audit` and
  `adopt` hash whole checkpoints, 23.5 GB per full restart and 493 GB for the audit, so run them in a
  job, section 4.4):

  ```bash
  $ROOT/env/train/bin/python -m hyak.train.retention summary    --run uniform
  $ROOT/env/train/bin/python -m hyak.train.retention verify     --run uniform [--step N]
  $ROOT/env/train/bin/python -m hyak.train.retention audit      --run uniform
  $ROOT/env/train/bin/python -m hyak.train.retention adopt      --run uniform --step N   # write a missing manifest
  $ROOT/env/train/bin/python -m hyak.train.retention quarantine --run uniform --step N   # move a bad checkpoint aside
  ```

  A segment that refuses to start names the exact one of these to run. `quarantine` refuses a step
  below the newest full restart and the only full restart unless `--force` is given; it prints the
  step the run will resume from, the updates given up and the exact `mv` that undoes it. Every new
  checkpoint is fsynced before older restart state is compacted or deleted; a kill inside the last
  save's retention step is completed by the next job before the final audit.
* **An RL job before the uniform bootstrap** ends 0 with status `paused` (reason
  `uniform_bootstrap_pending`), writes no `HALTED.json` and costs nothing; `submit.sh` refuses the
  submission in the first place. Wait for the `pairing_frozen` event of the uniform run.
* **What must not change during a run:** the chunk setting and `hyak/memory/chunked_hypernet.py`, the
  staged code, `input/hypernet_init`, the corpus, recipe and held-out inputs, the pool index, the
  model directory, `ANYTOK_ROOT` and `ANYTOK_RUNS`, and the checkout (no `git pull`): section 6.10. If
  you must change the chunk setting, accept it on the record with `--accept-chunk-change` (6.10); any
  other difference is refused.
* **Starting both runs again from scratch.** Move away together: `$RUNS/uniform`, `$RUNS/rl`,
  `input/pairing.json` and `input/uniform_*.json`. The monitor directories need no manual step: a
  run that starts from scratch moves the stage's earlier monitor files to
  `modal_results/hyak_tillicum/train-<run>/previous-<utc>/`, and the dashboard starts clean.
* **Keep `$RUNS/uniform` where it is** (copy, do not move) until the RL run is complete: every start
  of either run compares the uniform `run_config.json` and `heldout_tokenizers.json` with the frozen
  copies.
* `<run>/interrupted/<utc>/` (8.6 GB per moved weights-only checkpoint) and `$RUNS/cleared/` (up
  to 23.5 GB) are kept for diagnosis only; nothing reads them; delete them once the run has passed
  those steps. A stale `final_audit.json` (a run that trained again after step 20,000) is renamed
  `final_audit.stale-<utc>.json`.
* **Pool, corpus and evaluation jobs** are claim-based or skip finished work: resubmit them.
* `$RUNS/.hyak-cache/` caches input hashes (by size and modification time); deleting it only costs
  re-hashing.

---

## 9. Storage budget, the 60-day purge and keeping the checkpoints

| Location | Content | Size |
|---|---|---|
| `$RUNS/uniform`, `$RUNS/rl` (scrubbed) | 40 grid checkpoints each: 10 full restarts (23.5 GB) + 30 weights-only (8.6 GB) | 493 GB per run at the end; peak about 540 GB while a save is staged |
| `$RUNS/smoke` | smoke runs | about 47 GB (uniform) + 23.5 GB (RL); delete after the smoke |
| `<run>/interrupted/`, `$RUNS/cleared/` | debris of hard kills | 8.6 GB per moved weights-only checkpoint, up to 23.5 GB; delete when passed |
| `$ROOT/env` | two conda environments, pip/conda caches | train environment about 7 GB installed plus about 3 GB of pip cache, plus the data environment (not measured on the cluster; the caches can be deleted) |
| `$ROOT/toolchain`, `$ROOT/build` | Rust toolchain, tokenizers checkout and build tree | a few GB (not measured) |
| `$ROOT/input/model`, `input/hypernet_init` | SmolLM2, initial hypernetwork | 3.4 GB + 1.4 GB (4.4 GB more for a short time while init builds) |
| `$ROOT/corpus` | 23 buckets | about 5 GB of gzip (4.7 GB measured) |
| `$ROOT/hf-cache` | Hugging Face downloads | up to about 8 GB downloaded (*estimate*; a laptop build left 528 KB cached) |
| `$ROOT/input/pool` | 1,000 `tokenizer.json` | about 4 GB |
| `$ROOT/eval` | features (80,085,000 vocabulary rows x 10 int32 + priors) and per-tokenizer scores | about 3.5 GB of features plus scores |
| `$ROOT/checkpoints` (section 9, below) | the two final checkpoints and the small run files | 47 GB + a few MB |

So about 1.1 TB in scrubbed storage and about 100 GB in project storage (50 GB of data and
environments plus the archived finals).

**`$ROOT/corpus` must outlive both runs and their evaluation:** `input/corpus/<bucket>/train.jsonl.gz`
are links into it, and both trainers refuse to resume when `input/corpus/manifest.json` changes.
Do not put it on scrubbed storage. A lost bucket can be rebuilt byte for byte (section 6.4).

**Scrubbed storage deletes files that were not accessed for 60 days** (the official wording;
working around the deletion violates the usage policy). The early grid checkpoints are written once
and read again only by the final audit, so count 60 days from the day each file is written. Plan to
finish training and the audit within 60 days of the first checkpoint, or the audit can never pass.
Project storage has room for the two final restarts (47 GB) but not for both grids (987 GB).
Scrubbed storage is also readable by other users by default (section 4.1): check the mode of
`$RUNS` before the first checkpoint is written.

### Keeping the checkpoints

As soon as `final_audit.json` exists for a run, copy the final restart and the small run files to
project storage, then verify the copy against its training manifest. The copy is `rsync` (a few
minutes); the verification hashes 23.5 GB, so run that line in a job (section 4.4):

```bash
ARCH=$ROOT/checkpoints
for run in uniform rl; do
  mkdir -p $ARCH/$run
  rsync -a $RUNS/$run/step-020000 $ARCH/$run/
  rsync -a --prune-empty-dirs --exclude='.writing-*' --exclude='/interrupted/' --include='*/' \
        --include='manifest.json' --include='policy.pt' --include='/*.json' --include='/events.jsonl' \
        --exclude='*' $RUNS/$run/ $ARCH/$run/
  $ROOT/env/train/bin/python -c "from hyak.train import retention as r; print(r.verify_checkpoint('$ARCH/$run', 20000, '$run', r.FULL)['bytes'])"
done
```

(run the last line from the repository root). The second `rsync` keeps every `step-*/manifest.json`,
every RL `policy.pt`, `run_config.json`, `heldout_tokenizers.json`, `initial_eval.json`,
`events.jsonl`, `hyak_segment.json` and `final_audit.json`. The scorer accepts the archived copy:
section 6.14 (`HYAK_EVAL_SCORE_ARGS="--checkpoint ..."`). The 40-checkpoint grids exist only on
scrubbed storage: if they must outlive 60 days, move them to Kopah (S3) or Lolo (archive), the two
targets the Hyak storage page names (how is not part of this port), or let them expire; they are
not needed after the final audit and the scoring.

### Cleaning up

After the results are copied (section 6.15): `$ROOT/build` and `$ROOT/toolchain` (after the wheel and
the Rust library are installed), `$ROOT/env/.pip-cache` and `$ROOT/env/.conda-pkgs`, `$ROOT/hf-cache`
(after the corpus is built; a rebuilt bucket would download again), `$RUNS/smoke`, `$RUNS/cleared`,
`<run>/interrupted`, `$ROOT/input/pool/claims` and `$ROOT/eval/claims`. Keep `$ROOT/corpus`, `$ROOT/input` and
`$ROOT/eval` for as long as you want to rescore or rerun anything. Check what the experiment cost
with Hyak's usage reporting (for example `hyakusage`; not verified here) and your account's budget
page; the cost estimate at submission is based on the requested time, so short limits help.

---

## 10. Fidelity reports and how to read them

| Report | Where | What a faithful rebuild shows |
|---|---|---|
| Corpus scorecard | `$ROOT/corpus/scorecard.txt` | the exact verdict `content-identical to every recorded anchor`: raw training bytes equal the recipe in 23/23 buckets; regenerated recipe byte-identical; training content (hash of the decompressed pool file) 23/23; sizes of the 23 original staged training files 23/23 (sum 4,023,420,409 bytes); the rebuilt original `corpus/manifest.json` hashes to its pin (`363fbd07...`); 3,149/3,149 held-out ids found; 1,592/1,592 document SHA-256 equal; 34/34 excerpts; held-out files byte-identical to the originals once the stored original gzip header time is written in, 23/23 |
| Pool scorecard | `$ROOT/eval/reports/pool-scorecard.txt` | level 1: `tokenizer.json` SHA-256 equals the original (`reports/rl-pool-sweep-20k/spec.json`); level 2: `truncated_forms` equals the original run's; level 3: native token counts on the 576 stored prefixes equal the stored ones, and no protected row is over the 95-token cap. Unigram members cannot pass level 1 (training is not reproducible); they can pass levels 2 and 3. `--indices pilots+binding` gives the early version (section 6.7) |
| Replay check | `modal_results/hyak_tillicum/replay-check/` | schedule: 20,000/20,000 rows and the final random-state digest; examples: every column of the first 2,000 updates equal |
| Prefix validation | `$ROOT/eval/panels/<panel>/validation-report.json` | no trimmed row (`panel_sha256_equals_stored` is true only for a pool identical to the original) |

**The corpus scorecard was reproduced exact at full scale** (a build off-cluster on 2026-11-13,
the scorecard run again on 2026-11-14, with the pinned package set: `datasets` 5.2.0, `pyarrow`
26.0.1, `huggingface-hub` 2.4.0, `hf-xet` 1.8.0, `fsspec` 2026.9.0, `requests` 2.34.2, `urllib3`
2.8.0). The scorecard run takes about 72 s wall with 2 workers on a laptop (`--skip-train-files`
makes it fast but the verdict is then never exact). So a non-exact scorecard on the cluster is an
anomaly of that build, not a limit of regeneration. The reading rules (equal values are strong
evidence, not proof: the stored panels hold document hashes for 9 of the 23 buckets, so the
training-content and corpus-manifest anchors carry most of the weight):

1. Train bytes, training content, the re-created staged document count or the rebuilt original
   `corpus/manifest.json` differ in a bucket: its training text differs from the original. **Stop.**
2. A missing held-out id or a different document hash in a hash-held-out bucket (web-en, wiki-*,
   code, math-en, qa-en, books-en, medical-en, legal-en): the stream differs near its start. **Stop.**
3. Only the held-out anchors of an upstream-split bucket (web-es/fr/ru/zh/ja/ar/hi, papers-en)
   differ: only its held-out file differs; delete that bucket and rebuild it.
4. Only compressed sizes or held-out file identity differ, in every bucket: another zlib build, not
   content. When the zlib known-answer test fails, those anchors are reported as `not comparable` and
   the verdict rests on the content anchors.

The stop rule of section 6.4 applies to every non-exact verdict: no stage-inputs and no pool until it
is exact, or you approve continuing. A partial match of the *pool* scorecard or the replay check is a
fidelity result: record it with the results. Only you can decide whether a given level of fidelity is
acceptable (`AGENTS.md`: material changes need explicit approval).

---

## 11. Bringing original artifacts instead of regenerating them

Ask for them first (section 2.2): they are cheaper and more faithful. If you obtain originals, put
them into the same layout and let the same checks run. These paths are expected to work but were
not tested with real originals (none were available); transfer them directly to the cluster, not
through your laptop.

* **Corpus pools** (`<size>-v1/<bucket>/{train.jsonl.gz, eval.jsonl.gz, manifest.json}` of the
  `anytok-corpus` volume): copy them to `$ROOT/corpus/<size>-v1/<bucket>/`, then submit `corpus`
  (it verifies existing buckets and skips them; add `--env HYAK_CORPUS_PROBE=0` without internet),
  then `stage-inputs`. The corpus scorecard should then show the exact verdict.
* **Original staged trainer files** (the 23 `corpus/<bucket>/train.jsonl.gz` and `corpus/manifest.json`
  of `plans/zett-modal-20k-inputs.json`): copy them into `$ROOT/input/corpus/` as regular files,
  optionally the 23 `eval/<bucket>.jsonl.gz` into `$ROOT/input/eval/`, then submit
  `hyak/slurm/submit.sh stage-inputs --env HYAK_STAGE_INPUTS_ARGS=--original-staged`. Their sizes and
  SHA-256 must equal the plan (a mismatch is refused and names both hashes); they are kept and
  registered with origin `original` instead of links, and missing held-out files are copied from the
  pool. The regenerated corpus is still needed (pool recipe, held-out copies, the corpus stop).
  Without `--original-staged` a regular file there is refused with a pointer to the flag.
* **Pool** (`plan.json`, `full-results.json` and `tokenizers/NNNN/tokenizer.json`): copy them to
  `$ROOT/input/pool/` and do **not** run pool jobs or `hyak.pool.collect`: pool jobs refuse a pool
  that already has `full-results.json` without this worker's manifests (exit 2, nothing changed),
  and an index holding a `tokenizer.json` without this worker's `manifest.json` is `blocked` and never
  retrained, replaced or moved. Skip the wheel and the pilots too. The trainer validates the index
  against the plan and each file's hash when it loads it. Check with
  `$ROOT/env/train/bin/python -m hyak.pool.scorecard` (in a job, section 4.4): level 1 must be
  1,000/1,000. With the original pool, evaluation validation is a no-op (section 2.3).
* **Model**: `$ROOT/env/train/bin/python -m hyak.model.fetch_model --from-dir <your copy>` verifies
  and copies the three files.
* The original warmup hypernetwork, if you get it, would replace D1; that is a change of the
  experiment design, not a drop-in (both runs must use the same directory, and it must be decided
  before any run starts).

---

## 12. What was verified locally, and what only the cluster can prove

Verified on a laptop (Apple arm64, CPU only, the pinned packages). The `hyak/tests` suite had 1,233
tests passing and 3 skipped (network tests only) at its last full run, 2026-11-14, with the opt-in
end-to-end tests switched on (`HYAK_TEST_E2E=1`):

* Pins: every staged file's SHA-256 (trainers, `hf_hypernet`, Rust sources and library, plans);
  staging reproduces the Modal code layout (for RL the joint regret launcher's image with the variant
  trainer); tampering is detected, also in the regret snapshot and in the variant.
* The RL trainer variant (D8): `straight_reward.patch` applied to the pinned regret trainer, both with
  `patch(1)` and with a pure-Python applier, reproduces the variant byte for byte; the two files
  differ in exactly the two approved lines; the canonical forward, paired checks and paired logging
  are still in it.
* Init (D1): the recipe, its determinism and options; a full-size build: 1,376,746,800 bytes, the
  rescalers equal to the public checkpoint's, the trainer's next steps emulated (config
  byte-identical to the original trained checkpoints' config), and the step-0 measurements of
  section 2.1 on five pool-style tokenizers.
* Chunked forward (D2): outputs and every gradient equal to the pinned forward in float64; dropout
  and random-state handling; bf16 on CPU; memory bounded (peak 0.28 of the pinned path at V/8
  chunks on CPU).
* Corpus: the ported builder is AST-identical to the original where it should be; a **full-scale
  build** (off-cluster, 2026-11-13) matched every anchor, and the scorecard gave the exact verdict
  on 2026-11-14 (23/23 training and held-out content hashes, 23/23 original staged-file sizes, the
  original corpus manifest hash, 23/23 held-out files identical with the original header time).
* Pool: a real patched wheel built from the pinned upstream revision with `--locked`; real
  tokenizers trained through the worker; claim protocol with concurrent processes; collect on
  1,000 synthetic tokenizers; the policy feature digest of the plan.
* Training: **the real pinned trainers** ran on CPU through the production wrapper (a test-only
  CUDA-to-CPU shim, a miniature world: a tiny Llama with SmolLM2's vocabulary, a width-32
  hypernetwork, a 1,000-entry test pool) for both runs to update 2,511, with timer and `SIGTERM`
  stops at off-grid steps, `SIGKILL` mid-update and during a checkpoint write, and a kill between
  the trainer's rename and the manifest (then `retention adopt`); replayed updates were
  bit-identical, the smoke sequences of both runs passed, the replay check matched the real
  uniform run on 2,511 of 2,511 updates in all five columns (these long runs used the original RL
  trainer, before D8), and the real `07_train` and
  `11_smoke_train` scripts ran under bash with a fake `conda`/`module`, including a `SIGTERM` to
  the batch shell. Full 20,000-update simulations with a fake trainer (kills during saves, off-grid
  stops, interrupted directories) ended with exactly 40 grid checkpoints, 10 full, and a valid
  audit; the real final audit exists only in those simulations. The seed-42 schedule equals the
  original run's 20,000 rows and final random state.
* D8 on 2026-11-14, after the switch to the straight-reward RL trainer: the affected test files and
  the three opt-in end-to-end tests (`HYAK_TEST_E2E=1 pytest hyak/tests/test_e2e_local.py`) passed.
  There the real variant trainer ran through the production wrapper on the CPU shim (RL smoke:
  updates 1-2, a resume to 3): every update logged 128 paired examples, at most 128 positions and
  `policy_reward_bpb == train_bpb`; the receipt passed after both phases and at step 3; the trainer's
  21-setting comparison with the uniform bootstrap's `run_config.json` passed; the RL
  `initial_eval.json` differed from the uniform run's, as expected, with identical sanity tokenizers.
  The wrapper's refusal (exit 4) of each receipt violation (gap reward, regret `policy_reward`, wrong
  entropy or learning rate, 127 examples, 129 positions, an inconsistent gap, missing or non-finite
  metrics) was tested on the receipt function and, through whole segments and an RL smoke, with the
  fake trainer. The whole suite was not rerun for D8, and no RL run longer
  than three updates has used the variant trainer.
* Evaluation: the pipeline ran on real checkpoints written by the real trainers (rebuild, features,
  validation, the real scorer through the CPU shim on every checkpoint kind the trainers write,
  concurrent scorers with a kill and a takeover, pack of both panels, the real `08` and `09`
  scripts under bash); both stored panels rebuild 576/576 and 2,032/2,032 with matching panel
  hashes; the published numbers are recomputed exactly from the stored table.
* Integration: every job script runs end to end against a fake environment (call order, flags,
  exit codes, status files, signal handling, chain halting); `submit.sh --dry-run` for every stage;
  `--help` of every entry point named here; every command, flag, variable and path in this guide.

Only the cluster can show:

* `module load conda`, the conda-forge Python 3.12 and the pip installs on Rocky 9, and the package
  versions pip resolves for the data environment on the day you install (the validated set is pinned,
  its dependencies are not); whether the committed `librust_utils.so` imports there (else the
  rebuild in 6.1); the Linux builds of the patched tokenizers wheel and of `rust_utils` (only macOS
  arm64 builds exist).
* The NVIDIA driver version (>= 580 needed), CUDA, bf16 numerics on the H200, real peak memory and
  speed of training updates (the probe) and of scoring, the scorer's device placement and its memory
  at 127,000 rows, and the pinned trainer on CUDA: the chunk patch in the real loop on the GPU, the
  real `save_checkpoint` of 23.5 GB through the hook (smoke-train and the bootstrap are the first such
  runs).
* Whether `sbatch` accepts the command lines `submit.sh` builds (the test fake accepts anything:
  `sbatch --test-only` on the printed command is the first check), real `conda` activation and
  module loading in a batch shell, and two activations in one job (`04_pool.sbatch`).
* `flock`, atomic `mkdir`/rename, hard links, symlinks, read-only trees and mtime/inode stability on
  GPFS; `SLURM_JOB_END_TIME` / `squeue` answers; how Slurm's `--signal=B:TERM` and
  `scancel --batch --signal=TERM` reach the batch shell and the forwarded trainer; whether the
  interactive recipe of section 4.4 works as written.
* Outbound HTTPS from compute nodes and Hugging Face rate limits from the cluster.
* The full-size pool (time, memory, fidelity), the 23.5 GB save-and-hash time against the
  45-minute margin, the 493 GB audit time, and whether the first updates' step-0 offset (section
  2.1) vanishes as expected.
* The time and memory cost of the RL run's canonical control forward on the H200 (6.11 gives only an
  estimate), and the paired-protocol receipt on real batches of the full-size pool (the laptop
  checked it on the miniature world only).
* The test suite itself on Linux (section 14).

---

## 13. Troubleshooting

| Symptom | What to do |
|---|---|
| `probe` says `DOES NOT FIT` for 16,384 | Read the VERDICT's phase. If the hypernetwork backward set the peak, try `--env HYAK_PROBE_ARGS="--chunk-rows 8192 4096"` and set the **largest** fitting value (at least 4,096) as `HYAK_HN_CHUNK_ROWS` **before** the uniform bootstrap. If the LM micro-batches set the peak, chunk size cannot help: stop and report. Exit code 4. |
| Out of GPU memory during training | The run's chunk setting is fixed; do not change it mid-run without the recorded override (6.10). Check `peak_gib` in the train log and the tokenizer's vocabulary size; report it. A smaller chunk size from the start means restarting both runs from scratch. |
| `torch_cuda` fails in check-env (driver too old) | Driver >= 580 is needed for any torch 2.15.0 build. `HYAK_TORCH_INDEX_URL` cannot help (no CUDA 12 build exists). Ask Hyak support for a newer driver or the CUDA forward-compatibility package (`cuda-compat` for 13.2). A different torch version needs your approval (section 2.5). |
| `rust_library` or `trainer_import_*` fails | `bash hyak/env/build_rust_utils.sh fetch` (login node), then build, check and install in an interactive job (6.1), then check-env again; only before any run directory exists. |
| `libpython3.12.so.1.0` missing | Same rebuild: the extension-module build needs no libpython. |
| No internet on compute nodes (corpus probe fails) | Ask Hyak support whether compute nodes can reach huggingface.co (a proxy). Otherwise build the corpus where there is internet: on the login node with one worker, in `tmux` (`$ROOT/env/data/bin/python -m hyak.corpus.build --workers 1`; it may be killed for using CPU; re-running skips finished buckets), or on another cloud machine with the same data environment. Copy `$ROOT/corpus/` over directly between the two (not through your laptop, `AGENTS.md`), then run `hyak/slurm/submit.sh corpus --env HYAK_CORPUS_PROBE=0 --env HYAK_CORPUS_ARGS=--verify-only`. Nothing else needs internet in a job. |
| Hugging Face rate limits (HTTP 429) | Set `HF_TOKEN` in `hyak/config.env`; resubmit `corpus`. |
| `corpus` ends with "corpus scorecard is not exact" | Section 6.4: do not stage inputs or train the pool; compare the data environment, rebuild the differing buckets, read the scorecard again. |
| `stage-inputs` exits 3 | The corpus stop: the scorecard is not exact, or it is stale (a bucket was rebuilt: rerun `hyak.corpus.verify` in a job). `--accept-corpus-scorecard` needs your approval (6.5). |
| `filesystem_runs` fails (flock or rename on GPFS) | Do not train: the trainer's run lock and the wrapper depend on them. Ask Hyak support; try another storage location for `ANYTOK_RUNS` before any run starts. |
| A train job exits at once with "halted by HALTED.json" (exit 3) | Section 8: inspect, fix, delete the file, resubmit. |
| RL job ends 0 with "uniform_bootstrap_pending" | The pair is not frozen yet. Wait for the `pairing_frozen` event of the uniform run and resubmit (`submit.sh` refuses the submission before that). |
| "Resume settings differ" / `hyak_segment.json` mismatch (exit 3 or 4) | A setting, input or code file changed under a running experiment (the chunk setting, `hyak/memory/chunked_hypernet.py`, a model or input file, a path). Restore it; for the chunk setting only, the recorded override (6.10). |
| Pool index failed 3 times | `--report` shows the error; fix the cause (memory, time) and resubmit with `--env HYAK_POOL_ARGS=--retry-failed`. A SIGKILL'd child: usually the memory limit, `sacct -j <job> -o JobID,MaxRSS,State`. |
| `validate_prefixes` exit 3 | A decision for you (sections 2.3 and 6.13); read the validation report. |
| `Comparison plan changed ...` from `score` or `pack` | A file the scores depend on changed after scoring started (the message names them), or another checkpoint is being scored: section 6.14. |
| `submit.sh` refuses: a job of that name is queued | One-shot stages are not submitted twice; wait, or pass `--allow-duplicate` if you are sure. |
| `submit.sh` refuses `train-rl` | `input/pairing.json` does not exist: wait for the uniform bootstrap (6.11). |
| Status stuck at `running` after a job died | A hard-killed job cannot write its end; run `python3 -m hyak.tools.reconcile` (section 7), or the next job of the stage corrects it. `squeue` is the truth. |
| A job is killed (exit 137) | Usually the memory limit: `sacct -j <job> -o JobID,MaxRSS,State`. For the pool see 6.7; a training chain is halted (section 8). |

---

## 14. Reference

### submit.sh

```
hyak/slurm/submit.sh <stage> [--count N] [--debug] [--dry-run] [--run uniform|rl]
                     [--after JOBID[:JOBID...]] [--env KEY=VALUE]... [--allow-duplicate] [-- <sbatch args>]
```

Stages: `check-env probe corpus wheel pool stage-inputs init train-uniform train-rl eval-prepare
eval-uniform eval-rl replay-check smoke-train`. `--count N > 1` only for `pool`, `eval-prepare`,
`eval-uniform`, `eval-rl`, `train-uniform`, `train-rl`. `--debug` uses the debug QoS and 1 h
(check-env and probe use them by default; refused for training). `--after` adds `afterok` and
`--kill-on-invalid-dep=yes`. `--env KEY=VALUE` sets a variable for the job and, when KEY is a config
key (`HYAK_TIME_SMOKE_TRAIN=00:40:00`), decides this submission's resources; values may contain
spaces and commas. Secrets (`HF_TOKEN` and any name containing TOKEN, SECRET, PASSWORD, CREDENTIAL,
API_KEY) are refused: put `HF_TOKEN` in `hyak/config.env`. After `--`, resource flags (`--time`,
`--signal`, `-p`, `-q`, `-A`, `--gres`, `--mem*`, `-c` ...) are refused with a pointer to
`HYAK_<FLAG>_<STAGE>`. `submit.sh` passes plain `--export=ALL`: the job's variables (`HYAK_REPO`,
`HYAK_STAGE`, `HYAK_RUN`, `HYAK_CONFIG` and every `--env` pair) are exported by `submit.sh` itself,
so they never appear on the recorded `sbatch` line. It refuses `train-rl` and `smoke-train --run rl`
until `$ANYTOK_ROOT/input/pairing.json` exists, and training stages whose limit is below margin + 20
minutes (a 2-hour rehearsal that exercises both stop mechanisms needs
`HYAK_SEGMENT_MARGIN_MINUTES_TRAIN_UNIFORM=30` with `HYAK_TIME_TRAIN_UNIFORM=02:00:00`). A mistyped
per-stage variable gets a warning.

To stop jobs: `hyak/slurm/stop.sh <stage> [--dry-run]` (section 8). To mark the stages of dead jobs:
`python3 -m hyak.tools.reconcile` (section 7).

### Extra arguments for a job

Pass with `hyak/slurm/submit.sh <stage> --env VAR="..."` (words split on blanks; no quotes inside, so
a reason text must be one word). Do not `export` these in your login shell: `submit.sh` passes your
whole environment to every job.

| Variable | Stage | Effect |
|---|---|---|
| `HYAK_CHECK_ENV_ARGS` | check-env | appended to `python -m hyak.env.check_env` (e.g. `--no-hash-model`) |
| `HYAK_PROBE_ARGS` | probe | appended to `python -m hyak.memory.probe --also-pinned` |
| `HYAK_CORPUS_PROBE` | corpus | `0` skips the source probe |
| `HYAK_CORPUS_ARGS` | corpus | appended to `python -m hyak.corpus.build` (e.g. `--verify-only`, `--allow-unvalidated-env reason`) |
| `HYAK_WHEEL_ARGS` | wheel | appended to `build_wheel.sh build install` (e.g. `--force`) |
| `HYAK_GCC_MODULE` | wheel | module loaded when no C compiler is found (default `gcc/13.4.0`) |
| `HYAK_ACCEPT_CHUNK_CHANGE` | train-uniform, train-rl | reason text; passed as `--accept-chunk-change` to every segment so a changed chunk setting is accepted on the record (6.10) |
| `HYAK_POOL_ARGS` | pool | appended to `python -m hyak.pool.worker` (e.g. `--indices pilots+binding`, `--pilots-only`, `--families ...`, `--retry-failed`) |
| `HYAK_STAGE_INPUTS_ARGS` | stage-inputs | appended to `python -m hyak.corpus.stage_inputs` (e.g. `--original-staged`, `--accept-corpus-scorecard reason`) |
| `HYAK_INIT_ARGS` | init | appended to `python -m hyak.model.init_hypernet --status` (e.g. `--calibrate-after-bias`, `--prior-bias-std 0`, `--verify`) |
| `HYAK_REPLAY_CHECK_ARGS` | replay-check | appended to the example replay (e.g. `--strict`, `--updates 20000`) |
| `HYAK_EVAL_PREPARE_ARGS` | eval-prepare | appended to both `python -m hyak.eval.validate_prefixes` calls (e.g. `--workers 4`, `--recheck`, `--accept-valid-context-trims`: only after your approval, section 2.3) |
| `HYAK_EVAL_SCORE_ARGS` | eval-uniform, eval-rl | appended to every `python -m hyak.eval.score` call (e.g. `--checkpoint <archived step-020000>`, section 6.14) |
| `HYAK_EVAL_INDICES` | eval-uniform, eval-rl | pilot: score only these pool indices, pool panel only, no packing |
| `HYAK_EVAL_PANELS` | eval-uniform, eval-rl | panels to score (default `pool natural`) |

`HYAK_REGISTER_TIMEOUT_SECONDS` (default 60) bounds the dashboard registration in every job. Jobs
set `HF_HUB_OFFLINE=1` for train, smoke-train, eval, probe, init and replay stages, and remove
`PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `PYTHONUSERBASE` and `LD_PRELOAD` from the inherited
environment (the names are logged and recorded in an `environment_sanitized` event;
`CUDA_VISIBLE_DEVICES` is untouched). For training stages they also unset
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` and `PYTHONOPTIMIZE`.

### Exit codes (the contract of every program)

| Code | Meaning | The job |
|---|---|---|
| 0 | ok (also a clean pause, a stop on a signal or the time margin) | ends 0; a final state the stage wrote is kept, else `paused` |
| 1 | crash only: an uncaught exception, full traceback in the job log | `failed`; halts a training chain |
| 2 | usage, configuration, input or environment error, with a remedy line | `failed`; halts a training chain |
| 3 | a protocol stop that needs your decision (`validate_prefixes`, pool failed or blocked indices, a non-exact corpus scorecard at `stage-inputs`, a training job that finds `HALTED.json`, a mid-run settings change) | `failed`; halts a training chain |
| 4 | a check failed (check-env, probe `DOES NOT FIT`, a corpus bucket failed, `stage-inputs` inconsistent, `INIT FAILED`, no checkpoint progress, the RL paired-protocol receipt (6.11), replay-check schedule mismatch, `retention` verification) | `failed`; halts a training chain |
| 75 | not now / incomplete (the RL run before the uniform bootstrap, a run lock still held, `pack` while a run is incomplete, collect on an incomplete pool) | ends 0, no `HALTED.json`, a `not_now` event with the reason |
| 130, 143 | a stop that was asked for | `stopped`, or `paused` when the program finished its graceful stop with 0 |
| 137 or any other 128+n | killed (usually the memory limit) | `failed` with the text `killed`; halts a training chain |

Where a program uses a code more specifically: `run_segment --should-start` prints JSON and exits 0
(start) or 75 (complete, `insufficient_time` or `uniform_bootstrap_pending`); `retention`: 0, 2, 4,
75 (lock held), 1; `replay_check`: 0, 4, 2, 1; `corpus.build`: 0 (also `paused`), 2, 4, 143, 1;
`pool.worker`: 0, 3, 2, 1; `pool.collect`: 0, 75, 4, 2, 1 (a first TERM lets it finish, a second
stops it with 75); `eval.score`: 0, 75, 2, 1; `eval.pack`: 0, 75, 2, 1; `eval.validate_prefixes`: 0
(also up to date, or another job holds the claim), 3, 2, 1; `eval.rebuild_panel`: 0, 4, 2, 1;
`common.config`: 0, 2; `tools.register_dashboard`: 0, 2, 4 (`--check`: not up to date);
`tools.reconcile`: 0, 2; `tools.policy_monitor`: 0, 2, 1 (it also exits 0 when another collector
holds the lock). `train.monitor --once` exits 1 with a traceback on failure; the background monitor
never exits on a refresh error and logs `monitor_refresh_failed` events.

### Job scripts

| Stage | Script | Environment | Main command |
|---|---|---|---|
| check-env | `hyak/slurm/00_check_env.sbatch` | train | `python -m hyak.env.check_env` |
| probe | `hyak/slurm/01_probe.sbatch` | train | `python -m hyak.memory.probe --also-pinned` |
| corpus | `hyak/slurm/02_corpus.sbatch` | data | `python -m hyak.corpus.build --probe`; `python -m hyak.corpus.build`; `python -m hyak.corpus.verify` |
| wheel | `hyak/slurm/03_wheel.sbatch` | data | `bash hyak/pool/build_wheel.sh build install` |
| pool | `hyak/slurm/04_pool.sbatch` | data, then train | `python -m hyak.pool.worker`; when complete `python -m hyak.pool.collect --check-only`, `python -m hyak.pool.collect`, `python -m hyak.pool.scorecard` |
| stage-inputs | `hyak/slurm/05_stage_inputs.sbatch` | train | `python -m hyak.corpus.stage_inputs`; `python -m hyak.corpus.stage_inputs --check` |
| init | `hyak/slurm/06_init.sbatch` | train | `python -m hyak.model.init_hypernet --status` |
| train-uniform, train-rl | `hyak/slurm/07_train.sbatch` | train | `python -m hyak.train.monitor --run R` (background); `python -m hyak.train.run_segment --run R --should-start`; `python -m hyak.train.run_segment --run R` |
| eval-prepare | `hyak/slurm/08_eval_prepare.sbatch` | train | `python -m hyak.eval.rebuild_panel --panel all`; `python -m hyak.eval.prepare_features`; `python -m hyak.eval.validate_prefixes --panel pool`; `python -m hyak.eval.validate_prefixes --panel natural` |
| eval-uniform, eval-rl | `hyak/slurm/09_eval_score.sbatch` | train | `python -m hyak.eval.score --run R --panel P`; `python -m hyak.eval.pack --panel P` |
| replay-check | `hyak/slurm/10_replay_check.sbatch` | train | `python -m hyak.train.replay_check --schedule-only`; `python -m hyak.train.replay_check` (the example replay, 2,000 updates by default) |
| smoke-train | `hyak/slurm/11_smoke_train.sbatch` | train | `python -m hyak.train.run_segment --run R --smoke` |

Every job script: no `#SBATCH` lines (submit.sh supplies the resources), `set -euo pipefail`,
sources `hyak/slurm/lib.sh`, loads the config, activates the environment, registers the dashboard
entries, runs each program as a child that receives Slurm's TERM, and records the job's end in
`status.json` on every exit path (table above). After Slurm's TERM no further program starts; when
the running program stopped cleanly the job ends with 0 (`paused`: resubmit to continue). Stop
timers: the in-process timer of a training segment fires at the remaining time minus
`HYAK_SEGMENT_MARGIN_MINUTES` minus 5 minutes (50 minutes before the limit with the default margin
of 45); Slurm's backup `--signal=B:TERM@<margin>` follows 5 minutes later; a segment starts only
with at least margin + 20 minutes left (65 by default). `segment_end` reports `stop_source` (timer,
signal, external), `stop_seconds` and `stop_deferred_seconds`. For short rehearsals use
`HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_TRAIN` or `HYAK_SEGMENT_MARGIN_MINUTES_TRAIN_UNIFORM`.

### Files

```
$ANYTOK_ROOT/env/{train,data}                conda environments (+ <name>.freeze.txt)
$ANYTOK_ROOT/toolchain/{rustup,cargo}        Rust 1.96.0
$ANYTOK_ROOT/build/                          tokenizers checkout, rust_utils rebuild
$ANYTOK_ROOT/wheels/                         patched wheel + provenance.json
$ANYTOK_ROOT/code/{uniform,rl}               staged trainer trees + code_manifest.json (rl: the regret
                                             launcher's image with the variant trainer, D8)
$ANYTOK_ROOT/corpus/<size>-v1/<bucket>/      corpus pools (+ .hyak/ logs, locks, gzip-mtimes; scorecard.{json,txt})
$ANYTOK_ROOT/input/                          trainer inputs: recipe, corpus links, eval, model,
                                             hypernet_init, pool, pairing files, manifest.json
$ANYTOK_ROOT/eval/{panels,features,scores,reports}
$ANYTOK_ROOT/checkpoints/{uniform,rl}        archived finals (section 9)
$ANYTOK_RUNS/{uniform,rl}                    trainer output directories (+ hyak_segment.json, interrupted/)
$ANYTOK_RUNS/{probe,smoke,cleared,.hyak-cache,.locks}
modal_results/hyak_tillicum/<stage>/         status.json, events.jsonl, slurm-<job>.out,
                                             latest-slurm.out, <job>-train.log, HALTED.json,
                                             previous-<utc>/ (an earlier attempt of a run)
```

`<run>/hyak_segment.json` records, besides the settings compared on resume, the model hashes, the
software environment (Python, torch, CUDA, transformers, the SHA-256 of `env/train.freeze.txt`),
`trainer_variant` (the trainer the run executes: for RL the variant's path and SHA-256 and those of
its base, the regret trainer) and when applicable `environment_history`, `chunk_changes` and
`accepted_pairing_differences`. `input/pairing.json` records the same RL trainer as
`rl_trainer_variant`; a pairing record that names another RL trainer is refused.
`code/code_manifest.json` records it as `rl_trainer_variant` too.

In the checkout, besides the job scripts and modules:

```
hyak/variants/anytok_pool_straight_rl_20k/   the RL trainer (D8): its scripts/ directory holds the trainer;
                                             straight_reward.patch (the exact diff from the regret
                                             trainer), README.md (provenance, hashes, the decision)
hyak/code/pinned.py                          SHA-256 pins: runtime snapshots (uniform, original RL,
                                             regret at 3de64a5), the variant, Rust sources, plans
```
`segment_start` records the relevant environment variables (`PYTORCH_CUDA_ALLOC_CONF`,
`CUBLAS_WORKSPACE_CONFIG`, `OMP_NUM_THREADS`, `PYTHONHASHSEED`, `HF_HUB_OFFLINE` ...). Non-finite
numbers in a status or event are written as `null` with the raw text beside them.

### Tests

On the cluster (train environment, in a job; section 4.4):

```bash
PYTHONDONTWRITEBYTECODE=1 $ROOT/env/train/bin/python -m pytest -p no:cacheprovider hyak/tests -q
```

The suite is hermetic: `HYAK_CONFIG` points at a non-existent file for every test (nothing reads
your real `hyak/config.env` or touches real storage), the Hugging Face cache is a temporary
directory and `HF_HUB_OFFLINE=1` (unless `HYAK_TEST_NETWORK=1`). It has only run on the laptop
(macOS arm64). Tests that need the laptop's local assets (SmolLM2 test files, a mini corpus, a macOS
build of the Rust library, `HYAK_TEST_SCRATCH`, `HYAK_TEST_ASSETS`) skip themselves, naming the
missing path: about 70 tests (evaluation preparation, `fetch_model` verification, the init builds, the
probe end to end) therefore skip on the cluster by design. Others may behave differently on Linux, so
treat a failure there as information, not as a verdict on the cluster setup (check-env is the
authority). `HYAK_TEST_SKIP_SLOW=1` skips the slow ones. With a custom `--basetemp`, keep it outside
the checkout (configuration validation refuses storage inside it).
