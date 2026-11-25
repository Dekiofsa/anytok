# Held-out full-pool eval of every raw-BPB checkpoint

Run Qingyun's held-out full-pool eval (`reports/heldout-pool-sweep` and
`scripts/run_rift_pool_sweep.py` on zett-modal-20k, commit 948787e) on all 42
saves of the raw-BPB run (`reports/raw-bpb-rl-20k-tillicum/`): every 500
updates from 0 to 20,000, plus the save at 16,409. Each checkpoint scores all
1,000 pool tokenizers on the same 156 held-out passages, so this run's
checkpoints and his runs' checkpoints compare on identical text.

**Status: complete on 2026-11-24; report in `reports/rawbpb-heldout-pool/`;
score files on Hugging Face as one archive per checkpoint.** All 42
checkpoints were scored on all 1,000 tokenizers, and every score file passed
the checks below; the upload is under Persistence. Left: committing, after the
user's go-ahead. The user asked for the adaptation on 2026-11-23 ("Can you
adapt it for mine?"). On 2026-11-24 his archive became public, and the user
asked to double-check everything and then run it.

Checked before any GPU job, on 2026-11-24:

- The archive is public (anonymous API reads return it). Its newest commit is
  still the pinned `38176d7d`. The tarball's size and SHA-256 match his
  sources record, and his unchanged restore script verifies it: 78,199
  saved scores, 35 pending checkpoints, plan `d1269e6f…`, 4,992 tokens.
- All 42 saves exist at the pinned revision. Each scored file's size and
  SHA-256 in its `manifest.json` equal the Hub's own record (LFS object id, or
  the downloaded bytes for `hypernet/config.json`). The hypernetwork config is
  byte-identical to his checkpoints' (`22cb4f86…`).
- Built from the real archive, the workspace differs from his only in
  `checkpoints` and `checkpoint_hashes`. Its top-level modules and `inbox/`
  are his archive's bytes; only `run_rift_pool_sweep.py` is his 948787e
  version, as his README copies it. The code snapshot (22 files), the passage
  file and the continuation proof pass his `setup()` checks: 156 passages,
  4,992 tokens, 14,429 bytes. The same build on Mac and Tillicum gives plan
  hash `8d982055…`.
- His files copied to Tillicum are git-blob-identical to 948787e (or his
  branch for the helpers). The scoring path never loads the snapshot's
  compiled Rust library: `import rust_utils` resolves to the snapshot's
  folder as a namespace package.
- The environment is his pins exactly: Python 3.12.15, torch 2.15.0+cu132,
  transformers 5.15.1, tokenizers 0.22.2, numpy 2.3.5, safetensors 0.8.0,
  huggingface-hub 1.33.0. His staging verified the canonical model
  (`1193528982…`) and all 1,000 tokenizers (index `7cdc2c0b…`).
- GPU smoke test (job 518478, raw-bpb-020000, node g019, 36 s): the
  128,000-token vocabulary fits (peak 42.6 GiB allocated), lower-memory
  scoring is bitwise equal, and the checkpoint passes his embedding and
  hash checks on an NVIDIA H200.
- The smoke log shows Triton failing to compile its driver helper
  (`Python.h: No such file or directory`; Tillicum's system Python 3.12 has no
  headers). Two debug-QoS probes (518485, 518543) traced it to torch
  2.15's eager op router (`torch/_native/registry.py`), which offers a Triton
  kernel for outer-product `bmm` calls (the rotary embedding's
  `inv_freq @ position_ids`). When Triton cannot start, the router uses
  ATen's kernel. Both compute one product per element (the Triton kernel is
  `a[:, None] * b[None, :]`; ATen's is a K=1 GEMM with TF32 off, torch's
  default, which neither his code nor transformers changes), so the results
  are bitwise identical. The router's other overrides need runtimes that his
  pinned environment does not install (CuTe DSL, nvmath, FlyDSL), an opt-in
  variable (`PYTORCH_SUM_INNER_TREE`), or ops the scorer does not call
  (`rms_norm`; LlamaRMSNorm uses `pow`/`mean`/`rsqrt`).
- His panel audit, re-run on Tillicum with the staged tokenizers (debug job
  518502, about 6.5 minutes), equals his in every field except the plan
  hash: panel `0db45483…`, native prefixes `35feec06…` for all 1,000
  tokenizers, the same 5 original prefix failures and the same 3 trimmed
  passages. Features for all 1,000 tokenizers followed (3.5 minutes).
- His preflight (debug job 518616, 4 minutes) re-hashed all 42 staged
  checkpoints and reported `protocol: heldout-pool-4992`, plan `8d982055…`,
  42 checkpoints, 1,000 tokenizers, 42,000 pairs, 156 passages, 4,992 scored
  tokens and 14,429 bytes.

**Scoring:** array 518632 (42 tasks, normal QoS, his 1.5-hour limit), submitted
2026-11-24 at 21:09 UTC. The tasks ran from 21:10 to 22:51 UTC, up to 22 at
once, 19 to 22 minutes each, and all ended COMPLETED with exit code 0. Each
log holds only model-loading progress bars and the Triton message above. They
used 14.04 GPU-hours ($12.64); `hyakusage` moved from $70.55 to $83.19.

Checked after scoring, on 2026-11-24:

- A read-only pass on Tillicum read all 42,000 per-tokenizer files. Each has
  plan `8d982055…`, its checkpoint and index, its pool tokenizer's SHA-256 and
  156 finite non-negative passage losses, and its BPB and gap recompute from
  them. Every `ranking.csv` lists the 1,000 tokenizers once in his order and
  matches those files; every `complete.json` is his receipt.
- The scoring panel on Tillicum is byte-identical to the one in his archive
  (`cf76016f…`).
- The 126 copied files (`ranking.csv`, `complete.json`, `canonical.json`)
  match Tillicum's SHA-256s, and the per-passage pool means computed on
  Tillicum reproduce every checkpoint's mean gap and canonical BPB.

**Result** (details in `reports/rawbpb-heldout-pool/README.md`):

- The mean gap over the 1,000 tokenizers falls from +0.4692 BPB at step 0 to
  +0.0697 at step 20,000; no tokenizer matches canonical at any save.
- His archive holds complete L40S scores of his uniform (500 to 20,000),
  regret (to 8,000) and predictor (to 7,000) runs. At step 20,000 raw-BPB
  minus uniform is -0.0007 BPB (95% passage bootstrap [-0.0046, +0.0036]).
  Over the 40 matching saves the interval is above zero at 8 and below zero
  at none.
- By group at step 20,000, raw-BPB is below uniform on the six non-regex
  groups, which drew 99.3% of its 20,000 updates. It is above uniform on the
  three regex groups, which drew 0.7%, most on unigram / regex (+0.0182).

## Question

How does each tokenizer's held-out transfer gap change across the raw-BPB run,
and how does the run compare with his runs on the same passages and tokenizers?

## The protocol, his and unchanged

- 156 frozen passages from nine domains (17 or 18 each), 32 scored canonical
  continuation tokens each: 4,992 tokens and 14,429 bytes per tokenizer.
- The prefix is encoded with the scored tokenizer; the continuation, its bytes
  and the output head are canonical and shared. At most 128 positions,
  including BOS.
- The hypernetwork runs with dropout off and builds one full embedding table
  per tokenizer. The canonical baseline is scored once per checkpoint.
- Score: the mean over the nine domains of the BPB gap (transferred minus
  canonical). Each checkpoint's `ranking.csv` orders the 1,000 tokenizers by it.
- His scorer checks the passages, the native prefixes, the tokenizer hashes, the
  code snapshot and the pinned runtime versions before scoring.

This is a different quantity from the step-20,000 training-reward ranking
(`plans/rawbpb-pool-ranking-20000.md`), which used 32 training batches per
tokenizer. Both stay; they answer different questions.

## What differs from his sweep

- **The checkpoint list.** His `plan.json` names his runs' checkpoints, and his
  scorer refuses any other label. Here exactly two plan keys change:
  `checkpoints` (this run's labels, `raw-bpb-NNNNNN`, in step order) and
  `checkpoint_hashes` (their `lm.pt` and hypernetwork SHA-256s). Every other
  key is his. The plan's hash changes with them, so his panel audit runs again
  in the new workspace, and `check-audit` requires it to equal his in every
  field except the plan hash.
- **Account** `stf` instead of `zlab`.
- **GPU.** Tillicum H200s, the type his README prescribes there. His archived
  scores come from other machines (Rift and Eagle2; the scorer's default GPU is
  an L40S), and his README notes that BPB can shift slightly between GPU types.
  His archived scores are from Rift L40S GPUs, so the report's comparisons
  with his runs carry that caveat. A same-GPU comparison would need one of his
  checkpoints re-scored on an H200; that has not been done.

## Inputs

- **His archive.** Hugging Face dataset
  `qingy2024/anytok-rift-eval-resume`, revision
  `38176d7d654b33f513dcc7e4f3f19e6bfc142d85`, folder
  `paused-20261122T143018Z`: the passages, the plan, the code snapshot and his
  checkpoint receipts. Public since 2026-11-24; no login is needed.
- **Model and tokenizers.** His public input-bundle record, staged by his script.
- **Checkpoints.** `Dekiofsa/RLtokenizers`, folders
  `raw-bpb-rl-20k/step-NNNNNN` at revision
  `60670ac8cea5b53695ec43db8160c86856c2b70b`. Only `lm.pt`,
  `hypernet/config.json` and `hypernet/model.safetensors` are downloaded
  (8.6 GB per save, 362 GB for all 42), straight to Tillicum, and each is
  checked against its save's `manifest.json`. No optimizer or trainer state.

## Files

- His, copied unchanged from 948787e: `scripts/run_rift_pool_sweep.py`,
  `rift_eval_memory.py`, `restore_rift_pool_archive.py`,
  `stage_rift_resume_inputs.py` and `heldout_pool_h200.sbatch`; also
  `scripts/four_way_8k.py` from his branch, which his audit imports when a
  shared prefix needs trimming. The other modules his scorer imports are
  already on this branch, byte for byte the same as his.
- `scripts/prepare_rawbpb_heldout_pool.py`: builds the workspace from his
  restored one (`prepare`) and compares the re-run audit with his
  (`check-audit`). It requires every save of the run.
- `scripts/setup_rawbpb_heldout_pool.sh`: the four setup phases below.
- `scripts/heldout_pool_status.py` (read-only, runs on the cluster) and
  `scripts/run_rawbpb_heldout_pool.py` (monitor-only observer on the Mac):
  dashboard entry `rawbpb-heldout-pool`, written to
  `modal_results/rawbpb_heldout_pool/`.
- `scripts/heldout_pool_passage_means.py` (read-only, standard library): each
  checkpoint's canonical and pool-mean transferred loss per passage, from the
  per-tokenizer score files. Run on Tillicum for this run and locally on his
  restored archive.
- `scripts/report_rawbpb_heldout_pool.py`: writes `reports/rawbpb-heldout-pool/`
  from the copied files, the passage means, the Slurm records and his restored
  archive (`--reference`).
- `tests/test_rawbpb_heldout_pool.py`: on synthetic inputs, his unchanged staging
  script and preflight accept the prepared workspace (42 checkpoints, 42,000
  pairs, 4,992 tokens) and stop on a staged file that differs from its
  manifest; the preparation refuses a missing save, a wrong manifest, an edited
  plan and a changed scorer file. The observer's stages, counts and Slurm
  parsing are tested too.

## Commands

The stage is `/gpfs/projects/stf/<user>/anytok-heldout`. `code/scripts/` holds
the ten files the run needs, copied from this branch and hash-checked on
arrival. `scripts/setup_rawbpb_heldout_pool.sh STAGE PHASE` runs his README's
commands in four phases, keeping every cache on project storage. All downloads
are anonymous, straight from Hugging Face.

| Phase | Where | What |
|---|---|---|
| `inputs` | login node, tmux | pinned environment, his archive, restore, `prepare`, model and 1,000 tokenizers, the smoke checkpoint |
| `panel` | debug-QoS job | his `audit`, `check-audit`, his `features` |
| `checkpoints` | login node, tmux | the other 41 checkpoints |
| `preflight` | debug-QoS job | his `verify-launch` on all 42 |

The audit loads all 1,000 tokenizers in one process, and the preflight hashes
362 GB. On 2026-11-24 the login node killed the audit (exit 137) after its
prefix check, so both run on a compute node. They use the debug QoS, for its
priority, with default resources. Debug jobs are billed like any other: the
smoke test, panel, preflight and two probes (15.8 minutes in all) moved
`hyakusage` from $70.31 to $70.55. The QoS's usage factor of 0 affects
fairshare only.

```bash
stage=/gpfs/projects/stf/$USER/anytok-heldout; work=$stage/raw-bpb; code=$stage/code/scripts
tmux new-session -d -s heldout-inputs "bash $code/setup_rawbpb_heldout_pool.sh $stage inputs > $stage/logs/setup-inputs.log 2>&1; echo exit=\$? >> $stage/logs/setup-inputs.log"
sbatch --account=stf --qos=debug --time=00:15:00 --output="$work/logs/%x-smoke-%j.log" \
  "$code/heldout_pool_h200.sbatch" smoke "$work" "$stage/venv/bin/python" raw-bpb-020000
sbatch --account=stf --partition=gpu-h200 --qos=debug --gres=gpu:1 --time=01:00:00 \
  --job-name=anytok-heldout-panel --output="$stage/logs/setup-panel.log" \
  --wrap "bash $code/setup_rawbpb_heldout_pool.sh $stage panel; echo exit=\$?"
tmux new-session -d -s heldout-checkpoints "bash $code/setup_rawbpb_heldout_pool.sh $stage checkpoints > $stage/logs/setup-checkpoints.log 2>&1; echo exit=\$? >> $stage/logs/setup-checkpoints.log"
sbatch --account=stf --partition=gpu-h200 --qos=debug --gres=gpu:1 --time=01:00:00 \
  --job-name=anytok-heldout-preflight --output="$stage/logs/setup-preflight.log" \
  --wrap "bash $code/setup_rawbpb_heldout_pool.sh $stage preflight; echo exit=\$?"
```

The preflight must report `protocol: heldout-pool-4992`, `checkpoints: 42`,
`pairs: 42000` and `scored_tokens: 4992`. Then the scoring array, unchanged
from his README apart from the account:

```bash
sbatch --account=stf --array=0-41 --output="$work/logs/%x-%A_%a.log" \
  "$code/heldout_pool_h200.sbatch" score "$work" "$stage/venv/bin/python" "$work/checkpoints.txt"
```

Operational differences from his README, none of which touch the protocol:
the smoke test runs in the debug QoS with a 15-minute limit (his: normal QoS,
5 minutes), on the last checkpoint rather than the first, and the panel and
preflight run on a compute node.

## Compute

One H200 per checkpoint, normal QoS, his 1.5-hour limit, no explicit CPU or
memory reservations (Slurm's defaults: 4 CPUs, 120 GB). His figure is about $9
for 20 checkpoints, so about $19 for these 42; if every task used its whole
limit, 63 GPU-hours or $56.70. The five debug jobs cost $0.24. The account
had $179.69 of $250 left on 2026-11-24 before this run.

Actual: 14.04 GPU-hours and $12.64 for the array, $12.88 with the debug jobs.
`hyakusage` for `stf` read $83.19 of $250 afterwards ($166.81 left).

## Persistence

- Scores: `$work/run/heldout-h200/<label>/`, with per-passage losses for all
  1,000 tokenizers, `ranking.csv` and `complete.json`. Only small files came
  to the local machine (`modal_results/rawbpb_heldout_pool/`): the 42
  `ranking.csv`, `complete.json` and `canonical.json` files (4.6 MB), the
  per-passage pool means (260 KB) and the Slurm records.
- Hugging Face: `Dekiofsa/RLtokenizers` (public model repo), folder
  `raw-bpb-rl-20k/diagnostics/heldout-pool-4992/`, revision `fb8f2a00`. It
  holds `raw-bpb-NNNNNN.tar.gz` for each of the 42 checkpoints (1.6 MB each,
  67.8 MB in all), and each unpacks to the scorer's folder byte for byte. The
  user said "Yes, upload" on 2026-11-24. Everything went from Tillicum's login
  node straight to the Hub, 16:18 to 16:33 PST.
  - Before: a read-only scan found exactly 1,003 small regular files per
    folder (42,126 files, 169,055,112 bytes) and receipts with plan
    `8d982055…`. It found no netid, cluster path or email, and no file changed
    after the last task ended. Its manifest records each file's size, git blob
    SHA-1 and SHA-256 (`f050e2db…`). The 126 small files matched the local
    copies.
  - The first attempt uploaded loose files, one commit per checkpoint. After
    19 folders (steps 0 to 9,000, revision `670cd59f`) the Hub rejected the
    next commit. The repo is limited to 20,000 files, so 42,126 loose files
    cannot fit. The user chose one archive per checkpoint and removing the
    loose folders.
  - Archives: built on Tillicum with the files' exact bytes and mtimes, no
    owner names and a fixed gzip header. Every member was checked against the
    manifest. Commit `430a09d3` added the 42 archives. The Hub's SHA-256
    matched each one, and so did each downloaded copy. Commit `fb8f2a00` then
    removed the 19 loose folders, after checking they held exactly the
    manifest's files. They stay in the repo's history at `670cd59f`.
  - Final check: the repo is the pre-upload tree (`60670ac8`, 523 files, none
    changed) plus the 42 archives, 565 files in all. Its 21 commits since
    `60670ac8` all come from this upload. An anonymous read agrees.
  - Scripts, the manifest, the archive record (`archives.json`, with each
    archive's SHA-256) and the logs are in `hf-upload/` on the stage, copied
    to `modal_results/rawbpb_heldout_pool/hf-upload/`. The report's
    `manifest.json` also lists each archive's SHA-256.
- Dashboard: Evaluations entry `rawbpb-heldout-pool` (42,000 tokenizer
  scores), kept current by the monitor-only observer, which exited at
  `complete` on 2026-11-24.
- Report: `reports/rawbpb-heldout-pool/`, generated from the copied results.
  Commands:

```bash
ssh "$NETID@tillicum.hyak.uw.edu" "/usr/bin/python3 -I - /gpfs/projects/stf/\$USER/anytok-heldout/raw-bpb/run/heldout-h200" \
  < scripts/heldout_pool_passage_means.py > modal_results/rawbpb_heldout_pool/passage-means.json
.venv/bin/python -I scripts/heldout_pool_passage_means.py "$REFERENCE/run/scores" \
  > modal_results/rawbpb_heldout_pool/reference-passage-means.json
.venv/bin/python -B -E -s scripts/report_rawbpb_heldout_pool.py --reference "$REFERENCE"
```

  `$REFERENCE` is his archive restored by `restore_rift_pool_archive.py`.
  The report also reads `hf-upload/archives.json`, the upload's record of the
  42 archives.
