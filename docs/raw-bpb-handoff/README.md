# Raw-BPB training handoff

Train a fresh LM, input hypernetwork and shared tokenizer-selection policy for
20,000 updates. Match the current regret experiment, changing the policy reward
to **raw transferred continuation BPB**. Keep **entropy weight 0**. The validated
GPU-only memory implementation is enabled by default.

Read the repository's root `AGENTS.md` before launching or changing the run.

This is an independent run. Use a new output directory and fresh optimizers and
policy. The starting weights are the original SmolLM2-1.7B and identity-warmed
hypernetwork in the bundle. A partially trained regret checkpoint is not the
initialization for this comparison.

## Exact data, already prepared

Public dataset: <https://huggingface.co/datasets/cminst/anytok-training-inputs>.
`bundle-reference.json` pins the immutable Hugging Face revision, archive URL,
size and SHA-256. Credentials are not needed to download this public dataset.

- Archive: `anytok-training-inputs-v1.tar.gz`, **9,314,144,730 bytes**.
- Extracted inputs: **1,062 files**, **13,445,041,321 bytes**.
- Archive SHA-256:
  `51f24daf94303f63f00b2d5458c7d34e346163fffa3cefac5d55f36edde3bf5f`.
- Every original input file and every archived member was checksum-verified.

The archive extracts to `input/` and `BUNDLE_MANIFEST.json`:

| Path inside `input/` | Contents |
| --- | --- |
| `pool/` | All 1,000 tokenizer JSON files, original generator settings, complete result index and plan |
| `corpus/` | Exact compressed training text and 23-bucket checksum manifest |
| `recipe.json` | Corpus sampling weights, upstream names and pinned revisions |
| `model/` | Original SmolLM2-1.7B model and canonical tokenizer |
| `warmup/` | Identity-warmed input hypernetwork |
| `eval/` | Original held-out sanity-evaluation data |
| `uniform_heldout_tokenizers.json` | The two fixed ZeTT sanity tokenizer specifications |
| `uniform_run_config.json` | Original baseline settings used by the fidelity checks |
| `manifest.json`, `rl-input-audit.json` | Original file hashes and input provenance |

The training corpus represents **10 billion raw UTF-8 text bytes**, compressed
to about 4.02 GB. It includes eight web-language buckets, eight Wikipedia
buckets, and code, papers, math, QA, books, medical and legal text.

The saved pool contains 332 BPE, 328 WordPiece and 340 Unigram tokenizers.
Vocabulary targets span 32k–128k; pretokenization is regex, 256-character line
chunks or no regex. BPE merge noise spans up to 75%. Each tokenizer was already
trained on its saved 200 MB mixture. **Use these saved tokenizers and corpus;
do not regenerate either.** The 200 MB tokenizer-building samples and the
10 GB LM training corpus serve different purposes.

Original cloud source: Modal profile `queues`, volume `anytok-pool-rl-20k`,
prefix `input/`. The archive contains initialization inputs only, with no
partially trained RL checkpoint or credentials. The original corpus recipe
retains upstream dataset revisions and provenance.

## Download on the training machine

Keep the download, extracted inputs and checkpoints on persistent cloud storage.
The following example uses `/cloud/anytok-inputs`; substitute the actual mounted
cloud filesystem. Run from the repository root on branch `zett-modal-20k`.

```bash
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
ref = json.loads(Path('docs/raw-bpb-handoff/bundle-reference.json').read_text())
for name in ['anytok-training-inputs-v1.tar.gz',
             'anytok-training-inputs-v1.tar.gz.sha256', 'bundle-receipt.json']:
    hf_hub_download(repo_id=ref['repo_id'], repo_type='dataset',
                    revision=ref['revision'], filename=name,
                    local_dir='/cloud/anytok-inputs', token=False)
PY
```

```bash
cd /cloud/anytok-inputs
sha256sum -c anytok-training-inputs-v1.tar.gz.sha256
tar -xzf anytok-training-inputs-v1.tar.gz
```

Then return to the repository root and check **every extracted file**:

```bash
python scripts/verify_training_inputs.py --root /cloud/anytok-inputs
```

Expected: 1,062 verified files, 1,000 tokenizers, 23 corpus buckets. Leave room
for both the archive and extraction. A complete 20k checkpoint grid needs about
0.5 TB more; allow at least 600 GB for outputs and temporary checkpoint writes.

## Environment

Use Linux, Python **3.12**, and the pinned packages:

```bash
python -m pip install -r docs/raw-bpb-handoff/requirements.txt
```

The H200 validation used PyTorch **2.15.0+cu132**, CUDA 13.2. Use a compatible
CUDA driver and the same PyTorch build when comparing execution fidelity.
No Modal SDK or account is needed to run the standalone training entry point.

The repository contains ZeTT's compiled Python-3.12 Rust helper at
`runtime/anytok_pool_20k/rust_utils/target/release/librust_utils.so`. It needs
the Python shared library visible to the dynamic linker. The supplied binary
was tested on Linux x86_64 with glibc 2.34 and Python 3.12.13:

```bash
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

For another ABI, use a compatible environment or rebuild the same pinned Rust
source in `runtime/anytok_pool_20k/rust_utils`; retain its lockfile and sampler
behavior. The supplied binary is the verified H200 path. The launcher stages a self-contained
source tree beside the output directory and records all staged source hashes.

## Launch and resume

Inspect the command first; this starts no GPU work:

```bash
python scripts/run_raw_bpb_training.py \
  --inputs /cloud/anytok-inputs/input \
  --output /cloud/anytok-runs/raw-bpb-20k \
  --prepare-only
```

Launch through your platform's normal job allocation, with one H200 or a GPU
with sufficient memory:

```bash
python -u scripts/run_raw_bpb_training.py \
  --inputs /cloud/anytok-inputs/input \
  --output /cloud/anytok-runs/raw-bpb-20k
```

To resume the **same** raw-BPB experiment, repeat that command with `--resume`.
It checks the saved configuration and checkpoint checksums, then restores the
LM, hypernetwork, optimizer, policy, corpus position and all saved RNG states.
Keep the input mount paths consistent across restarts.
Do not change reward, entropy, scoring boundaries, corpus or pool mid-run.

The default stops at **20,000**; it does not rescale the original 190k-step LR
schedule. `--stop-at-update 2` is available for an explicit short startup test.
Use a separate smoke output directory. A successful startup test is not a full
evaluation or a completed 20k experiment.

Weights are saved every 500 updates. Full resumable state is saved every 2,000
and at a graceful stop/segment endpoint. Checkpoints use a temporary directory
and an atomic rename; each has a checksum manifest. SIGTERM/SIGINT is forwarded
to the trainer for its existing graceful save behavior.

## What must stay the same

| Setting | Value |
| --- | --- |
| Policy action | Select one of the complete 1,000 saved tokenizers each update |
| Policy | Settings features → 64 → 64 → 1 Tanh network; softmax over pool |
| Initialization | Exactly uniform action probabilities; policy seed 43 |
| Policy optimizer | Adam LR 0.001; EMA reward baseline decay 0.99 |
| Entropy weight | **0**, matching the current regret run |
| LM/HN seed | 42 |
| Update batch / microbatch | 128 / 32 |
| Context cap | **128 total positions: BOS + prefix + continuation** |
| Continuation length | Variable, as in current joint training; not fixed to 32 or 128 scored tokens |
| Paired cuts | Corrected reversible canonical boundary nearest the original midpoint; shared target IDs and bytes |
| Output head | Fixed canonical SmolLM2 head; native tokenizer changes the input prefix |
| Model objective | Transferred continuation token NLL + 0.5 × input lexical alignment |
| Peak learning rates | Hypernetwork 6e-5; LM 1e-6; original 10k-step ramp / 190k schedule |
| Sanity evaluation | Same two fixed ZeTT specifications, 23 buckets × 4 passages each |

The policy reward is:

```text
raw BPB = summed transferred continuation NLL / (summed continuation bytes × ln(2))
```

The adversary maximizes this value. Canonical BPB is still computed on the
exact same targets using current pre-update LM weights, without gradients or
changing training RNG. Keep it for diagnostics; **do not subtract it when
updating the raw-BPB policy**. `policy_reward_bpb` must equal `train_bpb`.
`paired_gap_bpb` must still equal `train_bpb - paired_canonical_bpb`.
Neither quantity includes lexical loss. The LM/HN objective is unchanged.

The old entropy-0.01 raw-BPB run used the older example-construction protocol.
Use this corrected entry point rather than the older original-RL launcher.

## Memory fix and verification

`scripts/pool_training_memory.py` applies the accepted `chunked_norm_mlp`
GPU-only execution and direct embedding backward. It reuses the already
validated normalization, source-preparation, recomputation and native MLP
operators. It keeps all vocabulary entries, global vocabulary attention,
native full-shaped matrix operations, dropout and parameter-gradient
reductions. No optimizer, activation or gradient CPU offload is enabled.

The complete native 128k test fit one Tillicum H200: **115.87 GiB allocated**,
**136.19 GiB reserved**, **6.34 s** for the measured training phases. That is a
short capacity measurement, not a long-run throughput estimate. See
[the H200 result](../../reports/tillicum-128k-smoke/README.md) and
[the numerical-fidelity study](../../reports/chunked-memory/README.md).
Outputs/losses/RNG matched in the same-device comparisons; gradient differences
were at the unmodified baseline's tiny accumulation-noise scale, not claimed
bit-identical across devices.

The integration check and test results are saved in this folder. The real-data
check starts fresh for two updates, resumes through update three, and verifies
raw rewards, corrected pairs and matching model/policy counters. The full 1,000
tokenizers remain available; a startup check samples only its three actions.

## Monitoring

The launcher streams logs and atomically writes `OUTPUT/status.json`.
`OUTPUT/events.jsonl` contains the trainer's full metrics. Set
`ANYTOK_RUN_PLATFORM` and `ANYTOK_RUN_PROFILE` to your platform/profile labels.
Register a long run in `plans/run-dashboard.json`, using its own stable ID:

```json
{
  "id": "raw-bpb-entropy0-training",
  "name": "Raw-BPB RL · entropy weight 0 · 20k steps",
  "category": "Training",
  "description": "Fresh raw-BPB comparison with corrected boundaries and the GPU-only memory fix.",
  "adapter": "progress_json",
  "platform": "Your training platform",
  "status": "/cloud/anytok-runs/raw-bpb-20k/status.json",
  "completed_field": "step",
  "total": 20000,
  "unit": "steps",
  "profiles": [],
  "app_names": [],
  "logs": ["/cloud/anytok-runs/raw-bpb-20k/events.jsonl"]
}
```

Serve with `python scripts/run_dashboard.py` on `0.0.0.0:1337`. If training runs
on a different host, copy only status/log metadata to the monitor host and
adjust these paths. Keep corpora, tokenizer files and checkpoints in persistent
cloud storage. On Modal, follow the repository instructions: GPU-only resource
request, and never use disabled profiles `general` or `research-everettcc`.
