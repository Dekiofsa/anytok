# Joint regret RL20k with entropy zero

Fresh 20,000-step joint run, starting from the original SmolLM2-1.7B and the
same identity-warmed input hypernetwork as the original uniform and RL runs.
The same 1,000 native training-pool tokenizers remain available on every step.
The 18→64→64→1 shared-settings policy starts exactly uniformly, seed 43.
Adam policy LR .001, EMA .99, entropy 0; LM/HN seed 42, optimizers and the
original 190k-step learning-rate schedule are unchanged. The run stops at 20k,
just as before; the schedule is not rescaled to 20k.

## Authorized changes

The user requested a full joint RL20k rerun with entropy zero and simple regret.
The user additionally approved corrected paired token boundaries explicitly.
Training examples use the closest reversible canonical boundary to the old
character midpoint, shrinking the same 512-character window by .85 as needed.
Both input methods must fit BOS + prefix + continuation within 128 positions.
The continuation is variable length, as in original joint training; the frozen
policy experiment's fixed 32-token targets are not substituted into training.
No training domain, tokenizer, source corpus or held-out bucket is removed.

Each update samples 128 fresh examples and one tokenizer. Canonical and native
input prefixes represent identical text; continuation IDs and bytes are exactly
shared. Canonical joint encoding must equal prefix IDs followed by target IDs.
Both losses use the current pre-update LM and fixed canonical output head.
Canonical scoring has no gradients and restores its CPU/CUDA RNG consumption.
The policy receives (transferred NLL - canonical NLL) / shared bytes / ln(2).
Batch BPB is byte-weighted, as in the original joint run (not the frozen
experiment's equal-domain full-panel average). The model objective remains
transferred continuation token NLL plus .5 lexical alignment; canonical loss
is used only for the adversary reward. No gradients train against canonical loss.

Periodic sanity uses the same two fixed ZeTT tokenizer specs, original 23 heldout
buckets and four passages per bucket. Cuts are also corrected for both inputs;
old uniform curves cannot be overlaid as a paired comparison. These are sanity
checks, not a replacement for the original broad final evaluation panels.

## Persistence, compute and monitoring

B300 on Modal; no explicit CPU or memory reservations. All corpus, native pool,
base-model and warmup artifacts are reused with hash verification in persistent
cloud volumes. Transfers are cloud-to-cloud; only logs and metadata come local.
New output prefix regret-run/ in anytok-pool-rl-20k; separate regret-smoke/.
Weights (LM, HN, policy) every 500 steps; full restart every 2,000 and bounded
segment endpoints. Hand off complete verified state across enabled profiles,
then gather all 40 grid checkpoints into the final workspace's volume.
Permanently disabled general and research-everettcc are excluded.
Check actual compute eligibility; $30 minus applied credits is an estimate.
Keep the inherited $2 reserve and $8.25/hour conservative scheduling allowance;
never request paid overage. Stop safely at a durable checkpoint if credits run out.

Dashboard: http://100.84.104.59:1337/?run=anytok-pool-regret-rl-training
State/logs: modal_results/anytok_pool_regret_rl_20k/.
Source snapshot: runtime/anytok_pool_regret_rl_20k/ (shared policy unchanged).
Launch: ANYTOK_RL_EXPERIMENT=regret python scripts/run_anytok_pool_rl_20k.py --run.
Before the full run, the controller runs two GPU updates and a separate-container
resume through update three; it verifies matched counters, paired rewards,
128 valid paired examples and the 128-position maximum.

## Verified launch

38 focused tests passed. Two GPU updates and an independent-container resume
through update three passed all paired-reward and checkpoint checks. Fresh full
training reproduces the smoke/resumed run's first three update metrics exactly
(excluding runtime and peak memory), including context hashes, actions, losses,
policy advantages and probabilities. Initial sanity scores also match exactly.

Training profile: queues. App `ap-Zs8ZtQSjXUwZXdgCAc5c7c`, call
`fc-01M7CAD80PMWK668CEGW63J2SA`. Launch receipt:
`modal_results/anytok_pool_regret_rl_20k/verification.json`, also persisted as
`queues / anytok-pool-rl-20k / regret-run/launch_verification.json`.

## Tillicum continuation (2026-11-14)

The user authorized continuing the stopped Modal experiment on Tillicum. The
original queued Modal app was stopped to prevent duplicate training. The last
complete restart is transformer / anytok-pool-rl-20k / regret-run/step-005288.
The later logged update 5815 and weights-only checkpoint 5500 do not contain
complete restart state. The interrupted source trajectory remains archived;
continuation replays from committed step 5288.

All eleven restart/config/metadata files (23,475,030,053 bytes) transferred
directly from Modal to Tillicum, with every source hash verified. The existing
full input bundle was verified identical. No large payload crossed the local
caller. Original Modal checkpoints remain unchanged. The complete restart is
also verified in cminst/anytok-regret-rl-20k before resumed training.

Tillicum job 470164: zlab account, gpu-h200 partition, normal QoS, one H200,
24-hour wall limit, no explicit CPU or memory reservations. The scheduler
assigned 4 CPUs / 120G RAM. At the published $0.90/GPU-hour rate, a full
24-hour allocation is $21.60; actual billing follows elapsed time.
Official rate/limits: https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs/

The guarded migration permits only model/warmup paths, recorded trainer source
and validated memory-execution fields to change. Every scientific setting,
regret reward, entropy weight 0, policy, optimizer, corpus position and saved
RNG state is retained. The new execution is the accepted GPU-only
chunked_norm_mlp path and direct embedding backward. Cross-device bitwise
trajectory equality is not claimed; hardware changes CUDA dropout mapping.

Resumed updates passed paired-example, reward and counter checks. The first
127k-vocabulary update peaked at 115.18 GiB allocated. Training continues to
20k with 500-step weights, 2k-step full restarts and graceful-stop restarts.
Checkpoints are atomically written, checksum-verified and uploaded directly
from Tillicum to Hugging Face. The latest-restart pointer is published only
after payload verification. A successful graceful walltime stop submits the
next job with an afterok dependency; failed training is not automatically
restarted. A five-minute pre-walltime signal allows a final checkpoint/backup.

Remote run: /gpfs/scrubbed/qxli2/anytok-h200-smoke/modal_results/tillicum_regret/run
Job driver: scripts/tillicum_regret_job.py
Migration guard: scripts/resume_regret_tillicum.py
Live metadata mirror: scripts/watch_tillicum_regret.py in tmux
anytok-tillicum-regret. The existing dashboard run ID and learning plots are
reused. Its committed history excludes the discarded later Modal trajectory.
Checkpoint repository: https://huggingface.co/cminst/anytok-regret-rl-20k
Migration receipts and initial progress: reports/tillicum-regret-resume/.
