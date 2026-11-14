#!/bin/bash
# hyak/slurm/10_replay_check.sbatch -- stage replay-check: corpus + pool fidelity before GPU training.
#
# Submit after stage-inputs and the pool collect (CPU work, about an hour):  hyak/slurm/submit.sh replay-check
# Extra arguments for the example replay:      --env HYAK_REPLAY_CHECK_ARGS="--strict"   (or --updates N)
#
# Train environment.  Two replays of the pinned uniform trainer's example construction, compared
# with reports/rl-tokenizer-selection-20k/data/training/uniform.csv:
#   1. python -m hyak.train.replay_check --schedule-only        RNG schedule of all 20,000 updates
#      (pool index and vocabulary size per update, and the final Python RNG digest); needs no corpus.
#      Exit 4 when the schedule or the RNG digest does not match the original run.
#   2. python -m hyak.train.replay_check [$HYAK_REPLAY_CHECK_ARGS]   the example replay; its default
#      is the first 2,000 updates (860 of the 1,000 tokenizers; the original rejections at updates
#      381, 619 and 1312), built from the staged corpus and pool: tokenizer index, vocabulary size,
#      scored_tokens, rejected_examples and truncated_forms per update.  (--updates N chooses another
#      length.)
# Reports: modal_results/hyak_tillicum/replay-check/replay-check-*.json.  Exit status of replay_check:
# 0 the replay ran (a mismatch of an example column is a fidelity result in the report, not a job
# failure; --strict in the extra arguments makes it fail with 4); 4 a check failed: the schedule or the
# RNG digest differs, the example replay hit an update at which the trainer itself would raise, or a
# column mismatched under --strict; 2 the inputs, staged code or environment are missing or wrong (staged
# code, corpus, pool or model; see the job log); 1 crashed (traceback).
#
# No #SBATCH lines on purpose: hyak/slurm/submit.sh passes every resource (partition, QoS, account,
# GPU, CPUs, memory, time), the job name, the output file, --signal and --export from
# hyak/config.env.  Do not submit this file with plain sbatch.

# --- common job prologue: identical in every hyak/slurm/*.sbatch (checked by hyak/tests/test_integration_sbatch.py) ---
set -euo pipefail
if [ -z "${HYAK_REPO:-}" ] || [ ! -r "$HYAK_REPO/hyak/slurm/job.sh" ]; then
  echo "[hyak] ERROR: HYAK_REPO must name the repository checkout; submit with hyak/slurm/submit.sh" >&2
  exit 2
fi
# shellcheck source=hyak/slurm/lib.sh
source "$HYAK_REPO/hyak/slurm/lib.sh"
# shellcheck source=hyak/slurm/job.sh
source "$HYAK_REPO/hyak/slurm/job.sh"
# --- end of common job prologue ---

job_begin replay-check
hyak_status_start "$JOB_STAGE" "replay of the uniform run's example construction"
hyak_activate train
hyak_register_dashboard

rc=0
job_step python -m hyak.train.replay_check --schedule-only || rc=$?
if [ "$rc" = 0 ]; then
  if job_stopping; then exit 0; fi
  job_extra_args HYAK_REPLAY_CHECK_ARGS
  job_step python -m hyak.train.replay_check ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
fi
case $rc in
  0) hyak_info "replay-check: reports in $JOB_DIR (exact-match rate per column)" ;;
  *) job_explain "$rc" "python -m hyak.train.replay_check" \
       "4=replay-check: a check failed: the RNG schedule or its digest differs from the original run, the example replay hit an update at which the trainer would raise, or a column mismatched (--strict); see the job log and the report in $JOB_DIR" \
       "2=hyak.train.replay_check could not run (exit 2): missing or wrong staged code, corpus, pool, model or environment; see the job log" ;;
esac
exit "$rc"
