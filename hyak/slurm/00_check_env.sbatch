#!/bin/bash
# hyak/slurm/00_check_env.sbatch -- stage check-env: prove the environment on one GPU node.
#
# Submit (debug QoS and 1 h by default):   hyak/slurm/submit.sh check-env
# Extra arguments for the checker:         --env HYAK_CHECK_ENV_ARGS="--no-hash-model"
#
# Runs python -m hyak.env.check_env in the train environment: Python 3.12 and the six pins, glibc,
# nvidia-smi and the driver/CUDA rule, a bf16 + fused-AdamW step on the GPU, the staged code tree,
# the committed librust_utils.so loaded exactly as the trainer loads it, both trainer imports,
# flock / rename / fsync on both storage roots, free space, the model files, (warning only) outbound
# HTTPS and (warning only) the job's end time: without SLURM_JOB_END_TIME a training segment has no
# in-process stop timer and relies on Slurm's --signal alone.  It writes modal_results/hyak_tillicum/check-env/{status.json, events.jsonl,
# check_env_report.json}.  The 3.4 GB model files are hashed by default (--no-hash-model skips it).
# Exit 0: no check failed (warnings are allowed).  Exit 4: at least one [FAIL] line; each has a
# "remedy:" line in the job log.  Exit 2: usage or configuration error.  Exit 1: the checker itself
# crashed (traceback in the job log).
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

job_begin check-env
hyak_status_start "$JOB_STAGE" "environment checks"
hyak_activate train
hyak_register_dashboard
job_extra_args HYAK_CHECK_ENV_ARGS

# check_env writes its own status (total = number of checks; final state complete or failed) and
# its report; do not pass --no-status here.
rc=0
job_step python -m hyak.env.check_env ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
case $rc in
  0) hyak_info "check-env: no check failed; report $JOB_DIR/check_env_report.json" ;;
  *) job_explain "$rc" "python -m hyak.env.check_env" \
       "4=hyak.env.check_env: at least one check failed; every [FAIL] line in the job log has a remedy" ;;
esac
exit "$rc"
