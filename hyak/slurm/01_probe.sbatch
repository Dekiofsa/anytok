#!/bin/bash
# hyak/slurm/01_probe.sbatch -- stage probe: GPU memory and speed probe for training updates (D2).
#
# Submit (debug QoS and 1 h by default):   hyak/slurm/submit.sh probe
# Extra arguments for the probe:           --env HYAK_PROBE_ARGS="--chunk-rows 16384 8192"
#
# Runs python -m hyak.memory.probe --also-pinned in the train environment.  Needs only the
# environment, input/model, input/hypernet_init and code/uniform (no corpus, no pool).  It runs
# the chunked-versus-pinned equivalence check on the GPU, then two training-shaped updates per
# vocabulary size (32,768 to 128,000 rows).  The configured HYAK_HN_CHUNK_ROWS is measured for
# EVERY vocabulary size first and decides the verdict; the pinned comparison (--also-pinned, the
# unchunked forward; out-of-memory is caught) runs afterwards as information only and never
# changes the exit status.  One "[probe] VERDICT ..." line per setting and a "[probe] SUMMARY ..."
# line are printed.  Report: modal_results/hyak_tillicum/probe/probe-report.json.
# Exit 0: every measurement of the configured chunk size FITS and the equivalence passed.
# Exit 4: DOES NOT FIT, a measurement error, or an equivalence failure: do not start training.
# Exit 2: setup or unexpected error (configuration, missing files, no CUDA, trainer import failure,
#         library or driver failure; the job log has the message and a remedy).
# Exit 1: the probe crashed (traceback in the job log).
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

job_begin probe
hyak_status_start "$JOB_STAGE" "memory probe"
hyak_activate train
hyak_register_dashboard
job_extra_args HYAK_PROBE_ARGS

rc=0
job_step python -m hyak.memory.probe --also-pinned ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
case $rc in
  0) hyak_info "probe: the configured chunk size fits (VERDICT and SUMMARY lines above; report $JOB_DIR/probe-report.json)" ;;
  *) job_explain "$rc" "python -m hyak.memory.probe" \
       "4=memory probe: the configured chunk size DOES NOT FIT, a measurement failed or the equivalence check failed (SUMMARY line and $JOB_DIR/probe-report.json); do not start training" \
       "2=memory probe setup or unexpected error: configuration, input/model, input/hypernet_init, code/uniform, CUDA or a library or driver failure (the ERROR line and any traceback are in the job log)" ;;
esac
exit "$rc"
