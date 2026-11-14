#!/bin/bash
# hyak/slurm/06_init.sbatch -- stage init: build the random-init hypernetwork (D1), once.
#
# Submit (CPU work, about 4 GB RAM, minutes; fits the debug QoS):  hyak/slurm/submit.sh init --debug
# Extra arguments for init_hypernet:                              --env HYAK_INIT_ARGS=--verify
#
# Train environment.  Runs python -m hyak.model.init_hypernet --status into
# $ANYTOK_ROOT/input/hypernet_init, the warmup checkpoint of both runs.  The default recipe is
# option (b): seed 42, build the hypernetwork, calibrate the rescalers with the zero prior bias,
# check the computed in_scaler/scaler against original_in_scaler.npz within 1e-5 and copy the four
# frozen rescaler tensors from it, then draw the prior bias N(0, 0.02) from Generator(1042).
# Other recipes through HYAK_INIT_ARGS (each is recorded in the manifest and the status):
#   --calibrate-after-bias   option (a): draw the bias first, calibrate afterwards
#   --prior-bias-std 0       option (c): keep the library's zero bias (not recommended)
# Needs code/uniform (python -m hyak.code.stage_code) and input/model (python -m
# hyak.model.fetch_model).  Success line:
#   INIT OK: <out> model.safetensors 1376746800 bytes sha256 <hex> zero-prior gradient ratio <r>; recipe ...
# Re-running is safe: an existing directory built with the same recipe prints "(already built)".
# Exit 0 built or already built; 2 usage, configuration, input or environment error ("INIT FAILED:
# ..." with a remedy); 4 a check failed ("INIT FAILED: ...", for example the calibration check against
# the original in_scaler); 1 crashed (traceback); 128+n signal.
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

job_begin init
hyak_status_start "$JOB_STAGE" "random-init hypernetwork"
hyak_activate train
hyak_register_dashboard
job_extra_args HYAK_INIT_ARGS

rc=0
job_step python -m hyak.model.init_hypernet --status ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
case $rc in
  0) hyak_info "init: $ANYTOK_ROOT/input/hypernet_init is ready (INIT OK line above)" ;;
  *) job_explain "$rc" "python -m hyak.model.init_hypernet" \
       "2=hyak.model.init_hypernet: usage, configuration, input or environment error (model, code/uniform, paths); the INIT FAILED line in the job log has the remedy" \
       "4=hyak.model.init_hypernet: a check failed (for example the calibration against original_in_scaler.npz); the INIT FAILED line in the job log has the reason" ;;
esac
exit "$rc"
