#!/bin/bash
# hyak/slurm/12_smoke_rawbpb.sbatch -- stage smoke-rawbpb: startup and restart check of the raw-BPB run.
#
# Submit (after bash hyak/handoff/fetch_inputs.sh verified the bundle):
#   hyak/slurm/submit.sh smoke-rawbpb
#
# Train environment, one H200, the debug QoS and its 1 h limit by default (HYAK_QOS_DEBUG, HYAK_TIME_DEBUG;
# per-stage HYAK_QOS_SMOKE_RAWBPB / HYAK_TIME_SMOKE_RAWBPB override them), margin
# HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_RAWBPB (10).  What Qingyun's scripts/verify_raw_bpb_handoff.py did on
# Tillicum, in a separate output ($HYAK_RAWBPB_SMOKE_OUTPUT, default $ANYTOK_RUNS/raw-bpb-smoke, which must
# not exist or be empty: the smoke never deletes anything):
#   0. the installed packages equal docs/raw-bpb-handoff/requirements.txt (exit 2 otherwise);
#   1. python -u scripts/run_raw_bpb_training.py --inputs $ANYTOK_ROOT/input --output SMOKE --stop-at-update 2
#      (fresh), then python -m hyak.handoff.run verify --smoke --expect-step 2;
#   2. the same command with --resume --stop-at-update 3, then verify --smoke --expect-step 3: status.json
#      complete at 3, train rows 1..3 with policy_reward_bpb == train_bpb, paired_gap_bpb == train_bpb -
#      paired_canonical_bpb, 128 paired examples, at most 128 positions, policy_updates == step, run_config
#      reward objective raw_bpb, entropy 0, and the model/policy/selection counters of step-000002 and
#      step-000003 equal to 2 and 3.
# The trainer's output goes to <monitor>/<jobid>-train.log, the launcher's own messages to the Slurm log.
# Exit 0 only when every check passed (a "rawbpb_verified" line at step 3; the stage status ends "complete").
# 2: environment, input or a non-empty smoke output (remedy in the log); 4: a check failed; 1: a crash; 137 or
# another 128+n: killed; 143: stopped by a signal.  Delete the smoke output afterwards (three full restarts,
# about 70 GB).
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

job_begin smoke-rawbpb
TRAIN_LOG="$JOB_DIR/${SLURM_JOB_ID:-local-$$}-train.log"
hyak_status_start "$JOB_STAGE" "raw-BPB smoke: 2 updates fresh, then a resume to update 3"
hyak_activate train
hyak_register_dashboard
# shellcheck source=hyak/handoff/job_rawbpb.sh
source "$HYAK_REPO/hyak/handoff/job_rawbpb.sh"
rawbpb_setup
SMOKE=$RAWBPB_SMOKE_OUTPUT

rawbpb_requirements || exit $?
rc=0
python -m hyak.handoff.run prepare --output "$SMOKE" --smoke >>"$TRAIN_LOG" || rc=$?
if [ "$rc" != 0 ]; then
  job_explain "$rc" "python -m hyak.handoff.run prepare --smoke" \
    "2=the smoke output $SMOKE is not empty; the remedy (rm -rf of that smoke directory) is in the job log"
  exit "$rc"
fi
rawbpb_link "$SMOKE"
hyak_info "trainer output goes to $TRAIN_LOG"

for phase in 2 3; do
  extra=(--stop-at-update "$phase")
  [ "$phase" = 2 ] || extra=(--resume --stop-at-update "$phase")
  rawbpb_launch "$SMOKE" "${extra[@]}"
  rc=$RAWBPB_RC
  if [ "$rc" != 0 ]; then
    rawbpb_explain "$rc"
    exit "$rc"
  fi
  if job_stopping; then
    exit 143                   # a stop was asked for: the smoke is incomplete, not failed
  fi
  python -m hyak.handoff.run verify --output "$SMOKE" --smoke --expect-step "$phase" >>"$TRAIN_LOG" || rc=$?
  if [ "$rc" != 0 ]; then
    job_explain "$rc" "python -m hyak.handoff.run verify --smoke" \
      "4=smoke-rawbpb: a check failed after the phase that stops at update $phase; every CHECK FAILED line is in the job log"
    exit "$rc"
  fi
  hyak_info "smoke-rawbpb: phase to update $phase verified"
done
python -m hyak.handoff.run mark --stage "$JOB_STAGE" --output "$SMOKE" --complete >/dev/null \
  || hyak_warn "could not mark $JOB_STAGE complete in its status.json"
hyak_info "smoke-rawbpb: all checks passed (rawbpb_verified at update 3 in $TRAIN_LOG); delete $SMOKE when you have read it"
exit 0
