#!/bin/bash
# hyak/slurm/11_smoke_train.sbatch -- stage smoke-train: pre-flight of one run in a scratch directory.
#
# Submit:   hyak/slurm/submit.sh smoke-train --run uniform
#           hyak/slurm/submit.sh smoke-train --run rl        (needs the frozen uniform pair, i.e.
#                                                             after the uniform bootstrap segment)
#
# Train environment, one H200.  Runs python -m hyak.train.run_segment --run RUN --smoke in
# $ANYTOK_RUNS/smoke/<run> (wiped first): the launchers' pre-flight sequence as two child
# processes (uniform: bootstrap then a resume to update 3; RL: 2 updates then a resume to 3, policy
# counters must equal 3) with the production --save-every / --eval-every.  Exit 0 only when every
# check passes (a "smoke_verified" line in <monitor>/<jobid>-train.log).  The background monitor
# (python -m hyak.train.monitor --run RUN --smoke) keeps modal_results/hyak_tillicum/smoke-train/
# status.json current and writes the final state ("complete" at update 3 on success: it is not a 20k
# run).  Defaults: a 2 h limit and a 10-minute margin (HYAK_TIME_SMOKE_TRAIN,
# HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_TRAIN in config.env.example).
# run_segment --smoke exit status: 0 = smoke_verified (every check passed; the monitor ends the status
# "complete" by itself); 75 = not now (the RL smoke before the uniform bootstrap froze the pair, or the
# smoke lock is still held by another job after 10 minutes): the job ends 0 without a failure, with a note,
# and no HALTED file; submit it again later.  Exit 1: crashed; 2: configuration or input error; 3: a protocol
# stop that needs your decision; 4: a pre-flight check failed; 137: killed (memory).
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

job_begin smoke-train
RUN=${HYAK_RUN:-}
case $RUN in
  uniform|rl) ;;
  *) hyak_die "HYAK_RUN must be uniform or rl (submit with: hyak/slurm/submit.sh smoke-train --run uniform|rl)" ;;
esac
JOB_FINALIZE=smoke_finalize
TRAIN_LOG="$JOB_DIR/${SLURM_JOB_ID:-local-$$}-train.log"
MONITOR_LOG="$JOB_DIR/monitor.log"
MONITOR_PID=""
SMOKE_ACTIVE=0

# smoke_stop_monitor: stop the background monitor and wait (at most 120 s) until it has exited.
smoke_stop_monitor() {
  local waited=0
  [ -n "$MONITOR_PID" ] || return 0
  kill -TERM "$MONITOR_PID" 2>/dev/null || true
  while kill -0 "$MONITOR_PID" 2>/dev/null; do
    if [ "$waited" -ge 120 ]; then
      hyak_warn "monitor $MONITOR_PID did not stop within 120 s; killing it"
      kill -KILL "$MONITOR_PID" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "$MONITOR_PID" 2>/dev/null || true
  MONITOR_PID=""
}

# smoke_finalize CODE (EXIT handler): stop the monitor, refresh once more, record the job's end.
smoke_finalize() {
  smoke_stop_monitor
  if [ "$SMOKE_ACTIVE" = 1 ]; then
    python -m hyak.train.monitor --run "$RUN" --smoke --once >>"$MONITOR_LOG" 2>&1 \
      || hyak_warn "the final monitor refresh failed (see $MONITOR_LOG)"
  fi
  job_finalize_status "$1"
}

hyak_status_start "$JOB_STAGE" "$RUN pre-flight smoke"
hyak_activate train
SMOKE_ACTIVE=1
hyak_register_dashboard

python -m hyak.train.monitor --run "$RUN" --smoke --parent-pid "$$" >>"$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
hyak_info "monitor started (pid $MONITOR_PID, log $MONITOR_LOG); trainer output goes to $TRAIN_LOG"

rc=0
job_step python -m hyak.train.run_segment --run "$RUN" --smoke >>"$TRAIN_LOG" || rc=$?
case $rc in
  0) hyak_info "smoke-train $RUN: all checks passed (smoke_verified in $TRAIN_LOG)" ;;
  75) job_not_now smoke_not_now "smoke-train $RUN did not run (exit 75): the RL smoke needs the frozen uniform pair (wait for the uniform bootstrap), or the smoke lock is held by another job; submit it again later (the trainer's lines are in $TRAIN_LOG)"
      rc=0 ;;
  *) job_explain "$rc" "python -m hyak.train.run_segment --run $RUN --smoke" \
       "4=smoke-train $RUN: a pre-flight check failed; the reason is in the Slurm log"
     [ -z "$JOB_MESSAGE" ] || JOB_MESSAGE="$JOB_MESSAGE; the trainer's lines are in $TRAIN_LOG" ;;
esac
exit "$rc"
