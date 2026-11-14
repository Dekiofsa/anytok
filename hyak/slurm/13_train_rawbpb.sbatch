#!/bin/bash
# hyak/slurm/13_train_rawbpb.sbatch -- stage train-rawbpb: one 24 h job of the raw-BPB 20k run.
#
# Submit a chain (singleton: one job at a time; a job that finds the run complete exits 0 at once):
#   hyak/slurm/submit.sh train-rawbpb --count 3
#
# Train environment, one H200, QoS normal, 24 h (HYAK_TIME), Slurm's --signal=B:TERM
# HYAK_SEGMENT_MARGIN_MINUTES_TRAIN_RAWBPB (20) minutes before the end.  The run is Qingyun's launcher
# scripts/run_raw_bpb_training.py, unchanged (raw-BPB policy reward, entropy 0, corrected paired boundaries,
# GPU-only memory fix, 20,000 updates from the bundle's SmolLM2-1.7B and identity-warmed hypernetwork), with
# --inputs $ANYTOK_ROOT/input --output $HYAK_RAWBPB_OUTPUT (default $ANYTOK_RUNS/raw-bpb-20k).  Each job:
#   1. refuses to start (exit 3) while modal_results/hyak_tillicum/train-rawbpb/HALTED.json exists (an earlier
#      job of the chain failed: inspect, fix, delete the file, resubmit);
#   2. python -m hyak.handoff.run prepare: complete (the newest full restart is at 20,000) -> exit 0 with a
#      note, nothing else; a resumable full restart -> the launcher with --resume (weights-only checkpoints
#      newer than that restart and stale .writing-* directories of a killed save are moved to
#      OUTPUT/interrupted/<UTC>/ first, never deleted); an empty or missing output -> a fresh start; trainer
#      files without a resumable restart -> exit 2 with the remedy (nothing is deleted); the trainer's lock
#      held -> exit 0 (not now);
#   3. before a launch, requires the installed packages to equal docs/raw-bpb-handoff/requirements.txt (exit 2);
#   4. runs the launcher as a forwarded child: Slurm's TERM reaches this batch shell, which forwards it to the
#      launcher, which forwards it to the trainer; the trainer finishes the update, evaluates, writes a full
#      restart and records paused_at (the job then ends 0, status "paused": resubmit, or let the chain go on);
#   5. after exit 0, python -m hyak.handoff.run verify: the raw-BPB receipt of every update of this job and the
#      counters of the newest full restart (exit 4 on a failure; no checkpoint progress is a failure unless a
#      stop was requested).
# Status: the launcher writes OUTPUT/status.json and OUTPUT/events.jsonl (the dashboard entry
# hyak-raw-bpb-training reads them through modal_results/hyak_tillicum/train-rawbpb/run); the job's own life
# cycle is in modal_results/hyak_tillicum/train-rawbpb/{status.json,events.jsonl}; the trainer's lines in
# <jobid>-train.log there.  python -m hyak.tools.policy_monitor --follow refreshes policy-monitor.json from
# the 500-step checkpoints for the learning panel.  Every end except 0, 130 and 143 writes HALTED.json and
# halts the chain: 1 crash, 2 configuration or input, 3 a decision is needed, 4 a check failed, 137 or another
# 128+n a killed trainer.  Stop gracefully with hyak/slurm/stop.sh train-rawbpb.
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

job_begin train-rawbpb
JOB_FINALIZE=rawbpb_finalize
TRAIN_LOG="$JOB_DIR/${SLURM_JOB_ID:-local-$$}-train.log"
HALT_FILE="$JOB_DIR/HALTED.json"
POLICY_LOG="$JOB_DIR/policy-monitor.log"
POLICY_PID=""
ACTIVE=0
OUTPUT=""

# rawbpb_halt CODE: write HALTED.json (returns 1 when it already exists).
rawbpb_halt() {
  [ ! -e "$HALT_FILE" ] || return 1
  "$(hyak_python)" -c 'import json, sys, time
keys = ("stage", "output", "job_id", "exit_code", "slurm_log", "train_log", "reason")
data = dict(zip(keys, sys.argv[1:]))
data["exit_code"] = int(data["exit_code"])
data["at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
data["remedy"] = ("inspect slurm_log and train_log, fix the cause, delete this file, then resubmit the chain "
                  "(hyak/slurm/submit.sh train-rawbpb --count N)")
with open(sys.argv[-1], "x") as handle:
    handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")' \
    "$JOB_STAGE" "$OUTPUT" "${SLURM_JOB_ID:-}" "$1" "$JOB_LOG" "$TRAIN_LOG" "$JOB_MESSAGE" "$HALT_FILE" \
    || printf 'stage=%s job=%s exit_code=%s\n' "$JOB_STAGE" "${SLURM_JOB_ID:-}" "$1" > "$HALT_FILE" \
    || hyak_warn "could not write $HALT_FILE"
  return 0
}

# rawbpb_finalize CODE (EXIT handler): stop the policy collector, refresh it once, halt the chain after a
# failure, copy the launcher's step into the stage status and record the job's end.
rawbpb_finalize() {
  local code=$1
  if [ -n "$POLICY_PID" ]; then
    kill -TERM "$POLICY_PID" 2>/dev/null || true
    wait "$POLICY_PID" 2>/dev/null || true
    POLICY_PID=""
  fi
  case $code in
    0|130|143) ;;
    *) if rawbpb_halt "$code"; then
         JOB_MESSAGE="${JOB_MESSAGE:+$JOB_MESSAGE; }chain halted: later $JOB_STAGE jobs exit at once until $HALT_FILE is deleted"
       fi ;;
  esac
  if [ "$ACTIVE" = 1 ] && [ -n "$OUTPUT" ] && [ -d "$OUTPUT" ]; then
    python -m hyak.tools.policy_monitor --once --run-dir "$OUTPUT" --output "$JOB_DIR/policy-monitor.json" \
      --plan "$RAWBPB_INPUTS/pool/plan.json" >>"$POLICY_LOG" 2>&1 || hyak_warn "the final policy refresh failed (see $POLICY_LOG)"
    python -m hyak.handoff.run mark --stage "$JOB_STAGE" --output "$OUTPUT" ${RAWBPB_COMPLETE:+--complete} >/dev/null 2>&1 \
      || hyak_warn "could not copy the launcher's step into the $JOB_STAGE status"
  fi
  job_finalize_status "$code"
}

if [ -e "$HALT_FILE" ]; then
  JOB_MESSAGE="the raw-BPB chain is halted by $HALT_FILE (an earlier job failed); inspect the logs it names, fix the cause, delete the file and resubmit"
  hyak_warn "$JOB_MESSAGE"
  cat "$HALT_FILE" >&2 || true
  exit 3
fi

hyak_status_start "$JOB_STAGE" "raw-BPB 20k training job"
hyak_activate train
hyak_register_dashboard
# shellcheck source=hyak/handoff/job_rawbpb.sh
source "$HYAK_REPO/hyak/handoff/job_rawbpb.sh"
rawbpb_setup
OUTPUT=$RAWBPB_TRAIN_OUTPUT
RAWBPB_COMPLETE=""
ACTIVE=1
rawbpb_link "$OUTPUT"

rc=0
answer=$(python -m hyak.handoff.run prepare --output "$OUTPUT" --total 20000) || rc=$?
printf '%s\n' "$answer" >>"$TRAIN_LOG"
hyak_info "prepare (exit $rc): $answer"
RE_ACTION='"action": "([a-z_]+)"'
RE_NEWEST='"newest_full_step": ([0-9]+)'
RE_REASON='"reason": "([a-z_]+)"'
action=""; newest=""; reason=""
if [[ $answer =~ $RE_ACTION ]]; then action=${BASH_REMATCH[1]}; fi
if [[ $answer =~ $RE_NEWEST ]]; then newest=${BASH_REMATCH[1]}; fi
if [[ $answer =~ $RE_REASON ]]; then reason=${BASH_REMATCH[1]}; fi
hyak_status_event "$JOB_STAGE" rawbpb_prepare "action=${action:-none}" "newest_full_step=${newest:-none}" "exit=$rc"
case $rc in
  0) ;;
  75)
    if [ "$action" = complete ]; then
      RAWBPB_COMPLETE=1
      job_not_now complete "the raw-BPB run is complete (full restart at step $newest); nothing to do"
    else
      job_not_now "${reason:-not_now}" "no launch: ${reason:-not now} ($answer)"
    fi
    exit 0 ;;
  *)
    job_explain "$rc" "python -m hyak.handoff.run prepare" \
      "2=the output $OUTPUT holds trainer files but no resumable restart (or is not a directory); the remedy is in the job log, nothing was deleted" \
      "3=the output $OUTPUT needs your decision (an incomplete newest restart, or not a raw-BPB run); see the job log"
    exit "$rc" ;;
esac
rawbpb_requirements || exit $?
LAUNCH_ARGS=()
case $action in
  resume) LAUNCH_ARGS=(--resume); hyak_info "resuming from the full restart at step $newest" ;;
  fresh) newest=""; hyak_info "fresh start in $OUTPUT" ;;
  *) JOB_MESSAGE="prepare answered an unknown action '$action'"; exit 2 ;;
esac

python -m hyak.tools.policy_monitor --follow --interval 300 --run-dir "$OUTPUT" --output "$JOB_DIR/policy-monitor.json" \
  --plan "$RAWBPB_INPUTS/pool/plan.json" >>"$POLICY_LOG" 2>&1 &
POLICY_PID=$!

before_lines=$(python -m hyak.handoff.run count-lines --output "$OUTPUT")
hyak_info "trainer output goes to $TRAIN_LOG"
rawbpb_launch "$OUTPUT" ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"}
rc=$RAWBPB_RC
if [ "$rc" != 0 ]; then
  rawbpb_explain "$rc"
  exit "$rc"
fi
VERIFY_ARGS=(--output "$OUTPUT" --since-line "$before_lines")
[ -z "$newest" ] || VERIFY_ARGS+=(--before-step "$newest")
[ "$HYAK_TERM_RECEIVED" = 0 ] || VERIFY_ARGS+=(--stop-requested)
verdict=$(python -m hyak.handoff.run verify "${VERIFY_ARGS[@]}") || rc=$?
printf '%s\n' "$verdict" >>"$TRAIN_LOG"
if [ "$rc" != 0 ]; then
  job_explain "$rc" "python -m hyak.handoff.run verify" \
    "4=train-rawbpb: the raw-BPB receipt failed after this job's launch (every CHECK FAILED line is in the job log)"
  exit "$rc"
fi
if [[ $verdict =~ $RE_NEWEST ]] && [ "${BASH_REMATCH[1]}" -ge 20000 ]; then
  RAWBPB_COMPLETE=1
  hyak_info "the raw-BPB run reached update 20,000"
fi
hyak_status_event "$JOB_STAGE" rawbpb_verified "newest_full_step=${BASH_REMATCH[1]:-none}"
exit 0
