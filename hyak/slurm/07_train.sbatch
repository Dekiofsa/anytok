#!/bin/bash
# hyak/slurm/07_train.sbatch -- stages train-uniform and train-rl: segments of one 20k training run.
#
# Submit a chain of N 24 h jobs (singleton: one job of a run at a time; extra jobs exit at once
# when the run is complete):
#   hyak/slurm/submit.sh train-uniform --count N
#   hyak/slurm/submit.sh train-rl --count N        (only after the uniform bootstrap froze the pair:
#                                                   submit.sh refuses train-rl while $ANYTOK_ROOT/input/pairing.json is missing)
#
# Train environment, one H200.  The job
#   1. refuses to start (exit 3: it needs your decision) while modal_results/hyak_tillicum/<stage>/HALTED.json
#      exists (written when an earlier job of the chain failed: like the Modal controllers, a failed
#      segment is never retried automatically; inspect, fix, delete the file, resubmit);
#   2. starts python -m hyak.train.monitor --run RUN in the background (status.json, events.jsonl,
#      for RL also policy-monitor.json; its own output goes to <monitor>/monitor.log);
#   3. loops: python -m hyak.train.run_segment --run RUN --should-start (exit 0 = start; 75 = "not
#      now": the run is complete, too little time is left, or (RL) the uniform bootstrap has not frozen
#      input/pairing.json yet -- the job ends with 0, writes no HALTED.json and the status says why;
#      anything else is a failure), then one segment
#      python -m hyak.train.run_segment --run RUN with its stdout (the trainer's JSON lines) in
#      <monitor>/<jobid>-train.log and its stderr in the Slurm log.  The uniform bootstrap
#      (one update) and the following resume run in the same job.  The loop stops on a stop signal,
#      on a failed segment, and when a segment exits 0 without a newer full restart (an error,
#      as in the Modal controllers, except while only the final audit is pending);
#   4. stops the monitor, refreshes the status once more and records the job's end.
# The job's exit status is the last segment's (or --should-start's when that failed).  A stop by
# the segment timer or by Slurm's --signal=B:TERM ends the job with 0 and status "paused".  Every
# status except 0, 130 and 143 halts the chain (HALTED.json): 1 is a crash (traceback in the Slurm
# log), 2 a configuration or input error, 4 a failed check, 137 or another 128+n a killed trainer
# (status "failed", never "stopped").  To stop a run gracefully use hyak/slurm/stop.sh.
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

job_begin train-uniform train-rl
RUN=${JOB_STAGE#train-}
if [ -n "${HYAK_RUN:-}" ] && [ "$HYAK_RUN" != "$RUN" ]; then
  hyak_die "HYAK_RUN=$HYAK_RUN contradicts stage $JOB_STAGE"
fi
export HYAK_RUN="$RUN"
JOB_FINALIZE=train_finalize
TRAIN_LOG="$JOB_DIR/${SLURM_JOB_ID:-local-$$}-train.log"
MONITOR_LOG="$JOB_DIR/monitor.log"
HALT_FILE="$JOB_DIR/HALTED.json"
MONITOR_PID=""
TRAIN_ACTIVE=0

# train_stop_monitor: stop the background monitor and wait (at most 120 s) until it has exited, so
# its last status write cannot land after the job's final status.
train_stop_monitor() {
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

# train_halt CODE: write HALTED.json (returns 1 when it already exists).
train_halt() {
  [ ! -e "$HALT_FILE" ] || return 1
  "$(hyak_python)" -c 'import json, sys, time
keys = ("stage", "run", "job_id", "exit_code", "slurm_log", "train_log", "reason")
data = dict(zip(keys, sys.argv[1:]))
data["exit_code"] = int(data["exit_code"])
data["at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
data["remedy"] = ("inspect slurm_log and train_log (python -m hyak.train.retention summary --run RUN helps), "
                  "fix the cause, delete this file, then resubmit the chain")
with open(sys.argv[-1], "x") as handle:
    handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")' \
    "$JOB_STAGE" "$RUN" "${SLURM_JOB_ID:-}" "$1" "$JOB_LOG" "$TRAIN_LOG" "$JOB_MESSAGE" "$HALT_FILE" \
    || printf 'stage=%s run=%s job=%s exit_code=%s\n' "$JOB_STAGE" "$RUN" "${SLURM_JOB_ID:-}" "$1" > "$HALT_FILE" \
    || hyak_warn "could not write $HALT_FILE"
  return 0
}

# train_finalize CODE (EXIT handler): stop the monitor, halt the chain after a failure, refresh the
# status from the trainer's events once more and record the job's end (hyak.common.status finalize).
train_finalize() {
  local code=$1
  train_stop_monitor
  case $code in
    0|130|143) ;;
    *) if train_halt "$code"; then
         JOB_MESSAGE="${JOB_MESSAGE:+$JOB_MESSAGE; }chain halted: later $JOB_STAGE jobs exit at once until $HALT_FILE is deleted"
       fi ;;
  esac
  if [ "$TRAIN_ACTIVE" = 1 ]; then
    python -m hyak.train.monitor --run "$RUN" --once >>"$MONITOR_LOG" 2>&1 \
      || hyak_warn "the final monitor refresh failed (see $MONITOR_LOG)"
  fi
  job_finalize_status "$code"
}

if [ -e "$HALT_FILE" ]; then
  JOB_MESSAGE="the $RUN chain is halted by $HALT_FILE (an earlier job failed); inspect the logs it names, fix the cause, delete the file and resubmit"
  hyak_warn "$JOB_MESSAGE"
  cat "$HALT_FILE" >&2 || true
  exit 3                       # a protocol stop: it needs your decision (inspect, fix, delete the file, resubmit)
fi

hyak_status_start "$JOB_STAGE" "$RUN training job"
hyak_activate train
TRAIN_ACTIVE=1
hyak_register_dashboard

python -m hyak.train.monitor --run "$RUN" --parent-pid "$$" >>"$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
hyak_info "monitor started (pid $MONITOR_PID, log $MONITOR_LOG); trainer output goes to $TRAIN_LOG"

RE_NEWEST='"newest_full_step": (null|[0-9]+)'
RE_REASON='"reason": "([a-z_]+)"'
rc=0
segments=0
previous_newest=""
# HYAK_ACCEPT_CHUNK_CHANGE="<reason>" (submit.sh --env) accepts a changed chunked-forward setting on the
# record (run_segment --accept-chunk-change, README 6.10).  Segments whose setting is unchanged ignore it.
SEGMENT_ARGS=()
if [ -n "${HYAK_ACCEPT_CHUNK_CHANGE:-}" ]; then
  SEGMENT_ARGS=(--accept-chunk-change "$HYAK_ACCEPT_CHUNK_CHANGE")
  hyak_info "HYAK_ACCEPT_CHUNK_CHANGE is set: a changed chunk setting will be accepted on the record (reason: $HYAK_ACCEPT_CHUNK_CHANGE)"
fi
while :; do
  if [ "$HYAK_TERM_RECEIVED" != 0 ]; then
    hyak_info "a stop signal arrived: no further segment in this job"
    break
  fi
  if [ -n "$MONITOR_PID" ] && ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    hyak_warn "the status monitor exited (see $MONITOR_LOG); training continues without live dashboard updates"
    MONITOR_PID=""
  fi
  q=0
  answer=$(python -m hyak.train.run_segment --run "$RUN" --should-start) || q=$?
  hyak_info "should-start (exit $q): $answer"
  if [[ $answer =~ $RE_NEWEST ]]; then newest=${BASH_REMATCH[1]}; else newest=""; fi
  if [[ $answer =~ $RE_REASON ]]; then reason=${BASH_REMATCH[1]}; else reason=""; fi
  if [ "$q" = 75 ]; then
    job_not_now "${reason:-not_now}" "no segment started: ${reason:-not now} (python -m hyak.train.run_segment --should-start answered 75: $answer)"
    break
  elif [ "$q" != 0 ]; then
    rc=$q
    job_explain "$q" "python -m hyak.train.run_segment --run $RUN --should-start"
    break
  fi
  if [ -z "$newest" ] || [ -z "$reason" ]; then
    rc=2
    JOB_MESSAGE="could not read newest_full_step/reason from run_segment --should-start: $answer"
    break
  fi
  if [ "$segments" -gt 0 ] && [ "$newest" = "$previous_newest" ]; then
    if [ "$reason" = audit_pending ]; then
      hyak_warn "the final audit is still pending; the next job (or python -m hyak.train.retention audit --run $RUN) finishes it"
      break
    fi
    rc=4                      # a check failed (1 is reserved for crashes)
    JOB_MESSAGE="training segment made no checkpoint progress (newest full restart still $newest after a segment that exited 0)"
    hyak_warn "$JOB_MESSAGE"
    break
  fi
  if [ "$HYAK_TERM_RECEIVED" != 0 ]; then
    hyak_info "a stop signal arrived: no further segment in this job"
    break
  fi
  previous_newest=$newest
  seg=0
  job_step python -m hyak.train.run_segment --run "$RUN" ${SEGMENT_ARGS[@]+"${SEGMENT_ARGS[@]}"} >>"$TRAIN_LOG" || seg=$?
  segments=$((segments + 1))
  if [ "$seg" != 0 ]; then
    rc=$seg
    job_explain "$seg" "training segment $segments of this job"
    [ -n "$JOB_MESSAGE" ] || JOB_MESSAGE="training segment $segments of this job ended with exit status $seg"
    JOB_MESSAGE="$JOB_MESSAGE; the trainer's traceback or message is in the Slurm log, its JSON lines in $TRAIN_LOG"
    break
  fi
done
hyak_info "$segments segment(s) ran in this job"
exit "$rc"
