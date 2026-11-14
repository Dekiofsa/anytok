# shellcheck shell=bash
# hyak/slurm/job.sh -- the common prologue of every hyak/slurm/*.sbatch script.
#
# Source it after lib.sh (the first lines of every job script do that; they are identical in all
# twelve files and checked by hyak/tests/test_integration_sbatch.py).  It defines the job's life cycle
# in one place, so a fix is made once:
#
#   job_begin STAGE [STAGE...]    the stages this script serves ($HYAK_STAGE; default the first one).
#                                 Keeps the shell alive on TERM/INT (only HYAK_TERM_RECEIVED=1 is
#                                 recorded, so no further step starts), removes the login shell's
#                                 PYTHONPATH/PYTHONHOME/PYTHONSTARTUP/PYTHONUSERBASE/LD_PRELOAD from the
#                                 job (names logged and recorded as an "environment_sanitized" event,
#                                 never the values), records the end of the job in status.json on every
#                                 exit path, loads hyak/config.env and creates the monitor directory.
#   job_step CMD [ARGS...]        run one program as a forwarded child (Slurm's TERM reaches it as
#                                 SIGTERM); returns its status; not started (143) once a stop signal arrived.
#   job_step_foreground CMD ...   run one program that must NOT be stopped by the early-warning signal
#                                 (a short finishing step such as the pool collect): not forwarded, and
#                                 started even after the signal.  The shell's TERM trap only records it.
#   job_stopping                  true once a stop signal has arrived (multi-step jobs then end with 0).
#   job_extra_args VAR            JOB_ARGS = the words of environment variable VAR.
#   job_explain RC TOOL [CODE=TEXT...]
#                                 JOB_MESSAGE for a failed program, from the project's exit-code contract
#                                 (below); CODE=TEXT pairs override the generic text of that code.
#   job_not_now REASON TEXT       record a "not now" (exit 75 of a tool): event + note, no failure.
#
# The project-wide exit-code contract of every Python tool whose status a job script reads:
#   0   success, or nothing left to do
#   1   never deliberate: an uncaught exception (the traceback is in the job log) -> status "failed";
#       a training chain is halted
#   2   usage, configuration, input or environment error (the log has a remedy line)
#   3   a protocol stop that needs the user's decision
#   4   a check failed (does not fit, equivalence failure, a bucket failed, inconsistent inputs, INIT FAILED)
#   75  not now / incomplete, try later: the job ends with 0, writes no HALTED.json, and the status says why
#   130/143  stopped by a signal that was asked for (INT/TERM): status "stopped" (or "paused" when the
#       program finished its graceful stop with 0)
#   137 or any other 128+n  the program was killed (out of memory, SIGKILL, a crash of the interpreter):
#       status "failed" with the text "killed"; a training chain is halted
#
# Variables a job script may read: JOB_STAGE, JOB_DIR (monitor directory of the stage), JOB_LOG (this
# job's Slurm output file; empty outside Slurm), JOB_ARGS.  A script sets JOB_MESSAGE (explanation added
# to the status error when the job fails), JOB_NOTE (added to the status of a job that ends with 0) and
# JOB_FINALIZE (the function that records the end of the job, CODE as $1).

JOB_STAGE=""                       # stage of this job (= $HYAK_STAGE)
JOB_DIR=""                         # <repo>/modal_results/hyak_tillicum/<stage>
JOB_LOG=""                         # this job's Slurm output file (empty outside Slurm)
JOB_MESSAGE=""                     # explanation added to the status error when the job fails
JOB_NOTE=""                        # explanation recorded when the job ends with 0 without finishing the stage
JOB_FINALIZE=job_finalize_status   # function that records the end of the job (CODE as $1)
JOB_ARGS=()                        # words of an extra-arguments variable (job_extra_args)

# job_begin STAGE [STAGE...]
job_begin() {
  local stage=${HYAK_STAGE:-$1} s ok=0
  for s in "$@"; do if [ "$s" = "$stage" ]; then ok=1; fi; done
  [ "$ok" = 1 ] || hyak_die "this script serves stage(s) '$*', not HYAK_STAGE='$stage'; submit it with hyak/slurm/submit.sh"
  export HYAK_STAGE="$stage"
  JOB_STAGE=$stage
  HYAK_TERM_RECEIVED=0                 # a new job has received no signal yet (ignore a stale value)
  hyak_trap_term
  trap job_on_exit EXIT
  hyak_sanitize_env
  hyak_load_config
  JOB_DIR=$(hyak_log_dir "$JOB_STAGE")
  if [ -n "${SLURM_JOB_ID:-}" ]; then JOB_LOG="$JOB_DIR/slurm-$SLURM_JOB_ID.out"; fi
  hyak_info "stage $JOB_STAGE: job ${SLURM_JOB_ID:-<none>} on $(hostname 2>/dev/null || echo '?'); repo $HYAK_REPO; config $HYAK_CONFIG_FILE"
  if [ -n "$HYAK_ENV_REMOVED" ]; then
    hyak_info "removed from the login environment (values not logged): $HYAK_ENV_REMOVED"
    hyak_status_event "$JOB_STAGE" environment_sanitized "removed=${HYAK_ENV_REMOVED// /,}"
  fi
}

# job_on_exit (EXIT trap): record the end of the job with the script's exit status, keep the status.
job_on_exit() {
  local code=$?
  trap - EXIT
  set +e
  if [ -n "$JOB_STAGE" ]; then "$JOB_FINALIZE" "$code"; fi
  hyak_info "stage ${JOB_STAGE:-?}: exit status $code"
  exit "$code"
}

# job_finalize_status CODE: python -m hyak.common.status finalize.  Exit 0 keeps a terminal state the stage
# wrote, else paused (JOB_NOTE says why when the stage did not finish); 130 and 143 are stops that were
# asked for; every other status, 137 and the other 128+n included (killed), is a failure with the log tail,
# so the status never says "stopped" for a program that was killed.  Never fails.
job_finalize_status() {
  local code=$1 status_code=$1 message=$JOB_MESSAGE
  local -a args
  case $code in
    0) message=$JOB_NOTE ;;
    130|143) ;;
    *) if [ "$code" -gt 128 ]; then
         status_code=1
         [ -n "$message" ] || message="killed by signal $((code - 128)) (exit status $code)"
       fi ;;
  esac
  args=(-m hyak.common.status finalize --stage "$JOB_STAGE" --exit-code "$status_code")
  if [ -n "$JOB_LOG" ]; then args[${#args[@]}]=--log; args[${#args[@]}]=$JOB_LOG; fi
  if [ -n "$message" ]; then args[${#args[@]}]=--message; args[${#args[@]}]=$message; fi
  PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$(hyak_python)" "${args[@]}" >/dev/null \
    || hyak_warn "could not record the end of stage $JOB_STAGE in its status.json"
}

# job_step CMD [ARGS...]: run one program as a forwarded child (Slurm's TERM reaches it as SIGTERM)
# and return its exit status.  Not started (status 143) once a stop signal has arrived.
job_step() {
  local rc=0
  if [ "$HYAK_TERM_RECEIVED" != 0 ]; then
    hyak_warn "a stop signal arrived earlier; not starting: $*"
    return 143
  fi
  hyak_info "start: $*"
  hyak_run_forwarding "$@" || rc=$?
  hyak_info "exit status $rc: $*"
  return "$rc"
}

# job_step_foreground CMD [ARGS...]: run one short finishing step that the early-warning signal must not
# interrupt (it is not forwarded: a step that writes a file once, such as the pool index, would be killed
# half way) and that still runs after the signal.  The batch shell only records a TERM while it waits.
job_step_foreground() {
  local rc=0
  hyak_info "start (not stopped by the early-warning signal): $*"
  "$@" || rc=$?
  hyak_info "exit status $rc: $*"
  return "$rc"
}

# job_stopping: true (with a message) once a stop signal has arrived.  Multi-step jobs then end with
# exit 0 instead of starting the next program (status "paused": resubmit to continue).
job_stopping() {
  [ "$HYAK_TERM_RECEIVED" != 0 ] || return 1
  hyak_info "a stop signal arrived: no further step in this job (resubmit to continue)"
  return 0
}

# job_extra_args VAR: JOB_ARGS = the words of environment variable VAR (split on blanks, no quoting,
# no globbing).  Set one with: hyak/slurm/submit.sh <stage> --env VAR="--flag value"
job_extra_args() {
  local value
  [[ $1 =~ ^[A-Z][A-Z0-9_]*$ ]] || hyak_die "job_extra_args: invalid variable name '$1'"
  value=${!1-}
  JOB_ARGS=()
  if [ -n "$value" ]; then
    read -r -a JOB_ARGS <<< "$value"
  fi
  if [ "${#JOB_ARGS[@]}" -gt 0 ]; then
    hyak_info "extra arguments from $1: ${JOB_ARGS[*]-}"
  fi
}

# job_explain RC TOOL [CODE=TEXT...]: set JOB_MESSAGE for the failed program TOOL (a description such as
# "python -m hyak.memory.probe") with exit status RC.  A CODE=TEXT pair for RC wins; the other codes get
# the text of the exit-code contract.  Statuses that are not failures (0, 75, 130, 143) leave it alone.
job_explain() {
  local rc=$1 tool=$2 item
  shift 2
  for item in "$@"; do
    if [ "${item%%=*}" = "$rc" ]; then JOB_MESSAGE=${item#*=}; return 0; fi
  done
  case $rc in
    0|75|130|143) ;;
    1) JOB_MESSAGE="$tool crashed: uncaught exception, traceback in the job log" ;;
    2) JOB_MESSAGE="$tool: usage, configuration, input or environment error; the job log has the remedy line" ;;
    3) JOB_MESSAGE="$tool stopped by the protocol and needs your decision; see the job log" ;;
    4) JOB_MESSAGE="$tool: a check failed; see the job log" ;;
    137) JOB_MESSAGE="$tool was killed (exit 137, signal 9): usually the memory limit of the job or the node; see the job log" ;;
    *) if [[ $rc =~ ^[0-9]+$ ]] && [ "$rc" -gt 128 ]; then
         JOB_MESSAGE="$tool was killed by signal $((rc - 128)) (exit $rc); see the job log"
       else
         JOB_MESSAGE="$tool failed with exit status $rc; see the job log"
       fi ;;
  esac
}

# job_not_now REASON TEXT: a tool answered "not now" (exit 75).  The job goes on to end with 0 (the caller
# breaks out of its loop or exits 0); no HALTED.json.  Records an event and the note that finalize puts in
# the status, and logs the reason.
job_not_now() {
  JOB_NOTE=$2
  hyak_info "not now ($1): $2"
  hyak_status_event "$JOB_STAGE" not_now "reason=$1" "note=$2"
}
