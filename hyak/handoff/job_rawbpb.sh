# shellcheck shell=bash
# hyak/handoff/job_rawbpb.sh -- shared part of 12_smoke_rawbpb.sbatch and 13_train_rawbpb.sbatch.
#
# Source it after the common job prologue, job_begin, hyak_activate train.  It runs Qingyun's launcher
# scripts/run_raw_bpb_training.py as it is; nothing here changes his files.
#
#   rawbpb_setup                 RAWBPB_TRAIN_OUTPUT, RAWBPB_SMOKE_OUTPUT (python -m hyak.handoff.run outputs),
#                                RAWBPB_INPUTS=$ANYTOK_ROOT/input, RAWBPB_LAUNCHER; exports ANYTOK_RUN_PLATFORM
#                                (HYAK_PLATFORM of the stage) and ANYTOK_RUN_PROFILE (<cluster>/<partition>) for
#                                the launcher's status rows
#   rawbpb_requirements          the installed packages must equal docs/raw-bpb-handoff/requirements.txt
#                                (exit status 2 otherwise)
#   rawbpb_link OUTPUT           <monitor dir>/run -> OUTPUT (the dashboard reads OUTPUT/status.json and
#                                OUTPUT/events.jsonl through it; the registry stays repository-relative)
#   rawbpb_launch OUTPUT ARGS... one launcher run as a forwarded child (job_step: Slurm's early TERM reaches the
#                                batch shell, which forwards it to the launcher, which forwards it to the trainer
#                                for its graceful full save); stdout (the trainer's JSON lines and tracebacks) is
#                                appended to $TRAIN_LOG.  Sets RAWBPB_RC to the status in the port's contract: a
#                                trainer killed by signal N makes the launcher exit 256-N (SystemExit of a negative
#                                return code), which is mapped to 128+N; everything else is kept.
#   rawbpb_explain RC            JOB_MESSAGE for a failed launch

RAWBPB_LAUNCHER="$HYAK_REPO/scripts/run_raw_bpb_training.py"
RAWBPB_INPUTS="$ANYTOK_ROOT/input"
RAWBPB_TRAIN_OUTPUT=""
RAWBPB_SMOKE_OUTPUT=""
RAWBPB_RC=0

rawbpb_setup() {
  local line
  while IFS= read -r line; do
    case $line in
      train=*) RAWBPB_TRAIN_OUTPUT=${line#train=} ;;
      smoke=*) RAWBPB_SMOKE_OUTPUT=${line#smoke=} ;;
    esac
  done < <(python -m hyak.handoff.run outputs)
  [ -n "$RAWBPB_TRAIN_OUTPUT" ] && [ -n "$RAWBPB_SMOKE_OUTPUT" ] \
    || hyak_die "could not resolve the raw-BPB output directories (python -m hyak.handoff.run outputs)"
  [ "$RAWBPB_TRAIN_OUTPUT" != "$RAWBPB_SMOKE_OUTPUT" ] || hyak_die "the smoke and the run share $RAWBPB_TRAIN_OUTPUT"
  [ -f "$RAWBPB_LAUNCHER" ] || hyak_die "$RAWBPB_LAUNCHER is missing (this checkout lacks Qingyun's handoff commit 670993b)"
  ANYTOK_RUN_PLATFORM=$(hyak_stage_value HYAK_PLATFORM "$JOB_STAGE")
  ANYTOK_RUN_PROFILE=$(python -c 'from hyak.common import slurm; print(slurm.profile_label())' 2>/dev/null || echo local)
  export ANYTOK_RUN_PLATFORM ANYTOK_RUN_PROFILE
  hyak_info "raw-BPB: inputs $RAWBPB_INPUTS; run output $RAWBPB_TRAIN_OUTPUT; smoke output $RAWBPB_SMOKE_OUTPUT; platform '$ANYTOK_RUN_PLATFORM', profile '$ANYTOK_RUN_PROFILE'"
}

rawbpb_requirements() {
  local rc=0
  python -m hyak.handoff.run requirements >>"$TRAIN_LOG" || rc=$?
  if [ "$rc" != 0 ]; then
    JOB_MESSAGE="the train environment differs from docs/raw-bpb-handoff/requirements.txt (the job log lists the packages); remedy: bash hyak/env/setup_train_env.sh"
  fi
  return "$rc"
}

rawbpb_link() {
  ln -sfn "$1" "$JOB_DIR/run" || hyak_warn "could not link $JOB_DIR/run -> $1 (the dashboard entry shows no progress)"
}

rawbpb_launch() {
  local output=$1 rc=0
  shift
  job_step python -u "$RAWBPB_LAUNCHER" --inputs "$RAWBPB_INPUTS" --output "$output" "$@" >>"$TRAIN_LOG" || rc=$?
  if [ "$rc" -ge 193 ] && [ "$rc" -le 255 ]; then
    hyak_warn "the launcher exited $rc: the trainer was killed by signal $((256 - rc)); recorded as exit $((128 + 256 - rc))"
    rc=$((128 + 256 - rc))
  fi
  RAWBPB_RC=$rc
  return 0
}

rawbpb_explain() {
  job_explain "$1" "python scripts/run_raw_bpb_training.py" \
    "2=scripts/run_raw_bpb_training.py: a usage or input error (a missing input, or an output the trainer refuses); see the job log" \
    "137=the trainer was killed (signal 9): usually the memory limit of the job or the node; see the job log"
  [ -z "$JOB_MESSAGE" ] || JOB_MESSAGE="$JOB_MESSAGE; the trainer's JSON lines and traceback are in $TRAIN_LOG"
}
