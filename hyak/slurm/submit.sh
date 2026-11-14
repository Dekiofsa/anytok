#!/usr/bin/env bash
# hyak/slurm/submit.sh -- submit one pipeline stage to Slurm with resources from hyak/config.env.
#
# Usage:
#   hyak/slurm/submit.sh <stage> [--count N] [--debug] [--dry-run] [--run uniform|rl]
#                        [--after JOBID[:JOBID...]] [--env KEY=VALUE]... [--allow-duplicate]
#                        [-- <extra sbatch arguments>]
#
#   <stage>            check-env probe corpus wheel pool stage-inputs init train-uniform
#                      train-rl eval-prepare eval-uniform eval-rl replay-check smoke-train
#                      smoke-rawbpb train-rawbpb   (the raw-BPB handoff run, README section 0)
#   --count N         submit N identical jobs (train-*: N chained 24 h jobs through
#                      --dependency=singleton; pool/eval-*: N parallel claim-based workers).
#                      Other stages are one-shot and accept only N=1.
#   --debug            use HYAK_QOS_DEBUG and HYAK_TIME_DEBUG (overrides per-stage QoS/time); refused for
#                      train-uniform/train-rl (a training job needs margin + 20 minutes)
#   --dry-run          print the sbatch command; submit nothing, create nothing
#   --run R            export HYAK_RUN=R for stages without a fixed run (smoke-train, ...)
#   --after IDS        add afterok:IDS to the dependency list (combined with singleton) and
#                      --kill-on-invalid-dep=yes (a failed predecessor cancels the job instead of leaving it pending)
#   --env KEY=VALUE    set KEY=VALUE for this submission: the job gets it, and a config key
#                      (HYAK_TIME_SMOKE_TRAIN=02:00:00, HYAK_POOL_ARGS=...) takes effect here as well.
#                      Secrets (HF_TOKEN ...) are refused: put them in hyak/config.env.
#   --allow-duplicate  submit a one-shot stage although a job of that name is queued/running
#   -- ARGS            passed to sbatch verbatim.  Not allowed there: --dependency (use --after) and the
#                      resource flags that submit.sh computes (--time, --signal, --partition, --qos,
#                      --account, --gres, --mem, --cpus-per-task ...): change them in hyak/config.env or
#                      with  --env HYAK_<FLAG>_<STAGE>=VALUE  (for example HYAK_TIME_SMOKE_TRAIN=00:40:00)
#
# Run this script directly on the login node (not through sbatch): the repository root is
# resolved from this file's location and passed to the job as HYAK_REPO.  Every resource
# comes from the config (hyak/common/slurm.py sbatch_resources mirrors the logic; tested):
#   --partition --qos [--account] --gres --cpus-per-task [--mem] --time  (HYAK_<FLAG>, per-stage
#   HYAK_<FLAG>_<STAGE> when non-empty; "none" or an empty value omits a flag; check-env and probe default to
#   the debug QoS/time; the default HYAK_MEM is empty: no --mem, the cluster's per-GPU default of
#   195.3125 GB applies), --job-name=anytok-<stage>, --output=<monitor dir>/slurm-%j.out,
#   --chdir=<repo>, --signal=B:TERM@<min(margin, time/4, 65535) s> (omitted when the margin is 0),
#   --export=ALL, [--kill-on-invalid-dep=yes with --after] and --dependency=singleton for train-uniform
#   and train-rl.
# The job's variables (HYAK_REPO, HYAK_STAGE, HYAK_RUN, HYAK_CONFIG and the --env pairs) are exported by this
# script itself and reach the job through plain --export=ALL.  Any other --export value would also make
# slurmd run the user's login profile on the compute node before the job starts (--get-user-env), and
# would put the --env values on the recorded command line.
# A time limit passed with --env or in the config also decides --signal (it is computed after the --env
# pairs are applied).  Train stages need a limit of at least margin + 20 minutes and a margin of at
# least 1 minute; train-rl (and smoke-train --run rl) is refused until $ANYTOK_ROOT/input/pairing.json
# exists, i.e. until the uniform bootstrap segment froze the pair.  smoke-rawbpb and train-rawbpb are refused
# until hyak/handoff/fetch_inputs.sh has verified the input bundle ($ANYTOK_ROOT/handoff/inputs-verified.json);
# smoke-rawbpb uses the debug QoS and time by default (like check-env), train-rawbpb is a singleton chain.
#
# Works with bash 3.2 (macOS) for --dry-run tests.
set -euo pipefail

# --- stage table: stage -> sbatch file (hyak/slurm/) and fixed run -------------------------------
# Must equal hyak.common.paths.SBATCH_FILES / STAGE_RUN (tested).
_stage_file() {
  case $1 in
    check-env) echo 00_check_env.sbatch ;;
    probe) echo 01_probe.sbatch ;;
    corpus) echo 02_corpus.sbatch ;;
    wheel) echo 03_wheel.sbatch ;;
    pool) echo 04_pool.sbatch ;;
    stage-inputs) echo 05_stage_inputs.sbatch ;;
    init) echo 06_init.sbatch ;;
    train-uniform|train-rl) echo 07_train.sbatch ;;
    eval-prepare) echo 08_eval_prepare.sbatch ;;
    eval-uniform|eval-rl) echo 09_eval_score.sbatch ;;
    replay-check) echo 10_replay_check.sbatch ;;
    smoke-train) echo 11_smoke_train.sbatch ;;
    smoke-rawbpb) echo 12_smoke_rawbpb.sbatch ;;
    train-rawbpb) echo 13_train_rawbpb.sbatch ;;
    *) return 1 ;;
  esac
}
_stage_run() {
  case $1 in
    train-uniform|eval-uniform) echo uniform ;;
    train-rl|eval-rl) echo rl ;;
    *) echo "" ;;
  esac
}
SINGLETON_STAGES="train-uniform train-rl train-rawbpb"
MULTI_JOB_STAGES="pool eval-prepare eval-uniform eval-rl train-uniform train-rl train-rawbpb"
DEBUG_STAGES="check-env probe smoke-rawbpb"
RE_ENV='^[A-Za-z_][A-Za-z0-9_]*='
RE_COUNT='^[0-9]+$'
RE_AFTER='^[0-9]+(_[0-9]+)?(:[0-9]+(_[0-9]+)?)*$'
RE_GPU_GRES='^gpu(:[A-Za-z0-9_]+)?:([0-9]+)$'
RE_MEM='^([0-9]+)([KMGT]?)$'
#: Names that never go through --env (they would be echoed and recorded): use hyak/config.env.
RE_SECRET_NAME='(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY)'

usage() {
  awk '/^# Usage:/ { on = 1 } /^# Run this script directly/ { exit } on { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

REPO=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${HYAK_REPO:-}" ] && [ "$(CDPATH='' cd -P -- "$HYAK_REPO" 2>/dev/null && pwd -P)" != "$REPO" ]; then
  echo "[submit] WARNING: ignoring HYAK_REPO=$HYAK_REPO; using this script's checkout $REPO" >&2
fi
HYAK_REPO=$REPO
export HYAK_REPO
# shellcheck source=hyak/slurm/lib.sh
source "$REPO/hyak/slurm/lib.sh"

# --- arguments -------------------------------------------------------------------------------------
[ $# -ge 1 ] || { usage >&2; exit 2; }
case $1 in -h|--help) usage; exit 0 ;; esac
STAGE=$1
shift
COUNT=1
DEBUG=0
DRY_RUN=0
RUN=""
AFTER=""
ALLOW_DUP=0
ENV_NAMES=()
ENV_VALUES=()
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --count) [ $# -ge 2 ] || hyak_die "--count needs a value"; COUNT=$2; shift 2 ;;
    --count=*) COUNT=${1#--count=}; shift ;;
    --debug) DEBUG=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --run) [ $# -ge 2 ] || hyak_die "--run needs a value"; RUN=$2; shift 2 ;;
    --run=*) RUN=${1#--run=}; shift ;;
    --after) [ $# -ge 2 ] || hyak_die "--after needs a value"; AFTER=$2; shift 2 ;;
    --after=*) AFTER=${1#--after=}; shift ;;
    --env)
      [ $# -ge 2 ] || hyak_die "--env needs KEY=VALUE"
      [[ $2 =~ $RE_ENV ]] || hyak_die "--env expects KEY=VALUE (the argument is not shown: it could hold a secret)"
      ENV_NAMES[${#ENV_NAMES[@]}]=${2%%=*}
      ENV_VALUES[${#ENV_VALUES[@]}]=${2#*=}
      shift 2 ;;
    --allow-duplicate) ALLOW_DUP=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) hyak_die "Unknown option '$1' (see --help; pass sbatch options after --)" ;;
  esac
done

case " $HYAK_STAGES " in
  *" $STAGE "*) ;;
  *) case $STAGE in
       eval-score) hyak_die "Stage 'eval-score' is submitted as eval-uniform or eval-rl" ;;
       train) hyak_die "Stage 'train' is submitted as train-uniform or train-rl" ;;
     esac
     hyak_die "Unknown stage '$STAGE'; expected one of: $HYAK_STAGES" ;;
esac
SBATCH_FILE_NAME=$(_stage_file "$STAGE")
SBATCH_FILE="$REPO/hyak/slurm/$SBATCH_FILE_NAME"
FIXED_RUN=$(_stage_run "$STAGE")

[[ $COUNT =~ $RE_COUNT ]] && [ "$COUNT" -ge 1 ] || hyak_die "--count must be a positive integer, got '$COUNT'"
if [ "$COUNT" -gt 1 ] && ! _hyak_in_list "$STAGE" "$MULTI_JOB_STAGES"; then
  hyak_die "Stage $STAGE is one-shot; --count $COUNT is not allowed (multi-job stages: $MULTI_JOB_STAGES)"
fi
if [ -n "$RUN" ]; then
  case $RUN in uniform|rl) ;; *) hyak_die "--run must be uniform or rl, got '$RUN'" ;; esac
  if [ -n "$FIXED_RUN" ] && [ "$RUN" != "$FIXED_RUN" ]; then
    hyak_die "Stage $STAGE always runs '$FIXED_RUN'; --run $RUN contradicts it"
  fi
fi
[ -n "$RUN" ] || RUN=$FIXED_RUN
if [ -n "$AFTER" ]; then
  [[ $AFTER =~ $RE_AFTER ]] || hyak_die "--after expects JOBID[:JOBID...], got '$AFTER'"
fi
i=0
while [ "$i" -lt "${#EXTRA[@]}" ]; do
  case ${EXTRA[$i]} in
    -d|-d*|--dependency|--dependency=*) hyak_die "Do not pass ${EXTRA[$i]} after --: use --after JOBID (it is combined with singleton for training stages)" ;;
    --export|--export=*|--job-name|--job-name=*|-J|-J*|--output|--output=*|-o|-o*|--chdir|--chdir=*|-D|-D*)
      hyak_die "${EXTRA[$i]} is set by submit.sh and must not be overridden" ;;
    -t|-t?*|--time|--time=*|--signal|--signal=*|-p|-p?*|--partition|--partition=*|-q|-q?*|--qos|--qos=*|\
    -A|-A?*|--account|--account=*|--gres|--gres=*|-G|-G?*|--gpus|--gpus=*|--mem|--mem=*|--mem-per-cpu|--mem-per-cpu=*|\
    --mem-per-gpu|--mem-per-gpu=*|-c|-c?*|--cpus-per-task|--cpus-per-task=*|--cpus-per-gpu|--cpus-per-gpu=*)
      hyak_die "${EXTRA[$i]} would change a resource that submit.sh computes (and --signal with it). Set it in hyak/config.env or per submission with  --env HYAK_<FLAG>_<STAGE>=VALUE  (FLAG: PARTITION QOS ACCOUNT GRES CPUS MEM TIME SEGMENT_MARGIN_MINUTES), for example: submit.sh smoke-train --run uniform --env HYAK_TIME_SMOKE_TRAIN=00:40:00 --env HYAK_SEGMENT_MARGIN_MINUTES_SMOKE_TRAIN=5" ;;
    --kill-on-invalid-dep|--kill-on-invalid-dep=*)
      hyak_die "${EXTRA[$i]} is set by submit.sh when --after is used" ;;
  esac
  i=$((i + 1))
done

# --- the job's environment ------------------------------------------------------------------------------
# Exported here, before the configuration is read: the job receives it through --export=ALL, and a config
# key given with --env (HYAK_TIME_SMOKE_TRAIN, HYAK_SEGMENT_MARGIN_MINUTES, ...) decides the resources of
# this submission too (the environment overrides the config file).
if [ -n "${HYAK_CONFIG:-}" ]; then
  case $HYAK_CONFIG in /*) ;; *) HYAK_CONFIG="$(pwd -P)/$HYAK_CONFIG" ;; esac
  export HYAK_CONFIG
fi
export HYAK_STAGE="$STAGE"
if [ -n "$RUN" ]; then
  export HYAK_RUN="$RUN"
else
  unset HYAK_RUN
fi
i=0
while [ "$i" -lt "${#ENV_NAMES[@]}" ]; do
  case ${ENV_NAMES[$i]} in
    HYAK_REPO|HYAK_STAGE|HYAK_RUN|HYAK_CONFIG) hyak_die "--env ${ENV_NAMES[$i]} is set by submit.sh" ;;
  esac
  if [[ ${ENV_NAMES[$i]} =~ $RE_SECRET_NAME ]]; then
    hyak_die "--env ${ENV_NAMES[$i]}: secrets are not passed on the command line (it would be echoed and recorded). Put it in hyak/config.env (HF_TOKEN=...); jobs read it from there and never print it"
  fi
  export "${ENV_NAMES[$i]}=${ENV_VALUES[$i]}"
  i=$((i + 1))
done

# --- configuration and resources -------------------------------------------------------------------
hyak_load_config --no-export

SUFFIX=$(hyak_stage_suffix "$STAGE")
_override() { _hyak_get "HYAK_${1}_${SUFFIX}"; }
_omit() { case $1 in ''|none|NONE|None) return 0 ;; esac; return 1; }

USE_DEBUG=$DEBUG
_hyak_in_list "$STAGE" "$DEBUG_STAGES" && USE_DEBUG=1
_pick() {  # _pick FLAG [DEBUG_KEY]
  local flag=$1 debug_key=${2:-} value
  value=$(_override "$flag")
  if [ "$DEBUG" = 1 ] && [ -n "$debug_key" ]; then value=$(_hyak_get "$debug_key")
  elif [ -n "$value" ]; then :
  elif [ "$USE_DEBUG" = 1 ] && [ -n "$debug_key" ]; then value=$(_hyak_get "$debug_key")
  else value=$(_hyak_get "HYAK_$flag")
  fi
  printf '%s\n' "$value"
}
PARTITION=$(_pick PARTITION)
QOS=$(_pick QOS HYAK_QOS_DEBUG)
ACCOUNT=$(_pick ACCOUNT)
GRES=$(_pick GRES)
CPUS=$(_pick CPUS)
MEM=$(_pick MEM)
TIME_LIMIT=$(_pick TIME HYAK_TIME_DEBUG)
TIME_SECONDS=$(hyak_parse_time "$TIME_LIMIT") || hyak_die "Time limit '$TIME_LIMIT' is not a Slurm time"
MARGIN=$(hyak_stage_value HYAK_SEGMENT_MARGIN_MINUTES "$STAGE")
MARGIN=$((10#${MARGIN:-0}))          # decimal: bash would read 08 or 045 as octal
SIGNAL=0
if [ "$MARGIN" -gt 0 ]; then
  SIGNAL=$((MARGIN * 60))
  [ "$SIGNAL" -le $((TIME_SECONDS / 4)) ] || SIGNAL=$((TIME_SECONDS / 4))
  [ "$SIGNAL" -le 65535 ] || SIGNAL=65535
fi
MONITOR_DIR=$(hyak_monitor_dir "$STAGE")
case $MONITOR_DIR in *,*) hyak_die "Monitor directory $MONITOR_DIR contains a comma" ;; esac
if [ -n "${HYAK_MONITOR_ROOT:-}" ]; then
  hyak_warn "HYAK_MONITOR_ROOT is set: job output goes to $MONITOR_DIR, which the dashboard does not read"
fi

# --- checks that refuse a submission that cannot work -----------------------------------------------------
# A training segment starts only while at least 20 minutes plus the margin remain, its in-process stop
# timer fires one margin before the time limit, and a margin of 0 leaves the job to be killed at the limit
# with up to 2,000 updates lost.  Such a job would end "paused" without training (train) or fail (smoke).
case $STAGE in
  train-uniform|train-rl|smoke-train|smoke-rawbpb|train-rawbpb)
    [ "$MARGIN" -ge 1 ] || hyak_die "$STAGE: HYAK_SEGMENT_MARGIN_MINUTES is 0, so the job would not be asked to stop and Slurm would kill it at the time limit in the middle of an update or a save. Remedy: set HYAK_SEGMENT_MARGIN_MINUTES (or HYAK_SEGMENT_MARGIN_MINUTES_$SUFFIX) to at least 1; the default is 45 for training and 10 for smoke-train"
    NEED_SECONDS=$(( (MARGIN + 20) * 60 ))
    if [ "$TIME_SECONDS" -lt "$NEED_SECONDS" ]; then
      hyak_die "$STAGE: the time limit $TIME_LIMIT is shorter than the margin ($MARGIN min) + 20 min = $((NEED_SECONDS / 60)) min, so the job would start no segment and end 'paused' after a minute$([ "$DEBUG" = 1 ] && echo ' (--debug is for the short stages, not for training)'). Remedy: raise HYAK_TIME_$SUFFIX to at least $((NEED_SECONDS / 3600)):$(printf '%02d' $(((NEED_SECONDS % 3600) / 60))):00, or lower HYAK_SEGMENT_MARGIN_MINUTES_$SUFFIX"
    fi ;;
esac
# The RL run (and its smoke test) draws its tokenizers from the pair the uniform bootstrap freezes; an RL
# job that starts before input/pairing.json exists is refused by the trainer and halts the chain.
if [ "$STAGE" = train-rl ] || { [ "$STAGE" = smoke-train ] && [ "$RUN" = rl ]; }; then
  if [ ! -e "$ANYTOK_ROOT/input/pairing.json" ]; then
    PAIRING_MESSAGE="$STAGE needs the uniform pairing first: $ANYTOK_ROOT/input/pairing.json has not been written yet. The uniform bootstrap segment (train-uniform, its first update) freezes it; submit this stage after the pairing_frozen event appears in $REPO/$HYAK_MONITOR_REL/train-uniform/events.jsonl"
    if [ "$DRY_RUN" = 1 ]; then
      hyak_warn "$PAIRING_MESSAGE (a real submission is refused until then)"
    else
      hyak_die "$PAIRING_MESSAGE"
    fi
  fi
fi

# The raw-BPB handoff stages train on the verified input bundle only (hyak/handoff/fetch_inputs.sh).
case $STAGE in
  smoke-rawbpb|train-rawbpb)
    if [ ! -e "$ANYTOK_ROOT/handoff/inputs-verified.json" ]; then
      BUNDLE_MESSAGE="$STAGE needs the verified input bundle: $ANYTOK_ROOT/handoff/inputs-verified.json has not been written yet. Remedy: on the login node run  bash hyak/handoff/fetch_inputs.sh  (download, check, extract into $ANYTOK_ROOT/input, verify every file)"
      if [ "$DRY_RUN" = 1 ]; then
        hyak_warn "$BUNDLE_MESSAGE (a real submission is refused until then)"
      else
        hyak_die "$BUNDLE_MESSAGE"
      fi
    fi ;;
esac

# --- Tillicum limits (warnings only; other clusters differ) ------------------------------------------------
# _mem_mib VALUE: Slurm memory (N, NK, NM, NG, NT; a bare number is MiB) in MiB, "" when not a number.
_mem_mib() {
  [[ ${1:-} =~ $RE_MEM ]] || return 0
  local n=$((10#${BASH_REMATCH[1]}))
  case ${BASH_REMATCH[2]} in
    K) echo $(( (n + 1023) / 1024 )) ;;
    G) echo $(( n * 1024 )) ;;
    T) echo $(( n * 1048576 )) ;;
    *) echo "$n" ;;
  esac
}
case $PARTITION in
  gpu-h200*)
    GPUS=0
    if ! _omit "$GRES" && [[ $GRES =~ $RE_GPU_GRES ]]; then GPUS=$((10#${BASH_REMATCH[2]})); fi
    if [ "$GPUS" -lt 1 ]; then
      hyak_warn "partition $PARTITION has no GPU in this request (--gres=${GRES:-none}); Tillicum starts no CPU-only jobs, so it will be rejected. Set HYAK_GRES=gpu:1"
      GPUS=1
    fi
    MEM_MIB=""
    _omit "$MEM" || MEM_MIB=$(_mem_mib "$MEM")
    case $PARTITION in
      *mig*)
        # A MIG unit has 1 CPU and 30 GB.  The pool stage runs HYAK_POOL_WORKERS x HYAK_POOL_THREADS threads.
        if ! _omit "$CPUS" && [ "$CPUS" -gt "$GPUS" ]; then
          hyak_warn "a MIG unit has 1 CPU: --cpus-per-task=$CPUS with $GPUS unit(s) will be rejected"
        fi
        if [ -n "$MEM_MIB" ] && [ "$MEM_MIB" -gt $((30 * 1024 * GPUS)) ]; then
          hyak_warn "a MIG unit has 30 GB: --mem=$MEM with $GPUS unit(s) will be rejected"
        fi
        if [ "$STAGE" = pool ]; then
          hyak_warn "pool on $PARTITION: $((HYAK_POOL_WORKERS)) workers x $((HYAK_POOL_THREADS)) threads need $((HYAK_POOL_WORKERS * HYAK_POOL_THREADS)) CPUs, i.e. $((HYAK_POOL_WORKERS * HYAK_POOL_THREADS)) MIG units (8 units cost more than one full GPU); on 1 unit the tokenizers would run all threads on 1 CPU with 30 GB. Use the full GPU partition for the pool"
        fi ;;
      *)
        if ! _omit "$CPUS" && [ "$CPUS" -gt $((8 * GPUS)) ]; then
          hyak_warn "Tillicum allows at most 8 CPUs per GPU; --cpus-per-task=$CPUS with $GPUS GPU(s) will be rejected"
        fi
        # DefMemPerGPU: 200,000 MiB = 195.3125 GiB
        if [ -n "$MEM_MIB" ] && [ "$MEM_MIB" -gt $((200000 * GPUS)) ]; then
          hyak_warn "Tillicum allows 195.3125 GB (200,000 MiB) of memory per GPU; --mem=$MEM with $GPUS GPU(s) will be rejected"
        fi ;;
    esac
    case $QOS in
      debug)
        if [ "$TIME_SECONDS" -gt 3600 ]; then
          hyak_warn "the debug QoS allows 1 h; --time=$TIME_LIMIT will be rejected (or lower HYAK_TIME_$SUFFIX / use a normal job)"
        fi
        if [ "$GPUS" -gt 1 ]; then
          hyak_warn "the debug QoS allows 1 GPU; --gres=$GRES will be rejected"
        fi ;;
      normal)
        if [ "$TIME_SECONDS" -gt 86400 ]; then
          hyak_warn "the normal QoS allows 24 h; --time=$TIME_LIMIT will be rejected"
        fi ;;
    esac ;;
esac
# A misspelled per-stage variable in the environment (HYAK_TIME_SMOKETRAIN) is ignored without a word otherwise.
for _name in ${!HYAK_@}; do
  _hyak_in_list "$_name" "$_HYAK_CONFIG_KEYS" && continue
  for _flag in $HYAK_STAGE_OVERRIDE_FLAGS; do
    case $_name in
      HYAK_"$_flag"_?*)
        hyak_warn "$_name is not a per-stage setting of any stage and is ignored (stage suffixes: $(for _s in $HYAK_STAGES; do printf '%s ' "$(printf '%s' "$_s" | tr 'a-z-' 'A-Z_')"; done))" ;;
    esac
  done
done

# --- the sbatch command ------------------------------------------------------------------------------
ARGS=(--parsable)
_omit "$PARTITION" || ARGS[${#ARGS[@]}]="--partition=$PARTITION"
_omit "$QOS" || ARGS[${#ARGS[@]}]="--qos=$QOS"
_omit "$ACCOUNT" || ARGS[${#ARGS[@]}]="--account=$ACCOUNT"
_omit "$GRES" || ARGS[${#ARGS[@]}]="--gres=$GRES"
_omit "$CPUS" || ARGS[${#ARGS[@]}]="--cpus-per-task=$CPUS"
_omit "$MEM" || ARGS[${#ARGS[@]}]="--mem=$MEM"
ARGS[${#ARGS[@]}]="--time=$TIME_LIMIT"
ARGS[${#ARGS[@]}]="--job-name=anytok-$STAGE"
ARGS[${#ARGS[@]}]="--output=$MONITOR_DIR/slurm-%j.out"
ARGS[${#ARGS[@]}]="--chdir=$REPO"
[ "$SIGNAL" -le 0 ] || ARGS[${#ARGS[@]}]="--signal=B:TERM@$SIGNAL"
DEPS=""
_hyak_in_list "$STAGE" "$SINGLETON_STAGES" && DEPS="singleton"
if [ -n "$AFTER" ]; then
  DEPS="${DEPS:+$DEPS,}afterok:$AFTER"
  # a failed predecessor cancels the job; without this it stays pending for ever (DependencyNeverSatisfied)
  ARGS[${#ARGS[@]}]="--kill-on-invalid-dep=yes"
fi
[ -z "$DEPS" ] || ARGS[${#ARGS[@]}]="--dependency=$DEPS"

# Plain --export=ALL: the job gets this script's environment, in which HYAK_REPO, HYAK_STAGE, HYAK_RUN,
# HYAK_CONFIG and the --env pairs were exported above.  Any --export=ALL,KEY=VALUE list would enable
# sbatch's --get-user-env (the user's login profile runs on the compute node before the job; a failure
# or time-out leaves the job held) and would put the --env values on the command line.
ARGS[${#ARGS[@]}]="--export=ALL"
ARGS=("${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"})

if [ ! -f "$SBATCH_FILE" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    hyak_warn "$SBATCH_FILE does not exist yet (dry run continues)"
  else
    hyak_die "$SBATCH_FILE does not exist"
  fi
fi

# Full validation by hyak.common.config (also in --dry-run; needs any Python >= 3.9).
PY=$(hyak_python)
if command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  # python -m hyak.common.config --check exits 0 (valid) or 2 (a configuration error, message on stderr);
  # anything else (1: crashed, 126/127: the interpreter could not run it) is the checker's own failure.
  CHECK_RC=0
  CHECK_OUT=$(PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 "$PY" -m hyak.common.config --check 2>&1) || CHECK_RC=$?
  if [ "$CHECK_RC" = 2 ]; then
    hyak_die "Configuration check failed: $CHECK_OUT"
  elif [ "$CHECK_RC" != 0 ]; then
    hyak_die "Configuration check failed (python -m hyak.common.config --check exited $CHECK_RC; 2 is a configuration error, so the checker itself broke): $CHECK_OUT"
  fi
else
  hyak_warn "no Python >= 3.9 found (set HYAK_PYTHON); skipped the full configuration check"
fi

echo "# stage=$STAGE run=${RUN:-none} count=$COUNT debug=$USE_DEBUG time_seconds=$TIME_SECONDS signal_seconds=$SIGNAL config=$HYAK_CONFIG_FILE"
printf '# job environment (exported here, passed by --export=ALL): HYAK_REPO=%s HYAK_STAGE=%s' "$REPO" "$STAGE"
[ -z "$RUN" ] || printf ' HYAK_RUN=%s' "$RUN"
[ -z "${HYAK_CONFIG:-}" ] || printf ' HYAK_CONFIG=%s' "$HYAK_CONFIG"
i=0
while [ "$i" -lt "${#ENV_NAMES[@]}" ]; do
  printf ' %s=%s' "${ENV_NAMES[$i]}" "${ENV_VALUES[$i]}"
  i=$((i + 1))
done
printf '\n'
printf '%q ' sbatch "${ARGS[@]}" "$SBATCH_FILE"
printf '\n'

if [ "$DRY_RUN" = 1 ]; then
  echo "# dry run: nothing submitted"
  exit 0
fi

# --- submission ---------------------------------------------------------------------------------------
command -v sbatch >/dev/null 2>&1 || hyak_die "sbatch not found; run this on a Slurm login node"
if ! _hyak_in_list "$STAGE" "$MULTI_JOB_STAGES" && [ "$ALLOW_DUP" = 0 ] && command -v squeue >/dev/null 2>&1; then
  QUEUED=$(squeue -h -u "${USER:-$(id -un)}" -n "anytok-$STAGE" -o %i 2>/dev/null || true)
  if [ -n "$QUEUED" ]; then
    hyak_die "A job named anytok-$STAGE is already queued or running ($(echo $QUEUED)); wait for it or pass --allow-duplicate"
  fi
fi
mkdir -p "$MONITOR_DIR" || hyak_die "Cannot create $MONITOR_DIR"
n=1
while [ "$n" -le "$COUNT" ]; do
  OUT=$(sbatch "${ARGS[@]}" "$SBATCH_FILE") || hyak_die "sbatch failed for job $n of $COUNT"
  echo "submitted anytok-$STAGE job ${OUT%%;*} ($n/$COUNT)"
  n=$((n + 1))
done
