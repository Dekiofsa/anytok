# shellcheck shell=bash
# hyak/slurm/lib.sh -- shared shell functions for the Hyak sbatch scripts and submit.sh.
#
# Source it; do not execute it:
#     set -euo pipefail
#     source "$HYAK_REPO/hyak/slurm/lib.sh"
#     hyak_load_config                     # reads hyak/config.env, exports every key
#     hyak_activate train                  # or: data
#     log_dir=$(hyak_log_dir pool)
#     rc=0; hyak_run_forwarding python -m hyak.pool.worker || rc=$?
#     hyak_finalize_status pool "$rc"
#     exit "$rc"
#
# The job life cycle (job_begin, job_step, the exit-code contract ...) lives in hyak/slurm/job.sh, which
# the sbatch scripts source after this file.
#
# Works with bash >= 3.2 and with "set -euo pipefail".  In a batch job "$0" is
# Slurm's spool copy of the script, so nothing here derives paths from "$0":
# HYAK_REPO (exported by submit.sh) is the repository root.  When this file is
# sourced with HYAK_REPO unset, HYAK_REPO is set from this file's own location
# (convenient for login-node scripts).
#
# Functions (all fail loudly through hyak_die unless noted):
#   hyak_info/hyak_warn/hyak_die MSG     timestamped messages on stderr (die exits 1)
#   hyak_load_config [--no-export]       defaults (config.env.example) < config file < environment;
#                                        validates; exports every key unless --no-export
#   hyak_print_config                    resolved KEY=VALUE lines, sorted, HF_TOKEN redacted
#   hyak_stage_value KEY STAGE           per-stage override HYAK_<FLAG>_<STAGE> if non-empty, else KEY
#   hyak_stage_suffix STAGE              train-uniform -> TRAIN_UNIFORM
#   hyak_valid_stage STAGE               dies unless STAGE is in $HYAK_STAGES
#   hyak_parse_time TIME                 prints seconds of a Slurm time ("inf" for UNLIMITED); returns 1 if invalid
#   hyak_monitor_dir STAGE               prints <repo>/modal_results/hyak_tillicum/<STAGE> (no side effects)
#   hyak_log_dir STAGE                   creates it, points latest-slurm.out at slurm-$SLURM_JOB_ID.out, prints it
#   hyak_modules                         module load $HYAK_MODULES (no-op when empty)
#   hyak_gcc_module                      prints the gcc module a C/C++ build needs (HYAK_GCC_MODULE, else gcc/13.4.0) and
#                                        returns 0 when HYAK_GCC_MODULE is set or a C or C++ compiler is missing; else 1
#   hyak_load_modules WHERE MODULE...    module load these modules; a failure names WHERE (e.g. "--module")
#   hyak_sanitize_env                    unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE LD_PRELOAD,
#                                        export PYTHONNOUSERSITE=1; HYAK_ENV_REMOVED lists what was set
#   hyak_activate train|data             modules + conda activate $ANYTOK_ROOT/env/<name> + exports below
#   hyak_trap_term                       install flag-only TERM/INT traps (sets HYAK_TERM_RECEIVED=1)
#   hyak_run_forwarding CMD [ARGS...]    run CMD as a child, forward TERM/INT to it as SIGTERM,
#                                        wait until it has really exited, return its exit code
#   hyak_python                          prints the python to use for hyak.common helpers
#   hyak_status_start STAGE [TASK]       python -m hyak.common.status start   (never fails the caller)
#   hyak_status_event STAGE NAME [K=V..] python -m hyak.common.status event   (never fails the caller)
#   hyak_finalize_status STAGE RC [LOG [MESSAGE]]
#                                        python -m hyak.common.status finalize (never fails the caller)
#   hyak_register_dashboard              python -m hyak.tools.register_dashboard, given up after
#                                        $HYAK_REGISTER_TIMEOUT_SECONDS (60) with a warning (never fails the caller)
#   hyak_isolate_pip                     unset every PIP_* variable, PIP_CONFIG_FILE=/dev/null (setup scripts)
#   hyak_private_tmp DIR                 TMPDIR=DIR (created; hyak_private_tmp_cleanup removes it) (setup scripts)
#
# hyak_activate exports: LD_LIBRARY_PATH=<prefix>/hyak-libpython[:...] (a directory that holds only the
# link to libpython3.12.so.1.0 for librust_utils.so; conda's own lib directory is never put on the
# path, and an entry of an earlier activation under $ANYTOK_ROOT/env is replaced, never stacked),
# PYTHONPATH=$HYAK_REPO, PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1, PYTHONNOUSERSITE=1,
# HF_HOME=$ANYTOK_ROOT/hf-cache, HYAK_ENV=<name>, HF_HUB_OFFLINE=1 for the stages that never need the
# network (train-uniform, train-rl, smoke-train, eval-*, probe, init, replay-check; the corpus stage
# unsets it), and
# TOKENIZERS_PARALLELISM as the original Modal images had it for the stage in $HYAK_STAGE: "false" for
# eval-prepare/eval-uniform/eval-rl, "true" for pool/wheel, unset otherwise (the training images set nothing).
# It never touches CUDA_VISIBLE_DEVICES (Slurm sets it).
#
# hyak_run_forwarding: Slurm's --signal=B:TERM@... reaches only this batch shell.  Bash returns
# from "wait" as soon as a trapped signal arrives, so the function loops until the child is
# really gone and then returns the child's own status.  A signal that arrived while no child ran
# (for example while job_step printed its start line) is delivered, once, to the next child.  INT is
# forwarded as TERM because background children of a non-interactive shell start with SIGINT
# ignored.  Pass one program
# (redirect its output with >>file 2>&1 on the call), not a pipeline: only the direct child is
# signalled.  Under "set -e" call it as:  rc=0; hyak_run_forwarding prog args || rc=$?

HYAK_STAGES="check-env probe corpus wheel pool stage-inputs init train-uniform train-rl eval-prepare eval-uniform eval-rl replay-check smoke-train"
HYAK_STAGE_OVERRIDE_FLAGS="PARTITION QOS ACCOUNT GRES CPUS MEM TIME SEGMENT_MARGIN_MINUTES PLATFORM"
HYAK_MONITOR_REL="modal_results/hyak_tillicum"
HYAK_TERM_RECEIVED=${HYAK_TERM_RECEIVED:-0}

_HYAK_RE_BLANK='^[[:blank:]]*(#.*)?$'
_HYAK_RE_LINE='^[[:blank:]]*(export[[:blank:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
_HYAK_RE_DQ='^"([^"\\$`]*)"([[:blank:]]+(#.*)?)?$'
_HYAK_RE_SQ="^'([^']*)'([[:blank:]]+(#.*)?)?\$"
_HYAK_RE_UQ='^([A-Za-z0-9_./:@%+,=-]*)([[:blank:]]+(#.*)?)?$'
_HYAK_RE_INT='^[0-9]+$'
_HYAK_RE_T_DHMS='^([0-9]+)-([0-9]+)(:([0-9]+)(:([0-9]+))?)?$'
_HYAK_RE_T_HMS='^([0-9]+):([0-9]+):([0-9]+)$'
_HYAK_RE_T_MS='^([0-9]+):([0-9]+)$'

hyak_ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
hyak_info() { printf '[hyak %s] %s\n' "$(hyak_ts)" "$*" >&2; }
hyak_warn() { printf '[hyak %s] WARNING: %s\n' "$(hyak_ts)" "$*" >&2; }
hyak_die() { printf '[hyak %s] ERROR: %s\n' "$(hyak_ts)" "$*" >&2; exit 1; }

# --- repository root -------------------------------------------------------------
_hyak_lib_repo=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -z "${HYAK_REPO:-}" ]; then
  HYAK_REPO=$_hyak_lib_repo
  export HYAK_REPO
elif [ "$(CDPATH='' cd -P -- "$HYAK_REPO" 2>/dev/null && pwd -P)" != "$_hyak_lib_repo" ]; then
  hyak_warn "HYAK_REPO=$HYAK_REPO is not the checkout that holds this lib.sh ($_hyak_lib_repo)"
fi

# --- small helpers --------------------------------------------------------------------
hyak_valid_stage() {
  case " $HYAK_STAGES " in
    *" ${1:-} "*) return 0 ;;
  esac
  hyak_die "Unknown stage '${1:-}'; expected one of: $HYAK_STAGES"
}

hyak_stage_suffix() {
  hyak_valid_stage "${1:-}"
  printf '%s\n' "$1" | tr 'a-z-' 'A-Z_'
}

# _hyak_normal_path VALUE: success for an absolute path that is already normalised, i.e. one that conda
# and Python (os.path.normpath) would not rewrite: no trailing slash, no "//", no "/./", no ".." component.
_hyak_normal_path() {
  case $1 in
    /) return 1 ;;
    /*) ;;
    *) return 1 ;;
  esac
  case $1 in
    */|*//*|*/./*|*/.|*/../*|*/..) return 1 ;;
  esac
  return 0
}

_hyak_isset() { eval "[ \"\${$1+set}\" = set ]"; }
_hyak_get() { eval "printf '%s' \"\${$1-}\""; }

_hyak_stage_keys() {
  local flag stage suffix out=""
  for flag in $HYAK_STAGE_OVERRIDE_FLAGS; do
    for stage in $HYAK_STAGES; do
      suffix=$(printf '%s' "$stage" | tr 'a-z-' 'A-Z_')
      out="$out HYAK_${flag}_${suffix}"
    done
  done
  printf '%s ' "$out"
}

_hyak_in_list() {  # _hyak_in_list WORD "LIST"
  case " $2 " in *" $1 "*) return 0 ;; esac
  return 1
}

# hyak_parse_time TIME: seconds of a Slurm time limit (M, M:S, H:M:S, D-H, D-H:M, D-H:M:S);
# prints "inf" for UNLIMITED/INFINITE; returns 1 (printing nothing) when invalid.
# Mirrors hyak.common.slurm.parse_time_limit.
hyak_parse_time() {
  local t=${1:-} d=0 h=0 m=0 s=0
  case $t in
    UNLIMITED|unlimited|INFINITE|infinite) printf 'inf\n'; return 0 ;;
  esac
  if [[ $t =~ $_HYAK_RE_T_DHMS ]]; then
    d=${BASH_REMATCH[1]}; h=${BASH_REMATCH[2]}; m=${BASH_REMATCH[4]:-0}; s=${BASH_REMATCH[6]:-0}
  elif [[ $t =~ $_HYAK_RE_T_HMS ]]; then
    h=${BASH_REMATCH[1]}; m=${BASH_REMATCH[2]}; s=${BASH_REMATCH[3]}
  elif [[ $t =~ $_HYAK_RE_T_MS ]]; then
    m=${BASH_REMATCH[1]}; s=${BASH_REMATCH[2]}
  elif [[ $t =~ $_HYAK_RE_INT ]]; then
    m=$t
  else
    return 1
  fi
  printf '%s\n' $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

# --- configuration ------------------------------------------------------------------
# _hyak_parse_env_file PATH: fills _HYAK_PARSED_KEYS / _HYAK_PARSED_VALS (same grammar as
# hyak/common/config.py parse_env_file).
_hyak_parse_env_file() {
  local path=$1 line lineno=0 key raw value k
  _HYAK_PARSED_KEYS=()
  _HYAK_PARSED_VALS=()
  [ -r "$path" ] || hyak_die "Cannot read config file $path"
  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    case $line in
      *$'\r'*) hyak_die "$path:$lineno: carriage return found (Windows line endings); convert the file with: sed -i 's/\\r\$//' $path" ;;
    esac
    if [[ $line =~ $_HYAK_RE_BLANK ]]; then
      continue
    fi
    if [[ $line =~ $_HYAK_RE_LINE ]]; then
      key=${BASH_REMATCH[2]}
      raw=${BASH_REMATCH[3]}
    else
      hyak_die "$path:$lineno: expected KEY=value (the line is not shown: it could hold a secret such as HF_TOKEN)"
    fi
    if [[ $raw =~ $_HYAK_RE_DQ ]]; then
      value=${BASH_REMATCH[1]}
    elif [[ $raw =~ $_HYAK_RE_SQ ]]; then
      value=${BASH_REMATCH[1]}
    elif [[ $raw =~ $_HYAK_RE_UQ ]]; then
      value=${BASH_REMATCH[1]}
    else
      hyak_die "$path:$lineno: value of $key must be unquoted [A-Za-z0-9_./:@%+,=-], double-quoted without \" \\ \$ or backtick, or single-quoted without ' (a trailing comment needs whitespace before #)"
    fi
    for k in ${_HYAK_PARSED_KEYS[@]+"${_HYAK_PARSED_KEYS[@]}"}; do
      [ "$k" != "$key" ] || hyak_die "$path:$lineno: duplicate key $key"
    done
    _HYAK_PARSED_KEYS[${#_HYAK_PARSED_KEYS[@]}]=$key
    _HYAK_PARSED_VALS[${#_HYAK_PARSED_VALS[@]}]=$value
  done < "$path"
}

# hyak_load_config [--no-export]
# Resolution per key: environment variable (if set, even empty) > config file > config.env.example.
# Config file: $HYAK_CONFIG, else $HYAK_REPO/hyak/config.env.  Validates CHANGE_ME placeholders,
# absolute ANYTOK_ROOT/ANYTOK_RUNS, integer settings and Slurm times.  Sets HYAK_CONFIG_FILE.
hyak_load_config() {
  local export_mode=1 example file key value i found stage_keys all_keys env_keys="" bad="" v
  local -a def_keys def_vals file_keys file_vals
  if [ "${1:-}" = "--no-export" ]; then export_mode=0; fi
  [ -n "${HYAK_REPO:-}" ] || hyak_die "HYAK_REPO is not set (submit.sh exports it; set it to the repository checkout)"
  example="$HYAK_REPO/hyak/config.env.example"
  file="${HYAK_CONFIG:-$HYAK_REPO/hyak/config.env}"
  [ -f "$example" ] || hyak_die "Missing $example (incomplete checkout?)"
  [ -f "$file" ] || hyak_die "Config file $file not found. Remedy: cp $example $HYAK_REPO/hyak/config.env and set ANYTOK_ROOT and ANYTOK_RUNS (or point HYAK_CONFIG at your file)"

  _hyak_parse_env_file "$example"
  def_keys=(${_HYAK_PARSED_KEYS[@]+"${_HYAK_PARSED_KEYS[@]}"})
  def_vals=(${_HYAK_PARSED_VALS[@]+"${_HYAK_PARSED_VALS[@]}"})
  _hyak_parse_env_file "$file"
  file_keys=(${_HYAK_PARSED_KEYS[@]+"${_HYAK_PARSED_KEYS[@]}"})
  file_vals=(${_HYAK_PARSED_VALS[@]+"${_HYAK_PARSED_VALS[@]}"})
  stage_keys=$(_hyak_stage_keys)

  for key in ${file_keys[@]+"${file_keys[@]}"}; do
    if ! _hyak_in_list "$key" "${def_keys[*]}" && ! _hyak_in_list "$key" "$stage_keys"; then
      bad="$bad $key"
    fi
  done
  [ -z "$bad" ] || hyak_die "$file: unknown key(s)$bad (typo?). Known keys: ${def_keys[*]}; per-stage overrides HYAK_<FLAG>_<STAGE> with FLAG in $HYAK_STAGE_OVERRIDE_FLAGS"

  # All keys to resolve: defaults, then per-stage overrides from the file or the environment.
  all_keys="${def_keys[*]}"
  for key in ${file_keys[@]+"${file_keys[@]}"} ${!HYAK_@}; do
    if _hyak_in_list "$key" "$stage_keys" && ! _hyak_in_list "$key" "$all_keys"; then
      all_keys="$all_keys $key"
    fi
  done
  # Record which keys come from the environment before assigning anything.
  for key in $all_keys; do
    if _hyak_isset "$key"; then env_keys="$env_keys $key"; fi
  done

  _HYAK_CONFIG_KEYS=""
  for key in $all_keys; do
    if _hyak_in_list "$key" "$env_keys"; then
      value=$(_hyak_get "$key")
    else
      found=0
      i=0
      while [ "$i" -lt "${#file_keys[@]}" ]; do
        if [ "${file_keys[$i]}" = "$key" ]; then value=${file_vals[$i]}; found=1; break; fi
        i=$((i + 1))
      done
      if [ "$found" = 0 ]; then
        value=""
        i=0
        while [ "$i" -lt "${#def_keys[@]}" ]; do
          if [ "${def_keys[$i]}" = "$key" ]; then value=${def_vals[$i]}; break; fi
          i=$((i + 1))
        done
      fi
    fi
    printf -v "$key" '%s' "$value"
    if [ "$export_mode" = 1 ]; then export "${key?}"; fi
    _HYAK_CONFIG_KEYS="$_HYAK_CONFIG_KEYS $key"
  done
  HYAK_CONFIG_FILE=$file
  if [ "$export_mode" = 1 ]; then export HYAK_CONFIG_FILE; fi

  # Validation (the Python validator, python -m hyak.common.config --check, checks more).
  bad=""
  for key in $_HYAK_CONFIG_KEYS; do
    v=$(_hyak_get "$key")
    case $v in *CHANGE_ME*) bad="$bad $key" ;; esac
  done
  [ -z "$bad" ] || hyak_die "Invalid configuration ($file):$bad still contain CHANGE_ME. Remedy: edit $file"
  for key in ANYTOK_ROOT ANYTOK_RUNS; do
    v=$(_hyak_get "$key")
    case $v in
      /*) ;;
      *) hyak_die "Invalid configuration ($file): $key='$v' must be an absolute path" ;;
    esac
    _hyak_normal_path "$v" || hyak_die "Invalid configuration ($file): $key='$v' must be a normalised path (no trailing slash, '//', '/./' or '..': conda and Python rewrite such paths, so every job would fail at activation). Remedy: write it without them, e.g. ${v%/}"
    case $v in
      "$_hyak_lib_repo"|"$_hyak_lib_repo"/*) hyak_die "Invalid configuration ($file): $key='$v' is inside the repository checkout $_hyak_lib_repo; environments, corpus and checkpoints would land in the git working tree. Choose a directory outside it, e.g. /gpfs/projects/<group>/anytok-data" ;;
    esac
  done
  for key in $_HYAK_CONFIG_KEYS; do
    v=$(_hyak_get "$key")
    case $key in
      HYAK_TIME|HYAK_TIME_DEBUG|HYAK_TIME_*)
        if [ "$key" = HYAK_TIME ] || [ "$key" = HYAK_TIME_DEBUG ] || [ -n "$v" ]; then
          i=$(hyak_parse_time "$v") || hyak_die "Invalid configuration ($file): $key='$v' is not a Slurm time limit (e.g. 24:00:00)"
          if [ "$i" = inf ] || [ "$i" = 0 ]; then hyak_die "Invalid configuration ($file): $key='$v' must be a finite, non-zero time limit"; fi
        fi ;;
      HYAK_SEGMENT_MARGIN_MINUTES*)
        if [ "$key" = HYAK_SEGMENT_MARGIN_MINUTES ] || [ -n "$v" ]; then
          [[ $v =~ $_HYAK_RE_INT ]] || hyak_die "Invalid configuration ($file): $key='$v' must be an integer >= 0"
        fi ;;
      HYAK_HN_CHUNK_ROWS)
        [[ $v =~ $_HYAK_RE_INT ]] && [ "$v" -ge 4096 ] || hyak_die "Invalid configuration ($file): $key='$v' must be an integer >= 4096 (hypernetwork chunk rows; hyak.memory enforces the same floor)" ;;
      HYAK_CPUS*)
        case $v in ''|none|NONE|None) ;; *)
          [[ $v =~ $_HYAK_RE_INT ]] && [ "$v" -ge 1 ] || hyak_die "Invalid configuration ($file): $key='$v' must be an integer >= 1 (or none)" ;;
        esac ;;
    esac
  done
}

# hyak_print_config: resolved KEY=VALUE lines (sorted; HF_TOKEN shown as *** when set).
hyak_print_config() {
  local key value
  [ -n "${_HYAK_CONFIG_KEYS:-}" ] || hyak_die "hyak_print_config: call hyak_load_config first"
  for key in $_HYAK_CONFIG_KEYS; do
    value=$(_hyak_get "$key")
    if [ "$key" = HF_TOKEN ] && [ -n "$value" ]; then value='***'; fi
    printf '%s=%s\n' "$key" "$value"
  done | LC_ALL=C sort
}

# hyak_stage_value KEY STAGE: per-stage override KEY_<STAGE> when non-empty, else KEY.
hyak_stage_value() {
  local key=${1:-} stage=${2:-} suffix flag value
  flag=${key#HYAK_}
  _hyak_in_list "$flag" "$HYAK_STAGE_OVERRIDE_FLAGS" || hyak_die "hyak_stage_value: $key has no per-stage override"
  suffix=$(hyak_stage_suffix "$stage")
  value=$(_hyak_get "${key}_${suffix}")
  if [ -z "$value" ]; then value=$(_hyak_get "$key"); fi
  printf '%s\n' "$value"
}

# --- monitor directories -----------------------------------------------------------------
hyak_monitor_dir() {
  hyak_valid_stage "${1:-}"
  if [ -n "${HYAK_MONITOR_ROOT:-}" ]; then
    printf '%s/%s\n' "$HYAK_MONITOR_ROOT" "$1"
  else
    printf '%s/%s/%s\n' "$HYAK_REPO" "$HYAK_MONITOR_REL" "$1"
  fi
}

hyak_log_dir() {
  local dir
  dir=$(hyak_monitor_dir "${1:-}") || exit 1
  mkdir -p "$dir" || hyak_die "Cannot create monitor directory $dir"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    (CDPATH='' cd -- "$dir" && ln -sfn "slurm-$SLURM_JOB_ID.out" latest-slurm.out) \
      || hyak_warn "could not update $dir/latest-slurm.out"
  fi
  printf '%s\n' "$dir"
}

# --- environment -----------------------------------------------------------------------------
_hyak_relax() { _HYAK_SAVED_FLAGS=$-; set +eu; }
_hyak_restore() {
  case $_HYAK_SAVED_FLAGS in *e*) set -e ;; esac
  case $_HYAK_SAVED_FLAGS in *u*) set -u ;; esac
}

# _hyak_load_lmod: make the "module" command available (no-op when it already is).
_hyak_load_lmod() {
  local f
  if ! type module >/dev/null 2>&1; then
    for f in "${LMOD_PKG:-/nonexistent}/init/bash" /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
             /usr/share/lmod/lmod/init/bash; do
      if [ -r "$f" ]; then
        _hyak_relax
        # shellcheck disable=SC1090
        . "$f"
        _hyak_restore
        break
      fi
    done
  fi
  type module >/dev/null 2>&1 || hyak_die "The 'module' command is not available; load Lmod or set HYAK_MODULES= (empty) and put conda on PATH"
}

# hyak_load_modules WHERE MODULE...: module load each module; a failure names WHERE (the place the
# user can change, e.g. "HYAK_MODULES in hyak/config.env" or "--module").
hyak_load_modules() {
  local where=${1:-} m rc
  shift
  [ $# -gt 0 ] || return 0
  _hyak_load_lmod
  for m in "$@"; do
    _hyak_relax
    module load "$m"
    rc=$?
    _hyak_restore
    [ "$rc" = 0 ] || hyak_die "module load $m failed (exit $rc); check $where"
  done
}

# hyak_gcc_module: print the compiler module a build needs and return 0 when HYAK_GCC_MODULE is set (the user's
# choice) or when a C compiler (cc or gcc) or a C++ compiler (c++ or g++) is missing from PATH (the default module
# is gcc/13.4.0); return 1, printing nothing, when no module is needed.
hyak_gcc_module() {
  if [ -n "${HYAK_GCC_MODULE:-}" ] \
     || ! { command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; } \
     || ! { command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1; }; then
    printf '%s\n' "${HYAK_GCC_MODULE:-gcc/13.4.0}"
    return 0
  fi
  return 1
}

hyak_modules() {
  [ -n "${HYAK_MODULES:-}" ] || return 0
  # shellcheck disable=SC2086
  hyak_load_modules "HYAK_MODULES in ${HYAK_CONFIG_FILE:-hyak/config.env}" $HYAK_MODULES
}

# --- environment of a job --------------------------------------------------------------------------
#: Variables of the login shell that must not reach a job: they redirect or extend Python, or inject
#: libraries.  CUDA_VISIBLE_DEVICES is deliberately not here (Slurm sets it for the allocated GPU).
HYAK_SANITIZE_VARS="PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE LD_PRELOAD"
#: Directory (inside the conda prefix) that holds only the link to libpython, for LD_LIBRARY_PATH.
HYAK_LIBPYTHON_DIRNAME="hyak-libpython"
HYAK_LIBPYTHON_SONAME="libpython3.12.so.1.0"
HYAK_ENV_REMOVED=""

# hyak_sanitize_env: unset $HYAK_SANITIZE_VARS and export PYTHONNOUSERSITE=1.  HYAK_ENV_REMOVED
# lists, space separated, the variables that were set (names only; values are never logged).
hyak_sanitize_env() {
  local name removed=""
  for name in $HYAK_SANITIZE_VARS; do
    if _hyak_isset "$name"; then
      removed="$removed $name"
      unset "$name"
    fi
  done
  export PYTHONNOUSERSITE=1
  HYAK_ENV_REMOVED=${removed# }
}

# _hyak_libpython_dir PREFIX: create (idempotently) <prefix>/hyak-libpython holding only the link to
# libpython3.12.so.1.0 and print its path; prints nothing and returns 1 when it cannot be created.
_hyak_libpython_dir() {
  local prefix=$1 dir="$1/$HYAK_LIBPYTHON_DIRNAME" link
  link="$dir/$HYAK_LIBPYTHON_SONAME"
  mkdir -p "$dir" 2>/dev/null || return 1
  if [ -e "$prefix/lib/$HYAK_LIBPYTHON_SONAME" ]; then
    if [ "$(readlink "$link" 2>/dev/null || true)" != "../lib/$HYAK_LIBPYTHON_SONAME" ]; then
      ln -sfn "../lib/$HYAK_LIBPYTHON_SONAME" "$link" 2>/dev/null || [ -L "$link" ] || return 1
    fi
  fi
  printf '%s\n' "$dir"
}

# _hyak_ld_library_path DIR: LD_LIBRARY_PATH with DIR first and every entry that lies under
# $ANYTOK_ROOT/env (conda lib directories or link directories of an earlier activation) dropped.
_hyak_ld_library_path() {
  local dir=$1 entry out="$1"
  local -a parts
  IFS=: read -r -a parts <<< "${LD_LIBRARY_PATH:-}"
  for entry in ${parts[@]+"${parts[@]}"}; do
    [ -n "$entry" ] || continue
    case $entry in
      "$ANYTOK_ROOT"/env|"$ANYTOK_ROOT"/env/*) continue ;;
    esac
    out="$out:$entry"
  done
  printf '%s\n' "$out"
}

hyak_activate() {
  local which=${1:-} prefix hook rc py libdir name
  case $which in
    train|data) ;;
    *) hyak_die "hyak_activate: expected 'train' or 'data', got '$which'" ;;
  esac
  [ -n "${ANYTOK_ROOT:-}" ] || hyak_die "hyak_activate: call hyak_load_config first"
  prefix="$ANYTOK_ROOT/env/$which"
  [ -x "$prefix/bin/python" ] || hyak_die "Environment $prefix is missing. Remedy: on the login node run: bash $HYAK_REPO/hyak/env/setup_${which}_env.sh"
  hyak_modules
  command -v conda >/dev/null 2>&1 || hyak_die "conda not found after loading modules '${HYAK_MODULES:-}'. Remedy: set HYAK_MODULES (e.g. conda) or put conda on PATH"
  _hyak_relax
  hook=$(conda shell.bash hook 2>/dev/null)
  rc=$?
  if [ "$rc" = 0 ]; then
    eval "$hook"
    conda activate "$prefix"
    rc=$?
  fi
  _hyak_restore
  [ "$rc" = 0 ] || hyak_die "conda activate $prefix failed (exit $rc)"
  [ "${CONDA_PREFIX:-}" = "$prefix" ] || hyak_die "conda activate did not activate $prefix (CONDA_PREFIX=${CONDA_PREFIX:-unset}); is ANYTOK_ROOT written the way conda writes it (no trailing slash, no '//')?"
  py=$(command -v python || true)
  [ "$py" = "$prefix/bin/python" ] || hyak_die "python resolves to '${py:-nothing}', not $prefix/bin/python; check PATH"
  if libdir=$(_hyak_libpython_dir "$prefix"); then
    LD_LIBRARY_PATH=$(_hyak_ld_library_path "$libdir")
    export LD_LIBRARY_PATH
  else
    hyak_warn "could not create $prefix/$HYAK_LIBPYTHON_DIRNAME; LD_LIBRARY_PATH is left as it is (the Rust library may not import). Remedy: bash $HYAK_REPO/hyak/env/setup_${which}_env.sh"
  fi
  export PYTHONPATH="$HYAK_REPO"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
  export HF_HOME="$ANYTOK_ROOT/hf-cache"
  export HYAK_ENV="$which"
  case ${HYAK_STAGE:-} in
    eval-prepare|eval-uniform|eval-rl) export TOKENIZERS_PARALLELISM=false ;;
    pool|wheel) export TOKENIZERS_PARALLELISM=true ;;
    '') hyak_warn "HYAK_STAGE is not set; leaving TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-<unset>}" ;;
    *) unset TOKENIZERS_PARALLELISM ;;
  esac
  # Stages that read only staged files (nothing is downloaded after the corpus, model and wheel steps):
  # fail at once instead of waiting for a network that may not be there.
  case ${HYAK_STAGE:-} in
    train-uniform|train-rl|smoke-train|eval-prepare|eval-uniform|eval-rl|probe|init|replay-check) export HF_HUB_OFFLINE=1 ;;
    corpus) unset HF_HUB_OFFLINE ;;       # the build reads the sources from the Hub: a stale value would break it
  esac
  # The pinned trainers compute in the numerical mode they were validated in: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
  # would force TF32 matmuls over the trainers' own choice and PYTHONOPTIMIZE strips assert statements (and
  # changes __debug__) from the trainer and the guards.  Neither may come from the login shell or a module.
  # PYTORCH_CUDA_ALLOC_CONF is deliberately kept: it is a memory-allocator setting, not a numerical one.
  case ${HYAK_STAGE:-} in
    train-uniform|train-rl|smoke-train)
      for name in TORCH_ALLOW_TF32_CUBLAS_OVERRIDE PYTHONOPTIMIZE; do
        if _hyak_isset "$name"; then
          hyak_warn "unsetting $name for the training stage ${HYAK_STAGE} (it would change the trainer's numerics or its checks; value not logged)"
          unset "$name"
        fi
      done ;;
  esac
  hyak_info "activated $prefix ($(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?'))"
}

# hyak_isolate_pip: for the setup scripts.  pip must see no configuration of the user or the site and no
# PIP_* variable: an extra index, a find-links directory or a constraints file would change which torch
# (or datasets ...) is installed.  --index-url on the command line does not remove an extra index.
hyak_isolate_pip() {
  local name
  for name in $(compgen -e | grep '^PIP_' || true); do
    unset "$name"
  done
  export PIP_CONFIG_FILE=/dev/null
  export PIP_DISABLE_PIP_VERSION_CHECK=1
}

# hyak_private_tmp DIR: use DIR (created) as TMPDIR for pip and conda, so that a download of several GB
# does not land in the small /tmp of a login node.  Call hyak_private_tmp_cleanup in an EXIT trap.
HYAK_PRIVATE_TMP=""
hyak_private_tmp() {
  mkdir -p "$1" || hyak_die "Cannot create the temporary directory $1"
  HYAK_PRIVATE_TMP=$1
  export TMPDIR=$1
}
hyak_private_tmp_cleanup() {
  if [ -n "$HYAK_PRIVATE_TMP" ] && [ -d "$HYAK_PRIVATE_TMP" ]; then
    rm -rf -- "$HYAK_PRIVATE_TMP" 2>/dev/null || true
  fi
}

# --- signals and child processes ------------------------------------------------------------
_HYAK_CHILD_PID=""
_HYAK_SIGNAL_PENDING=0
_HYAK_WAIT_SIGNALLED=0

_hyak_on_signal() {
  HYAK_TERM_RECEIVED=1
  _HYAK_WAIT_SIGNALLED=1
  if [ -n "${_HYAK_CHILD_PID:-}" ]; then
    hyak_info "received SIG$1; forwarding SIGTERM to child $_HYAK_CHILD_PID"
    kill -TERM "$_HYAK_CHILD_PID" 2>/dev/null || true
  else
    _HYAK_SIGNAL_PENDING=1
    hyak_info "received SIG$1 (no child running)"
  fi
}

# hyak_trap_term: keep the shell alive on TERM/INT and only record HYAK_TERM_RECEIVED=1, so
# a loop of several children can stop starting new ones after Slurm's warning signal.
hyak_trap_term() {
  trap '_hyak_on_signal TERM' TERM
  trap '_hyak_on_signal INT' INT
}

hyak_run_forwarding() {
  local prev_term prev_int child got status again
  [ $# -gt 0 ] || hyak_die "hyak_run_forwarding: no command given"
  prev_term=$(trap -p TERM)
  prev_int=$(trap -p INT)
  _HYAK_CHILD_PID=""
  # _HYAK_SIGNAL_PENDING is not reset here: a signal recorded while no child ran (for example while
  # job_step printed its "start:" line) belongs to this child and is delivered once, below.
  trap '_hyak_on_signal TERM' TERM
  trap '_hyak_on_signal INT' INT
  "$@" &
  child=$!
  _HYAK_CHILD_PID=$child
  if [ "$_HYAK_SIGNAL_PENDING" = 1 ]; then
    _HYAK_SIGNAL_PENDING=0
    hyak_info "forwarding a signal that arrived before child $child started"
    kill -TERM "$child" 2>/dev/null || true
  fi
  while :; do
    _HYAK_WAIT_SIGNALLED=0
    got=0
    wait "$child" || got=$?
    if [ "$got" -le 128 ] || [ "$_HYAK_WAIT_SIGNALLED" = 0 ]; then
      status=$got          # wait returned because the child exited (a code > 128 means it died by a signal)
      break
    fi
    # A trapped signal interrupted wait.  Keep waiting while the child exists.
    if kill -0 "$child" 2>/dev/null; then
      continue
    fi
    # The child is gone; its status was either the value just returned or is held by the shell.
    again=0
    wait "$child" 2>/dev/null || again=$?
    if [ "$again" = 127 ]; then status=$got; else status=$again; fi
    break
  done
  _HYAK_CHILD_PID=""
  if [ -n "$prev_term" ]; then eval "$prev_term"; else trap - TERM; fi
  if [ -n "$prev_int" ]; then eval "$prev_int"; else trap - INT; fi
  return "$status"
}

# --- Python helpers (never fail the caller) ---------------------------------------------------
hyak_python() {
  if [ -n "${HYAK_PYTHON:-}" ]; then printf '%s\n' "$HYAK_PYTHON"
  elif command -v python >/dev/null 2>&1 && [ -n "${CONDA_PREFIX:-}" ]; then command -v python
  else command -v python3 || printf 'python3\n'
  fi
}

_hyak_py() {
  local py rc=0
  py=$(hyak_python)
  PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$py" "$@" || rc=$?
  return "$rc"
}

hyak_status_start() {
  local stage=${1:-} task=${2:-}
  hyak_valid_stage "$stage"
  if [ -n "$task" ]; then
    _hyak_py -m hyak.common.status start --stage "$stage" --task "$task" >/dev/null || hyak_warn "could not write start status for $stage"
  else
    _hyak_py -m hyak.common.status start --stage "$stage" >/dev/null || hyak_warn "could not write start status for $stage"
  fi
  return 0
}

hyak_status_event() {
  local stage=${1:-} name=${2:-} item
  local -a args
  hyak_valid_stage "$stage"
  [ -n "$name" ] || hyak_die "hyak_status_event: event name required"
  shift 2
  args=(-m hyak.common.status event --stage "$stage" --name "$name")
  for item in "$@"; do args[${#args[@]}]=--set; args[${#args[@]}]=$item; done
  _hyak_py "${args[@]}" >/dev/null || hyak_warn "could not append event $name for $stage"
  return 0
}

hyak_finalize_status() {
  local stage=${1:-} rc=${2:-} log=${3:-} message=${4:-}
  local -a args
  hyak_valid_stage "$stage"
  [[ $rc =~ $_HYAK_RE_INT ]] || hyak_die "hyak_finalize_status: exit code must be an integer, got '$rc'"
  args=(-m hyak.common.status finalize --stage "$stage" --exit-code "$rc")
  if [ -n "$log" ]; then args[${#args[@]}]=--log; args[${#args[@]}]=$log; fi
  if [ -n "$message" ]; then args[${#args[@]}]=--message; args[${#args[@]}]=$message; fi
  _hyak_py "${args[@]}" >/dev/null || hyak_warn "could not finalize status for $stage"
  return 0
}

# hyak_register_dashboard: add the stage to the dashboard registry.  The registry is written under a
# file lock on the checkout's file system; a holder that hangs (or a file system on which flock never
# returns) must not keep a job waiting in its prologue, so the registration is a watched child that is
# given up after $HYAK_REGISTER_TIMEOUT_SECONDS (default 60) with a warning.  The content it writes is
# idempotent: the next job registers again.
hyak_register_dashboard() {
  local limit=${HYAK_REGISTER_TIMEOUT_SECONDS:-60} pid waited=0 rc=0 py
  [[ $limit =~ $_HYAK_RE_INT ]] || limit=60
  py=$(hyak_python)
  PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$py" -m hyak.tools.register_dashboard &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      hyak_warn "dashboard registration did not finish within ${limit} s (a stuck lock on $HYAK_REPO/$HYAK_MONITOR_REL?); continuing without it. plans/run-dashboard.json was not updated by this job"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "$pid" || rc=$?
  [ "$rc" = 0 ] || hyak_warn "dashboard registration failed (exit $rc); plans/run-dashboard.json was not updated"
  return 0
}
