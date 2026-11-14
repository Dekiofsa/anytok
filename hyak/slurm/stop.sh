#!/usr/bin/env bash
# hyak/slurm/stop.sh -- stop the jobs of one stage gracefully: the queued ones are cancelled first,
# then the running one gets Slurm's TERM through the batch shell, exactly like the early-warning signal.
#
# Usage (on the login node):
#   hyak/slurm/stop.sh <stage> [--dry-run]
#
#   <stage>      train-uniform train-rl pool eval-prepare eval-uniform eval-rl corpus wheel ... (any stage;
#                the jobs are found by their name, anytok-<stage>, among your own jobs)
#   --dry-run    print what would be done; change nothing
#
# Order of the steps, and why:
#   1. scancel every PENDING job of the stage.  A training chain is submitted with
#      --dependency=singleton, which ignores exit states: the next queued job would start the moment the
#      running one is gone.  Cancelling the queued ones first keeps the run stopped.
#   2. scancel --batch --signal=TERM <running job>.  The batch shell forwards it to the program
#      (hyak_run_forwarding): a trainer finishes the update, evaluates, saves its restart and the job ends
#      with 0 (status "paused"); a pool worker starts no new tokenizer (a second TERM stops the running ones
#      at once); a scoring job ends its current tokenizer.  Nothing is lost, and the stage resumes from
#      its files when you submit it again.
# Plain "scancel <job>" is the emergency stop: it sends TERM to every process and KILL 30 s later, which is
# not enough for a trainer to write a 23.5 GB restart.  Use it only when this script does not work.
#
# This is a command-line helper only (nothing in the dashboard starts or stops a job).
set -euo pipefail

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

HERE=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
HYAK_REPO=$(CDPATH='' cd -P -- "$HERE/../.." && pwd -P)
export HYAK_REPO
# shellcheck source=hyak/slurm/lib.sh
source "$HERE/lib.sh"

[ $# -ge 1 ] || { usage >&2; exit 2; }
case $1 in -h|--help) usage; exit 0 ;; esac
STAGE=$1
shift
DRY_RUN=0
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) hyak_die "Unknown option '$1' (see --help)" ;;
  esac
done
hyak_valid_stage "$STAGE"
command -v squeue >/dev/null 2>&1 || hyak_die "squeue not found; run this on a Slurm login node"
command -v scancel >/dev/null 2>&1 || hyak_die "scancel not found; run this on a Slurm login node"

USER_NAME=${USER:-$(id -un)}
NAME="anytok-$STAGE"
LISTING=$(squeue -h -u "$USER_NAME" -n "$NAME" -o '%i %T' 2>/dev/null) \
  || hyak_die "squeue could not list your jobs; try again in a moment"

PENDING=""
RUNNING=""
OTHER=""
while read -r job state; do
  [ -n "${job:-}" ] || continue
  case $state in
    PENDING) PENDING="$PENDING $job" ;;
    RUNNING) RUNNING="$RUNNING $job" ;;
    *) OTHER="$OTHER $job($state)" ;;
  esac
done <<< "$LISTING"

if [ -z "$PENDING$RUNNING$OTHER" ]; then
  echo "No job named $NAME is queued or running for $USER_NAME."
  exit 0
fi
echo "stage $STAGE ($NAME): queued:${PENDING:- none}; running:${RUNNING:- none}${OTHER:+; in another state:$OTHER (not touched)}"

stop_failed=0
if [ -n "$PENDING" ]; then
  for job in $PENDING; do
    if [ "$DRY_RUN" = 1 ]; then
      echo "dry run: scancel $job (queued)"
    else
      scancel "$job" || { hyak_warn "scancel $job failed"; stop_failed=1; }
      echo "cancelled queued job $job"
    fi
  done
fi
if [ -n "$RUNNING" ]; then
  for job in $RUNNING; do
    if [ "$DRY_RUN" = 1 ]; then
      echo "dry run: scancel --batch --signal=TERM $job (running)"
    else
      scancel --batch --signal=TERM "$job" || { hyak_warn "scancel --batch --signal=TERM $job failed"; stop_failed=1; }
      echo "sent TERM to the batch shell of running job $job: it stops the program gracefully (a trainer saves a restart first; this can take several minutes). Watch: tail -f $HYAK_REPO/$HYAK_MONITOR_REL/$STAGE/latest-slurm.out"
    fi
  done
fi
[ "$DRY_RUN" = 0 ] || echo "# dry run: nothing changed"
exit "$stop_failed"
