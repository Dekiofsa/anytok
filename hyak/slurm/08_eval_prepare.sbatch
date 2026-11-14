#!/bin/bash
# hyak/slurm/08_eval_prepare.sbatch -- stage eval-prepare: panels, features and prefix validation.
#
# Submit once the pool is collected (CPU work; 2-3 jobs may share it):  hyak/slurm/submit.sh eval-prepare [--count 2]
#
# Train environment (TOKENIZERS_PARALLELISM=false, as in the original scoring images).  In order,
# stopping at the first failure:
#   1. python -m hyak.eval.rebuild_panel --panel all      decode the stored 576 + 2,032 passages and
#      verify every context hash and both panel hashes (exit 4: a stored panel did not rebuild exactly,
#      nothing written, see $ANYTOK_ROOT/eval/panels/<panel>/rebuild-report.json; exit 2: usage or input error);
#   2. python -m hyak.eval.prepare_features               forms/priors for all 1,000 tokenizers
#      (claim-based, shared by every eval-prepare job);
#   3. python -m hyak.eval.validate_prefixes --panel pool
#   4. python -m hyak.eval.validate_prefixes --panel natural
#      the repository's joint validation/trimming loop for the regenerated pool.  Exit 3 is a
#      protocol stop (an unsatisfiable row, no convergence, or an originally valid pool context
#      would be trimmed): a decision for you, see $ANYTOK_ROOT/eval/panels/<panel>/validation-report.json.
#      A stop of the pool panel does not keep the natural panel from being validated (it does not
#      depend on the pool): the job goes on to it and then ends with the stop's status (3).
# Extra arguments for both validate_prefixes calls (for example --workers 4 or --recheck):
#   --env HYAK_EVAL_PREPARE_ARGS="--workers 4"
# The stage status ends "complete" when all 1,000 feature sets exist and both panels are validated.
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

job_begin eval-prepare
hyak_status_start "$JOB_STAGE" "panels, features and prefix validation"
hyak_activate train
hyak_register_dashboard

rc=0
job_step python -m hyak.eval.rebuild_panel --panel all || rc=$?
if [ "$rc" != 0 ]; then
  job_explain "$rc" "python -m hyak.eval.rebuild_panel" \
    "4=hyak.eval.rebuild_panel: a stored passage or panel hash did not rebuild exactly (exit 4); nothing was written; see $ANYTOK_ROOT/eval/panels/<panel>/rebuild-report.json and the job log" \
    "2=hyak.eval.rebuild_panel: usage or input error (the canonical tokenizer, the pinned code or the configuration); the ERROR line in the job log has the remedy"
  exit "$rc"
fi
if job_stopping; then exit 0; fi
job_step python -m hyak.eval.prepare_features || rc=$?
if [ "$rc" != 0 ]; then
  job_explain "$rc" "python -m hyak.eval.prepare_features"
  exit "$rc"
fi
job_extra_args HYAK_EVAL_PREPARE_ARGS
stop_rc=0
stop_message=""
for panel in pool natural; do
  if job_stopping; then
    JOB_MESSAGE=$stop_message
    exit "$stop_rc"
  fi
  rc=0
  job_step python -m hyak.eval.validate_prefixes --panel "$panel" ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
  case $rc in
    0) ;;
    3) # a protocol stop of this panel: remember it and still validate the other panel
       stop_rc=3
       stop_message="${stop_message:+$stop_message; }prefix validation of the $panel panel stopped by the protocol (exit 3): a decision is needed; read $ANYTOK_ROOT/eval/panels/$panel/validation-report.json" ;;
    143) JOB_MESSAGE=$stop_message
         exit "$rc" ;;
    *) job_explain "$rc" "python -m hyak.eval.validate_prefixes --panel $panel"
       [ -z "$stop_message" ] || JOB_MESSAGE="$JOB_MESSAGE; also: $stop_message"
       exit "$rc" ;;
  esac
done
if [ "$stop_rc" != 0 ]; then
  JOB_MESSAGE=$stop_message
  exit "$stop_rc"
fi
hyak_info "eval-prepare: panels rebuilt, features prepared, both panels validated (or held by another job)"
exit 0
