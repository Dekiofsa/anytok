#!/bin/bash
# hyak/slurm/05_stage_inputs.sbatch -- stage stage-inputs: stage the trainer's input layout.
#
# Submit after the corpus build (minutes; fits the debug QoS):  hyak/slurm/submit.sh stage-inputs --debug
# Extra arguments for stage_inputs (e.g. --force):              --env HYAK_STAGE_INPUTS_ARGS=--force
#
# Train environment (standard library only).  Runs python -m hyak.corpus.stage_inputs, then the
# read-only re-check python -m hyak.corpus.stage_inputs --check.  It writes input/recipe.json,
# input/corpus/<bucket>/train.jsonl.gz (relative symlinks into $ANYTOK_ROOT/corpus),
# input/corpus/manifest.json (exactly 23 keys), input/eval/<bucket>.jsonl.gz, input/pool-recipe.json
# (read by the pool worker) and input/manifest.json.  It refuses test-scale buckets, buckets below
# their recipe quota, and any change once a run directory or a pool tokenizer exists (unless
# --force).  Exit 0: staged and consistent.  Exit 2: configuration or input error: the input/model
# directory is not fetched or does not match the pin (run python -m hyak.model.fetch_model), or the
# environment is wrong.  Exit 3: the corpus stop: there is no exact corpus scorecard (read
# $ANYTOK_ROOT/corpus/scorecard.txt and README section 10); continuing needs the user's approval, then
# add --accept-corpus-scorecard "<reason>" through HYAK_STAGE_INPUTS_ARGS.  Exit 4: refused or
# inconsistent (the reason is in the log; also for the --check step).  Exit 1: crashed (traceback).
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

job_begin stage-inputs
hyak_status_start "$JOB_STAGE" "staging trainer inputs"
hyak_activate train
hyak_register_dashboard
job_extra_args HYAK_STAGE_INPUTS_ARGS

rc=0
job_step python -m hyak.corpus.stage_inputs ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
if [ "$rc" = 0 ]; then
  if job_stopping; then exit 0; fi
  job_step python -m hyak.corpus.stage_inputs --check || rc=$?
fi
case $rc in
  0) hyak_info "stage-inputs: $ANYTOK_ROOT/input staged and consistent" ;;
  *) job_explain "$rc" "python -m hyak.corpus.stage_inputs" \
       "2=hyak.corpus.stage_inputs: input/model is not fetched or does not match the pin, or another input or configuration error; run python -m hyak.model.fetch_model (see the job log for the exact problem)" \
       "3=hyak.corpus.stage_inputs stopped: no exact corpus scorecard: see $ANYTOK_ROOT/corpus/scorecard.txt and README section 10; continuing needs the user's approval (--accept-corpus-scorecard \"<reason>\" through HYAK_STAGE_INPUTS_ARGS or the stage's extra-arguments variable)" \
       "4=hyak.corpus.stage_inputs refused or found an inconsistency (also for --check); the reason is in the job log" ;;
esac
exit "$rc"
