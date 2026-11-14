#!/bin/bash
# hyak/slurm/04_pool.sbatch -- stage pool: generate the 1,000 pool tokenizers (any number of jobs).
#
# Pilots first (6 jobs x 2 slots = the 12 pilots):  hyak/slurm/submit.sh pool --count 6 --env HYAK_POOL_ARGS=--pilots-only
# Full pool (resubmitting is always safe):           hyak/slurm/submit.sh pool --count N
# Extra arguments for the worker:                    --env HYAK_POOL_ARGS="--families unigram/no-regex --workers 1"
#
# Data environment (the patched wheel).  Runs python -m hyak.pool.worker --workers
# $HYAK_POOL_WORKERS --threads $HYAK_POOL_THREADS (2 x 4 Rayon threads on the 8 CPUs of one GPU);
# work is shared through atomic claims in $ANYTOK_ROOT/input/pool/claims, so jobs never train the
# same tokenizer.  Worker exit status: 0 the pool is complete, or a normal end (also after a graceful
# stop: other jobs, or one more job, finish it); 3 an index failed in this job or failed/blocked/exhausted
# indices exist (python -m hyak.pool.worker --report lists them); 2 invalid inputs or environment (the
# plan, input/pool-recipe.json, the wheel, a brought-in pool that was refused); 1 crashed (traceback in
# the job log).  One TERM (Slurm's early warning before the time limit) means "start no new tokenizer";
# a second TERM stops the running ones.
#
# After the worker (exit 0, or 3 when a tokenizer failed in this job), the job decides from the
# manifests, not from the shared status or from its own exit code: it switches to the train
# environment and runs python -m hyak.pool.collect --check-only (0 complete; 75 not complete yet; 4 a
# manifest is invalid or inconsistent: a failure, the log lists the indices).  Complete (exit 0): the job runs
#   python -m hyak.pool.collect (writes input/pool/full-results.json once and input/pool/plan.json;
#   exit 0 ok; 75 not now: the pool is not complete after all, or a second TERM stopped it: the job
#   ends 0 and says "submit one more pool job"; 4 refused: a manifest is invalid or inconsistent or an
#   existing index differs, a failure; 2 input error),
#   then python -m hyak.pool.scorecard (three-level fingerprint against the original pool; skipped
#   only when $ANYTOK_ROOT/eval/reports/pool-scorecard.json is newer than the index AND covers all
#   1,000 indices and all three levels).
# A pool that is not complete (exit 75 of --check-only): a job whose own worker exited 0 ends with
# 0 ("collect runs in the job that finishes the last tokenizer"); a job whose worker exited 3 ends with
# 3.  A pool that is complete ends with 0 even when this job's worker had exited 3 (another job retried
# the tokenizer).  collect and scorecard write once and take minutes: they are not stopped by the early
# warning signal (the batch shell only records it); a pilot job (--pilots-only) never collects.
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

job_begin pool
# An event, not "status start": the worker writes one row per slot (<job>.s0, <job>.s1), and the
# row that "status start" adds would stay on the dashboard for the whole job.
hyak_status_event "$JOB_STAGE" job_start "task=tokenizer generation"
hyak_activate data
hyak_register_dashboard
job_extra_args HYAK_POOL_ARGS

rc=0
job_step python -m hyak.pool.worker --workers "$HYAK_POOL_WORKERS" --threads "$HYAK_POOL_THREADS" \
  ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
worker_rc=$rc
case $rc in
  0|3) ;;                      # 3 may still end with a complete pool: the manifests decide below
  *) job_explain "$rc" "python -m hyak.pool.worker" \
       "2=pool worker: invalid inputs or environment (plan, input/pool-recipe.json, the wheel, or a brought-in pool that was refused; see the job log)"
     exit "$rc" ;;
esac
for word in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do
  if [ "$word" = --pilots-only ]; then
    hyak_info "pilot job: no collect"
    if [ "$worker_rc" = 3 ]; then
      job_explain 3 "python -m hyak.pool.worker" "3=pool: a tokenizer failed in this job, or failed/blocked indices exist; list them with python -m hyak.pool.worker --report (data environment)"
    fi
    exit "$worker_rc"
  fi
done

# Does the pool verify as complete?  (Needs the train environment: numpy, tokenizers, torch.)
hyak_activate train
rc=0
job_step_foreground python -m hyak.pool.collect --check-only || rc=$?
case $rc in
  0) ;;
  75)                          # not complete yet (only missing or failed indices)
    if [ "$worker_rc" = 3 ]; then
      job_explain 3 "python -m hyak.pool.worker" "3=pool: a tokenizer failed in this job, or failed/blocked indices exist; list them with python -m hyak.pool.worker --report (data environment)"
      exit 3
    fi
    hyak_info "pool not complete yet (collect --check-only exit $rc); collect runs in the job that finishes the last tokenizer"
    exit 0 ;;
  *) job_explain "$rc" "python -m hyak.pool.collect --check-only" \
       "4=pool: a manifest is invalid or inconsistent (collect --check-only refused it; the job log lists the indices and the reason); do not collect until it is understood (python -m hyak.pool.worker --report)" \
       "2=python -m hyak.pool.collect --check-only: invalid inputs (plan, pool directory or code); see the job log"
     exit "$rc" ;;
esac

hyak_info "all 1,000 tokenizers verify: collecting the pool index in the train environment"
rc=0
job_step_foreground python -m hyak.pool.collect || rc=$?
case $rc in
  0) ;;
  75) job_not_now collect_not_now "pool collect did not publish the index (exit 75): the pool is not complete after all, or collect was stopped by a second TERM; submit one more pool job (resubmitting is safe: it collects)"
      exit 0 ;;
  *) job_explain "$rc" "python -m hyak.pool.collect" \
       "4=python -m hyak.pool.collect refused the pool: a manifest is invalid or inconsistent, or an existing index differs; the job log lists the problem indices" \
       "2=python -m hyak.pool.collect: invalid inputs (plan, pool directory or staged code); the job log has the remedy"
     exit "$rc" ;;
esac
scorecard=$ANYTOK_ROOT/eval/reports/pool-scorecard.json
scorecard_ok=0
if [ -f "$scorecard" ] && [ "$scorecard" -nt "$ANYTOK_ROOT/input/pool/full-results.json" ]; then
  # A report of a pilot subset (--indices) or of one level is older work in a newer file: it must cover everything.
  python -c 'import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get("planned") == 1000 and r.get("indices") == 1000 and sorted(r.get("levels", [])) == [1, 2, 3] else 1)' \
    "$scorecard" 2>/dev/null && scorecard_ok=1
fi
if [ "$scorecard_ok" = 1 ]; then
  hyak_info "pool scorecard already written after the index and covering all 1,000 indices and three levels ($scorecard); not recomputed"
else
  job_step_foreground python -m hyak.pool.scorecard --workers "${SLURM_CPUS_PER_TASK:-8}" || rc=$?
  if [ "$rc" != 0 ]; then
    job_explain "$rc" "python -m hyak.pool.scorecard"
    JOB_MESSAGE="the pool index was written, but $JOB_MESSAGE; rerun it in a job"
  fi
fi
exit "$rc"
