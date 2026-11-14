#!/bin/bash
# hyak/slurm/02_corpus.sbatch -- stage corpus: build the 23 corpus buckets from public Hugging Face data.
#
# Submit:                                  hyak/slurm/submit.sh corpus
# Skip the source probe:                   --env HYAK_CORPUS_PROBE=0
# Extra arguments for the build:           --env HYAK_CORPUS_ARGS="--buckets web-en code"
#
# Data environment.  The node needs outbound HTTPS to huggingface.co for the whole build.
#   1. python -m hyak.corpus.build --probe   first record of each of the ten sources at its pinned
#      revision (writes modal_results/hyak_tillicum/corpus/probe.json).  Exit 2 stops the job here:
#      a source cannot be read from this node (the probe's failure status), or a usage, configuration
#      or environment error.  Exit 1: crashed.
#   2. python -m hyak.corpus.build           all 23 buckets to the original pool targets into
#      $ANYTOK_ROOT/corpus/<size>-v1/<bucket>/; built buckets are re-verified and skipped.
#      Exit 0: complete, or stopped early for a benign reason (time limit, a bucket locked by
#      another job; status "paused": resubmit).  Exit 4: a bucket failed (the status error names
#      it and its log in $ANYTOK_ROOT/corpus/.hyak/logs/).  Exit 2: usage, configuration or
#      environment error, including a data environment that differs from the validated package
#      set (refused unless --allow-unvalidated-env "<reason>", which needs the user's approval).
#      Exit 1: crashed (traceback in the job log).  Exit 143: stopped by a signal.
#   3. python -m hyak.corpus.verify          fidelity scorecard against the repository's anchors
#      ($ANYTOK_ROOT/corpus/scorecard.txt, copied to modal_results/hyak_tillicum/corpus/); it only
#      reports and never fails the job.  Unless its verdict line is "content-identical to every
#      recorded anchor", the job prints the warning "corpus scorecard is not exact: do not submit
#      stage-inputs or the pool (README section 10)"; the job's exit status is not changed.
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

job_begin corpus
# An event, not "status start": hyak.corpus.build writes one worker row per bucket, and the row
# that "status start" adds would stay on the dashboard for the whole job.
hyak_status_event "$JOB_STAGE" job_start "task=corpus build"
hyak_activate data
hyak_register_dashboard

rc=0
if [ "${HYAK_CORPUS_PROBE:-1}" != 0 ]; then
  job_step python -m hyak.corpus.build --probe || rc=$?
  if [ "$rc" != 0 ]; then
    job_explain "$rc" "python -m hyak.corpus.build --probe" \
      "2=corpus source probe: this node could not read every source at its pinned revision, or a usage or environment error (see the job log and $JOB_DIR/probe.json; hyak/README.md explains what to do without internet on compute nodes)"
    exit "$rc"
  fi
fi

if job_stopping; then exit 0; fi
job_extra_args HYAK_CORPUS_ARGS
job_step python -m hyak.corpus.build ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
case $rc in
  0)
    if job_stopping; then exit 0; fi
    vrc=0
    job_step python -m hyak.corpus.verify || vrc=$?
    if [ "$vrc" = 0 ]; then
      hyak_info "corpus scorecard: $ANYTOK_ROOT/corpus/scorecard.txt (copy in $JOB_DIR)"
    else
      hyak_warn "python -m hyak.corpus.verify exited $vrc (a report only; the build result stands)"
    fi
    # Only the exact verdict (hyak.corpus.verify.EXACT_PREFIX) lets stage-inputs and the pool go on.
    if [ "$vrc" != 0 ] || ! grep -q '^verdict: content-identical to every recorded anchor' "$ANYTOK_ROOT/corpus/scorecard.txt" 2>/dev/null; then
      hyak_warn "corpus scorecard is not exact: do not submit stage-inputs or the pool (README section 10)"
    fi ;;
  *) job_explain "$rc" "python -m hyak.corpus.build" \
       "2=hyak.corpus.build: usage, configuration or environment error (for example a data environment that differs from the validated package set); the ERROR line in the job log has the remedy" \
       "4=a corpus bucket failed; the status error names it and its log in $ANYTOK_ROOT/corpus/.hyak/logs/" ;;
esac
exit "$rc"
