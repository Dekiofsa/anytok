#!/bin/bash
# hyak/slurm/09_eval_score.sbatch -- stages eval-uniform and eval-rl: score a final checkpoint.
#
# Pilot (pool panel only, no packing):   hyak/slurm/submit.sh eval-rl --env HYAK_EVAL_INDICES="0 616 822 192"
# Full scoring (resubmitting is safe):   hyak/slurm/submit.sh eval-uniform --count 2
#                                        hyak/slurm/submit.sh eval-rl --count 2
# Choose panels:                         --env HYAK_EVAL_PANELS=pool   (default "pool natural")
# Extra arguments for every score call:  --env HYAK_EVAL_SCORE_ARGS="--checkpoint /path/to/archived/step-020000"
#   (for example a copy of the final checkpoint, because Scrubbed storage deletes untouched files)
#
# Train environment (TOKENIZERS_PARALLELISM=false), one H200.  For each panel:
#   python -m hyak.eval.score --run RUN --panel PANEL [--indices ...] [$HYAK_EVAL_SCORE_ARGS]
# on $ANYTOK_RUNS/<run>/step-020000 (it must carry the training wrapper's manifest.json); work is
# shared by claims, the canonical and BOS-only controls are computed once per run and panel, and
# a graceful SIGTERM ends the job with 0.  score exit status: 0 = every index of the panel's work list
# is scored, or the job stopped on a signal or the time margin (resubmit to continue); 75 = this panel's
# work list is not complete: after the bounded wait other live jobs (the other job of --count 2, or an
# earlier one) still hold claims.  75 is "not now", not an error: the job remembers the panel, goes on
# with the next panel and with packing, and at the end finishes as a "not now" (the job ends with 0 and a
# note, no HALTED file, the stage is not complete); the job holding the claims, or one more job, finishes
# the panel.  Then, unless HYAK_EVAL_INDICES is set (a pilot),
#   python -m hyak.eval.pack --panel PANEL
# packs and summarises into $ANYTOK_ROOT/eval/reports/<panel>/.  pack exits 75 while a run is not
# completely scored yet (for example while the other run, uniform or rl, or another eval job still scores
# its last tokenizer; the complete runs are packed): "not now" too, the job goes on and the job that scores
# the last tokenizer packs the panel.  Any other non-zero status of score or pack is a
# failure (1 = crashed, 2 = input or configuration error).
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

job_begin eval-uniform eval-rl
RUN=${JOB_STAGE#eval-}
if [ -n "${HYAK_RUN:-}" ] && [ "$HYAK_RUN" != "$RUN" ]; then
  hyak_die "HYAK_RUN=$HYAK_RUN contradicts stage $JOB_STAGE"
fi
export HYAK_RUN="$RUN"

INDICES=()
PILOT=0
if [ -n "${HYAK_EVAL_INDICES:-}" ]; then
  read -r -a INDICES <<< "$HYAK_EVAL_INDICES"
  [ "${#INDICES[@]}" -gt 0 ] || hyak_die "HYAK_EVAL_INDICES contains no index"
  PILOT=1
  PANELS=${HYAK_EVAL_PANELS:-pool}
  hyak_info "pilot: indices ${INDICES[*]} on panel(s) $PANELS; no packing"
else
  PANELS=${HYAK_EVAL_PANELS:-pool natural}
fi
for panel in $PANELS; do
  case $panel in pool|natural) ;; *) hyak_die "HYAK_EVAL_PANELS may contain only 'pool' and 'natural', got '$panel'" ;; esac
done

hyak_status_start "$JOB_STAGE" "scoring the $RUN checkpoint"
hyak_activate train
hyak_register_dashboard

job_extra_args HYAK_EVAL_SCORE_ARGS
rc=0
LATER_PANELS=""           # panels whose work list was not complete when score ended (score exit 75)
later_add() {             # later_add PANEL: remember it once
  case " $LATER_PANELS " in *" $1 "*) ;; *) LATER_PANELS="${LATER_PANELS:+$LATER_PANELS }$1" ;; esac
}
for panel in $PANELS; do
  if job_stopping; then exit 0; fi
  rc=0
  if [ "$PILOT" = 1 ]; then
    job_step python -m hyak.eval.score --run "$RUN" --panel "$panel" --indices "${INDICES[@]}" ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
  else
    job_step python -m hyak.eval.score --run "$RUN" --panel "$panel" ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
  fi
  case $rc in
    0) ;;
    75) later_add "$panel"
        hyak_info "score --panel $panel: not complete (exit 75: other live jobs still hold claims); going on with the next panel and packing, the job will end as 'not now'" ;;
    *) job_explain "$rc" "python -m hyak.eval.score --run $RUN --panel $panel" \
         "2=hyak.eval.score --run $RUN --panel $panel: input or configuration error (checkpoint, panel, pool or features; the ERROR line in the job log names the remedy)"
       exit "$rc" ;;
  esac
done
rc=0

# later_exit: the job ends 0 as a "not now" when a panel was not complete (no HALTED file; the stage is not complete).
later_exit() {
  if [ -n "$LATER_PANELS" ]; then
    job_not_now eval_incomplete "panel(s) $LATER_PANELS of $RUN not completely scored yet (score exit 75: other live jobs hold claims after the wait); resubmit when those jobs have ended, or submit one more eval job"
  fi
  exit 0
}

if [ "$PILOT" = 1 ]; then
  hyak_info "pilot done: read elapsed_seconds and peak_gib in $ANYTOK_ROOT/eval/scores/$RUN/<panel>/NNNN.json"
  later_exit
fi
if job_stopping; then exit 0; fi
for panel in $PANELS; do
  prc=0
  job_step python -m hyak.eval.pack --panel "$panel" || prc=$?
  case $prc in
    0) hyak_info "packed: $ANYTOK_ROOT/eval/reports/$panel/" ;;
    75) hyak_info "pack --panel $panel: not complete yet (a run is not completely scored; the job that scores its last tokenizer packs it); its complete runs were packed" ;;
    143) exit "$prc" ;;
    *) rc=$prc
       job_explain "$prc" "python -m hyak.eval.pack --panel $panel" \
         "2=hyak.eval.pack --panel $panel: input or configuration error (the scores are kept; fix it and rerun pack in a job)"
       exit "$rc" ;;
  esac
done
later_exit
