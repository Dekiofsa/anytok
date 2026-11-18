# Find where the stale-teacher explanation holds and where it breaks

The user asked to generalize the synthetic experiments and push their limits.
This is a separate, predeclared CPU study. Earlier synthetic studies and real
training runs remain unchanged; real training stays paused.

The questions in plain language are:

- Does the teacher still get stuck when practice helps related skills only a little?
- Does it happen if the student learns much slower or much faster?
- Was the earlier result driven by giving hundreds of neighbours a full update?
- Does it survive removing noisy answers, changing the plateau, changing the
  learning curve, or rearranging which skills improve together?
- Can we identify cases where learning occurs but the ranking stays fixed?

## Primary boundary study

Use all 1,000 tokenizer settings and all 32 measured batch residuals from the
step-6,500 reward table. Vary neighbour improvement strength over 0%, 1%, 10%
and 100%. Vary student learning speed over 0.01×, 0.25×, 1×, 4×, 64× and 256×
the previous reference rate, plus a separate frozen-student control. Do not
adjust these values after seeing results.

Run two practice rules. Under extra practice, the selected skill gets one unit
and each same-group neighbour gets the declared fraction of a unit. Under
shared practice, divide those weights by their sum, distributing exactly one
unit across the group. This matches total practice weight, not exact loss
reduction. Sharing 0% is identical under both rules and is evaluated once.

Initial gaps, measured noise, the 50% plateau, grouping by algorithm and the
exponential curve stay fixed in this primary study. There are 43 unique worlds,
including the frozen control.

## Predeclared checks of the assumptions

At 10% and 100% sharing, reference speed and extra practice, separately:

- remove observation noise or double its measured size;
- move the final plateau to 0% or 90% of initial difficulty;
- use nine smaller algorithm/pretokenizer groups, shuffled groups preserving
  the original group sizes, or a single group containing all skills;
- replace exponential learning with a clipped linear or inverse curve, using
  the same initial slope;
- make all skills equally difficult initially, or permute the initial gaps.

The initial rate remains calibrated with the earlier 50% plateau. Changing a
plateau never silently recalibrates the learning rate. Permuting initial gaps
does not permute the original centered per-tokenizer observation residuals.
Group/layout permutations use fixed seed 20261118, independently of teachers.

Add matched frozen controls for the new noise and starting-gap settings. In
the all-skill, full-sharing case, ranking is mathematically unchanged: this
tests whether broad improvement alone suffices when the hard choices do not
move. These checks add 26 worlds, for 69 unique worlds in the full study.

## Teachers and repetitions

Each world uses original entropy-zero REINFORCE, its correctly differentiated
25% uniform mixture, uniform choices, and a comparator that always picks a
currently hardest skill. The last comparator receives simulator truth only for
assessment. It is not an optimal long-term curriculum, and its labels never
enter the learned teachers.

Use five new independent teacher-network seeds, 48–52, paired across conditions.
All begin with exactly uniform probabilities. Each gets 20,000 observations.
The full study has 1,380 runs and 27.6 million choices; the expensive work is
the two learned teachers, while the two references need no network updates.
Use at most eight local CPU workers, one Torch/BLAS thread each. No GPUs or
Modal allocations. Expected elapsed time is roughly 1–2 hours, subject to
measured throughput; update ETA from the actual run rather than assuming it.

All conditions are retained, including those where the teacher succeeds and
where extra exploration hurts. Do not choose cases for having larger failures.
Historical conditions are preserved, not replaced by cheaper substitutes.

## Measurements, visibility and audit

Keep all per-step actions, rewards, current ranks, favorite probabilities,
policy advantages and before/after chosen-action probabilities. Report both
actual choices and learned choices before the extra uniform component.
Report final and whole-run selection error and remaining student difficulty.

A persistent stale majority means at least half the actual sampling probability
is on a favorite outside the current hardest 100 for most of the final 2,000
decisions. Also report continuous choice quality so the conclusion does not
rest on that single cutoff. Each method creates its own learning history;
compare its choice with alternatives available to that same student.

The final independent audit reconstructs every reward, student transition and
rank. Partial plots use complete five-repetition cells and are marked live;
they are not treated as audited final results. Cache audits only when raw trace,
plan and auditor hashes agree. Preserve scientific source and input hashes.

Dashboard ID: `student-generalization`. State and traces:
`modal_results/student_generalization/`. Report:
`reports/student-generalization/`. Raw simulation traces are measurement
metadata; original corpora, tokenizers and LM/HN checkpoints remain in cloud
storage. No real-model performance improvement follows from this simulation.

After the audit, archive the raw measurement traces and receipts directly to
Hugging Face under `cminst/anytok-hardness-predictor-20k`, diagnostic prefix
`student-generalization`. Publish only the generated report folder to the
existing GitHub branch; do not include unrelated user edits, force-push or
switch branches. Archive or publication errors must preserve all local results
and show that compute has completed while archival needs a retry.
