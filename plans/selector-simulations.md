# Local selector stress tests

The user approved local simulated environments, including explicitly synthetic
responses to training, and requested the original REINFORCE selector baseline.
The real frozen pilot is stopped at 15,300; joint training remains paused.

Use every original tokenizer and all 32 saved reward batches at model step
6,500. Preserve each tokenizer's measured noise distribution. Draw one common
batch index per step within each seed, shared across methods. Only the selected
action's reward enters the learner. The evaluator can inspect all current means.

Run 20,000 observations per cell, five seeds (43–47), three selectors, and five
environments: 75 cells and 1,500,000 simulated observations. Selectors are uniform,
the unchanged entropy-zero `SettingsTokenizerBandit` REINFORCE implementation,
and the unchanged `HardnessPredictor` with one-in-four full-pool coverage. Every
cell starts fresh; no pilot weights or evaluator labels initialize learners.

Environments: fixed measured means; gradual Unigram-family decline starting
at 10k; an abrupt decline at 10k of twice that size; exponential improvement of
the selected tokenizer; and improvement of the entire selected algorithm family.
Calibration uses #771's observed gap decline at model steps 6,500→8,000 and its
1,381 selections in that interval. Gradual duration is 1,500 steps. The
observational decline does not establish causal effects of individual choices.
These are transparent plausible/stress assumptions, not simulated LM training
whose realism has been established. Keep the empirical centered noise unchanged.

Report current maximum mean minus selected mean (missed hardness), pool
coverage, current selected ranks, concentration, and algorithm selection over
time. Five-seed means and standard deviations are descriptive, not confidence
intervals. Practice environments evolve differently under each selector: do not
compare their absolute rewards as if their model states were identical.

Use four local CPU workers, one Torch/BLAS thread each; no GPU/cloud jobs.
Dashboard: `selector-simulations`. State and per-cell traces:
`modal_results/selector_simulations/`; report: `reports/selector-simulations/`.
Completed cells resume from verified trace hashes; interrupted cells rerun
deterministically from their original seed. Pin implementation/input hashes.
