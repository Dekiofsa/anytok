# Simple online hardness predictor: approved frozen pilot

The user approved the simplified three-operation design and confirmed that
the LM and hypernetwork are frozen at **6,500**, with local execution on the
RTX PRO 6000 96GB. The joint training run remains paused at 8,000.

Use the complete original 1,000-tokenizer pool and original 23-bucket recipe.
Each of **20,000** committed observations performs a real full-vocabulary HN
forward and paired LM scoring on a fresh batch128/microbatch32, with active
dropout, unchanged valid paired boundaries, variable canonical continuations,
and BOS + prefix + continuation <=128 positions. No cached ranking reward is
supplied to the learner. Reference panel/bank raw-window hashes are reserved
for evaluation, advancing within the selected recipe bucket on exact matches.

Reuse the existing 18→64→64→1 Tanh network, initialization seed43, float64 CPU
parameters and Adam0.001. Output is predicted gap in BPB. Following each real
measurement, make exactly one single-observation squared-error update.
There is no warmup, replay, entropy term or explicit age weighting.

Observations 1,5,9,... explore using a private-RNG shuffled pool permutation.
Reshuffle after all1,000 exploration actions are consumed. Remaining steps
choose the maximum prediction, with random tie breaking. This produces
exactly5 exploration visits per tokenizer in20k observations. Data sampling
uses a separate Python seed42; each actual reward uses its own fixed CUDA
seed2026111600+observation_index. Predictor initialization leaves model RNGs
untouched. Models are frozen with gradients disabled and parameter versions
checked on every checkpoint.

Preflight reproduces a prior same-device 6.5k/771 full-batch result exactly,
then verifies a real third reward/update after restoring predictor, optimizer,
coverage permutation/RNG, and corpus state. Probe observations are excluded
from the pilot's20k updates. Restart state is saved atomically every100
observations without copying frozen LM/HN weights. Every500-observation state
is uploaded to the existing persistent HF model repository.

The complete 6.5k ranking is evaluator-only. Plot actual observed rewards by
selection mode, reference hardness of learned exploitation, pool coverage,
and predictions against the full reference. Display-only100-step smoothing
does not change learning. The H200 reference is a finite estimate; local
Blackwell parity is checked against the prior local result.

Dashboard: `hardness-predictor-6500`, Diagnostics. State:
`modal_results/hardness_predictor_6500/`. Report:
`reports/hardness-predictor-6500/`. Source LM/HN checkpoint and original inputs
remain persistent in cloud storage; the user-authorized local diagnostic copy
is verified against original manifests. Code, score metadata and plots are
published to the existing GitHub branch after the completed audit.
