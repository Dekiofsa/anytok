# Synthetic student: why does an adversarial teacher become stale?

The user approved synthetic student learning rules to investigate the RL-specific
teacher–student failure. This is a separate CPU-only experiment. Real training
stays paused. Historical selector simulations remain unchanged.

Every world has all 1,000 original tokenizer settings. Initial difficulty and
observation noise come from the complete step-6,500 table: 32 measured batches
per tokenizer. A selector sees only the reward for its chosen tokenizer. The
simulator knows every current difficulty only for evaluation.

For each tokenizer, initial mean gap is split into a fixed floor and a learnable
deficit, each half the initial gap. Practice shrinks the deficit exponentially:

`current gap = floor + initial learnable deficit × exp(-rate × practice count)`

This is an intuitive hypothetical student, not a fitted LM or a claim that half
of real transfer loss is irreducible. Unlike the first simulator, improvement
plateaus instead of continuing toward zero. Set the rate once, before outcomes,
so 1,381 practices reproduce the relative decline observed for #771 between
6.5k and 8k. That real observational decline does not establish causality.

Three student rules isolate the extent of improvement:

1. Frozen student: practice changes nothing.
2. Individual learning: only the selected tokenizer improves.
3. Shared learning: every tokenizer using the selected algorithm improves.

The third rule assumes complete transfer within BPE, WordPiece or Unigram.
It is a limiting stress case, not a measured property of the real hypernetwork.

Compare the original entropy-zero REINFORCE selector with the same selector
whose sampling distribution retains 25% uniform probability:

`q = 0.75 × learned probabilities + 0.25 × uniform`

The exploratory version samples q and differentiates log q. Do not apply the
original log-p update to actions drawn from another distribution. Both use the
same shared settings network, Adam .001, EMA baseline .99 and entropy zero.
There is no hardness predictor in this comparison. This version uses a fixed
uniform probability floor, not the predictor's deterministic shuffled cycle;
coverage is probabilistic rather than a guarantee of every tokenizer per cycle.

Start each condition in two ways: original uniform initialization, and the actual
collapsed step-6,500 policy. Reconstruct the latter using all 6,500 recorded
real actions and rewards, preserving network, optimizer and EMA state. Require
every replayed action and advantage to agree, plus final counts and probabilities
to match the independently saved checkpoint metadata. Failure stops the run.
No LM or hypernetwork weights are downloaded or updated.

Fork each starting state into five paired future sampling/noise seeds, 43–47.
These are five continuation repetitions of one real trained policy, not five
independently trained LMs. Every condition receives the same sequence of batch
indices within a seed, with every action's original centered residuals unchanged.
Negative reward observations are retained. Rewards are observed before the
synthetic student improves.

Scope: 3 student rules × 2 starting states × 2 coverage settings × 5 paired
continuations × 20,000 choices = 60 runs and 1.2 million observations. Four local
CPU workers, one Torch/BLAS thread each; no GPU or cloud allocation.

Keep every action, observation, advantage, before/after chosen-action probability,
current hardest tokenizer, current favorite and its true rank. Report how much
harder an available tokenizer was than the actual choice. Also report the
learned policy's expected choice separately from its 25% uniform component,
so exploration's immediate cost does not masquerade as poor learned ranking.

Define a visibly stale majority as assigning at least 50% actual sampling
probability to a favorite outside the current hardest 100. Report the fraction
of time in that condition, its onset, and the remaining difficulty over time.
The simulator audits all 1.2 million observations independently from saved actions.

Practice worlds diverge across selectors. Selection quality is measured against
each world's current alternatives, not by comparing raw rewards across worlds.
Synthetic mastery describes this declared rule only; it is not LM generalization.
No result establishes the cause of the real training collapse without real-world
confirmation. Do not tune the rules or omit unfavorable seeds after inspection.

Dashboard: `synthetic-student`; state `modal_results/synthetic_student/`;
report `reports/synthetic-student/`. The existing interactive explorer continues
to show its original 75 runs; these new outcomes are not substituted into it.
