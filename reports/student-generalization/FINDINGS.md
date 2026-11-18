# What the broader experiments tell us

All 1,380 runs and 27.6 million simulated choices completed and passed the
independent audit. These results describe hypothetical students, not measured
changes in a real language model. Each condition has five new teacher seeds.

The strongest pattern is a range of learning speeds where a teacher can get
stuck after its preferred group becomes easier. The failure is uncommon when
learning is very slow, and can disappear again when learning is extremely fast.
It is not true that every learning student defeats this teacher.

## A learning-speed window

With complete sharing across an algorithm family, extra practice and original
REINFORCE, the number of repetitions ending with an outdated majority was:

| Student speed relative to the reference rule | Ending stuck |
| --- | ---: |
| 0.01× | 0/5 |
| 0.25× | 4/5 |
| 1× | 5/5 |
| 4× | 3/5 |
| 64× | 1/5 |
| 256× | 0/5 |

At reference speed, reducing neighbour improvement from 100% to 10% or 1%
reduced the failure counts to 1/5 and 0/5. Strong shared improvement is a
reliable trigger in this particular regime; weak sharing is not a universal
failure condition. Occasional failures also occur without sharing.

## What the controls rule out—and what they do not

Removing answer noise does not remove the problem: the complete-sharing
reference case still fails in 4/5 repetitions. This rules out observation noise
as a necessary ingredient here. Action selection remains stochastic.

Making all skills improve together, so their difficulty order stays unchanged,
gives 0/5 failures for the original teacher. Broad improvement alone is therefore
insufficient in that control. Changing which skills are relatively hard is an
important candidate ingredient; this does not prove it is sufficient.

Smaller skill groups still fail in 5/5 reference repetitions. Shuffled groups
fail in 2/5. Linear and inverse learning curves fail in 5/5 and 4/5. The effect
is not restricted to the original three groups or one exponential curve.

Spreading one unit of practice across the group instead of giving each neighbour
extra practice removes the failure at reference speed. It reappears at faster
settings: 3/5 at 64× and 5/5 at 256×. The practice normalization conserves practice
weight, not exact loss improvement. The shifted failure region is consistent
with the speed of group-level improvement mattering; it does not establish an
effect of sharing independent of that speed.

Extra random exploration is not a universal repair. At the complete-sharing,
reference-speed setting, both teacher variants fail in 5/5 repetitions, although
exploration lowers the average distance from the hardest available choice.
The intervention is specifically a correctly differentiated uniform mixture.

## Interpretation

A plausible account is that difficulty moves after the teacher has become
confident about a preferred region, and the teacher fails to revise that
preference. Very slow change may not require a switch within the experiment;
very fast change may happen before a strong preference forms. The speed-window
results are consistent with that account, but do not yet identify its timing or
gradient mechanism causally. Full per-step traces are archived for that analysis.

The real-world question remains whether training on one tokenizer changes the
difficulty of related tokenizers strongly enough to enter this failure region.
The simulations make that a concrete measurement to perform; they do not answer
it on behalf of the real model.
