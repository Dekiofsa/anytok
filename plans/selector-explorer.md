# Interactive replay of the completed selector simulations

The user requested an interactive explanation with a slider and visibility into
the experiment. Add `/selector-explorer` to the existing dashboard server on
0.0.0.0:1337. Reuse the dashboard fonts, palette and layout. Read-only UI;
there are no job controls.

Show all three selectors together, each with its complete 1,000-tokenizer world,
its pre-choice estimates, selected action, actual observed reward and current
selection shortfall. Display generator settings, prior selection counts, current
hardest action, a full-run timeline, and a sortable table of every tokenizer.
Explain empirical starting rewards, synthetic response rules, noise, and which
information is hidden from the learner. Practice-dependent worlds are separate
because each selector creates its own action history.

The existing simulation traces are immutable. Recover beliefs by importing the
original selector implementations and replaying every saved observation.
Require exact action agreement at all 20,000 steps in every run. Cache vectors
immediately before steps 1,25,50,...,20000; these are pre-choice values, before
the displayed reward updates the learner. Display increments and playback
speeds must expose one-in-four coverage exploration rather than alias it away.
An initial 100-step cache was superseded for this reason; its outcomes are not
used by the final explorer.

Record CPU replay progress as `selector-playback-build` in the existing run
registry. Cache/status: `modal_results/selector_playback_25/`. Uniform decisions
are also verified against the original RNG stream. Only metadata vectors are
cached, without LM weights or new GPU scoring. Source and trace hashes are
recorded in every completed cache receipt. Derived caches can be regenerated
from the published original traces and code.

Validate world values against audited traces at changes and late training,
pre-choice visit counts, probabilities and tensor shapes, full pool visibility,
slider/keyboard/playback controls, all scenarios and seeds, and desktop/mobile
rendering. Add a readable guide beside the simulation report.
