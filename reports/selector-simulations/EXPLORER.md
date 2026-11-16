# See what is being tested

Open the [interactive selector explorer](http://100.84.104.59:1337/selector-explorer).
It uses the existing dashboard server. Locally, run
`python scripts/run_dashboard.py` and open `http://localhost:1337/selector-explorer`.

Each dot is one of the original 1,000 tokenizers. Its height in the truth chart
is its **current mean BPB gap**: the hardness that our simulated environment
assigns it. Higher means harder. Every method sees the same tokenizer settings,
but it does not see the truth chart.

The second chart shows what that selector has learned **before making the
displayed choice**. REINFORCE predicts selection probabilities; the new predictor
predicts BPB gaps. Those are different quantities, so their axes are labeled
differently. Uniform sampling assigns 0.1% to every tokenizer.

The black ring marks the actual choice from the saved run. The amber diamond
marks the currently hardest choice, which the selector cannot look up. The
observed reward is one noisy measurement around the chosen tokenizer's mean.
The selector uses only that one reward to update itself. In a practice-dependent
environment, the declared simulation rule then makes the selected tokenizer,
or its whole algorithm family, a little easier.

**Missed hardness = largest current mean − chosen tokenizer's current mean.**
This compares the choice with what was available in that method's world at that
moment. It is not the raw BPB loss or the reward itself. Lower is better.

Start at step 1 and press Play, or drag the slider. The slider shows exact frames
every 25 observations, plus step 1. These increments expose both coverage and
exploitation choices; every 100th step would systematically hide coverage.
The timeline summarizes all 20,000 original observations in 100-step windows.
Click any dot or enter an ID to follow the same tokenizer across all three
worlds. Expand the table to see every tokenizer and sort by hardness, estimate,
prior selections, or ID.

In the fixed/externally changing cases, the true means are identical across
methods at a given step. In the practice cases, **each method creates its own
world through its choices**. The three truth charts can therefore differ.
Those response rules are synthetic assumptions, informed by one observed
learning curve; they do not establish how real LM training would respond.

## Examples to inspect

- [The beginning](http://100.84.104.59:1337/selector-explorer?scenario=practice_family&seed=43&frame=0): no previous observations; predictions/probabilities start uninformative.
- [Abrupt change, just before](http://100.84.104.59:1337/selector-explorer?scenario=abrupt&seed=43&frame=396): press Play to see the Unigram means fall and whether each selector switches.
- [A stuck REINFORCE run](http://100.84.104.59:1337/selector-explorer?scenario=practice_family&seed=43&frame=800): the old selector still chooses WordPiece #443 with nearly all its probability even though harder choices remain in its own world.
- [Fixed hardness](http://100.84.104.59:1337/selector-explorer?scenario=fixed&seed=43&frame=800): concentrating can be effective when the hard choices stay hard.

## Where the estimates come from

The simulation outcome files are unchanged. `scripts/build_selector_playback.py`
replays their actual rewards through the unchanged original selector classes,
requiring agreement with **every original action**, then records pre-choice
vectors for the slider. Each cache receipt records source/trace checksums.
The reader checks array shapes, finite values, probability normalization,
checksums, and truth values against the original audited simulation trace.

Regenerate the display cache with the same project environment:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-memory/bin/python scripts/build_selector_playback.py
```

This runs on CPU and changes no experiment outcomes, LM weights, or
hypernetwork weights. Incomplete cached estimates are labeled as preparing;
they are never replaced by reference rewards or an approximate estimate.
