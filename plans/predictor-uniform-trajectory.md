# Predictor versus uniform throughout early training

Compare both actual training runs at steps 2,000, 4,000, 6,000 and 8,000.
Use every one of the same 50 randomly drawn pool tokenizers and 12 held-out
tokenizers from the four-way 8k evaluation. Preserve all 576 passages, shared
prefixes, canonical continuation IDs, document IDs and scored bytes exactly.
BOS + unscored prefix + 32 scored canonical tokens must fit 128 positions.
Only prefix embeddings are transferred; the canonical suffix inputs and output
head are fixed. Score full native vocabularies with the original feature maps.

The existing verified 8k scores are reused without changing or rerunning them.
Six earlier checkpoints require 372 new model/tokenizer evaluations, for 496
comparisons across all eight checkpoints. Keep sample A and sample B separate.
Report absolute transferred BPB, each checkpoint's canonical BPB, and their gap.
Define improvement as Uniform gap minus Predictor gap: positive favors Predictor.
Average BPB within each text domain by scored bytes, then equally over nine domains.

Plot all-tokenizer trajectories, random-50 and held-out averages, settings groups,
and the ten largest 8k improvements on sample B. These ten are a retrospective
display group; all 62 tokenizers remain evaluated and saved. They must not be
presented as a new independently selected confirmation set.

Evaluate on Tillicum H200s with the original four-way runtime and evaluator.
Transfer uniform weights directly from their persistent Modal Volume to Tillicum;
reuse predictor weights already on Tillicum. Do not download models, corpora or
tokenizer files locally. Do not resume or queue training. No explicit CPU or
memory reservation. Save loss metadata, provenance, CSVs, PNG/PDF plots and a
plain-language explanation; archive results on Hugging Face and display progress
in the existing run dashboard.

Intervals resample source-document clusters within domains, paired across models
and training steps. Random-50 mean intervals additionally resample tokenizer IDs
with finite-population correction. They do not include retraining uncertainty.
This remains an actual-run comparison with one training seed per method, including
the known difference in training boundary construction; it is not a selector-only
ablation.
