"""Memory-saving hypernetwork forward for training on one H200 (DESIGN.md decision D2).

* :mod:`hyak.memory.chunked_hypernet` -- the vocabulary-chunked, checkpointed
  replacement of ``ZettHypernet.__call__`` used for every training update
  (``install`` / ``uninstall`` / ``settings``).  Evaluation (no gradients) stays on
  the pinned forward.  Its SHA-256 is part of the run identity: it must not change
  between the uniform bootstrap and update 20,000 of both runs.
* :mod:`hyak.memory.memory_model` -- the peak-memory model and predicted peaks
  (``predict_peak_gib``, ``prediction_table``; not hashed, may be corrected freely).
* :mod:`hyak.memory.probe` -- ``python -m hyak.memory.probe``: training-shaped
  GPU memory and speed measurements per vocabulary size, with a FITS / DOES NOT
  FIT verdict and an on-GPU equivalence check.
"""
