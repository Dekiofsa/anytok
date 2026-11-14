"""Corpus pipeline of the Hyak port (DESIGN.md D4 and section 5.E).

* :mod:`hyak.corpus.sources`      verbatim, Modal-free port of ``scripts/modal_prepare_corpus.py``
                                  (plans, adapters, held-out rule, bucket builder, verify/publish/probe logic).
* :mod:`hyak.corpus.build`        ``python -m hyak.corpus.build``: builds ``$ANYTOK_ROOT/corpus/<size>-v1/<bucket>/``.
* :mod:`hyak.corpus.stage_inputs` ``python -m hyak.corpus.stage_inputs``: the trainer layout under ``$ANYTOK_ROOT/input/``.
* :mod:`hyak.corpus.verify`       ``python -m hyak.corpus.verify``: fidelity scorecard against the repository's anchors.

``sources`` imports ``datasets`` and ``requests`` lazily (inside the adapters), exactly like the
original, so everything except the live adapters works with the standard library only.
"""
