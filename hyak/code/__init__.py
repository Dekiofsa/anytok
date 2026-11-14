"""Staged code trees of the two training runs (DESIGN.md section 5.A).

* :mod:`hyak.code.pinned`     -- full SHA-256 and size of every pinned repository file
  (the two runtime snapshots, the Rust sources, ``scripts/tokenizer_pool.py``, reference
  plans) and the layout of the staged ``uniform`` and ``rl`` trees.
* :mod:`hyak.code.stage_code` -- copies those files into ``$ANYTOK_ROOT/code/{uniform,rl}``,
  writes ``$ANYTOK_ROOT/code/code_manifest.json`` and verifies an existing stage.

Both modules use only the standard library and run on any Python >= 3.9.
"""
