"""AnyTok tokenizer-transfer experiment, ported from Modal to the Hyak Tillicum Slurm cluster.

Everything specific to the Slurm port lives under this package.  The pinned
trainer snapshots under ``runtime/`` are never modified; wrappers import them
in-process.  Start with ``hyak/README.md`` (the run guide) and
``hyak/config.env.example`` (cluster settings).

Shared building blocks for every stage are in :mod:`hyak.common`:

* :mod:`hyak.common.config` -- ``load_config()``, ``repo_root()``, per-stage overrides.
* :mod:`hyak.common.paths`  -- ``Paths`` (the single source of truth for every
  storage location), stage names and dashboard run ids.
* :mod:`hyak.common.io`     -- hashing, atomic writes, JSONL appends.
* :mod:`hyak.common.status` -- ``StatusWriter`` for the dashboard's ``progress_json`` adapter.
* :mod:`hyak.common.slurm`  -- job id, remaining time, job liveness, sbatch resources.
"""
