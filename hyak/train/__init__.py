"""Training segment runner for the uniform and RL 20k runs on Slurm (owner: train).

Modules:

* ``run_segment``  -- one trainer invocation per process (port of the Modal launchers' ``train_segment``)
* ``retention``    -- checkpoint manifests, retention hooks, pre-flight repair and the final grid audit
* ``pairing``      -- freezing and verifying the uniform run's files that the RL run depends on (D6)
* ``monitor``      -- dashboard status for a training stage, derived from the trainer's events.jsonl
* ``replay_check`` -- CPU replay of the trainer's example construction against the original run's CSV

Nothing here imports Modal.  The pinned trainers are only ever imported by
``run_segment.TrainerAdapter`` from the staged code tree.
"""
