"""The raw-BPB handoff run: Qingyun's verified launcher and input bundle on Tillicum (README section 0).

* ``hyak.handoff.common``  pinned file names, the requirements, checkpoint inspection
* ``hyak.handoff.fetch``   download, check, extract and verify the input bundle (``hyak/handoff/fetch_inputs.sh``)
* ``hyak.handoff.run``     what the job scripts 12_smoke_rawbpb.sbatch and 13_train_rawbpb.sbatch ask before
                           and after ``scripts/run_raw_bpb_training.py`` (start, resume, complete, refuse; the
                           receipt checks)

Nothing here changes Qingyun's files (``scripts/``, ``runtime/``, ``docs/raw-bpb-handoff/``): they are run as
they are.
"""
