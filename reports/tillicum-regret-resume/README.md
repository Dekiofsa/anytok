# Regret RL: Modal to Tillicum continuation

The existing entropy-zero regret experiment is running on one H200, Slurm job
470164, resumed from complete Modal step **5,288**. This is a continuation of
the trained LM, hypernetwork, optimizer and policy; corpus and RNG states are
restored together. The later uncommitted Modal trajectory is retained separately.

The migration transferred 23,475,030,053 bytes directly between cloud services
and verified every file hash. `restart-receipt.json` records exact provenance.
`initial-backup.json` pins the verified full restart in the public model repository
[cminst/anytok-regret-rl-20k](https://huggingface.co/cminst/anytok-regret-rl-20k).

`modal-config.json` and `tillicum-config.json` retain the before/after settings.
Only artifact locations, trainer implementation and the validated GPU memory
execution change. Reward stays regret and entropy weight stays zero. Nine
focused migration tests reject changed reward, entropy, data, pool, length,
microbatch and learning-rate settings; all 13 dashboard tests also pass.

`first-progress.json` records successful resumed training: model/policy counter
continuity, 128 valid paired examples, at most 128 total positions, and policy
reward equal to the paired BPB gap. The measured peak is about **115.18 GiB**
for the resumed 127k-vocabulary workload. This report is an initial-progress
snapshot; the live dashboard and checkpoint repository track later progress.

The previous numerical-fidelity tests and H200 fit test remain in
[chunked-memory](../chunked-memory/README.md) and
[tillicum-128k-smoke](../tillicum-128k-smoke/README.md). Cross-device bitwise
trajectory equality is not asserted.

The [run plan](../../plans/anytok-pool-regret-rl-20k.md) records monitoring,
checkpoint backups, 24-hour allocation limits and graceful continuation.
