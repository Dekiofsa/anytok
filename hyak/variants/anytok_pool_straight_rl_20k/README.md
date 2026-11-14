# Straight-BPB-reward variant of the joint regret RL20k trainer

`scripts/train_joint_smollm_input.py` here is the RL trainer of the Hyak port. It is a copy of
Qingyun's joint regret RL20k trainer with the policy reward changed from the paired gap to the
straight transferred bits per byte. Nothing else differs.

## Provenance

| | Path | SHA-256 | Bytes |
|---|---|---|---|
| Base | `runtime/anytok_pool_regret_rl_20k/scripts/train_joint_smollm_input.py` at commit `3de64a55815ec5ebecb37f4e47b7addbe886cdde` (branch `zett-modal-20k`; the file was added in `62b6b76`) | `76ff232402154653c5899826a4f63f8a4afadf5a443829f87919f8e75b31c6d4` | 34,752 |
| Variant | `hyak/variants/anytok_pool_straight_rl_20k/scripts/train_joint_smollm_input.py` | `8dd3b8570dd6545123ab1aee5d338a8957d72ece26e48a759196d2619f43ff51` | 34,723 |
| Diff | `straight_reward.patch` (unified diff, base to variant) | | |

The base file and the helper it imports (`runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py`,
SHA-256 `406cfc58bf1c46cef0dc6785bbb313ca5797c0d35bf6cd786b3115a92e5523c1`) are pinned in
`hyak/code/pinned.py` and are never edited. `hyak/tests/test_straight_variant.py` checks that
applying `straight_reward.patch` to the pinned base reproduces the variant byte for byte, and that the
two files differ in exactly the two lines below.

## Decision

On 2026-11-14 the user decided that the port's RL run is "the same as Qingyun's new joint regret
RL20k, but with the straight BPB loss as the reward" (plans/anytok-pool-regret-rl-20k.md). This is the
explicit approval `AGENTS.md` requires for a material change of the experiment. Until then the port
ran the original RL trainer (`runtime/anytok_pool_settings_rl_20k`: entropy 0.01, raw training BPB as
reward, the original example cuts). The uniform run is not affected: it keeps the original uniform
trainer (`runtime/anytok_pool_20k`) and the original cuts, which is also the uniform run Qingyun
compares his regret run with.

## What changes

1. `bandit.update(pool_index, train_bpb)` instead of `bandit.update(pool_index, reward_bpb)`: the
   policy receives the paired, pre-update transferred continuation BPB of the batch (transferred
   continuation NLL / shared continuation bytes / ln 2), not the gap to the canonical BPB.
2. `run_config.json` says so: `"policy_reward": "paired_pre_model_update_transferred_bpb"` (the regret
   trainer writes `paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb`).

No comment or docstring of the base describes the reward value itself, so none changed. The local
name `reward_bpb` keeps its meaning from the base: it holds the paired gap, which is still logged as
`paired_gap_bpb` (now a diagnostic, not the reward).

## What does not change

Everything else of Qingyun's protocol, byte for byte: the corrected paired token boundaries
(`make_paired_prefix_example`, the nearest valid canonical boundary to the midpoint, 128 positions for
both inputs) for training batches and the sanity evaluation; the no-grad canonical control forward of
every update on the pre-update LM, which restores the RNG state it consumes; the paired checks (shared
targets and bytes, `Too many non-roundtripping or oversized paired examples`); all paired logging
(`paired_canonical_bpb`, `paired_gap_bpb`, `paired_valid_examples`, `paired_maximum_positions`,
`paired_batch_sha256`, `pairing_protocol`, `paired_reward_source_sha256`); the policy (shared-settings
MLP 18-64-64-1, seed 43 = seed + 1, Adam learning rate 0.001, EMA baseline 0.99, entropy weight 0,
passed explicitly by `hyak.train.run_segment.rl_argv`); the LM and hypernetwork objective, optimizers,
learning-rate schedule, seed 42, checkpoint cadence, and the trainer's own comparison of 21 settings
with the uniform run's `run_config.json`.

The canonical forward is still computed although the policy no longer uses its result: the run must
equal the regret run except for the reward, and `paired_gap_bpb` stays a logged diagnostic (the Hyak
wrapper checks it after every segment). It costs one extra no-grad LM forward per update.
