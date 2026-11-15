from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_regret_favorite import analyze


def fixture():
    inventory = dict(steps=[500, 1000], panel_sha256='panel', checkpoints=[
        dict(step=step, manifest_sha256=f'manifest-{step}') for step in [500, 1000]])
    records = []
    for step in [500, 1000]:
        batches = []
        for index in range(32):
            base = 1 + .01 * (index % 2)
            gap = (.2 if step == 500 else .1) + .01 * (index % 2)
            batches.append(dict(index=index, valid_examples=128, maximum_positions=128,
                dropout_seed=2026111500+index, scored_bytes=100 if index%2 else 10000,
                scored_tokens=1000, raw_bpb=base+gap, canonical_bpb=base, gap_bpb=gap,
                paired_batch_sha256=f'batch-{index}', rejected_examples=0))
        records.append(dict(step=step, tokenizer_index=771, panel_sha256='panel',
            source_manifest_sha256=f'manifest-{step}', batches=batches,
            frozen_parameter_versions_unchanged=True, gpu='same', torch='2.15',
            tokenizer_sha256='native', worker_sha256='worker', staged_source_manifest_sha256='source',
            policy_probability=.5, cumulative_selections=step//2))
    return records, inventory


def test_checkpoint_change_is_paired_and_means_updates_equally():
    result, table, scores, draws = analyze(*fixture())
    assert result['complete'] and result['examples'] == 4096
    assert table[0]['gap_bpb'] == pytest.approx(.205)
    assert table[1]['gap_bpb'] == pytest.approx(.105)
    assert result['endpoint_change']['gap_bpb']['last_minus_first'] == pytest.approx(-.1)
    assert result['endpoint_change']['gap_bpb']['paired_batch_bootstrap_95pct'] == pytest.approx([-.1, -.1])


@pytest.mark.parametrize('field', ['paired_batch_sha256', 'scored_tokens', 'scored_bytes', 'dropout_seed'])
def test_changed_context_targets_or_counts_cannot_be_compared(field):
    records, inventory = fixture()
    records[1]['batches'][4][field] = 'changed' if field == 'paired_batch_sha256' else 123
    with pytest.raises(ValueError):
        analyze(records, inventory)


def test_partial_plot_is_explicitly_incomplete_and_no_missing_batch_is_accepted():
    records, inventory = fixture()
    assert analyze(records[:1], inventory)[0]['complete'] is False
    records[1]['batches'].pop()
    with pytest.raises(ValueError):
        analyze(records, inventory)
