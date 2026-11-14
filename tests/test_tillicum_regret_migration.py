"""A platform migration must refuse changes to the scientific experiment."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from resume_regret_tillicum import migration, OLD_TRAINER_SHA256, REGRET_REWARD


def configs():
    requested = json.loads((ROOT / 'docs/raw-bpb-handoff/training-verification.json').read_text())['config']
    requested.update(policy_reward=REGRET_REWARD, reward_objective='regret')
    saved = deepcopy(requested)
    for key in ['reward_objective', 'memory_execution', 'memory_source_sha256']:
        del saved[key]
    saved.update(model='/mnt/rl/input/model', warmup_checkpoint='/old/modal/input/warmup',
                 training_script_sha256=OLD_TRAINER_SHA256)
    return saved, requested


def test_only_paths_code_and_validated_execution_can_change():
    saved, requested = configs()
    result = migration(saved, requested, 5288)
    assert result['step'] == 5288 and result['stage'] == 'execution_migration'
    assert migration(requested, requested, 6000) is None


@pytest.mark.parametrize('key,value', [('pool_size', 999), ('sequence_length', 256),
    ('recipe_sha256', 'different'), ('data_manifest_sha256', 'different'),
    ('hn_peak_lr', 0.1), ('microbatch_size', 4), ('pairing_protocol', 'old-cuts')])
def test_material_changes_are_rejected(key, value):
    saved, requested = configs();requested[key] = value
    with pytest.raises(ValueError):
        migration(saved, requested, 5288)


def test_raw_reward_or_entropy_change_cannot_slip_into_the_resume():
    saved, requested = configs();requested['reward_objective'] = 'raw_bpb'
    with pytest.raises(ValueError):
        migration(saved, requested, 5288)
    saved, requested = configs();requested['policy']['entropy_weight'] = .01
    with pytest.raises(ValueError):
        migration(saved, requested, 5288)
    saved, requested = configs()
    with pytest.raises(ValueError):
        migration(saved, requested, 5500)
