"""Preserve the pilot's decisions and updates through the full trainer interface."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hardness_predictor import HardnessPredictor
from joint_hardness_policy import SettingsTokenizerBandit
from test_hardness_predictor import configs
from run_hardness_training import prepared_sources
from monitoring.learning import LearningReader
import json


def test_adapter_matches_pilot_decisions_losses_parameters_and_rng():
    original=HardnessPredictor(configs());adapter=SettingsTokenizerBandit(configs())
    for step in range(100):
        action=original.select();selected=adapter.sample()
        assert selected==action['index']
        p=adapter.probabilities()
        assert p[selected]>0 and p.sum()==pytest.approx(1.)
        if action['mode']=='explore':assert p[selected]==1.
        reward=.05+.01*(step%4)
        expected=original.observe(selected,reward);metrics=adapter.update(selected,reward)
        assert metrics['selector_squared_error']==expected
        assert metrics['selector_mode']==action['mode']
        for key,value in original.network.state_dict().items():assert torch.equal(value,adapter.network.state_dict()[key])
    assert adapter.updates==100 and len(adapter.recent)==100


def test_complete_resume_preserves_next_real_choice_and_single_update():
    adapter=SettingsTokenizerBandit(configs())
    for step in range(17):adapter.update(adapter.sample(),.04+.001*step)
    state=copy.deepcopy(adapter.state_dict());restored=SettingsTokenizerBandit(configs());restored.load_state_dict(state)
    for step in range(20):
        a,b=adapter.sample(),restored.sample();assert a==b
        assert adapter.update(a,.07)==restored.update(b,.07)
        np.testing.assert_array_equal(adapter.predictions(),restored.predictions())


def test_checkpoint_contains_predictions_without_fabricated_softmax_logits():
    adapter=SettingsTokenizerBandit(configs());weights=adapter.weights_dict()
    assert 'predictions' in weights and 'logits' not in weights
    assert weights['predictions'].shape==(1000,) and weights['recent_steps']==0
    with pytest.raises(RuntimeError):adapter.probabilities()
    adapter.sample()
    with pytest.raises(RuntimeError):adapter.weights_dict()


def test_staged_trainer_keeps_the_original_scoring_and_update_loop():
    files=prepared_sources();text=files['scripts/train_joint_smollm_input.py'].decode()
    root=Path(__file__).resolve().parents[1]
    original=(root/'runtime/anytok_pool_raw_bpb_rl_20k/scripts/train_joint_smollm_input.py').read_text()
    start='    for update in range(start_update, stop_update):'
    assert text[text.index(start):]==original[original.index(start):]
    assert 'from hf_hypernet.hardness_policy import SettingsTokenizerBandit' in text
    assert 'predictor_source_sha256' in text
    assert files['scripts/hardness_predictor.py']==(root/'scripts/hardness_predictor.py').read_bytes()


def test_monitor_uses_observed_choice_shares_not_conditional_argmax_probability(tmp_path):
    record=dict(stage='train',step=2,tokenizer_config_index=0,policy_action_probability=1.,
        selector_recent_effective_tokenizers=2.,selector_recent_largest_share=.5,
        selector_unique_tokenizers=2,selector_recent_entropy=float(np.log(2)),
        selector_exploration_visits=1,selector_squared_error=.001)
    (tmp_path/'train.log').write_text(json.dumps(record)+'\n')
    (tmp_path/'pool.json').write_text(json.dumps(dict(configs=configs())))
    result=LearningReader(tmp_path).read(dict(logs='train.log',baseline_logs='none*.log',
        plan='pool.json',snapshots='snapshots.json',selector_kind='hardness_predictor',baseline_required=False))
    assert result['policy_measurement']=='recent_actual_selection_frequencies'
    assert result['latest_policy']['policy_max_probability']==.5
    assert result['latest_policy']['coverage_visits']==1
