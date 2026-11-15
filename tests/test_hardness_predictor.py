import copy
import io
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hardness_predictor import HardnessPredictor, CoverageSchedule


def configs():
    domains=['web','wikipedia','code','papers','math','qa','books','medical','legal']
    return [dict(index=i,seed=i,algorithm=['bpe','wordpiece','unigram'][i%3],
        pretokenizer='no-regex',vocab_size=64000,noise_per_million=200000 if i%3==0 else None,
        domain_bytes={name:200000000 if name=='web' else 0 for name in domains}) for i in range(1000)]


def test_full_20k_schedule_visits_every_tokenizer_five_times_without_warmup():
    schedule=CoverageSchedule();counts=np.zeros(1000,dtype=int);modes=[]
    for step in range(20000):
        index,mode=schedule.choose(np.arange(1000,dtype=float))
        assert (mode=='explore')==(step%4==0)
        if mode=='explore':counts[index]+=1
        else:assert index==999
        modes.append(mode)
    np.testing.assert_array_equal(counts,np.full(1000,5))
    assert modes.count('explore')==5000


def test_initialization_preserves_model_rng_and_has_same_parameter_count():
    before=torch.get_rng_state().clone()
    with patch.object(torch.cuda,'manual_seed_all') as reseed:
        learner=HardnessPredictor(configs())
    reseed.assert_not_called()
    assert torch.equal(before,torch.get_rng_state())
    assert sum(p.numel() for p in learner.network.parameters())==5441
    np.testing.assert_array_equal(learner.predictions(),np.zeros(1000))


def test_real_observation_updates_unvisited_matching_settings_once():
    learner=HardnessPredictor(configs())
    action=learner.select();index=action['index']
    neighbor=next(i for i in range(1000) if i!=index and i%3==index%3)
    loss=learner.observe(index,.09);values=learner.predictions()
    assert loss==pytest.approx(.09**2)
    assert abs(values[index]-.09)<.09
    assert values[neighbor]==pytest.approx(values[index],abs=1e-15)
    assert learner.counts[neighbor]==0 and learner.updates==1
    assert all(float(state['step'])==1 for state in learner.optimizer.state.values())


def test_network_optimizer_cycle_and_rng_resume_exactly():
    learner=HardnessPredictor(configs())
    for step in range(11):
        action=learner.select();learner.observe(action['index'],.03+.01*(step%4))
    memory=io.BytesIO();torch.save(learner.state_dict(),memory);memory.seek(0)
    restored=HardnessPredictor(configs())
    restored.load_state_dict(torch.load(memory,weights_only=True))
    for step in range(13):
        a=learner.select();b=restored.select();assert a==b
        reward=.04+.01*(step%5)
        assert learner.observe(a['index'],reward)==restored.observe(b['index'],reward)
        np.testing.assert_array_equal(learner.predictions(),restored.predictions())


def test_missing_reward_and_corrupt_resume_are_rejected():
    learner=HardnessPredictor(configs());action=learner.select()
    with pytest.raises(RuntimeError):learner.select()
    with pytest.raises(ValueError):learner.observe(action['index'],float('nan'))
    with pytest.raises(RuntimeError):learner.state_dict()
    learner.observe(action['index'],.07)
    state=copy.deepcopy(learner.state_dict());state['schedule']['steps']+=1
    with pytest.raises(ValueError):learner.load_state_dict(state)
    with pytest.raises(ValueError):HardnessPredictor(configs()[:100])
