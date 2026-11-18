import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hardness_predictor import _features
from synthetic_student import SyntheticStudent,mixed_probabilities,update_policy


def environment(scenario):
    configs=[dict(algorithm=a) for a in ['unigram','wordpiece','bpe','unigram']]
    rewards=np.tile(np.array([.10,.08,.06,.09])[:,None],(1,32))
    rewards+=np.linspace(-.02,.02,32)[None,:]
    calibration=dict(start_gap=.10,end_gap=.08,selected_visits=10)
    return SyntheticStudent(rewards,configs,calibration,scenario,43,steps=3000)


def test_frozen_preserves_every_measured_noise_value_and_pre_update_rewards():
    env=environment('frozen');before=env.means()
    for step in range(1,30):
        value=env.observe(0,step)
        assert value==pytest.approx(before[0]+env.noise[0,env.batches[step-1]])
    np.testing.assert_array_equal(before,env.means())


def test_plateau_rule_matches_declared_calibration_and_does_not_decay_to_zero():
    env=environment('individual');before=env.means()
    for step in range(1,11):env.observe(0,step)
    assert env.means()[0]==pytest.approx(.08)
    np.testing.assert_array_equal(env.means()[1:],before[1:])
    env.visits[0]=100000
    assert env.means()[0]==pytest.approx(.05)
    assert np.all(env.means()>=env.floor)


def test_family_rule_transfers_improvement_only_to_declared_algorithm():
    env=environment('family');before=env.means()
    for step in range(1,11):env.observe(0,step)
    after=env.means()
    assert after[0]==pytest.approx(.08)
    assert after[3]==pytest.approx(.09*.8)
    np.testing.assert_array_equal(after[1:3],before[1:3])


def test_paired_noise_is_independent_of_the_selected_action_history():
    a=environment('individual');b=environment('family')
    np.testing.assert_array_equal(a.batches,b.batches)
    for step in range(1,30):a.observe(0,step);b.observe(1,step)
    np.testing.assert_array_equal(a.batches,b.batches)
    assert a.means()[0]!=b.means()[0]


def configs():
    names=_features.DOMAINS
    return [dict(algorithm=a,pretokenizer='regex',vocab_size=64000,
        noise_per_million=0 if a=='bpe' else None,domain_bytes={n:1 for n in names})
        for a in ['bpe','wordpiece','unigram']]


def test_zero_coverage_is_the_exact_original_update():
    a=_features.SettingsTokenizerBandit(configs(),seed=43,entropy_weight=0.)
    b=_features.SettingsTokenizerBandit(configs(),seed=43,entropy_weight=0.)
    for action,reward in [(0,.1),(2,.2),(1,.05),(0,.11)]:
        a.update(action,reward);update_policy(b,action,reward,0.)
    for k,v in a.network.state_dict().items():torch.testing.assert_close(v,b.network.state_dict()[k],rtol=0,atol=0)
    assert a.baseline==b.baseline and a.counts.tolist()==b.counts.tolist()


def test_coverage_gradient_is_for_actual_mixture_not_naive_log_softmax():
    policy=_features.SettingsTokenizerBandit(configs(),seed=43,entropy_weight=0.)
    for action,reward in [(0,.1),(2,.2),(1,.05)]:policy.update(action,reward)
    coverage=.25;p,q=mixed_probabilities(policy,coverage)
    np.testing.assert_allclose(q,.75*p+.25/3)
    assert (q>=.25/3).all()
    action=2;advantage=.3-policy.baseline
    logits=policy.logits
    logq=torch.log(.75*logits.softmax(0)+.25/3)
    expected=torch.autograd.grad(-advantage*logq[action],list(policy.network.parameters()))
    update_policy(policy,action,.3,coverage)
    for param,gradient in zip(policy.network.parameters(),expected):
        torch.testing.assert_close(param.grad,gradient,rtol=1e-12,atol=1e-14)
