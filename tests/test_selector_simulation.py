"""Verify the simulator's counterfactual accounting and empirical reward law."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from selector_simulation import SimulatedRewards, load_inputs


def fixture():
    configs=[dict(algorithm=name) for name in ['unigram','wordpiece','bpe']]
    rewards=np.array([[.08,.10],[.06,.08],[.04,.06]])
    calibration=dict(observed_drop=.02,elapsed_steps=1500,decay_per_visit=np.log(2)/10)
    return configs,rewards,calibration


def environment(scenario,seed=43):
    c,r,k=fixture()
    return SimulatedRewards(r,c,k,scenario,seed)


def test_fixed_environment_preserves_all_empirical_batch_rewards():
    env=environment('fixed');c,r,k=fixture()
    for step in range(1,100):
        for action in range(3):
            assert env.observe(action,step)==pytest.approx(r[action,env.batches[step-1]],abs=1e-16)
    np.testing.assert_allclose(env.means(100),r.mean(axis=1),rtol=0,atol=0)


def test_changing_means_are_continuous_or_abrupt_at_declared_boundary():
    gradual=environment('gradual');base=gradual.means(1)
    np.testing.assert_array_equal(gradual.means(10000),base)
    assert gradual.means(10750)[0]==pytest.approx(base[0]-.01)
    assert gradual.means(11500)[0]==pytest.approx(base[0]-.02)
    np.testing.assert_array_equal(gradual.means(20000),gradual.means(11500))
    abrupt=environment('abrupt')
    np.testing.assert_array_equal(abrupt.means(9999),base)
    assert abrupt.means(10000)[0]==pytest.approx(base[0]-.04)
    np.testing.assert_array_equal(abrupt.means(10000)[1:],base[1:])


def test_practice_changes_only_declared_targets_and_rewards_are_pre_update():
    c,r,k=fixture();c.append(dict(algorithm='unigram'));r=np.vstack([r,r[0]])
    for scenario in ['practice_tokenizer','practice_family']:
        env=SimulatedRewards(r,c,k,scenario,43);base=env.means(1)
        for step in range(1,11):
            means=env.means(step)
            reward=env.observe(0,step)
            assert reward==pytest.approx(means[0]+env.residuals[0,env.batches[step-1]])
        changed=env.means(11)
        assert changed[0]==pytest.approx(base[0]/2)
        assert changed[3]==pytest.approx(base[3]/2 if scenario=='practice_family' else base[3])
        np.testing.assert_array_equal(changed[1:3],base[1:3])


def test_methods_share_noise_sequence_without_sharing_action_dependent_state():
    a=environment('practice_tokenizer');b=environment('practice_tokenizer')
    np.testing.assert_array_equal(a.batches,b.batches)
    for step in range(1,11):a.observe(0,step);b.observe(1,step)
    assert a.means(11)[0]<b.means(11)[0]
    assert b.means(11)[1]<a.means(11)[1]


def test_complete_real_table_matches_published_values():
    configs,rewards,calibration=load_inputs()
    assert len(configs)==1000 and rewards.shape==(1000,32)
    assert calibration['selected_visits']==1381
    assert calibration['observed_drop']==pytest.approx(.012486762790812657,abs=1e-15)
    assert np.isfinite(rewards).all() and calibration['causal'] is False
