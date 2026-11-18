from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from generalized_student import GeneralizedStudent,worlds,group_ids,reference_rate
from synthetic_student import SyntheticStudent


def inputs():
    configs=[dict(algorithm=a,pretokenizer=p) for a,p in
        [('unigram','regex'),('wordpiece','regex'),('bpe','line256'),('unigram','no-regex')]]
    rewards=np.tile(np.array([.10,.08,.06,.09])[:,None],(1,32))+np.linspace(-.02,.02,32)[None,:]
    return configs,rewards,dict(start_gap=.10,end_gap=.08,selected_visits=10)


def world(**kw):
    w=dict(worlds()[1]);w.update(kw);return w


@pytest.mark.parametrize('scenario,sharing',[('individual',0.),('family',1.)])
def test_reference_endpoints_preserve_prior_experiment_exactly(scenario,sharing):
    c,r,k=inputs();a=SyntheticStudent(r,c,k,scenario,43,200)
    b=GeneralizedStudent(r,c,k,world(sharing=sharing,speed=1.,budget='extra'),43,200)
    for step in range(1,201):
        np.testing.assert_array_equal(a.means(),b.means())
        action=(step*7)%4
        assert a.observe(action,step)==b.observe(action,step)


def test_fixed_practice_budget_really_distributes_one_unit():
    c,r,k=inputs()
    for sharing in [0.,.01,.1,1.]:
        for action in range(4):
            env=GeneralizedStudent(r,c,k,world(sharing=sharing,budget='shared'),43)
            env.observe(action,1)
            assert env.practices().sum()==pytest.approx(1.)
            if action==0:assert env.practices()[1]==0 and env.practices()[2]==0


def test_partial_sharing_has_exact_declared_strength():
    c,r,k=inputs();env=GeneralizedStudent(r,c,k,world(sharing=.1,budget='extra'),43)
    env.observe(0,1)
    np.testing.assert_allclose(env.practices(),[1,0,0,.1])


def test_global_full_sharing_preserves_rank_order():
    c,r,k=inputs();env=GeneralizedStudent(r,c,k,world(sharing=1.,grouping='global',speed=4.),43)
    order=np.argsort(env.means())
    for step in range(1,101):env.observe(step%4,step)
    np.testing.assert_array_equal(np.argsort(env.means()),order)
    np.testing.assert_allclose(env.means()/env.base,np.full(4,env.means()[0]/env.base[0]))


def test_noise_and_floor_can_change_without_changing_rate_or_practice():
    c,r,k=inputs();a=GeneralizedStudent(r,c,k,world(floor=0,noise=0),43)
    b=GeneralizedStudent(r,c,k,world(floor=.9,noise=2),43)
    assert a.rate==b.rate
    np.testing.assert_array_equal(a.batches,b.batches)
    before=a.means();assert a.observe(0,1)==before[0]
    assert b.observe(0,1)==pytest.approx(b.base[0]+2*b.residuals[0,b.batches[0]])


def test_learning_curves_share_the_same_initial_slope_but_differ_later():
    c,r,k=inputs();means={}
    for curve in ['linear','exponential','inverse']:
        env=GeneralizedStudent(r,c,k,world(sharing=0.,speed=1.,curve=curve),43)
        for step in range(1,11):env.observe(0,step)
        means[curve]=env.means()[0]
        assert env.rate==reference_rate(k)
    assert means['linear']<means['exponential']<means['inverse']


def test_worlds_are_frozen_unique_and_all_static_checks_exist():
    grid=worlds();assert len(grid)==69
    keys=[tuple(sorted((k,v) for k,v in w.items() if k not in ['id','study','label'])) for w in grid]
    assert len(set(keys))==len(grid)
    for noise,layout in [(0.,'measured'),(1.,'measured'),(2.,'measured'),(1.,'flat'),(1.,'shuffled')]:
        assert any(w['speed']==0 and w['noise']==noise and w['starting_gaps']==layout for w in grid)


@pytest.mark.parametrize('method',['reinforce','coverage','uniform','current_hardest'])
def test_short_real_pool_trace_passes_independent_student_audit(tmp_path,monkeypatch,method):
    import run_student_generalization as runner
    import report_student_generalization as reporter
    from selector_simulation import load_inputs
    import torch
    torch.set_num_threads(1)
    actual=load_inputs()
    monkeypatch.setattr(runner,'STATE',tmp_path)
    monkeypatch.setattr(reporter,'STATE',tmp_path)
    monkeypatch.setattr(runner,'INPUTS',actual,raising=False)
    monkeypatch.setattr(reporter,'INPUTS',actual,raising=False)
    monkeypatch.setattr(runner,'PLAN',dict(plan_sha256='test-plan'),raising=False)
    monkeypatch.setattr(runner,'STEPS',200)
    config=world(grouping='random',sharing=.1,speed=64.,budget='shared',
        noise=2.,starting_gaps='shuffled',floor=.9)
    config['id']='integration'
    result=runner.run_cell((config,method,48))
    folder=tmp_path/'cells'/runner.key((config,method,48))
    verified=reporter.audit_cell((config,str(folder),'test-plan'))
    assert verified['steps']==200
    assert verified['rewards_student_transitions_and_ranks_verified']
    if method=='current_hardest':assert result['final_choice_gap']==0
