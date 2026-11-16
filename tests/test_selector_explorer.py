"""The visual replay must use pre-choice state and every original tokenizer."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from monitoring.selector_explorer import SelectorExplorer


@pytest.fixture(scope='module')
def explorer():
    return SelectorExplorer(ROOT)


def test_metadata_and_frames_do_not_alias_away_coverage_steps(explorer):
    meta=explorer.metadata()
    assert len(meta['tokenizers'])==1000
    assert sorted(meta['order'])==list(range(1000))
    assert meta['steps'][:5]==[1,25,50,75,100]
    assert meta['steps'][-1]==20000
    assert sum((step-1)%4==0 for step in meta['steps'][1:])==200


@pytest.mark.parametrize('scenario',['fixed','gradual','abrupt','practice_tokenizer','practice_family'])
def test_worlds_match_saved_trace_at_change_boundaries_and_late_training(explorer,scenario):
    for frame in [0,1,2,396,400,401,460,799,800]:
        data=explorer.frame(scenario,43,frame)
        for method,row in data['methods'].items():
            trace=explorer.trace(scenario,method,43);step=data['step']
            assert len(row['means'])==len(row['counts'])==1000
            assert sum(row['counts'])==step-1
            assert row['selected']==trace['actions'][step-1]
            assert row['observed_reward']==trace['rewards'][step-1]
            assert row['chosen_mean']==pytest.approx(trace['chosen_mean'][step-1],abs=1e-14)
            assert row['missed_hardness']==pytest.approx(trace['oracle_mean'][step-1]-trace['chosen_mean'][step-1],abs=1e-14)
            if row['beliefs'] is not None:assert len(row['beliefs'])==1000


def test_observation_is_revealed_before_it_updates_visits_or_beliefs(explorer):
    data=explorer.frame('fixed',43,0)
    for row in data['methods'].values():
        assert sum(row['counts'])==0
        assert row['visited_before_step']==0
    pred=data['methods']['predictor']
    assert pred['mode']=='coverage exploration'
    if pred['beliefs'] is not None:np.testing.assert_array_equal(pred['beliefs'],np.zeros(1000))
    later=explorer.frame('fixed',43,1)['methods']['predictor']
    assert later['mode']=='coverage exploration'
    assert sum(later['counts'])==24


def test_practice_worlds_are_specific_to_each_selectors_history(explorer):
    data=explorer.frame('practice_family',43,800)
    old=data['methods']['reinforce'];new=data['methods']['predictor']
    assert old['selected']==443
    assert old['missed_hardness']>.05
    assert new['missed_hardness']<.005
    assert not np.array_equal(old['means'],new['means'])


@pytest.mark.parametrize('scenario,seed,frame',[('../outside',43,0),('fixed',42,0),('fixed',43,-1),('fixed',43,801)])
def test_unknown_input_cannot_choose_files_or_arbitrary_frames(explorer,scenario,seed,frame):
    with pytest.raises(ValueError):explorer.frame(scenario,seed,frame)
