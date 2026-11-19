import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pool_generalization import digest
from report_predictor_trajectory import analyze, audit
from test_four_way_8k import scored_panel


@pytest.fixture
def trajectory(scored_panel):
    parent=scored_panel['plan']
    parent['models']=[dict(name=n,manifest=dict(step=8000)) for n in ['uniform','predictor']]
    plan=dict(parent=parent,parent_plan_sha256=digest(parent),panel_sha256=parent['panel_sha256'],models=[])
    data=dict(plan=plan,rows=copy.deepcopy(scored_panel['rows']),models=[])
    for step in [2000,4000,6000,8000]:
        for name in ['uniform','predictor']:
            plan['models'].append(dict(name=name,step=step,manifest=dict(step=step),cached=step==8000))
    for step in [2000,4000,6000,8000]:
        for name in ['uniform','predictor']:
            m=copy.deepcopy(scored_panel['models'][name])
            m.update(name=name,step=step,cached=step==8000)
            key=dict(step=step,model=name,plan_sha256=digest(parent) if step==8000 else digest(plan))
            m['canonical'].update(key)
            for r in m['tokenizers']:r.update(key)
            data['models'].append(m)
    data['preparation']=dict(plan_sha256=digest(plan))
    return data


def test_full_trajectory_keeps_lm_and_transfer_improvements_separate(trajectory):
    report=analyze(trajectory,replicates=16)
    for info in report['splits'].values():
        np.testing.assert_allclose(info['pool_mean_improvement'],[.03]*4,atol=1e-12)
        np.testing.assert_allclose(info['pool_gain_minus_8k_ci95'],np.zeros((2,4)),atol=1e-12)
        np.testing.assert_allclose(info['absolute_improvement_bpb'],np.full((4,62),-.27),atol=1e-12)
    assert report['audit']['comparisons']==496


def test_wrong_checkpoint_key_is_rejected(trajectory):
    trajectory['models'][0]['canonical']['step']=4000
    with pytest.raises(ValueError,match='provenance'):audit(trajectory)


def test_earlier_advantage_is_compared_with_8k_on_paired_text(trajectory):
    import math
    early=next(m for m in trajectory['models'] if m['name']=='predictor' and m['step']==2000)
    for record in early['tokenizers']:
        for losses,row in zip(record['transferred'],trajectory['rows']):
            reduction=.02*row['continuation_bytes']*math.log(2)/32
            losses[:]=[v-reduction for v in losses]
    report=analyze(trajectory,replicates=16)
    for info in report['splits'].values():
        np.testing.assert_allclose(info['pool_mean_improvement'],[.05,.03,.03,.03],atol=1e-12)
        np.testing.assert_allclose(np.array(info['pool_gain_minus_8k_ci95'])[:,0],[.02,.02],atol=1e-12)


def test_incomplete_coverage_is_rejected(trajectory):
    trajectory['models'][2]['tokenizers'].pop()
    with pytest.raises(ValueError,match='coverage'):audit(trajectory)


def test_changed_continuation_is_rejected(trajectory):
    trajectory['rows'][0]['canonical_continuation_ids'][0]+=1
    with pytest.raises(ValueError,match='targets changed'):audit(trajectory)


def test_cached_8k_requires_its_original_plan_hash(trajectory):
    trajectory['models'][-1]['canonical']['plan_sha256']=digest(trajectory['plan'])
    with pytest.raises(ValueError,match='provenance'):audit(trajectory)


def test_monitor_keeps_worker_progress(tmp_path,monkeypatch):
    import json
    import run_predictor_trajectory as runner
    monkeypatch.setattr(runner,'STATE',tmp_path)
    workers=[dict(id='uniform-002000',completed=10,total=62,state='scoring')]
    runner.status('Scoring',134,workers=workers)
    row=json.loads((tmp_path/'status.json').read_text())
    assert row['completed']==134 and row['total']==496 and row['workers']==workers
