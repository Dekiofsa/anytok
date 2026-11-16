"""Retry a missing summary after payload verification without uploading again."""
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import tillicum_hardness_job as job
from test_hardness_predictor import configs


def test_verified_payload_can_repair_missing_summary_and_clear_error(tmp_path,monkeypatch):
    state=tmp_path/'state';inputs=tmp_path/'input';folder=state/'run/step-000500'
    folder.mkdir(parents=True);(state/'backups').mkdir();(inputs/'pool').mkdir(parents=True)
    (inputs/'pool/plan.json').write_text(json.dumps(dict(configs=configs())))
    counts=torch.zeros(1000,dtype=torch.int64);counts[:125]=1;counts[126]=375
    exploration=torch.zeros(1000,dtype=torch.int64);exploration[:125]=1
    torch.save(dict(counts=counts,recent_counts=counts,recent_steps=500,
        exploration_counts=exploration,predictions=torch.zeros(1000)),folder/'policy.pt')
    record=dict(step=500,revision='already-verified')
    (state/'backups/step-000500.json').write_text(json.dumps(record))
    (state/'backup-status.json').write_text(json.dumps(dict(stage='retrying',error='summary import failed')))
    monkeypatch.setattr(job,'STATE',state);monkeypatch.setattr(job,'INPUT',inputs)
    job.backup(None,folder)  # No API exists: a second upload would fail this test.
    report=json.loads((state/'policy-monitor.json').read_text())
    assert len(report['snapshots'])==1
    assert report['snapshots'][0]['kind']=='recent_actual_selection_frequencies'
    assert report['snapshots'][0]['exploration_visits']==125
    assert json.loads((state/'backup-status.json').read_text())['stage']=='verified'
    monkeypatch.setattr(torch,'load',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Unnecessary reload')))
    job.backup(None,folder)
    assert len(json.loads((state/'policy-monitor.json').read_text())['snapshots'])==1
