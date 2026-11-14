"""The shipped memory adapter must retain outputs, gradients and RNG."""
import copy
from pathlib import Path
import sys
import json
import os
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'runtime/anytok_pool_20k'))
from hf_hypernet.identity_warmup import build_input_hypernet, freeze_rescalers
from pool_training_memory import apply_gpu_memory_fix, backward_hypernetwork
from run_raw_bpb_training import run_and_monitor


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA execution fidelity')
@pytest.mark.parametrize('deterministic', [True, False])
@pytest.mark.parametrize('autocast', [False, True])
def test_shipped_memory_adapter_preserves_joint_gradients_and_rng(deterministic, autocast):
    torch.set_num_threads(1)
    torch.manual_seed(67)
    original = build_input_hypernet(96, 64, hidden_width=64, intermediate_width=128,
                                   layers=2, attention_heads=8).cuda()
    freeze_rescalers(original)
    original.model.activation_checkpointing = True
    original.activation_checkpoint_input = True
    modified = copy.deepcopy(original)
    apply_gpu_memory_fix(modified, chunk_rows=31)
    assert tuple(original.state_dict()) == tuple(modified.state_dict())
    forms = torch.randint(0, 97, (40, 10), device='cuda')
    priors = torch.randn(40, device='cuda')
    source = torch.randn(96, 64, device='cuda')
    embedding_gradient = torch.randn(40, 64, device='cuda')
    records = []
    for model, optimized in [(original, False), (modified, True)]:
        torch.cuda.manual_seed(814)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            output, _, _ = model(forms, priors, source, None, deterministic=deterministic)
        lexical = output.float().square().mean()
        backward_hypernetwork(output, embedding_gradient, lexical, optimized)
        records.append((output.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                                          if p.grad is not None}, torch.cuda.get_rng_state()))
    assert torch.equal(records[0][0], records[1][0])
    assert torch.equal(records[0][2], records[1][2])
    assert records[0][1].keys() == records[1][1].keys()
    for name in records[0][1]:
        torch.testing.assert_close(records[0][1][name], records[1][1][name], rtol=1e-6, atol=1e-6, msg=name)
    with pytest.raises(ValueError):
        apply_gpu_memory_fix(modified)


@pytest.mark.parametrize('exit_code', [0, 1])
def test_launcher_status_reports_completed_and_failed_children(tmp_path, exit_code):
    output = tmp_path / 'run'
    code = ("from pathlib import Path;import json,sys;"
            f"Path({str(output)!r}).mkdir();"
            "print(json.dumps(dict(stage='train',step=2)),flush=True);"
            "print(json.dumps(dict(stage='paused_at',step=2)),flush=True);"
            f"sys.exit({exit_code})")
    assert run_and_monitor([sys.executable, '-c', code],
                           SimpleNamespace(output=output, stop_at_update=2), dict(os.environ)) == exit_code
    status = json.loads((output / 'status.json').read_text())
    assert status['step'] == 2 and status['total'] == 2
    assert status['stage'] == ('complete' if exit_code == 0 else 'failed')
    assert status['returncode'] == exit_code and not status['workers']
