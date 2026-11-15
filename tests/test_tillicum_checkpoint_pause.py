import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pause_tillicum_regret import committed_checkpoint


def test_pause_waits_for_manifest_then_requires_full_restart_payload(tmp_path):
    assert committed_checkpoint(tmp_path, 8000) is None
    state = tmp_path / 'state.pt'
    state.write_bytes(b'complete optimizer and RNG state')
    manifest = dict(step=8000, kind='full_restart_checkpoint',
                    files={'state.pt': {'bytes': state.stat().st_size}})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert committed_checkpoint(tmp_path, 8000) == manifest
    state.write_bytes(b'incomplete')
    with pytest.raises(ValueError):
        committed_checkpoint(tmp_path, 8000)


@pytest.mark.parametrize('step,kind,files', [
    (7500, 'full_restart_checkpoint', {'state.pt': {'bytes': 1}}),
    (8000, 'weights_only_checkpoint', {'lm.pt': {'bytes': 1}}),
    (8000, 'full_restart_checkpoint', {}),
])
def test_pause_refuses_wrong_step_or_nonresumable_checkpoint(tmp_path, step, kind, files):
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(step=step, kind=kind, files=files)))
    with pytest.raises(ValueError):
        committed_checkpoint(tmp_path, 8000)
