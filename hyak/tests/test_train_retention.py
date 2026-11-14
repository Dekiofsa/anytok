"""hyak.train.retention: manifests, both retention hooks, pre-flight repair, audits, manual repairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hyak.common.io import read_json
from hyak.train import retention as R


def make_checkpoint(run_dir: Path, step: int, run: str = 'uniform', kind: str = R.FULL, manifest=True):
    """A checkpoint with distinct file contents and (optionally) its manifest, like the launcher tests did."""
    folder = run_dir / R.step_name(step)
    names = R.full_files(run) if kind == R.FULL else R.weight_files(run)
    files = {}
    for name in sorted(names):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = f'{step}:{name}'.encode()
        path.write_bytes(value)
        files[name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    if manifest:
        (folder / 'manifest.json').write_text(json.dumps({'step': step, 'kind': kind, 'files': files}))
    return folder


def code_dir(tmp_path: Path) -> Path:
    code = tmp_path / 'code'
    (code / 'scripts').mkdir(parents=True)
    (code / 'hf_hypernet').mkdir()
    (code / R.TRAINER_REL).write_text('trainer')
    (code / R.POLICY_REL).write_text('policy')
    return code


# --- the launcher's own scenario (tests/test_zett_modal_checkpoint_policy.py:33-64) ---------------------
def test_compaction_keeps_milestones_and_latest_then_cleans_source(tmp_path):
    run = tmp_path / 'run'
    for step in (500, 1000, 1500, 2000, 2500, 2600):
        make_checkpoint(run, step)
    assert R.compact_grid_checkpoints(run) == [500, 1000, 1500, 2500]
    for step in (500, 1000, 1500, 2500):
        result = R.verify_checkpoint(run, step, 'uniform', R.WEIGHTS_ONLY)
        assert result['files'] == 3
        assert not (run / R.step_name(step) / 'optimizer.pt').exists()
        assert not (run / R.step_name(step) / 'state.pt').exists()
    for step in (2000, 2600):
        R.verify_checkpoint(run, step, 'uniform', R.FULL)
    assert R.compact_grid_checkpoints(run, False) == [2600]
    assert not (run / 'step-002600').exists()
    R.verify_checkpoint(run, 2000, 'uniform', R.FULL)


def test_bad_weights_never_delete_restart_files(tmp_path):
    run = tmp_path / 'run'
    old = make_checkpoint(run, 500)
    make_checkpoint(run, 2000)
    (old / 'lm.pt').write_bytes(b'corrupt')
    with pytest.raises(R.CheckpointError, match='Weight checksum mismatch'):
        R.compact_grid_checkpoints(run)
    assert (old / 'optimizer.pt').exists() and (old / 'state.pt').exists()


def test_rl_prune_after_full_restart(tmp_path):
    run = tmp_path / 'run'
    make_checkpoint(run, 0, 'rl')
    make_checkpoint(run, 500, 'rl', R.WEIGHTS_ONLY)
    make_checkpoint(run, 1234, 'rl')         # off-grid restart from a stop
    make_checkpoint(run, 1500, 'rl')         # grid restart from a stop
    make_checkpoint(run, 2000, 'rl')         # milestone
    make_checkpoint(run, 2500, 'rl', R.WEIGHTS_ONLY)
    make_checkpoint(run, 4000, 'rl')         # the new full restart
    assert R.prune_after_full_restart(run, 4000) == {'deleted': [0, 1234], 'downgraded': [1500]}
    assert [s for s, _ in R.step_dirs(run)] == [500, 1500, 2000, 2500, 4000]
    assert R.verify_checkpoint(run, 1500, 'rl', R.WEIGHTS_ONLY)['files'] == 4
    R.verify_checkpoint(run, 2000, 'rl', R.FULL)
    # nothing newer than the restart is touched
    make_checkpoint(run, 4500, 'rl')
    assert R.prune_after_full_restart(run, 4000) == {'deleted': [], 'downgraded': []}


def test_verify_checkpoint_rejects_every_kind_of_damage(tmp_path):
    run = tmp_path / 'run'
    folder = make_checkpoint(run, 2000, 'rl')
    assert R.verify_checkpoint(run, 2000, 'rl', R.FULL)['kind'] == R.FULL
    with pytest.raises(R.CheckpointError):          # the uniform file set lacks policy.pt
        R.verify_checkpoint(run, 2000, 'uniform')
    with pytest.raises(R.CheckpointError, match='Invalid checkpoint manifest'):
        R.verify_checkpoint(run, 2000, 'rl', R.WEIGHTS_ONLY)
    (folder / 'policy.pt').write_bytes(b'corrupt')   # tests/test_pool_rl_bandit.py:66-79
    with pytest.raises(R.CheckpointError):
        R.verify_checkpoint(run, 2000, 'rl', R.FULL)
    (folder / 'manifest.json').unlink()
    with pytest.raises(R.ManifestMissing):
        R.verify_checkpoint(run, 2000, 'rl')


def test_manifest_writers_match_the_launcher_formats(tmp_path):
    code = code_dir(tmp_path)
    run = tmp_path / 'run'
    folder = make_checkpoint(run, 500, 'uniform', manifest=False)
    manifest = R.write_uniform_manifest(folder, 500, code)
    text = (folder / 'manifest.json').read_text()
    assert text == json.dumps(manifest, indent=2) + '\n'          # modal_anytok_pool_20k.py:308
    assert list(manifest) == ['step', 'kind', 'files', 'trainer_sha256']
    assert manifest['trainer_sha256'] == hashlib.sha256(b'trainer').hexdigest()
    rl_folder = make_checkpoint(tmp_path / 'rl', 500, 'rl', R.WEIGHTS_ONLY, manifest=False)
    rl_manifest = R.write_rl_manifest(rl_folder, 500, code)
    assert rl_manifest['kind'] == R.WEIGHTS_ONLY
    assert rl_manifest['policy_sha256'] == hashlib.sha256(b'policy').hexdigest()
    (rl_folder / 'manifest.json').unlink()
    (rl_folder / 'extra.bin').write_bytes(b'x')
    with pytest.raises(R.CheckpointError, match='unexpected checkpoint files'):
        R.write_rl_manifest(rl_folder, 500, code)


def test_hooks_call_the_original_then_record_and_prune(tmp_path):
    code = code_dir(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    events, calls = [], []

    def emit(name, **fields):
        events.append((name, fields))

    def original(output, step, *args, **kwargs):
        calls.append((step, args, kwargs))
        make_checkpoint(Path(output), step, 'uniform', manifest=False)

    hook = R.uniform_save_hook(original, run, code, emit=emit, before_manifest=lambda s: calls.append(('record', s)))
    for step in (500, 1000):
        hook(str(run), step, 'lm', 'hn', protected_steps=(500,))
    assert calls[0] == (500, ('lm', 'hn'), {'protected_steps': (500,)}) and calls[1] == ('record', 500)
    assert R.verify_checkpoint(run, 500, 'uniform', R.WEIGHTS_ONLY)  # compacted after the 1000 save
    assert ('compacted_checkpoints', {'steps': [500]}) in events
    assert [e[0] for e in events].count('committed_checkpoint') == 2
    with pytest.raises(R.CheckpointError, match='hook manages'):
        hook(str(tmp_path / 'elsewhere'), 1500)

    rl_run = tmp_path / 'rl'
    rl_run.mkdir()

    def rl_original(output, step, *args, bandit=None, full_restart=True, **kwargs):
        make_checkpoint(Path(output), step, 'rl', R.FULL if full_restart else R.WEIGHTS_ONLY, manifest=False)

    rl_hook = R.rl_save_hook(rl_original, rl_run, code, emit=emit)
    rl_hook(str(rl_run), 0, bandit='b', full_restart=True)
    rl_hook(str(rl_run), 500, bandit='b', full_restart=False)
    rl_hook(str(rl_run), 777, bandit='b', full_restart=True)
    assert [s for s, _ in R.step_dirs(rl_run)] == [500, 777]
    assert ('pruned_restarts', {'step': 777, 'deleted': [0], 'downgraded': []}) in events


def test_preflight_repairs_what_a_hard_kill_leaves(tmp_path):
    run = tmp_path / 'run'
    make_checkpoint(run, 2000)
    half = make_checkpoint(run, 2500)                         # compaction died after its manifest rewrite
    manifest = read_json(half / 'manifest.json')
    manifest['kind'] = R.WEIGHTS_ONLY
    manifest['files'] = {n: manifest['files'][n] for n in R.UNIFORM_WEIGHT_FILES}
    (half / 'manifest.json').write_text(json.dumps(manifest))
    partial = run / 'step-002300'                             # a trainer rmtree died half way
    (partial / 'hypernet').mkdir(parents=True)
    (partial / 'lm.pt').write_bytes(b'x')
    make_checkpoint(run, 3000)                                # newest full restart
    make_checkpoint(run, 3500, kind=R.WEIGHTS_ONLY)           # newer than the restart
    (run / '.writing-abc123').mkdir()
    (run / '.writing-abc123' / 'lm.pt').write_bytes(b'partial')
    result = R.preflight(run, 'uniform', label='T0')
    assert result.removed_writing == ['.writing-abc123']
    assert result.finished_compactions == [2500]
    assert not (half / 'state.pt').exists() and not (half / 'optimizer.pt').exists()
    assert result.finished_deletions == [2300]
    assert result.newest_full_step == 3000
    assert result.moved_interrupted == [3500]
    assert (run / 'interrupted' / 'T0' / 'step-003500' / 'manifest.json').is_file()
    assert [s for s, _ in R.step_dirs(run)] == [2000, 2500, 3000]
    # the same call again changes nothing
    again = R.preflight(run, 'uniform', label='T1')
    assert again.moved_interrupted == [] and again.newest_full_step == 3000


def test_preflight_refuses_unverifiable_states(tmp_path):
    run = tmp_path / 'run'
    make_checkpoint(run, 2000)
    newest = make_checkpoint(run, 2600, manifest=False)
    with pytest.raises(R.ManifestMissing, match='retention adopt --run uniform --step 2600'):
        R.preflight(run, 'uniform')
    make_checkpoint(run, 2600, manifest=True)
    (newest / 'state.pt').write_bytes(b'corrupt')
    with pytest.raises(R.CheckpointError, match='quarantine --run uniform --step 2600'):
        R.preflight(run, 'uniform')
    (run / 'step-junk').mkdir()
    with pytest.raises(R.CheckpointError, match='Unexpected entry'):
        R.preflight(run, 'uniform')
    assert R.preflight(tmp_path / 'absent', 'uniform').newest_full_step is None


def test_adopt_and_quarantine(tmp_path):
    code = code_dir(tmp_path)
    run = tmp_path / 'run'
    make_checkpoint(run, 2000)
    make_checkpoint(run, 2600, manifest=False)
    with pytest.raises(R.CheckpointError, match='hyak_segment.json'):
        R.adopt(run, 'uniform', 2600, code)
    (run / R.SEGMENT_RECORD).write_text('{}')
    R.adopt(run, 'uniform', 2600, code)
    assert R.preflight(run, 'uniform').newest_full_step == 2600
    with pytest.raises(R.CheckpointError, match='already exists'):
        R.adopt(run, 'uniform', 2600, code)
    target = R.quarantine(run, 2600, label='X')
    assert target == run / 'interrupted' / 'manual-X' / 'step-002600'
    assert R.preflight(run, 'uniform').newest_full_step == 2000


def _finished_grid(run_dir: Path, run: str):
    for step in R.grid_steps():
        make_checkpoint(run_dir, step, run, R.FULL if step % 2000 == 0 else R.WEIGHTS_ONLY)


def test_final_audits_write_the_launchers_receipts(tmp_path):
    manifest = tmp_path / 'input-manifest.json'
    manifest.write_text('{"files": {}}\n')
    uniform = tmp_path / 'uniform'
    _finished_grid(uniform, 'uniform')
    receipt = R.audit_final(uniform, 'uniform', manifest)
    assert {k: receipt[k] for k in ('checkpoint_count', 'full_checkpoint_count', 'final_step')} == \
        {'checkpoint_count': 40, 'full_checkpoint_count': 10, 'final_step': 20000}
    assert receipt['input_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt['extra_step_dirs'] == [] and R.audit_valid(uniform, 'uniform')
    rl = tmp_path / 'rl'
    _finished_grid(rl, 'rl')
    make_checkpoint(rl, 0, 'rl')
    receipt = R.audit_final(rl, 'rl')
    assert (receipt['checkpoints'], receipt['final_step'], receipt['full_checkpoint_count']) == (40, 20000, 10)
    assert receipt['extra_step_dirs'] == [0] and R.audit_valid(rl, 'rl')
    # a non-milestone full restart fails the uniform audit (exact kinds) ...
    make_checkpoint(uniform, 19500, 'uniform', R.FULL)
    with pytest.raises(R.CheckpointError):
        R.audit_final(uniform, 'uniform', manifest)
    # ... and a weights-only milestone fails both
    make_checkpoint(rl, 20000, 'rl', R.WEIGHTS_ONLY)
    with pytest.raises(R.CheckpointError):
        R.audit_final(rl, 'rl')


def test_final_audit_stops_on_request(tmp_path):
    rl = tmp_path / 'rl'
    _finished_grid(rl, 'rl')
    seen = []
    with pytest.raises(R.AuditInterrupted):
        R.audit_final(rl, 'rl', should_stop=lambda: seen.append(1) or len(seen) > 3)
    assert len(seen) == 4 and not (rl / R.FINAL_AUDIT).exists()


def test_signal_during_the_final_audit_ends_the_segment_cleanly(tmp_config):
    import torch
    from hyak.train import run_segment as S
    paths = tmp_config.paths
    run = paths.run_dir('uniform')
    _finished_grid(run, 'uniform')
    paths.input_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.input_manifest.write_text('{"files": {}}\n')
    torch.save({'step': 20000}, run / 'step-020000' / 'state.pt')
    signals = S.EarlySignals()
    signals.received = 15
    events = []
    outcome = S._finish_complete('uniform', paths, run, S.TrainerAdapter(paths.code_dir('uniform')),
                                 lambda name, **f: events.append(name), 20000, already=False, signals=signals)
    assert outcome == {'outcome': 'audit_interrupted', 'step': 20000} and events == ['final_audit_interrupted']
    assert not (run / R.FINAL_AUDIT).exists()


def test_retention_cli(tmp_config, capsys):
    run = tmp_config.paths.run_dir('uniform')
    make_checkpoint(run, 2000)
    assert R.main(['summary', '--run', 'uniform']) == 0
    assert json.loads(capsys.readouterr().out)['newest_full_step'] == 2000
    assert R.main(['verify', '--run', 'uniform']) == 0
    assert json.loads(capsys.readouterr().out)['kind'] == R.FULL
    with pytest.raises(SystemExit) as exc:
        R.main(['quarantine', '--run', 'uniform'])                 # usage: --step is required
    assert exc.value.code == 2
    assert R.main(['verify', '--run', 'uniform', '--step', '2500']) == 4   # a check failed (no such checkpoint)
    with R.WrapperLock(R.wrapper_lock_path(tmp_config.paths, 'uniform')):
        assert R.main(['verify', '--run', 'uniform']) == 75      # not now: a segment or repair holds the lock
        assert 'held by another process' in capsys.readouterr().err
