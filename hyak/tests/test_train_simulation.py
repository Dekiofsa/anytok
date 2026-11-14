"""Complete 20,000-update runs of both trainers (fake trainer, real wrapper) with stops, hard kills and resumes.

Each ``run_once`` is one Slurm job segment in a fresh "process".  The fake trainer saves with the
pinned cadence (uniform: a full restart every 500 + at stops, pruning off-grid restarts itself; RL:
step 0 full, weights-only every 500, full at multiples of 2,000 and at stops) and checks between
updates that a verified full restart exists; the hooks' retention steps are checked before and after
each run.  The end state must be the original retention: 40 grid checkpoints, 10 full restarts at
multiples of 2,000, 30 weights-only, every manifest verified and ``final_audit.json`` written.
"""
from __future__ import annotations

import pytest

from hyak.common.io import read_json, sha256_file
from hyak.common.paths import FINAL_STEP
from hyak.train import monitor as M
from hyak.train import pairing
from hyak.train import retention as R
from hyak.train import run_segment as S
from hyak.tests import test_train_support as T


@pytest.fixture
def world(make_tmp_config, repo_root, monkeypatch):
    world = T.build_world(make_tmp_config, repo_root)
    T.INVARIANT.update(checks=0, enabled=True)
    for name in ('compact_grid_checkpoints', 'prune_after_full_restart'):
        original = getattr(R, name)

        def checked(run_dir, *args, _original=original, **kwargs):
            run = 'uniform' if run_dir == world.paths.run_dir('uniform') else 'rl'
            T.restart_available(run_dir, run)
            result = _original(run_dir, *args, **kwargs)
            T.restart_available(run_dir, run)
            T.INVARIANT['checks'] += 1
            return result

        monkeypatch.setattr(R, name, checked)
    yield world
    T.INVARIANT.update(enabled=False)


def expected_grid(run_dir):
    kinds = T.step_kinds(run_dir)
    assert sorted(kinds) == R.grid_steps()
    assert all(kinds[s] == (R.FULL if s % 2000 == 0 else R.WEIGHTS_ONLY) for s in kinds), kinds
    assert sum(k == R.FULL for k in kinds.values()) == 10 and sum(k == R.WEIGHTS_ONLY for k in kinds.values()) == 30
    for step in R.grid_steps():
        R.verify_checkpoint(run_dir, step, 'uniform' if 'uniform' in run_dir.name else 'rl')


def train_steps(run_dir):
    return {e['step'] for e in T.trainer_events(run_dir) if e['stage'] == 'train'}


def test_uniform_full_run_with_stops_kills_and_resumes(world, monkeypatch):
    paths = world.paths
    run_dir = paths.run_dir('uniform')
    go = lambda **kw: T.run_once(world, 'uniform', monkeypatch, **kw)  # noqa: E731

    assert go()['outcome'] == 'bootstrap_done'
    config_sha = sha256_file(run_dir / 'run_config.json')
    record_sha = sha256_file(run_dir / R.SEGMENT_RECORD)
    assert go(control={'stop_at': [2300]})['step'] == 2300
    assert T.step_kinds(run_dir) == {500: R.WEIGHTS_ONLY, 1000: R.WEIGHTS_ONLY, 1500: R.WEIGHTS_ONLY,
                                     2000: R.FULL, 2300: R.FULL}
    assert go(control={'stop_at': [2500]})['step'] == 2500           # a stop exactly on the grid
    assert T.step_kinds(run_dir)[2500] == R.FULL and 2300 not in T.step_kinds(run_dir)
    with pytest.raises(T.FakeHardKill):                              # killed while writing step 3000
        go(control={'crash': {'step': 3000, 'where': 'in_write'}})
    assert list(run_dir.glob('.writing-*')) and T.restart_available(run_dir, 'uniform') == 2500
    assert go(control={'stop_at': [4321]})['step'] == 4321
    assert not list(run_dir.glob('.writing-*'))
    with pytest.raises(T.FakeHardKill):                              # killed between save and manifest
        go(control={'crash': {'step': 5000, 'where': 'after_replace'}})
    # the renamed step-005000 holds exactly a full restart's files: the next segment adopts it by itself
    with pytest.raises(T.FakeHardKill):                              # killed during an update
        go(control={'kill_at': 6700})
    adopted = [e for e in T.monitor_events(world, 'train-uniform') if e.get('event') == 'adopted_restart']
    assert [e['step'] for e in adopted] == [5000]
    assert T.restart_available(run_dir, 'uniform') == 6500
    assert go(control={'stop_at': [7600]})['step'] == 7600
    assert T.step_kinds(run_dir)[7500] == R.WEIGHTS_ONLY
    with pytest.raises(T.FakeHardKill):   # 7600 was pruned before the 8000 save, which then died
        go(control={'crash': {'step': 8000, 'where': 'in_write'}})
    assert T.restart_available(run_dir, 'uniform') == 6000
    assert [s for s, _ in R.step_dirs(run_dir) if s > 6000] == [6500, 7000, 7500]   # weights-only, newer
    outcome = go(control={'stop_at': [12345]})                       # moved aside, then replayed
    assert outcome['step'] == 12345
    moved = [e for e in T.monitor_events(world, 'train-uniform') if e.get('event') == 'moved_interrupted']
    assert moved and moved[-1]['steps'] == [6500, 7000, 7500] and moved[-1]['restart_step'] == 6000
    assert go(control={'stop_at': [19999]})['step'] == 19999
    assert S.should_start('uniform')['reason'] == 'resume'
    final = go()
    assert final['outcome'] == 'complete' and final['step'] == FINAL_STEP

    expected_grid(run_dir)
    audit = read_json(run_dir / R.FINAL_AUDIT)
    assert (audit['checkpoint_count'], audit['full_checkpoint_count'], audit['final_step']) == (40, 10, 20000)
    assert audit['input_manifest_sha256'] == sha256_file(paths.input_manifest) and audit['extra_step_dirs'] == []
    interrupted = sorted(p.name for p in (run_dir / 'interrupted').glob('*/*'))
    assert interrupted == ['step-006500', 'step-007000', 'step-007500']
    assert not list(run_dir.glob('.writing-*'))
    events = T.trainer_events(run_dir)
    assert events[-1] == {'stage': 'paused_at', 'step': FINAL_STEP}
    assert train_steps(run_dir) == set(range(1, FINAL_STEP + 1))
    assert sha256_file(run_dir / 'run_config.json') == config_sha
    assert sha256_file(run_dir / R.SEGMENT_RECORD) == record_sha
    assert pairing.verify_uniform_unchanged(paths)['uniform_full_restart_step_at_freeze'] == 1
    assert T.INVARIANT['checks'] > 20000

    assert go() == {'outcome': 'already_complete', 'step': FINAL_STEP, 'final_audit': audit}
    assert S.should_start('uniform') == {'run': 'uniform', 'newest_full_step': FINAL_STEP, 'start': False,
                                         'reason': 'complete'}
    status = M.Monitor('uniform', paths, world.cfg).refresh(force=True)
    assert (status['state'], status['stage'], status['step'], status['eta_seconds']) == ('complete', 'complete',
                                                                                          FINAL_STEP, 0)


def test_rl_full_run_with_stops_kills_and_resumes(world, monkeypatch):
    paths = world.paths
    run_dir = paths.run_dir('rl')
    T.run_once(world, 'uniform', monkeypatch)                         # the pair (D6)
    go = lambda **kw: T.run_once(world, 'rl', monkeypatch, **kw)  # noqa: E731

    first = go(control={'stop_at': [1234]})
    assert first['outcome'] == 'paused' and first['step'] == 1234
    assert T.step_kinds(run_dir) == {500: R.WEIGHTS_ONLY, 1000: R.WEIGHTS_ONLY, 1234: R.FULL}   # step 0 deleted
    pair = [e for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'initial_eval_pairing']
    # D8: the corrected paired cuts make the RL step-0 evaluation differ from the uniform run's by design;
    # only the shared sanity tokenizers are checked (no warning)
    assert pair == [pair[0]] and pair[0]['initial_eval_equal'] is False and pair[0]['sanity_tokenizers_equal'] is True
    receipts = [e for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'paired_protocol_verified']
    assert [e['step'] for e in receipts] == [1234]
    assert (run_dir / 'heldout_tokenizers.json').read_bytes() == paths.uniform_heldout.read_bytes()
    assert go(control={'stop_at': [2500]})['step'] == 2500
    assert T.step_kinds(run_dir) == {500: R.WEIGHTS_ONLY, 1000: R.WEIGHTS_ONLY, 1500: R.WEIGHTS_ONLY,
                                     2000: R.FULL, 2500: R.FULL}
    with pytest.raises(T.FakeHardKill):                              # 3000 and 3500 exist, newest restart 2500
        go(control={'kill_at': 3700})
    with pytest.raises(T.FakeHardKill):
        go(control={'crash': {'step': 4000, 'where': 'after_replace'}})
    assert go(control={'stop_at': [4500]})['step'] == 4500           # step 4000 adopted automatically
    assert [e['step'] for e in T.monitor_events(world, 'train-rl') if e.get('event') == 'adopted_restart'] == [4000]
    assert T.step_kinds(run_dir)[2500] == R.WEIGHTS_ONLY             # downgraded after the 4500 restart
    assert go(control={'stop_at': [19999]})['step'] == 19999
    final = go()
    assert final['outcome'] == 'complete' and final['step'] == FINAL_STEP

    expected_grid(run_dir)
    audit = read_json(run_dir / R.FINAL_AUDIT)
    assert (audit['checkpoints'], audit['final_step'], audit['full_checkpoint_count']) == (40, 20000, 10)
    assert audit['extra_step_dirs'] == []
    assert sorted(p.name for p in (run_dir / 'interrupted').glob('*/*')) == ['step-003000', 'step-003500']
    state = S.TrainerAdapter(paths.code_dir('rl')).load_state(run_dir / 'step-020000' / 'state.pt')
    assert state['bandit']['updates'] == FINAL_STEP and int(state['bandit']['counts'].sum()) == FINAL_STEP
    manifest = read_json(run_dir / 'step-019500' / 'manifest.json')
    assert manifest['kind'] == R.WEIGHTS_ONLY and set(manifest['files']) == R.weight_files('rl')
    assert manifest['policy_sha256'] == sha256_file(paths.code_dir('rl') / R.POLICY_REL)
    assert train_steps(run_dir) == set(range(1, FINAL_STEP + 1))
    assert T.trainer_events(run_dir)[-1] == {'stage': 'paused_at', 'step': FINAL_STEP}
    assert go()['outcome'] == 'already_complete'
    assert T.INVARIANT['checks'] > 20000
