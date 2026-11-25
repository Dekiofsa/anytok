"""Raw-BPB held-out full-pool eval: the adapted workspace, accepted by his unchanged staging script and preflight."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from package_training_inputs import sha256
from pool_generalization import digest
import prepare_rawbpb_heldout_pool as prep

STEPS = sorted([*range(0, 20001, 500), 16409])
LABELS = [f'raw-bpb-{step:06d}' for step in STEPS]
# The modules his scorer imports at the top level; his restored workspace carries them.
MODULES = ['natural_transfer.py', 'package_training_inputs.py', 'pool_generalization.py', 'uniform_trajectory.py']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def passages():
    """156 passages with 32 continuation tokens each and 14,429 continuation bytes in all."""
    return [dict(id=f'doc-{i:03d}', prefix=f'prefix {i}', canonical_continuation_ids=[(32 * i + j) % 49152 for j in range(32)],
                 continuation_bytes=92 + (i < 77)) for i in range(156)]


def pool_index_bytes():
    tokenizers = [dict(config=dict(index=i, algorithm='bpe', pretokenizer='regex'), tokenizer_sha256=f'{i:064x}',
                       vocab_size_actual=32000 + i) for i in range(1000)]
    return json.dumps(dict(tokenizers=tokenizers)).encode()


def build_reference(tmp_path):
    """His workspace as restore_rift_pool_archive.py leaves it, with his own checkpoints and scores."""
    ref = tmp_path / 'reference'
    panel = passages()
    prefixes = {str(i): [[1 + (i + d) % 7] for d in range(156)] for i in range(1000)}
    code = ref / 'inbox/code'
    code.mkdir(parents=True)
    (code / 'model_stub.py').write_text('VALUE = 1\n')
    write_json(code / 'source-manifest.json', {'model_stub.py': sha256(code / 'model_stub.py')})
    write_json(ref / 'inbox/panel.json', panel)
    plan = dict(checkpoints=['predictor-008000', 'uniform-008000'],
                checkpoint_hashes={'predictor-008000': {'lm.pt': '0' * 64}},
                documents=[row['id'] for row in panel], panel_sha256=digest(panel),
                checks=dict(continuation_sha256=digest([[r['id'], r['canonical_continuation_ids'], r['continuation_bytes']]
                                                        for r in panel])),
                input_hashes={'model/model.safetensors': 'a' * 64, 'model/tokenizer.json': 'b' * 64},
                pool_index_sha256=hashlib.sha256(pool_index_bytes()).hexdigest(),
                code_manifest_sha256=sha256(code / 'source-manifest.json'),
                runtime={'numpy': np.__version__.split('+')[0]}, smoke_tokenizer=3)
    write_json(ref / 'plan.json', plan)
    audit = dict(stage='verified', documents=156, scored_tokens=4992, scored_bytes=14429, original_prefix_failures=[],
                 trimmed_prefixes=[], panel_sha256=digest(panel), native_prefixes_sha256=digest(prefixes),
                 plan_sha256=digest(plan))
    write_json(ref / 'run/panel-audit.json', audit)
    write_json(ref / 'run/scoring-panel.json', panel)
    write_json(ref / 'run/native-prefixes.json', prefixes)
    write_json(ref / 'run/scores/predictor-008000/0000.json', dict(nats=[1.0]))
    write_json(ref / 'resume-reference/checkpoints/predictor-008000/ready.json', dict(step=8000, files={}))
    bundle = dict(repo='cminst/anytok-training-inputs', path='input.tar.gz', revision='r' * 40, bytes=1, sha256='c' * 64)
    write_json(ref / 'sources.json', dict(input_bundle=bundle, checkpoints=dict(repo='cminst/other', revision='r', items={})))
    write_json(ref / 'resume-verified.json', dict(plan_sha256=digest(plan), saved=10, remaining=5))
    for name in MODULES:
        shutil.copy2(SCRIPTS / name, ref / name)
    (ref / 'run_rift_pool_sweep.py').write_text('# an older copy from Rift\n')
    return ref


def build_hub(tmp_path, steps=STEPS):
    """The run's folder on Hugging Face: three scored files per save plus the files the eval never reads."""
    run = tmp_path / 'hub' / prep.FOLDER
    for step in steps:
        folder = run / f'step-{step:06d}'
        names = ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors', 'policy.pt']
        if step % 2000 == 0 or step == 16409:
            names += ['optimizer.pt', 'state.pt']
        files = {}
        for name in names:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'{name} at {step}'.encode())
            files[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        write_json(folder / 'manifest.json', dict(files=files, kind='weights_only_checkpoint', step=step))
    return run


@pytest.fixture
def setup(tmp_path):
    ref, hub = build_reference(tmp_path), build_hub(tmp_path)
    return SimpleNamespace(ref=ref, hub=hub, work=tmp_path / 'heldout' / 'raw-bpb', tmp=tmp_path)


def prepared(s):
    prep.prepare(s.ref, s.work, prep.manifests_from_folder(s.hub), 'local folder', prep.REVISION)
    return json.loads((s.work / 'plan.json').read_text())


def test_only_the_checkpoint_list_changes(setup):
    s = setup
    his = json.loads((s.ref / 'plan.json').read_text())
    plan = prepared(s)
    assert sorted(k for k in set(his) | set(plan) if his.get(k) != plan.get(k)) == ['checkpoint_hashes', 'checkpoints']
    assert plan['checkpoints'] == LABELS and len(LABELS) == 42
    assert LABELS.index('raw-bpb-016409') == LABELS.index('raw-bpb-016000') + 1
    manifest = json.loads((s.hub / 'step-016409/manifest.json').read_text())
    assert plan['checkpoint_hashes']['raw-bpb-016409'] == {name: manifest['files'][name]['sha256'] for name in prep.SCORED_FILES}
    assert (s.work / 'checkpoints.txt').read_text().splitlines() == LABELS

    sources = json.loads((s.work / 'sources.json').read_text())
    assert sources['input_bundle'] == json.loads((s.ref / 'sources.json').read_text())['input_bundle']
    checkpoints = sources['checkpoints']
    assert (checkpoints['repo'], checkpoints['revision']) == (prep.REPO, prep.REVISION)
    item = checkpoints['items']['raw-bpb-016409']
    assert item['prefix'] == 'raw-bpb-rl-20k/step-016409' and item['step'] == 16409
    assert item['files'] == {name: manifest['files'][name] for name in prep.SCORED_FILES}  # no optimizer or state
    assert json.loads((s.work / 'resume-reference/checkpoints/raw-bpb-016409/ready.json').read_text()) == manifest

    assert prep.tree_hashes(s.work / 'inbox') == prep.tree_hashes(s.ref / 'inbox')
    for name in MODULES:
        assert sha256(s.work / name) == sha256(s.ref / name)
    assert sha256(s.work / 'run_rift_pool_sweep.py') == prep.SCORER_FILES['run_rift_pool_sweep.py']
    assert not (s.work / 'run').exists() and not (s.work / 'resume-reference/checkpoints/predictor-008000').exists()

    adaptation = json.loads((s.work / 'adaptation.json').read_text())
    assert adaptation['plan_sha256'] == digest(plan) and adaptation['reference_plan_sha256'] == digest(his)
    assert adaptation['checkpoints']['count'] == 42 and adaptation['checkpoints']['steps'] == STEPS
    assert adaptation['archived_scorer_files'] == {'run_rift_pool_sweep.py': sha256(s.ref / 'run_rift_pool_sweep.py')}
    assert adaptation['added_files'] == [dict(file='four_way_8k.py', sha256=sha256(SCRIPTS / 'four_way_8k.py'))]
    assert adaptation['not_copied_directories'] == ['resume-reference', 'run']


def his_audit_rerun(s):
    """What run_rift_pool_sweep.py audit writes in the new workspace: his audit under the new plan hash."""
    plan = json.loads((s.work / 'plan.json').read_text())
    audit = json.loads((s.ref / 'run/panel-audit.json').read_text()) | dict(plan_sha256=digest(plan))
    (s.work / 'run').mkdir(exist_ok=True)
    for name in ('scoring-panel.json', 'native-prefixes.json'):
        shutil.copy2(s.ref / 'run' / name, s.work / 'run' / name)
    write_json(s.work / 'run/panel-audit.json', audit)
    return audit


def stage_with_his_script(s, monkeypatch, labels):
    """His stage_rift_resume_inputs.stage_checkpoint, unchanged, reading the fake hub anonymously."""
    def hf_hub_download(repo, path, revision, repo_type, local_dir, token, force_download):
        assert (repo, revision, repo_type, token) == (prep.REPO, prep.REVISION, 'model', False)
        target = Path(local_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.hub.parent / path, target)
        return str(target)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=hf_hub_download))
    monkeypatch.delitem(sys.modules, 'stage_rift_resume_inputs', raising=False)
    stage = importlib.import_module('stage_rift_resume_inputs')
    return [stage.stage_checkpoint(s.work, label) for label in labels]


def his_preflight(s):
    env = {key: value for key, value in os.environ.items() if key != 'ANYTOK_RIFT_INPUT'}
    return subprocess.run([sys.executable, str(s.work / 'run_rift_pool_sweep.py'), 'verify-launch',
                           '--checkpoint-list', str(s.work / 'checkpoints.txt')],
                          cwd=s.work, env=env, capture_output=True, text=True)


def test_his_staging_and_preflight_accept_the_workspace(setup, monkeypatch):
    s = setup
    prepared(s)
    his_audit_rerun(s)
    result = prep.check_audit(s.ref, s.work)
    assert result['matches_reference'] and result['panel_sha256'] == json.loads((s.ref / 'run/panel-audit.json').read_text())['panel_sha256']
    assert (s.work / 'run/audit-matches-reference.json').is_file()
    # What his stage_inputs and features steps leave behind.
    (s.work / 'input/pool').mkdir(parents=True)
    (s.work / 'input/pool/full-results.json').write_bytes(pool_index_bytes())
    write_json(s.work / 'run/features-ready.json', dict(features=1000))

    staged = stage_with_his_script(s, monkeypatch, LABELS)
    assert [row['step'] for row in staged] == STEPS
    assert not (s.work / 'checkpoints/raw-bpb-020000/optimizer.pt').exists()
    done = his_preflight(s)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report == dict(protocol='heldout-pool-4992', plan_sha256=digest(json.loads((s.work / 'plan.json').read_text())),
                          checkpoints=42, tokenizers=1000, pairs=42000, passages=156, scored_tokens=4992, scored_bytes=14429)

    # A staged file that differs from its Hugging Face manifest stops the preflight.
    with (s.work / 'checkpoints/raw-bpb-010000/lm.pt').open('ab') as stream:
        stream.write(b'x')
    done = his_preflight(s)
    assert done.returncode != 0 and 'Changed checkpoint file for raw-bpb-010000/lm.pt' in done.stderr


def test_check_audit_refusals(setup):
    s = setup
    prepared(s)
    audit = his_audit_rerun(s)
    write_json(s.work / 'run/panel-audit.json', audit | dict(plan_sha256=digest(json.loads((s.ref / 'plan.json').read_text()))))
    with pytest.raises(ValueError, match='does not cover'):
        prep.check_audit(s.ref, s.work)
    write_json(s.work / 'run/panel-audit.json', audit | dict(native_prefixes_sha256='d' * 64))
    with pytest.raises(ValueError, match=r"differs from his in \['native_prefixes_sha256'\]"):
        prep.check_audit(s.ref, s.work)
    write_json(s.work / 'run/panel-audit.json', audit)
    plan = json.loads((s.work / 'plan.json').read_text())
    write_json(s.work / 'plan.json', plan | dict(smoke_tokenizer=4))
    with pytest.raises(ValueError, match='changed after preparation'):
        prep.check_audit(s.ref, s.work)


def edit_manifest(hub, step, change):
    path = hub / f'step-{step:06d}/manifest.json'
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('case, message', [
    ('missing step', r'missing steps \[10000\]'),
    ('wrong step', 'names step 501'),
    ('no hash', 'no size and SHA-256 for hypernet/model.safetensors'),
    ('extra step', r'outside the run: \[20500\]'),
    ('his plan edited', 'not the plan his verified panel audit covers'),
    ('not restored', 'not a restored workspace'),
    ('scorer changed', 'differs from his file at 948787e'),
])
def test_prepare_refusals(setup, monkeypatch, case, message):
    s = setup
    if case == 'missing step':
        shutil.rmtree(s.hub / 'step-010000')
    elif case == 'wrong step':
        edit_manifest(s.hub, 500, lambda m: m.update(step=501))
    elif case == 'no hash':
        edit_manifest(s.hub, 500, lambda m: m['files']['hypernet/model.safetensors'].pop('sha256'))
    elif case == 'extra step':
        build_hub(s.tmp, [20500])
    elif case == 'his plan edited':
        plan = json.loads((s.ref / 'plan.json').read_text())
        write_json(s.ref / 'plan.json', plan | dict(documents=plan['documents'][:-1]))
    elif case == 'not restored':
        (s.ref / 'resume-verified.json').unlink()
    elif case == 'scorer changed':
        monkeypatch.setitem(prep.SCORER_FILES, 'rift_eval_memory.py', '0' * 64)
    with pytest.raises(ValueError, match=message):
        prepared(s)
    assert not s.work.exists()


def test_main_refuses_an_existing_workspace_and_reports_success(setup, capsys):
    s = setup
    args = ['prepare', '--reference', str(s.ref), '--workspace', str(s.work), '--manifests', str(s.hub)]
    assert prep.main(args) == 0
    assert json.loads(capsys.readouterr().out)['checkpoints'] == 42
    assert prep.main(args) == 2
    assert 'already exists' in json.loads(capsys.readouterr().err)['error']


# ---------------------------------------------------------------- the dashboard observer
import heldout_pool_status as collector
import run_rawbpb_heldout_pool as watch

SMOKE_OK = dict(gpu='NVIDIA H200', smaller_vocab_bitwise_equal=True, vocab_size=128000)


def remote(**fields):
    status = dict(stage_exists=True, setup=dict(inputs=dict(exit=0, step='x'), panel=None, checkpoints=None, preflight=None),
                  markers={name: False for name in collector.MARKERS}, smoke=None, preflight=None,
                  checkpoints={label: dict(scored=0, complete=False, ranking=False) for label in LABELS},
                  worker_failures=[], jobs=[], slurm_errors=[])
    status.update(fields)
    return status


def tasks(state, count=42, reason=''):
    return [dict(id=f'600_{i}', state=state, reason=reason, elapsed='00:10:00') for i in range(count)]


@pytest.mark.parametrize('status, stage, exit_code', [
    (dict(stage_exists=False), 'preparing', None),
    (remote(setup=dict(inputs=dict(exit=None, step='2026 environment'))), 'preparing', None),
    (remote(setup=dict(inputs=dict(exit=1, step='2026 restore'))), 'failed', 2),
    (remote(setup=dict(inputs=dict(exit=0), panel=dict(exit=137, step='2026 panel audit'))), 'failed', 2),
    (remote(setup=dict(inputs=dict(exit=0), panel=dict(exit=None, step='2026 features')), smoke=SMOKE_OK,
            jobs=[dict(id='599', state='COMPLETED')]), 'preparing', None),
    (remote(jobs=[dict(id='599', state='PENDING', reason='Priority')]), 'smoke', None),
    (remote(jobs=[dict(id='599', state='COMPLETED')], smoke=SMOKE_OK), 'smoke_passed', None),
    (remote(jobs=[dict(id='599', state='COMPLETED')], smoke=dict(SMOKE_OK, gpu='NVIDIA L40S')), 'failed', 2),
    (remote(jobs=[dict(id='599', state='FAILED', exit_code='1:0')]), 'failed', 2),
    (remote(worker_failures=[dict(worker='hardware', error='Expected a H200 for this sweep')]), 'failed', 2),
    (remote(jobs=[dict(id='599', state='COMPLETED')], smoke=SMOKE_OK,
            markers=dict(preflight_passed=False, **{'preflight-passed': True})), 'ready', None),
    (remote(jobs=[dict(id='599', state='COMPLETED')] + tasks('PENDING', reason='Priority'), smoke=SMOKE_OK), 'queued', None),
    (remote(jobs=[dict(id='599', state='COMPLETED')] + tasks('RUNNING', 3) + tasks('PENDING')[3:], smoke=SMOKE_OK),
     'scoring', None),
    (remote(jobs=tasks('COMPLETED')[:41] + [dict(id='600_41', state='TIMEOUT', exit_code='0:15')]), 'failed', 2),
])
def test_observer_stages(status, stage, exit_code):
    verdict = watch.evaluate(status)
    assert (verdict['stage'], verdict['exit']) == (stage, exit_code)


def test_observer_counts_scores_and_needs_every_checkpoint_complete():
    checkpoints = {label: dict(scored=1000, complete=True) for label in LABELS}
    assert watch.evaluate(remote(checkpoints=checkpoints)) == dict(
        completed=42000, exit=0, ambiguous=False, stage='complete',
        detail='all 42 checkpoints scored on all 1000 tokenizers')
    checkpoints[LABELS[-1]] = dict(scored=999, complete=False)
    verdict = watch.evaluate(remote(checkpoints=checkpoints, jobs=tasks('COMPLETED')))
    assert (verdict['completed'], verdict['stage'], verdict['ambiguous']) == (41999, 'scoring', True)


def test_observer_names_each_task_by_its_checkpoint():
    status = remote(jobs=[dict(id='600_41', state='RUNNING', elapsed='01:00')])
    status['checkpoints'][LABELS[41]]['scored'] = 250
    (worker,) = watch.build_workers(status)
    assert (worker['task'], worker['state'], worker['elapsed_seconds']) == ('score raw-bpb-020000', 'Tokenizer 250/1000', 60)


def test_status_script_reads_logs_and_merges_slurm(tmp_path, monkeypatch):
    log = tmp_path / 'setup-inputs.log'
    log.write_text('=== 2026-11-24T12:00:00Z environment\nok\n=== 2026-11-24T12:05:00Z restore\nexit=1\n')
    assert collector.setup_log(log)['exit'] == 1 and collector.setup_log(log)['step'].endswith('restore')
    outputs = {'sacct': '599|COMPLETED|0:0|00:03:00|a|b\n600_[0-41]|PENDING|0:0|00:00:00|Unknown|Unknown\n',
               'squeue': '600_0|PENDING|Priority|0:00\n600_1|PENDING|Priority|0:00\n'}
    monkeypatch.setattr(collector, 'command', lambda argv: (outputs[argv[0]], None))
    jobs, errors = collector.slurm('2026-11-24')
    assert [job['id'] for job in jobs] == ['599', '600_0', '600_1'] and not errors


def test_observer_needs_a_host_and_one_matching_socket(tmp_path):
    with pytest.raises(SystemExit):
        watch.parse_args([])
    exact = str(tmp_path / 'socket')
    assert watch.parse_args(['--host', 'h', '--control-path', exact]).control_path == exact
    with pytest.raises(SystemExit):
        watch.parse_args(['--host', 'h', '--control-path', str(tmp_path / 'cm-*')])


import heldout_pool_passage_means as passage_script
import report_rawbpb_heldout_pool as report


def test_passage_means_average_every_tokenizer_and_check_each_file(tmp_path, monkeypatch):
    monkeypatch.setattr(passage_script, 'TOKENIZERS', 3)
    folder = tmp_path / 'raw-bpb-000500'
    key = dict(plan_sha256='p' * 64, checkpoint='raw-bpb-000500')
    write_json(folder / 'complete.json', dict(key, completed=3))
    write_json(folder / 'canonical.json', dict(key, bpb=1.0, nats=[1.0, 2.0]))
    for index, nats in enumerate([[2.0, 3.0], [4.0, 5.0], [3.0, 7.0]]):
        write_json(folder / f'{index:04d}.json', dict(key, index=index, nats=nats))
    result = passage_script.passage_means(folder)
    assert (result['canonical_nats'], result['mean_transferred_nats']) == ([1.0, 2.0], [3.0, 5.0])
    write_json(folder / '0002.json', dict(key, checkpoint='raw-bpb-001000', index=2, nats=[3.0, 7.0]))
    with pytest.raises(ValueError, match='does not match its receipt'):
        passage_script.passage_means(folder)


def test_report_pool_mean_and_paired_bootstrap():
    rng = np.random.default_rng(0)
    size = rng.integers(80, 100, 12).astype(float)
    groups = [(domain, np.arange(4 * i, 4 * i + 4), size[4 * i:4 * i + 4]) for i, domain in enumerate('abc')]
    gaps = rng.random((5, 12))
    # The pool mean of the tokenizers' balanced gaps is the balanced gap of the per-passage pool means.
    assert np.isclose(np.mean([report.balanced(row, groups) for row in gaps]), report.balanced(gaps.mean(0), groups))
    draws = report.bootstrap_draws(groups, reps=200, seed=1)
    base = report.resampled(gaps[0], groups, draws)
    assert np.allclose(base[0], report.balanced(gaps[0][np.concatenate([g[1][d[0]] for g, d in zip(groups, draws)])],
                                                [(g[0], np.arange(4 * i, 4 * i + 4), g[2][d[0]])
                                                 for i, (g, d) in enumerate(zip(groups, draws))]))
    # Adding 0.1 nats per byte to every passage raises every replicate by exactly 0.1 / ln 2.
    shifted = report.resampled(gaps[0] + 0.1 * size, groups, draws)
    assert np.allclose(shifted - base, 0.1 / np.log(2))
    assert report.hardest(np.array([0.1, 0.3, 0.3, 0.2])) == [1, 2, 3, 0]


def test_report_needs_every_job_completed_on_one_gpu(tmp_path):
    def line(job, name, state='COMPLETED', seconds=1200):
        return f'{job}|{name}|normal|{state}|0:0|{seconds}|2026-11-24T13:00:00|2026-11-24T13:20:00|billing=8,cpu=4,gres/gpu=1'
    rows = [line(f'51847{i}', 'setup', seconds=60) for i in range(5)]
    rows += [line(f'518632_{i}', 'anytok-heldout-4992') for i in range(42)]
    path = tmp_path / 'sacct.txt'
    path.write_text('\n'.join(rows) + '\n')
    _, jobs = report.read_jobs(path)
    assert (jobs['tasks'], jobs['array_gpu_hours'], round(jobs['array_usd'], 2)) == (42, 14.0, 12.6)
    path.write_text('\n'.join(rows[:-1] + [line('518632_41', 'anytok-heldout-4992', state='FAILED')]) + '\n')
    with pytest.raises(ValueError, match='completed on one GPU'):
        report.read_jobs(path)
