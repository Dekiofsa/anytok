"""hyak.corpus.stage_inputs: the staged layout satisfies the pinned trainers' contracts.

The trainers cannot be imported here (Linux librust_utils.so, CUDA), so their ``verify_corpus``
and ``build_eval_set`` are compiled straight from the pinned files, and ``BalancedTextStream`` /
``make_split_prefix_example`` are loaded from the pinned ``joint_training.py`` (its ``rust_utils``
import is satisfied by an empty module; the sampler is never called).
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import types

import pytest

from hyak.corpus import sources, stage_inputs
from hyak.corpus.stage_inputs import InputLayout, StagingError
from hyak.tests.test_corpus_fixtures import (JOINT_TRAINING, REPO, RL_TRAINER, TRAINER, VARIANT_RL_TRAINER,
                                             extract_function,
                                             function_node, write_bucket, write_pool)

RECIPE = REPO / 'data' / 'balanced-full-v3.json'
TEST_SCALE = {'production': False, 'train_target': 60000, 'eval_target': 15000, 'note': 'test'}


def pinned_verify_corpus(trainer=TRAINER):
    return extract_function(trainer, 'verify_corpus', {'json': json, 'Path': Path, 'hashlib': hashlib})


@pytest.fixture(scope='module')
def joint_training():
    """The pinned hf_hypernet/joint_training.py as an isolated module (needs torch/numpy/tokenizers)."""
    pytest.importorskip('torch')
    pytest.importorskip('tokenizers')
    saved = sys.modules.get('rust_utils')
    sys.modules['rust_utils'] = types.ModuleType('rust_utils')
    try:
        spec = importlib.util.spec_from_file_location('hyak_test_pinned_joint_training', JOINT_TRAINING)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop('rust_utils', None)
        else:
            sys.modules['rust_utils'] = saved
    assert hashlib.sha256(JOINT_TRAINING.read_bytes()).hexdigest().startswith('7ce18a9d')
    return module


def consume(joint_training, recipe_path, data_dir, updates=15, seed=42):
    """Drive BalancedTextStream exactly as the trainer does (train_joint_smollm_input.py:473-483, 518-519)."""
    corpus = joint_training.BalancedTextStream(recipe_path, data_dir)
    rng = random.Random(seed)
    texts = []
    try:
        for _ in range(joint_training.INITIAL_QUEUE_BATCHES + updates):
            names = rng.choices(corpus.names, weights=corpus.weights, k=joint_training.QUEUE_BATCH_SIZE)
            texts.extend((name, corpus.next_text(name, chars=2048)) for name in names)
        return corpus, texts
    finally:
        corpus.close()


ACCEPT = 'synthetic test corpus (cannot match the original anchors)'


def fake_model(paths) -> dict:
    """Three small files in input/model and their expected records (the pin's shape)."""
    model = paths.model_dir
    model.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, body in (('config.json', b'{"model_type": "llama"}'), ('model.safetensors', b'\0' * 4096),
                       ('tokenizer.json', b'{"model": {}}')):
        (model / name).write_bytes(body)
        expected[name] = {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    return expected


def stage_ok(paths, **kwargs):
    """stage() of a synthetic production-like pool: fake model, non-exact scorecard accepted."""
    kwargs.setdefault('accept_corpus_scorecard', ACCEPT)
    if 'model_expected' not in kwargs:
        kwargs['model_expected'] = fake_model(paths)
    kwargs.setdefault('log', lambda *a: None)
    return stage_inputs.stage(paths, **kwargs)


def exact_scorecard(root: Path, **changes):
    """A scorecard.json as hyak.corpus.verify writes it for an exact corpus (bound to the current manifests)."""
    manifests = {name: hashlib.sha256((Path(root) / f'{size}-v1' / name / 'manifest.json').read_bytes()).hexdigest()
                 for name, (size, _spec) in sources.bucket_specs().items()}
    report = {'format': 2, 'corpus_root': str(root), 'bucket_manifests': manifests, 'exact': True,
              'verdict': 'content-identical to every recorded anchor (test)', 'at_utc': '2026-11-13T00:00:00Z'}
    report.update(changes)
    (Path(root) / 'scorecard.json').write_text(json.dumps(report))
    return report


@pytest.fixture
def staged(tmp_config):
    write_pool(tmp_config.paths.corpus_root)
    report = stage_ok(tmp_config.paths)
    return tmp_config, report


# --- layout and contract -------------------------------------------------------------------------------
def test_stage_layout(staged):
    tmp, report = staged
    paths = tmp.paths
    recipe = json.loads(RECIPE.read_text())
    names = [b['name'] for b in recipe['buckets']]
    assert paths.recipe.read_bytes() == RECIPE.read_bytes()
    assert hashlib.sha256(paths.recipe.read_bytes()).hexdigest() == stage_inputs.RECIPE_SHA256
    manifest = json.loads(paths.input_corpus_manifest.read_text())
    assert list(manifest) == sorted(names) and len(manifest) == 23
    for bucket in recipe['buckets']:
        name = bucket['name']
        link = paths.input_corpus_train(name)
        assert link.is_symlink() and not os.path.isabs(os.readlink(link))
        target = paths.root / 'corpus' / Path(bucket['source_path']).relative_to('/corpus') / 'train.jsonl.gz'
        assert link.resolve() == target.resolve()
        entry = manifest[name]
        assert set(entry) == set(stage_inputs.CORPUS_ENTRY_FIELDS)
        assert entry['compressed_bytes'] == target.stat().st_size
        assert entry['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert entry['source'] == f'corpus/{Path(bucket["source_path"]).relative_to("/corpus")}/train.jsonl.gz'
        assert entry['train_bytes'] == bucket['train_available_bytes']
        held = paths.input_eval_file(name)
        assert not held.is_symlink() and held.read_bytes() == (target.parent / 'eval.jsonl.gz').read_bytes()
    assert sorted(p.name for p in paths.input_eval.iterdir()) == sorted(f'{n}.jsonl.gz' for n in names)
    pool = json.loads(paths.pool_recipe.read_text())
    assert paths.pool_recipe.read_text() == json.dumps(pool, indent=2) + '\n'
    for old, new in zip(recipe['buckets'], pool['buckets']):
        assert {k: v for k, v in new.items() if k not in ('source_path', 'eval_path')} == \
               {k: v for k, v in old.items() if k not in ('source_path', 'eval_path')}
        assert Path(new['source_path']) == paths.corpus_root / Path(old['source_path']).relative_to('/corpus')
        assert (Path(new['source_path']) / 'train.jsonl.gz').is_file() and Path(new['eval_path']).is_file()
    assert {k: v for k, v in pool.items() if k != 'buckets'} == {k: v for k, v in recipe.items() if k != 'buckets'}
    inputs = json.loads(paths.input_manifest.read_text())
    assert inputs['kind'] == stage_inputs.INPUT_MANIFEST_KIND and len(inputs['files']) == 52
    assert inputs['files']['corpus/manifest.json']['sha256'] == hashlib.sha256(paths.input_corpus_manifest.read_bytes()).hexdigest()
    corpus = inputs['provenance']['corpus']
    assert corpus['test_scale'] is False and corpus['mode'] == 'links'
    assert corpus['scorecard']['accepted_reason'] == ACCEPT and corpus['scorecard']['exact'] is False
    # model files registered with the pinned records (env-1): the trainer's manifest loop now hashes them
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        assert inputs['files'][f'model/{name}']['sha256'] == hashlib.sha256((paths.model_dir / name).read_bytes()).hexdigest()
    # gzip header times of every bucket (a lost bucket can be rebuilt byte for byte)
    for name in names:
        size = sources.bucket_specs()[name][0]
        train = paths.corpus_bucket_dir(size, name) / 'train.jsonl.gz'
        assert corpus['gzip_mtime'][name]['train.jsonl.gz'] == int.from_bytes(train.read_bytes()[4:8], 'little')
    assert report['changed'] == [] and report['buckets'] == 23 and report['model_files'] == 3


def test_pinned_trainer_verify_corpus_accepts_the_stage(staged):
    tmp, _ = staged
    for trainer in (TRAINER, RL_TRAINER, VARIANT_RL_TRAINER):
        pinned_verify_corpus(trainer)(tmp.paths.input_corpus, tmp.paths.input_corpus_manifest)
    port = [ast.dump(n) for n in function_node(REPO / 'hyak/corpus/stage_inputs.py', 'verify_corpus_contract').body[2:]]
    for trainer in (TRAINER, RL_TRAINER, VARIANT_RL_TRAINER):
        assert port == [ast.dump(n) for n in function_node(trainer, 'verify_corpus').body]  # verbatim copy
    assert stage_inputs.check_contract(InputLayout(tmp.paths), json.loads(RECIPE.read_text())) == []
    manifest = json.loads(tmp.paths.input_corpus_manifest.read_text())
    manifest['extra'] = {'compressed_bytes': 0, 'sha256': ''}
    tmp.paths.input_corpus_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='all 23'):
        pinned_verify_corpus()(tmp.paths.input_corpus, tmp.paths.input_corpus_manifest)
    assert stage_inputs.check_staged(tmp.paths)


def test_restaging_is_idempotent_and_check_passes(staged):
    tmp, _ = staged
    expected = fake_model(tmp.paths)  # same bytes as the fixture's
    before = {p: p.stat().st_mtime_ns for p in tmp.paths.input.rglob('*') if p.is_file() and not p.is_symlink()
              and p.name != 'manifest.json'}
    inputs_before = tmp.paths.input_manifest.read_bytes()
    report = stage_ok(tmp.paths, model_expected=expected)
    assert report['changed'] == []
    assert {p: p.stat().st_mtime_ns for p in before} == before
    # corpus-9 (d): nothing changed, so input/manifest.json keeps its bytes (staged_at, job and host included)
    assert tmp.paths.input_manifest.read_bytes() == inputs_before
    assert stage_inputs.check_staged(tmp.paths, model_expected=expected) == []
    eval_file = tmp.paths.input_eval_file('code')
    eval_file.write_bytes(gzip.compress(b'{"id": "x", "text": "y"}\n'))
    problems = stage_inputs.check_staged(tmp.paths, model_expected=expected)
    assert any('code' in p and 'differs' in p for p in problems)


def test_balanced_text_stream_consumes_the_stage(staged, joint_training):
    tmp, _ = staged
    corpus, texts = consume(joint_training, tmp.paths.recipe, tmp.paths.input_corpus)
    assert len(texts) == (16 + 15) * 128
    assert all(isinstance(t, str) and 64 <= len(t) <= 2048 for _, t in texts)
    assert {name for name, _ in texts} <= set(corpus.names) and len({name for name, _ in texts}) >= 20
    # resume: a stream rebuilt from state_dict() continues with exactly the same texts
    stream = joint_training.BalancedTextStream(tmp.paths.recipe, tmp.paths.input_corpus)
    rng = random.Random(7)
    names = rng.choices(stream.names, weights=stream.weights, k=600)
    head = [stream.next_text(n, chars=2048) for n in names[:300]]
    state = json.loads(json.dumps(stream.state_dict()))
    tail = [stream.next_text(n, chars=2048) for n in names[300:]]
    stream.close()
    resumed = joint_training.BalancedTextStream(tmp.paths.recipe, tmp.paths.input_corpus, state)
    assert [resumed.next_text(n, chars=2048) for n in names[300:]] == tail
    resumed.close()
    assert head


def test_tokenizer_pool_reads_the_pool_recipe(staged):
    """scripts/tokenizer_pool.py (Modal-free, imported from its file) trains from pool-recipe paths."""
    tmp, _ = staged
    spec = importlib.util.spec_from_file_location('hyak_test_tokenizer_pool', REPO / 'scripts' / 'tokenizer_pool.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe = json.loads(tmp.paths.pool_recipe.read_text())
    quota = {b['name']: 5000 + 17 * i for i, b in enumerate(recipe['buckets'])}
    stats, validation = {}, {}
    batches = list(module.batches(recipe, quota, stats, validation))
    assert batches and set(stats) == set(quota) and set(validation) == set(quota)
    for name, row in stats.items():
        assert 0 <= quota[name] - row['used_bytes'] <= 3
    again = {}
    list(module.batches(recipe, quota, again, {}))
    assert again == stats


# --- refusals -----------------------------------------------------------------------------------------------
def test_refuses_test_scale_without_flag_and_accepts_with_it(tmp_config):
    write_pool(tmp_config.paths.corpus_root, test_scale=TEST_SCALE)
    with pytest.raises(StagingError, match='TEST-SCALE') as info:
        stage_inputs.stage(tmp_config.paths, log=lambda *a: None)
    assert str(info.value).count('TEST-SCALE') == 1
    # a test-scale stage needs no scorecard and registers no model (the e2e worlds use a tiny model)
    report = stage_inputs.stage(tmp_config.paths, allow_test_scale=True, log=lambda *a: None)
    assert report['test_scale'] is True and report['shortfall_bytes']['web-en'] > 0 and report['model_files'] == 0
    inputs = json.loads(tmp_config.paths.input_manifest.read_text())
    assert inputs['provenance']['corpus']['test_scale'] is True
    assert not any(key.startswith('model/') for key in inputs['files'])
    pinned_verify_corpus()(tmp_config.paths.input_corpus, tmp_config.paths.input_corpus_manifest)
    # corpus-7: --check accepts it, but says loudly that it is a test-scale stage
    assert stage_inputs.check_staged(tmp_config.paths) == []
    assert any('TEST-SCALE' in w for w in stage_inputs.check_warnings(tmp_config.paths))


def test_refuses_mixed_test_scale_and_production(tmp_config):
    write_pool(tmp_config.paths.corpus_root)
    size, spec = sources.bucket_specs()['code']
    write_bucket(tmp_config.paths.corpus_bucket_dir(size, 'code'), dict(spec, train_target=60000, eval_target=15000),
                 test_scale=TEST_SCALE)
    with pytest.raises(StagingError, match='mixed'):
        stage_ok(tmp_config.paths, allow_test_scale=True)


def test_refuses_a_bucket_below_the_quota(tmp_config):
    write_pool(tmp_config.paths.corpus_root)
    size, spec = sources.bucket_specs()['wiki-hi']
    write_bucket(tmp_config.paths.corpus_bucket_dir(size, 'wiki-hi'), spec, counts={'train_bytes': 124_999_999})
    with pytest.raises(StagingError, match='silently rewind'):
        stage_ok(tmp_config.paths)


def test_refuses_tampered_missing_or_foreign_buckets(tmp_config):
    write_pool(tmp_config.paths.corpus_root)
    train = tmp_config.paths.corpus_bucket_dir('full', 'web-fr') / 'train.jsonl.gz'
    train.write_bytes(train.read_bytes() + b'\0')
    manifest = tmp_config.paths.corpus_bucket_dir('extra', 'qa-en') / 'manifest.json'
    manifest.unlink()
    other = tmp_config.paths.corpus_bucket_dir('specialty', 'legal-en') / 'manifest.json'
    data = json.loads(other.read_text())
    data['revision'] = '0' * 40
    other.write_text(json.dumps(data))
    with pytest.raises(StagingError) as info:
        stage_ok(tmp_config.paths)
    message = str(info.value)
    assert 'web-fr' in message and 'SHA-256' in message
    assert 'qa-en' in message and 'hyak.corpus.build --buckets qa-en' in message
    assert 'legal-en' in message and 'revision' in message
    assert not tmp_config.paths.input_corpus_manifest.exists()


def test_refuses_to_change_inputs_of_existing_runs(staged):
    tmp, _ = staged
    run = tmp.paths.run_dir('uniform')
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text('{}')
    assert stage_ok(tmp.paths)['changed'] == []  # identical content: allowed
    size, spec = sources.bucket_specs()['web-zh']
    write_bucket(tmp.paths.corpus_bucket_dir(size, 'web-zh'), spec, counts={'train_bytes': 700_000_000}, seed=99)
    before = tmp.paths.input_corpus_manifest.read_bytes()
    with pytest.raises(StagingError, match='refuse to resume'):
        stage_ok(tmp.paths)
    assert tmp.paths.input_corpus_manifest.read_bytes() == before
    report = stage_ok(tmp.paths, force=True)
    assert str(tmp.paths.input_corpus_manifest) in report['changed']
    assert tmp.paths.input_corpus_manifest.read_bytes() != before


def test_refuses_to_change_inputs_after_pool_training(staged):
    tmp, _ = staged
    tokenizer = tmp.paths.pool_tokenizer_dir(7)
    tokenizer.mkdir(parents=True)
    (tokenizer / 'manifest.json').write_text('{}')
    size, spec = sources.bucket_specs()['math-en']
    write_bucket(tmp.paths.corpus_bucket_dir(size, 'math-en'), spec, counts={'train_bytes': 1_000_000_152}, seed=5)
    with pytest.raises(StagingError, match='pool tokenizer'):
        stage_ok(tmp.paths)
    assert stage_ok(tmp.paths, force=True)['changed']


def test_refuses_to_replace_a_regular_file(staged):
    tmp, _ = staged
    link = tmp.paths.input_corpus_train('books-en')
    data = link.read_bytes()
    link.unlink()
    link.write_bytes(data)
    with pytest.raises(StagingError, match='regular file') as info:
        stage_ok(tmp.paths)
    assert '--original-staged' in str(info.value)
    stage_ok(tmp.paths, force=True)
    assert link.is_symlink()


def test_recipe_checks():
    recipe = json.loads(RECIPE.read_text())
    stage_inputs.check_recipe(recipe)
    broken = json.loads(RECIPE.read_text())
    broken['buckets'][0], broken['buckets'][1] = broken['buckets'][1], broken['buckets'][0]
    with pytest.raises(StagingError, match='order'):
        stage_inputs.check_recipe(broken)
    broken = json.loads(RECIPE.read_text())
    broken['buckets'][3]['source_path'] = '/corpus/extra-v1/web-ru'
    with pytest.raises(StagingError, match='paths'):
        stage_inputs.check_recipe(broken)


def test_update_input_manifest_merges(tmp_path):
    path = tmp_path / 'manifest.json'
    stage_inputs.update_input_manifest(path, {'model/config.json': {'bytes': 1, 'sha256': 'a'}},
                                       provenance_key='model', provenance={'x': 1})
    stage_inputs.update_input_manifest(path, {'recipe.json': {'bytes': 2, 'sha256': 'b'}},
                                       provenance_key='corpus', provenance={'y': 2})
    data = json.loads(path.read_text())
    assert set(data['files']) == {'model/config.json', 'recipe.json'} and set(data['provenance']) == {'model', 'corpus'}
    stage_inputs.update_input_manifest(path, {}, remove=['recipe.json'])
    assert set(json.loads(path.read_text())['files']) == {'model/config.json'}
    assert not (tmp_path / '.manifest.lock').exists()
    path.write_text(json.dumps({'kind': 'zett_20k_inputs', 'files': {}}))
    with pytest.raises(StagingError, match='not an'):
        stage_inputs.update_input_manifest(path, {})


def test_cli_status_check_and_exit_codes(tmp_config, monkeypatch, capsys):
    from hyak.model import fetch_model
    write_pool(tmp_config.paths.corpus_root)
    status_path = tmp_config.paths.monitor_dir('stage-inputs') / 'status.json'
    # 3: the corpus stop (no exact scorecard), nothing staged
    assert stage_inputs.main([]) == stage_inputs.EXIT_STOP == 3
    assert not tmp_config.paths.input_corpus_manifest.exists()
    assert 'corpus stop' in capsys.readouterr().err
    assert json.loads(status_path.read_text())['state'] == 'failed'
    # 2: the model was never fetched
    assert stage_inputs.main(['--accept-corpus-scorecard', ACCEPT]) == stage_inputs.EXIT_USAGE == 2
    assert 'hyak.model.fetch_model' in capsys.readouterr().err
    expected = fake_model(tmp_config.paths)
    monkeypatch.setattr(fetch_model, 'expected_files', lambda repo=None: expected)
    assert stage_inputs.main(['--accept-corpus-scorecard', ACCEPT]) == 0
    status = json.loads(status_path.read_text())
    assert status['state'] == 'complete' and status['completed'] == 52 and status['report']['buckets'] == 23
    capsys.readouterr()
    assert stage_inputs.main(['--check']) == 0
    assert 'non-exact corpus scorecard' in capsys.readouterr().out
    tmp_config.paths.input_corpus_train('web-hi').unlink()
    assert stage_inputs.main(['--check']) == stage_inputs.EXIT_CHECK == 4
    # 4: a refused staging (the regular file above is missing, a bucket file is tampered)
    train = tmp_config.paths.corpus_bucket_dir('full', 'web-fr') / 'train.jsonl.gz'
    train.write_bytes(train.read_bytes() + b'\0')
    assert stage_inputs.main(['--accept-corpus-scorecard', ACCEPT]) == 4
    # 1: an unexpected error prints its traceback
    monkeypatch.setattr(stage_inputs, 'stage', lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError('boom')))
    assert stage_inputs.main([]) == 1
    assert 'Traceback' in capsys.readouterr().err


# --- the corpus stop ----------------------------------------------------------------------------------------
def test_scorecard_stop(tmp_config):
    paths = tmp_config.paths
    write_pool(paths.corpus_root)
    expected = fake_model(paths)
    with pytest.raises(stage_inputs.ScorecardStop, match='does not exist'):
        stage_inputs.stage(paths, model_expected=expected, log=lambda *a: None)
    exact_scorecard(paths.corpus_root, exact=False, verdict='differs from the original: train bytes 22/23')
    with pytest.raises(stage_inputs.ScorecardStop, match='22/23'):
        stage_inputs.stage(paths, model_expected=expected, log=lambda *a: None)
    exact_scorecard(paths.corpus_root, corpus_root='/elsewhere/corpus')
    with pytest.raises(stage_inputs.ScorecardStop, match='/elsewhere/corpus'):
        stage_inputs.stage(paths, model_expected=expected, log=lambda *a: None)
    exact_scorecard(paths.corpus_root)
    report = stage_inputs.stage(paths, model_expected=expected, log=lambda *a: None)
    assert report['scorecard']['exact'] is True and report['scorecard']['accepted_reason'] is None
    # a rebuilt bucket makes the scorecard stale: the corpus stop applies again
    size, spec = sources.bucket_specs()['wiki-ar']
    write_bucket(paths.corpus_bucket_dir(size, 'wiki-ar'), spec, counts={'train_bytes': 187_655_005}, seed=77)
    with pytest.raises(stage_inputs.ScorecardStop, match='older than the bucket manifests of wiki-ar'):
        stage_inputs.stage(paths, model_expected=expected, force=True, log=lambda *a: None)
    report = stage_inputs.stage(paths, model_expected=expected, force=True, accept_corpus_scorecard='user said so',
                                log=lambda *a: None)
    inputs = json.loads(paths.input_manifest.read_text())
    assert inputs['provenance']['corpus']['scorecard']['accepted_reason'] == 'user said so'


def test_model_must_match_the_pin(tmp_config):
    paths = tmp_config.paths
    write_pool(paths.corpus_root)
    expected = fake_model(paths)
    (paths.model_dir / 'tokenizer.json').write_bytes(b'{"model": {"x"}}')  # same size class, other content
    with pytest.raises(stage_inputs.EnvironmentProblem, match='tokenizer.json'):
        stage_ok(paths, model_expected=expected)
    fake_model(paths)
    (paths.model_dir / 'extra.bin').write_bytes(b'x')
    with pytest.raises(stage_inputs.EnvironmentProblem, match='unexpected entries'):
        stage_ok(paths, model_expected=expected)


def test_corpus_root_location(tmp_config):
    paths = tmp_config.paths
    inside_runs = paths.runs / 'corpus'
    write_pool(inside_runs)
    with pytest.raises(StagingError, match='ANYTOK_RUNS'):
        stage_ok(paths, corpus_root=inside_runs)
    elsewhere = tmp_config.base / 'scrubbed' / 'corpus'
    write_pool(elsewhere)
    messages = []
    stage_ok(paths, corpus_root=elsewhere, log=messages.append)
    assert any('outside $ANYTOK_ROOT/corpus' in m for m in messages)
    assert any('outside $ANYTOK_ROOT/corpus' in w for w in stage_inputs.check_warnings(paths))
    # corpus-9 (c): a plain --check re-checks the corpus root the stage recorded
    assert stage_inputs.check_staged(paths, model_expected=fake_model(paths)) == []


def test_original_staged_files_are_kept_and_registered(tmp_config, monkeypatch):
    """readme-10: the 23 original staged files + corpus/manifest.json, accepted when they equal the pins."""
    paths = tmp_config.paths
    write_pool(paths.corpus_root)
    recipe = json.loads(RECIPE.read_text())
    pins = {}
    for i, bucket in enumerate(recipe['buckets']):
        path = paths.input_corpus_train(bucket['name'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps({'text': f'original text {i} ' * 20}).encode() + b'\n', mtime=0))
        pins[f'corpus/{bucket["name"]}/train.jsonl.gz'] = {'bytes': path.stat().st_size,
                                                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {b['name']: {'compressed_bytes': pins[f'corpus/{b["name"]}/train.jsonl.gz']['bytes'],
                            'sha256': pins[f'corpus/{b["name"]}/train.jsonl.gz']['sha256']} for b in recipe['buckets']}
    paths.input_corpus_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    pins['corpus/manifest.json'] = {'bytes': paths.input_corpus_manifest.stat().st_size,
                                    'sha256': hashlib.sha256(paths.input_corpus_manifest.read_bytes()).hexdigest()}
    monkeypatch.setattr(stage_inputs, 'load_inputs_plan', lambda repo=None: pins)
    with pytest.raises(StagingError, match='regular file'):
        stage_ok(paths)                                   # without the flag: refused, nothing replaced
    report = stage_ok(paths, original_staged=True)
    assert report['mode'] == 'original-staged'
    for bucket in recipe['buckets']:
        assert not paths.input_corpus_train(bucket['name']).is_symlink()
    inputs = json.loads(paths.input_manifest.read_text())
    assert inputs['files']['corpus/code/train.jsonl.gz']['origin'] == 'original'
    assert inputs['files']['corpus/manifest.json']['sha256'] == pins['corpus/manifest.json']['sha256']
    pinned_verify_corpus()(paths.input_corpus, paths.input_corpus_manifest)
    expected = fake_model(paths)
    assert stage_inputs.check_staged(paths, model_expected=expected) == []
    # one changed byte in a brought file: refused with both hashes
    path = paths.input_corpus_train('wiki-hi')
    path.write_bytes(path.read_bytes()[:-1] + b'\1')
    with pytest.raises(StagingError, match='not the original trainer inputs'):
        stage_ok(paths, original_staged=True)
    assert any('wiki-hi' in p for p in stage_inputs.check_staged(paths, model_expected=expected))


# --- the shared miniature corpus (built once with the network test / CLI) ---------------------------------
def _mini_input(scratch_dir):
    path = scratch_dir / 'mini-input'
    if not (path / 'corpus' / 'manifest.json').is_file():
        pytest.skip(f'no staged miniature corpus at {path} (see hyak/tests/test_corpus_network.py)')
    return path


def test_mini_input_satisfies_the_trainer(scratch_dir, joint_training, scratch_assets):
    """The staged live miniature: verify_corpus, BalancedTextStream and build_eval_set (92 examples)."""
    root = _mini_input(scratch_dir)
    pinned_verify_corpus()(root / 'corpus', root / 'corpus' / 'manifest.json')
    corpus, texts = consume(joint_training, root / 'recipe.json', root / 'corpus', updates=20)
    assert len(texts) == 36 * 128 and all(64 <= len(t) <= 2048 for _, t in texts)
    from tokenizers import Tokenizer
    transformers = pytest.importorskip('transformers')
    source = transformers.AutoTokenizer.from_pretrained(str(scratch_assets / 'full'), local_files_only=True)
    target = Tokenizer.from_file(str(scratch_assets / 'full' / 'tokenizer.json'))
    build_eval_set = extract_function(TRAINER, 'build_eval_set', {
        'json': json, 'Path': Path, 'make_split_prefix_example': joint_training.make_split_prefix_example})
    generated, canonical = build_eval_set(root / 'eval', target, source)
    assert len(generated) == len(canonical) == 92
