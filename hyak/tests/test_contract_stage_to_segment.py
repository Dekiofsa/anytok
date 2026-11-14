"""Contract between hyak.corpus.stage_inputs and hyak.train.run_segment (asset-free).

Inputs staged NOW by ``stage_inputs.stage`` (not a stored scratch artifact) are accepted by
``run_segment.verify_inputs``, and one fake-trainer bootstrap segment starts from them.  A production
stage lists the three model files, which ``verify_inputs`` must accept with one hash per model file;
a test-scale stage is refused unless the explicit test parameter allows it.
"""
from __future__ import annotations

import os
import shutil

import pytest

from hyak.common.io import read_json
from hyak.corpus import stage_inputs
from hyak.tests import test_train_support as T
from hyak.tests.test_corpus_fixtures import write_pool
from hyak.tests.test_corpus_stage import exact_scorecard
from hyak.train import run_segment as S

TEST_SCALE = {'production': False, 'train_target': 60000, 'eval_target': 15000, 'note': 'test'}


def _restage_world(make_tmp_config, repo_root):
    """A training world whose recipe, corpus, held-out files and input manifest are removed (to be staged)."""
    world = T.build_world(make_tmp_config, repo_root)
    paths = world.paths
    for name in ('corpus', 'eval'):
        shutil.rmtree(paths.input / name)
    for path in (paths.recipe, paths.input_manifest):
        path.unlink()
    return world


def test_production_stage_is_accepted_by_the_training_wrapper(make_tmp_config, repo_root, monkeypatch):
    world = _restage_world(make_tmp_config, repo_root)
    paths = world.paths
    write_pool(paths.corpus_root)
    exact_scorecard(paths.corpus_root)
    report = stage_inputs.stage(paths, model_expected=world.model_expected, log=lambda *a: None)
    assert report['buckets'] == 23
    files = read_json(paths.input_manifest)['files']
    assert set(T.STAGED_MODEL_FILES) <= set(files)
    assert not read_json(paths.input_manifest)['provenance']['corpus']['test_scale']
    calls = []
    real = S.sha256_file
    with monkeypatch.context() as patch:
        patch.setattr(S, 'sha256_file', lambda path, *a: calls.append(os.path.realpath(path)) or real(path, *a))
        result = S.verify_inputs(paths, 'uniform', S.HashCache(paths.runs / 'cache.json'),
                                 init_bytes=world.init_bytes, model_expected=world.model_expected)
    assert result['file_count'] == len(files) and result['corpus_test_scale'] is False
    assert result['model_sha256'] == {name: entry['sha256'] for name, entry in world.model_expected.items()}
    for name in world.model_expected:
        assert calls.count(os.path.realpath(paths.model_dir / name)) == 1, name
    outcome = T.run_once(world, 'uniform', monkeypatch)
    assert outcome['outcome'] == 'bootstrap_done'


def test_test_scale_stage_needs_the_explicit_test_parameter(make_tmp_config, repo_root, monkeypatch):
    world = _restage_world(make_tmp_config, repo_root)
    paths = world.paths
    write_pool(paths.corpus_root, test_scale=TEST_SCALE)
    stage_inputs.stage(paths, allow_test_scale=True, log=lambda *a: None)
    manifest = read_json(paths.input_manifest)
    assert manifest['provenance']['corpus']['test_scale'] is True
    assert not set(T.STAGED_MODEL_FILES) & set(manifest['files'])     # a test-scale stage registers no model
    cache = S.HashCache(paths.runs / 'cache.json')
    with pytest.raises(S.SegmentError, match='test-scale corpus') as caught:
        S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes, model_expected=world.model_expected)
    assert caught.value.exit_code == S.EXIT_USAGE
    result = S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes,
                             model_expected=world.model_expected, allow_test_scale=True)
    assert result['corpus_test_scale'] is True and result['file_count'] == len(manifest['files'])
    with pytest.raises(S.SegmentError, match='test-scale corpus'):
        T.run_once(world, 'uniform', monkeypatch)                 # a production segment refuses it
    assert T.run_once(world, 'uniform', monkeypatch, allow_test_scale=True)['outcome'] == 'bootstrap_done'
