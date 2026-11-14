"""Pinned identities, plan loading (exact 63-bit seeds) and the shared manifest/tokenizer/recipe checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

import hyak.pool as pool
from hyak.pool import (PoolError, digest, load_plan, manifest_problems, parse_indices, plan_problems,
                       recipe_problems, tokenizer_json_problems)
from hyak.tests.test_pool_support import make_test_plan, tokenizer_bytes, tokenizer_pool_module

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_pinned_hashes_match_the_repository():
    assert sha(REPO / pool.PLAN_REL) == pool.PLAN_FILE_SHA256
    assert (REPO / pool.PLAN_REL).stat().st_size == pool.PLAN_FILE_BYTES
    assert sha(REPO / pool.WORKER_REL) == pool.WORKER_SHA256
    assert sha(REPO / pool.POOL_SAMPLING_REL) == pool.POOL_SAMPLING_SHA256
    assert sha(REPO / pool.TOKENIZER_INPUTS_REL) == pool.TOKENIZER_INPUTS_SHA256
    assert sha(REPO / pool.SETTINGS_POLICY_REL) == pool.SETTINGS_POLICY_SHA256
    assert sha(REPO / pool.SPEC_REL) == pool.SPEC_SHA256
    assert sha(REPO / pool.UNIFORM_CSV_REL) == pool.UNIFORM_CSV_SHA256
    assert sha(REPO / pool.REWARD_TABLE_REL) == pool.REWARD_TABLE_SHA256
    for name, value in pool.PATCH_SHA256.items():
        assert sha(REPO / 'patches' / name) == value
    assert pool.FORBIDDEN_PATCH not in pool.PATCH_FILES
    spec = json.loads((REPO / pool.SPEC_REL).read_text())
    assert spec['pool_index_sha256'] == pool.ORIGINAL_INDEX_SHA256 and spec['pool_plan_sha256'] == pool.PLAN_SHA256


def test_patch_order_and_blob_ids_match_the_build_script_and_patch_headers():
    source = (REPO / 'scripts/build_fast_tokenizers.py').read_text()
    assert "names = ['all-candidate-noise', 'unigram-literal-unk', 'unigram-stack-drop']" in source
    assert f"REVISION = '{pool.UPSTREAM_REVISION}'" in source
    assert pool.PATCH_NAMES == ('all-candidate-noise', 'unigram-literal-unk', 'unigram-stack-drop')
    seen = {}
    for name in pool.PATCH_FILES:
        text = (REPO / 'patches' / name).read_text()
        for path, before in re.findall(r'^diff --git a/(\S+) b/\S+\nindex ([0-9a-f]+)\.\.[0-9a-f]+', text, re.M):
            seen[path] = before
    assert set(seen) == set(pool.UPSTREAM_BLOBS)
    for path, blob in pool.UPSTREAM_BLOBS.items():
        assert blob.startswith(seen[path]) and len(blob) == 40


def test_production_plan_loads_with_exact_seeds():
    plan, info = load_plan(REPO / pool.PLAN_REL)
    assert info['production'] and info['file_sha256'] == pool.PLAN_FILE_SHA256
    assert plan['plan_sha256'] == pool.PLAN_SHA256 and len(plan['configs']) == 1000
    assert plan['configs'][0]['seed'] == pool.PLAN_SEED_0 and isinstance(plan['configs'][0]['seed'], int)
    assert sum(c['seed'] > 2 ** 53 for c in plan['configs']) > 900
    assert tuple(plan['pilot_indices']) == pool.PILOT_INDICES
    assert pool.max_slack(plan) == 69
    # Parsing numbers as doubles (JavaScript, old jq) corrupts the seeds; the checks notice it.
    lossy = json.loads((REPO / pool.PLAN_REL).read_bytes(), parse_int=lambda s: int(float(s)))
    problems = plan_problems(lossy, production=True)
    assert any('config_sha256' in p for p in problems) and any('plan_sha256' in p for p in problems)


def test_digest_is_the_original_function():
    module = tokenizer_pool_module()
    value = {'b': [1, 2**63 - 1, None], 'a': 'é'}
    assert digest(value) == module.digest(value)


def test_any_other_plan_file_is_refused(tmp_path):
    raw = (REPO / pool.PLAN_REL).read_bytes()
    copy = tmp_path / 'plan.json'
    copy.write_bytes(raw + b'\n')                      # same JSON, different bytes
    with pytest.raises(PoolError, match='not the saved production plan'):
        load_plan(copy)
    plan, info = load_plan(copy, allow_test_plan=True)  # consistent, but not production
    assert not info['production']
    tampered = json.loads(raw)
    tampered['configs'][5]['vocab_size'] += 1000
    copy.write_text(json.dumps(tampered))
    with pytest.raises(PoolError, match='config 5: config_sha256 mismatch'):
        load_plan(copy, allow_test_plan=True)


def test_small_test_plans_are_consistent(tmp_path):
    plan = make_test_plan(tmp_path / 'p.json', count=5, vocab=300)
    loaded, info = load_plan(tmp_path / 'p.json', allow_test_plan=True)
    assert loaded == plan and not info['production'] and plan_problems(plan, production=False) == []


def test_parse_indices():
    assert parse_indices(['3', '1,2', '5-7', '3'], 10) == [3, 1, 2, 5, 6, 7]
    assert parse_indices(['pilots'], 1000) == list(pool.PILOT_INDICES)
    assert parse_indices(['pilots', '0'], 4, [2, 1]) == [2, 1, 0]
    for bad in (['x'], ['5-2'], ['10']):
        with pytest.raises(PoolError):
            parse_indices(bad, 10)


def test_recipe_may_differ_only_in_source_path():
    plan, _ = load_plan(REPO / pool.PLAN_REL)
    recipe = json.loads(json.dumps(plan['recipe']))
    for bucket in recipe['buckets']:
        bucket['source_path'] = '/gpfs/projects/x/anytok/corpus/' + '/'.join(Path(bucket['source_path']).parts[-2:])
    assert recipe_problems(recipe, plan['recipe']) == []
    swapped = json.loads(json.dumps(recipe))
    swapped['buckets'][0]['source_path'], swapped['buckets'][1]['source_path'] = (
        swapped['buckets'][1]['source_path'], swapped['buckets'][0]['source_path'])
    assert any('does not end with' in p for p in recipe_problems(swapped, plan['recipe']))
    changed = json.loads(json.dumps(recipe))
    changed['buckets'][3]['train_target_bytes'] += 1
    assert any('fields other than source_path/eval_path' in p for p in recipe_problems(changed, plan['recipe']))
    relative = json.loads(json.dumps(recipe))
    relative['buckets'][2]['source_path'] = 'corpus/full-v1/web-fr'
    assert any('not absolute' in p for p in recipe_problems(relative, plan['recipe']))
    with_eval = json.loads(json.dumps(recipe))
    for bucket in with_eval['buckets']:
        bucket['eval_path'] = '/gpfs/x/corpus/' + '/'.join(Path(bucket['eval_path']).parts[-3:])
    assert recipe_problems(with_eval, plan['recipe']) == []
    with_eval['buckets'][4]['eval_path'] = with_eval['buckets'][5]['eval_path']
    assert any('eval_path' in p and 'does not end with' in p for p in recipe_problems(with_eval, plan['recipe']))
    reordered = json.loads(json.dumps(recipe))
    reordered['buckets'].reverse()
    assert recipe_problems(reordered, plan['recipe']) == ['bucket names or order differ from the plan recipe']


@pytest.mark.parametrize('algorithm', pool.ALGORITHMS)
@pytest.mark.parametrize('pretokenizer', pool.PRETOKENIZERS)
def test_tokenizer_json_checks_accept_the_pool_shapes(algorithm, pretokenizer):
    config = {'algorithm': algorithm, 'pretokenizer': pretokenizer}
    raw = tokenizer_bytes(algorithm, pretokenizer, 300)
    assert tokenizer_json_problems(raw, config, 300) == []
    assert any('token ids' in p for p in tokenizer_json_problems(raw, config, 301))
    other = {'regex': 'no-regex', 'no-regex': 'line256', 'line256': 'regex'}[pretokenizer]
    assert tokenizer_json_problems(raw, {**config, 'pretokenizer': other}, 300)
    data = json.loads(raw)
    data['normalizer'] = {'type': 'NFC'}
    assert any('normalizer' in p for p in tokenizer_json_problems(json.dumps(data).encode(), config, 300))


def test_tokenizer_json_checks_catch_missing_eos_and_bad_scores():
    raw = tokenizer_bytes('unigram', 'regex', 300)
    data = json.loads(raw)
    data['model']['vocab'][5][1] = float('nan')
    assert any('not finite' in p for p in tokenizer_json_problems(json.dumps(data).encode(), {'algorithm': 'unigram', 'pretokenizer': 'regex'}, 300))
    data = json.loads(tokenizer_bytes('bpe', 'regex', 300))
    data['added_tokens'] = []
    data['model']['vocab'] = {('x' + k if k == '<|endoftext|>' else k): v for k, v in data['model']['vocab'].items()}
    assert any('<|endoftext|>' in p for p in tokenizer_json_problems(json.dumps(data).encode(), {'algorithm': 'bpe', 'pretokenizer': 'regex'}, 300))


def test_manifest_checks(tmp_path):
    plan = make_test_plan(tmp_path / 'p.json', count=2, vocab=300)
    config = plan['configs'][0]
    raw = tokenizer_bytes(config['algorithm'], config['pretokenizer'], 300)
    path = tmp_path / 'tokenizer.json'
    path.write_bytes(raw)
    buckets = [b['name'] for b in plan['recipe']['buckets']]
    manifest = {'status': 'complete', 'config': config, 'vocab_size_actual': 300,
                'total_used_bytes': plan['total_bytes_per_tokenizer'] - 69,
                'source_stats': {b: {} for b in buckets}, 'tokenizer_bytes': len(raw),
                'tokenizer_sha256': hashlib.sha256(raw).hexdigest(), 'provenance': {'plan_sha256': 'x'}}
    assert manifest_problems(manifest, config, plan, path) == []
    assert manifest_problems({**manifest, 'total_used_bytes': plan['total_bytes_per_tokenizer'] - 70}, config, plan, path)
    assert manifest_problems({**manifest, 'total_used_bytes': plan['total_bytes_per_tokenizer'] + 1}, config, plan, path)
    assert manifest_problems({**manifest, 'status': 'failed'}, config, plan, path)
    assert manifest_problems({**manifest, 'config': plan['configs'][1]}, config, plan, path)
    assert manifest_problems({k: v for k, v in manifest.items() if k != 'provenance'}, config, plan, path)
    identity = {k: None for k in pool.IDENTITY_KEYS}
    assert any('provenance differs' in p for p in manifest_problems(manifest, config, plan, path, identity=identity))
    path.write_bytes(raw[:-1] + b' ')
    assert any('SHA-256' in p for p in manifest_problems(manifest, config, plan, path))
