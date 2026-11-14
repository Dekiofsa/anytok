"""hyak.train.replay_check: anchors in the original CSV, the RNG schedule, and the example replay logic."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import pytest

from hyak.common.io import read_json
from hyak.train import replay_check as RC
from hyak.tests.test_train_support import isolated_imports

#: Bytecode under runtime/ when this module was collected: a later file is blamed on the test that wrote it,
#: not one left by an earlier import or a forgotten PYTHONDONTWRITEBYTECODE.
PYC_AT_COLLECTION = frozenset((Path(__file__).resolve().parents[2] / 'runtime').rglob('*.pyc'))


def test_original_csv_anchor_rows(repo_root):
    rows = RC.read_original(repo_root / RC.ORIGINAL_CSV, 3)
    as_tuples = [(r['step'], r['tokenizer_config_index'], r['tokenizer_vocab_size'], r['scored_tokens'],
                  r['rejected_examples'], r['truncated_forms']) for r in rows]
    assert as_tuples == [(1, 433, 127000, 8636, 0, 588), (2, 858, 59000, 8755, 0, 332), (3, 85, 80000, 8418, 0, 51)]
    assert len(RC.read_original(repo_root / RC.ORIGINAL_CSV)) == 20000


def test_schedule_reproduces_all_20000_pool_indices_and_the_rng_state(repo_root):
    report = RC.schedule_report(repo_root, 20000)
    assert report['all_match'] and report['compared_updates'] == 20000
    for column in ('tokenizer_config_index', 'tokenizer_vocab_size'):
        assert report['columns'][column]['matched'] == 20000
    # the Python RNG after 16 + 20,000 choices(k=128) equals the state read from the original step-020000
    assert report['python_rng_digest_matches'] is True
    assert report['python_rng_digest'] == read_json(repo_root / RC.RNG_PROVENANCE)['python_rng_digest']


def test_schedule_is_sensitive_to_rng_consumption(repo_root):
    recipe = read_json(repo_root / RC.REPO_RECIPE)
    weights = [b['train_target_bytes'] for b in recipe['buckets']]
    original = RC.read_original(repo_root / RC.ORIGINAL_CSV, 200)
    import numpy as np
    rng = np.random.default_rng(42)                        # without the two sanity-tokenizer lognormal draws
    shifted = [{'step': s, 'tokenizer_config_index': int(rng.integers(1000))} for s in range(1, 201)]
    result = RC.compare(shifted, original, columns=('tokenizer_config_index',))
    assert not result['all_match'] and result['columns']['tokenizer_config_index']['rate'] < 0.05
    assert RC.replay_schedule(weights, 3)['pool_indices'] == [433, 858, 85]


def test_cli_schedule_only(repo_root, tmp_path, capsys):
    output = tmp_path / 'schedule.json'
    assert RC.main(['--schedule-only', '--updates', '500', '--output', str(output)]) == 0
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line['all_match'] and line['compared_updates'] == 500 and read_json(output)['mode'] == 'schedule'


# --- example replay on a synthetic mini world ---------------------------------------------------------------
WORDS = ('the of and to in is was for on with as by at from this that which model token text data code '
         'value result table figure section number system method paper river city music art history '
         'función ciudad música historia ville musique histoire город музыка история').split()


def _document(rng: random.Random, bucket: str, n: int) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(40, 400))]
    if bucket == 'code':
        return '\n'.join(f'def f{i}(x):\n    return x + {i}' for i in range(rng.randint(3, 20)))
    return f'{bucket} {n}: ' + ' '.join(words) + '.'


def build_mini_corpus(root: Path, recipe: dict, seed: int = 7, drop_first: str | None = None) -> Path:
    data = root / 'corpus'
    for index, bucket in enumerate(recipe['buckets']):
        name = bucket['name']
        rng = random.Random(seed * 1000 + index)
        docs = [_document(rng, name, n) for n in range(60)]
        if name == drop_first:
            docs = docs[1:]
        path = data / name / 'train.jsonl.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for i, text in enumerate(docs):
                stream.write(json.dumps({'id': str(i), 'text': text}, ensure_ascii=False) + '\n')
    return data


def train_tokenizers(root: Path, texts: list) -> list:
    """BPE, WordPiece and Unigram with the pool's construction (scripts/tokenizer_pool.py:144-176), stock library."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    out = []
    for algorithm, size in (('bpe', 420), ('wordpiece', 450), ('unigram', 480)):
        common = dict(vocab_size=size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        common['special_tokens'] = ['<|endoftext|>'] if algorithm == 'bpe' else ['<|unk|>', '<|endoftext|>']
        if algorithm == 'bpe':
            tokenizer = Tokenizer(models.BPE())
            trainer = trainers.BpeTrainer(**common, min_frequency=2, max_token_length=64)
        elif algorithm == 'wordpiece':
            tokenizer = Tokenizer(models.WordPiece(unk_token='<|unk|>', max_input_chars_per_word=1_000_000))
            trainer = trainers.WordPieceTrainer(**common, min_frequency=2, continuing_subword_prefix='')
        else:
            tokenizer = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(**common, max_piece_length=64, unk_token='<|unk|>')
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(texts, trainer=trainer)
        path = root / f'{algorithm}.json'
        tokenizer.save(str(path))
        out.append((algorithm, path, tokenizer.get_vocab_size()))
    return out


def build_mini_pool(root: Path, tokenizers: list) -> Path:
    """A 1,000-entry pool in the trainer's format; entry i uses tokenizer i % 3 (hard links)."""
    pool = root / 'pool'
    configs = [{'index': i, 'algorithm': tokenizers[i % 3][0], 'vocab_size': tokenizers[i % 3][2]} for i in range(1000)]
    plan = {'plan_sha256': 'mini-plan', 'configs': configs}
    (pool / 'tokenizers').mkdir(parents=True)
    (pool / 'plan.json').write_text(json.dumps(plan))
    entries = []
    for i, config in enumerate(configs):
        source = tokenizers[i % 3][1]
        target = pool / 'tokenizers' / f'{i:04d}' / 'tokenizer.json'
        target.parent.mkdir()
        os.link(source, target)
        entries.append({'status': 'complete', 'config': config, 'vocab_size_actual': config['vocab_size'],
                        'tokenizer_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    (pool / 'full-results.json').write_text(json.dumps({'plan_sha256': 'mini-plan', 'completed': 1000,
                                                        'tokenizers': entries}))
    return pool


def write_csv(path: Path, rows: list):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', *RC.COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.slow
def test_example_replay_detects_a_shifted_stream(tmp_path, repo_root, scratch_assets, make_tmp_config, capsys):
    recipe_path = repo_root / RC.REPO_RECIPE
    recipe = read_json(recipe_path)
    data = build_mini_corpus(tmp_path / 'faithful', recipe)
    texts = []
    for bucket in recipe['buckets']:
        with gzip.open(data / bucket['name'] / 'train.jsonl.gz', 'rt', encoding='utf-8') as stream:
            texts += [json.loads(line)['text'] for line in stream]
    pool_dir = build_mini_pool(tmp_path, train_tokenizers(tmp_path, texts))
    model = scratch_assets / 'full'
    updates = 12
    with isolated_imports():
        helpers = RC.load_helpers(repo_root / 'runtime/anytok_pool_20k')
        source = RC.load_source_tokenizer(model)
        fallback = RC.fallback_ids(source, helpers['missing_byte_symbols'])
        assert len(fallback) == 21                          # SmolLM2 lacks 21 byte symbols (trainer-core 3.4.3)
        pool = helpers['TokenizerPool'](pool_dir)
        first = RC.replay_examples(helpers, recipe_path, data, pool, source, fallback, updates)
        assert first['failure'] is None and len(first['rows']) == updates
        assert [r['tokenizer_config_index'] for r in first['rows'][:3]] == [433, 858, 85]
        assert all(r['scored_tokens'] > 0 for r in first['rows'])
        again = RC.replay_examples(helpers, recipe_path, data, helpers['TokenizerPool'](pool_dir), source, fallback,
                                   updates)
        assert again['rows'] == first['rows']               # deterministic
        expected = first['rows']
        assert RC.compare(again['rows'], expected)['all_match']
        shifted_data = build_mini_corpus(tmp_path / 'shifted', recipe, drop_first='web-en')
        shifted = RC.replay_examples(helpers, recipe_path, shifted_data, helpers['TokenizerPool'](pool_dir), source,
                                     fallback, updates)
    result = RC.compare(shifted['rows'], expected)
    assert not result['all_match']
    assert result['columns']['scored_tokens']['matched'] < updates
    assert result['columns']['tokenizer_config_index']['rate'] == 1.0     # the schedule does not depend on text
    assert result['columns']['truncated_forms']['rate'] == 1.0            # nor do the tokenizers' surface forms
    assert result['mismatches'][0]['step'] == result['columns']['scored_tokens']['first_mismatch_step']
    # command line, against a CSV fabricated from the faithful replay
    expected_csv = tmp_path / 'expected.csv'
    write_csv(expected_csv, expected)
    world = make_tmp_config()
    common = ['--updates', str(updates), '--csv', str(expected_csv), '--code-dir',
              str(repo_root / 'runtime/anytok_pool_20k'), '--recipe', str(recipe_path), '--pool', str(pool_dir),
              '--model', str(model)]
    with isolated_imports():
        assert RC.main(common + ['--data-dir', str(data), '--output', str(tmp_path / 'ok.json'), '--strict']) == 0
    report = read_json(tmp_path / 'ok.json')
    assert report['all_match'] and report['inputs']['pool_index_sha256'] == hashlib.sha256(
        (pool_dir / 'full-results.json').read_bytes()).hexdigest()
    with isolated_imports():
        assert RC.main(common + ['--data-dir', str(shifted_data), '--output', str(tmp_path / 'bad.json'),
                                 '--strict']) == 4
        assert RC.main(common + ['--data-dir', str(shifted_data), '--output', str(tmp_path / 'bad.json')]) == 0
    status = read_json(world.paths.monitor_dir('replay-check') / 'status.json')
    assert status['state'] == 'complete' and status['completed'] == updates and status['all_match'] is False
    written = sorted(str(p) for p in set((repo_root / 'runtime').rglob('*.pyc')) - PYC_AT_COLLECTION)
    assert not written, f'the replay check wrote bytecode into the pinned code: {written}'


@pytest.fixture
def mini_input(scratch_dir):
    path = scratch_dir / 'mini-input'
    if not (path / 'recipe.json').is_file() or not (path / 'corpus' / 'manifest.json').is_file():
        pytest.skip(f'{path} (miniature of the regenerated corpus) not available')
    return path


@pytest.mark.slow
def test_replay_on_the_regenerated_mini_corpus(mini_input, tmp_path, repo_root, scratch_assets):
    """Real text of the 23 pinned sources (a miniature of the corpus build), with three pool-format tokenizers."""
    import shutil
    recipe = read_json(mini_input / 'recipe.json')
    data = tmp_path / 'corpus'
    shutil.copytree(mini_input / 'corpus', data)                # follows the symlinks into the pool directories
    texts = []
    for bucket in recipe['buckets']:
        with gzip.open(data / bucket['name'] / 'train.jsonl.gz', 'rt', encoding='utf-8') as stream:
            texts += [json.loads(line)['text'][:4000] for _, line in zip(range(20), stream)]
    pool_dir = build_mini_pool(tmp_path, train_tokenizers(tmp_path, texts))
    with isolated_imports():
        helpers = RC.load_helpers(repo_root / 'runtime/anytok_pool_20k')
        source = RC.load_source_tokenizer(scratch_assets / 'full')
        fallback = RC.fallback_ids(source, helpers['missing_byte_symbols'])
        replay = RC.replay_examples(helpers, mini_input / 'recipe.json', data, helpers['TokenizerPool'](pool_dir),
                                    source, fallback, 8)
        assert replay['failure'] is None and [r['step'] for r in replay['rows']] == list(range(1, 9))
        # drop the first document of web-en (16% of all draws): the stream shifts, the schedule does not
        path = data / 'web-en' / 'train.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            lines = stream.readlines()
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.writelines(lines[1:])
        shifted = RC.replay_examples(helpers, mini_input / 'recipe.json', data, helpers['TokenizerPool'](pool_dir),
                                     source, fallback, 8)
    result = RC.compare(shifted['rows'], replay['rows'])
    assert result['columns']['tokenizer_config_index']['rate'] == 1.0
    assert result['columns']['scored_tokens']['matched'] < 8


def test_verify_inputs_accepts_the_corpus_stage_output(mini_input, make_tmp_config, repo_root):
    """input/manifest.json and corpus/manifest.json as hyak.corpus.stage_inputs writes them."""
    import shutil
    from hyak.tests.test_train_support import build_world
    from hyak.train import run_segment as S
    world = build_world(make_tmp_config, repo_root)
    paths = world.paths
    for name in ('corpus', 'eval'):
        shutil.rmtree(paths.input / name)
        shutil.copytree(mini_input / name, paths.input / name)
    for name in ('recipe.json', 'pool-recipe.json', 'manifest.json'):
        shutil.copyfile(mini_input / name, paths.input / name)
    cache = S.HashCache(paths.runs / 'cache.json')
    # SCRATCH/mini-input is a test-scale stage: refused unless the explicit test parameter allows it
    with pytest.raises(S.SegmentError, match='test-scale corpus'):
        S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes, model_expected=world.model_expected)
    result = S.verify_inputs(paths, 'uniform', cache, init_bytes=world.init_bytes,
                             model_expected=world.model_expected, allow_test_scale=True)
    assert result['file_count'] == len(read_json(mini_input / 'manifest.json')['files'])
    assert result['corpus_test_scale'] is True


# --- review fix round (train-state-7, train-state-v1, env-10) ----------------------------------------------------
def _doctored_csv(repo_root, path, limit=None):
    with open(repo_root / RC.ORIGINAL_CSV, newline='') as src, open(path, 'w', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for n, row in enumerate(reader):
            if limit is not None and n >= limit:
                break
            row['tokenizer_config_index'] = str((int(row['tokenizer_config_index']) + 1) % 1000)
            writer.writerow(row)


def test_schedule_mismatch_fails_the_job_without_strict(repo_root, tmp_path, capsys):
    doctored = tmp_path / 'doctored.csv'
    _doctored_csv(repo_root, doctored, limit=500)
    assert RC.main(['--schedule-only', '--updates', '500', '--csv', str(doctored),
                    '--output', str(tmp_path / 'a.json')]) == 4
    out = capsys.readouterr()
    assert json.loads(out.out.splitlines()[-1])['all_match'] is False and 'does not match the original' in out.err


def test_schedule_digest_mismatch_fails_the_job(repo_root, tmp_path, capsys, monkeypatch):
    real = RC.replay_schedule

    def other_digest(weights, updates, **kw):
        result = real(weights, updates, **kw)
        result['python_rng_digest'] = 'f' * 64
        return result

    monkeypatch.setattr(RC, 'replay_schedule', other_digest)
    assert RC.main(['--schedule-only', '--output', str(tmp_path / 's.json')]) == 4
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line['python_rng_digest_matches'] is False and line['all_match'] is False


def test_default_example_replay_is_2000_updates():
    assert RC.DEFAULT_EXAMPLE_UPDATES == 2000


def test_helpers_are_checked_against_the_pins(repo_root, tmp_path, tmp_config):
    import shutil
    code = tmp_path / 'code'
    shutil.copytree(repo_root / 'runtime/anytok_pool_20k/hf_hypernet', code / 'hf_hypernet',
                    ignore=shutil.ignore_patterns('__pycache__'))
    assert set(RC.verify_helpers(code)) == {f'hf_hypernet/{n}' for n in
                                           ('__init__.py', 'configuration_hypernet.py', 'hyper_roberta.py',
                                            'hyper_t5.py', 'identity_warmup.py', 'joint_training.py',
                                            'modeling_hypernet.py', 'pool_sampling.py', 'smollm_setup.py',
                                            'tokenizer_inputs.py')}
    target = code / 'hf_hypernet' / 'joint_training.py'
    target.write_text(target.read_text() + '\nTAMPERED = True\n')
    with isolated_imports():          # purges hf_hypernet* on entry, so only load_helpers could add it back
        with pytest.raises(RC.HelperMismatch, match='joint_training.py'):
            RC.load_helpers(code)
        assert 'hf_hypernet.joint_training' not in sys.modules   # the tampered copy was never imported
    with isolated_imports():
        assert RC.main(['--code-dir', str(code), '--updates', '5', '--output', str(tmp_path / 'x.json')]) == 2


@pytest.mark.slow
def test_trainer_failure_fails_the_job(tmp_path, repo_root, scratch_assets, make_tmp_config, capsys):
    """An update at which the real trainer would raise fails the replay (exit 4, status failed), --strict or not."""
    recipe_path = repo_root / RC.REPO_RECIPE
    recipe = read_json(recipe_path)
    data = build_mini_corpus(tmp_path / 'faithful', recipe)
    texts = []
    for bucket in recipe['buckets']:
        with gzip.open(data / bucket['name'] / 'train.jsonl.gz', 'rt', encoding='utf-8') as stream:
            texts += [json.loads(line)['text'] for line in stream]
    pool_dir = build_mini_pool(tmp_path, train_tokenizers(tmp_path, texts))
    model = scratch_assets / 'full'
    updates = 6
    with isolated_imports():
        helpers = RC.load_helpers(repo_root / 'runtime/anytok_pool_20k')
        source = RC.load_source_tokenizer(model)
        fallback = RC.fallback_ids(source, helpers['missing_byte_symbols'])
        faithful = RC.replay_examples(helpers, recipe_path, data, helpers['TokenizerPool'](pool_dir), source,
                                      fallback, updates)
    expected_csv = tmp_path / 'expected.csv'
    write_csv(expected_csv, faithful['rows'])
    bad = build_mini_corpus(tmp_path / 'bad', recipe)
    path = bad / 'wiki-en' / 'train.jsonl.gz'           # no wiki-en document can round-trip (drawn in update 2)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        docs = [json.loads(line)['text'] for line in stream]
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for i, text in enumerate(docs):
            stream.write(json.dumps({'id': str(i), 'text': '<|endoftext|> ' + text[:1500]}, ensure_ascii=False) + '\n')
    world = make_tmp_config()
    common = ['--updates', str(updates), '--csv', str(expected_csv), '--code-dir',
              str(repo_root / 'runtime/anytok_pool_20k'), '--recipe', str(recipe_path), '--pool', str(pool_dir),
              '--model', str(model), '--data-dir', str(bad), '--output', str(tmp_path / 'bad.json')]
    for extra in ([], ['--strict']):
        capsys.readouterr()
        with isolated_imports():
            assert RC.main(common + extra) == 4
        line = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert line['all_match'] is False and line['trainer_failure']['step'] == 2
    status = read_json(world.paths.monitor_dir('replay-check') / 'status.json')
    assert status['state'] == 'failed' and 'would raise at update 2' in status['error']
