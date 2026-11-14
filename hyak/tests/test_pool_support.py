"""Shared helpers for the hyak/tests/test_pool_*.py tests (no tests in this module).

* ``make_test_plan`` -- a small plan built with the repository's own ``make_plan``
  (scripts/tokenizer_pool.py:45-100) over a synthetic 23-bucket recipe, hashes recomputed.
* ``make_corpus`` -- 23 ``<root>/<pool>-v1/<bucket>/train.jsonl.gz`` files and the generation recipe.
* ``tokenizer_bytes`` -- a real (``tokenizers`` library) tokenizer.json with exactly the requested
  vocabulary, pretokenizer and decoder of scripts/tokenizer_pool.py:150-172.
* ``write_fake_trainer`` -- a stand-in for scripts/tokenizer_pool.py with the same command line,
  stdout stages and manifest fields, whose behaviour per index is set by environment variables.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random

REPO = Path(__file__).resolve().parents[2]


def tokenizer_pool_module():
    spec = importlib.util.spec_from_file_location('anytok_tokenizer_pool_for_tests', REPO / 'scripts/tokenizer_pool.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def synthetic_recipe():
    """The 23 bucket names and pool directories of data/balanced-full-v3.json with small byte counts."""
    real = json.loads((REPO / 'data/balanced-full-v3.json').read_text())
    buckets = []
    for bucket in real['buckets']:
        buckets.append({'name': bucket['name'], 'source_path': bucket['source_path'],
                        'train_target_bytes': bucket['train_target_bytes'],
                        'train_available_bytes': bucket['train_available_bytes']})
    return {'name': 'test-recipe', 'buckets': buckets}


def make_test_plan(path: Path, *, count=4, total_bytes=23_000, seed=7, vocab=None, mutate=None) -> dict:
    """Write a consistent test plan (make_plan + optional vocabulary override) and return it."""
    pool = tokenizer_pool_module()
    plan = pool.make_plan(synthetic_recipe(), count=count, seed=seed, total_bytes=total_bytes)
    for config in plan['configs']:
        if vocab is not None:
            config['vocab_size'] = vocab(config) if callable(vocab) else vocab
        if mutate is not None:
            mutate(config)
        config.pop('config_sha256', None)
        config['config_sha256'] = digest(config)
    if not plan['pilot_indices'] or any(i >= count for i in plan['pilot_indices']):
        plan['pilot_indices'] = list(range(min(2, count)))
    plan['plan_sha256'] = digest(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def make_corpus(root: Path, plan: dict, *, docs_bytes=None) -> Path:
    """Write the 23 train files (enough text for every quota) and the generation recipe; return its path."""
    root = Path(root)
    rng = random.Random(5)
    words = [''.join(rng.choices('abcdefghij', k=5)) for _ in range(300)]
    need = {b['name']: max(c['bucket_bytes'][b['name']] for c in plan['configs']) for b in plan['recipe']['buckets']}
    recipe = json.loads(json.dumps(plan['recipe']))
    for bucket in recipe['buckets']:
        relative = Path(*Path(bucket['source_path']).parts[-2:])
        folder = root / relative
        folder.mkdir(parents=True, exist_ok=True)
        total, docs = 0, []
        target = docs_bytes or (need[bucket['name']] + 2000)
        while total < target:
            text = ' '.join(rng.choices(words, k=40)) + ' 中文 🙂\n'
            docs.append(text)
            total += len(text.encode())
        with gzip.open(folder / 'train.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for i, text in enumerate(docs):
                stream.write(json.dumps({'id': f'{bucket["name"]}-{i}', 'text': text}, ensure_ascii=False) + '\n')
        bucket['source_path'] = str(folder)
    path = root / 'pool-recipe.json'
    path.write_text(json.dumps(recipe, indent=2) + '\n')
    return path


def tokenizer_bytes(algorithm: str, pretokenizer: str, vocab_size: int, extra_tokens=()) -> bytes:
    """A real tokenizer.json with ids 0..vocab_size-1 (specials first, then the byte alphabet)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    specials = ['<|endoftext|>'] if algorithm == 'bpe' else ['<|unk|>', '<|endoftext|>']
    pieces = specials + list(extra_tokens) + alphabet
    n = 0
    while len(pieces) < vocab_size:
        pieces.append(f'tok{n}')
        n += 1
    pieces = pieces[:vocab_size]
    if algorithm == 'unigram':
        model = models.Unigram([(p, 0.0 if p in specials else -5.0 - 1e-3 * i) for i, p in enumerate(pieces)], unk_id=0)
    elif algorithm == 'wordpiece':
        model = models.WordPiece({p: i for i, p in enumerate(pieces)}, unk_token='<|unk|>',
                                 continuing_subword_prefix='', max_input_chars_per_word=1_000_000)
    else:
        model = models.BPE({p: i for i, p in enumerate(pieces)}, [])
    tokenizer = Tokenizer(model)
    if pretokenizer == 'line256':
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r'[^\n]{1,256}\n?|\n'), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=pretokenizer == 'regex')
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(specials)
    return tokenizer.to_str().encode()


def write_tokenizer_library(folder: Path, plan: dict) -> Path:
    """Pre-generate one tokenizer.json per (algorithm, pretokenizer, vocab) of the plan for the fake trainer."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for config in plan['configs']:
        name = f"{config['algorithm']}-{config['pretokenizer']}-{config['vocab_size']}.json"
        if not (folder / name).exists():
            (folder / name).write_bytes(tokenizer_bytes(config['algorithm'], config['pretokenizer'], config['vocab_size']))
    return folder


FAKE_TRAINER = r'''
"""Stand-in for scripts/tokenizer_pool.py --train-config C --recipe R --destination D (tests only)."""
import argparse, hashlib, json, os, signal, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--train-config', required=True)
parser.add_argument('--recipe', required=True)
parser.add_argument('--destination', required=True)
args = parser.parse_args()
config = json.loads(Path(args.train_config).read_text())
recipe = json.loads(Path(args.recipe).read_text())
index = config['index']
log = os.environ.get('FAKE_LOG')
if log:
    line = (json.dumps({'index': index, 'pid': os.getpid(), 'rayon': os.environ.get('RAYON_NUM_THREADS'),
                        'parallelism': os.environ.get('TOKENIZERS_PARALLELISM')}) + '\n').encode()
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.write(fd, line)
    os.close(fd)
modes = json.loads(os.environ.get('FAKE_MODES', '{}'))
mode = modes.get(str(index), os.environ.get('FAKE_MODE', 'ok'))
print(json.dumps({'stage': 'training', 'index': index}), flush=True)
for bucket in recipe['buckets']:
    if not (Path(bucket['source_path']) / 'train.jsonl.gz').is_file():
        sys.exit('missing corpus file for ' + bucket['name'])
if mode == 'sigkill':
    os.kill(os.getpid(), signal.SIGKILL)
if mode == 'poison':  # the job dies with this tokenizer (as a step-wide memory kill would)
    time.sleep(0.3)
    os.kill(os.getppid(), signal.SIGKILL)
    time.sleep(5)
if mode == 'fail':
    print('RuntimeError: simulated failure', file=sys.stderr, flush=True)
    sys.exit(1)
if mode == 'fail_once':
    marker = Path(os.environ['FAKE_STATE']) / f'{index}.failed'
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        sys.exit(1)
if mode.startswith('sleep'):
    time.sleep(float(mode[5:] or 60))
time.sleep(float(os.environ.get('FAKE_DELAY', '0')))
raw = (Path(os.environ['FAKE_TOKENIZERS']) /
       f"{config['algorithm']}-{config['pretokenizer']}-{config['vocab_size']}.json").read_bytes()
destination = Path(args.destination)
destination.mkdir(parents=True, exist_ok=True)
(destination / 'tokenizer.json').write_bytes(raw)
print(json.dumps({'stage': 'corpus_consumed', 'used_bytes': sum(config['bucket_bytes'].values())}), flush=True)
stats = {b['name']: {'quota_bytes': config['bucket_bytes'][b['name']], 'used_bytes': config['bucket_bytes'][b['name']],
                     'documents': 1, 'documents_sha256': '0' * 64} for b in recipe['buckets']}
used = sum(s['used_bytes'] for s in stats.values()) - (100 if mode == 'bad_slack' else 0)
manifest = {'status': 'complete', 'config': config, 'source_stats': stats,
            'vocab_size_actual': config['vocab_size'] - (1 if mode == 'bad_vocab' else 0),
            'train_seconds': 0.25, 'elapsed_seconds': 0.3, 'cpu_seconds': 0.2, 'peak_rss_gib': 0.05,
            'estimated_process_cost_usd': 0.0, 'total_used_bytes': used, 'validation': {},
            'tokenizer_bytes': len(raw), 'tokenizer_sha256': hashlib.sha256(raw).hexdigest()}
(destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'stage': 'complete', 'index': index}), flush=True)
'''


def write_fake_trainer(folder: Path) -> Path:
    path = Path(folder) / 'fake_tokenizer_pool.py'
    path.write_text(FAKE_TRAINER)
    return path
