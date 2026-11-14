"""CPU replay of the uniform trainer's example construction against the original run (corpus + pool fidelity).

    python -m hyak.train.replay_check [--updates K] [--schedule-only] [--strict] [--output FILE]
                                      [--csv PATH] [--code-dir DIR] [--recipe PATH] [--data-dir DIR]
                                      [--pool DIR] [--model DIR] [--config FILE]

Before any model call the trainer builds each update's 128 examples from the corpus stream, the
seed and the drawn pool tokenizer only (uniform trainer 478-483, 515-542).  Three logged integers
therefore depend on nothing else: ``scored_tokens``, ``rejected_examples`` and ``truncated_forms``
(gaps.md N1).  ``reports/rl-tokenizer-selection-20k/data/training/uniform.csv`` holds them for all
20,000 original updates.  This module replays exactly that part of the trainer with the staged
pinned helpers (``BalancedTextStream``, ``TokenizerPool``, ``make_split_prefix_example``,
``missing_byte_symbols``; the Rust sampler is never called -- ``rust_utils`` imports as an empty
namespace package) and reports the exact-match rate of ``tokenizer_config_index``,
``tokenizer_vocab_size``, ``scored_tokens``, ``rejected_examples`` and ``truncated_forms`` for the first
K updates (default 2,000: 860 of the 1,000 tokenizers, original rejections at updates 381, 619 and
1312; about an hour on one CPU thread).  With a faithful corpus and pool every column matches.  The
trainer would raise "Too many non-roundtripping or oversized examples" where 128 replacements are
not enough; the replay reports that update and stops, and the job FAILS (the uniform run would
crash at the same update).

``--schedule-only`` needs no corpus or pool: it replays ``random.Random(42)`` (16 priming draws and
one ``choices(k=128)`` per update) and ``numpy.random.default_rng(42)`` (two ``lognormal`` draws for
the sanity tokenizers, then one ``integers(1000)`` per update) and compares the pool index of all
20,000 rows, their vocabulary sizes through the plan, and the Python RNG state after update 20,000
with the digest read from the original ``step-020000/state.pt``
(``reports/rl-tokenizer-selection-20k/data/provenance/uniform-rng.json``).

Before importing them, the helpers' files (every ``hf_hypernet`` module the replay imports) are
compared with the pinned snapshot hashes (``hyak.code.pinned``); a difference is an error.

Exit status (project contract): 0 when the replay ran and, for the example replay, matched or
mismatched in columns only (a fidelity result, written to the report); 4 (a check failed) when the
schedule replay does not match the original (pool index, vocabulary size or the Python RNG digest),
when the example replay hits an update at which the trainer would raise, and with ``--strict`` when
any example column mismatches; 2 for missing or wrong inputs, staged code or environment; 1 only
for an unexpected exception (traceback on stderr).  The report (JSON) goes to ``--output``
(default ``<repo>/modal_results/hyak_tillicum/replay-check/replay-check-<examples|schedule>-<K>.json``);
progress to the ``replay-check`` status files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now

ORIGINAL_CSV = 'reports/rl-tokenizer-selection-20k/data/training/uniform.csv'
RNG_PROVENANCE = 'reports/rl-tokenizer-selection-20k/data/provenance/uniform-rng.json'
REPO_RECIPE = 'data/balanced-full-v3.json'
REPO_PLAN = 'plans/tokenizer-pool-1000-v1.json'
COLUMNS = ('tokenizer_config_index', 'tokenizer_vocab_size', 'scored_tokens', 'rejected_examples', 'truncated_forms')
SEED = 42
ORIGINAL_UPDATES = 20000
POOL_SIZE = 1000
#: Default length of the example replay (lead decision of the review fix round).
DEFAULT_EXAMPLE_UPDATES = 2000
EXIT_USAGE = 2
EXIT_CHECK = 4
#: Pinned helper modules the replay imports (their files are hash-checked before use).
HELPER_MODULES = ('hf_hypernet.joint_training', 'hf_hypernet.pool_sampling', 'hf_hypernet.tokenizer_inputs')


class HelperMismatch(RuntimeError):
    """The staged helpers are not the pinned snapshot files."""
#: Constants of the pinned helpers and trainer (joint_training.py:24-25; trainer 518-533).
INITIAL_QUEUE_BATCHES = 16
QUEUE_BATCH_SIZE = 128
BATCH = 128
TEXT_CHARS = 2048
SNIPPET_CHARS = 512
MIN_CHARS = 32
MAX_REPLACEMENTS = 128
SPECIAL_TOKEN = '<|endoftext|>'
SPECIAL_ID = 0


def rng_digest(value) -> str:
    """Digest of an RNG state as the repository computes it (scripts/pool_generalization.py:15-16)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_original(csv_path, limit=None) -> list:
    """Rows ``{step, tokenizer_config_index, tokenizer_vocab_size, scored_tokens, rejected_examples, truncated_forms}``."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            rows.append({'step': int(row['step']), **{column: int(row[column]) for column in COLUMNS}})
            if limit is not None and len(rows) >= limit:
                break
    if [row['step'] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f'{csv_path}: steps are not 1..{len(rows)} in order')
    return rows


def replay_schedule(weights, updates: int, *, seed: int = SEED, pool_size: int = POOL_SIZE) -> dict:
    """Pool index per update and the final Python RNG digest, without corpus or pool.

    trainer 370-371 (seeds), 478-480 (16 priming ``choices``), 494-496 with joint_training.py:129
    (two ``lognormal`` draws for the sanity tokenizers), 518 (``choices(k=128)``) and 520
    (``integers(len(pool.entries))``).  ``random.choices`` consumes one ``random()`` per draw
    whatever the population, so bucket indices stand in for names (as
    scripts/analyze_coverage_evidence.py:73-78 does).
    """
    py_rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    population = list(range(len(weights)))
    for _ in range(INITIAL_QUEUE_BATCHES):
        py_rng.choices(population, weights=weights, k=QUEUE_BATCH_SIZE)
    for _ in range(2):
        noise_rng.lognormal(mean=math.log(1e-5), sigma=4.0)
    indices = []
    for _ in range(updates):
        py_rng.choices(population, weights=weights, k=BATCH)
        indices.append(int(noise_rng.integers(pool_size)))
    return {'pool_indices': indices, 'python_rng_digest': rng_digest(py_rng.getstate())}


def pinned_helper_hashes() -> dict:
    """``{path inside the staged tree: sha256}`` of the uniform tree's ``hf_hypernet`` files (hyak.code.pinned)."""
    from hyak.code import pinned
    return {rel: pinned.SNAPSHOT_FILES[source][0] for rel, source in pinned.tree_layout('uniform').items()
            if rel.startswith('hf_hypernet/')}


def verify_helpers(code_dir, expected=None) -> dict:
    """Every ``hf_hypernet`` file of ``code_dir`` that the replay may import equals its pin (env-10)."""
    code_dir = Path(code_dir)
    expected = pinned_helper_hashes() if expected is None else expected
    found, problems = {}, []
    for rel, digest in sorted(expected.items()):
        path = code_dir / rel
        if not path.is_file():
            problems.append(f'{rel} missing')
            continue
        found[rel] = sha256_file(path)
        if found[rel] != digest:
            problems.append(f'{rel} has SHA-256 {found[rel][:16]}..., pinned {digest[:16]}...')
    if problems:
        raise HelperMismatch(f'{code_dir} is not the pinned uniform tree: {"; ".join(problems)}. '
                             'Remedy: python -m hyak.code.stage_code --verify')
    return found


def load_helpers(code_dir, expected=None):
    """Import the pinned helpers from ``code_dir`` (no bytecode is written next to them), after checking
    their files against the pins (``expected`` is a test parameter only)."""
    code_dir = Path(code_dir).resolve()
    verify_helpers(code_dir, expected)
    sys.dont_write_bytecode = True
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import hf_hypernet.joint_training as joint
    from hf_hypernet import pool_sampling, tokenizer_inputs
    for module in (joint, pool_sampling, tokenizer_inputs):
        if Path(module.__file__).resolve().parent != code_dir / 'hf_hypernet':
            raise RuntimeError(f'{module.__name__} was imported from {module.__file__}, not from {code_dir}')
    for name, module in list(sys.modules.items()):
        location = getattr(module, '__file__', None)
        if name.startswith('hf_hypernet') and location and Path(location).resolve().is_relative_to(code_dir):
            rel = Path(location).resolve().relative_to(code_dir).as_posix()
            if rel not in (expected if expected is not None else pinned_helper_hashes()):
                raise HelperMismatch(f'{name} was imported from {location}, which is not a pinned file')
    if (joint.INITIAL_QUEUE_BATCHES, joint.QUEUE_BATCH_SIZE) != (INITIAL_QUEUE_BATCHES, QUEUE_BATCH_SIZE):
        raise RuntimeError('pinned queue constants differ from this replay')
    return {'BalancedTextStream': joint.BalancedTextStream,
            'make_split_prefix_example': joint.make_split_prefix_example,
            'TokenizerPool': pool_sampling.TokenizerPool,
            'missing_byte_symbols': tokenizer_inputs.missing_byte_symbols}


def load_source_tokenizer(model_dir):
    """``AutoTokenizer.from_pretrained(--model, local_files_only=True)`` with the trainer's EOS check (421-423)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN) != SPECIAL_ID:
        raise ValueError("This runner expects SmolLM2's end-of-text token at ID zero")
    return tokenizer


def fallback_ids(source_tokenizer, missing_byte_symbols) -> dict:
    """The trainer's ``fallback`` mapping without a hypernetwork (tokenizer_inputs.py:28-39)."""
    size = len(source_tokenizer)
    return {symbol: size + 1 + i for i, symbol in enumerate(missing_byte_symbols(source_tokenizer))}


def replay_examples(helpers, recipe, data_dir, pool, source_tokenizer, fallback, updates: int, *,
                    seed: int = SEED, progress=None) -> dict:
    """Replay update 1..K of a fresh uniform start; returns ``{"rows", "python_rng_digest", "failure"}``.

    Operation for operation the trainer's lines 370-371, 473, 478-483, 494-496 and 515-542, minus
    everything that needs a model (the queue is kept by the trainer but never read in the loop).
    """
    corpus = helpers['BalancedTextStream'](recipe, data_dir, None)
    make_example = helpers['make_split_prefix_example']
    py_rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    rows, failure = [], None
    try:
        for _ in range(INITIAL_QUEUE_BATCHES):
            names = py_rng.choices(corpus.names, weights=corpus.weights, k=QUEUE_BATCH_SIZE)
            for name in names:
                corpus.next_text(name, chars=TEXT_CHARS)
        for _ in range(2):
            noise_rng.lognormal(mean=math.log(1e-5), sigma=4.0)
        for update in range(updates):
            names = py_rng.choices(corpus.names, weights=corpus.weights, k=BATCH)
            texts = [corpus.next_text(name, chars=TEXT_CHARS) for name in names]
            pool_index = int(noise_rng.integers(len(pool.entries)))
            target_tokenizer, vocab = pool.load(pool_index)
            examples = []
            rejected = 0
            for name, raw in zip(names, texts):
                example = make_example(raw[:SNIPPET_CHARS], target_tokenizer, source_tokenizer, min_chars=MIN_CHARS)
                while example is None and rejected < MAX_REPLACEMENTS:
                    rejected += 1
                    raw = corpus.next_text(name, chars=TEXT_CHARS)
                    example = make_example(raw[:SNIPPET_CHARS], target_tokenizer, source_tokenizer,
                                           min_chars=MIN_CHARS)
                if example is None:
                    failure = {'step': update + 1, 'error': 'Too many non-roundtripping or oversized examples',
                               'tokenizer_config_index': pool.entries[pool_index]['config']['index']}
                    break
                examples.append(example)
            if failure is not None:
                break
            _, truncated = pool.forms(pool_index, source_tokenizer, fallback)
            rows.append({'step': update + 1,
                         'tokenizer_config_index': pool.entries[pool_index]['config']['index'],
                         'tokenizer_vocab_size': len(vocab),
                         'scored_tokens': sum(len(x[1]) for x in examples),
                         'rejected_examples': rejected,
                         'truncated_forms': int(truncated)})
            if progress is not None:
                progress(update + 1)
    finally:
        corpus.close()
    return {'rows': rows, 'python_rng_digest': rng_digest(py_rng.getstate()), 'failure': failure}


def compare(replayed: list, original: list, columns=COLUMNS, keep: int = 50) -> dict:
    """Per-column exact matches of the replayed rows against the original rows of the same steps."""
    by_step = {row['step']: row for row in original}
    stats = {column: {'matched': 0, 'total': 0, 'first_mismatch_step': None} for column in columns}
    mismatches = []
    for row in replayed:
        reference = by_step.get(row['step'])
        if reference is None:
            raise ValueError(f'no original row for step {row["step"]}')
        differing = {}
        for column in columns:
            stats[column]['total'] += 1
            if row[column] == reference[column]:
                stats[column]['matched'] += 1
            else:
                differing[column] = {'replayed': row[column], 'original': reference[column]}
                if stats[column]['first_mismatch_step'] is None:
                    stats[column]['first_mismatch_step'] = row['step']
        if differing and len(mismatches) < keep:
            mismatches.append({'step': row['step'], **differing})
    for column in columns:
        total = stats[column]['total']
        stats[column]['rate'] = stats[column]['matched'] / total if total else None
    all_match = bool(replayed) and all(s['matched'] == s['total'] for s in stats.values())
    return {'columns': stats, 'mismatches': mismatches, 'all_match': all_match, 'compared_updates': len(replayed)}


def schedule_report(repo, updates: int, *, recipe=None, plan=None, csv_path=None) -> dict:
    """Schedule-only comparison (no corpus, no pool) of the first ``updates`` rows of the original CSV."""
    repo = Path(repo)
    recipe_data = read_json(recipe or repo / REPO_RECIPE)
    weights = [bucket['train_target_bytes'] for bucket in recipe_data['buckets']]
    plan_data = read_json(plan or repo / REPO_PLAN)
    original = read_original(csv_path or repo / ORIGINAL_CSV, updates)
    if len(original) < updates:
        raise ValueError(f'the original CSV has only {len(original)} rows')
    schedule = replay_schedule(weights, updates)
    replayed = [{'step': step, 'tokenizer_config_index': index,
                 'tokenizer_vocab_size': plan_data['configs'][index]['vocab_size']}
                for step, index in enumerate(schedule['pool_indices'], 1)]
    result = compare(replayed, original, columns=('tokenizer_config_index', 'tokenizer_vocab_size'))
    result['python_rng_digest'] = schedule['python_rng_digest']
    provenance = repo / RNG_PROVENANCE
    if updates == ORIGINAL_UPDATES and provenance.is_file():
        recorded = read_json(provenance)
        result['python_rng_digest_original'] = recorded.get('python_rng_digest')
        result['python_rng_digest_matches'] = recorded.get('python_rng_digest') == schedule['python_rng_digest']
        if not result['python_rng_digest_matches']:
            result['all_match'] = False
    return result


def _default_output(updates, schedule_only):
    from hyak.common.paths import STAGE_REPLAY_CHECK, default_monitor_dir
    suffix = 'schedule' if schedule_only else 'examples'
    return default_monitor_dir(STAGE_REPLAY_CHECK) / f'replay-check-{suffix}-{updates}.json'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.train.replay_check', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--updates', type=int, default=None,
                        help=f'updates to replay (default {DEFAULT_EXAMPLE_UPDATES}; schedule-only default 20000); '
                             'at most 20000')
    parser.add_argument('--schedule-only', action='store_true',
                        help='RNG schedule only (no corpus, no pool); a mismatch always exits 4')
    parser.add_argument('--strict', action='store_true', help='exit 4 when any example column mismatches')
    parser.add_argument('--csv', type=Path, help=f'original per-update table (default <repo>/{ORIGINAL_CSV})')
    parser.add_argument('--code-dir', type=Path, help='staged uniform code tree (default $ANYTOK_ROOT/code/uniform)')
    parser.add_argument('--recipe', type=Path, help='default $ANYTOK_ROOT/input/recipe.json')
    parser.add_argument('--data-dir', type=Path, help='default $ANYTOK_ROOT/input/corpus')
    parser.add_argument('--pool', type=Path, help='default $ANYTOK_ROOT/input/pool')
    parser.add_argument('--model', type=Path, help='default $ANYTOK_ROOT/input/model')
    parser.add_argument('--output', type=Path, help='report JSON')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    updates = args.updates if args.updates is not None else (ORIGINAL_UPDATES if args.schedule_only
                                                             else DEFAULT_EXAMPLE_UPDATES)
    if not 1 <= updates <= ORIGINAL_UPDATES:
        parser.error(f'--updates must be in 1..{ORIGINAL_UPDATES}')
    from hyak.common.config import load_config, repo_root
    from hyak.common.paths import STAGE_REPLAY_CHECK, Paths
    from hyak.common.status import StatusWriter
    started = time.monotonic()
    repo = repo_root()
    output = args.output or _default_output(updates, args.schedule_only)
    try:
        if args.schedule_only:
            report = schedule_report(repo, updates, recipe=args.recipe, csv_path=args.csv)
            report.update(mode='schedule', updates=updates)
        else:
            cfg = load_config(args.config)
            paths = Paths.from_config(cfg)
            code_dir = args.code_dir or paths.code_dir('uniform')
            recipe = args.recipe or paths.recipe
            data_dir = args.data_dir or paths.input_corpus
            pool_dir = args.pool or paths.pool_dir
            model_dir = args.model or paths.model_dir
            writer = StatusWriter(STAGE_REPLAY_CHECK, updates, 'updates', cfg=cfg)
            writer.update(0, stage='loading')
            helpers = load_helpers(code_dir)
            source = load_source_tokenizer(model_dir)
            pool = helpers['TokenizerPool'](pool_dir)
            fallback = fallback_ids(source, helpers['missing_byte_symbols'])
            original = read_original(args.csv or repo / ORIGINAL_CSV, updates)

            def progress(done):
                if done % 10 == 0 or done == updates:
                    writer.update(done, stage='replaying', append=done % 100 == 0)

            replay = replay_examples(helpers, recipe, data_dir, pool, source, fallback, updates, progress=progress)
            report = compare(replay['rows'], original)
            if replay['failure'] is not None or len(replay['rows']) < updates:
                report['all_match'] = False
            report.update(mode='examples', updates=updates, trainer_failure=replay['failure'],
                          python_rng_digest=replay['python_rng_digest'],
                          inputs={'recipe_sha256': sha256_file(recipe),
                                  'corpus_manifest_sha256': sha256_file(Path(data_dir) / 'manifest.json')
                                  if (Path(data_dir) / 'manifest.json').is_file() else None,
                                  'pool_index_sha256': pool.index_hash, 'pool_plan_sha256': pool.plan_hash,
                                  'tokenizer_sha256': sha256_file(Path(model_dir) / 'tokenizer.json'),
                                  'code_dir': str(code_dir)})
            failure = replay['failure']
            if failure is not None:
                writer.update(len(replay['rows']), stage='failed', state='failed', all_match=False,
                              error=f'the trainer would raise at update {failure["step"]} (pool index '
                                    f'{failure["tokenizer_config_index"]}): {failure["error"]}',
                              match_rates={c: report['columns'][c]['rate'] for c in COLUMNS})
            else:
                writer.update(len(replay['rows']), stage='complete', state='complete',
                              all_match=report['all_match'],
                              match_rates={c: report['columns'][c]['rate'] for c in COLUMNS})
        report.update(created_at_utc=utc_now(), elapsed_seconds=round(time.monotonic() - started, 1))
        atomic_write_json(output, report)
    except Exception as exc:  # noqa: BLE001
        from hyak.common.config import ConfigError
        if not isinstance(exc, (OSError, ValueError, KeyError, ImportError, HelperMismatch, ConfigError)):
            raise
        traceback.print_exc(file=sys.stderr)
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return EXIT_USAGE

    summary = {name: stats['rate'] for name, stats in report['columns'].items()}
    print(json.dumps({'stage': 'replay_check', 'mode': report['mode'], 'updates': updates,
                      'compared_updates': report['compared_updates'], 'all_match': report['all_match'],
                      'match_rates': summary, 'trainer_failure': report.get('trainer_failure'),
                      'python_rng_digest_matches': report.get('python_rng_digest_matches'),
                      'report': str(output)}), flush=True)
    if report['mode'] == 'schedule' and not report['all_match']:
        print('ERROR: the RNG schedule does not match the original run (a wrong environment or checkout); '
              f'see {output}', file=sys.stderr, flush=True)
        return EXIT_CHECK
    if report.get('trainer_failure'):
        failure = report['trainer_failure']
        print(f'ERROR: the uniform trainer would raise at update {failure["step"]} ({failure["error"]}, pool index '
              f'{failure["tokenizer_config_index"]}); the corpus or pool is not faithful. See {output}',
              file=sys.stderr, flush=True)
        return EXIT_CHECK
    return EXIT_CHECK if (args.strict and not report['all_match']) else 0


if __name__ == '__main__':
    sys.exit(main())
