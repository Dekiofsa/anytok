"""Fidelity scorecard of a tokenizer pool against the original pool (DESIGN.md D4, 5.F; NOTES/gaps.md N2).

    python -m hyak.pool.scorecard [--levels 1 2 3] [--indices ... | pilots+binding] [--workers 8]
                                  [--pool-dir DIR] [--model-dir DIR] [--out FILE] [--skip-if-complete]

Three recorded fingerprints of every original tokenizer:

1. **File identity** -- SHA-256 of ``tokenizers/NNNN/tokenizer.json`` versus
   ``reports/rl-pool-sweep-20k/spec.json`` ``tokenizers[i].sha256`` (identical to
   ``tokenizer_hashes`` of the reward table).  Byte-identical tokenizer.
2. **Vocabulary strings** -- ``truncated_forms`` (vocabulary entries needing more than 10
   SmolLM2 pieces) versus the per-index value logged in
   ``reports/rl-tokenizer-selection-20k/data/training/uniform.csv``.  Computed exactly as the
   trainer does: pinned ``TokenizerPool.forms`` = ``build_surface_forms(pieces ordered by id,
   AutoTokenizer(model dir), 10, byte fallbacks, {'<|endoftext|>': 0})``
   (runtime/anytok_pool_20k/hf_hypernet/pool_sampling.py:54-62, train_joint_smollm_input.py:421,538).
3. **Segmentation** -- native token count of each of the 576 stored panel prefixes versus
   ``rows[j].native_prefix_lengths[str(i)]`` of
   ``reports/frozen-policy-search-20k/reward_table.json.gz``; the prefixes are rebuilt with
   ``hyak.eval.rebuild_panel.rebuild('pool', ...)`` (decode of the stored canonical ids, every
   context hash and the panel hash verified) and counted as
   scripts/pool_regime.py:11 did: ``len(tokenizer.encode(prefix, add_special_tokens=False).ids)``.

Level 3 also counts, per tokenizer, the **protected rows over the cap**: passages that were valid in
the original panel (``original_joint_encoding_matches``) on which the tokenizer now needs more than 95
native tokens (scripts/pool_regime.py:12).  Any such row makes ``hyak.eval.validate_prefixes --panel
pool`` stop (exit 3), which blocks the complete-pool table.  The seven tokenizers that sit exactly at
the cap in the original (``hyak.pool.BINDING_INDICES`` = 64 100 632 687 728 821 914) are reported
separately: run them with the pilots (``--indices pilots+binding``) before the full pool.

The scorecard reports; it never fails the pipeline because a tokenizer differs.  A tokenizer whose
level 2 or 3 cannot be computed is reported with ``level2_error`` / ``level3_error`` (listed at the top of
the text summary); the others are scored.  It fails (exit 2) only when an input is missing or corrupt
(anchor files, the canonical tokenizer, a panel that does not rebuild); 1 only for an unexpected crash
(traceback).  It also works on a partial pool and on an original pool the user obtains (expect 1,000 /
1,000 at level 1).  ``--skip-if-complete`` exits 0 without work when ``--out`` already holds a report of
all planned indices at all three levels for this plan that is newer than the pool index.  Run with the
``train`` environment (stock tokenizers, transformers).  Output: JSON (``--out``, default
``$ANYTOK_ROOT/eval/reports/pool-scorecard.json``) and a text summary next to it.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

from hyak.common.config import ConfigError, load_config
from hyak.common.io import atomic_write_json, atomic_write_text, sha256_file, utc_now
from hyak.common.paths import Paths
from hyak.pool import (
    BINDING_INDICES, PLAN_REL, REWARD_TABLE_REL, REWARD_TABLE_SHA256, SPEC_REL, SPEC_SHA256, UNIFORM_CSV_REL,
    UNIFORM_CSV_SHA256, PoolError, family, index_name, load_plan)

MAX_PIECES = 10          # train_joint_smollm_input.py:389 "max_pieces": 10; pool_sampling.py:60
EOS = '<|endoftext|>'    # pool_sampling.py:61 {'<|endoftext|>': 0}
NATIVE_CAP = 95          # scripts/pool_regime.py:12 (native prefix tokens of the pool regime)


# --- anchors ---------------------------------------------------------------------------------------
def _pinned(path: Path, expected: str) -> Path:
    actual = sha256_file(path)
    if actual != expected:
        raise PoolError(f'{path} has SHA-256 {actual}, expected the committed {expected}; restore it with git')
    return path


def load_spec_hashes(repo: Path, plan: dict, *, check_pinned: bool = True) -> list:
    """Original tokenizer SHA-256 per index (spec.json), cross-checked with the reward table and the plan."""
    spec_path = repo / SPEC_REL
    if check_pinned:
        _pinned(spec_path, SPEC_SHA256)
    spec = json.loads(spec_path.read_bytes())
    rows = spec['tokenizers']
    if [row['index'] for row in rows] != list(range(len(rows))):
        raise PoolError(f'{SPEC_REL}: tokenizers are not ordered by index')
    if check_pinned:
        for row in rows:
            if row['config'] != plan['configs'][row['index']]:
                raise PoolError(f'{SPEC_REL}: config of index {row["index"]} differs from the plan')
    return [row['sha256'] for row in rows]


def load_truncated_forms(repo: Path, *, check_pinned: bool = True) -> dict:
    """``{pool index: truncated_forms}`` from the original uniform run's per-update CSV.

    Every index must carry one consistent value (it is a function of the tokenizer only).
    """
    path = repo / UNIFORM_CSV_REL
    if check_pinned:
        _pinned(path, UNIFORM_CSV_SHA256)
    values = defaultdict(set)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            values[int(row['tokenizer_config_index'])].add(int(row['truncated_forms']))
    inconsistent = sorted(i for i, v in values.items() if len(v) != 1)
    if inconsistent:
        raise PoolError(f'{UNIFORM_CSV_REL}: indices with several truncated_forms values: {inconsistent[:10]}')
    return {index: next(iter(v)) for index, v in values.items()}


def load_reward_rows(repo: Path, *, check_pinned: bool = True) -> list:
    path = repo / REWARD_TABLE_REL
    if check_pinned:
        _pinned(path, REWARD_TABLE_SHA256)
    return json.loads(gzip.decompress(path.read_bytes()))['rows']


# --- per-tokenizer measurements ------------------------------------------------------------------------
_STATE = {}


def _init_worker(settings: dict) -> None:
    """Process initializer: pinned code on sys.path, canonical source tokenizer, rebuilt prefixes."""
    _STATE.clear()
    _STATE.update(settings)
    if settings.get('level2') or settings.get('level3_prefixes') is not None:
        from hyak.eval import setup_imports
        setup_imports(settings['code_dir'], settings['repo'])
    if settings.get('level2'):
        from transformers import AutoTokenizer
        from hf_hypernet.tokenizer_inputs import missing_byte_symbols
        source = AutoTokenizer.from_pretrained(settings['model_dir'], local_files_only=True)
        size = len(source)
        # enable_byte_fallbacks (tokenizer_inputs.py:21-39) without a hypernetwork: same mapping.
        _STATE['source'] = source
        _STATE['fallback'] = {symbol: size + 1 + i for i, symbol in enumerate(missing_byte_symbols(source))}


def measure(index: int) -> dict:
    """All requested levels for one pool index (runs in a worker process)."""
    path = Path(_STATE['tokenizers_dir']) / index_name(index) / 'tokenizer.json'
    result = {'index': index}
    try:
        raw = path.read_bytes()
    except OSError:
        result['missing'] = True
        return result
    import hashlib
    result['sha256'] = hashlib.sha256(raw).hexdigest()
    if not (_STATE.get('level2') or _STATE.get('level3_prefixes') is not None):
        return result
    from tokenizers import Tokenizer
    started = time.time()
    prefixes = _STATE.get('level3_prefixes')
    try:
        tokenizer = Tokenizer.from_str(raw.decode())
    except Exception as exc:  # noqa: BLE001 -- one bad tokenizer must not end the scorecard
        message = f'cannot load tokenizer.json: {type(exc).__name__}: {exc}'
        if _STATE.get('level2'):
            result['level2_error'] = message
        if prefixes is not None:
            result['level3_error'] = message
        return result
    if _STATE.get('level2'):
        try:
            from hf_hypernet.tokenizer_inputs import build_surface_forms
            tokens = tokenizer.get_vocab()
            ordered = [None] * len(tokens)
            for token, token_id in tokens.items():     # pool_sampling.py:47-49
                if 0 <= token_id < len(ordered):
                    ordered[token_id] = token
            if None in ordered:
                result['level2_error'] = 'non-contiguous vocabulary ids'
            else:
                _, truncated = build_surface_forms(ordered, _STATE['source'], MAX_PIECES, _STATE['fallback'], {EOS: 0})
                result['truncated_forms'] = int(truncated)
        except Exception as exc:  # noqa: BLE001 -- e.g. a piece build_surface_forms cannot represent
            result['level2_error'] = f'{type(exc).__name__}: {exc}'
    if prefixes is not None:
        try:
            result['native_prefix_lengths'] = [len(tokenizer.encode(p, add_special_tokens=False).ids) for p in prefixes]
        except Exception as exc:  # noqa: BLE001
            result['level3_error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = round(time.time() - started, 3)
    return result


# --- scoring -------------------------------------------------------------------------------------------------
def score(plan: dict, measurements: list, *, spec_hashes=None, truncated=None, reward_rows=None) -> dict:
    """Compare measurements with the anchors.  Pure function (unit-tested on fixtures)."""
    per_index = []
    families = defaultdict(lambda: {'tokenizers': 0, 'level1_match': 0, 'level2_match': 0, 'level2_checked': 0,
                                    'level3_all_rows_match': 0, 'level3_rows_match': 0, 'level3_rows': 0,
                                    'level3_checked': 0, 'level3_protected_over_cap': 0,
                                    'level3_tokenizers_over_cap': 0, 'errors': 0})
    totals = {'present': 0, 'missing': 0}
    errors = []
    for item in measurements:
        index = item['index']
        config = plan['configs'][index]
        row = {'index': index, 'family': family(config), 'vocab_size': config['vocab_size']}
        if item.get('missing'):
            row['missing'] = True
            totals['missing'] += 1
            per_index.append(row)
            continue
        totals['present'] += 1
        cell = families[row['family']]
        cell['tokenizers'] += 1
        if spec_hashes is not None:
            row['level1'] = {'sha256': item['sha256'], 'original_sha256': spec_hashes[index],
                             'match': item['sha256'] == spec_hashes[index]}
            cell['level1_match'] += row['level1']['match']
        if truncated is not None and ('truncated_forms' in item or 'level2_error' in item):
            expected = truncated.get(index)
            actual = item.get('truncated_forms')
            row['level2'] = {'truncated_forms': actual, 'original': expected,
                             'match': actual is not None and actual == expected}
            if 'level2_error' in item:
                row['level2']['error'] = item['level2_error']
            if expected is not None:
                cell['level2_checked'] += 1
                cell['level2_match'] += row['level2']['match']
        if reward_rows is not None and 'native_prefix_lengths' in item:
            expected = [r['native_prefix_lengths'][str(index)] for r in reward_rows]
            actual = item['native_prefix_lengths']
            matches = sum(a == e for a, e in zip(actual, expected))
            diffs = [a - e for a, e in zip(actual, expected) if a != e]
            protected = [reward_rows[j]['id'] for j, a in enumerate(actual)
                         if reward_rows[j].get('original_joint_encoding_matches') and a > NATIVE_CAP]
            row['level3'] = {'rows': len(expected), 'rows_match': matches, 'all_match': matches == len(expected),
                             'max_abs_difference': max((abs(d) for d in diffs), default=0),
                             'mean_difference': round(sum(diffs) / len(expected), 4),
                             'over_native_cap_95': sum(a > NATIVE_CAP for a in actual),
                             'protected_rows_over_cap': len(protected), 'protected_rows_over_cap_ids': protected[:10],
                             'mismatched_rows': [reward_rows[j]['id'] for j, (a, e) in enumerate(zip(actual, expected))
                                                 if a != e][:10]}
            cell['level3_checked'] += 1
            cell['level3_rows'] += len(expected)
            cell['level3_rows_match'] += matches
            cell['level3_all_rows_match'] += row['level3']['all_match']
            cell['level3_protected_over_cap'] += len(protected)
            cell['level3_tokenizers_over_cap'] += bool(protected)
        if 'level3_error' in item:
            row['level3'] = {'error': item['level3_error']}
        failures = {k: item[k] for k in ('level2_error', 'level3_error') if k in item}
        if failures:
            cell['errors'] += 1
            errors.append({'index': index, **failures})
        per_index.append(row)
    summary = {name: dict(cell) for name, cell in sorted(families.items())}
    overall = defaultdict(int)
    for cell in summary.values():
        for key, value in cell.items():
            overall[key] += value
    by_index = {row['index']: row for row in per_index}
    binding = [by_index[i] for i in BINDING_INDICES if i in by_index]
    report = {'totals': {**totals, **overall}, 'families': summary, 'tokenizers': per_index, 'errors': errors}
    if binding and len(plan['configs']) > max(BINDING_INDICES):
        present = [row for row in binding if not row.get('missing')]
        report['binding'] = {
            'indices': list(BINDING_INDICES), 'measured': len(binding), 'present': len(present),
            'level1_identical': sum(bool((row.get('level1') or {}).get('match')) for row in present),
            'level1_checked': sum('level1' in row for row in present),
            'level3_checked': sum('rows' in (row.get('level3') or {}) for row in present),
            'protected_rows_over_cap': sum((row.get('level3') or {}).get('protected_rows_over_cap', 0) for row in present),
            'not_identical': [row['index'] for row in present if 'level1' in row and not row['level1']['match']],
        }
    return report


def summary_text(report: dict) -> str:
    totals = report['totals']
    lines = [f"Pool scorecard  {report['created_at']}  pool {report['pool_dir']}",
             f"tokenizers present {totals['present']} / {report['planned']}"
             + (f", missing {totals['missing']}" if totals['missing'] else '')]
    errors = report.get('errors') or []
    if errors:
        lines.append(f"ERRORS: level 2/3 could not be computed for {len(errors)} tokenizer(s): "
                     + '; '.join(f"{e['index']}: {e.get('level2_error') or e.get('level3_error')}"[:200] for e in errors[:10]))
    lines.append('')
    header = (f"{'family':<20}{'n':>5}{'L1 file':>10}{'L2 forms':>11}{'L3 all 576':>12}{'L3 rows':>12}"
              f"{'over cap':>10}")
    lines.append(header)
    lines.append('-' * len(header))
    for name, cell in list(report['families'].items()) + [('ALL', totals)]:
        n = cell.get('tokenizers', 0)
        l3 = (f"{100.0 * cell['level3_rows_match'] / cell['level3_rows']:.2f}%" if cell.get('level3_rows') else '-')
        lines.append(f"{name:<20}{n:>5}{cell.get('level1_match', 0):>10}"
                     f"{(str(cell['level2_match']) + '/' + str(cell['level2_checked'])) if cell.get('level2_checked') else '-':>11}"
                     f"{(str(cell['level3_all_rows_match']) + '/' + str(cell['level3_checked'])) if cell.get('level3_checked') else '-':>12}"
                     f"{l3:>12}{cell.get('level3_protected_over_cap', 0) if cell.get('level3_checked') else '-':>10}")
    binding = report.get('binding')
    if binding:
        lines += ['', f"binding tokenizers {' '.join(str(i) for i in binding['indices'])} (exactly 95 native tokens on a "
                      f"protected passage in the original): {binding['present']} of 7 present; identical to the original "
                      f"(level 1): {binding['level1_identical']} of {binding['level1_checked']}"
                  + (f"; protected rows over the cap: {binding['protected_rows_over_cap']}" if binding['level3_checked'] else ''),
                  'Any binding tokenizer that is not identical may make the complete-pool evaluation stop '
                  '(hyak.eval.validate_prefixes --panel pool, exit 3); any protected row over the cap will.'
                  + (f" Not identical: {binding['not_identical']}." if binding['not_identical'] else '')]
    lines += ['', 'L1 = byte-identical tokenizer.json (spec.json); L2 = truncated_forms equal to uniform.csv;',
              'L3 = native token counts equal to the reward table on all 576 prefixes / share of matching rows;',
              f'over cap = protected passages (valid in the original panel) needing more than {NATIVE_CAP} native tokens.',
              'WordPiece and the BPE families (noise included) were bit-reproducible in repeated local runs: with a '
              'faithful corpus and wheel their L1 should match; Unigram may differ in ids and scores.']
    return '\n'.join(lines) + '\n'


def existing_report_is_complete(out: Path, plan: dict, pool_dir: Path) -> bool:
    """``--skip-if-complete``: ``out`` covers every planned index at all three levels for this plan and is newer
    than the pool index ``full-results.json`` (when one exists)."""
    try:
        report = json.loads(out.read_text())
        count = len(plan['configs'])
        complete = (report.get('levels') == [1, 2, 3] and report.get('indices') == count
                    and report.get('planned') == count and (report.get('totals') or {}).get('present') == count
                    and report.get('plan_sha256') == plan['plan_sha256'] and not report.get('errors'))
        index = pool_dir / 'full-results.json'
        if complete and index.exists() and out.stat().st_mtime < index.stat().st_mtime:
            return False
        return bool(complete)
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def run(args) -> int:
    levels = sorted(set(args.levels))
    if not set(levels) <= {1, 2, 3}:
        raise PoolError('--levels takes 1, 2 and/or 3')
    cfg = load_config(args.config) if (args.config or not (args.pool_dir and args.model_dir and args.out)) else None
    paths = Paths.from_config(cfg) if cfg is not None else None
    from hyak.common.config import repo_root
    repo = paths.repo if paths else repo_root()
    plan_path = Path(args.plan) if args.plan else repo / PLAN_REL
    plan, plan_info = load_plan(plan_path, allow_test_plan=args.allow_test_plan)
    pinned = plan_info['production']
    pool_dir = Path(args.pool_dir) if args.pool_dir else paths.pool_dir
    model_dir = Path(args.model_dir) if args.model_dir else paths.model_dir
    out = Path(args.out) if args.out else paths.eval_reports / 'pool-scorecard.json'
    count = len(plan['configs'])
    if args.skip_if_complete and existing_report_is_complete(out, plan, pool_dir):
        print(f'{out} already covers all {count} tokenizers at levels 1-3 and is newer than the pool index; skipped')
        return 0
    from hyak.pool import parse_indices
    indices = parse_indices(args.indices, count, plan['pilot_indices']) if args.indices else list(range(count))
    spec_hashes = load_spec_hashes(repo, plan, check_pinned=pinned) if 1 in levels else None
    truncated = load_truncated_forms(repo, check_pinned=pinned) if 2 in levels else None
    reward_rows, prefixes, panel_report = None, None, None
    code_dir = Path(args.code_dir) if args.code_dir else repo / 'runtime/anytok_pool_20k'
    if 3 in levels:
        from hyak.eval import setup_imports
        from hyak.eval.rebuild_panel import rebuild
        setup_imports(code_dir, repo)
        rebuilt, panel_report = rebuild('pool', model_dir / 'tokenizer.json', repo=repo)
        if not panel_report['ok']:
            raise PoolError(f'the stored 576-prefix panel does not rebuild: {panel_report["failed_rows"][:3]} '
                            f'{panel_report["panel_problems"]}')
        reward_rows = load_reward_rows(repo, check_pinned=pinned)
        if [r['id'] for r in reward_rows] != [r['id'] for r in rebuilt]:
            raise PoolError('rebuilt panel rows are not in reward-table order')
        prefixes = [r['prefix'] for r in rebuilt]
    if 2 in levels and not (model_dir / 'tokenizer.json').is_file():
        raise PoolError(f'{model_dir} has no tokenizer.json (the SmolLM2 model directory, input/model)')
    settings = {'tokenizers_dir': str(pool_dir / 'tokenizers'), 'level2': 2 in levels,
                'level3_prefixes': prefixes, 'model_dir': str(model_dir), 'code_dir': str(code_dir), 'repo': str(repo)}
    started = time.time()
    if args.workers > 1 and len(indices) > 1:
        context = multiprocessing.get_context('spawn')
        with context.Pool(args.workers, initializer=_init_worker, initargs=(settings,)) as workers:
            measurements = list(workers.imap(measure, indices, chunksize=4))
    else:
        _init_worker(settings)
        measurements = [measure(i) for i in indices]
    report = score(plan, measurements, spec_hashes=spec_hashes, truncated=truncated, reward_rows=reward_rows)
    report.update(created_at=utc_now(), pool_dir=str(pool_dir), planned=count, indices=len(indices), levels=levels,
                  plan_sha256=plan['plan_sha256'], model_dir=str(model_dir), seconds=round(time.time() - started, 1),
                  panel=None if panel_report is None else {k: panel_report[k] for k in
                                                           ('panel_sha256', 'panel_sha256_matches', 'rows', 'verified_rows')})
    atomic_write_json(out, report)
    text = summary_text(report)
    atomic_write_text(out.with_suffix('.txt'), text)
    print(text, end='')
    print(f'wrote {out} and {out.with_suffix(".txt")}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m hyak.pool.scorecard', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--indices', nargs='+', help='subset of pool indices (ints, a-b ranges, "pilots", "binding", '
                        '"pilots+binding"); default all')
    parser.add_argument('--skip-if-complete', action='store_true',
                        help='exit 0 without work when --out already holds a complete, current report')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                        help='processes for levels 2 and 3 (default: cores - 1, at most 8)')
    parser.add_argument('--pool-dir', help='default $ANYTOK_ROOT/input/pool')
    parser.add_argument('--model-dir', help='SmolLM2 directory with tokenizer.json (+ config.json); default input/model')
    parser.add_argument('--code-dir', help='pinned code tree (default <repo>/runtime/anytok_pool_20k)')
    parser.add_argument('--out', help='report JSON (default $ANYTOK_ROOT/eval/reports/pool-scorecard.json)')
    parser.add_argument('--config', help='config file (default $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--plan', help=argparse.SUPPRESS)
    parser.add_argument('--allow-test-plan', action='store_true', help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (RuntimeError, OSError) as exc:   # PoolError, ConfigError, EvalError, a missing input file
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 2
    except Exception:  # noqa: BLE001 -- unexpected: full traceback, exit 1
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
