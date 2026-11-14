"""Native prefix lengths of the regenerated pool and the repository's joint prefix trimming (CPU).

    python -m hyak.eval.validate_prefixes --panel pool|natural [--indices ...] [--workers N]
                                          [--recheck] [--force] [--claim-timeout-minutes 240]
                                          [--accept-valid-context-trims] [--config FILE] [--code-dir DIR]

Input: ``eval/panels/<panel>/rebuilt.json`` (hyak.eval.rebuild_panel), the pool
(``input/pool``) and the canonical tokenizer (``input/model/tokenizer.json``).

Loop (scripts/modal_frozen_policy_cache.py:180-193 with ``constraints`` :82-104
and ``reduce_panel`` :108-137; for the natural panel the identical loop of
scripts/modal_random_pool_50.py:69-88): for up to five rounds, compute
``pool_regime.required_start(row, canonical, tokenizer, forbidden | {0})`` for
every (tokenizer, row) (imported), take the per-row maximum start over all
tokenizers, ``pool_regime.shorten`` the rows (imported), and repeat until every
start is 0; then record ``native_prefix_lengths`` and ``maximum_positions``.
Targets, bytes and source ids never change; only prefixes can lose characters
on the left.  With a pool identical to the original this is a no-op that
reproduces the stored lengths and the stored panel hash.

Hard stops (nothing is dropped or replaced): a row with no shared lossless prefix
of at least 16 canonical tokens for some tokenizer (``required_start`` raises);
no convergence within five rounds; for the pool panel any trim of a context that
was already valid in the original panel (``original_joint_encoding_matches``,
scripts/modal_frozen_policy_cache.py:121-122), a position-cap violation, or a
change of coverage (576 rows, 288 per split).  The report
``eval/panels/<panel>/validation-report.json`` is written in every case and lists
each row whose prefix had to be trimmed relative to the stored panel
(``comparison.trimmed``; the tokenizers that forced it are in ``rounds[].rows[].tokenizers``)
and how many rows stayed identical.

The trim of an originally valid pool-panel context is the one stop with an option,
and the option is the user's decision, never a default: the report then holds
``valid_context_trims`` (id, split, domain, characters before/after, canonical prefix
tokens before/after, forcing tokenizers) and ``decision``; after the user approves
those trims, ``--accept-valid-context-trims`` validates the panel with them and records
them as ``plan['hyak']['accepted_valid_context_trims']`` and in the report.  The other
option is to validate with the original pool.  Unsatisfiable rows, no convergence,
the position cap and coverage have no option.  Exit codes: 0 done (or up to date, or
another job holds the claim), 2 usage/input error, 3 protocol stop, 1 crash.

Outputs: ``panel.json`` (rows with prefix), ``plan.json`` (original frozen plan
with the new ``panel_sha256``, ``indices``, ``models: []`` and a ``hyak`` block with
the pool identity), ``preparation.json`` ({plan, panel_sha256, metadata, files}).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import signal
import sys
import time

from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now
from hyak.common.paths import STAGE_EVAL_PREPARE
from hyak.common.config import ConfigError
from hyak.eval import (EXIT_OK, EXIT_PROTOCOL, EXIT_USAGE, MAX_ROUNDS, PANEL_NATURAL, PANEL_POOL, PANEL_SPECS, PANELS,
                       POSITION_CAP, Claims, EvalError, EvalLayout, check_tokenizer_file, crashed, digest,
                       load_canonical, load_pool_spec, log, parse_indices, repo_root, setup_imports, tokenizer_path,
                       validate_panel_name)


class ValidationStop(EvalError):
    """Validation cannot finish without changing the protocol; the report says which rows."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def tokenizer_constraints(rows: list, canonical, tokenizer, forbidden: set) -> tuple:
    """Required start and native length of every row for one tokenizer.

    Body of ``constraints`` (scripts/modal_frozen_policy_cache.py:93-97): ``forbidden``
    must already include id 0 (``native[4] | {0}``, :92).  A row without a valid
    start is collected instead of aborting, so the report can list every such row.
    Returns ``(starts, counts, failures)``; failed rows get ``None``.
    """
    from pool_regime import required_start
    starts, counts, failures = [], [], []
    for row in rows:
        try:
            start, count = required_start(row, canonical, tokenizer, forbidden)
        except ValueError as exc:
            starts.append(None)
            counts.append(None)
            failures.append({'id': row['id'], 'error': str(exc)})
            continue
        starts.append(start)
        counts.append(count)
    return starts, counts, failures


_WORKER = {}


def _init_worker(code_dir: str, canonical_path: str, rows: list):
    setup_imports(code_dir)
    _WORKER['canonical'], _ = load_canonical(canonical_path)
    _WORKER['rows'] = rows
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _validate_one(index: int, path: str, expected_sha: str) -> dict:
    from uniform_trajectory import load_candidate
    if sha256_file(path) != expected_sha:
        raise EvalError(f'Native tokenizer changed: {path}')  # modal_frozen_policy_cache.py:90-91
    native = load_candidate(path, 'pool')
    tokenizer, forbidden = native[0], native[4] | {0}
    starts, counts, failures = tokenizer_constraints(_WORKER['rows'], _WORKER['canonical'], tokenizer, forbidden)
    return {'index': index, 'starts': starts, 'counts': counts, 'failures': failures}


def run_round(rows, indices, spec, pool_dir, canonical_path, *, workers, code_dir, progress=None) -> dict:
    """One validation round over all tokenizers.  Returns per-row max starts, lengths and failures."""
    starts = [0] * len(rows)
    demanding = [dict() for _ in rows]
    lengths, failures = {}, []
    jobs = {index: str(tokenizer_path(pool_dir, index)) for index in indices}
    context = multiprocessing.get_context('spawn')
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker,
                             initargs=(str(code_dir), str(canonical_path), rows)) as pool:
        futures = {pool.submit(_validate_one, index, path, spec['tokenizers'][index]['sha256']): index
                   for index, path in jobs.items()}
        for future in as_completed(futures):
            result = future.result()
            index = result['index']
            for j, start in enumerate(result['starts']):
                if start is None:
                    continue
                starts[j] = max(starts[j], start)
                if start:
                    demanding[j][str(index)] = start
            lengths[str(index)] = result['counts']
            failures.extend({'index': index, **f} for f in result['failures'])
            done += 1
            if progress is not None:
                progress(done)
    return {'starts': starts, 'lengths': lengths, 'failures': failures, 'demanding': demanding}


def _compare_with_stored(stored: list, final: list, indices: list) -> dict:
    """Rows unchanged versus trimmed, and how many native lengths equal the stored (original-pool) ones."""
    trimmed, identical = [], 0
    for before, after in zip(stored, final):
        if before['id'] != after['id'] or before['canonical_continuation_ids'] != after['canonical_continuation_ids']:
            raise EvalError('internal error: validation reordered rows or changed targets')
        if before['prefix'] == after['prefix']:
            identical += 1
            continue
        trimmed.append({'id': after['id'], 'domain': after['domain'], 'split': after['split'],
                        'stored_prefix_chars': len(before['prefix']), 'prefix_chars': len(after['prefix']),
                        'trimmed_chars': len(before['prefix']) - len(after['prefix']),
                        'stored_canonical_prefix_tokens': len(before['canonical_prefix_ids']),
                        'canonical_prefix_tokens': len(after['canonical_prefix_ids']),
                        'pool_trim_chars': after.get('pool_trim_chars', 0),
                        'original_joint_encoding_matches': after.get('original_joint_encoding_matches')})
    equal_lengths = {}
    for index in indices:
        key = str(index)
        pairs = [(b['native_prefix_lengths'].get(key), a['native_prefix_lengths'][key])
                 for b, a in zip(stored, final) if key in b.get('native_prefix_lengths', {})]
        if pairs:
            equal_lengths[key] = sum(x == y for x, y in pairs)
    full = sum(v == len(stored) for v in equal_lengths.values())
    return {'rows': len(final), 'rows_identical_to_stored': identical, 'rows_trimmed': len(trimmed),
            'trimmed': trimmed,
            'native_lengths_equal_stored': {'tokenizers_compared': len(equal_lengths),
                                            'tokenizers_all_rows_equal': full,
                                            'rows_equal_per_tokenizer': equal_lengths}}


def finalize_rows(panel: str, rows: list, lengths: dict, indices: list) -> list:
    """Final-round bookkeeping and checks.

    pool: scripts/modal_frozen_policy_cache.py:116-124; natural: scripts/modal_random_pool_50.py:86-88
    plus the position-cap check of :119-120 (cannot fire with the 95-token cap, kept as a guard).

    Every problem has a ``kind``: ``position_cap`` and ``coverage`` always stop;
    ``valid_context_trim`` (pool panel: a context that was valid in the original panel
    had to be trimmed) stops unless the user accepted it (``--accept-valid-context-trims``).
    """
    spec = PANEL_SPECS[panel]
    problems = []
    for j, row in enumerate(rows):
        row['native_prefix_lengths'] = {str(i): lengths[str(i)][j] for i in indices}
        row['maximum_positions'] = 33 + max(len(row['canonical_prefix_ids']), max(row['native_prefix_lengths'].values()))
        if row['maximum_positions'] > POSITION_CAP:
            problems.append({'id': row['id'], 'kind': 'position_cap', 'error': 'Position cap exceeded'})
        if panel == PANEL_POOL and row['original_joint_encoding_matches'] and row.get('pool_trim_chars', 0):
            problems.append({'id': row['id'], 'kind': VALID_CONTEXT_TRIM,
                             'error': 'An already-valid original prefix changed'})
    if len(rows) != spec.rows or any(sum(r['split'] == s for r in rows) != n for s, n in spec.splits):
        problems.append({'id': None, 'kind': 'coverage', 'error': 'Full original text coverage required'})
    return problems


VALID_CONTEXT_TRIM = 'valid_context_trim'
ACCEPT_FLAG = '--accept-valid-context-trims'


def forcing_tokenizers(report: dict, row_id) -> dict:
    """``{tokenizer index: largest start (characters) it demanded}`` for one row over all rounds."""
    out = {}
    for entry in report['rounds']:
        for row in entry['rows']:
            if row['id'] == row_id:
                for index, start in row['tokenizers'].items():
                    out[index] = max(out.get(index, 0), start)
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def valid_context_trims(report: dict, stored: list, rows: list, problems: list) -> list:
    """One record per originally valid context that had to be trimmed (what the user is asked to accept)."""
    ids = {p['id'] for p in problems if p.get('kind') == VALID_CONTEXT_TRIM}
    out = []
    for before, after in zip(stored, rows):
        if after['id'] not in ids:
            continue
        out.append({'id': after['id'], 'split': after['split'], 'domain': after['domain'],
                    'stored_prefix_chars': len(before['prefix']), 'prefix_chars': len(after['prefix']),
                    'trimmed_chars': len(before['prefix']) - len(after['prefix']),
                    'stored_canonical_prefix_tokens': len(before['canonical_prefix_ids']),
                    'canonical_prefix_tokens': len(after['canonical_prefix_ids']),
                    'tokenizers': forcing_tokenizers(report, after['id'])})
    return out


def decision_text(trims: list, report_path) -> str:
    """The options of the user after a valid-context stop (also in report['decision'])."""
    chars = sum(t['trimmed_chars'] for t in trims)
    forcing = sorted({int(i) for t in trims for i in t['tokenizers']})
    return (f'{len(trims)} pool-panel passages whose context was valid in the original panel would lose {chars} '
            f'characters on the left (forced by tokenizers {forcing}); the frozen protocol keeps such contexts '
            'unchanged, so no panel was written. Targets and scored bytes would be unchanged. This is the user\'s '
            f'decision. Options: (1) after the user approves the listed trims, rerun with {ACCEPT_FLAG} (job '
            'variable HYAK_EVAL_PREPARE_ARGS); the accepted rows are recorded in the panel plan, the report and '
            'the packed summaries; (2) bring the original pool and validate with it. The natural panel is '
            f'independent and can be validated and scored meanwhile. Rows, characters and tokenizers: {report_path}')


def derived_plan(panel: str, original_plan: dict, rows: list, indices: list, spec: dict, *,
                 stored_hash: str, canonical_sha: str, code_hashes: dict, rounds: int, trimmed: int,
                 scripts_hashes: dict | None = None, trimmed_row_ids: list | None = None,
                 trimmed_chars: int | None = None, accepted_valid_context_trims: list | None = None) -> dict:
    """The original frozen plan with this panel's hash, the scored indices and the regenerated pool's identity.

    ``plan['hyak']`` also records how the panel differs from the stored one (``trimmed_rows``,
    ``trimmed_row_ids`` in panel order, ``trimmed_chars``), the SHA-256 of the hypernetwork code
    (``code_sha256``) and of the imported scoring scripts (``scripts_sha256``; default: what this
    process imported), and, only when the user accepted them, ``accepted_valid_context_trims``.
    """
    from hyak.eval import current_code_hashes
    plan = {k: v for k, v in original_plan.items() if k != 'models'}
    plan.update(panel_sha256=digest(rows), indices=list(indices), models=[])
    plan['hyak'] = {
        'version': 1, 'panel': panel, 'stored_panel_sha256': stored_hash,
        'source_plan_sha256': digest(original_plan), 'source_plan': PANEL_SPECS[panel].plan_rel,
        'pool_index_sha256': spec['index_sha256'], 'pool_plan_sha256': spec['plan_sha256'],
        'canonical_tokenizer_sha256': canonical_sha,
        'tokenizer_sha256': {str(i): spec['tokenizers'][i]['sha256'] for i in indices},
        'validation_rounds': rounds, 'trimmed_rows': trimmed, 'code_sha256': code_hashes,
        'scripts_sha256': scripts_hashes if scripts_hashes is not None else current_code_hashes()['scripts'],
        'trimmed_row_ids': list(trimmed_row_ids or []), 'trimmed_chars': int(trimmed_chars or 0),
        'complete_pool': list(indices) == list(range(1000)),
    }
    if accepted_valid_context_trims:
        plan['hyak']['accepted_valid_context_trims'] = list(accepted_valid_context_trims)
    return plan


def default_indices(panel: str, repo: Path | None = None) -> list:
    """pool: all 1,000; natural: the 38 indices of plans/natural-model-comparison-v1.json."""
    if panel == PANEL_POOL:
        return list(range(1000))
    repo = Path(repo) if repo is not None else repo_root()
    return list(read_json(repo / PANEL_SPECS[PANEL_NATURAL].plan_rel)['indices'])


def validate(layout: EvalLayout, panel: str, spec: dict, pool_dir: Path, canonical_path: Path, indices: list, *,
             workers: int, code_info: dict, status=None, max_rounds: int = MAX_ROUNDS, repo=None,
             accept_valid_context_trims: bool = False) -> dict:
    """Run the loop in memory.  Returns ``{'rows', 'plan', 'preparation', 'report'}``.

    Nothing but the report is written here (see :func:`write_validated`); on a hard
    stop the report is written and ValidationStop is raised.
    ``accept_valid_context_trims`` (pool panel; the user's explicit decision, never a
    default) lets trims of originally valid contexts through and records them in the
    plan and the report; every other stop stays a hard stop.
    """
    from pool_regime import shorten
    validate_panel_name(panel)
    repo = Path(repo) if repo is not None else repo_root()
    stored = read_json(layout.rebuilt(panel))
    stored_hash = digest(stored)
    if stored_hash != PANEL_SPECS[panel].panel_sha256:
        raise EvalError(f'{layout.rebuilt(panel)} is not the stored {panel} panel; rerun hyak.eval.rebuild_panel')
    canonical, canonical_sha = load_canonical(canonical_path)
    for index in indices:
        check_tokenizer_file(pool_dir, spec, index)
    original_plan = read_json(repo / PANEL_SPECS[panel].plan_rel)
    report = {'panel': panel, 'started_at': utc_now(), 'indices': len(indices), 'stored_panel_sha256': stored_hash,
              'pool_index_sha256': spec['index_sha256'], 'canonical_tokenizer_sha256': canonical_sha,
              'rounds': [], 'ok': False}
    rows = [dict(r) for r in stored]
    lengths = None
    begin = time.monotonic()
    for round_index in range(max_rounds):
        def progress(done, r=round_index):
            if status is not None and (done % 25 == 0 or done == len(indices)):
                status.update(done, stage=f'validating {panel} prefixes (round {r + 1})', total=len(indices),
                              elapsed_seconds=time.monotonic() - begin, task=f'validate-{panel}', append=False)
        log(f'{panel}: validation round {round_index + 1} over {len(indices)} tokenizers x {len(rows)} rows')
        result = run_round(rows, indices, spec, pool_dir, canonical_path, workers=workers,
                           code_dir=code_info['code_dir'], progress=progress)
        starts, lengths = result['starts'], result['lengths']
        entry = {'round': round_index + 1, 'rows_needing_trim': sum(s > 0 for s in starts),
                 'max_start_chars': max(starts) if starts else 0,
                 'rows': [{'id': rows[j]['id'], 'start_chars': s, 'tokenizers': result['demanding'][j]}
                          for j, s in enumerate(starts) if s],
                 'unsatisfiable': result['failures']}
        report['rounds'].append(entry)
        log(f'{panel}: round {round_index + 1}: {entry["rows_needing_trim"]} rows need a trim '
            f'(max {entry["max_start_chars"]} chars); {len(result["failures"])} unsatisfiable (tokenizer,row) pairs')
        if result['failures']:
            report['stopped'] = ('No shared lossless prefix with at least 16 canonical tokens for '
                                 f'{len({f["id"] for f in result["failures"]})} rows; nothing was dropped')
            atomic_write_json(layout.validation_report(panel), report)
            raise ValidationStop(report['stopped'] + f'; see {layout.validation_report(panel)}', report)
        final = not any(starts)
        if panel == PANEL_POOL:
            rows = shorten(rows, starts, canonical)  # reduce_panel shortens in every round (:113)
        elif not final:
            rows = shorten(rows, starts, canonical)  # modal_random_pool_50.py:80-83
        if final:
            break
    else:
        report['stopped'] = f'Common-prefix validation did not converge in {max_rounds} rounds'
        atomic_write_json(layout.validation_report(panel), report)
        raise ValidationStop(report['stopped'], report)
    problems = finalize_rows(panel, rows, lengths, indices)
    comparison = _compare_with_stored(stored, rows, indices)
    report.update(comparison=comparison, problems=problems)
    trims = valid_context_trims(report, stored, rows, problems)
    hard = [p for p in problems if p.get('kind') != VALID_CONTEXT_TRIM]
    report_path = layout.validation_report(panel)
    if hard:
        report['stopped'] = (f'{len(hard)} protocol violations after trimming (e.g. {hard[0]["error"]}); '
                             'a material change that needs the user\'s decision; no option continues')
        atomic_write_json(report_path, report)
        raise ValidationStop(report['stopped'] + f'; see {report_path}', report)
    if trims:
        report['valid_context_trims'] = trims
        if not accept_valid_context_trims:
            report['stopped'] = (f'{len(trims)} protocol violations after trimming (An already-valid original '
                                 'prefix changed); a material change that needs the user\'s decision')
            report['decision'] = decision_text(trims, report_path)
            atomic_write_json(report_path, report)
            raise ValidationStop(report['decision'], report)
        report['accepted_valid_context_trims'] = trims
        log(f'{panel}: ACCEPTED ({ACCEPT_FLAG}) {len(trims)} trims of originally valid contexts '
            f'({sum(t["trimmed_chars"] for t in trims)} characters); recorded in the plan and the report')
    plan = derived_plan(panel, original_plan, rows, indices, spec, stored_hash=stored_hash,
                        canonical_sha=canonical_sha, code_hashes=code_info['hf_hypernet_sha256'],
                        rounds=len(report['rounds']), trimmed=comparison['rows_trimmed'],
                        scripts_hashes=code_info.get('scripts_sha256'),
                        trimmed_row_ids=[t['id'] for t in comparison['trimmed']],
                        trimmed_chars=sum(t['trimmed_chars'] for t in comparison['trimmed']),
                        accepted_valid_context_trims=trims)
    files = [{'path': str(tokenizer_path(pool_dir, i)), 'sha256': spec['tokenizers'][i]['sha256']} for i in indices]
    preparation = dict(plan=plan, panel_sha256=plan['panel_sha256'],
                       metadata=[{k: v for k, v in r.items() if k != 'prefix'} for r in rows], files=files,
                       trimmed_passages=comparison['rows_trimmed'],
                       continuation_tokens=32 * len(rows), continuation_bytes=sum(r['continuation_bytes'] for r in rows))
    report.update(ok=True, panel_sha256=plan['panel_sha256'],
                  panel_sha256_equals_stored=plan['panel_sha256'] == stored_hash, finished_at=utc_now(),
                  seconds=time.monotonic() - begin)
    log(f'{panel}: validated in {len(report["rounds"])} round(s); {comparison["rows_identical_to_stored"]} rows '
        f'identical to the stored panel, {comparison["rows_trimmed"]} trimmed; panel sha256 {plan["panel_sha256"]} '
        f'(stored {stored_hash}, equal={report["panel_sha256_equals_stored"]})')
    return {'rows': rows, 'plan': plan, 'preparation': preparation, 'report': report}


def write_validated(layout: EvalLayout, panel: str, result: dict, *, force: bool = False) -> bool:
    """Write panel.json, preparation.json, plan.json and the report.  Returns True if anything changed.

    An existing validated panel with different content is replaced only with
    ``force`` (scores made from the old plan are then refused by score/pack).
    """
    plan = result['plan']
    if layout.plan(panel).exists():
        previous = read_json(layout.plan(panel))
        if previous == plan and layout.panel(panel).exists() and digest(read_json(layout.panel(panel))) == plan['panel_sha256']:
            result['report']['written'] = 'unchanged'
            atomic_write_json(layout.validation_report(panel), result['report'])
            return False
        if not force:
            result['report']['written'] = 'refused: a different validated panel exists'
            atomic_write_json(layout.validation_report(panel), result['report'])
            raise EvalError(f'{layout.plan(panel)} holds a different validated panel (old panel sha256 '
                            f'{previous.get("panel_sha256")}, new {plan["panel_sha256"]}); pass --force to replace it')
    result['report']['written'] = 'written'
    # panel and preparation first: plan.json is the marker that the set is complete.
    atomic_write_json(layout.panel(panel), result['rows'])
    atomic_write_json(layout.preparation(panel), result['preparation'])
    atomic_write_json(layout.plan(panel), plan)
    atomic_write_json(layout.validation_report(panel), result['report'])
    return True


def up_to_date(layout: EvalLayout, panel: str, spec: dict, indices: list) -> bool:
    """True if a validated panel for exactly this pool and index list already exists and verifies."""
    try:
        plan = read_json(layout.plan(panel))
        rows = read_json(layout.panel(panel))
    except (OSError, ValueError):
        return False
    hyak = plan.get('hyak', {})
    return (digest(rows) == plan.get('panel_sha256') and hyak.get('pool_index_sha256') == spec['index_sha256']
            and plan.get('indices') == list(indices)
            and hyak.get('stored_panel_sha256') == PANEL_SPECS[panel].panel_sha256)


def _failed(status, exc) -> None:
    """Failed status that keeps the last progress count (a stop must not show 0/1000)."""
    if status is not None:
        status.update(status.last[status.completed_field] if status.last else 0, state='failed',
                      error=str(exc)[:2000])


def main(argv=None, *, pool_reference_plan=True) -> int:
    """Exit codes: 0 validated or up to date (or another job is validating this panel); 2 usage/input error;
    3 protocol stop that needs the user's decision (see validation-report.json); 1 crash (traceback).
    ``pool_reference_plan`` is a test parameter only (a miniature world's TEST pool plan); see load_pool_spec."""
    parser = argparse.ArgumentParser(prog='python -m hyak.eval.validate_prefixes', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--indices', nargs='+', help='default: pool 0-999; natural the 38 plan indices')
    parser.add_argument('--workers', type=int, default=None, help='default: $SLURM_CPUS_PER_TASK, at most 8')
    parser.add_argument('--recheck', action='store_true', help='recompute even if an up-to-date panel exists')
    parser.add_argument('--force', action='store_true', help='replace an existing validated panel that differs')
    parser.add_argument('--claim-timeout-minutes', type=float, default=240.0)
    parser.add_argument(ACCEPT_FLAG, dest='accept_valid_context_trims', action='store_true',
                        help='pool panel only, and only after the user approved them: accept trims of contexts that '
                             'were valid in the original panel (listed in validation-report.json after a stop); '
                             'they are recorded in the plan, the report and the packed summaries')
    parser.add_argument('--no-status', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--code-dir', type=Path)
    args = parser.parse_args(argv)
    status = None
    claims = None
    if args.accept_valid_context_trims and args.panel != PANEL_POOL:
        log(f'{ACCEPT_FLAG} applies to the pool panel only; ignored for --panel {args.panel}')
        args.accept_valid_context_trims = False
    try:
        info = setup_imports(args.code_dir)
        from hyak.common.config import load_config
        from hyak.common.paths import Paths
        from hyak.common.status import StatusWriter
        from hyak.eval.prepare_features import default_workers
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
        layout = EvalLayout(paths)
        spec = load_pool_spec(paths.pool_dir, reference_plan=pool_reference_plan)
        indices = parse_indices(args.indices, range(1000)) if args.indices else default_indices(args.panel)
        workers = args.workers or default_workers()
        if not args.recheck and up_to_date(layout, args.panel, spec, indices):
            log(f'{args.panel}: {layout.panel(args.panel)} is up to date for this pool; nothing to do')
            return 0
        if layout.plan(args.panel).exists() and not args.force and not args.recheck:
            raise EvalError(f'{layout.plan(args.panel)} exists for another pool or index list; pass --force to '
                            'replace it (scores made with the old panel will then be refused)')
        claims = Claims(layout.claims('validate'), args.claim_timeout_minutes * 60)
        if not claims.try_acquire(args.panel, task=f'validate {args.panel}'):
            log(f'{args.panel}: another job is validating this panel ({claims.owner(args.panel)}); exiting')
            claims = None
            return 0
        if not args.no_status:
            status = StatusWriter(STAGE_EVAL_PREPARE, len(indices), 'tokenizers', cfg=cfg)
        result = validate(layout, args.panel, spec, paths.pool_dir, paths.model_dir / 'tokenizer.json', indices,
                          workers=workers, code_info=info, status=status,
                          accept_valid_context_trims=args.accept_valid_context_trims)
        changed = write_validated(layout, args.panel, result, force=args.force)
        report = result['report']
        log(f'{args.panel}: {"wrote" if changed else "unchanged"} {layout.panel(args.panel)}')
        if status is not None:
            status.event({'event': 'prefixes_validated', 'panel': args.panel, 'rounds': len(report['rounds']),
                          'rows_trimmed': report['comparison']['rows_trimmed'],
                          'panel_sha256': report['panel_sha256'], 'changed': changed,
                          'accepted_valid_context_trims': len(report.get('accepted_valid_context_trims', []))})
            from hyak.eval.prepare_features import report_progress
            report_progress(status, layout, spec, paths.model_dir / 'tokenizer.json',
                            elapsed_seconds=report.get('seconds'))
        return EXIT_OK
    except ValidationStop as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if 'validation-report.json' not in str(exc):
            print(f'See {layout.validation_report(args.panel)}', file=sys.stderr)
        _failed(status, exc)
        return EXIT_PROTOCOL
    except (EvalError, ConfigError, OSError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        _failed(status, exc)
        return EXIT_USAGE
    except Exception:
        return crashed(status)
    finally:
        if claims is not None:
            try:
                claims.release(args.panel)
            except EvalError:
                pass


if __name__ == '__main__':
    sys.exit(main())
