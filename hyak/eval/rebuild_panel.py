"""Rebuild the two stored evaluation panels from their canonical token ids and verify every hash.

    python -m hyak.eval.rebuild_panel [--panel pool|natural|all] [--tokenizer FILE]
                                      [--out-dir DIR] [--config FILE] [--code-dir DIR] [--check-only]

The stored bundles keep every panel row without its raw ``prefix`` string
(stripped at scripts/modal_frozen_policy_cache.py:135 and
scripts/modal_natural_transfer.py:120).  A prefix entered a final panel only if
``canonical.decode(canonical(prefix).input_ids) == prefix``
(scripts/original_pool_check.py:45, scripts/pool_regime.py:18,
scripts/natural_transfer.py:168), and ``canonical_prefix_ids`` is
``canonical(prefix).input_ids`` (scripts/pool_regime.py:33), so decoding the ids
with the same canonical tokenizer returns the prefix.  Each row is then checked
against everything the construction code guaranteed, and the whole panel against
its recorded hash.  Any mismatch is a hard failure: no panel file is written.

Default tokenizer: ``$ANYTOK_ROOT/input/model/tokenizer.json`` (SHA-256 pinned).
Outputs: ``<out-dir>/<panel>/rebuilt.json`` (rows with ``prefix``) and
``<out-dir>/<panel>/rebuild-report.json``; default out-dir ``$ANYTOK_ROOT/eval/panels``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from hyak.common.config import ConfigError
from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now
from hyak.eval import (CANONICAL_PREFIX_MAX, CANONICAL_PREFIX_MIN, EXIT_CHECK, PANEL_POOL, PANEL_SPECS, PANELS,
                       POSITION_CAP, SUFFIX_TOKENS, EvalError, crashed, digest, load_canonical, log, read_bundle, repo_root,
                       setup_imports, validate_panel_name)


def load_stored_rows(panel: str, repo: Path | None = None):
    """Rows (without prefix) of a stored panel, the expected whole-panel hash and the original plan.

    Returns ``(rows, expected_panel_sha256, original_plan, sources)``.  Checks that
    the repository bundles are the committed ones (SHA-256) and, for the natural
    panel, that both bundles hold identical rows.
    """
    spec = PANEL_SPECS[validate_panel_name(panel)]
    repo = Path(repo) if repo is not None else repo_root()
    source = repo / spec.source_rel
    actual = sha256_file(source)
    if actual != spec.source_sha256:
        raise EvalError(f'{source} sha256 {actual} != committed {spec.source_sha256}; restore it with git')
    sources = {spec.source_rel: actual}
    bundle = read_bundle(source)
    plan = read_json(repo / spec.plan_rel)
    if panel == PANEL_POOL:
        rows = bundle['rows']
        stored_plan = bundle['plan']
        expected = stored_plan['panel_sha256']
        # The reward table's plan is the frozen plan plus the prepared panel hash
        # (scripts/modal_frozen_policy_cache.py:125; scripts/frozen_policy_search.py:78).
        if {k: v for k, v in stored_plan.items() if k != 'panel_sha256'} != plan:
            raise EvalError(f'{spec.plan_rel} differs from the plan stored in {spec.source_rel}')
    else:
        rows = bundle['preparation']['metadata']
        expected = bundle['preparation']['panel_sha256']
        twin = repo / spec.twin_rel
        twin_sha = sha256_file(twin)
        if twin_sha != spec.twin_sha256:
            raise EvalError(f'{twin} sha256 {twin_sha} != committed {spec.twin_sha256}; restore it with git')
        sources[spec.twin_rel] = twin_sha
        twin_bundle = read_bundle(twin)
        if twin_bundle['metadata'] != rows:
            raise EvalError(f'{spec.twin_rel} metadata differ from {spec.source_rel} preparation.metadata')
        if twin_bundle['plan']['panel_sha256'] != expected or plan['panel_sha256'] != expected:
            raise EvalError('Natural panel hash differs between the stored bundles and the plan')
    if expected != spec.panel_sha256:
        raise EvalError(f'Stored {panel} panel hash {expected} != pinned {spec.panel_sha256}')
    if len(rows) != spec.rows:
        raise EvalError(f'Stored {panel} panel has {len(rows)} rows, expected {spec.rows}')
    return rows, expected, plan, sources


def check_row(meta: dict, canonical) -> tuple:
    """Decode one row and check it.  Returns ``(prefix, suffix, problems)``.

    Conditions are the ones each row satisfied when it was built:
    scripts/original_pool_check.py:33-47 and :65-66, scripts/pool_regime.py:15-20 and :33-37,
    scripts/natural_transfer.py:154-171 and :188-190.
    """
    problems = []
    prefix_ids = meta['canonical_prefix_ids']
    targets = meta['canonical_continuation_ids']
    prefix = canonical.decode(prefix_ids, skip_special_tokens=False)
    suffix = canonical.decode(targets, skip_special_tokens=False)
    if canonical(prefix, add_special_tokens=False).input_ids != prefix_ids:
        problems.append('decoded prefix does not re-encode to canonical_prefix_ids')
    if canonical(suffix, add_special_tokens=False).input_ids != targets:
        problems.append('decoded continuation does not re-encode to canonical_continuation_ids')
    if canonical(prefix + suffix, add_special_tokens=False).input_ids != prefix_ids + targets:
        problems.append('joint canonical encoding of prefix + continuation differs')
    if digest([prefix, targets]) != meta['context_sha256']:
        problems.append('context_sha256 mismatch')
    if len(suffix.encode()) != meta['continuation_bytes']:
        problems.append(f'continuation has {len(suffix.encode())} bytes, row says {meta["continuation_bytes"]}')
    if len(targets) != SUFFIX_TOKENS or 0 in targets:
        problems.append('continuation must be 32 canonical ids without id 0')
    if not CANONICAL_PREFIX_MIN <= len(prefix_ids) <= CANONICAL_PREFIX_MAX or 0 in prefix_ids:
        problems.append(f'canonical prefix has {len(prefix_ids)} ids (16..64 without id 0 required)')
    if meta.get('canonical_joint_encoding_matches') is not True:
        problems.append('canonical_joint_encoding_matches is not true')
    lengths = meta.get('native_prefix_lengths') or {}
    if not lengths:
        problems.append('native_prefix_lengths missing')
    elif meta.get('maximum_positions') != 33 + max(len(prefix_ids), max(lengths.values())) \
            or meta['maximum_positions'] > POSITION_CAP:
        problems.append('maximum_positions inconsistent or above 128')
    return prefix, suffix, problems


def rebuild_rows(rows: list, canonical) -> tuple:
    """Rebuilt rows (each stored row plus ``prefix``) and ``[(row id, problems)]`` for failing rows."""
    rebuilt, failures = [], []
    for meta in rows:
        if 'prefix' in meta:
            raise EvalError(f'stored row {meta.get("id")} already has a prefix; not a stored panel')
        prefix, _, problems = check_row(meta, canonical)
        if problems:
            failures.append((meta['id'], problems))
        rebuilt.append({**meta, 'prefix': prefix})
    return rebuilt, failures


def panel_problems(panel: str, rows: list) -> list:
    """Whole-panel structure: unique ids and the per-split / per-domain counts of the stored panels."""
    problems = []
    spec = PANEL_SPECS[panel]
    if len({r['id'] for r in rows}) != len(rows):
        problems.append('duplicate row ids')
    for split, count in spec.splits:
        found = sum(r['split'] == split for r in rows)
        if found != count:
            problems.append(f'{found} {split} rows, expected {count}')
    if panel == PANEL_POOL:
        from uniform_trajectory import PROFILE_DOMAINS
        for split, _ in spec.splits:
            counts = Counter(r['domain'] for r in rows if r['split'] == split)
            if counts != Counter({d: 32 for d in PROFILE_DOMAINS}):
                problems.append(f'{split}: expected 32 rows in each of the nine evaluation buckets')
    else:
        from natural_transfer import DOMAINS
        if not {r['domain'] for r in rows} <= set(DOMAINS):
            problems.append('natural panel contains a domain outside natural_transfer.DOMAINS')
    return problems


def rebuild(panel: str, tokenizer: Path, *, repo: Path | None = None, require_pinned: bool = True) -> tuple:
    """Rebuild one panel in memory.  Returns ``(rows, report)``; raises EvalError on any mismatch."""
    canonical, tokenizer_sha = load_canonical(tokenizer, require_pinned=require_pinned)
    stored, expected, _, sources = load_stored_rows(panel, repo)
    rows, failures = rebuild_rows(stored, canonical)
    structure = panel_problems(panel, rows)
    actual = digest(rows)
    report = {
        'panel': panel, 'created_at': utc_now(), 'sources': sources,
        'canonical_tokenizer': str(tokenizer), 'canonical_tokenizer_sha256': tokenizer_sha,
        'decode': 'tokenizers.Tokenizer.from_file(tokenizer.json).decode(ids, skip_special_tokens=False) '
                  '(uniform_trajectory.CanonicalTokenizer)',
        'rows': len(rows), 'verified_rows': len(rows) - len(failures),
        'failed_rows': [{'id': i, 'problems': p} for i, p in failures],
        'panel_problems': structure,
        'expected_panel_sha256': expected, 'panel_sha256': actual, 'panel_sha256_matches': actual == expected,
        'splits': {split: sum(r['split'] == split for r in rows) for split, _ in PANEL_SPECS[panel].splits},
        'domains': dict(sorted(Counter(r['domain'] for r in rows).items())),
    }
    report['ok'] = not failures and not structure and actual == expected
    return rows, report


class RebuildCheckFailed(EvalError):
    """A stored panel did not rebuild exactly (exit 4: a check failed)."""


def write_outputs(panel: str, rows: list, report: dict, out_dir: Path) -> Path:
    """Write ``rebuilt.json`` (only when verified) and the report; refuse to replace different content."""
    folder = Path(out_dir) / panel
    target = folder / 'rebuilt.json'
    report_path = folder / 'rebuild-report.json'
    if not report['ok']:
        atomic_write_json(report_path, report)
        raise RebuildCheckFailed(f'{panel} panel did not verify ({report["verified_rows"]}/{report["rows"]} rows, '
                        f'panel hash match {report["panel_sha256_matches"]}, problems {report["panel_problems"]}); '
                        f'nothing written except {report_path}. Check that --tokenizer is the pinned SmolLM2 '
                        'tokenizer.json and that tokenizers==0.22.2 is installed.')
    if target.exists():
        existing = read_json(target)
        if digest(existing) != report['panel_sha256']:
            raise EvalError(f'{target} exists with different content; move it away to rebuild')
        log(f'{target} already holds the verified panel; unchanged')
    else:
        atomic_write_json(target, rows)
    atomic_write_json(report_path, report)
    return target


def main(argv=None) -> int:
    """Exit codes: 0 rebuilt and verified (or already present); 4 a stored panel did not verify (see
    rebuild-report.json); 2 usage or input error; 1 crash (traceback)."""
    parser = argparse.ArgumentParser(prog='python -m hyak.eval.rebuild_panel', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panel', choices=PANELS + ('all',), default='all')
    parser.add_argument('--tokenizer', type=Path, help='canonical tokenizer.json (default: $ANYTOK_ROOT/input/model/tokenizer.json)')
    parser.add_argument('--out-dir', type=Path, help='default: $ANYTOK_ROOT/eval/panels')
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--code-dir', type=Path, help='pinned code tree (default: <repo>/runtime/anytok_pool_20k)')
    parser.add_argument('--check-only', action='store_true', help='verify and report; write nothing')
    args = parser.parse_args(argv)
    try:
        setup_imports(args.code_dir)
        tokenizer, out_dir = args.tokenizer, args.out_dir
        if tokenizer is None or (out_dir is None and not args.check_only):
            from hyak.common.config import load_config
            from hyak.common.paths import Paths
            paths = Paths.from_config(load_config(args.config))
            tokenizer = tokenizer or paths.model_dir / 'tokenizer.json'
            out_dir = out_dir or paths.eval_panels
        status = 0
        for panel in (PANELS if args.panel == 'all' else (args.panel,)):
            rows, report = rebuild(panel, tokenizer)
            log(f'{panel}: {report["verified_rows"]}/{report["rows"]} rows verified; panel sha256 '
                f'{report["panel_sha256"]} (expected {report["expected_panel_sha256"]}, '
                f'match={report["panel_sha256_matches"]})')
            if args.check_only:
                status = status or (0 if report['ok'] else EXIT_CHECK)
                if not report['ok']:
                    log(f'{panel}: FAILED rows {report["failed_rows"][:5]} problems {report["panel_problems"]}')
                continue
            target = write_outputs(panel, rows, report, out_dir)
            log(f'{panel}: wrote {target}')
        return status
    except RebuildCheckFailed as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return EXIT_CHECK
    except (EvalError, ConfigError, OSError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    except Exception:
        return crashed()


if __name__ == '__main__':
    sys.exit(main())
