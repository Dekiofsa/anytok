"""Hypernetwork input features of every pool tokenizer (CPU; claim-based, several jobs may share the work).

    python -m hyak.eval.prepare_features [--indices 0-999 ...] [--workers N] [--checkpoint DIR]
                                         [--min-remaining-minutes 5] [--claim-timeout-minutes 120]
                                         [--verify] [--config FILE] [--code-dir DIR]

Port of the feature part of scripts/modal_eval_rl_pool.py:129-165
(``prepare_features``; its old-cut ``fit_start`` part is not used by the corrected
protocol).  For each index: tokenizer.json hash against the pool index,
``uniform_trajectory.load_candidate(path, 'pool')``, vocabulary size check,
``build_surface_forms(tokens, canonical, 10, fallbacks, specials)`` with
``fallbacks = {s: len(canonical) + 1 + i for s in missing_byte_symbols(canonical)}``,
then ``features/NNNN/forms.npy`` (int32, ``(vocab, 10)``, padded with 49152),
``priors.npy`` (float32, ``(vocab,)``) and ``manifest.json`` with the original keys
``tokenizer_sha256, forms_sha256, priors_sha256, truncated, max_pieces, fallbacks``
plus provenance.  All functions are imported from the pinned tree / scripts.

Each feature directory is built in a temporary directory and renamed into
place, so a directory ``NNNN`` is always complete.  Work is shared through
atomic-mkdir claims in ``$ANYTOK_ROOT/eval/claims/features``; a claim whose owner
job is gone (squeue) or that is older than the claim timeout can be taken over.
SIGTERM stops claiming new work; running builds finish; exit 0.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now
from hyak.common.paths import STAGE_EVAL_PREPARE, pool_index_name
from hyak.common.config import ConfigError
from hyak.eval import (EXIT_OK, EXIT_USAGE, MAX_PIECES, PANELS, Claims, EvalError, EvalLayout, check_tokenizer_file,
                       crashed, load_canonical, load_pool_spec, log, parse_indices, remaining_seconds, setup_imports)

MANIFEST_KEYS = ('tokenizer_sha256', 'forms_sha256', 'priors_sha256', 'truncated', 'max_pieces', 'fallbacks')


def expected_fallbacks(canonical) -> dict:
    """``{symbol: len(canonical) + 1 + i}`` for the bytes the canonical BPE cannot represent.

    scripts/modal_eval_rl_pool.py:139-142 (``missing_byte_symbols`` imported from
    hf_hypernet.tokenizer_inputs); identical to what ``enable_byte_fallbacks``
    returns for a checkpoint (tokenizer_inputs.py:39).
    """
    from hf_hypernet.tokenizer_inputs import missing_byte_symbols
    symbols = missing_byte_symbols(canonical)
    return {s: len(canonical) + 1 + i for i, s in enumerate(symbols)}


def check_checkpoint_symbols(checkpoint: Path, fallbacks: dict) -> None:
    """``symbols != config['hn_byte_fallback_symbols']`` -> error (scripts/modal_eval_rl_pool.py:138-141)."""
    config = read_json(Path(checkpoint) / 'hypernet' / 'config.json')
    if list(fallbacks) != config.get('hn_byte_fallback_symbols'):
        raise EvalError(f'Byte fallback configuration changed: {checkpoint}/hypernet/config.json lists '
                        f'{config.get("hn_byte_fallback_symbols")}, the canonical tokenizer gives {list(fallbacks)}')


def build_features(entry: dict, path: Path, canonical, fallbacks: dict, destination: Path, provenance=None) -> dict:
    """Write forms.npy, priors.npy and manifest.json for one tokenizer into ``destination`` (must not exist).

    Operations of scripts/modal_eval_rl_pool.py:152-165 in order.  Returns the manifest.
    """
    import numpy as np
    from hf_hypernet.tokenizer_inputs import build_surface_forms
    from uniform_trajectory import load_candidate
    tokenizer, tokens, priors, specials, forbidden = load_candidate(path, 'pool')
    if len(tokens) != entry['vocab_size']:
        raise EvalError(f'Native vocabulary size mismatch for {path}: {len(tokens)} != {entry["vocab_size"]}')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    forms, truncated = build_surface_forms(tokens, canonical, MAX_PIECES, fallbacks, specials)
    np.save(destination / 'forms.npy', forms)
    np.save(destination / 'priors.npy', np.asarray(priors, dtype=np.float32))
    manifest = {'tokenizer_sha256': entry['sha256'],
                'forms_sha256': sha256_file(destination / 'forms.npy'),
                'priors_sha256': sha256_file(destination / 'priors.npy'), 'truncated': truncated,
                'max_pieces': MAX_PIECES, 'fallbacks': fallbacks}
    manifest.update(index=entry['index'], vocab_size=len(tokens), forms_shape=list(forms.shape),
                    forms_dtype=str(forms.dtype), special_ids=sorted(int(i) for i in forbidden),
                    created_at=utc_now(), **(provenance or {}))
    atomic_write_json(destination / 'manifest.json', manifest)
    return manifest


def feature_problem(folder: Path, entry: dict, fallbacks: dict, *, deep: bool = False):
    """None if ``folder`` holds this tokenizer's features, else a reason (missing / mismatching).

    The shallow check reads the manifest; ``deep`` also re-hashes both arrays and
    checks their shapes and dtypes, as the scorer does before use
    (scripts/modal_random_pool_50.py:185-195).
    """
    manifest_path = Path(folder) / 'manifest.json'
    if not manifest_path.is_file():
        return 'missing'
    try:
        manifest = read_json(manifest_path)
    except ValueError as exc:
        return f'unreadable manifest ({exc})'
    if any(key not in manifest for key in MANIFEST_KEYS):
        return 'manifest lacks the original keys'
    if manifest['tokenizer_sha256'] != entry['sha256']:
        return 'built for another tokenizer file'
    if manifest['fallbacks'] != fallbacks or manifest['max_pieces'] != MAX_PIECES:
        return 'built with another byte-fallback mapping'
    if deep:
        import numpy as np
        for name in ('forms', 'priors'):
            if sha256_file(Path(folder) / f'{name}.npy') != manifest[f'{name}_sha256']:
                return f'{name}.npy does not match its manifest hash'
        forms = np.load(Path(folder) / 'forms.npy', mmap_mode='r')
        priors = np.load(Path(folder) / 'priors.npy', mmap_mode='r')
        if forms.dtype != np.int32 or forms.shape != (entry['vocab_size'], MAX_PIECES):
            return f'forms.npy has {forms.dtype} {forms.shape}'
        if priors.dtype != np.float32 or priors.shape != (entry['vocab_size'],):
            return f'priors.npy has {priors.dtype} {priors.shape}'
    return None


# --- worker processes ------------------------------------------------------------------------------
_WORKER = {}


def _init_worker(code_dir: str, tokenizer: str):
    setup_imports(code_dir)
    canonical, _ = load_canonical(tokenizer)
    _WORKER['canonical'] = canonical
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _build_one(entry: dict, path: str, fallbacks: dict, features_root: str, provenance: dict) -> dict:
    canonical = _WORKER['canonical']
    root = Path(features_root)
    final = root / pool_index_name(entry['index'])
    temporary = root / f'.tmp-{pool_index_name(entry["index"])}-{uuid.uuid4().hex}'
    begin = time.monotonic()
    try:
        manifest = build_features(entry, Path(path), canonical, fallbacks, temporary, provenance)
        try:
            os.rename(temporary, final)
        except OSError:
            if feature_problem(final, entry, fallbacks) is None:
                return {'index': entry['index'], 'status': 'existing', 'seconds': time.monotonic() - begin}
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return {'index': entry['index'], 'status': 'built', 'truncated': manifest['truncated'],
            'seconds': time.monotonic() - begin}


# --- driver ------------------------------------------------------------------------------------------
class _Stop:
    def __init__(self):
        self.requested = False
        self.reason = None

    def __call__(self, signum, frame):
        self.requested = True
        self.reason = signal.Signals(signum).name


def prepare(layout: EvalLayout, spec: dict, pool_dir: Path, canonical_path: Path, indices: list, *,
            workers: int, code_dir: str, claim_timeout: float, min_remaining: float, status=None,
            stop=None, deep_verify: bool = False, provenance=None) -> dict:
    """Build every missing feature directory of ``indices``.  Returns counts.  Raises on the first failure."""
    canonical, canonical_sha = load_canonical(canonical_path)
    fallbacks = expected_fallbacks(canonical)
    features_root = layout.paths.eval_features
    features_root.mkdir(parents=True, exist_ok=True)
    claims = Claims(layout.claims('features'), claim_timeout)
    entries = {e['index']: e for e in spec['tokenizers']}
    stop = stop or _Stop()
    provenance = {'canonical_tokenizer_sha256': canonical_sha, **(provenance or {})}

    def done_count():
        return sum(feature_problem(layout.feature_dir(i), entries[i], fallbacks) is None for i in indices)

    def present_count():   # cheap (one stat per index) for frequent status updates
        return sum((layout.feature_dir(i) / 'manifest.json').is_file() for i in indices)

    todo = []
    for index in indices:
        problem = feature_problem(layout.feature_dir(index), entries[index], fallbacks, deep=deep_verify)
        if problem is None:
            continue
        if problem != 'missing':
            raise EvalError(f'{layout.feature_dir(index)} exists but is not valid ({problem}); delete that '
                            'directory and rerun (features must match the current pool)')
        todo.append(index)
    log(f'features: {len(indices) - len(todo)}/{len(indices)} present; {len(todo)} to build with {workers} workers')
    counts = {'built': 0, 'existing': len(indices) - len(todo), 'skipped_claimed': 0, 'stopped': False}
    begin = time.monotonic()

    last_report = [0.0]

    def report(stage='preparing hypernetwork features', state=None, force=False):
        if status is None or (not force and time.monotonic() - last_report[0] < 15):
            return
        last_report[0] = time.monotonic()
        done = present_count()
        rate = (time.monotonic() - begin) / counts['built'] if counts['built'] else None
        eta = rate * (len(indices) - done) if rate else None
        status.update(done, stage=stage, state=state, total=len(indices), eta_seconds=eta,
                      elapsed_seconds=time.monotonic() - begin, task='features', workers=None)

    report(force=True)
    context = multiprocessing.get_context('spawn')
    pending = {}
    queue = list(todo)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker,
                             initargs=(str(code_dir), str(canonical_path))) as pool:
        try:
            while queue or pending:
                while queue and len(pending) < workers and not stop.requested:
                    if remaining_seconds() < min_remaining:
                        stop.requested, stop.reason = True, 'job time limit approaching'
                        break
                    index = queue.pop(0)
                    if feature_problem(layout.feature_dir(index), entries[index], fallbacks) is None:
                        counts['existing'] += 1
                        continue
                    if not claims.try_acquire(pool_index_name(index), task='features'):
                        counts['skipped_claimed'] += 1
                        continue
                    if feature_problem(layout.feature_dir(index), entries[index], fallbacks) is None:
                        claims.release_after_work(pool_index_name(index))
                        counts['existing'] += 1
                        continue
                    try:
                        path = check_tokenizer_file(pool_dir, spec, index)
                        future = pool.submit(_build_one, entries[index], str(path), fallbacks, str(features_root),
                                             provenance)
                    except BaseException:
                        claims.release_after_work(pool_index_name(index))
                        raise
                    pending[future] = index
                if stop.requested and not pending:
                    break
                if not pending:
                    continue
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    try:
                        result = future.result()
                    finally:
                        claims.release_after_work(pool_index_name(index))
                    counts['built' if result['status'] == 'built' else 'existing'] += 1
                    log(f'features {pool_index_name(index)}: {result["status"]} in {result["seconds"]:.1f} s'
                        + (f' ({result["truncated"]} truncated forms)' if 'truncated' in result else ''))
                    report()
        except BaseException:
            for future in pending:
                future.cancel()
            for future, index in list(pending.items()):
                try:
                    future.result()
                except BaseException:
                    pass
                try:
                    claims.release(pool_index_name(index))
                except EvalError:
                    pass
            raise
    counts['stopped'] = stop.requested
    counts['takeovers'] = claims.takeovers
    counts['lost_claims'] = claims.lost
    counts['done'] = done_count()
    counts['total'] = len(indices)
    if stop.requested:
        log(f'features: stopped ({stop.reason}); {counts["done"]}/{len(indices)} present')
    report('features complete' if counts['done'] == len(indices) else 'preparing hypernetwork features', force=True)
    return counts


def report_progress(status, layout: EvalLayout, spec: dict, canonical_path: Path, *,
                    elapsed_seconds: float | None = None) -> dict:
    """Overall eval-prepare state: features of all 1,000 tokenizers and both validated panels.

    Writes ``complete`` only when everything the scorers need exists for the
    current pool; otherwise a running state whose stage says what is missing.
    ``elapsed_seconds``: the caller's working time, shown on the dashboard.
    """
    from hyak.eval.validate_prefixes import default_indices, up_to_date
    canonical, _ = load_canonical(canonical_path)
    fallbacks = expected_fallbacks(canonical)
    entries = {e['index']: e for e in spec['tokenizers']}
    features = sum(feature_problem(layout.feature_dir(i), entries[i], fallbacks) is None for i in range(1000))
    panels = {panel: up_to_date(layout, panel, spec, default_indices(panel)) for panel in PANELS}
    done = features == 1000 and all(panels.values())
    missing = [f'features {features}/1000'] if features < 1000 else []
    missing += [f'{p} panel not validated' for p, ok in panels.items() if not ok]
    if status is not None:
        status.update(features, stage='complete' if done else 'waiting: ' + ', '.join(missing),
                      state='complete' if done else 'running', total=1000, elapsed_seconds=elapsed_seconds,
                      panels_validated=[p for p, ok in panels.items() if ok], workers=[] if done else None)
    return {'features': features, 'panels': panels, 'complete': done}


def default_workers() -> int:
    value = os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1
    return max(1, min(8, int(value)))


def main(argv=None, *, pool_reference_plan=True) -> int:
    """``pool_reference_plan`` is a test parameter only (a miniature world's TEST pool plan); see load_pool_spec."""
    parser = argparse.ArgumentParser(prog='python -m hyak.eval.prepare_features', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--indices', nargs='+', help='pool indices, e.g. 0-999 or 3 4 11 (default: all 1,000)')
    parser.add_argument('--workers', type=int, default=default_workers(),
                        help='parallel build processes (default: $SLURM_CPUS_PER_TASK, at most 8)')
    parser.add_argument('--checkpoint', type=Path,
                        help='optional trained checkpoint: cross-check its hn_byte_fallback_symbols')
    parser.add_argument('--min-remaining-minutes', type=float, default=5.0)
    parser.add_argument('--claim-timeout-minutes', type=float, default=120.0)
    parser.add_argument('--verify', action='store_true', help='re-hash existing feature arrays before skipping them')
    parser.add_argument('--no-status', action='store_true', help='do not write dashboard status (local runs)')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--code-dir', type=Path, help='pinned code tree (default: <repo>/runtime/anytok_pool_20k)')
    args = parser.parse_args(argv)
    stop = _Stop()
    signal.signal(signal.SIGTERM, stop)
    status = None
    try:
        info = setup_imports(args.code_dir)
        from hyak.common.config import load_config
        from hyak.common.paths import Paths
        from hyak.common.status import StatusWriter
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
        layout = EvalLayout(paths)
        spec = load_pool_spec(paths.pool_dir, reference_plan=pool_reference_plan)
        indices = parse_indices(args.indices, range(1000))
        if args.workers < 1:
            raise EvalError('--workers must be >= 1')
        canonical_path = paths.model_dir / 'tokenizer.json'
        if args.checkpoint:
            canonical, _ = load_canonical(canonical_path)
            check_checkpoint_symbols(args.checkpoint, expected_fallbacks(canonical))
        if not args.no_status:
            status = StatusWriter(STAGE_EVAL_PREPARE, len(indices), 'tokenizers', cfg=cfg)
        from hyak.common import slurm
        provenance = {'pool_index_sha256': spec['index_sha256'], 'job_id': slurm.worker_id(),
                      'host': slurm.node_name(),
                      'tokenizer_inputs_sha256': info['hf_hypernet_sha256']['hf_hypernet/tokenizer_inputs.py']}
        begin = time.monotonic()
        counts = prepare(layout, spec, paths.pool_dir, canonical_path, indices, workers=args.workers,
                         code_dir=info['code_dir'], claim_timeout=args.claim_timeout_minutes * 60,
                         min_remaining=args.min_remaining_minutes * 60, status=status, stop=stop,
                         deep_verify=args.verify, provenance=provenance)
        log(f'features: {counts}')
        if status is not None:
            status.event({'event': 'features', **{k: v for k, v in counts.items() if k not in ('takeovers', 'lost_claims')},
                          'takeovers': len(counts['takeovers']), 'lost_claims': len(counts['lost_claims'])})
            report_progress(status, layout, spec, canonical_path, elapsed_seconds=time.monotonic() - begin)
        return EXIT_OK
    except (EvalError, ConfigError, OSError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        if status is not None:
            status.update(status.last[status.completed_field] if status.last else 0, state='failed',
                          error=str(exc)[:2000])
        return EXIT_USAGE
    except Exception:
        return crashed(status)


if __name__ == '__main__':
    sys.exit(main())
