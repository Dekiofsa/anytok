"""Write the pool index once and prove the trainers accept it (DESIGN.md 5.F; port of ``finish()`` in
scripts/run_tokenizer_pool_farm.py:185-200).

    python -m hyak.pool.collect [--check-only] [--pool-dir DIR] [--code-dir DIR] [--rl-code-dir DIR]

1. Every one of the 1,000 manifests must verify (``hyak.pool.manifest_problems``:
   status complete, config equal to the plan's, ``vocab_size_actual`` as configured,
   ``0 <= 200,000,000 - total_used_bytes <= 69``, tokenizer bytes and SHA-256, a
   structurally valid ``tokenizer.json``) and all must share one provenance identity
   (plan, worker script, threads, wheel, patches, upstream revision, recipe, corpus).
   Fewer than 1,000 verified tokenizers: nothing is written.
2. ``plan.json`` = byte copy of plans/tokenizer-pool-1000-v1.json (as
   scripts/modal_generate_tokenizer_pool.py:201 uploaded it).
3. ``full-results.json`` = ``json.dumps({"plan_sha256", "completed": 1000, "tokenizers":
   [manifests ordered by index]}, indent=2) + "\\n"`` (scripts/run_tokenizer_pool_farm.py:194-196).
   Its bytes are hashed into both runs (``pool_index_sha256``), so it is written exactly
   once: an existing file is never replaced; its SHA-256 is printed and it is checked
   against the manifests instead.
4. Before the index is published it is loaded with the pinned
   ``hf_hypernet.pool_sampling.TokenizerPool`` (all 1,000 ``load(i)``: hash, contiguous
   ids, finite priors, ``<|endoftext|>``) and the pinned RL ``SettingsTokenizerBandit``
   (feature matrix (1000, 18), feature digest ``9fe7f9e5...``, first action 652).

``--check-only`` verifies the manifests (any number, also a partial pool) and writes
nothing.  Run with the ``train`` environment (numpy, tokenizers, torch).

Signals: the first SIGTERM (Slurm's early-warning signal, forwarded by the job script about 45
minutes before the limit) is noted and collect finishes (the index is published atomically with a
hard link, never partially); a second SIGTERM or a SIGINT stops it at once, removes its temporary
proof directory and exits 75 without an index.  Leftover ``.collect-proof-*`` directories of a killed
run are removed at the next start.

Exit status: 0 success (index written or already present and consistent); 75 the pool is not
complete yet (only missing or failed indices) or collect was stopped by a signal: nothing written,
try again later; 4 a manifest is invalid or inconsistent, or an existing index differs (the job log
lists the indices); 2 invalid inputs; 1 only for an unexpected crash (traceback).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import traceback

from hyak.common.config import ConfigError, load_config
from hyak.common.io import atomic_copy, atomic_write_json, fsync_dir, read_json, utc_now
from hyak.common.paths import STAGE_POOL, Paths
from hyak.pool import (
    FEATURE_SHAPE, FEATURE_SHA256, FIRST_POLICY_ACTION, PLAN_COUNT, PLAN_REL, POOL_SAMPLING_REL, POOL_SAMPLING_SHA256,
    SETTINGS_POLICY_REL, SETTINGS_POLICY_SHA256, PoolError, file_sha256, identity_of, index_name, load_plan,
    manifest_problems, tokenizer_json_problems)

EXIT_OK, EXIT_INPUT, EXIT_INVALID, EXIT_NOT_NOW = 0, 2, 4, 75
#: Kept for callers of the old name: an incomplete pool is "not now" (75).
EXIT_INCOMPLETE = EXIT_NOT_NOW
#: Reason of an index that simply has no result yet (missing or failed): the pool is incomplete, not invalid.
NOT_YET = 'no manifest.json'


class Interrupted(BaseException):
    """Second SIGTERM or SIGINT: stop now (finally blocks clean up; nothing is published)."""


class _Signals:
    """First SIGTERM: note it and finish; second SIGTERM or SIGINT: raise Interrupted."""

    def __init__(self):
        self.terms = 0
        self.previous = {}

    def __enter__(self):
        def on_term(signum, _frame):
            self.terms += 1
            if self.terms == 1:
                print('[hyak pool collect] SIGTERM received: finishing (the index is published atomically or not at '
                      'all); a second SIGTERM stops now', file=sys.stderr, flush=True)
                return
            raise Interrupted(f'signal {signum}')

        def on_int(signum, _frame):
            raise Interrupted(f'signal {signum}')

        try:
            self.previous = {signal.SIGTERM: signal.signal(signal.SIGTERM, on_term),
                             signal.SIGINT: signal.signal(signal.SIGINT, on_int)}
        except ValueError:  # not the main thread (tests)
            self.previous = {}
        return self

    def __exit__(self, *exc):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
        return False


def _cleanup_proofs(pool_dir: Path, older_than=600.0) -> None:
    """Remove ``.collect-proof-*`` directories a killed collect left in the pool directory."""
    try:
        entries = list(Path(pool_dir).glob('.collect-proof-*'))
    except OSError:
        return
    for entry in entries:
        try:
            if time.time() - entry.lstat().st_mtime > older_than:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def _incomplete_only(problems: dict) -> bool:
    """True when every problem is a missing result (not an invalid or inconsistent manifest)."""
    return all(len(reasons) == 1 and reasons[0].startswith(NOT_YET) for reasons in problems.values())


def index_text(plan: dict, manifests: dict) -> str:
    """Bytes of full-results.json, verbatim scripts/run_tokenizer_pool_farm.py:194-196."""
    return json.dumps({"plan_sha256": plan["plan_sha256"], "completed": len(manifests),
                       "tokenizers": [manifests[i] for i in sorted(manifests)]}, indent=2) + "\n"


def verify_manifests(plan: dict, tokenizers_dir: Path, *, require_production_identity: bool = True):
    """Check every planned index.  Returns ``(manifests, problems, summary)``.

    ``manifests`` maps index -> manifest for every index that verifies on its own;
    ``problems`` maps index -> list of reasons (missing, invalid, or not matching the
    majority identity).  ``summary`` counts identities, families and totals.
    """
    manifests, problems, identities = {}, {}, {}
    for config in plan['configs']:
        index = config['index']
        folder = Path(tokenizers_dir) / index_name(index)
        path = folder / 'manifest.json'
        if not path.exists():
            reason = NOT_YET
            if (folder / 'failure.json').exists():
                try:
                    failure = read_json(folder / 'failure.json')
                    reason += f' (failed {failure.get("attempts", 1)}x: {str(failure.get("error"))[:200]})'
                except (OSError, ValueError, AttributeError):
                    reason += ' (unreadable failure.json)'
            problems[index] = [reason]
            continue
        try:
            manifest = read_json(path)
        except (OSError, ValueError) as exc:
            problems[index] = [f'unreadable manifest: {exc}']
            continue
        found = manifest_problems(manifest, config, plan, folder / 'tokenizer.json', check_hash=True)
        if not found:
            found = tokenizer_json_problems((folder / 'tokenizer.json').read_bytes(), config, config['vocab_size'])
        if not found:
            provenance = manifest['provenance']
            if provenance.get('plan_sha256') != plan['plan_sha256']:
                found.append('provenance plan_sha256 differs from the plan')
            if require_production_identity and (provenance.get('production_plan') is not True
                                                or not provenance.get('runtime_wheel_sha256')):
                found.append('not generated by the production worker with a verified wheel')
        if found:
            problems[index] = found
            continue
        manifests[index] = manifest
        key = json.dumps(identity_of(manifest['provenance']), sort_keys=True)
        identities.setdefault(key, []).append(index)
    if len(identities) > 1:
        majority = max(identities, key=lambda k: len(identities[k]))
        reference = json.loads(majority)
        for key, indices in identities.items():
            if key == majority:
                continue
            other = json.loads(key)
            changed = sorted(k for k in reference if reference[k] != other.get(k))
            for index in indices:
                problems[index] = [f'provenance identity differs from the other {len(identities[majority])} '
                                   f'tokenizers in: {", ".join(changed)}']
                manifests.pop(index, None)
    summary = {
        'planned': len(plan['configs']),
        'verified': len(manifests),
        'problems': len(problems),
        'identities': len(identities),
        'families': dict(sorted(Counter(f"{plan['configs'][i]['algorithm']}/{plan['configs'][i]['pretokenizer']}"
                                         for i in manifests).items())),
        'total_tokenizer_bytes': sum(m['tokenizer_bytes'] for m in manifests.values()),
    }
    return manifests, problems, summary


def _load_module(name: str, path: Path, expected_sha256: str, label: str):
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise PoolError(f'{path} has SHA-256 {actual}, expected the pinned {label} {expected_sha256}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_pool_sampling(code_dir: Path):
    """The pinned ``TokenizerPool`` class from ``<code_dir>/hf_hypernet`` (hash-checked)."""
    code_dir = Path(code_dir)
    target = code_dir / 'hf_hypernet' / 'pool_sampling.py'
    if file_sha256(target) != POOL_SAMPLING_SHA256:
        raise PoolError(f'{target} is not the pinned pool_sampling.py ({POOL_SAMPLING_SHA256})')
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    loaded = sys.modules.get('hf_hypernet')
    if loaded is not None and Path(getattr(loaded, '__file__', '') or '').parent != code_dir / 'hf_hypernet':
        raise PoolError(f'hf_hypernet was already imported from {loaded.__file__}, not {code_dir}')
    from hf_hypernet.pool_sampling import TokenizerPool
    return TokenizerPool


def prove_consumers(pool_dir: Path, code_dir: Path, policy_path: Path, *, production: bool) -> dict:
    """Load the pool exactly as both trainers do; raise PoolError on any problem."""
    TokenizerPool = import_pool_sampling(code_dir)
    try:
        pool = TokenizerPool(pool_dir)
    except (ValueError, KeyError, OSError) as exc:
        raise PoolError(f'pinned TokenizerPool rejects {pool_dir}: {exc}') from exc
    vocab_total = 0
    for index in range(len(pool.entries)):
        try:
            tokenizer, vocab = pool.load(index)
        except (ValueError, OSError) as exc:
            raise PoolError(f'pinned TokenizerPool.load({index}) failed: {exc}') from exc
        vocab_total += len(vocab)
    policy = _load_module('hyak_pool_settings_policy', Path(policy_path), SETTINGS_POLICY_SHA256, 'settings_policy.py')
    configs = [entry['config'] for entry in pool.entries]
    features = policy.settings_features(configs)
    bandit = policy.SettingsTokenizerBandit(configs)
    feature_sha = bandit.settings['feature_sha256']
    first = bandit.sample()
    result = {'index_sha256': pool.index_hash, 'plan_sha256': pool.plan_hash, 'loaded': len(pool.entries),
              'vocabulary_rows': vocab_total, 'feature_shape': list(features.shape), 'feature_sha256': feature_sha,
              'first_policy_action': first}
    if tuple(features.shape) != FEATURE_SHAPE:
        raise PoolError(f'settings_features shape {features.shape} != {FEATURE_SHAPE}')
    if production and feature_sha != FEATURE_SHA256:
        raise PoolError(f'policy feature digest {feature_sha} != the original run\'s {FEATURE_SHA256}')
    result['feature_sha256_matches_original'] = feature_sha == FEATURE_SHA256
    result['first_policy_action_matches_original'] = first == FIRST_POLICY_ACTION
    if first != FIRST_POLICY_ACTION:
        print(f'WARNING: SettingsTokenizerBandit(seed=43).sample() gave {first}, the original RL run started '
              f'with {FIRST_POLICY_ACTION} (numpy version?)', file=sys.stderr, flush=True)
    return result


def _proof_dir(pool_dir: Path, candidate_text: str) -> Path:
    """A directory that looks like the pool to TokenizerPool but holds the candidate index."""
    folder = Path(tempfile.mkdtemp(prefix='.collect-proof-', dir=str(pool_dir)))
    (folder / 'full-results.json').write_text(candidate_text)
    os.symlink(pool_dir / 'plan.json', folder / 'plan.json')
    os.symlink(pool_dir / 'tokenizers', folder / 'tokenizers')
    return folder


def _publish_once(target: Path, text: str) -> bool:
    """Create ``target`` with ``text`` atomically; never replace an existing file.

    Returns True when this call created it, False when an identical file already existed
    (another job collected the same manifests at the same moment).  A different existing
    file raises PoolError.
    """
    data = text.encode()

    def existing_matches():
        if target.read_bytes() != data:
            raise PoolError(f'{target} appeared while collecting with different content; it was not replaced')
        return False

    fd, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=f'.{target.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)     # atomic create-if-absent
        except FileExistsError:
            return existing_matches()
        except OSError:                    # file system without hard links
            if target.exists():
                return existing_matches()
            os.rename(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    fsync_dir(target.parent)
    return True


def run(args) -> int:
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    plan_path = Path(args.plan) if args.plan else paths.repo / PLAN_REL
    plan, plan_info = load_plan(plan_path, allow_test_plan=args.allow_test_plan)
    production = plan_info['production']
    if not production and not args.pool_dir:
        raise PoolError('A test plan needs an explicit --pool-dir')
    if not production and os.path.normpath(args.pool_dir) == os.path.normpath(str(paths.pool_dir)):
        raise PoolError('A test plan must not be collected into the production pool directory')
    pool_dir = Path(args.pool_dir) if args.pool_dir else paths.pool_dir
    tokenizers_dir = pool_dir / 'tokenizers'
    if not args.check_only:
        _cleanup_proofs(pool_dir)
    manifests, problems, summary = verify_manifests(plan, tokenizers_dir, require_production_identity=production)
    report = {'at_utc': utc_now(), 'pool_dir': str(pool_dir), 'plan_sha256': plan['plan_sha256'],
              'production_plan': production, **summary,
              'problem_indices': {str(i): p for i, p in sorted(problems.items())[:200]}}
    if manifests:
        report['identity'] = identity_of(next(iter(manifests.values()))['provenance'])
    print(json.dumps({k: v for k, v in report.items() if k != 'problem_indices'}, indent=2, sort_keys=True))
    for index, reasons in sorted(problems.items())[:50]:
        print(f'  {index_name(index)}: {"; ".join(reasons)}', file=sys.stderr)
    if len(problems) > 50:
        print(f'  ... and {len(problems) - 50} more', file=sys.stderr)
    monitor = Path(args.monitor_dir) if args.monitor_dir else paths.monitor_dir(STAGE_POOL)
    if args.check_only:
        return EXIT_OK if not problems else (EXIT_NOT_NOW if _incomplete_only(problems) else EXIT_INVALID)
    claims = pool_dir / 'claims'
    active = [p.name for p in claims.iterdir() if p.is_dir() and p.name.isdigit()] if claims.is_dir() else []
    if problems or len(manifests) != len(plan['configs']):
        print(f'ERROR: only {len(manifests)} of {len(plan["configs"])} tokenizers verify; refusing to write '
              f'{pool_dir / "full-results.json"}. Finish or repair the pool (python -m hyak.pool.worker --report).',
              file=sys.stderr)
        return EXIT_NOT_NOW if _incomplete_only(problems) else EXIT_INVALID
    if len(manifests) != PLAN_COUNT:
        print(f'ERROR: the trainers require exactly {PLAN_COUNT} tokenizers; this plan has {len(manifests)}. '
              'Nothing written.', file=sys.stderr)
        return EXIT_INVALID
    if active:
        print(f'WARNING: {len(active)} claim directories exist ({active[:5]}); workers may still be running, '
              'but every planned tokenizer already verifies.', file=sys.stderr)
    # plan.json: byte copy of the plan (scripts/modal_generate_tokenizer_pool.py:201).
    plan_target = pool_dir / 'plan.json'
    plan_bytes = plan_path.read_bytes()
    if plan_target.exists():
        if plan_target.read_bytes() != plan_bytes:
            raise PoolError(f'{plan_target} exists and differs from {plan_path}; it must be a byte copy. '
                            'Move it away only if no run has started from this pool.')
    else:
        atomic_copy(plan_path, plan_target)
    text = index_text(plan, manifests)
    target = pool_dir / 'full-results.json'
    code_dir = Path(args.code_dir) if args.code_dir else _default_code_dir(paths, 'uniform', 'runtime/anytok_pool_20k')
    rl_dir = Path(args.rl_code_dir) if args.rl_code_dir else _default_code_dir(paths, 'rl', 'runtime/anytok_pool_settings_rl_20k')
    policy_path = rl_dir / 'hf_hypernet' / 'settings_policy.py'
    if target.exists():
        existing = target.read_bytes()
        existing_sha = hashlib.sha256(existing).hexdigest()
        print(f'{target} already exists (SHA-256 {existing_sha}); it is never rewritten.')
        if existing != text.encode():
            print('ERROR: the existing full-results.json differs from what the current manifests give '
                  '(a manifest changed after the index was written?). Investigate before training; '
                  'both runs record the index hash.', file=sys.stderr)
            return EXIT_INVALID
        proof = prove_consumers(pool_dir, code_dir, policy_path, production=production)
        report.update(index_sha256=existing_sha, written=False, consumers=proof)
        atomic_write_json(monitor / 'collect-report.json', report)
        print(json.dumps({'full_results_sha256': existing_sha, 'consumers': proof}, indent=2))
        return EXIT_OK
    proof_folder = _proof_dir(pool_dir, text)
    try:
        proof = prove_consumers(proof_folder, code_dir, policy_path, production=production)
    finally:
        shutil.rmtree(proof_folder, ignore_errors=True)
    created = _publish_once(target, text)
    written_sha = file_sha256(target)
    if written_sha != proof['index_sha256']:
        raise PoolError(f'{target} hash {written_sha} differs from the proven candidate {proof["index_sha256"]}')
    if not created:
        print(f'{target} was written by another job at the same time with identical content.')
    report.update(index_sha256=written_sha, written=created, consumers=proof)
    atomic_write_json(monitor / 'collect-report.json', report)
    try:
        from hyak.common.status import StatusWriter
        StatusWriter(STAGE_POOL, len(plan['configs']), 'tokenizers', monitor_dir=monitor, cfg=cfg).event(
            {'event': 'index_written', 'full_results_sha256': written_sha, 'plan_sha256': plan['plan_sha256']})
    except (ValueError, OSError):
        pass
    print(json.dumps({'full_results': str(target), 'full_results_sha256': written_sha, 'consumers': proof}, indent=2))
    return EXIT_OK


def _default_code_dir(paths: Paths, run: str, repo_rel: str) -> Path:
    staged = paths.code_dir(run)
    return staged if (staged / 'hf_hypernet').is_dir() else paths.repo / repo_rel


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m hyak.pool.collect', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check-only', action='store_true', help='verify the manifests; write nothing')
    parser.add_argument('--pool-dir', help='default $ANYTOK_ROOT/input/pool')
    parser.add_argument('--code-dir', help='code tree with the pinned hf_hypernet/pool_sampling.py '
                        '(default $ANYTOK_ROOT/code/uniform if staged, else runtime/anytok_pool_20k)')
    parser.add_argument('--rl-code-dir', help='code tree with hf_hypernet/settings_policy.py '
                        '(default $ANYTOK_ROOT/code/rl if staged, else runtime/anytok_pool_settings_rl_20k)')
    parser.add_argument('--monitor-dir', help='where collect-report.json goes (default the pool monitor directory)')
    parser.add_argument('--config', help='config file (default $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--plan', help=argparse.SUPPRESS)
    parser.add_argument('--allow-test-plan', action='store_true', help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with _Signals():
            return run(args)
    except Interrupted as exc:
        print(f'[hyak pool collect] stopped by {exc}: nothing was published; submit one more pool job (or rerun '
              'python -m hyak.pool.collect) to write the index', file=sys.stderr, flush=True)
        return EXIT_NOT_NOW
    except (PoolError, ConfigError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return EXIT_INPUT
    except Exception:  # noqa: BLE001 -- unexpected: full traceback, exit 1
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
