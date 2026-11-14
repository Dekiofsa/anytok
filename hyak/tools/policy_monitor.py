"""Policy snapshots of the RL run for the dashboard's learning panel (Slurm replacement of
``scripts/collect_policy_monitor.py``).

    python -m hyak.tools.policy_monitor [--once | --follow [--interval 60]]
                                        [--run-dir DIR] [--output FILE] [--plan FILE] [--config FILE]

The original ran its reader inside the live Modal container (``modal container exec``,
collect_policy_monitor.py:34-60, 101-142 at 3de64a5).  Here the checkpoints are on the shared filesystem, so
the same read happens locally: for every ``<run>/step-NNNNNN`` with a ``manifest.json`` at a
multiple of 500 (not 0), ``policy.pt`` is size- and hash-checked against the manifest, loaded with
``torch.load(..., weights_only=True)``, required to have ``updates == step`` and settings kind
``shared_settings_mlp``, and summarised with ``monitoring.policy_summary.summarize`` (unchanged).

Output (defaults: ``<repo>/modal_results/hyak_tillicum/train-rl/policy-monitor.json``, the
registry's ``learning.snapshots``): ``{"snapshots": [row, ...], "error": str | null, "checked_at"}``,
each row as ``summarize`` returns it plus ``policy_sha256``, ``plan_sha256`` and ``profile``,
written atomically.  A step already in the file is re-read only when its ``policy.pt`` hash
changed (a step replayed after a hard kill); the original never re-read a known step.  Errors are
recorded in ``error`` and the previous snapshots are kept, as in the original.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
import time

from hyak.common import slurm
from hyak.common.io import atomic_write_text, read_json, sha256_file, utc_now

POLICY_KIND = 'shared_settings_mlp'
POOL_SIZE = 1000
_STEP_RE = re.compile(r'step-(\d{6})')


def _plan_configs(plan_path):
    plan = read_json(plan_path)
    configs = plan['configs']
    # collect_policy_monitor.py:42-43
    if len(configs) != POOL_SIZE or any(c['index'] != i for i, c in enumerate(configs)):
        raise ValueError(f'{plan_path} is not the {POOL_SIZE}-tokenizer plan')
    return plan, configs


def read_snapshots(run_dir, plan_path, known=None):
    """New snapshot rows from the checkpoints of ``run_dir`` (port of collect_policy_monitor.py:37-59).

    ``known`` maps step -> policy_sha256 of rows already collected; those are skipped unless the
    hash changed.
    """
    import torch
    from monitoring.policy_summary import summarize

    torch.set_num_threads(1)
    known = known or {}
    plan, configs = _plan_configs(plan_path)
    rows = []
    for folder in sorted(Path(run_dir).glob('step-*')):
        match = _STEP_RE.fullmatch(folder.name)
        if not match or not folder.is_dir():
            continue
        step = int(match.group(1))
        if step == 0 or step % 500 or not (folder / 'manifest.json').is_file():
            continue
        manifest = read_json(folder / 'manifest.json')
        path = folder / 'policy.pt'
        expected = manifest['files']['policy.pt']
        if known.get(step) == expected['sha256']:
            continue
        if manifest['step'] != step or path.stat().st_size != expected['bytes']:
            raise ValueError(f'{folder}: manifest step or policy.pt size does not match')
        if sha256_file(path) != expected['sha256']:
            raise ValueError(f'{path}: SHA-256 differs from its manifest')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['updates'] != step or state['settings']['kind'] != POLICY_KIND:
            raise ValueError(f'{path}: policy updates {state["updates"]} / kind {state["settings"].get("kind")} '
                             f'do not match step {step}')
        row = summarize(configs, state['logits'].softmax(0).tolist(), state['counts'].tolist(), step)
        row.update(policy_sha256=expected['sha256'], plan_sha256=plan['plan_sha256'])
        rows.append(row)
    return rows


def collect(run_dir, output, plan_path, *, expected_plan_sha256=None, profile=None) -> dict:
    """Merge new snapshots into ``output`` (port of collect_policy_monitor.py:101-142)."""
    output = Path(output)
    try:
        saved = read_json(output)
        if not isinstance(saved, dict) or not isinstance(saved.get('snapshots'), list):
            raise ValueError('not a snapshot file')
    except (OSError, ValueError):
        saved = {'snapshots': []}
    try:
        known = {row['step']: row.get('policy_sha256') for row in saved['snapshots']}
        rows = read_snapshots(run_dir, plan_path, known)
        merged = {row['step']: row for row in saved['snapshots']}
        for row in rows:
            if expected_plan_sha256 is not None and row['plan_sha256'] != expected_plan_sha256:
                raise ValueError('Policy snapshot pool differs from registered pool')
            merged[row['step']] = {**row, 'profile': profile or slurm.profile_label()}
        saved.update(snapshots=[merged[k] for k in sorted(merged)], error=None)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        saved['error'] = str(exc)[-900:]
    saved['checked_at'] = utc_now()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(saved, allow_nan=False) + '\n')
    print(json.dumps({'at_utc': saved['checked_at'], 'stage': 'policy_metadata',
                      'snapshots': len(saved['snapshots']), 'error': saved.get('error')}), flush=True)
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tools.policy_monitor', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='one read (default)')
    mode.add_argument('--follow', action='store_true', help='read every --interval seconds')
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--run-dir', type=Path, help='default: $ANYTOK_RUNS/rl')
    parser.add_argument('--output', type=Path, help='default: <repo>/modal_results/hyak_tillicum/train-rl/policy-monitor.json')
    parser.add_argument('--plan', type=Path, help='default: $ANYTOK_ROOT/input/pool/plan.json')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    try:
        from hyak.common.config import ConfigError, load_config, repo_root
        from hyak.common.paths import RUN_RL, STAGE_TRAIN_RL, Paths, default_monitor_dir
        if args.run_dir is None or args.plan is None:
            paths = Paths.from_config(load_config(args.config))
            run_dir = args.run_dir or paths.run_dir(RUN_RL)
            plan = args.plan or paths.pool_plan
        else:
            run_dir, plan = args.run_dir, args.plan
        output = args.output or default_monitor_dir(STAGE_TRAIN_RL) / 'policy-monitor.json'
        expected = read_json(repo_root() / 'plans/tokenizer-pool-1000-v1.json')['plan_sha256']
    except (ConfigError, OSError, ValueError, KeyError) as exc:  # unexpected errors keep their traceback (exit 1)
        print(f'ERROR: {exc}. Remedy: check --config / HYAK_CONFIG and the plan file '
              f'(exit 2 = usage, configuration or input error)', file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = _take_lock(output.with_name(output.name + '.lock'))
    if lock is False:
        print(json.dumps({'at_utc': utc_now(), 'stage': 'policy_metadata',
                          'skipped': 'another collector holds the lock'}), flush=True)
        return 0
    try:
        while True:
            collect(run_dir, output, plan, expected_plan_sha256=expected)
            if not args.follow:
                return 0
            time.sleep(args.interval)
    finally:
        if lock is not None:
            lock.close()


def _take_lock(path):
    """Advisory single-collector lock: the open file (held), None when locking is unavailable, False when
    another collector holds it.

    Any OSError other than "held" (ENOLCK, ENOTSUP, EACCES, a read-only directory, ...) means "no lock, carry
    on": the collector is idempotent and writes its output atomically, so two of them only repeat work.  A
    warning line (with ``at_utc``, so the dashboard's log pane shows it) says that it happened.
    """
    try:
        lock = open(path, 'a')
    except OSError as exc:
        _warn_no_lock(path, exc)
        return None
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return False
    except OSError as exc:
        lock.close()
        _warn_no_lock(path, exc)
        return None
    return lock


def _warn_no_lock(path, exc):
    print(json.dumps({'at_utc': utc_now(), 'stage': 'policy_metadata',
                      'warning': f'no lock on {path} ({exc}); continuing without it'}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
