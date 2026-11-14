"""What the raw-BPB job scripts ask before and after Qingyun's launcher (scripts/run_raw_bpb_training.py).

    python -m hyak.handoff.run prepare  --output DIR [--total N] [--smoke]
    python -m hyak.handoff.run verify   --output DIR [--smoke] [--expect-step N] [--since-line K]
                                        [--before-step N] [--stop-requested]
    python -m hyak.handoff.run requirements [--requirements FILE]
    python -m hyak.handoff.run outputs  [--config FILE]
    python -m hyak.handoff.run count-lines --output DIR
    python -m hyak.handoff.run mark     --stage STAGE --output DIR [--complete]

prepare (before a launch; prints one JSON line with "action"):
  * no output directory, or an empty one: ``fresh`` (exit 0).  With ``--smoke`` the output must be in that
    state (exit 2 otherwise, with the ``rm -rf`` remedy: a smoke always starts fresh).
  * a resumable full restart (the newest ``step-NNNNNN`` with ``state.pt``, structurally complete) and the
    run is below ``--total`` (20,000): ``resume`` (exit 0).  First, under the trainer's own ``.run.lock``,
    weights-only checkpoints NEWER than that restart and stale ``.writing-*`` directories (left by a hard kill;
    the trainer would otherwise stop with FileExistsError when it reaches that step again) are moved, never
    deleted, to ``<output>/interrupted/<UTC>/`` and listed in "moved".
  * the newest full restart is at ``--total`` or beyond: ``complete`` (exit 75: not now, nothing to do).
  * the trainer's lock is held by another process: exit 75 (``run_lock_held``).
  * trainer files but no resumable restart (partial state of a start that died before its first save):
    exit 2 with a remedy (move the directory away); nothing is deleted.
  * the newest restart is incomplete, or run_config.json is not the raw-BPB protocol: exit 3 (your decision).

verify (after the launcher returned 0; exit 0 ok, 4 a check failed, 2 files missing):
  the checks of scripts/verify_raw_bpb_handoff.py on this output: run_config.json (reward_objective raw_bpb,
  policy_reward paired_pre_model_update_transferred_bpb, memory_execution optimized, entropy weight 0, policy
  learning rate 0.001, baseline decay 0.99, policy seed 43, pool size 1000, micro-batch 32); every ``train`` row
  of events.jsonl after line ``--since-line``: policy_reward_bpb == train_bpb, |paired_gap_bpb - (train_bpb -
  paired_canonical_bpb)| < 1e-12, paired_valid_examples == 128, paired_maximum_positions <= 128, policy_updates
  == step, reward_objective raw_bpb; the newest full restart's state.pt: model step == policy updates == policy
  selections, and its manifest.  With ``--smoke``: train rows exactly 1..N, status.json "complete" at
  ``--expect-step`` N, and the counters of the restarts at 2 and (N = 3) 3.  Without ``--smoke``: the newest
  full restart must be newer than ``--before-step`` (checkpoint progress), unless ``--stop-requested``.

requirements: the installed packages against docs/raw-bpb-handoff/requirements.txt (exit 0, or 2 with the list).
outputs: prints ``train=<dir>`` and ``smoke=<dir>`` as configured.  count-lines: lines in <output>/events.jsonl.
mark: copies the launcher's step (<output>/status.json) into the stage's own status.json in
modal_results/hyak_tillicum/<stage>/ (``--complete``: state complete), so that python -m hyak.common.status show
and the job's end record carry the step.
Exit codes follow the port's contract (hyak/slurm/job.sh): 0, 1 crash, 2, 3, 4, 75.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time

from hyak.handoff import common as C

EXIT_OK, EXIT_USAGE, EXIT_DECISION, EXIT_CHECK, EXIT_NOT_NOW = 0, 2, 3, 4, 75


def _emit(obj: dict) -> None:
    print(json.dumps(obj, sort_keys=True), flush=True)


def _say(message: str) -> None:
    print(f'[hyak.handoff.run] {message}', file=sys.stderr, flush=True)


def _utc_stamp() -> str:
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


# --- prepare ------------------------------------------------------------------------------------------
def config_problems(config: dict) -> list:
    """Differences of a run_config.json from the raw-BPB protocol."""
    problems = []
    for key, wanted in C.EXPECTED_CONFIG.items():
        if config.get(key) != wanted:
            problems.append(f'run_config.json {key} is {config.get(key)!r}, expected {wanted!r}')
    policy = config.get('policy') or {}
    for key, wanted in C.EXPECTED_POLICY.items():
        if policy.get(key) != wanted:
            problems.append(f'run_config.json policy.{key} is {policy.get(key)!r}, expected {wanted!r}')
    return problems


def _move_aside(output: Path, items: list) -> list:
    if not items:
        return []
    target = output / 'interrupted' / _utc_stamp()
    target.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in items:
        destination = target / path.name
        path.rename(destination)
        moved.append({'from': str(path), 'to': str(destination)})
    return moved


def prepare(output: Path, total: int = C.FINAL_STEP, smoke: bool = False) -> tuple:
    """``(exit_code, record)``; see the module docstring."""
    output = Path(output)
    if C.output_is_empty(output):
        return EXIT_OK, {'action': 'fresh', 'output': str(output), 'newest_full_step': None}
    if not output.is_dir():
        return EXIT_USAGE, {'action': 'refuse', 'reason': 'not_a_directory',
                            'message': f'{output} exists and is not a directory',
                            'remedy': f'move it away (mv {output} {output}.moved-{_utc_stamp()}) or configure another '
                                      'output directory'}
    if smoke:
        return EXIT_USAGE, {'action': 'refuse', 'reason': 'smoke_output_exists',
                            'message': f'the smoke output {output} is not empty: the smoke must start fresh',
                            'remedy': f'remove the earlier smoke (about 70 GB of checkpoints): rm -rf {output}  '
                                      '(it is not the training run; check the path first)'}
    lock_path = output / '.run.lock'
    lock = open(lock_path, 'a+')
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return EXIT_NOT_NOW, {'action': 'not_now', 'reason': 'run_lock_held',
                                  'message': f'another process holds {lock_path} (a trainer is running in this output)'}
        restarts = C.full_restarts(output)
        if not restarts:
            names = sorted(p.name for p in output.iterdir() if p.name != '.run.lock')
            return EXIT_USAGE, {
                'action': 'refuse', 'reason': 'partial_state',
                'message': f'{output} holds trainer files ({", ".join(names[:8])}{" ..." if len(names) > 8 else ""}) but no '
                           'resumable full restart (a step-NNNNNN directory with state.pt): an earlier start ended before '
                           'its first save. A fresh start needs an empty output and a resume needs a restart.',
                'remedy': f'inspect the logs, then move the directory away and resubmit for a fresh start: '
                          f'mv {output} {output}.partial-{_utc_stamp()}  (nothing was deleted)'}
        newest_step, newest = restarts[-1]
        problems = C.restart_problems(newest_step, newest)
        if problems:
            older = [s for s, _ in restarts[:-1] if not C.restart_problems(s, dict(restarts)[s])]
            fallback = (f'the next job then resumes from step {older[-1]}' if older else
                        'no older full restart is usable: the run would start again from scratch')
            return EXIT_DECISION, {
                'action': 'refuse', 'reason': 'newest_restart_incomplete', 'newest_full_step': newest_step,
                'problems': problems[:10],
                'message': f'the newest full restart {newest} is incomplete: ' + '; '.join(problems[:3]),
                'remedy': f'move it aside yourself: mkdir -p {output}/interrupted/manual && mv {newest} '
                          f'{output}/interrupted/manual/  ({fallback})'}
        config_path = output / 'run_config.json'
        try:
            config = json.loads(config_path.read_text())
        except (OSError, ValueError) as exc:
            return EXIT_DECISION, {'action': 'refuse', 'reason': 'run_config_unreadable',
                                   'message': f'{config_path}: {exc}',
                                   'remedy': 'this output does not look like a raw-BPB run; configure another '
                                             'HYAK_RAWBPB_OUTPUT or inspect it'}
        problems = config_problems(config)
        if problems:
            return EXIT_DECISION, {'action': 'refuse', 'reason': 'not_the_raw_bpb_protocol', 'problems': problems,
                                   'message': f'{output} is not a raw-BPB entropy-0 run: ' + '; '.join(problems[:3]),
                                   'remedy': 'do not resume it with this stage; configure another HYAK_RAWBPB_OUTPUT'}
        if newest_step >= total:
            return EXIT_NOT_NOW, {'action': 'complete', 'reason': 'complete', 'newest_full_step': newest_step,
                                  'message': f'the run is complete: full restart at step {newest_step} >= {total}'}
        later = [path for step, path in C.step_dirs(output) if step > newest_step]
        stale = sorted(p for p in output.iterdir() if p.name.startswith('.writing-') and p.is_dir())
        moved = _move_aside(output, later + stale)
        for item in moved:
            _say(f'moved {item["from"]} -> {item["to"]} (newer than the restart at {newest_step}, or a stale '
                 'staging directory of a killed save; delete it once the run has passed that step)')
        return EXIT_OK, {'action': 'resume', 'output': str(output), 'newest_full_step': newest_step, 'moved': moved}
    finally:
        lock.close()


# --- verify -------------------------------------------------------------------------------------------
def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def row_problems(row: dict) -> list:
    """The per-update checks of scripts/verify_raw_bpb_handoff.py."""
    step = row.get('step')
    where = f'train row of step {step}'
    problems = []
    for key in ('train_bpb', 'policy_reward_bpb', 'paired_gap_bpb', 'paired_canonical_bpb'):
        if not _number(row.get(key)):
            problems.append(f'{where}: {key} is {row.get(key)!r}')
    if problems:
        return problems
    if row['policy_reward_bpb'] != row['train_bpb']:
        problems.append(f'{where}: policy_reward_bpb {row["policy_reward_bpb"]!r} != train_bpb {row["train_bpb"]!r} '
                        '(the policy did not receive the raw BPB)')
    gap = row['train_bpb'] - row['paired_canonical_bpb']
    if abs(row['paired_gap_bpb'] - gap) >= C.GAP_TOLERANCE:
        problems.append(f'{where}: paired_gap_bpb {row["paired_gap_bpb"]!r} != train_bpb - paired_canonical_bpb {gap!r}')
    if row.get('paired_valid_examples') != C.PAIRED_EXAMPLES:
        problems.append(f'{where}: paired_valid_examples {row.get("paired_valid_examples")!r} != {C.PAIRED_EXAMPLES}')
    positions = row.get('paired_maximum_positions')
    if not isinstance(positions, int) or positions > C.MAX_POSITIONS:
        problems.append(f'{where}: paired_maximum_positions {positions!r} > {C.MAX_POSITIONS}')
    if row.get('policy_updates') != step:
        problems.append(f'{where}: policy_updates {row.get("policy_updates")!r} != step')
    if row.get('reward_objective') != 'raw_bpb':
        problems.append(f'{where}: reward_objective {row.get("reward_objective")!r} != raw_bpb')
    return problems


def read_events(output: Path, since_line: int = 0) -> list:
    rows = []
    path = Path(output) / 'events.jsonl'
    for number, line in enumerate(path.read_text().splitlines()):
        if number < since_line or not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def restart_counters(step: int, path: Path) -> tuple:
    problems = C.restart_problems(step, path)
    if problems:
        return None, problems
    state = C.load_state(path / 'state.pt')
    found = C.counters(state)
    problems = [f'{path}/state.pt: {name} counter {value} != step {step}' for name, value in found.items()
                if value != step]
    return found, problems


def verify(output: Path, *, smoke=False, expect_step=None, since_line=0, before_step=None,
           stop_requested=False) -> tuple:
    output = Path(output)
    problems, record = [], {'output': str(output)}
    for name in ('run_config.json', 'events.jsonl'):
        if not (output / name).is_file():
            return EXIT_USAGE, {**record, 'ok': False, 'problems': [f'{output / name} is missing']}
    config = json.loads((output / 'run_config.json').read_text())
    problems += config_problems(config)
    rows = read_events(output, since_line)
    train = [r for r in rows if isinstance(r, dict) and r.get('stage') == 'train']
    for row in train:
        problems += row_problems(row)
    record['train_rows'] = len(train)
    record['steps'] = [train[0]['step'], train[-1]['step']] if train else None
    restarts = C.full_restarts(output)
    record['newest_full_step'] = restarts[-1][0] if restarts else None
    if smoke:
        if expect_step is None:
            raise ValueError('--smoke needs --expect-step')
        steps = [r.get('step') for r in train]
        if steps != list(range(1, expect_step + 1)):
            problems.append(f'train rows are for steps {steps}, expected 1..{expect_step}')
        try:
            status = json.loads((output / 'status.json').read_text())
        except (OSError, ValueError) as exc:
            status = {}
            problems.append(f'{output}/status.json: {exc}')
        if status.get('stage') != 'complete' or status.get('step') != expect_step:
            problems.append(f'status.json says {status.get("stage")!r} at step {status.get("step")!r}, '
                            f'expected complete at {expect_step}')
        record['resume_counters'] = {}
        for step in [s for s in C.SMOKE_PHASES if s <= expect_step]:
            path = output / f'step-{step:06d}'
            found, issues = restart_counters(step, path) if (path / 'state.pt').is_file() else \
                (None, [f'{path}/state.pt is missing'])
            problems += issues
            record['resume_counters'][str(step)] = found
    else:
        if not restarts:
            problems.append('no full restart exists after the launch')
        else:
            step, path = restarts[-1]
            found, issues = restart_counters(step, path)
            problems += issues
            record['newest_counters'] = found
            if before_step is not None and step <= before_step:
                message = (f'no checkpoint progress: the newest full restart is still {step} '
                           f'(it was {before_step} before this launch)')
                if stop_requested:
                    record['warning'] = message + ' -- a stop was requested, so this is not a failure'
                    _say('WARNING: ' + record['warning'])
                else:
                    problems.append(message)
    record['ok'] = not problems
    record['problems'] = problems[:50]
    return (EXIT_OK if not problems else EXIT_CHECK), record


def mark(stage: str, output: Path, complete: bool) -> int:
    from hyak.common.paths import STAGE_PROGRESS
    from hyak.common.status import StatusWriter
    try:
        launcher = json.loads((Path(output) / 'status.json').read_text())
    except (OSError, ValueError):
        launcher = {}
    restarts = C.full_restarts(output)
    step = launcher.get('step') if isinstance(launcher.get('step'), int) else (restarts[-1][0] if restarts else 0)
    progress = STAGE_PROGRESS[stage]
    writer = StatusWriter(stage, progress['total'], progress['unit'], progress['completed_field'])
    writer.update(step, stage='complete' if complete else 'running',
                  state='complete' if complete else 'running', workers=[] if complete else None,
                  launcher_stage=launcher.get('stage'), newest_full_step=restarts[-1][0] if restarts else None,
                  output=str(output), event='rawbpb_progress')
    return EXIT_OK


# --- main ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.handoff.run', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p_prep = sub.add_parser('prepare', help='decide fresh / resume / complete / refuse before a launch')
    p_prep.add_argument('--output', type=Path, required=True)
    p_prep.add_argument('--total', type=int, default=C.FINAL_STEP)
    p_prep.add_argument('--smoke', action='store_true', help='the smoke output must be empty')
    p_ver = sub.add_parser('verify', help='the receipt checks after a launch')
    p_ver.add_argument('--output', type=Path, required=True)
    p_ver.add_argument('--smoke', action='store_true')
    p_ver.add_argument('--expect-step', type=int)
    p_ver.add_argument('--since-line', type=int, default=0, help='skip this many lines of events.jsonl')
    p_ver.add_argument('--before-step', type=int, help='newest full restart before the launch')
    p_ver.add_argument('--stop-requested', action='store_true', help='no progress is then only a warning')
    p_req = sub.add_parser('requirements', help='installed packages against requirements.txt')
    p_req.add_argument('--requirements', type=Path)
    p_out = sub.add_parser('outputs', help='print the configured output directories')
    p_out.add_argument('--config', type=Path)
    p_mark = sub.add_parser('mark', help="copy the launcher's step into the stage status")
    p_mark.add_argument('--stage', required=True)
    p_mark.add_argument('--output', type=Path, required=True)
    p_mark.add_argument('--complete', action='store_true')
    p_cnt = sub.add_parser('count-lines', help='lines in <output>/events.jsonl (0 when absent)')
    p_cnt.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == 'prepare':
        code, record = prepare(args.output, args.total, args.smoke)
        _emit({'stage': 'rawbpb_prepare', **record})
        if code in (EXIT_USAGE, EXIT_DECISION):
            _say(f'ERROR: {record.get("message")}')
            _say(f'remedy: {record.get("remedy")}')
        return code
    if args.command == 'verify':
        code, record = verify(args.output, smoke=args.smoke, expect_step=args.expect_step, since_line=args.since_line,
                              before_step=args.before_step, stop_requested=args.stop_requested)
        _emit({'stage': 'rawbpb_verified' if record.get('ok') else 'rawbpb_verify_failed', **record})
        for problem in record.get('problems', []):
            _say(f'CHECK FAILED: {problem}')
        return code
    if args.command == 'requirements':
        from hyak.common.config import repo_root
        path = args.requirements or C.docs_dir(repo_root()) / C.REQUIREMENTS_NAME
        pins = C.parse_requirements(path)
        installed = C.installed_versions([name for name, _ in pins])
        problems = C.requirement_problems(pins, installed)
        _emit({'stage': 'rawbpb_requirements', 'requirements': str(path), 'installed': installed,
               'ok': not problems, 'problems': problems})
        for problem in problems:
            _say(f'ERROR: {problem}')
        if problems:
            _say('remedy: bash hyak/env/setup_train_env.sh (it installs exactly docs/raw-bpb-handoff/requirements.txt)')
            return EXIT_USAGE
        return EXIT_OK
    if args.command == 'outputs':
        from hyak.common.config import ConfigError, load_config
        try:
            train, smoke = C.outputs(load_config(args.config))
        except ConfigError as exc:
            _say(f'ERROR: {exc}')
            return EXIT_USAGE
        print(f'train={train}')
        print(f'smoke={smoke}')
        return EXIT_OK
    if args.command == 'mark':
        return mark(args.stage, args.output, args.complete)
    path = args.output / 'events.jsonl'
    print(sum(1 for _ in path.open('rb')) if path.is_file() else 0)
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
