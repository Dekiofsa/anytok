"""TEST-ONLY: drive production-mode segments of the REAL trainers (CPU shim) and inject stops and kills.

    python -m hyak.tests.e2e.marathon --world DIR --run {uniform,rl} --log-dir DIR --plan PLAN.json

Each plan item starts ``python -m hyak.train.run_segment --run R`` (exactly the production entry point; the
CPU shim and the explicit test parameters come from ``world.world_env``) and optionally acts on it:

* ``{"action": "sigterm_at", "step": N}``          SIGTERM once train step >= N is logged (graceful stop)
* ``{"action": "sigkill_mid", "step": N}``         SIGKILL ~``delay`` s after train step N is logged (mid-update)
* ``{"action": "sigkill_writing", "step": N}``     SIGKILL as soon as ``.writing-*`` holds files after step N
* ``{"action": "sigkill_after_replace", "step": N}`` SIGKILL once ``step-N`` exists without ``manifest.json``
  (the next segment adopts such a complete restart by itself and records ``adopted_restart``)
* ``{"max_seconds": S}``                           the wrapper's in-process stop timer
* ``{"cli": [...]}``                               run another production CLI (e.g. retention adopt)

After every item the run directory is inspected (``python -m hyak.train.retention summary`` and ``verify``,
every manifest re-verified, ``.writing-*`` / ``interrupted/`` listed) and one JSON line is appended to
``<log-dir>/marathon-<run>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from hyak.tests.e2e.world import REPO, world_env


class EventTail:
    """Incremental reader of the trainer's ``<run>/events.jsonl``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if self.path.is_file() else 0
        self.partial = b''
        self.last_train = None
        self.rows = []

    def poll(self):
        if not self.path.is_file():
            return []
        with open(self.path, 'rb') as stream:
            stream.seek(self.offset)
            data = stream.read()
        self.offset += len(data)
        lines = (self.partial + data).split(b'\n')
        self.partial = lines.pop()
        new = []
        for raw in lines:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            new.append(row)
            if row.get('stage') == 'train':
                self.last_train = row['step']
        self.rows += new
        return new


def inspect(world: Path, run: str, env: dict) -> dict:
    """Production CLIs (retention summary/verify) plus a manifest re-verification of every step directory."""
    run_dir = world / 'runs' / run
    out = {}
    for command in ('summary', 'verify'):
        result = subprocess.run([sys.executable, '-m', 'hyak.train.retention', command, '--run', run],
                                env=env, cwd=str(REPO), capture_output=True, text=True)
        try:
            out[command] = json.loads(result.stdout) if result.returncode == 0 else {'error': result.stderr.strip()}
        except ValueError:
            out[command] = {'raw': result.stdout[-2000:], 'stderr': result.stderr[-2000:]}
    sys.path.insert(0, str(REPO))
    from hyak.train import retention
    steps = {}
    for step, path in retention.step_dirs(run_dir):
        files = sorted(p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file())
        row = {'files': files}
        try:
            checked = retention.verify_checkpoint(run_dir, step, run)
            row.update(kind=checked['kind'], verified=True)
        except retention.CheckpointError as exc:
            row.update(verified=False, error=str(exc)[:300])
        steps[step] = row
    out['steps'] = steps
    out['writing'] = sorted(p.name for p in run_dir.glob('.writing-*')) if run_dir.is_dir() else []
    interrupted = run_dir / 'interrupted'
    out['interrupted'] = sorted(p.relative_to(interrupted).as_posix() for p in interrupted.glob('*/*')) \
        if interrupted.is_dir() else []
    return out


def run_item(world: Path, run: str, item: dict, log_dir: Path, index: int, extra_env: dict) -> dict:
    env = world_env(world, extra_env)
    name = f'{run}-{index:02d}-{item.get("name") or item.get("action") or ("timer" if item.get("max_seconds") else "cli")}'
    record = {'index': index, 'name': name, 'item': item, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if 'cli' in item:
        result = subprocess.run([sys.executable, '-m', *item['cli']], env=env, cwd=str(REPO),
                                capture_output=True, text=True)
        (log_dir / f'{name}.out').write_text(result.stdout)
        (log_dir / f'{name}.err').write_text(result.stderr)
        record.update(exit_code=result.returncode, stdout_tail=result.stdout[-1500:], stderr_tail=result.stderr[-1500:])
        record['inspect'] = inspect(world, run, env)
        return record
    command = [sys.executable, '-m', 'hyak.train.run_segment', '--run', run]
    if item.get('max_seconds'):
        command += ['--max-seconds', str(item['max_seconds'])]
    run_dir = world / 'runs' / run
    tail = EventTail(run_dir / 'events.jsonl')
    started = time.monotonic()
    with open(log_dir / f'{name}.out', 'wb') as out, open(log_dir / f'{name}.err', 'wb') as err:
        child = subprocess.Popen(command, env=env, cwd=str(REPO), stdout=out, stderr=err)
        action, target = item.get('action'), item.get('step')
        fired = None
        armed_at = None
        while child.poll() is None:
            tail.poll()
            if action and fired is None:
                step = tail.last_train
                if action == 'sigterm_at' and step is not None and step >= target:
                    child.send_signal(signal.SIGTERM)
                    fired = {'signal': 'SIGTERM', 'after_train_step': step}
                elif action == 'sigkill_mid' and step is not None and step >= target:
                    if armed_at is None:
                        armed_at = (time.monotonic(), step)
                    elif time.monotonic() - armed_at[0] >= item.get('delay', 1.0):
                        if tail.last_train == armed_at[1]:
                            child.kill()
                            fired = {'signal': 'SIGKILL', 'after_train_step': step, 'during_update': step + 1,
                                     'delay': item.get('delay', 1.0)}
                        else:
                            armed_at = (time.monotonic(), tail.last_train)
                elif action == 'sigkill_writing' and (target is None or (step is not None and step >= target)) \
                        and run_dir.is_dir():
                    for staging in run_dir.glob('.writing-*'):
                        contents = [p.name for p in staging.rglob('*')]
                        if contents:
                            child.kill()
                            fired = {'signal': 'SIGKILL', 'after_train_step': step, 'staging': staging.name,
                                     'staging_contents': sorted(contents)}
                            break
                elif action == 'sigkill_after_replace' and step is not None and step >= target:
                    folder = run_dir / f'step-{target:06d}'
                    if folder.is_dir() and not (folder / 'manifest.json').exists():
                        child.kill()
                        fired = {'signal': 'SIGKILL', 'after_train_step': step, 'folder': folder.name,
                                 'contents': sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*'))}
                    elif (folder / 'manifest.json').exists():
                        fired = {'missed': f'{folder.name}/manifest.json appeared before the kill'}
            time.sleep(0.002 if action in ('sigkill_writing', 'sigkill_after_replace') else 0.05)
        code = child.wait()
    tail.poll()
    record.update(exit_code=code, fired=fired, seconds=round(time.monotonic() - started, 1))
    result = None
    for line in (log_dir / f'{name}.out').read_text(errors='replace').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('stage') == 'segment_result':
            result = row
    record['segment_result'] = result
    train = [row for row in tail.rows if row.get('stage') == 'train']
    if train:
        record['train_steps'] = [train[0]['step'], train[-1]['step'], len(train)]
        seconds = sorted(row['seconds'] for row in train)
        record['median_update_seconds'] = seconds[len(seconds) // 2]
    record['trainer_last_events'] = [row.get('stage') + ':' + str(row.get('step')) for row in tail.rows[-4:]]
    record['stderr_tail'] = (log_dir / f'{name}.err').read_text(errors='replace')[-1200:]
    record['inspect'] = inspect(world, run, env)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tests.e2e.marathon', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--run', required=True, choices=('uniform', 'rl'))
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--threads', default='2', help='HYAK_E2E_TORCH_THREADS for the trainer')
    parser.add_argument('--start', type=int, default=0, help='first plan item to run')
    args = parser.parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    journal = args.log_dir / f'marathon-{args.run}.jsonl'
    for index, item in enumerate(plan):
        if index < args.start:
            continue
        record = run_item(args.world, args.run, item, args.log_dir, index, {'HYAK_E2E_TORCH_THREADS': args.threads})
        with open(journal, 'a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        print(json.dumps({k: record.get(k) for k in ('name', 'exit_code', 'fired', 'train_steps', 'seconds')}),
              flush=True)
        if item.get('stop_on_failure', True) and record['exit_code'] not in item.get('expect_exit', [0, -9]):
            print(f'stopping: unexpected exit code {record["exit_code"]}', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
