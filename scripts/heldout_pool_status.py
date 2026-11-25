"""Read-only status of the raw-BPB held-out full-pool eval on Tillicum, printed as one JSON line.

run_rawbpb_heldout_pool.py streams this file to a cluster Python on stdin:
  python3 -I - --stage /gpfs/projects/stf/$USER/anytok-heldout
It reads the setup logs and markers, the smoke receipt, the score folders and Slurm's
records of the eval's jobs (job name anytok-heldout-4992). It changes nothing.
Standard library only; Python 3.9 compatible.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

JOB_NAME = 'anytok-heldout-4992'
PHASES = ('inputs', 'panel', 'checkpoints', 'preflight')
MARKERS = ('environment-ready', 'smoke-checkpoint-staged', 'features-ready', 'checkpoints-staged',
           'preflight-passed')
SCORE_FILE = re.compile(r'\d{4}\.json')


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def setup_log(path):
    """The last step header and the exit=N line the tmux wrapper appends, if any."""
    try:
        lines = Path(path).read_text(errors='replace').splitlines()
    except OSError:
        return None
    steps = [line[4:] for line in lines if line.startswith('=== ')]
    exits = [line[5:].strip() for line in lines if line.startswith('exit=')]
    code = None
    if exits:
        code = int(exits[-1]) if exits[-1].lstrip('-').isdigit() else exits[-1]
    return dict(exit=code, step=steps[-1] if steps else None, tail=lines[-3:])


def command(argv):
    try:
        done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    if done.returncode != 0:
        return None, done.stderr.strip()[:300]
    return done.stdout, None


def slurm(since):
    """Every job of the eval since the given date: sacct for history, squeue for pending reasons."""
    user = os.environ.get('USER', '')
    jobs, errors = {}, []
    out, error = command(['sacct', '-u', user, '--name', JOB_NAME, '-S', since, '-X', '-P', '-n',
                          '--format=JobID,State,ExitCode,Elapsed,Start,End'])
    if error:
        errors.append('sacct: ' + error)
    for line in (out or '').splitlines():
        parts = line.split('|')
        if len(parts) >= 6:
            jobs[parts[0]] = dict(id=parts[0], state=parts[1], exit_code=parts[2], elapsed=parts[3],
                                  start=parts[4], end=parts[5])
    out, error = command(['squeue', '-u', user, '-n', JOB_NAME, '-r', '-h', '-o', '%i|%T|%r|%M'])
    if error:
        errors.append('squeue: ' + error)
    for line in (out or '').splitlines():
        parts = line.split('|')
        if len(parts) >= 4:
            entry = jobs.setdefault(parts[0], dict(id=parts[0]))
            entry.update(state=parts[1], reason=parts[2], elapsed=parts[3])
    # sacct lists pending array tasks as one range (123_[1-41]); squeue -r lists them one by one.
    ranges = [key for key in jobs if '[' in key]
    for key in ranges:
        base = key.split('_')[0]
        if any(other.startswith(base + '_') and '[' not in other for other in jobs):
            jobs.pop(key)
    return sorted(jobs.values(), key=lambda job: job['id']), errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--since', default='2026-11-24')
    args = parser.parse_args()
    stage = Path(os.path.expandvars(args.stage))
    work = stage / 'raw-bpb'
    if not stage.is_dir():
        print(json.dumps(dict(stage_exists=False)))
        return
    labels = []
    try:
        labels = [line for line in (work / 'checkpoints.txt').read_text().splitlines() if line]
    except OSError:
        pass
    checkpoints = {}
    for label in labels:
        folder = work / 'run/heldout-h200' / label
        scored = 0
        if folder.is_dir():
            scored = sum(1 for name in os.listdir(str(folder)) if SCORE_FILE.fullmatch(name))
        checkpoints[label] = dict(scored=scored, complete=(folder / 'complete.json').is_file(),
                                  ranking=(folder / 'ranking.csv').is_file())
    failures = []
    status_dir = work / 'run/status'
    if status_dir.is_dir():
        for path in sorted(status_dir.glob('*.json')):
            row = read_json(path) or {}
            if row.get('stage') == 'failed':
                failures.append(dict(worker=path.stem, error=str(row.get('error'))[:300],
                                     checkpoint=row.get('checkpoint'), at_utc=row.get('at_utc')))
    jobs, errors = slurm(args.since)
    print(json.dumps(dict(
        stage_exists=True,
        setup={phase: setup_log(stage / 'logs' / ('setup-%s.log' % phase)) for phase in PHASES},
        markers={name: (stage / 'logs' / name).is_file() for name in MARKERS},
        smoke=read_json(work / 'run/hardware-check.json'),
        preflight=read_json(stage / 'logs/preflight.json'),
        checkpoints=checkpoints, worker_failures=failures, jobs=jobs, slurm_errors=errors)))


if __name__ == '__main__':
    main()
