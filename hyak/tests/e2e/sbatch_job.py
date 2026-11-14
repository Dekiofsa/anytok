"""TEST-ONLY: run a REAL hyak/slurm/*.sbatch script under bash against an e2e world (stage G).

    python -m hyak.tests.e2e.sbatch_job --world DIR --script 07_train.sbatch --stage train-uniform \
        [--run uniform] [--job-id ID] [--term-after-train N | --term-after-start S] [--hours 3]

Stand-ins (as in hyak/tests/test_integration_sbatch.py, but with the REAL Python entry points):

* a fake ``module`` shell function (exported to the batch shell) that accepts ``module load conda``;
* a fake ``conda`` on PATH whose ``shell.bash hook`` defines ``conda activate PREFIX`` (CONDA_PREFIX, PATH);
* ``$ANYTOK_ROOT/env/train/bin/python``: execs the pinned venv's Python with the CPU shim directory put in
  front of the ``PYTHONPATH=$HYAK_REPO`` that ``hyak_activate`` sets;
* Slurm's environment: ``SLURM_JOB_ID`` and ``SLURM_JOB_END_TIME`` (now + ``--hours``); the job's
  stdout+stderr go to ``<monitor>/<stage>/slurm-<job>.out`` like ``--output``.

``--term-after-train N`` sends SIGTERM to the batch shell (Slurm ``--signal=B:TERM``) once the job's
train log holds N new ``train`` lines; ``--term-after-start S`` sends it S seconds after the job printed
its first ``start: python -m hyak.train.run_segment`` line (the trainer is then being imported).
``--term-after-log PATTERN --term-count N`` sends it once the job output holds N lines containing PATTERN
(evaluation stages, e.g. ``scored in``).  ``--env KEY=VALUE`` adds job environment (e.g. the
``--env`` values of submit.sh, or ``HYAK_E2E_EVAL_CHECKPOINT_<RUN>`` for the scorer); ``--cpus`` is
``SLURM_CPUS_PER_TASK``.
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

from hyak.tests.e2e.world import REPO, SHIM_DIR, world_env

FAKE_CONDA = ('#!/bin/sh\n[ "$1 $2" = "shell.bash hook" ] || exit 3\n'
              "cat <<'EOF'\nconda() { if [ \"$1\" = activate ]; then export CONDA_PREFIX=\"$2\"; "
              'export PATH="$2/bin:$PATH"; else command conda "$@"; fi; }\nEOF\n')
FAKE_MODULE = '() {  echo "[fake module] $*" >&2; case "$1" in load) return 0 ;; *) return 0 ;; esac; }'
PYTHON_WRAPPER = '''#!/bin/bash
# TEST-ONLY stand-in for the conda train environment: the pinned venv + the e2e CPU shim.
export PYTHONPATH="$E2E_SHIM_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$E2E_REAL_PYTHON" "$@"
'''


def prepare(world: Path) -> Path:
    fake = world / 'fakejob'
    (fake / 'bin').mkdir(parents=True, exist_ok=True)
    (fake / 'bin' / 'conda').write_text(FAKE_CONDA)
    (fake / 'bin' / 'conda').chmod(0o755)
    for which in ('train', 'data'):
        prefix = world / 'root' / 'env' / which
        (prefix / 'bin').mkdir(parents=True, exist_ok=True)
        (prefix / 'lib').mkdir(exist_ok=True)
        (prefix / 'bin' / 'python').write_text(PYTHON_WRAPPER)
        (prefix / 'bin' / 'python').chmod(0o755)
    return fake


def run_job(world: Path, script: str, stage: str, *, run=None, job_id='e2e-job', term_after_train=None,
            term_after_start=None, term_after_log=None, term_count=1, hours=3.0, threads='2', cpus='8',
            extra_env=None, log=print) -> dict:
    world = Path(world)
    fake = prepare(world)
    env = world_env(world, {'HYAK_E2E_TORCH_THREADS': threads})
    env.pop('PYTHONPATH', None)
    env.update({'HYAK_REPO': str(REPO), 'HYAK_STAGE': stage, 'SLURM_JOB_ID': job_id,
                'SLURM_JOB_END_TIME': str(int(time.time() + hours * 3600)), 'SLURM_CPUS_PER_TASK': str(cpus),
                'E2E_SHIM_DIR': str(SHIM_DIR), 'E2E_REAL_PYTHON': sys.executable,
                'PATH': f'{fake / "bin"}:{os.environ["PATH"]}', 'BASH_FUNC_module%%': FAKE_MODULE})
    if run:
        env['HYAK_RUN'] = run
    env.update(extra_env or {})
    monitor = world / 'monitor' / stage
    monitor.mkdir(parents=True, exist_ok=True)
    slurm_out = monitor / f'slurm-{job_id}.out'
    train_log = monitor / f'{job_id}-train.log'
    started = time.monotonic()
    fired = None
    with open(slurm_out, 'wb') as out:
        job = subprocess.Popen(['bash', str(REPO / 'hyak' / 'slurm' / script)], env=env, cwd=str(REPO),
                               stdout=out, stderr=subprocess.STDOUT)
        base = sum(1 for _ in open(train_log)) if train_log.is_file() else 0
        start_seen = None
        while job.poll() is None:
            if fired is None and term_after_train is not None and train_log.is_file():
                lines = train_log.read_text(errors='replace').splitlines()[base:]
                count = sum(1 for line in lines if '"stage": "train"' in line)
                if count >= term_after_train:
                    job.send_signal(signal.SIGTERM)
                    fired = {'signal': 'SIGTERM', 'to': 'batch shell', 'after_new_train_lines': count,
                             'at_seconds': round(time.monotonic() - started, 1)}
            if fired is None and term_after_log is not None:
                count = sum(term_after_log in line for line in slurm_out.read_text(errors='replace').splitlines())
                if count >= term_count:
                    job.send_signal(signal.SIGTERM)
                    fired = {'signal': 'SIGTERM', 'to': 'batch shell', 'after_log_lines': count,
                             'pattern': term_after_log, 'at_seconds': round(time.monotonic() - started, 1)}
            if fired is None and term_after_start is not None:
                text = slurm_out.read_text(errors='replace')
                if start_seen is None and 'start: python -m hyak.train.run_segment' in text:
                    start_seen = time.monotonic()
                if start_seen is not None and time.monotonic() - start_seen >= term_after_start:
                    job.send_signal(signal.SIGTERM)
                    fired = {'signal': 'SIGTERM', 'to': 'batch shell', 'seconds_after_segment_start': term_after_start,
                             'at_seconds': round(time.monotonic() - started, 1)}
            time.sleep(0.1)
        code = job.wait()
    status_path = monitor / 'status.json'
    result = {'script': script, 'stage': stage, 'run': run, 'job_id': job_id, 'exit_code': code, 'fired': fired,
              'seconds': round(time.monotonic() - started, 1), 'slurm_out': str(slurm_out),
              'status': json.loads(status_path.read_text()) if status_path.is_file() else None,
              'halted': (monitor / 'HALTED.json').is_file()}
    if train_log.is_file():
        rows = [json.loads(line) for line in train_log.read_text().splitlines() if line.startswith('{')]
        result['train_log'] = {'lines': len(rows), 'non_json_lines': sum(1 for line in train_log.read_text().splitlines()
                                                                         if line and not line.startswith('{')),
                               'stages': [r.get('stage') for r in rows if r.get('stage') not in ('train',)][-30:],
                               'train_steps': [r['step'] for r in rows if r.get('stage') == 'train'][-5:]}
    log(json.dumps({k: v for k, v in result.items() if k != 'status'}, indent=1))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tests.e2e.sbatch_job', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--run')
    parser.add_argument('--job-id', default='e2e-job')
    parser.add_argument('--term-after-train', type=int)
    parser.add_argument('--term-after-start', type=float)
    parser.add_argument('--term-after-log', help='SIGTERM once the job output holds --term-count lines with this text')
    parser.add_argument('--term-count', type=int, default=1)
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--cpus', default='8', help='SLURM_CPUS_PER_TASK')
    parser.add_argument('--hours', type=float, default=3.0)
    parser.add_argument('--threads', default='2')
    parser.add_argument('--result', type=Path)
    args = parser.parse_args(argv)
    result = run_job(args.world, args.script, args.stage, run=args.run, job_id=args.job_id,
                     term_after_train=args.term_after_train, term_after_start=args.term_after_start,
                     term_after_log=args.term_after_log, term_count=args.term_count, hours=args.hours,
                     threads=args.threads, cpus=args.cpus, extra_env=dict(item.split('=', 1) for item in args.env))
    if args.result:
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
