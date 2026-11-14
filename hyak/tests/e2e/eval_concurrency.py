"""TEST-ONLY: two REAL scoring processes on the same run and panel, one of them killed (e2e evaluation stage D).

    python -m hyak.tests.e2e.eval_concurrency --world DIR --run uniform --panel pool --checkpoint CKPT \
        --indices 0-11 --log-dir DIR [--kill-after 2]

1. A and B start together: ``python -m hyak.eval.score --run R --panel P --checkpoint CKPT --indices ...``
   (production entry point, CPU shim, explicit test parameters from ``world_env``).
2. Once A has logged ``--kill-after`` scored tokenizers, A is killed with SIGKILL in the middle of its next
   tokenizer (its claim directory and owner.json stay behind).
3. B runs to the end of its work list.  If B reaches A's claimed index after the kill it takes the claim
   over itself (same host, non-Slurm owner whose pid is gone); if it passed it while A was alive, B comes
   back to it after its pass (it waits for claims of other jobs) and takes it over then.
4. C (a resubmission) starts afterwards and takes over any claim left behind, then finishes.

The miniature checkpoints are not step 20,000, so the jobs pass ``--allow-non-final-checkpoint``.

Checks written to ``<log-dir>/D-result.json``: every index has exactly one result file and was logged as
scored exactly once over A, B and C; ``canonical.json`` was computed once (one 'controls written' line);
the claim A held when it was killed was taken over by B or C (reason: owner pid gone); no claim
directories are left.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from hyak.tests.e2e.world import REPO, world_env

SCORED = re.compile(r'\] (\d{4}): scored in')


def start(world, name, args, log_dir):
    command = [sys.executable, '-m', 'hyak.eval.score', '--run', args.run, '--panel', args.panel,
               '--checkpoint', str(args.checkpoint), '--indices', *args.indices, '--no-status',
               '--allow-non-final-checkpoint']
    out = open(log_dir / f'D-{name}.out', 'w')
    err = open(log_dir / f'D-{name}.err', 'w')
    process = subprocess.Popen(command, env=world_env(world), cwd=str(REPO), stdout=out, stderr=err)
    return {'name': name, 'process': process, 'out': log_dir / f'D-{name}.out', 'started': time.monotonic()}


def scored(job) -> list:
    return [int(m.group(1)) for m in SCORED.finditer(job['out'].read_text(errors='replace'))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tests.e2e.eval_concurrency', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--indices', nargs='+', required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--kill-after', type=int, default=2)
    parser.add_argument('--kill-delay', type=float, default=2.0, help='seconds into the next tokenizer')
    args = parser.parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    world = args.world
    from hyak.common.config import load_config
    from hyak.common.paths import Paths
    os.environ['HYAK_CONFIG'] = str(world / 'config.env')
    paths = Paths.from_config(load_config(world / 'config.env'))
    out_dir = paths.eval_scores_dir(args.run) / args.panel
    claims_dir = paths.eval_root / 'claims' / f'scores-{args.run}-{args.panel}'
    report = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'out_dir': str(out_dir)}
    a = start(world, 'A', args, args.log_dir)
    b = start(world, 'B', args, args.log_dir)
    killed = None
    while a['process'].poll() is None:
        done = scored(a)
        if killed is None and len(done) >= args.kill_after:
            time.sleep(args.kill_delay)
            live = sorted(p.name for p in claims_dir.iterdir() if p.is_dir() and not p.name.startswith('.')) \
                if claims_dir.is_dir() else []
            owners = {}
            for name in live:
                try:
                    owners[name] = json.loads((claims_dir / name / 'owner.json').read_text())['pid']
                except (OSError, ValueError, KeyError):
                    owners[name] = None
            a['process'].send_signal(signal.SIGKILL)
            killed = {'after_scored': done, 'claims_at_kill': owners, 'a_pid': a['process'].pid,
                      'at_seconds': round(time.monotonic() - a['started'], 1)}
            break
        time.sleep(0.2)
    a_code = a['process'].wait()
    b_code = b['process'].wait()
    stale = sorted(p.name for p in claims_dir.iterdir() if p.is_dir() and not p.name.startswith('.')) \
        if claims_dir.is_dir() else []
    c = start(world, 'C', args, args.log_dir)
    c_code = c['process'].wait()
    left = sorted(p.name for p in claims_dir.iterdir() if p.is_dir() and not p.name.startswith('.')) \
        if claims_dir.is_dir() else []
    per_job = {j['name']: scored(j) for j in (a, b, c)}
    every = [i for values in per_job.values() for i in values]
    from hyak.eval import parse_indices
    wanted = parse_indices(args.indices, range(1000))
    results = {}
    for index in wanted:
        path = out_dir / f'{index:04d}.json'
        results[index] = json.loads(path.read_text())['job_id'] if path.is_file() else None
    texts = {j['name']: j['out'].read_text(errors='replace') for j in (a, b, c)}
    controls_lines = {name: text.count('controls written') for name, text in texts.items()}
    import ast
    takeovers = {}
    for name in ('B', 'C'):
        last = texts[name].strip().splitlines()[-1] if texts[name].strip() else ''
        counts = ast.literal_eval(last.split(': ', 1)[1]) if "'takeovers'" in last else {}
        takeovers[name] = [(t['name'], t['reason']) for t in counts.get('takeovers', [])]
    held_by_a = sorted(n for n, pid in ((killed or {}).get('claims_at_kill') or {}).items() if pid == a['process'].pid)
    taken = sorted(n for rows in takeovers.values() for n, reason in rows if 'is gone' in reason)
    report.update(
        exit_codes={'A': a_code, 'B': b_code, 'C': c_code}, killed=killed, scored_by_job=per_job,
        claims_left_after_A_and_B=stale, claims_left_at_end=left, result_job_ids=results,
        controls_written_lines=controls_lines, canonical_json_job=json.loads((out_dir / 'canonical.json').read_text())['job_id'],
        takeovers=takeovers, claims_held_by_a_at_kill=held_by_a,
        checks={
            'every_index_has_one_result': all(results[i] is not None for i in wanted),
            'every_index_scored_exactly_once': sorted(every) == sorted(wanted),
            'controls_computed_once': sum(controls_lines.values()) == 1,
            'a_was_killed_holding_a_claim': bool(held_by_a),
            'a_claims_taken_over': bool(held_by_a) and taken == held_by_a,
            'no_claims_left': not left,
        })
    report['ok'] = all(report['checks'].values())
    (args.log_dir / 'D-result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
