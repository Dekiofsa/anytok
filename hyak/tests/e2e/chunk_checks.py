"""TEST-ONLY: the chunked forward inside the REAL uniform trainer (stage F of the e2e harness).

    python -m hyak.tests.e2e.chunk_checks --world DIR --out DIR [--variants NAME ...] [--threads 2]

Every variant is ``python -m hyak.train.run_segment --run uniform --smoke`` (production entry point: the
one-update bootstrap, then a resume to update 3 in a new process) in a world built by
``hyak.tests.e2e.world``, with ``HYAK_HN_CHUNK_ROWS`` (environment overrides the config file) and the
shim's test-only switches.  The smoke run's ``events.jsonl`` is kept per variant and the logged
quantities of updates 1-3 are compared (``compare``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from hyak.tests.e2e.world import REPO, world_env

VARIANTS = {
    'pinned': {'HYAK_HN_CHUNK_ROWS': '0'},
    'one_chunk': {'HYAK_HN_CHUNK_ROWS': '1000000'},
    'chunked': {'HYAK_HN_CHUNK_ROWS': '4096'},
    'chunked_again': {'HYAK_HN_CHUNK_ROWS': '4096'},
    'pinned_nodrop': {'HYAK_HN_CHUNK_ROWS': '0', 'HYAK_E2E_DROPOUT': '0'},
    'chunked_nodrop': {'HYAK_HN_CHUNK_ROWS': '4096', 'HYAK_E2E_DROPOUT': '0'},
    'pinned_bf16': {'HYAK_HN_CHUNK_ROWS': '0', 'HYAK_E2E_AUTOCAST': 'bf16'},
    'one_chunk_bf16': {'HYAK_HN_CHUNK_ROWS': '1000000', 'HYAK_E2E_AUTOCAST': 'bf16'},
}
FIELDS = ('loss_nats_per_token', 'input_lexical_loss', 'gradient_norm_before_clip', 'scored_tokens',
          'tokenizer_pool_index', 'truncated_forms', 'rejected_examples', 'hn_lr', 'lm_lr', 'seconds', 'peak_gib')


def run_variant(world: Path, name: str, out: Path, threads: str, attempts: int = 3) -> dict:
    env = world_env(world, {**VARIANTS[name], 'HYAK_E2E_TORCH_THREADS': threads})
    smoke = world / 'runs' / 'smoke' / 'uniform'
    for attempt in range(1, attempts + 1):
        result = subprocess.run([sys.executable, '-m', 'hyak.train.run_segment', '--run', 'uniform', '--smoke'],
                                env=env, cwd=str(REPO), capture_output=True, text=True)
        (out / f'{name}.a{attempt}.out').write_text(result.stdout)
        (out / f'{name}.a{attempt}.err').write_text(result.stderr)
        if result.returncode == 0:
            break
        # Only the miniature held-out files can make a fresh start fail ("Only 3 usable held-out
        # examples"; the sanity tokenizers come from the unseeded Rust sampler): retry from scratch.
        if 'usable held-out examples' not in result.stderr:
            raise RuntimeError(f'variant {name} failed: {result.stderr[-2000:]}')
    else:
        raise RuntimeError(f'variant {name} failed {attempts} times')
    shutil.copyfile(smoke / 'events.jsonl', out / f'{name}.events.jsonl')
    shutil.copyfile(smoke / 'hyak_segment.json', out / f'{name}.hyak_segment.json')
    rows = [json.loads(line) for line in (smoke / 'events.jsonl').read_text().splitlines()]
    train = {row['step']: {k: row.get(k) for k in FIELDS} for row in rows if row.get('stage') == 'train'}
    evals = [row for row in rows if row.get('stage') in ('initial_eval', 'eval')]
    return {'variant': name, 'attempts': attempt, 'train': train, 'evals': evals,
            'chunk': json.loads((smoke / 'hyak_segment.json').read_text())['chunk_identity']}


def compare(a: dict, b: dict) -> dict:
    rows = {}
    for step in sorted(set(a['train']) & set(b['train'])):
        row = {}
        for key in FIELDS:
            x, y = a['train'][step][key], b['train'][step][key]
            if isinstance(x, float) and isinstance(y, float):
                row[key] = {'equal': x == y, 'abs_diff': abs(x - y), 'rel_diff': abs(x - y) / max(abs(x), 1e-30)}
            else:
                row[key] = {'equal': x == y}
        rows[step] = row
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tests.e2e.chunk_checks', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    parser.add_argument('--threads', default='2')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.variants:
        results[name] = run_variant(args.world, name, args.out, args.threads)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2, sort_keys=True))
        print(json.dumps({'variant': name, 'train': {s: r['loss_nats_per_token'] for s, r in results[name]['train'].items()}}),
              flush=True)
    pairs = [('pinned', 'one_chunk'), ('pinned', 'chunked'), ('chunked', 'chunked_again'),
             ('pinned_nodrop', 'chunked_nodrop'), ('pinned_bf16', 'one_chunk_bf16')]
    comparison = {f'{a}~{b}': compare(results[a], results[b]) for a, b in pairs if a in results and b in results}
    (args.out / 'comparison.json').write_text(json.dumps(comparison, indent=2, sort_keys=True))
    print(json.dumps(comparison, indent=1, sort_keys=True)[:20000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
