"""TEST-ONLY: an evaluation world that scores the REAL miniature checkpoints of an e2e world on another TEST pool.

    python -m hyak.tests.e2e.eval_world build --world DIR --base WORLD [--smoke-pool DIR] [--force]

Why: the e2e training world's pool is backed by three 4,000-entry smoke tokenizers.  With them, 29 rows of the
stored pool panel whose context was valid in the original panel would need a trim, so
``python -m hyak.eval.validate_prefixes --panel pool`` stops with exit 3 for every index subset (all three
families force such trims).  That is the protocol working, not a defect.  To exercise the pool-panel
scorer, this builds a second world that shares the base world's runs (read only: the scorer only reads
``lm.pt`` and ``hypernet/`` of a checkpoint) and tiny model, but whose TEST pool is backed by the three
32,000-entry smoke tokenizers made by ``python -m hyak.pool.worker`` with the patched wheel
(SCRATCH/pool-smoke/miniB: BPE/regex, WordPiece/line256, Unigram/no-regex), with which both stored panels
validate.  The hypernetwork is vocabulary-agnostic, so a checkpoint trained on the 4k pool can be scored
on any tokenizer.

Layout: ``DIR/config.env`` (ANYTOK_ROOT=DIR/root, ANYTOK_RUNS=<base>/runs), ``DIR/root/code`` -> base code,
``DIR/root/input/model`` (hard links to the base model files), ``DIR/root/input/pool`` (TEST plan + 1,000
hard-linked tokenizer dirs + full-results.json from ``python -m hyak.pool.collect --allow-test-plan``),
``DIR/test-plan.json``, ``DIR/model-hashes.json``, ``DIR/world.json``, ``DIR/env.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from hyak.tests.e2e.world import (CHUNK_ROWS, REPO, build_pool, remove, scratch_dir, world_env, write_config,
                                  write_model_hashes)


def build(target: Path, base: Path, *, smoke_pool: Path | None = None, force: bool = False, log=print) -> dict:
    target, base = Path(os.path.abspath(target)), Path(os.path.abspath(base))
    scratch = scratch_dir()
    smoke_pool = Path(smoke_pool or scratch / 'pool-smoke' / 'miniB' / 'tokenizers')
    if target.exists():
        if not force:
            raise FileExistsError(f'{target} exists (use --force)')
        remove(target)
    (target / 'root' / 'input' / 'model').mkdir(parents=True)
    (target / 'monitor').mkdir()
    write_config(target, chunk_rows=CHUNK_ROWS)
    config = (target / 'config.env').read_text().replace(f'ANYTOK_RUNS={target / "runs"}', f'ANYTOK_RUNS={base / "runs"}')
    (target / 'config.env').write_text(config)
    (target / 'root' / 'code').symlink_to(base / 'root' / 'code')
    for path in sorted((base / 'root' / 'input' / 'model').iterdir()):
        os.link(path, target / 'root' / 'input' / 'model' / path.name)
    write_model_hashes(target)
    old = os.environ.get('HYAK_E2E_SMOKE_POOL')
    os.environ['HYAK_E2E_SMOKE_POOL'] = str(smoke_pool)
    try:
        pool = build_pool(target, scratch, log=log)
    finally:
        if old is None:
            os.environ.pop('HYAK_E2E_SMOKE_POOL', None)
        else:
            os.environ['HYAK_E2E_SMOKE_POOL'] = old
    base_info = json.loads((base / 'world.json').read_text())
    info = {**base_info, 'eval_world_of': str(base), 'smoke_pool': str(smoke_pool), 'pool_vocab': pool['vocab'],
            'plan_sha256': pool['plan_sha256'], 'plan_file_sha256': pool['plan_file_sha256'],
            'pool_index_sha256': pool['index_sha256']}
    (target / 'world.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
    env = world_env(target)
    lines = [f'export {k}="{v}"' for k, v in sorted(env.items())
             if k.startswith(('HYAK_', 'PYTHON', 'OMP_', 'HF_', 'TRANSFORMERS_', 'TOKENIZERS_'))]
    lines += [f'export PY={sys.executable}', f'cd {REPO}']
    (target / 'env.sh').write_text('\n'.join(lines) + '\n')
    log(json.dumps(info, indent=2, sort_keys=True))
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tests.e2e.eval_world', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build',))
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--smoke-pool', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    build(args.world, args.base, smoke_pool=args.smoke_pool, force=args.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
