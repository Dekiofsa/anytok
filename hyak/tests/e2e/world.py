"""TEST-ONLY: build a miniature "cluster world" in which the REAL pinned trainers run on CPU.

    python -m hyak.tests.e2e.world build  --world DIR [--width 64] [--force]
    python -m hyak.tests.e2e.world env    --world DIR          (print the child environment as KEY=VALUE)

Layout (production names from hyak.common.paths, rooted in DIR):

    DIR/config.env                      ANYTOK_ROOT=DIR/root, ANYTOK_RUNS=DIR/runs, HYAK_HN_CHUNK_ROWS=4096
    DIR/monitor/                        HYAK_MONITOR_ROOT (status.json, events.jsonl per stage)
    DIR/root/code/{uniform,rl}          python -m hyak.code.stage_code --rust-library <macOS build>
    DIR/root/corpus/                    copy of SCRATCH/mini-corpus (test-scale build, 23 buckets)
    DIR/root/input/{recipe.json,pool-recipe.json,corpus/,eval/,manifest.json}
                                        python -m hyak.corpus.stage_inputs --allow-test-scale
    DIR/root/input/model/               tiny random Llama (vocab 49,152, hidden WIDTH, BF16) + real SmolLM2 tokenizer.json
    DIR/root/input/hypernet_init/       hyak.model.init_hypernet.create(..., profile=width WIDTH) (D1 recipe)
    DIR/root/input/pool/                TEST plan (the 1,000 production configs, vocab_size of the smoke set
                                        (4,000), family fixed per algorithm) + 1,000 tokenizer dirs hard-linked
                                        to three smoke tokenizers (BPE/regex, WordPiece/line256, Unigram/no-regex) +
                                        full-results.json written by python -m hyak.pool.collect --allow-test-plan
    DIR/runs/                           ANYTOK_RUNS (uniform, rl, smoke/, .locks, .hyak-cache)
    DIR/world.json                      facts the harness needs (init bytes, test plan SHA-256, width)

Children run with ``world_env()``: PYTHONPATH = <this dir>/shim (CPU shim sitecustomize) + the repository,
HYAK_E2E_CPU_SHIM=1, HYAK_E2E_INIT_BYTES / HYAK_E2E_PLAN_SHA256 / HYAK_E2E_MODEL_HASHES /
HYAK_E2E_ALLOW_TEST_SCALE (explicit run_segment test parameters).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SHIM_DIR = HERE / 'shim'
REPO = HERE.parents[2]
DEFAULT_WIDTH = 32
#: The config and hyak.memory accept HYAK_HN_CHUNK_ROWS >= 4096 only; with the miniature's 4,000-entry
#: vocabularies that is one chunk (the multi-chunk comparison of stage F needs a larger test vocabulary).
CHUNK_ROWS = 4096
PLATFORM = 'local Mac CPU (hyak e2e test shim)'
#: Smoke tokenizers (pool owner's miniB run, patched wheel) backing every config of one algorithm.
BACKING = {'bpe': '0000', 'wordpiece': '0001', 'unigram': '0002'}


def scratch_dir() -> Path:
    value = os.environ.get('HYAK_E2E_SCRATCH')
    return Path(value) if value else Path(sys.prefix).parent


def assets(scratch: Path | None = None) -> dict:
    scratch = scratch or scratch_dir()
    return {'scratch': scratch,
            'rust_library': scratch / 'rust-utils-macos' / 'librust_utils.so',
            'rust_provenance': scratch / 'rust-utils-macos' / 'provenance.json',
            'mini_corpus': scratch / 'mini-corpus',
            'tokenizer': scratch / 'smollm2' / 'full' / 'tokenizer.json',
            'smoke_pool': _smoke_pool(scratch)}


def _smoke_pool(scratch: Path) -> Path:
    """Three smoke tokenizers made by ``python -m hyak.pool.worker`` with the patched wheel: the 4,000-entry set
    (SCRATCH/e2e/pool4k, fast enough for thousands of CPU updates) if present, else the pool owner's 32,000-entry set."""
    if os.environ.get('HYAK_E2E_SMOKE_POOL'):
        return Path(os.environ['HYAK_E2E_SMOKE_POOL'])
    small = scratch / 'e2e' / 'pool4k' / 'pool' / 'tokenizers'
    return small if (small / '0002' / 'tokenizer.json').is_file() else scratch / 'pool-smoke' / 'miniB' / 'tokenizers'


def missing_assets(scratch: Path | None = None) -> list:
    found = assets(scratch)
    rows = [found['rust_library'], found['mini_corpus'] / 'full-v1', found['tokenizer']]
    rows += [found['smoke_pool'] / name / 'tokenizer.json' for name in BACKING.values()]
    return [str(path) for path in rows if not path.exists()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    """scripts/tokenizer_pool.py:30 (canonical JSON SHA-256), as hyak.pool.digest."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def remove(path: Path) -> None:
    """Delete a file or tree, including the read-only staged code trees and init files."""
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        for folder, dirs, _files in os.walk(path):
            os.chmod(folder, 0o755)
            for name in dirs:
                os.chmod(os.path.join(folder, name), 0o755)
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def world_env(world: Path, extra: dict | None = None, *, shim: bool = True) -> dict:
    """Environment of every child process of the harness (production entry points + the CPU shim)."""
    world = Path(world)
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('SLURM_') or k.startswith('HYAK_') or k in ('PYTHONPATH', 'ANYTOK_ROOT', 'ANYTOK_RUNS'))}
    env.update(PYTHONDONTWRITEBYTECODE='1', HYAK_CONFIG=str(world / 'config.env'),
               HYAK_MONITOR_ROOT=str(world / 'monitor'), OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HOME=str(world / 'hf-home'), TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=f'{SHIM_DIR}{os.pathsep}{REPO}' if shim else str(REPO))
    info_path = world / 'world.json'
    if shim:
        env['HYAK_E2E_CPU_SHIM'] = '1'
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            env['HYAK_E2E_INIT_BYTES'] = str(info['init_bytes'])
            env['HYAK_E2E_PLAN_SHA256'] = info['plan_file_sha256']
            env['HYAK_E2E_ALLOW_TEST_SCALE'] = '1'          # the miniature corpus is staged --allow-test-scale
            env['HYAK_E2E_AUTOCAST'] = info.get('autocast', 'bf16')
        # explicit TEST parameters of the evaluation entry points (see the shim)
        if (world / 'test-plan.json').is_file():
            env['HYAK_E2E_POOL_PLAN'] = str(world / 'test-plan.json')
        if (world / MODEL_HASHES).is_file():
            env['HYAK_E2E_MODEL_HASHES'] = str(world / MODEL_HASHES)
    env.update(extra or {})
    return env


#: The tiny model's {name: {bytes, sha256}} table (the scorer's test parameter ``model_hashes``).
MODEL_HASHES = 'model-hashes.json'


def write_model_hashes(world: Path) -> Path:
    """Hash table of ``root/input/model`` in the shape of plans/zett-modal-20k-inputs.json ``files['model/...']``."""
    model_dir = Path(world) / 'root' / 'input' / 'model'
    table = {name: {'bytes': (model_dir / name).stat().st_size, 'sha256': _sha(model_dir / name)}
             for name in ('config.json', 'model.safetensors', 'tokenizer.json')}
    path = Path(world) / MODEL_HASHES
    path.write_text(json.dumps(table, indent=2, sort_keys=True) + '\n')
    return path


def run(command, world: Path, *, log=print, extra=None, shim=True, check=True, **kwargs):
    log('$ ' + ' '.join(str(c) for c in command))
    result = subprocess.run([str(c) for c in command], env=world_env(world, extra, shim=shim), cwd=str(REPO),
                            text=True, capture_output=True, **kwargs)
    if result.stdout.strip():
        log(result.stdout.strip()[-4000:])
    if result.stderr.strip():
        log(result.stderr.strip()[-4000:])
    if check and result.returncode != 0:
        raise RuntimeError(f'{command} exited {result.returncode}')
    return result


def write_config(world: Path, *, chunk_rows: int = CHUNK_ROWS, extra: dict | None = None) -> Path:
    lines = [f'ANYTOK_ROOT={world / "root"}', f'ANYTOK_RUNS={world / "runs"}', f'HYAK_HN_CHUNK_ROWS={chunk_rows}',
             f'HYAK_PLATFORM="{PLATFORM}"']
    lines += [f'{k}={v}' for k, v in (extra or {}).items()]
    path = world / 'config.env'
    path.write_text('# hyak e2e test world (TEST ONLY)\n' + '\n'.join(lines) + '\n')
    return path


TINY_LM = r'''
import json, shutil, sys
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
out, tokenizer, width = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
config = LlamaConfig(vocab_size=49152, hidden_size=width, intermediate_size=2 * width, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                     tie_word_embeddings=True, bos_token_id=0, eos_token_id=0, rms_norm_eps=1e-5)
lm = LlamaForCausalLM(config).to(torch.bfloat16)
lm.config.dtype = torch.bfloat16
tmp = out.with_name(out.name + '.tmp')
shutil.rmtree(tmp, ignore_errors=True)
lm.save_pretrained(tmp, safe_serialization=True)
for extra in tmp.iterdir():
    if extra.name not in ('config.json', 'model.safetensors'):
        extra.unlink()
shutil.copyfile(tokenizer, tmp / 'tokenizer.json')
tmp.rename(out)
print(json.dumps({'model_dir': str(out), 'files': sorted(p.name for p in out.iterdir())}))
'''

INIT = r'''
import json, sys
from pathlib import Path
from hyak.model import init_hypernet as ih
model, code, out, runs, width = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
profile = ih.Profile(width=width, expected_model_bytes=None, expected_trained_model_bytes=None,
                     expected_parameters=None, expected_frozen=4 * width, reference=False)
manifest = ih.create(model, code, out, profile=profile, run_dirs=[runs / 'uniform', runs / 'rl'])
print(json.dumps({'outputs': manifest['outputs'], 'checks': {k: v.get('passed') for k, v in manifest['checks'].items()
                                                             if isinstance(v, dict)}}))
'''


def build_pool(world: Path, scratch: Path, *, log=print) -> dict:
    """TEST plan + 1,000 tokenizer directories, then the production collector into a staging dir."""
    root = world / 'root'
    smoke = assets(scratch)['smoke_pool']
    real = json.loads((REPO / 'plans' / 'tokenizer-pool-1000-v1.json').read_bytes())
    backing = {alg: json.loads((smoke / name / 'manifest.json').read_text()) for alg, name in BACKING.items()}
    plan ={k: v for k, v in real.items() if k != 'plan_sha256'}
    plan['total_bytes_per_tokenizer'] = sum(next(iter(backing.values()))['config']['bucket_bytes'].values())
    plan['distribution'] = {**real['distribution'],
                            'hyak_e2e_test': 'TEST plan: production configs with the smoke vocab_size and one '
                                             'pretokenizer per algorithm, backed by three smoke tokenizers'}
    configs = []
    vocab = {m['vocab_size_actual'] for m in backing.values()}
    if len(vocab) != 1:
        raise ValueError(f'the three smoke tokenizers have different vocabulary sizes {vocab}')
    vocab = vocab.pop()
    for config in real['configs']:
        source = backing[config['algorithm']]['config']
        new = {k: v for k, v in config.items() if k != 'config_sha256'}
        new.update(pretokenizer=source['pretokenizer'], vocab_size=vocab, bucket_bytes=source['bucket_bytes'])
        new['config_sha256'] = _digest(new)
        configs.append(new)
    plan['configs'] = configs
    plan['plan_sha256'] = _digest(plan)
    plan_path = world / 'test-plan.json'
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    plan_file_sha = _sha(plan_path)
    staging = world / 'pool-staging'
    shutil.rmtree(staging, ignore_errors=True)
    for config in configs:
        alg = config['algorithm']
        folder = staging / 'tokenizers' / f'{config["index"]:04d}'
        folder.mkdir(parents=True)
        os.link(smoke / BACKING[alg] / 'tokenizer.json', folder / 'tokenizer.json')
        manifest = json.loads(json.dumps(backing[alg]))
        manifest['config'] = config
        manifest['provenance'] = {**manifest['provenance'], 'plan_sha256': plan['plan_sha256'],
                                  'plan_file_sha256': plan_file_sha}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    run([sys.executable, '-m', 'hyak.pool.collect', '--plan', plan_path, '--allow-test-plan', '--pool-dir', staging,
         '--code-dir', root / 'code' / 'uniform', '--rl-code-dir', root / 'code' / 'rl',
         '--monitor-dir', world / 'monitor' / 'pool'], world, log=log, shim=False)
    target = root / 'input' / 'pool'
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    return {'plan_sha256': plan['plan_sha256'], 'plan_file_sha256': _sha(target / 'plan.json'),
            'index_sha256': _sha(target / 'full-results.json'), 'vocab': vocab}


def build(world: Path, *, width: int = DEFAULT_WIDTH, force: bool = False, scratch: Path | None = None,
          chunk_rows: int = CHUNK_ROWS, autocast: str = 'off', log=print) -> dict:
    world = Path(os.path.abspath(world))
    scratch = scratch or scratch_dir()
    missing = missing_assets(scratch)
    if missing:
        raise FileNotFoundError(f'missing scratch assets: {missing}')
    if world.exists():
        if not force:
            raise FileExistsError(f'{world} exists (use --force to rebuild it)')
        for path in list(world.iterdir()):
            if path.name in ('WORLD.md',):
                continue
            remove(path)
    world.mkdir(parents=True, exist_ok=True)
    started = time.time()
    found = assets(scratch)
    write_config(world, chunk_rows=chunk_rows)
    (world / 'runs').mkdir()
    (world / 'monitor').mkdir()
    py = sys.executable
    run([py, '-m', 'hyak.code.stage_code', '--rust-library', found['rust_library'], '--rust-provenance',
         found['rust_provenance']], world, log=log, shim=False)
    shutil.copytree(found['mini_corpus'], world / 'root' / 'corpus', symlinks=True)
    run([py, '-m', 'hyak.corpus.stage_inputs', '--allow-test-scale'], world, log=log, shim=False)
    model_dir = world / 'root' / 'input' / 'model'
    run([py, '-c', TINY_LM, model_dir, found['tokenizer'], width], world, log=log, shim=False)
    write_model_hashes(world)
    run([py, '-c', INIT, model_dir, world / 'root' / 'code' / 'uniform', world / 'root' / 'input' / 'hypernet_init',
         world / 'runs', width], world, log=log, shim=False)
    pool = build_pool(world, scratch, log=log)
    info = {'width': width, 'chunk_rows': chunk_rows, 'scratch': str(scratch), 'repo': str(REPO), 'autocast': autocast,
            'smoke_pool': str(found['smoke_pool']), 'pool_vocab': pool['vocab'],
            'init_bytes': (world / 'root' / 'input' / 'hypernet_init' / 'model.safetensors').stat().st_size,
            'plan_sha256': pool['plan_sha256'], 'plan_file_sha256': pool['plan_file_sha256'],
            'pool_index_sha256': pool['index_sha256'], 'python': sys.executable,
            'built_seconds': round(time.time() - started, 1)}
    (world / 'world.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
    log(json.dumps(info, indent=2, sort_keys=True))
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tests.e2e.world', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'env'))
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS)
    parser.add_argument('--autocast', choices=('bf16', 'off'), default='off',
                        help='CPU autocast of the shim (bf16 follows the trainer dtype path but is ~8x slower)')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'build':
        build(args.world, width=args.width, force=args.force, chunk_rows=args.chunk_rows, autocast=args.autocast)
    else:
        for key, value in sorted(world_env(args.world).items()):
            if key.startswith(('HYAK_', 'PYTHON', 'OMP_', 'HF_', 'TRANSFORMERS_', 'TOKENIZERS_')):
                print(f'{key}={value}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
