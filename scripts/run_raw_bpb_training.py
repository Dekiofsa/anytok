"""Run corrected full-pool training with raw BPB and GPU-only memory savings.

python scripts/run_raw_bpb_training.py --inputs /cloud/input --output /cloud/raw-bpb-run
Use --prepare-only to stage/inspect the exact command without starting training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MEMORY_FILES = ['pool_training_memory.py', 'local_memory_variants.py', 'chunked_hypernetwork.py',
                'lossless_layer_norm.py', 'lossless_mlp.py']


def source_files():
    base = ROOT / 'runtime/anytok_pool_20k'
    files = {str(p.relative_to(base)): p for p in (base / 'hf_hypernet').rglob('*.py')}
    files.update({
        'scripts/train_joint_smollm_input.py': ROOT / 'runtime/anytok_pool_raw_bpb_rl_20k/scripts/train_joint_smollm_input.py',
        'hf_hypernet/settings_policy.py': ROOT / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py',
        'hf_hypernet/paired_reward.py': ROOT / 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py',
        'rust_utils/target/release/librust_utils.so': base / 'rust_utils/target/release/librust_utils.so',
    })
    files.update({'scripts/' + name: ROOT / 'scripts' / name for name in MEMORY_FILES})
    return files


def prepare_code(parent):
    files = source_files()
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    code = parent / ('.raw-bpb-code-' + revision[:16])
    code.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
                raise ValueError('Staged source was modified: ' + str(target))
        else:
            shutil.copy2(source, target)
    (code / 'source-manifest.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    return code


def command(args, code):
    inputs = args.inputs.resolve()
    result = [sys.executable, '-u', str(code / 'scripts/train_joint_smollm_input.py'),
        '--recipe', str(inputs / 'recipe.json'), '--data-dir', str(inputs / 'corpus'),
        '--data-manifest', str(inputs / 'corpus/manifest.json'), '--eval-dir', str(inputs / 'eval'),
        '--output', str(args.output.resolve()), '--warmup-checkpoint', str(inputs / 'warmup'),
        '--model', str(inputs / 'model'), '--tokenizer-pool', str(inputs / 'pool'),
        '--eval-tokenizers', str(inputs / 'uniform_heldout_tokenizers.json'),
        '--microbatch-size', '32', '--policy-entropy', '0', '--policy-lr', '0.001',
        '--policy-baseline-decay', '0.99', '--reward-objective', args.reward_objective,
        '--memory-execution', args.memory_execution, '--save-every', '500', '--eval-every', '500',
        '--stop-at-update', str(args.stop_at_update), '--seed', '42']
    milestones = list(range(500, args.stop_at_update + 1, 500))
    if milestones:
        result += ['--protected-steps', *map(str, milestones)]
    if args.resume:
        result.append('--resume')
    return result


def run_and_monitor(cmd, args, env):
    """Relay logs and write the existing progress_json adapter's status format."""
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda received, frame: process.send_signal(received) if process.poll() is None else None)
    profile = env.get('ANYTOK_RUN_PROFILE') or env.get('SLURM_JOB_ACCOUNT') or env.get('MODAL_PROFILE', 'local')
    platform = env.get('ANYTOK_RUN_PLATFORM', 'Slurm GPU' if env.get('SLURM_JOB_ID') else 'GPU training')
    latest = dict(step=0, total=args.stop_at_update)
    def save(stage, **fields):
        latest.update(fields, stage=stage, at_utc=datetime.now(timezone.utc).isoformat())
        latest['workers'] = [] if stage in ['complete', 'stopped', 'failed'] else [dict(
            id=env.get('SLURM_JOB_ID', str(process.pid)), profile=profile, platform=platform, state=stage)]
        temporary = args.output / '.status.json.tmp'
        temporary.write_text(json.dumps(latest) + '\n')
        temporary.replace(args.output / 'status.json')
    for line in process.stdout:
        print(line, end='', flush=True)
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or 'stage' not in row:
            continue
        stage = row['stage']
        if stage == 'paused_at':
            stage = 'complete' if row['step'] >= args.stop_at_update else 'stopped'
        save(stage, **{k: v for k, v in row.items() if k != 'stage'})
    code = process.wait()
    if args.output.is_dir():
        save('failed' if code else latest['stage'], returncode=code)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at-update', type=int, default=20000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reward-objective', choices=['raw_bpb', 'regret'], default='raw_bpb')
    parser.add_argument('--memory-execution', choices=['optimized', 'original'], default='optimized')
    args = parser.parse_args()
    if not 1 <= args.stop_at_update <= 20000:
        parser.error('--stop-at-update must be between 1 and 20000')
    required = ['recipe.json', 'corpus/manifest.json', 'pool/full-results.json', 'pool/plan.json',
                'model/config.json', 'warmup/config.json', 'uniform_heldout_tokenizers.json',
                'uniform_run_config.json', 'rl-input-audit.json']
    for name in required:
        if not (args.inputs / name).is_file():
            parser.error('Missing original input: ' + name)
    code = prepare_code(args.output.resolve().parent)
    cmd = command(args, code)
    print(json.dumps(dict(stage='training_command', code=str(code), command=cmd)), flush=True)
    if args.prepare_only:
        return
    if importlib.metadata.version('torch').split('+')[0] != '2.15.0':
        raise RuntimeError('Use the verified torch==2.15.0 environment')
    env = {**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
    env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    raise SystemExit(run_and_monitor(cmd, args, env))


if __name__ == '__main__':
    main()
