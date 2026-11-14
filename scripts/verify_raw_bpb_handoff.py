"""Bounded real-input fresh-training and restart check; never runs 20k steps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from package_training_inputs import atomic_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/raw_bpb_handoff'


def main():
    started = time.monotonic()
    output = STATE / 'smoke-run'
    base = [sys.executable, '-u', 'scripts/run_raw_bpb_training.py',
            '--inputs', 'modal_results/local_memory/artifacts/input', '--output', str(output)]
    if output.exists():
        raise FileExistsError('Use a fresh output for the initialization/restart verification')
    for label, extra in [('fresh-two', ['--stop-at-update', '2']),
                         ('resume-third', ['--resume', '--stop-at-update', '3'])]:
        with (STATE / (label + '.log')).open('w') as stream:
            subprocess.run(base + extra, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
    config = json.loads((output / 'run_config.json').read_text())
    assert config['reward_objective'] == 'raw_bpb' and config['memory_execution'] == 'optimized'
    assert config['policy']['entropy_weight'] == 0 and config['pool_size'] == 1000
    rows = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    train = [row for row in rows if row['stage'] == 'train']
    assert [row['step'] for row in train] == [1, 2, 3]
    for row in train:
        assert row['policy_reward_bpb'] == row['train_bpb']
        assert abs(row['paired_gap_bpb'] - (row['train_bpb'] - row['paired_canonical_bpb'])) < 1e-12
        assert row['paired_valid_examples'] == 128 and row['paired_maximum_positions'] <= 128
        assert row['policy_updates'] == row['step'] and row['reward_objective'] == 'raw_bpb'
    import torch  # CPU checkpoint metadata only; parent creates no CUDA context.
    counters = {}
    for number in [2, 3]:
        folder = output / f'step-{number:06d}'
        state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True)
        assert state['step'] == number and state['bandit']['updates'] == number
        assert int(state['bandit']['counts'].sum()) == number
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['kind'] == 'full_restart_checkpoint' and manifest['step'] == number
        counters[number] = dict(model=state['step'], policy=state['bandit']['updates'],
                                selections=int(state['bandit']['counts'].sum()))
    result = dict(outcome='complete', slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        account=os.environ.get('SLURM_JOB_ACCOUNT'), node=os.environ.get('SLURM_JOB_NODELIST'),
        wall_seconds=time.monotonic() - started, training_updates=3, pool_size=1000,
        raw_rewards_verified=True, corrected_pairs_verified=True, resume_counters=counters,
        peak_allocated_gib=max(row['peak_gib'] for row in train),
        steps=train, config=config, initial_eval=next(row for row in rows if row['stage'] == 'initial_eval'))
    atomic_json(STATE / 'training-verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
