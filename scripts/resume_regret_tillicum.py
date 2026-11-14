"""Continue the exact Modal regret run using the validated GPU-only execution."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

OLD_TRAINER_SHA256 = '76ff232402154653c5899826a4f63f8a4afadf5a443829f87919f8e75b31c6d4'
REGRET_REWARD = 'paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb'


def migration(saved, requested, checkpoint_step):
    if saved == requested:
        return None
    if (checkpoint_step != 5288 or saved.get('training_script_sha256') != OLD_TRAINER_SHA256
            or saved.get('policy_reward') != REGRET_REWARD
            or requested.get('policy_reward') != REGRET_REWARD
            or requested.get('reward_objective') != 'regret'
            or requested.get('memory_execution') != 'optimized'
            or saved.get('policy', {}).get('entropy_weight') != 0):
        raise ValueError('This migration only continues the exact entropy-zero regret experiment')
    allowed = {'model', 'warmup_checkpoint', 'training_script_sha256', 'memory_execution',
               'memory_source_sha256', 'reward_objective'}
    changed = {key for key in set(saved) | set(requested) if saved.get(key) != requested.get(key)}
    if not changed <= allowed:
        raise ValueError('Material training settings changed: ' + ', '.join(sorted(changed - allowed)))
    return dict(stage='execution_migration', step=checkpoint_step,
                reason='Modal B300 to Tillicum H200; verified identical input artifacts and GPU-only memory fix',
                changes={key: dict(before=saved.get(key), after=requested.get(key)) for key in sorted(changed)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path[:0] = [str(args.code / 'scripts'), str(args.code)]
    source = args.code / 'scripts/train_joint_smollm_input.py'
    spec = importlib.util.spec_from_file_location('tillicum_regret_trainer', source)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    original = trainer.check_resume_settings
    def checked(saved, requested, allow_microbatch, checkpoint_step, allow_min_chars=False):
        if saved == requested:
            return original(saved, requested, allow_microbatch, checkpoint_step, allow_min_chars)
        return migration(saved, requested, checkpoint_step)
    trainer.check_resume_settings = checked
    argv = args.trainer_args[1:] if args.trainer_args[:1] == ['--'] else args.trainer_args
    if '--resume' not in argv or argv[argv.index('--reward-objective') + 1] != 'regret':
        raise ValueError('Explicit regret resume required')
    output = Path(argv[argv.index('--output') + 1])
    config = output / 'run_config.json'
    original_config = output / 'run_config.modal.json'
    if not original_config.exists():
        original_config.write_bytes(config.read_bytes())
    sys.argv = [str(source), *argv]
    trainer.main()


if __name__ == '__main__':
    main()
