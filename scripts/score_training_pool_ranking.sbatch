#!/bin/bash
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --job-name=anytok-policy-rank
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=modal_results/regret_pool_ranking/slurm-%A-%a.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(.venv-memory/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
exec .venv-memory/bin/python -u scripts/score_training_pool_ranking.py --shard "$SLURM_ARRAY_TASK_ID"
