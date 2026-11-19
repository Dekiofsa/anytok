#!/bin/bash
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --job-name=anytok-predictor-eval
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --time=00:20:00
#SBATCH --output=modal_results/predictor_trajectory/slurm-%A_%a.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(.venv-memory/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
jobs=(uniform-002000 predictor-002000 uniform-004000 predictor-004000 uniform-006000 predictor-006000)
exec .venv-memory/bin/python -u scripts/predictor_trajectory.py --job "${jobs[$SLURM_ARRAY_TASK_ID]}"
