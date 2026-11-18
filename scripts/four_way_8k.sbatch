#!/bin/bash
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --job-name=anytok-eval-8k
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=01:00:00
#SBATCH --output=modal_results/four_way_8k/slurm-%A_%a.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(.venv-memory/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
models=(uniform raw_bpb regret predictor)
exec .venv-memory/bin/python -u scripts/four_way_8k.py --model "${models[$SLURM_ARRAY_TASK_ID]}"
