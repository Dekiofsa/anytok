#!/bin/bash
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --job-name=anytok-predictor-20k
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=modal_results/tillicum_hardness/slurm-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(.venv-memory/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
exec .venv-memory/bin/python -u scripts/tillicum_hardness_job.py
