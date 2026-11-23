#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --job-name=anytok-rank20k
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/stf/%u/anytok-runs/rawbpb-pool-ranking-20000/slurm-%A-%a.log
#
# The complete frozen sweep of the step-20,000 checkpoint: four nonoverlapping shards of 250
# tokenizers, one H200 each. A resubmitted task resumes from its saved scores and batches.
# Submit only after the prep job has ended: its smoke test writes shard 0's status, partial and
# score files, which shard 0 then reuses. Either
#   sbatch --dependency=afterok:<prep job id> score_rawbpb_pool_ranking.sbatch
# or plain sbatch once the observer reports 'prepared'. The state folder
# /gpfs/projects/stf/$USER/anytok-runs/rawbpb-pool-ranking-20000 and its staged code/ must exist
# (Slurm opens the --output logs there before this script runs; a missing folder means no log).
# No CPU or memory reservations (project rule).

set -euo pipefail
USER=${USER:-$(id -un)}
P=/gpfs/projects/stf/$USER
STATE=$P/anytok-runs/rawbpb-pool-ranking-20000
RUN=$P/anytok-runs/raw-bpb-20k
INPUT=$P/anytok-data/input
CODE=$P/anytok-runs/.raw-bpb-code-d875965262f96b29
ENV=$P/anytok-data/env/train

# As the training jobs ran: no login-shell Python or numerics overrides.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONOPTIMIZE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE LD_PRELOAD
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$ENV/hyak-libpython:${LD_LIBRARY_PATH:-}"

date -u
nvidia-smi -L
cd "$STATE"
exec "$ENV/bin/python" -u "$STATE/code/score_rawbpb_pool_ranking.py" --shard "$SLURM_ARRAY_TASK_ID" \
  --state "$STATE" --input "$INPUT" --checkpoint "$RUN/step-020000" --code "$CODE" --step 20000
