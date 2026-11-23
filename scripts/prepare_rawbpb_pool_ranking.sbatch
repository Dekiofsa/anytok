#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=anytok-rank20k-prep
#SBATCH --output=/gpfs/projects/stf/%u/anytok-runs/rawbpb-pool-ranking-20000/slurm-prep-%j.log
#
# Prepare the frozen step-20,000 full-pool ranking, then smoke-test the scorer on one tokenizer:
#   sbatch prepare_rawbpb_pool_ranking.sbatch fresh       # the next 32 training batches after step 20,000
#   sbatch prepare_rawbpb_pool_ranking.sbatch published   # the published panel of the step-6,500 ranking
# Before sbatch (Slurm opens the --output log in the state folder before this script runs; if the
# folder is missing, the job fails without writing any log):
#   mkdir -p /gpfs/projects/stf/$USER/anytok-runs/rawbpb-pool-ranking-20000/code
#   stage in code/: prepare_rawbpb_pool_ranking.py, score_rawbpb_pool_ranking.py,
#   rawbpb_pool_ranking_status.py and an unchanged copy of package_training_inputs.py;
#   published mode only: first place the verified panel at
#   /gpfs/projects/stf/$USER/anytok-runs/rawbpb-pool-ranking-20000/published-panel/training-panel.json.gz
# The preparation is CPU work; it runs in a 1-GPU job because Tillicum has no CPU-only jobs, and
# the smoke test uses the GPU. No CPU or memory reservations (project rule).

set -euo pipefail
MODE=${1:-}
case $MODE in
  fresh|published) ;;
  *) echo "usage: sbatch prepare_rawbpb_pool_ranking.sbatch fresh|published" >&2; exit 2 ;;
esac

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

prep=("$ENV/bin/python" -u "$STATE/code/prepare_rawbpb_pool_ranking.py" --state "$STATE" --run "$RUN"
      --input "$INPUT" --code "$CODE" --step 20000 --scorer "$STATE/code/score_rawbpb_pool_ranking.py"
      --panel "$MODE")
if [ "$MODE" = published ]; then
  prep+=(--published-panel "$STATE/published-panel/training-panel.json.gz")
fi
"${prep[@]}"

# Reached only after a successful preparation (set -e): score the first tokenizer of shard 0.
"$ENV/bin/python" -u "$STATE/code/score_rawbpb_pool_ranking.py" --shard 0 --max-tokenizers 1 \
  --state "$STATE" --input "$INPUT" --checkpoint "$RUN/step-020000" --code "$CODE" --step 20000
date -u
