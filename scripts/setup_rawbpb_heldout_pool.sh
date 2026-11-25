#!/usr/bin/env bash
# Prepare Qingyun's held-out full-pool eval (948787e) for all 42 raw-BPB saves on the Tillicum login node.
#
# Usage: bash setup_rawbpb_heldout_pool.sh STAGE [phase]
#   STAGE  project-storage folder holding code/scripts/ (this repository's verified copies)
#   phase  inputs (default): environment, his archive, restore, prepare, model and pool, the smoke checkpoint
#          panel: panel audit, check-audit, features
#          checkpoints: the other 41 checkpoints (after the smoke test passes)
#          preflight: every staged file against its manifest (after panel and checkpoints)
#
# inputs and checkpoints are downloads and run on the login node. panel and preflight are heavy (the
# audit loads all 1,000 tokenizers in one process; the preflight hashes 362 GB), and Tillicum's login
# nodes kill such processes, so they run as debug-QoS jobs on a compute node with default resources.
#
# The commands are his README's (reports/heldout-pool-sweep/README.md at 948787e) plus the two steps the
# adaptation adds: prepare (plans/rawbpb-heldout-pool.md) and check-audit after his audit. Every step stops
# the script on failure. Downloads are anonymous and go straight from Hugging Face to project storage.
# No Slurm job is submitted here.
set -euo pipefail
stage=${1:?Pass the stage folder}
phase=${2:-inputs}
code="$stage/code/scripts"
venv="$stage/venv"
python="$venv/bin/python"
reference="$stage/reference"
work="$stage/raw-bpb"
prefix=paused-20261122T143018Z
smoke=raw-bpb-020000

# Keep every cache and temporary file on project storage (home is 10 GB).
export HF_HOME="$stage/hf-home" UV_CACHE_DIR="$stage/uv-cache" UV_PYTHON_INSTALL_DIR="$stage/python"
export TMPDIR="$stage/tmp" HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR" "$stage/logs"

step() { printf '\n=== %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
mark() { printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$stage/logs/$1"; }

case "$phase" in
inputs)
    step "environment (Python 3.12, his pinned packages)"
    if [ ! -x "$python" ]; then
        uv venv --python 3.12 "$venv"
    fi
    uv pip install --python "$python" \
        --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match \
        torch==2.15.0+cu132 transformers==5.15.1 tokenizers==0.22.2 numpy==2.3.5 \
        safetensors==0.8.0 huggingface-hub==1.33.0 requests
    "$python" - <<'EOF'
import importlib.metadata as metadata, sys
pins = {'torch': '2.15.0+cu132', 'transformers': '5.15.1', 'tokenizers': '0.22.2', 'numpy': '2.3.5',
        'safetensors': '0.8.0', 'huggingface-hub': '1.33.0'}
found = {name: metadata.version(name) for name in pins}
print('python', sys.version.split()[0], found)
if sys.version_info[:2] != (3, 12) or found != pins:
    raise SystemExit('The environment differs from the pinned one')
EOF
    mark environment-ready

    step "his archive (anonymous, pinned revision)"
    "$venv/bin/hf" download qingy2024/anytok-rift-eval-resume \
        "$prefix/sources.json" "$prefix/rift-sweep-resume.tar.gz" \
        --repo-type dataset --revision 38176d7d654b33f513dcc7e4f3f19e6bfc142d85 --local-dir "$stage/hf"

    step "restore and verify his workspace (his script)"
    if [ ! -d "$reference" ]; then
        "$python" "$code/restore_rift_pool_archive.py" \
            "$stage/hf/$prefix/rift-sweep-resume.tar.gz" "$stage/hf/$prefix/sources.json" "$reference"
    fi

    step "prepare the raw-BPB workspace (checkpoint list only)"
    if [ ! -d "$work" ]; then
        "$python" "$code/prepare_rawbpb_heldout_pool.py" prepare --reference "$reference" --workspace "$work"
    fi

    step "stage the canonical model and the 1,000 tokenizers (his script)"
    "$python" "$code/stage_rift_resume_inputs.py" "$work"

    step "stage the smoke checkpoint $smoke (his script)"
    "$python" "$code/stage_rift_resume_inputs.py" "$work" "$smoke"
    mark smoke-checkpoint-staged
    ;;
panel)
    step "panel audit for this plan (his scorer)"
    "$python" "$work/run_rift_pool_sweep.py" audit

    step "the re-run audit must equal his in every field but the plan hash"
    "$python" "$code/prepare_rawbpb_heldout_pool.py" check-audit --reference "$reference" --workspace "$work"

    step "tokenizer features (his scorer)"
    "$python" "$work/run_rift_pool_sweep.py" features
    mark features-ready
    ;;
checkpoints)
    step "stage the other checkpoints (his script)"
    while read -r label || [ -n "$label" ]; do
        "$python" "$code/stage_rift_resume_inputs.py" "$work" "$label"
    done < "$work/checkpoints.txt"
    mark checkpoints-staged
    ;;
preflight)
    step "preflight: every staged file against its manifest (his scorer)"
    "$python" "$work/run_rift_pool_sweep.py" verify-launch --checkpoint-list "$work/checkpoints.txt" \
        | tee "$stage/logs/preflight.json"
    mark preflight-passed
    ;;
*)
    printf 'Expected inputs, panel, checkpoints or preflight, got %s\n' "$phase" >&2
    exit 2
    ;;
esac
step "phase $phase complete"
