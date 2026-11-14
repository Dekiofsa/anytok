#!/usr/bin/env bash
# hyak/env/setup_train_env.sh -- create or update the "train" conda environment (DESIGN.md 5.A, D10).
#
# Usage (on the login node: it only downloads and installs; about 3 GB to download, roughly twice that
# installed, plus the pip cache; needs internet):
#   bash hyak/env/setup_train_env.sh [--recreate] [--dry-run]
#
#   --recreate   delete $ANYTOK_ROOT/env/train first (needed when the torch build must change)
#   --dry-run    print what would be done; change nothing (no module/conda/pip calls)
#
# Result: $ANYTOK_ROOT/env/train with Python 3.12 and exactly the six pins of the Modal training
# image (scripts/modal_anytok_pool_20k.py:37-42; stock tokenizers, never the patched pool wheel)
#   torch==2.15.0 transformers==5.15.1 tokenizers==0.22.2 numpy==2.3.5 safetensors==0.8.0 accelerate==1.14.0
# plus huggingface_hub==1.33.0, which makes the set equal to docs/raw-bpb-handoff/requirements.txt (the
# raw-BPB handoff run, README section 0; hyak/handoff/fetch_inputs.sh downloads the bundle with it),
# plus pytest==9.2.0 for the test suite, a "pip check", and "pip freeze --all" in
# $ANYTOK_ROOT/env/train.freeze.txt.  Idempotent: re-running with the same settings installs
# nothing new and rewrites the same freeze file; a changed freeze is reported and the previous
# record is kept as train.freeze.<UTC>.txt.  pip ignores every pip configuration file and PIP_*
# variable of the login shell (an extra index could replace the pinned torch by another build),
# and its temporary files go to $ANYTOK_ROOT/env/.tmp (removed at the end), not to the login
# node's /tmp.
#
# Which torch build -- HYAK_TORCH_INDEX_URL in hyak/config.env (facts checked 2026-11-13 on
# https://pypi.org/pypi/torch/2.15.0/json and https://download.pytorch.org/whl/torch/):
#   * torch==2.15.0 for cp312 Linux x86-64 exists ONLY as CUDA 13 builds:
#       - PyPI default wheel (HYAK_TORCH_INDEX_URL empty): CUDA 13.2 runtime (depends on
#         cuda-toolkit==13.2.2, nvidia-cudnn-cu13==9.26.0.51, nvidia-nccl-cu13==2.30.7).  This is
#         what the Modal images installed ("uv_pip_install('torch==2.15.0', ...)"), so it is the
#         faithful choice and the default.
#       - https://download.pytorch.org/whl/cu132 -> torch 2.15.0+cu132 (same CUDA 13.2, PyTorch-hosted)
#       - https://download.pytorch.org/whl/cu134 -> torch 2.15.0+cu134 (CUDA 13.4)
#     The cu126/cu128/cu129/cu130 indexes stop at torch 2.13/2.14: there is no CUDA 12 build of 2.15.0.
#   * Driver rule (CUDA release notes, minor version compatibility): CUDA 13.x runs on NVIDIA
#     drivers >= 580; the native branch of 13.2 is R595, of 13.4 R615.
#       driver >= 580  -> keep HYAK_TORCH_INDEX_URL empty (default wheel; 580-594 run it under
#                         minor version compatibility, which is fine for the H200's sm_90 code).
#       driver <  580  -> no torch 2.15.0 CUDA build can run; an index URL does not help.  Ask
#                         Hyak support for a newer driver or the CUDA forward-compatibility libraries
#                         (cuda-compat-13-2; Data Center GPUs such as the H200).  A different torch
#                         version is a material deviation and needs the user's approval.
#     Set HYAK_TORCH_INDEX_URL only to cu132/cu134 for a reason (e.g. PyPI unreachable); changing
#     it after a run has started changes the environment of that run.
#   * The Tillicum modules cuda/12.9.1 and cuda/13.0.0 are NOT needed: the wheel bundles its CUDA
#     libraries.  Do not add a cuda module to HYAK_MODULES (its libraries could shadow torch's).
#   * python -m hyak.env.check_env (00_check_env.sbatch) reports the node's driver and applies
#     this rule on a GPU node.
#
# Caches go to $ANYTOK_ROOT/env/.pip-cache and .conda-pkgs (the 10 GB home quota is too small).
# Never edits ~/.bashrc, ~/.condarc or any other dotfile.
set -euo pipefail

HERE=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=hyak/slurm/lib.sh
source "$HERE/../slurm/lib.sh"

PYTHON_VERSION=3.12
TORCH_VERSION=2.15.0
PINS=(torch==2.15.0 transformers==5.15.1 tokenizers==0.22.2 numpy==2.3.5 safetensors==0.8.0 accelerate==1.14.0)
# The seventh line of docs/raw-bpb-handoff/requirements.txt (PINS + HANDOFF_PINS equal that file; tested).
HANDOFF_PINS=(huggingface_hub==1.33.0)
TEST_PINS=(pytest==9.2.0)
PYPI=https://pypi.org/simple
KNOWN_TORCH_LABELS="cu132 cu134"

RECREATE=0
DRY_RUN=0
while [ $# -gt 0 ]; do
  case $1 in
    --recreate) RECREATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) hyak_die "Unknown option '$1' (see --help)" ;;
  esac
done

say() { printf '[setup_train_env] %s\n' "$*"; }
run() {
  printf '[setup_train_env] +'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

hyak_load_config
ENV_ROOT="$ANYTOK_ROOT/env"
PREFIX="$ENV_ROOT/train"
PY="$PREFIX/bin/python"
FREEZE="$ENV_ROOT/train.freeze.txt"
INDEX_URL=${HYAK_TORCH_INDEX_URL:-}
INDEX_URL=${INDEX_URL%/}
LABEL=""
if [ -n "$INDEX_URL" ]; then
  LABEL=${INDEX_URL##*/}
  _hyak_in_list "$LABEL" "$KNOWN_TORCH_LABELS" || hyak_die "HYAK_TORCH_INDEX_URL=$INDEX_URL: torch $TORCH_VERSION has CUDA builds only for: $KNOWN_TORCH_LABELS (and the PyPI default, CUDA 13.2, with an empty HYAK_TORCH_INDEX_URL). There is no CUDA 12 or CPU option for training."
fi
EXPECTED_TORCH=$TORCH_VERSION${LABEL:++$LABEL}

say "repository:   $HYAK_REPO"
say "environment:  $PREFIX (Python $PYTHON_VERSION)"
say "packages:     ${PINS[*]} ${HANDOFF_PINS[*]} ${TEST_PINS[*]}"
if [ -n "$INDEX_URL" ]; then
  say "torch build:  torch==$EXPECTED_TORCH from $INDEX_URL"
else
  say "torch build:  torch==$TORCH_VERSION from PyPI (CUDA 13.2, as the Modal images)"
fi
say "freeze file:  $FREEZE"
[ "$DRY_RUN" = 0 ] || say "dry run: nothing will be changed"

export CONDA_PKGS_DIRS="$ENV_ROOT/.conda-pkgs"
# No pip configuration file and no PIP_* variable of the login shell: --index-url on the command line does
# not remove an extra index, and a PyTorch-hosted local build (2.15.0+cpu ...) would win over the wheel.
hyak_isolate_pip
export PIP_CACHE_DIR="$ENV_ROOT/.pip-cache"
# pip must not treat packages in ~/.local as installed, and nothing from PYTHONPATH may leak in.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

trap hyak_private_tmp_cleanup EXIT
if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$ENV_ROOT" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
  # pip keeps every wheel it downloads (torch alone is 539 MB, about 3 GB in all) in its temporary
  # directory until it exits: not in the login node's /tmp.
  hyak_private_tmp "$ENV_ROOT/.tmp"
  hyak_modules
  command -v conda >/dev/null 2>&1 || hyak_die "conda not found after loading modules '${HYAK_MODULES:-}'. Remedy: set HYAK_MODULES=conda in hyak/config.env (Tillicum: module load conda)"
fi

# --- 1. the conda prefix ------------------------------------------------------------------------
if [ "$RECREATE" = 1 ] && [ -e "$PREFIX" ]; then
  [ -d "$PREFIX/conda-meta" ] || hyak_die "$PREFIX exists but is not a conda environment; refusing to delete it. Remove it yourself."
  for run_dir in "$ANYTOK_RUNS/uniform" "$ANYTOK_RUNS/rl"; do
    if [ -d "$run_dir" ] && [ -n "$(ls -A "$run_dir" 2>/dev/null)" ]; then
      hyak_warn "$run_dir already holds a run: recreating the environment changes the software of that run (record it in the run notes)"
    fi
  done
  say "--recreate: removing $PREFIX"
  run conda remove --yes --all --prefix "$PREFIX"
  if [ "$DRY_RUN" = 0 ] && [ -e "$PREFIX" ]; then run rm -rf -- "$PREFIX"; fi
fi
if [ -x "$PY" ]; then
  have=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [ "$have" = "$PYTHON_VERSION" ] || hyak_die "$PREFIX has Python $have, need $PYTHON_VERSION. Remedy: bash $HERE/setup_train_env.sh --recreate"
  say "environment exists (Python $have)"
elif [ -e "$PREFIX" ] && [ "$DRY_RUN" = 0 ]; then
  hyak_die "$PREFIX exists but has no bin/python (interrupted creation?). Remedy: bash $HERE/setup_train_env.sh --recreate"
else
  run conda create --yes --prefix "$PREFIX" --override-channels --channel conda-forge "python=$PYTHON_VERSION" pip
fi

# --- 2. torch build guard -------------------------------------------------------------------------
installed_torch=""
if [ -x "$PY" ]; then
  installed_torch=$("$PY" -c 'import importlib.metadata as m
try:
    print(m.version("torch"))
except m.PackageNotFoundError:
    pass')
fi
if [ -n "$installed_torch" ] && [ "$installed_torch" != "$EXPECTED_TORCH" ]; then
  hyak_die "installed torch is $installed_torch but this configuration asks for $EXPECTED_TORCH (HYAK_TORCH_INDEX_URL='${INDEX_URL}'). pip would keep the old build. Remedy: bash $HERE/setup_train_env.sh --recreate"
fi

# --- 3. packages ---------------------------------------------------------------------------------------
if [ -n "$INDEX_URL" ]; then
  # The exact local version selects this build even with PyPI as the extra index for dependencies.
  run "$PY" -m pip install --index-url "$INDEX_URL" --extra-index-url "$PYPI" "torch==$EXPECTED_TORCH"
fi
# With torch==2.15.0+cuXYZ installed, "torch==2.15.0" is already satisfied (PEP 440 ignores local labels).
run "$PY" -m pip install --index-url "$PYPI" "${PINS[@]}" "${HANDOFF_PINS[@]}" "${TEST_PINS[@]}"
run "$PY" -m pip check

if [ "$DRY_RUN" = 1 ]; then
  say "dry run complete"
  exit 0
fi

# --- 4. verification (hyak/env/check_env.py, the checks that need no GPU) ---------------------------
final_torch=$("$PY" -c 'import importlib.metadata as m; print(m.version("torch"))')
[ "$final_torch" = "$EXPECTED_TORCH" ] || hyak_die "torch $final_torch was installed, expected $EXPECTED_TORCH. A pip configuration, PIP_EXTRA_INDEX_URL or PIP_FIND_LINKS can cause this; this script ignores them, so look at the pip lines above for the index torch came from. Remedy: bash $HERE/setup_train_env.sh --recreate"
say "torch $final_torch, CUDA runtime $("$PY" -c 'import torch; print(torch.version.cuda)')"
if [ ! -f "$PREFIX/lib/libpython3.12.so.1.0" ]; then
  hyak_warn "$PREFIX/lib/libpython3.12.so.1.0 is missing; the committed librust_utils.so links against it. Remedy: bash $HERE/build_rust_utils.sh (extension-module build)"
fi
# A directory that holds only the link to libpython: hyak_activate puts it (and nothing else of the
# environment) on LD_LIBRARY_PATH, so conda's own libz/libstdc++ never shadow the system's.
_hyak_libpython_dir "$PREFIX" >/dev/null || hyak_warn "could not create $PREFIX/$HYAK_LIBPYTHON_DIRNAME (hyak_activate will try again in the job)"
PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$PY" -m hyak.env.check_env --only python,packages \
  --no-gpu --no-network --no-status \
  || hyak_die "the new environment failed hyak.env.check_env (see above)"
PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$PY" -m hyak.handoff.run requirements >/dev/null \
  || hyak_die "the new environment differs from docs/raw-bpb-handoff/requirements.txt (see above)"

# --- 5. freeze ------------------------------------------------------------------------------------------
tmp="$FREEZE.tmp.$$"
"$PY" -m pip freeze --all > "$tmp"
if [ -f "$FREEZE" ] && ! cmp -s "$tmp" "$FREEZE"; then
  old="${FREEZE%.txt}.$(date -u +%Y%m%dT%H%M%SZ).txt"
  hyak_warn "the environment changed since the last freeze (diff below); runs started before used the old one, whose record is kept as $old"
  diff "$FREEZE" "$tmp" || true
  mv -f "$FREEZE" "$old"
fi
mv -f "$tmp" "$FREEZE"
say "wrote $FREEZE ($(wc -l < "$FREEZE" | tr -d ' ') packages)"
say "done. Next: python -m hyak.code.stage_code, then submit check-env (hyak/slurm/submit.sh check-env)"
