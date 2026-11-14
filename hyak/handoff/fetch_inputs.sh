#!/usr/bin/env bash
# hyak/handoff/fetch_inputs.sh -- fetch Qingyun's input bundle for the raw-BPB run into $ANYTOK_ROOT/input.
#
# Usage (on the login node: it needs internet for the download):
#   bash hyak/handoff/fetch_inputs.sh [--dry-run] [--force] [--remove-archive]
#                                     [--no-download | --download-only] [--from-dir DIR] [--reverify]
#
#   --dry-run          print what would be done; no network, nothing written
#   --force            an existing $ANYTOK_ROOT/input that does not verify is MOVED to input.replaced-<UTC>
#                      (never deleted); without it such a tree is refused (exit 2)
#   --remove-archive   delete the 9.3 GB archive after the full verification (default: keep it)
#   --no-download      use the archive already in $ANYTOK_ROOT/handoff (extract and verify only; no internet)
#   --download-only    download and check the archive; do not extract
#   --from-dir DIR     copy the three files from DIR on the cluster instead of downloading them
#   --reverify         hash every file of an already verified tree again
#
# What it does (python -m hyak.handoff.fetch, with the train environment's huggingface_hub, token=False):
#   1. reads ANYTOK_ROOT from hyak/config.env; a verified tree ($ANYTOK_ROOT/handoff/inputs-verified.json
#      still matches every file) is a no-op;
#   2. downloads anytok-training-inputs-v1.tar.gz (9,314,144,730 bytes), its .sha256 and bundle-receipt.json
#      from the public dataset cminst/anytok-training-inputs at the revision pinned in
#      docs/raw-bpb-handoff/bundle-reference.json into $ANYTOK_ROOT/handoff/ (resumable: just rerun);
#   3. checks the archive's size and SHA-256 against that repository pin (authoritative) and the downloaded
#      .sha256 and receipt against it;
#   4. extracts into $ANYTOK_ROOT (input/ with 1,062 files, 13.4 GB, and BUNDLE_MANIFEST.json) through a staging
#      directory, so an interrupted extraction never leaves a half tree;
#   5. runs python scripts/verify_training_inputs.py --root $ANYTOK_ROOT and requires 1,062 files, 1,000
#      tokenizers and 23 buckets, and writes $ANYTOK_ROOT/handoff/inputs-verified.json (submit.sh refuses
#      smoke-rawbpb and train-rawbpb without it).
#
# Space: 9.3 GB for the archive plus 13.4 GB extracted on the file system of ANYTOK_ROOT (project storage).
# Hashing and extracting take a few minutes of one CPU.  Extraction may be moved into a debug job if the
# login node reaps it: download with --download-only here, then in an interactive job (README 4.4)
#   bash hyak/handoff/fetch_inputs.sh --no-download
# Exit status: 0 done or nothing to do; 2 configuration, space, download or existing-tree problem (the message
# has a remedy); 4 a size, checksum or verification failure; 1 a crash.
# Never edits dotfiles; downloads nothing but the three pinned files.
set -euo pipefail

HERE=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=hyak/slurm/lib.sh
source "$HERE/../slurm/lib.sh"

for arg in "$@"; do
  case $arg in
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
  esac
done
DRY_RUN=0
for arg in "$@"; do [ "$arg" != --dry-run ] || DRY_RUN=1; done

hyak_load_config
PY=${HYAK_PYTHON:-$ANYTOK_ROOT/env/train/bin/python}
if [ ! -x "$PY" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    PY=$(command -v python3 || echo python3)
    hyak_warn "the train environment is missing ($ANYTOK_ROOT/env/train); the dry run uses $PY"
  else
    hyak_die "the train environment $ANYTOK_ROOT/env/train is missing (its huggingface_hub downloads the bundle). Remedy: bash hyak/env/setup_train_env.sh"
  fi
fi
unset HF_HUB_OFFLINE PYTHONHOME PYTHONSTARTUP
export HF_HOME="$ANYTOK_ROOT/hf-cache"
export PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HYAK_REPO"
exec "$PY" -m hyak.handoff.fetch "$@"
