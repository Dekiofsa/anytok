#!/usr/bin/env bash
# hyak/env/setup_data_env.sh -- create or update the "data" conda environment and the Rust toolchain
# (DESIGN.md 5.A, D10).  Builds nothing.
#
# Usage (on the login node: downloads only; needs internet; idempotent, safe to re-run):
#   bash hyak/env/setup_data_env.sh [--recreate] [--skip-rust] [--dry-run]
#   bash hyak/env/setup_data_env.sh --freeze-only
#
#   --recreate     delete $ANYTOK_ROOT/env/data first
#   --skip-rust    do not install or check the Rust toolchain
#   --dry-run      print what would be done; change nothing (no module/conda/pip/download calls)
#   --freeze-only  only rewrite env/data.freeze.txt from the installed packages (the wheel job does
#                  this after it installed the patched tokenizers); needs no internet and no module
#
# Result:
#   * $ANYTOK_ROOT/env/data: Python 3.12 with the package set the corpus build was validated with
#     (it reproduces every record of the original corpus, including the exact size of all 46 original
#     files): datasets==5.2.0 pyarrow==26.0.1 huggingface-hub==2.4.0 hf-xet==1.8.0 fsspec==2026.9.0
#     requests==2.34.2 urllib3==2.8.0, plus "maturin>=1.4,<2.0".  (The original image,
#     scripts/modal_prepare_corpus.py:63-65, asked for ranges and never recorded the versions it
#     resolved, so a range is no more faithful than this pin.)  hyak.code.pinned.DATA_PINS lists
#     the same set; hyak.corpus.build compares it before building.  "pip freeze --all" goes to
#     $ANYTOK_ROOT/env/data.freeze.txt; when the packages change, the previous file is kept as
#     data.freeze.<UTC>.txt.  The patched tokenizers wheel is installed into this environment
#     later by hyak/pool/build_wheel.sh install -- never into the train environment -- with
#     --no-deps: it requires huggingface-hub<2.0, so "pip check" then reports that one known
#     conflict, which this script accepts (any other conflict stops it).  pip ignores every pip
#     configuration file and PIP_* variable here, and its temporary files go to
#     $ANYTOK_ROOT/env/.tmp (removed at the end), not to the login node's /tmp.
#   * rustup with toolchain 1.96.0 (minimal profile; the compiler recorded in the committed
#     librust_utils.so) in RUSTUP_HOME=$ANYTOK_ROOT/toolchain/rustup and
#     CARGO_HOME=$ANYTOK_ROOT/toolchain/cargo, installed with --no-modify-path (no dotfile is
#     touched).  rustup-init is downloaded from static.rust-lang.org and checked against the
#     SHA-256 published next to it.  Versions go to $ANYTOK_ROOT/toolchain/versions.txt.
#   To use the toolchain in a shell:
#     export RUSTUP_HOME=$ANYTOK_ROOT/toolchain/rustup CARGO_HOME=$ANYTOK_ROOT/toolchain/cargo
#     export PATH="$CARGO_HOME/bin:$PATH"
set -euo pipefail

HERE=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=hyak/slurm/lib.sh
source "$HERE/../slurm/lib.sh"

PYTHON_VERSION=3.12
# The validated set; must equal hyak.code.pinned.DATA_REQUIREMENTS (tested).
DATA_PINS=(datasets==5.2.0 pyarrow==26.0.1 huggingface-hub==2.4.0 hf-xet==1.8.0 fsspec==2026.9.0 requests==2.34.2 urllib3==2.8.0)
REQUIREMENTS=("${DATA_PINS[@]}" "maturin>=1.4,<2.0")
RUST_TOOLCHAIN=1.96.0
RUSTUP_DIST=https://static.rust-lang.org/rustup/dist
PYPI=https://pypi.org/simple

RECREATE=0
DRY_RUN=0
SKIP_RUST=0
FREEZE_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --recreate) RECREATE=1; shift ;;
    --freeze-only) FREEZE_ONLY=1; shift ;;
    --skip-rust) SKIP_RUST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) hyak_die "Unknown option '$1' (see --help)" ;;
  esac
done

say() { printf '[setup_data_env] %s\n' "$*"; }
run() {
  printf '[setup_data_env] +'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

hyak_load_config
ENV_ROOT="$ANYTOK_ROOT/env"
PREFIX="$ENV_ROOT/data"
PY="$PREFIX/bin/python"
FREEZE="$ENV_ROOT/data.freeze.txt"
TOOLCHAIN="$ANYTOK_ROOT/toolchain"
export RUSTUP_HOME="$TOOLCHAIN/rustup"
export CARGO_HOME="$TOOLCHAIN/cargo"

# write_freeze: pip freeze --all into $FREEZE; a changed package set keeps the previous record as
# <name>.freeze.<UTC>.txt next to it (the old file used to be overwritten).
write_freeze() {
  local tmp="$FREEZE.tmp.$$" stamp old
  "$PY" -m pip freeze --all > "$tmp"
  if [ -f "$FREEZE" ] && ! cmp -s "$tmp" "$FREEZE"; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    old="${FREEZE%.txt}.$stamp.txt"
    hyak_warn "the data environment changed since the last freeze (diff below); the previous record is kept as $old"
    diff "$FREEZE" "$tmp" || true
    mv -f "$FREEZE" "$old"
  fi
  mv -f "$tmp" "$FREEZE"
  say "wrote $FREEZE ($(wc -l < "$FREEZE" | tr -d ' ') packages)"
}

if [ "$FREEZE_ONLY" = 1 ]; then
  [ -x "$PY" ] || hyak_die "$PY is missing; nothing to freeze. Remedy: bash $HERE/setup_data_env.sh"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH PYTHONHOME
  hyak_isolate_pip
  write_freeze
  exit 0
fi

say "repository:   $HYAK_REPO"
say "environment:  $PREFIX (Python $PYTHON_VERSION)"
say "packages:     ${REQUIREMENTS[*]}"
if [ "$SKIP_RUST" = 0 ]; then
  say "rust:         toolchain $RUST_TOOLCHAIN (minimal) in RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME"
fi
[ "$DRY_RUN" = 0 ] || say "dry run: nothing will be changed"

export CONDA_PKGS_DIRS="$ENV_ROOT/.conda-pkgs"
# No pip configuration file, no PIP_* variable of the login shell (an extra index would change which
# build of a package pip picks even with --index-url on the command line).
hyak_isolate_pip
export PIP_CACHE_DIR="$ENV_ROOT/.pip-cache"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
RUSTUP_WORK=""
cleanup() {
  [ -z "$RUSTUP_WORK" ] || rm -rf -- "$RUSTUP_WORK"
  hyak_private_tmp_cleanup
}
trap cleanup EXIT

if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$ENV_ROOT" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
  # pip keeps every wheel it downloads in its temporary directory until it exits: not in /tmp.
  hyak_private_tmp "$ENV_ROOT/.tmp"
  hyak_modules
  command -v conda >/dev/null 2>&1 || hyak_die "conda not found after loading modules '${HYAK_MODULES:-}'. Remedy: set HYAK_MODULES=conda in hyak/config.env"
fi

# --- 1. the conda prefix ----------------------------------------------------------------------------
if [ "$RECREATE" = 1 ] && [ -e "$PREFIX" ]; then
  [ -d "$PREFIX/conda-meta" ] || hyak_die "$PREFIX exists but is not a conda environment; refusing to delete it. Remove it yourself."
  say "--recreate: removing $PREFIX (the patched tokenizers wheel must be reinstalled afterwards: hyak/pool/build_wheel.sh install)"
  run conda remove --yes --all --prefix "$PREFIX"
  if [ "$DRY_RUN" = 0 ] && [ -e "$PREFIX" ]; then run rm -rf -- "$PREFIX"; fi
fi
if [ -x "$PY" ]; then
  have=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [ "$have" = "$PYTHON_VERSION" ] || hyak_die "$PREFIX has Python $have, need $PYTHON_VERSION. Remedy: bash $HERE/setup_data_env.sh --recreate"
  say "environment exists (Python $have)"
elif [ -e "$PREFIX" ] && [ "$DRY_RUN" = 0 ]; then
  hyak_die "$PREFIX exists but has no bin/python (interrupted creation?). Remedy: bash $HERE/setup_data_env.sh --recreate"
else
  run conda create --yes --prefix "$PREFIX" --override-channels --channel conda-forge "python=$PYTHON_VERSION" pip
fi

# --- 2. packages ------------------------------------------------------------------------------------------
run "$PY" -m pip install --index-url "$PYPI" "${REQUIREMENTS[@]}"
# pip check, accepting exactly one known conflict: the patched tokenizers wheel (installed later with
# --no-deps by hyak/pool/build_wheel.sh) requires huggingface-hub<2.0, the validated set has 2.4.0.
# Without this the script stopped on every re-run after the wheel was installed.
pip_check() {
  local out rc=0 rest
  printf '[setup_data_env] + %q -m pip check\n' "$PY"
  [ "$DRY_RUN" = 0 ] || return 0
  out=$("$PY" -m pip check 2>&1) || rc=$?
  [ "$rc" != 0 ] || { printf '%s\n' "$out"; return 0; }
  rest=$(printf '%s\n' "$out" | grep -v -E '^tokenizers [^ ]+ has requirement huggingface[-_]hub[^ ]*, but you have huggingface[-_]hub [^ ]+\.$' || true)
  if [ -n "$rest" ]; then
    printf '%s\n' "$out" >&2
    hyak_die "pip check found conflicts in $PREFIX (listed above); the only accepted one is the patched tokenizers wheel's huggingface-hub<2.0 requirement"
  fi
  say "pip check: only the accepted tokenizers / huggingface-hub conflict of the patched wheel remains"
}
pip_check

# --- 3. Rust toolchain --------------------------------------------------------------------------------------
host_triple() {
  local arch os
  arch=$(uname -m)
  os=$(uname -s)
  case "$os/$arch" in
    Linux/x86_64) echo x86_64-unknown-linux-gnu ;;
    Linux/aarch64) echo aarch64-unknown-linux-gnu ;;
    Darwin/arm64) echo aarch64-apple-darwin ;;
    Darwin/x86_64) echo x86_64-apple-darwin ;;
    *) hyak_die "no rustup-init for $os/$arch" ;;
  esac
}
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
if [ "$SKIP_RUST" = 0 ]; then
  RUSTUP="$CARGO_HOME/bin/rustup"
  if [ -x "$RUSTUP" ]; then
    say "rustup exists: $("$RUSTUP" --version 2>/dev/null | head -1)"
  else
    triple=$(host_triple)
    url="$RUSTUP_DIST/$triple/rustup-init"
    say "installing rustup from $url (checksum verified)"
    if [ "$DRY_RUN" = 0 ]; then
      command -v curl >/dev/null 2>&1 || hyak_die "curl is required to download rustup-init"
      mkdir -p "$TOOLCHAIN"
      work=$(mktemp -d "$TOOLCHAIN/.rustup-download.XXXXXX")
      RUSTUP_WORK=$work            # removed by the EXIT trap on every path, including a failed download or checksum
      curl --proto '=https' --tlsv1.2 -sSfL --retry 3 -o "$work/rustup-init" "$url" \
        || hyak_die "download of $url failed (internet access is needed: run this on the login node)"
      curl --proto '=https' --tlsv1.2 -sSfL --retry 3 -o "$work/rustup-init.sha256" "$url.sha256" \
        || hyak_die "download of $url.sha256 failed"
      want=$(cut -d' ' -f1 < "$work/rustup-init.sha256")
      got=$(sha256_of "$work/rustup-init")
      [ -n "$want" ] && [ "$want" = "$got" ] || hyak_die "rustup-init checksum mismatch (got $got, published $want); not running it"
      chmod +x "$work/rustup-init"
      run "$work/rustup-init" -y --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
      rm -rf -- "$work"
      RUSTUP_WORK=""
    fi
  fi
  run "$RUSTUP" toolchain install "$RUST_TOOLCHAIN" --profile minimal
  if [ "$DRY_RUN" = 0 ]; then
    rustc_version=$("$RUSTUP" run "$RUST_TOOLCHAIN" rustc --version)
    case $rustc_version in
      "rustc $RUST_TOOLCHAIN "*) say "toolchain OK: $rustc_version" ;;
      *) hyak_die "rustc reports '$rustc_version', expected rustc $RUST_TOOLCHAIN" ;;
    esac
    {
      echo "recorded_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "rustup=$("$RUSTUP" --version 2>/dev/null | head -1)"
      echo "rustc=$("$RUSTUP" run "$RUST_TOOLCHAIN" rustc -vV | tr '\n' ';')"
      echo "cargo=$("$RUSTUP" run "$RUST_TOOLCHAIN" cargo --version)"
    } > "$TOOLCHAIN/versions.txt.tmp.$$"
    mv -f "$TOOLCHAIN/versions.txt.tmp.$$" "$TOOLCHAIN/versions.txt"
    say "wrote $TOOLCHAIN/versions.txt"
  fi
fi

if [ "$DRY_RUN" = 1 ]; then
  say "dry run complete"
  exit 0
fi

# --- 4. verification and freeze ----------------------------------------------------------------------------
# A directory that holds only the link to libpython3.12.so.1.0: hyak_activate puts it (and nothing else of
# the environment) on LD_LIBRARY_PATH, so conda's own libz/libstdc++ never shadow the system's.
_hyak_libpython_dir "$PREFIX" >/dev/null || hyak_warn "could not create $PREFIX/$HYAK_LIBPYTHON_DIRNAME (hyak_activate will try again in the job)"
"$PY" - "${DATA_PINS[@]}" <<'EOF' || hyak_die "the data environment failed its import check (the installed versions differ from the validated set: rerun this script, or --recreate)"
import importlib.metadata as m, sys
assert sys.version_info[:2] == (3, 12), sys.version
import datasets, requests, huggingface_hub  # noqa: F401
for pin in sys.argv[1:]:
    name, _, want = pin.partition("==")
    got = m.version(name)
    print(f"[setup_data_env] {name} {got}")
    assert got == want, f"{name} {got} is installed, the validated set has {want}"
print(f"[setup_data_env] maturin {m.version('maturin')}")
EOF
write_freeze
say "done. Next: hyak/pool/build_wheel.sh fetch (login node) and the corpus build"
