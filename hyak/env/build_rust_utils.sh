#!/usr/bin/env bash
# hyak/env/build_rust_utils.sh -- fallback rebuild of librust_utils.so (DESIGN.md 5.A, D10).
#
# Only needed when check_env reports that the committed library
# (runtime/anytok_pool_20k/rust_utils/target/release/librust_utils.so: Linux x86-64, glibc >= 2.34,
# linked against libpython3.12.so.1.0) cannot be imported.  Rebuilding changes no behaviour the
# trainer hashes (the library is not part of run_config.json), but it changes the code manifest:
# do it before any run starts.
#
# Usage:
#   bash hyak/env/build_rust_utils.sh [fetch] [build] [check] [install] [--module M]... [--plain] [--dry-run]
#
#   (no phase)  all four, in order
#   fetch       copy the pinned crate sources to $ANYTOK_ROOT/build/rust_utils/crate and
#               "cargo fetch --locked" the crates (needs internet: login node)
#   build       "cargo +1.96.0 build --release --locked --offline" with CARGO_TARGET_DIR outside the
#               repository, first with --features defaut (the manifest's misspelled feature that
#               enables pyo3/extension-module, so the library needs no libpython), falling back to
#               the plain build (the committed library's linkage) if that fails; writes
#               build/rust_utils/out/{librust_utils.so,provenance.json}
#   check       python -m hyak.env.check_env --only rust_library --rust-library <built .so> in the
#               train environment (imports it exactly as the trainer does and draws a tokenizer)
#   install     python -m hyak.code.stage_code --rust-library <built .so> --force: puts it into both
#               staged code trees and the code manifest (refused while a run directory is non-empty)
#   --module M  load this Lmod module for the build (repeatable), e.g. --module gcc/13.4.0
#               (a C and C++ compiler; the onig_sys crate compiles C code).  Without --module the
#               script loads the module named by HYAK_GCC_MODULE when that is set, or gcc/13.4.0 when
#               no C compiler (cc or gcc) or no C++ compiler (c++ or g++) is on PATH
#   --plain     skip the extension-module attempt
#   --dry-run   print the commands; run nothing
#
# Where to run it: "fetch" needs internet and is light: run it on the login node.  "build" compiles
# about 110 crates in release mode: do NOT run it on the login node (heavy processes there are
# killed); run build, check and install inside an interactive GPU or CPU job on a compute node
# (the compute nodes need no internet for these phases, the crates were fetched), and pass
# --module gcc/13.4.0 there if you want a specific compiler (it is loaded by default when no C or C++
# compiler is on PATH).  The script warns when it is started
# for "build" on a Slurm login node.
#
# Needs: the Rust toolchain from setup_data_env.sh ($ANYTOK_ROOT/toolchain), the train
# environment's Python 3.12 (PYO3_PYTHON; pyo3 0.19.2 builds against 3.12, as the committed
# binary proves), a C compiler.  The repository is never written to.
set -euo pipefail

HERE=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=hyak/slurm/lib.sh
source "$HERE/../slurm/lib.sh"

TOOLCHAIN_VERSION=1.96.0
SRC_REL=runtime/anytok_pool_20k/rust_utils
SOURCES="Cargo.toml Cargo.lock pyproject.toml rust-toolchain src/lib.rs"

PHASES=""
MODULES=""
PLAIN=0
DRY_RUN=0
while [ $# -gt 0 ]; do
  case $1 in
    fetch|build|check|install) PHASES="$PHASES $1"; shift ;;
    --module) [ $# -ge 2 ] || hyak_die "--module needs a value"; MODULES="$MODULES $2"; shift 2 ;;
    --plain) PLAIN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) hyak_die "Unknown argument '$1' (see --help)" ;;
  esac
done
[ -n "$PHASES" ] || PHASES=" fetch build check install"

say() { printf '[build_rust_utils] %s\n' "$*"; }
run() {
  printf '[build_rust_utils] +'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

hyak_load_config
SRC="$HYAK_REPO/$SRC_REL"
BUILD="$ANYTOK_ROOT/build/rust_utils"
CRATE="$BUILD/crate"
TARGET="$BUILD/target"
OUT="$BUILD/out"
export RUSTUP_HOME="$ANYTOK_ROOT/toolchain/rustup"
export CARGO_HOME="$ANYTOK_ROOT/toolchain/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
TRAIN_PREFIX="$ANYTOK_ROOT/env/train"
TRAIN_PY="$TRAIN_PREFIX/bin/python"
CARGO="$CARGO_HOME/bin/cargo"

case $BUILD in
  "$HYAK_REPO"|"$HYAK_REPO"/*) hyak_die "ANYTOK_ROOT is inside the repository; the build tree must be outside it" ;;
esac

# libpython_path: the directory that holds only the link to libpython3.12 (what hyak_activate uses), else
# conda's lib directory for an environment that was set up before that directory existed.
libpython_path() {
  if [ -d "$TRAIN_PREFIX/$HYAK_LIBPYTHON_DIRNAME" ]; then printf '%s\n' "$TRAIN_PREFIX/$HYAK_LIBPYTHON_DIRNAME"
  else printf '%s\n' "$TRAIN_PREFIX/lib"
  fi
}

stdlib_python() {  # any Python >= 3.9 for the standard-library-only hyak modules
  if [ -x "$TRAIN_PY" ]; then printf '%s\n' "$TRAIN_PY"; else hyak_python; fi
}

need_toolchain() {
  [ "$DRY_RUN" = 1 ] || [ -x "$CARGO" ] || hyak_die "cargo not found at $CARGO. Remedy: bash $HERE/setup_data_env.sh (installs rustup $TOOLCHAIN_VERSION into $ANYTOK_ROOT/toolchain)"
}

phase_fetch() {
  say "fetch: verifying the pinned sources in $SRC"
  run env PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$(stdlib_python)" -m hyak.code.pinned --check --group rust-sources
  need_toolchain
  run rm -rf -- "$CRATE"
  run mkdir -p "$CRATE/src" "$OUT"
  for f in $SOURCES; do
    run cp "$SRC/$f" "$CRATE/$f"
  done
  run chmod -R u+w "$CRATE"
  run "$CARGO" "+$TOOLCHAIN_VERSION" fetch --locked --manifest-path "$CRATE/Cargo.toml"
}

phase_build() {
  local variant="" features="" lib="" m status
  if [ "$DRY_RUN" = 0 ] && [ -z "${SLURM_JOB_ID:-}" ] && command -v sbatch >/dev/null 2>&1; then
    hyak_warn "this looks like a Slurm login node: the build compiles about 110 crates and heavy processes there are killed. Run the build phase inside an interactive job on a compute node"
  fi
  [ "$DRY_RUN" = 1 ] || [ -f "$CRATE/Cargo.toml" ] || hyak_die "$CRATE is missing. Remedy: bash $HERE/build_rust_utils.sh fetch (login node)"
  if [ "$DRY_RUN" = 0 ]; then
    for f in $SOURCES; do
      cmp -s "$SRC/$f" "$CRATE/$f" || hyak_die "$CRATE/$f differs from the pinned $SRC_REL/$f. Remedy: rerun the fetch phase"
    done
    [ -x "$TRAIN_PY" ] || hyak_die "$TRAIN_PY is missing (PYO3_PYTHON must be the train interpreter). Remedy: bash $HERE/setup_train_env.sh"
    m=$("$TRAIN_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    [ "$m" = 3.12 ] || hyak_die "$TRAIN_PY is Python $m, need 3.12"
  fi
  need_toolchain
  if [ -n "$MODULES" ]; then
    # shellcheck disable=SC2086
    [ "$DRY_RUN" = 1 ] || hyak_load_modules "the --module option of this script" $MODULES
    say "loaded modules:$MODULES"
  fi
  # A C and C++ compiler come from the gcc module: load it when HYAK_GCC_MODULE names one, or when cc/gcc
  # or c++/g++ is missing (default gcc/13.4.0; choose another with HYAK_GCC_MODULE=<module>).
  if [ "$DRY_RUN" = 0 ]; then
    if gcc_module=$(hyak_gcc_module); then
      hyak_load_modules "HYAK_GCC_MODULE (module avail gcc lists them)" "$gcc_module"
      say "loaded the C and C++ compiler module $gcc_module"
    fi
    if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
      hyak_die "no C compiler (cc/gcc) on PATH even after the gcc module; onig_sys needs one. Remedy: rerun with --module <a gcc module> or HYAK_GCC_MODULE=<module>"
    fi
  fi
  export PYO3_PYTHON="$TRAIN_PY"
  # Python's own libpython for a plain (non extension-module) link and load.
  export LD_LIBRARY_PATH="$(libpython_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  status=1
  if [ "$PLAIN" = 0 ]; then
    say "build: trying --features defaut (pyo3/extension-module)"
    if run "$CARGO" "+$TOOLCHAIN_VERSION" build --release --locked --offline \
         --manifest-path "$CRATE/Cargo.toml" --features defaut; then
      status=0; variant=extension-module; features=defaut
    else
      hyak_warn "the extension-module build failed; falling back to the plain build (links libpython3.12 like the committed library)"
    fi
  fi
  if [ "$status" != 0 ]; then
    run "$CARGO" "+$TOOLCHAIN_VERSION" build --release --locked --offline --manifest-path "$CRATE/Cargo.toml" \
      || hyak_die "cargo build failed (see above)"
    variant=plain; features=""
  fi
  case $(uname -s) in
    Linux) lib="$TARGET/release/librust_utils.so" ;;
    Darwin) lib="$TARGET/release/librust_utils.dylib"
            hyak_warn "macOS build: usable for local tests only, never on the cluster" ;;
    *) hyak_die "unsupported build host $(uname -s)" ;;
  esac
  if [ "$DRY_RUN" = 1 ]; then
    say "dry run: would copy $lib to $OUT/librust_utils.so and write $OUT/provenance.json"
    return 0
  fi
  [ -f "$lib" ] || hyak_die "cargo did not produce $lib"
  mkdir -p "$OUT"
  cp "$lib" "$OUT/.librust_utils.so.tmp.$$"
  mv -f "$OUT/.librust_utils.so.tmp.$$" "$OUT/librust_utils.so"
  BUILD_VARIANT=$variant BUILD_FEATURES=$features BUILD_LIBRARY="$OUT/librust_utils.so" BUILD_CRATE="$CRATE" \
  BUILD_RUSTC="$("$CARGO_HOME/bin/rustc" "+$TOOLCHAIN_VERSION" -vV | tr '\n' ';')" \
  BUILD_CARGO="$("$CARGO" "+$TOOLCHAIN_VERSION" --version)" BUILD_CC="$( (cc --version || gcc --version) 2>/dev/null | head -1)" \
  BUILD_LDD="$(ldd "$OUT/librust_utils.so" 2>&1 || true)" BUILD_SOURCES="$SOURCES" BUILD_REPO_SRC="$SRC" \
    PYTHONNOUSERSITE=1 "$TRAIN_PY" - "$OUT/provenance.json" <<'EOF'
import hashlib, json, os, platform, socket, sys, sysconfig, time
from pathlib import Path
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
lib = Path(os.environ["BUILD_LIBRARY"])
record = {
    "kind": "hyak_rust_utils_build", "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host": socket.gethostname(), "platform": platform.platform(),
    "variant": os.environ["BUILD_VARIANT"], "features": os.environ["BUILD_FEATURES"] or None,
    "rustc": os.environ["BUILD_RUSTC"], "cargo": os.environ["BUILD_CARGO"], "cc": os.environ["BUILD_CC"],
    "rustflags": os.environ.get("RUSTFLAGS") or None, "pyo3_python": sys.executable, "python_version": platform.python_version(),
    "python_libdir": sysconfig.get_config_var("LIBDIR"),
    "sources": {name: sha(Path(os.environ["BUILD_REPO_SRC"]) / name) for name in os.environ["BUILD_SOURCES"].split()},
    "library": str(lib), "bytes": lib.stat().st_size, "sha256": sha(lib), "ldd": os.environ["BUILD_LDD"][-4000:],
}
try:
    record["glibc"] = os.confstr("CS_GNU_LIBC_VERSION")
except (ValueError, OSError):
    record["glibc"] = None
target = Path(sys.argv[1])
tmp = target.with_name(target.name + ".tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(tmp, target)
print(f"[build_rust_utils] built {lib} ({record['variant']}, sha256 {record['sha256']})")
EOF
}

phase_check() {
  [ "$DRY_RUN" = 1 ] || [ -f "$OUT/librust_utils.so" ] || hyak_die "$OUT/librust_utils.so is missing. Remedy: run the build phase"
  [ "$DRY_RUN" = 1 ] || [ -x "$TRAIN_PY" ] || hyak_die "$TRAIN_PY is missing. Remedy: bash $HERE/setup_train_env.sh"
  export LD_LIBRARY_PATH="$(libpython_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  run env PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "$TRAIN_PY" -m hyak.env.check_env \
    --only rust_library --rust-library "$OUT/librust_utils.so" --no-gpu --no-network --no-status
}

phase_install() {
  [ "$DRY_RUN" = 1 ] || [ -f "$OUT/provenance.json" ] || hyak_die "$OUT/provenance.json is missing. Remedy: run the build phase"
  run env PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$(stdlib_python)" -m hyak.code.stage_code \
    --rust-library "$OUT/librust_utils.so" --rust-provenance "$OUT/provenance.json" --force
}

say "repository $HYAK_REPO; build tree $BUILD; phases:$PHASES"
for phase in $PHASES; do
  "phase_$phase"
done
say "done:$PHASES"
