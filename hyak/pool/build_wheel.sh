#!/usr/bin/env bash
# hyak/pool/build_wheel.sh -- build the patched Hugging Face tokenizers 0.22.2 wheel that trains
# the 1,000-tokenizer pool (DESIGN.md 5.F; port of scripts/build_fast_tokenizers.py).
#
# Usage:
#   bash hyak/pool/build_wheel.sh PHASE... [options]
#
# Phases (run in the order given):
#   fetch    login node, needs internet: clone huggingface/tokenizers at f383101a (= tag v0.22.2)
#            into --src, check the four upstream blob ids, apply the three patches in the order of
#            scripts/build_fast_tokenizers.py:29 (all-candidate-noise, unigram-literal-unk,
#            unigram-stack-drop; never merge-noise), check the patched blob ids, write the source
#            marker .anytok-source.json, and download every crate (cargo fetch).
#   build    no network needed (CARGO_NET_OFFLINE=true): Rust unit tests as in
#            build_fast_tokenizers.py --test (cargo test --release --lib on the tokenizers crate),
#            then maturin build --release --locked; the wheel and provenance.json go to --wheels.
#   install  pip-install the wheel (--no-deps) into the DATA environment only, then prove the
#            patched trainer is present (same probe as scripts/tokenizer_pool.py:156), that the
#            installed extension is byte-identical to the wheel's, and that the repository's
#            TrainerTest (tests/test_tokenizer_pool.py, nine combinations, literal <UNK>) passes.
#   verify   the install checks without installing.
#   all      fetch build install.
#
# Options:
#   --src DIR         source checkout     (default: Paths.tokenizers_src = $ANYTOK_ROOT/build/tokenizers-src)
#   --wheels DIR      wheel + provenance  (default: Paths.wheels = $ANYTOK_ROOT/wheels)
#   --python PATH     Python that runs maturin and receives the wheel
#                     (default: $ANYTOK_ROOT/env/data/bin/python)
#   --toolchain DIR   rustup/cargo homes DIR/rustup and DIR/cargo (default: $ANYTOK_ROOT/toolchain)
#   --rust VERSION    Rust toolchain (default 1.96.0; upstream's rust-toolchain files say "stable",
#                     so RUSTUP_TOOLCHAIN is always set explicitly)
#   --jobs N          cargo build jobs (default: $SLURM_CPUS_PER_TASK, else all cores)
#   --skip-tests      build without the Rust unit tests (recorded in provenance; not recommended)
#   --force           rebuild although a verified wheel exists / replace a different wheel
#   --no-config       do not read hyak/config.env; then --src, --wheels, --python, --toolchain are required
#   --source URL|DIR  upstream repository to clone (default https://github.com/huggingface/tokenizers.git;
#                     a local clone works too, like build_fast_tokenizers.py --source)
#
# Facts checked on 2026-11-13 (git ls-remote / git ls-tree of the upstream repository):
#   * f383101a26663708484cac0727792aad74f78234 is the commit of tag v0.22.2 (2026-01-05).
#   * bindings/python/Cargo.lock is tracked, so maturin --locked works as in the original.
#   * tokenizers/ (the crate the Rust unit tests run on) has no Cargo.lock: fetch seeds it with a
#     copy of bindings/python/Cargo.lock, so the tests use the wheel's dependency versions; crates
#     only the tests need (dev-dependencies) are resolved at fetch time (recorded in provenance).
#   * bindings/python/rust-toolchain and tokenizers/rust-toolchain contain "stable".
set -euo pipefail

UPSTREAM_REVISION=f383101a26663708484cac0727792aad74f78234
UPSTREAM_TAG=v0.22.2
RUST_TOOLCHAIN_DEFAULT=1.96.0
# Order matters (scripts/build_fast_tokenizers.py:29).  Must equal hyak.pool.PATCH_FILES (tested).
PATCH_FILES="tokenizers-0.22.2-all-candidate-noise.patch tokenizers-0.22.2-unigram-literal-unk.patch tokenizers-0.22.2-unigram-stack-drop.patch"
FORBIDDEN_PATCH=tokenizers-0.22.2-merge-noise.patch
# "<path> <full blob id at UPSTREAM_REVISION> <7-digit blob id after the patches>"; must equal
# hyak.pool.UPSTREAM_BLOBS and the patches' index lines (tested).
BLOB_TABLE="bindings/python/src/trainers.rs 53415fff02cb5d2651dd92cdb760f944bbe77ef3 1c55e0b
tokenizers/src/models/bpe/trainer.rs cda6aea65436842277d4ed7f4ab53f6447157af4 3b26584
tokenizers/src/models/unigram/model.rs 955ac7b5ce831597cdaf8b9099140bfb964f4ed0 7146740
tokenizers/src/models/unigram/lattice.rs 2897bb3767ff20c017e0087871fc68a9f7b2ff01 b81b3c3"

_self_dir=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [ -z "${HYAK_REPO:-}" ]; then
  HYAK_REPO=$(CDPATH='' cd -P -- "$_self_dir/../.." && pwd -P)
  export HYAK_REPO
fi
# shellcheck source=../slurm/lib.sh
source "$HYAK_REPO/hyak/slurm/lib.sh"

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

PHASES=""
SRC=""; WHEELS=""; PY=""; TOOLCHAIN=""; RUST_VERSION=$RUST_TOOLCHAIN_DEFAULT; JOBS=""
SKIP_TESTS=0; FORCE=0; USE_CONFIG=1; SOURCE_URL=https://github.com/huggingface/tokenizers.git
while [ $# -gt 0 ]; do
  case $1 in
    fetch|build|install|verify) PHASES="$PHASES $1" ;;
    all) PHASES="$PHASES fetch build install" ;;
    --src) SRC=${2:?--src needs a directory}; shift ;;
    --wheels) WHEELS=${2:?--wheels needs a directory}; shift ;;
    --python) PY=${2:?--python needs a path}; shift ;;
    --toolchain) TOOLCHAIN=${2:?--toolchain needs a directory}; shift ;;
    --rust) RUST_VERSION=${2:?--rust needs a version}; shift ;;
    --jobs) JOBS=${2:?--jobs needs a number}; shift ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --force) FORCE=1 ;;
    --no-config) USE_CONFIG=0 ;;
    --source) SOURCE_URL=${2:?--source needs a URL or directory}; shift ;;
    -h|--help) usage; exit 0 ;;
    *) hyak_die "build_wheel.sh: unknown argument '$1' (see --help)" ;;
  esac
  shift
done
[ -n "$PHASES" ] || { usage; hyak_die "build_wheel.sh: give at least one phase: fetch build install verify all"; }

HELPER_PY=$(hyak_python)
_pyhelp() { PYTHONPATH="$HYAK_REPO" PYTHONDONTWRITEBYTECODE=1 "$HELPER_PY" "$@"; }

# --- locations ----------------------------------------------------------------------------------
TRAIN_PY=""
# (Python code lives in functions that print it: bash 3.2 cannot parse a here-document inside $(...).)
_code_locations() { cat <<'EOF'
from hyak.common.config import load_config
from hyak.common.paths import Paths
p = Paths.from_config(load_config())
for value in (p.tokenizers_src, p.wheels, p.env_data / "bin" / "python", p.toolchain, p.env_train / "bin" / "python"):
    print(value)
EOF
}
if [ "$USE_CONFIG" = 1 ]; then
  hyak_load_config
  _locations=$(_pyhelp -c "$(_code_locations)") || hyak_die "could not resolve storage paths from the config"
  _n=0
  while IFS= read -r _line; do
    _n=$((_n + 1))
    case $_n in
      1) [ -n "$SRC" ] || SRC=$_line ;;
      2) [ -n "$WHEELS" ] || WHEELS=$_line ;;
      3) [ -n "$PY" ] || PY=$_line ;;
      4) [ -n "$TOOLCHAIN" ] || TOOLCHAIN=$_line ;;
      5) TRAIN_PY=$_line ;;
    esac
  done <<EOF
$_locations
EOF
else
  [ -n "$SRC" ] && [ -n "$WHEELS" ] && [ -n "$PY" ] && [ -n "$TOOLCHAIN" ] \
    || hyak_die "--no-config requires --src, --wheels, --python and --toolchain"
fi
for _v in SRC WHEELS TOOLCHAIN; do
  case $(eval "printf '%s' \"\$$_v\"") in /*) ;; *) hyak_die "$_v must be an absolute path" ;; esac
done
export RUSTUP_HOME="$TOOLCHAIN/rustup" CARGO_HOME="$TOOLCHAIN/cargo"
export RUSTUP_TOOLCHAIN="$RUST_VERSION"
export PATH="$CARGO_HOME/bin:$PATH"
# Crates whose newest versions need a newer compiler are skipped when the tests' extra
# dependencies are resolved (cargo >= 1.84); the wheel itself always uses upstream's lock file.
export CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback
if [ -z "$JOBS" ]; then
  JOBS=${SLURM_CPUS_PER_TASK:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
fi
export CARGO_BUILD_JOBS="$JOBS"
MARKER="$SRC/.anytok-source.json"
FETCH_RECORD="$SRC/.anytok-fetch.json"
PROVENANCE="$WHEELS/provenance.json"

_status() {  # _status NAME [K=V...]: wheel-stage event (never fails; skipped without a config)
  if [ "$USE_CONFIG" = 1 ]; then hyak_status_event wheel "$@"; fi
}

_sha256() { _pyhelp -c 'import hashlib,sys; h=hashlib.sha256(open(sys.argv[1],"rb").read()); print(h.hexdigest())' "$1"; }

_expect_patch_hashes() {  # verify the repository's patch files against hyak.pool.PATCH_SHA256
  _pyhelp - "$HYAK_REPO" $PATCH_FILES <<'EOF' || hyak_die "patch files in $HYAK_REPO/patches differ from the pinned SHA-256 values; restore them with git checkout -- patches/"
import hashlib, sys
from pathlib import Path
from hyak.pool import PATCH_FILES, PATCH_SHA256
repo, names = Path(sys.argv[1]), tuple(sys.argv[2:])
assert names == PATCH_FILES, f'patch order {names} != {PATCH_FILES}'
for name in names:
    actual = hashlib.sha256((repo / 'patches' / name).read_bytes()).hexdigest()
    if actual != PATCH_SHA256[name]:
        sys.exit(f'{name}: {actual} != {PATCH_SHA256[name]}')
EOF
}

_need_toolchain() {
  [ -x "$CARGO_HOME/bin/cargo" ] && [ -x "$CARGO_HOME/bin/rustup" ] \
    || hyak_die "Rust toolchain not found under $TOOLCHAIN. Remedy: run bash $HYAK_REPO/hyak/env/setup_data_env.sh on the login node (installs rustup and toolchain $RUST_VERSION there)"
  rustup toolchain list 2>/dev/null | grep -q "^$RUST_VERSION-" \
    || hyak_die "Rust toolchain $RUST_VERSION is not installed in $RUSTUP_HOME. Remedy: RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME $CARGO_HOME/bin/rustup toolchain install $RUST_VERSION --profile minimal"
  local v
  v=$(rustc -V 2>/dev/null || true)
  case $v in "rustc $RUST_VERSION "*) ;; *) hyak_die "rustc reports '$v', expected $RUST_VERSION" ;; esac
}

# _check_source DIR: the checkout is at the pinned revision, carries exactly the three patches,
# and matches its marker (as scripts/build_fast_tokenizers.py:42-47 checked).
_check_source() {
  local dir=$1 head
  [ -d "$dir/.git" ] || hyak_die "$dir is not a git checkout. Remedy: run the fetch phase (on the login node)"
  [ -f "$dir/$(basename "$MARKER")" ] || hyak_die "$dir has no source marker; it was not created by build_wheel.sh fetch. Remedy: move it away and rerun fetch"
  head=$(git -C "$dir" rev-parse HEAD)
  [ "$head" = "$UPSTREAM_REVISION" ] || hyak_die "$dir is at $head, expected $UPSTREAM_REVISION"
  _pyhelp - "$dir" "$HYAK_REPO" <<'EOF' || hyak_die "existing build source $dir differs from its provenance. Remedy: move $dir away and rerun the fetch phase"
import hashlib, json, subprocess, sys
from pathlib import Path
from hyak.pool import PATCH_FILES, PATCH_SHA256, UPSTREAM_REVISION
folder, repo = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads((folder / '.anytok-source.json').read_text())
if data['revision'] != UPSTREAM_REVISION or data['patches'] != {n: PATCH_SHA256[n] for n in PATCH_FILES}:
    sys.exit('Existing source build provenance differs')
tracked = subprocess.check_output(['git', '-C', str(folder), 'diff', '--name-only']).decode().splitlines()
if sorted(tracked) != sorted(data['files']):
    sys.exit(f'changed files {sorted(tracked)} != recorded {sorted(data["files"])}')
for name, sha in data['files'].items():
    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha:
        sys.exit('Existing build source was edited: ' + name)
EOF
  local path base after actual
  while read -r path base after; do
    actual=$(git -C "$dir" hash-object "$path")
    case $actual in "$after"*) ;; *) hyak_die "$dir/$path has blob $actual; the patched file should be $after... (patches not applied as expected)" ;; esac
  done <<EOF
$BLOB_TABLE
EOF
}

# --- fetch -------------------------------------------------------------------------------------------------
phase_fetch() {
  command -v git >/dev/null 2>&1 || hyak_die "git is not installed"
  _need_toolchain
  _expect_patch_hashes
  if [ -e "$SRC" ]; then
    hyak_info "fetch: reusing existing source $SRC"
  else
    local tmp="$SRC.partial.$$" path base after actual patch
    mkdir -p "$(dirname "$SRC")"
    rm -rf "$tmp"
    hyak_info "fetch: cloning $SOURCE_URL at $UPSTREAM_REVISION ($UPSTREAM_TAG)"
    git clone --quiet --no-checkout "$SOURCE_URL" "$tmp" || { rm -rf "$tmp"; hyak_die "git clone of $SOURCE_URL failed (internet on this node?)"; }
    git -C "$tmp" -c advice.detachedHead=false checkout --quiet --detach "$UPSTREAM_REVISION" \
      || { rm -rf "$tmp"; hyak_die "revision $UPSTREAM_REVISION not found in $SOURCE_URL"; }
    [ "$(git -C "$tmp" rev-parse HEAD)" = "$UPSTREAM_REVISION" ] || { rm -rf "$tmp"; hyak_die "checkout is not at $UPSTREAM_REVISION"; }
    if [ -n "$(git -C "$tmp" status --porcelain)" ]; then rm -rf "$tmp"; hyak_die "fresh checkout is not clean"; fi
    while read -r path base after; do
      actual=$(git -C "$tmp" rev-parse "HEAD:$path")
      [ "$actual" = "$base" ] || { rm -rf "$tmp"; hyak_die "upstream $path is blob $actual, expected $base (not the patches' base)"; }
    done <<EOF
$BLOB_TABLE
EOF
    for patch in $PATCH_FILES; do
      [ "$patch" != "$FORBIDDEN_PATCH" ] || hyak_die "refusing to apply $FORBIDDEN_PATCH"
      git -C "$tmp" apply --check "$HYAK_REPO/patches/$patch" || { rm -rf "$tmp"; hyak_die "patch $patch does not apply to $UPSTREAM_REVISION"; }
      git -C "$tmp" apply "$HYAK_REPO/patches/$patch"
      hyak_info "fetch: applied $patch"
    done
    # Marker in the format of scripts/build_fast_tokenizers.py:39-41.
    _pyhelp - "$tmp" "$HYAK_REPO" <<'EOF' || { rm -rf "$tmp"; hyak_die "could not write the source marker"; }
import hashlib, json, subprocess, sys
from pathlib import Path
from hyak.pool import PATCH_FILES, PATCH_SHA256, UPSTREAM_REVISION
folder, repo = Path(sys.argv[1]), Path(sys.argv[2])
hashes = {name: hashlib.sha256((repo / 'patches' / name).read_bytes()).hexdigest() for name in PATCH_FILES}
assert hashes == {n: PATCH_SHA256[n] for n in PATCH_FILES}
tracked = subprocess.check_output(['git', '-C', str(folder), 'diff', '--name-only']).decode().splitlines()
files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in tracked}
(folder / '.anytok-source.json').write_text(json.dumps({'revision': UPSTREAM_REVISION, 'patches': hashes, 'files': files}, indent=2) + '\n')
EOF
    mv "$tmp" "$SRC"
  fi
  _check_source "$SRC"
  hyak_info "fetch: source verified ($UPSTREAM_REVISION + 3 patches)"
  if [ ! -f "$SRC/tokenizers/Cargo.lock" ]; then
    # The tokenizers crate has no lock file upstream; start from the wheel's lock (see header).
    cp "$SRC/bindings/python/Cargo.lock" "$SRC/tokenizers/Cargo.lock"
  fi
  hyak_info "fetch: downloading crates (cargo fetch) into $CARGO_HOME"
  (cd "$SRC/bindings/python" && cargo fetch --locked --manifest-path "$SRC/bindings/python/Cargo.toml") \
    || hyak_die "cargo fetch --locked failed for bindings/python (network? upstream lock file changed?)"
  (cd "$SRC/tokenizers" && cargo fetch --manifest-path "$SRC/tokenizers/Cargo.toml") \
    || hyak_die "cargo fetch failed for the tokenizers crate"
  _pyhelp - "$SRC" "$FETCH_RECORD" "$(cargo -V)" <<'EOF' || hyak_die "could not write $FETCH_RECORD"
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
src, out, cargo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
tracked = set(subprocess.check_output(['git', '-C', str(src), 'ls-files']).decode().splitlines())
locks = {}
for rel in ('bindings/python/Cargo.lock', 'tokenizers/Cargo.lock'):
    locks[rel] = {'sha256': hashlib.sha256((src / rel).read_bytes()).hexdigest(), 'tracked_upstream': rel in tracked}
record = {'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'cargo': cargo, 'locks': locks}
out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
EOF
  _status fetch_complete "revision=$UPSTREAM_REVISION"
  hyak_info "fetch: complete. Next: the build phase (no internet needed)"
}

# --- build ---------------------------------------------------------------------------------------------------
# _need_compilers: rustc links with a program named cc, the onig crate compiles C and esaxx-rs (default
# feature esaxx_fast of the tokenizers crate) compiles C++ (the cc crate calls cc and c++).  When only
# gcc / g++ exist, CC / CXX / the cargo linker name them; these variables choose programs only.
_need_compilers() {
  if ! command -v cc >/dev/null 2>&1 && command -v gcc >/dev/null 2>&1; then
    export CC=gcc CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=gcc
  fi
  if ! command -v c++ >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    export CXX=g++
  fi
  if ! command -v "${CC:-cc}" >/dev/null 2>&1 || ! command -v "${CXX:-c++}" >/dev/null 2>&1; then
    hyak_die "a C and a C++ compiler are needed (onig: C, esaxx-rs: C++, rustc links with cc); found C: $(command -v "${CC:-cc}" 2>/dev/null || echo none), C++: $(command -v "${CXX:-c++}" 2>/dev/null || echo none). Remedy: module load gcc (e.g. gcc/13.4.0; HYAK_GCC_MODULE) before this phase"
  fi
}

phase_build() {
  command -v git >/dev/null 2>&1 \
    || hyak_die "git is not installed on this node; the build phase verifies the checkout with git. Remedy: module load git (or run the build phase where git exists)"
  _need_compilers
  _need_toolchain
  _expect_patch_hashes
  _check_source "$SRC"
  [ -f "$FETCH_RECORD" ] || hyak_die "no crates fetched for $SRC. Remedy: run the fetch phase on the login node"
  _pyhelp - "$SRC" "$FETCH_RECORD" <<'EOF' || hyak_die "a Cargo.lock in $SRC changed after the fetch phase. Remedy: move $SRC away and rerun fetch"
import hashlib, json, sys
from pathlib import Path
src, record = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
for rel, item in record['locks'].items():
    if hashlib.sha256((src / rel).read_bytes()).hexdigest() != item['sha256']:
        sys.exit(rel + ' changed')
EOF
  [ -x "$PY" ] || hyak_die "Python $PY not found. Remedy: create the data environment (hyak/env/setup_data_env.sh)"
  "$PY" -m maturin --version >/dev/null 2>&1 || hyak_die "maturin is not installed in $PY. Remedy: $PY -m pip install maturin"
  if [ -f "$PROVENANCE" ] && [ "$FORCE" = 0 ]; then
    if _pyhelp - "$PROVENANCE" "$WHEELS" "$SRC" <<'EOF'
import hashlib, json, sys
from pathlib import Path
prov, wheels, src = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3])
wheel = wheels / prov['wheel']['name']
marker = json.loads((src / '.anytok-source.json').read_text())
ok = (wheel.is_file() and hashlib.sha256(wheel.read_bytes()).hexdigest() == prov['wheel']['sha256']
      and prov['source_marker'] == marker)
sys.exit(0 if ok else 1)
EOF
    then
      hyak_info "build: $PROVENANCE already describes a verified wheel built from this source; nothing to do (use --force to rebuild)"
      return 0
    fi
    hyak_die "$PROVENANCE exists but does not match its wheel or this source. Remedy: inspect $WHEELS; rerun with --force to rebuild (tokenizers already generated with the old wheel will then be refused by the worker)"
  fi
  export CARGO_NET_OFFLINE=true
  mkdir -p "$WHEELS/logs"
  local stamp test_log test_summary="skipped" out tests_ran=0 started
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  started=$(date +%s)
  if [ "$SKIP_TESTS" = 0 ]; then
    test_log="$WHEELS/logs/rust-tests-$stamp.log"
    hyak_info "build: Rust unit tests (cargo test --release --lib, $JOBS jobs); log $test_log"
    _status rust_tests_start "log=$test_log"
    if ! (cd "$SRC/tokenizers" && cargo test --release --locked --manifest-path "$SRC/tokenizers/Cargo.toml" --lib) >"$test_log" 2>&1; then
      tail -40 "$test_log" >&2
      hyak_die "Rust unit tests failed (log $test_log)"
    fi
    test_summary=$(grep -E '^test result:' "$test_log" | tail -1)
    case $test_summary in "test result: ok."*) ;; *) hyak_die "could not confirm the Rust test result in $test_log" ;; esac
    tests_ran=1
    hyak_info "build: $test_summary"
  else
    hyak_warn "build: Rust unit tests skipped (--skip-tests)"
  fi
  out=$(mktemp -d "$WHEELS/.build-$stamp.XXXXXX")
  hyak_info "build: maturin build --release --locked"
  _status maturin_start
  (cd "$SRC/bindings/python" && "$PY" -m maturin build --release --locked \
      --manifest-path "$SRC/bindings/python/Cargo.toml" --interpreter "$PY" --out "$out") \
    || { rm -rf "$out"; hyak_die "maturin build failed"; }
  _pyhelp - "$out" "$WHEELS" "$SRC" "$FETCH_RECORD" "$PY" "$FORCE" "$tests_ran" "$test_summary" \
      "${test_log:-}" "$(rustc -vV | tr '\n' ';')" "$(cargo -V)" "$("$PY" -m maturin --version)" \
      "$(( $(date +%s) - started ))" "$HYAK_REPO" <<'EOF' || { rm -rf "$out"; hyak_die "could not publish the wheel into $WHEELS"; }
import hashlib, json, os, platform, shutil, socket, subprocess, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
from hyak.common.io import atomic_write_json
from hyak.pool import (PATCH_FILES, PATCH_SHA256, RUST_TOOLCHAIN, TOKENIZERS_VERSION, UPSTREAM_REVISION,
                       UPSTREAM_TAG, UPSTREAM_URL)
(out, wheels, src, fetch, py, force, tests_ran, test_summary, test_log, rustc, cargo, maturin, seconds,
 repo) = sys.argv[1:]
out, wheels, src = Path(out), Path(wheels), Path(src)
built = sorted(out.glob('*.whl'))
if len(built) != 1:
    sys.exit(f'expected exactly one wheel in {out}, found {[p.name for p in built]}')
wheel = built[0]
if not wheel.name.startswith(f'tokenizers-{TOKENIZERS_VERSION}-cp39-abi3-'):
    sys.exit(f'unexpected wheel name {wheel.name}')
data = wheel.read_bytes()
sha = hashlib.sha256(data).hexdigest()
with zipfile.ZipFile(wheel) as archive:
    members = [n for n in archive.namelist() if n.startswith('tokenizers/') and n.endswith(('.so', '.pyd'))]
    if len(members) != 1:
        sys.exit(f'expected one compiled extension in the wheel, found {members}')
    extension_sha = hashlib.sha256(archive.read(members[0])).hexdigest()
target = wheels / wheel.name
previous = sorted(p.name for p in wheels.glob('*.whl') if p.name != wheel.name)
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != sha:
    if force != '1':
        sys.exit(f'{target} exists with different content; rerun with --force to replace it')
if previous and force != '1':
    sys.exit(f'{wheels} already holds other wheels {previous}; rerun with --force to add another')
if force == '1':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    old = [p for p in wheels.glob('*.whl')] + ([wheels / 'provenance.json'] if (wheels / 'provenance.json').exists() else [])
    if old:
        keep = wheels / 'superseded' / stamp
        keep.mkdir(parents=True, exist_ok=True)
        for p in old:
            shutil.move(str(p), str(keep / p.name))
temporary = wheels / f'.{wheel.name}.tmp'
shutil.copyfile(wheel, temporary)
if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha:
    sys.exit('wheel copy is corrupt')
os.replace(temporary, target)
shutil.rmtree(out)
marker = json.loads((src / '.anytok-source.json').read_text())
fetch_record = json.loads(Path(fetch).read_text())
provenance = {
    'kind': 'anytok_patched_tokenizers_wheel',
    'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'wheel': {'name': wheel.name, 'bytes': len(data), 'sha256': sha,
              'extension_member': members[0], 'extension_sha256': extension_sha},
    'tokenizers_version': TOKENIZERS_VERSION,
    'upstream': {'url': UPSTREAM_URL, 'revision': UPSTREAM_REVISION, 'tag': UPSTREAM_TAG},
    'patches': {name: PATCH_SHA256[name] for name in PATCH_FILES},
    'patch_order': list(PATCH_FILES),
    'source_marker': marker,
    'cargo_locks': fetch_record['locks'],
    'rust_tests': {'ran': tests_ran == '1', 'summary': test_summary,
                   'command': 'cargo test --release --locked --manifest-path tokenizers/Cargo.toml --lib',
                   'log': test_log or None},
    'build_command': 'python -m maturin build --release --locked --manifest-path bindings/python/Cargo.toml',
    'toolchain': {'rust_toolchain': RUST_TOOLCHAIN if os.environ.get('RUSTUP_TOOLCHAIN') == RUST_TOOLCHAIN
                  else os.environ.get('RUSTUP_TOOLCHAIN'), 'rustc': rustc, 'cargo': cargo, 'maturin': maturin},
    'python': {'executable': py, 'version': subprocess.check_output([py, '-c', 'import sys; print(sys.version.split()[0])']).decode().strip()},
    'platform': {'system': platform.system(), 'machine': platform.machine(), 'release': platform.release(),
                 'libc': '-'.join(platform.libc_ver()), 'host': socket.gethostname(),
                 'slurm_job_id': os.environ.get('SLURM_JOB_ID')},
    'build_seconds': int(seconds),
    'repo_build_script_sha256': hashlib.sha256((Path(repo) / 'scripts/build_fast_tokenizers.py').read_bytes()).hexdigest(),
    'installs': [],
}
atomic_write_json(wheels / 'provenance.json', provenance)
print(json.dumps({'wheel': str(target), 'sha256': sha, 'extension_sha256': extension_sha}))
EOF
  _status build_complete
  hyak_info "build: complete; wheel and provenance in $WHEELS"
}

# --- install / verify ------------------------------------------------------------------------------------------
_code_probe() { cat <<'EOF'
import hashlib, json, sys
from pathlib import Path
prov = json.loads(Path(sys.argv[1]).read_text())
wheel = Path(sys.argv[2]) / prov["wheel"]["name"]
if hashlib.sha256(wheel.read_bytes()).hexdigest() != prov["wheel"]["sha256"]:
    sys.exit(f"{wheel} does not match provenance.json")
import tokenizers
from tokenizers import trainers
if tokenizers.__version__ != prov["tokenizers_version"]:
    sys.exit(f"installed tokenizers {tokenizers.__version__} != {prov['tokenizers_version']}")
# Same probe as scripts/tokenizer_pool.py:156 (NOTES/tokenizer-pool.md section 7, test 4).
state = json.loads(trainers.BpeTrainer(noise_per_million=5, noise_seed=7, random_from_all=True).__getstate__())["BpeTrainer"]
if (state.get("noise_per_million"), state.get("noise_seed"), state.get("random_from_all")) != (5, 7, True):
    sys.exit(f"patched noise trainer is not installed (BpeTrainer state {state})")
package = Path(tokenizers.__file__).resolve().parent
installed = package / Path(prov["wheel"]["extension_member"]).name
if not installed.is_file():
    sys.exit(f"{installed} not found")
actual = hashlib.sha256(installed.read_bytes()).hexdigest()
if actual != prov["wheel"]["extension_sha256"]:
    sys.exit(f"installed extension {installed} ({actual}) differs from the extension in the wheel ({prov['wheel']['extension_sha256']})")
print(json.dumps({"python": sys.executable, "prefix": sys.prefix, "extension": str(installed),
                  "extension_sha256": actual, "tokenizers_version": tokenizers.__version__}))
EOF
}

_verify_installed() {  # _verify_installed: probe + byte identity + repository TrainerTest; prints JSON
  local probe trainer_log
  probe=$(cd / && env -u PYTHONPATH "$PY" -I -c "$(_code_probe)" "$PROVENANCE" "$WHEELS") \
    || hyak_die "the patched wheel is not correctly installed in $PY. Remedy: rerun the install phase"
  trainer_log=$(mktemp "${TMPDIR:-/tmp}/anytok-trainertest.XXXXXX")
  # The repository's own nine-combination check (tests/test_tokenizer_pool.py:48-77); it skips itself
  # when the patch is missing, so a skip is treated as a failure here.
  if ! (cd "$HYAK_REPO" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true \
        "$PY" -m unittest discover -s tests -p test_tokenizer_pool.py -v) >"$trainer_log" 2>&1; then
    tail -30 "$trainer_log" >&2
    rm -f "$trainer_log"
    hyak_die "tests/test_tokenizer_pool.py failed with the installed wheel"
  fi
  if grep -q -i 'skipped' "$trainer_log" \
      || ! grep -E -q '^test_all_nine_algorithm_pretokenizer_combinations( \(.*\))? \.\.\. ok$' "$trainer_log"; then
    tail -30 "$trainer_log" >&2
    rm -f "$trainer_log"
    hyak_die "TrainerTest did not run against the patched wheel"
  fi
  rm -f "$trainer_log"
  printf '%s\n' "$probe"
}

phase_install() {
  [ -f "$PROVENANCE" ] || hyak_die "no $PROVENANCE. Remedy: run the build phase first"
  [ -x "$PY" ] || hyak_die "Python $PY not found"
  if [ -n "$TRAIN_PY" ] && [ "$(CDPATH='' cd -P -- "$(dirname "$PY")" && pwd -P)" = "$(CDPATH='' cd -P -- "$(dirname "$TRAIN_PY")" 2>/dev/null && pwd -P)" ]; then
    hyak_die "refusing to install the patched wheel into the train environment ($PY); it belongs to the data environment only (D10)"
  fi
  local wheel
  wheel="$WHEELS/$(_pyhelp -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel"]["name"])' "$PROVENANCE")"
  [ -f "$wheel" ] || hyak_die "wheel $wheel is missing"
  if "$PY" -m pip --version >/dev/null 2>&1; then
    hyak_info "install: pip install --no-deps --force-reinstall $wheel into $PY"
    env -u PYTHONPATH "$PY" -m pip install --no-deps --no-index --force-reinstall --quiet "$wheel" \
      || hyak_die "pip install of $wheel failed"
  elif command -v uv >/dev/null 2>&1; then
    hyak_info "install: uv pip install --no-deps --reinstall $wheel into $PY"
    uv pip install --python "$PY" --no-deps --reinstall --quiet "$wheel" || hyak_die "uv pip install of $wheel failed"
  else
    hyak_die "neither pip nor uv is available for $PY"
  fi
  local result
  result=$(_verify_installed)
  _pyhelp - "$PROVENANCE" "$result" <<'EOF' || hyak_die "could not record the install in $PROVENANCE"
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from hyak.common.io import atomic_write_json
path, record = Path(sys.argv[1]), json.loads(sys.argv[2])
prov = json.loads(path.read_text())
record['verified_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
record['trainer_test'] = 'tests/test_tokenizer_pool.py: PoolPlanTest + TrainerTest (9 combinations) OK'
prov['installs'] = [r for r in prov.get('installs', []) if r.get('prefix') != record['prefix']] + [record]
atomic_write_json(path, prov)
EOF
  _status install_complete
  hyak_info "install: verified $result"
}

phase_verify() {
  [ -f "$PROVENANCE" ] || hyak_die "no $PROVENANCE. Remedy: run the build phase first"
  [ -x "$PY" ] || hyak_die "Python $PY not found"
  hyak_info "verify: $(_verify_installed)"
}

for _phase in $PHASES; do
  hyak_info "build_wheel.sh: phase $_phase (src $SRC, wheels $WHEELS, python $PY, rust $RUST_VERSION)"
  "phase_$_phase"
done
