#!/bin/bash
# hyak/slurm/03_wheel.sbatch -- stage wheel: build and install the patched tokenizers 0.22.2 wheel.
#
# First, on the login node (needs internet):   bash hyak/pool/build_wheel.sh fetch
# Then submit (minutes; fits the debug QoS):   hyak/slurm/submit.sh wheel --debug
# Extra arguments for build_wheel.sh:          --env HYAK_WHEEL_ARGS="--force"
# C and C++ compiler module:                   --env HYAK_GCC_MODULE=gcc/13.4.0  (loaded when it is set, or when
#                                               no cc/gcc or no c++/g++ is on PATH; that is the default)
#
# Data environment.  Runs "bash hyak/pool/build_wheel.sh build install": the Rust unit tests and
# maturin build --release --locked offline, then pip install --no-deps into the DATA environment
# only, and the checks that the patched trainer is present, the installed extension equals the
# wheel's and the repository's TrainerTest passes.  The wheel and wheels/provenance.json go to
# $ANYTOK_ROOT/wheels.  On success the data environment's freeze file is refreshed (the old one is
# kept as data.freeze.<UTC>.txt) and the stage status becomes "complete".  build_wheel.sh is a shell
# script: its failures are its own deliberate messages (see the job log), not a Python traceback.
#
# No #SBATCH lines on purpose: hyak/slurm/submit.sh passes every resource (partition, QoS, account,
# GPU, CPUs, memory, time), the job name, the output file, --signal and --export from
# hyak/config.env.  Do not submit this file with plain sbatch.

# --- common job prologue: identical in every hyak/slurm/*.sbatch (checked by hyak/tests/test_integration_sbatch.py) ---
set -euo pipefail
if [ -z "${HYAK_REPO:-}" ] || [ ! -r "$HYAK_REPO/hyak/slurm/job.sh" ]; then
  echo "[hyak] ERROR: HYAK_REPO must name the repository checkout; submit with hyak/slurm/submit.sh" >&2
  exit 2
fi
# shellcheck source=hyak/slurm/lib.sh
source "$HYAK_REPO/hyak/slurm/lib.sh"
# shellcheck source=hyak/slurm/job.sh
source "$HYAK_REPO/hyak/slurm/job.sh"
# --- end of common job prologue ---

job_begin wheel
hyak_status_start "$JOB_STAGE" "patched tokenizers wheel: build and install"
hyak_activate data
hyak_register_dashboard

# The onig crate compiles C code and the esaxx crate C++ (hyak/pool/build_wheel.sh refuses to build without a
# C and C++ compiler).  Load the gcc module when HYAK_GCC_MODULE is set, or when a C compiler (cc or gcc) or a
# C++ compiler (c++ or g++) is missing.
if gcc_module=$(hyak_gcc_module); then
  hyak_info "loading the C and C++ compiler module $gcc_module (HYAK_GCC_MODULE chooses another: --env HYAK_GCC_MODULE=<module>)"
  hyak_load_modules "HYAK_GCC_MODULE (--env HYAK_GCC_MODULE=<module>; module avail gcc lists them)" "$gcc_module"
fi

job_extra_args HYAK_WHEEL_ARGS
rc=0
job_step bash "$HYAK_REPO/hyak/pool/build_wheel.sh" build install ${JOB_ARGS[@]+"${JOB_ARGS[@]}"} || rc=$?
if [ "$rc" = 0 ]; then
  # The patched wheel changed the environment: keep the record of the one before it.
  bash "$HYAK_REPO/hyak/env/setup_data_env.sh" --freeze-only \
    || hyak_warn "could not refresh $ANYTOK_ROOT/env/data.freeze.txt (run: bash hyak/env/setup_data_env.sh --freeze-only)"
  # build_wheel.sh appends events only; record the terminal state of this one-shot stage.
  python - <<'PYEOF' || hyak_warn "could not mark the wheel stage complete in its status.json"
from hyak.common.status import StatusWriter
StatusWriter('wheel', 2, 'steps').update(2, stage='complete', state='complete', workers=[],
                                         phases=['build', 'install'])
PYEOF
  hyak_info "wheel: built and installed; provenance in $ANYTOK_ROOT/wheels/provenance.json"
else
  if [ "$rc" -gt 128 ]; then
    job_explain "$rc" "bash hyak/pool/build_wheel.sh build install"      # killed or stopped by a signal
  else
    JOB_MESSAGE="hyak/pool/build_wheel.sh build install failed (exit $rc); run the fetch phase on the login node first, see the job log"
  fi
fi
exit "$rc"
