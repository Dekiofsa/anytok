"""TEST-ONLY harness hook (hyak e2e): put this directory on PYTHONPATH and set HYAK_E2E_CPU_SHIM=1.

Installs ``hyak_e2e_cpu_shim`` (CUDA-to-CPU shim for the pinned trainers, test parameters for
``python -m hyak.train.run_segment``).  Does nothing without ``HYAK_E2E_CPU_SHIM=1``.
"""
import os

if os.environ.get('HYAK_E2E_CPU_SHIM') == '1':
    import hyak_e2e_cpu_shim

    hyak_e2e_cpu_shim.install()
