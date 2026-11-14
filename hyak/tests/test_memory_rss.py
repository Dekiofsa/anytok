"""Chunking bounds the hypernetwork's backward memory (DESIGN.md 5.D test 5).

Each measurement runs in its own subprocess (``python -m hyak.tests.test_memory_rss --worker``)
so peaks do not mix: one forward + backward of a float32 hypernetwork (width 256, 4 heads,
3 layers, dropout on) over V = 6,000 rows on the CPU.

* ``rss``: peak resident set size above the pre-run baseline (``ru_maxrss`` and a 2 ms psutil
  sampler) for the pinned forward and for the chunked forward with V/8 and V/4 rows per chunk.
* ``profile``: exact live tensor bytes from the profiler's allocation events, with a marker at
  every checkpoint recomputation.  Per recomputed region the growth above the live bytes at its
  start must stay within one standalone region's forward+backward cost, and the live bytes at
  the start of each region must not grow (earlier regions' recomputed tensors were freed): at most
  one chunk region's graph is alive at any time.

Both are marked ``slow`` (about 40 s, peak about 2.5 GB in the pinned subprocess).  On macOS the
physical footprint (``proc_pid_rusage``: includes pages compressed under memory pressure) is the
primary peak measure; RSS is reported and checked with a looser bound.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import pytest

REPO = Path(__file__).resolve().parents[2]
VOCAB = 6000
WIDTH = 256
SOURCE_VOCAB = 1000


def _setup(vocab, width):
    import torch
    from hyak.tests.test_memory_common import build_hn, make_forms, make_priors
    hn, source = build_hn(SOURCE_VOCAB, width=width, layers=3, heads=width // 64, seed=41,
                          dtype=torch.float32)
    forms = make_forms(vocab, SOURCE_VOCAB, seed=42)
    priors = make_priors(vocab, 'unigram')
    cotangent = torch.randn(vocab, width, generator=torch.Generator().manual_seed(43))
    return hn, source, forms, priors, cotangent


def _call(mode, rows, hn, forms, priors, source):
    from hyak.memory import chunked_hypernet as ch
    from hyak.tests.test_memory_common import pinned_call
    if mode == 'pinned':
        return pinned_call()(hn, forms, priors, source, None, False)
    return ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=rows)


def _macos_footprint():
    """``(current, lifetime maximum)`` physical footprint of this process on macOS, else None.

    Unlike RSS, the footprint counts pages the kernel compressed under memory pressure, so it
    is a stable peak measure while other processes compete for RAM (proc_pid_rusage, RUSAGE_INFO_V4).
    """
    if sys.platform != 'darwin':
        return None
    import ctypes

    names = ('user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins', 'wired_size',
             'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime', 'child_user_time',
             'child_system_time', 'child_pkg_idle_wkups', 'child_interrupt_wkups', 'child_pageins',
             'child_elapsed_abstime', 'diskio_bytesread', 'diskio_byteswritten', 'qos_default',
             'qos_maintenance', 'qos_background', 'qos_utility', 'qos_legacy', 'qos_user_initiated',
             'qos_user_interactive', 'billed_system_time', 'serviced_system_time', 'logical_writes',
             'lifetime_max_phys_footprint', 'instructions', 'cycles', 'billed_energy', 'serviced_energy',
             'interval_max_phys_footprint', 'runnable_time')

    class RusageInfoV4(ctypes.Structure):
        _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in names]

    try:
        library = ctypes.CDLL('/usr/lib/libproc.dylib')
        info = RusageInfoV4()
        if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
    except OSError:
        return None
    return info.phys_footprint, info.lifetime_max_phys_footprint


def _worker_rss(mode, rows, vocab, width):
    import psutil
    import torch
    hn, source, forms, priors, cotangent = _setup(vocab, width)
    # Warm-up on a few rows so lazily loaded kernels and thread pools are resident before the baseline.
    out, _, _ = _call(mode, max(1, min(rows, 16)) if rows else 0, hn, forms[:64], priors[:64], source)
    (out * cotangent[:64]).sum().backward()
    hn.zero_grad(set_to_none=True)
    del out
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    footprint_before = _macos_footprint()
    sampled = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            sampled[0] = max(sampled[0], process.memory_info().rss)
            time.sleep(0.002)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    out, _, _ = _call(mode, rows, hn, forms, priors, source)
    (out * cotangent).sum().backward()
    seconds = time.perf_counter() - started
    stop.set()
    thread.join()
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss *= 1 if sys.platform == 'darwin' else 1024  # bytes on macOS, KiB on Linux
    footprint_after = _macos_footprint()
    finite = bool(torch.isfinite(out).all())
    result = {'mode': mode, 'rows': rows, 'vocab': vocab, 'width': width, 'baseline_bytes': baseline,
              'peak_rss_bytes': max(maxrss, sampled[0]) - baseline, 'sampled_peak_bytes': sampled[0] - baseline,
              'seconds': round(seconds, 3), 'finite': finite, 'peak_footprint_bytes': None}
    if footprint_before and footprint_after:
        result['peak_footprint_bytes'] = footprint_after[1] - footprint_before[0]
    return result


def _live_timeline(prof):
    events = prof.profiler.kineto_results.events()
    memory = sorted((e.start_ns(), e.nbytes()) for e in events if e.name() == '[memory]')
    markers = sorted((e.start_ns(), e.name()) for e in events if e.name().startswith('hyak-recompute/'))
    return memory, markers


def _live_at(memory, start_index, t_ns, current):
    while start_index < len(memory) and memory[start_index][0] < t_ns:
        current += memory[start_index][1]
        start_index += 1
    return start_index, current


def _region_cost(function, inputs, cotangent_shape):
    import torch
    from torch.profiler import ProfilerActivity, profile
    cotangent = torch.randn(cotangent_shape, generator=torch.Generator().manual_seed(44))
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        out = function(*inputs)
        (out * cotangent).sum().backward()
        del out
    memory, _ = _live_timeline(prof)
    current = peak = 0
    for _, nbytes in memory:
        current += nbytes
        peak = max(peak, current)
    return peak


def _worker_profile(rows, vocab, width):
    import functools
    import torch
    from torch.nn import functional as F
    from torch.profiler import ProfilerActivity, profile, record_function
    from hyak.memory import chunked_hypernet as ch
    hn, source, forms, priors, cotangent = _setup(vocab, width)

    def marker(region, layer, chunk, recompute):
        if recompute:
            with record_function(f'hyak-recompute/{region}/{layer}/{chunk}'):
                pass

    gc.collect()
    with ch.observe(marker), profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        out, _, _ = ch.chunked_call(hn, forms, priors, source, None, False, chunk_rows=rows)
        (out * cotangent).sum().backward()
        del out
    memory, markers = _live_timeline(prof)
    index, current, peak = 0, 0, 0
    for _, nbytes in memory:
        current += nbytes
        peak = max(peak, current)
    segments = []
    index, current = 0, 0
    bounds = [t for t, _ in markers] + [float('inf')]
    for k, (t_ns, name) in enumerate(markers):
        index, current = _live_at(memory, index, t_ns, current)
        start_live = current
        segment_peak = current
        while index < len(memory) and memory[index][0] < bounds[k + 1]:
            current += memory[index][1]
            segment_peak = max(segment_peak, current)
            index += 1
        segments.append({'region': name, 'start_live': start_live, 'growth': segment_peak - start_live})
    # Standalone cost of the three region kinds on one chunk (forward + backward, no checkpoint).
    width_rows = min(rows, vocab)
    layer = hn.model.layers[-1]
    mask = F.pad(forms[:width_rows] != hn.pad_token_id, (0, 1), value=1)
    with torch.no_grad():
        h = ch._stage_embed(hn, False, 0, forms[:width_rows], source, priors[:width_rows])
    h.requires_grad_(True)
    blocks = torch.randn(16, width, requires_grad=True)
    costs = {
        'embed': _region_cost(functools.partial(ch._stage_embed, hn, False, 0),
                              (forms[:width_rows], source, priors[:width_rows]), (width_rows, 11, width)),
        'attention': _region_cost(functools.partial(ch._stage_attention, layer, False, 2, 0), (h, mask),
                                  (width_rows, 11, width)),
        'up_ffn': _region_cost(functools.partial(ch._stage_up_ffn, layer, False, 2, 0, False), (h, blocks),
                               (width_rows, 11, width)),
    }
    return {'rows': rows, 'vocab': vocab, 'width': width, 'peak_live_bytes': peak, 'segments': segments,
            'region_costs': costs, 'unit_bytes': width_rows * 11 * width * 4}


def _run_worker(*args, timeout=600):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
    command = [sys.executable, '-m', 'hyak.tests.test_memory_rss', '--worker', *map(str, args)]
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise AssertionError(f'worker {args} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
    lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
    return json.loads(lines[-1])


@pytest.mark.slow
def test_chunked_peak_rss_is_bounded_by_chunk_size():
    # psutil is installed in the cluster's train environment (accelerate 1.14.0 requires it); skip only in
    # an environment without it.
    pytest.importorskip('psutil', reason='psutil is not installed in this environment')
    runs = {'pinned': _run_worker('rss', 'pinned', 0, VOCAB, WIDTH),
            'V/8': _run_worker('rss', 'chunked', VOCAB // 8, VOCAB, WIDTH),
            'V/4': _run_worker('rss', 'chunked', VOCAB // 4, VOCAB, WIDTH)}
    assert all(run['finite'] for run in runs.values())
    rss = {name: run['peak_rss_bytes'] / 2**20 for name, run in runs.items()}
    rss_ratio = {name: rss[name] / rss['pinned'] for name in ('V/8', 'V/4')}
    message = (f'\npeak RSS above baseline (MiB): {json.dumps({k: round(v, 1) for k, v in rss.items()})}; '
               f'ratios to pinned: {json.dumps({k: round(v, 3) for k, v in rss_ratio.items()})}')
    # RSS shrinks when macOS compresses pages under memory pressure (other processes); the physical
    # footprint does not, so it is the primary measure where available.
    if all(run['peak_footprint_bytes'] is not None for run in runs.values()):
        primary = {name: run['peak_footprint_bytes'] / 2**20 for name, run in runs.items()}
        message += (f'; peak physical footprint (MiB): {json.dumps({k: round(v, 1) for k, v in primary.items()})}; '
                    f'ratios: {json.dumps({k: round(primary[k] / primary["pinned"], 3) for k in ("V/8", "V/4")})}')
    else:
        primary = rss
    print(message)
    assert 1000 < primary['pinned'] < 3500, f'pinned peak {primary["pinned"]:.0f} MiB outside the intended range'
    assert primary['V/8'] <= 0.55 * primary['pinned'], 'chunked (V/8) peak is not clearly below the pinned peak'
    assert primary['V/8'] < primary['V/4'] < primary['pinned'], primary
    assert rss_ratio['V/8'] <= 0.7, f'chunked (V/8) peak RSS is {rss_ratio["V/8"]:.2f} of the pinned peak RSS'


@pytest.mark.slow
def test_at_most_one_chunk_region_graph_is_alive_during_backward():
    rows = VOCAB // 8
    report = _run_worker('profile', 'chunked', rows, VOCAB, WIDTH)
    segments = report['segments']
    costs = report['region_costs']
    unit = report['unit_bytes']
    assert len(segments) == 8 * (1 + 2 * 3), 'every region of every chunk is recomputed exactly once'
    first_start = segments[0]['start_live']
    worst_growth = max(s['growth'] / costs[s['region'].split('/')[1]] for s in segments)
    worst_start = max(s['start_live'] - first_start for s in segments) / unit
    print(f'\nchunked rows={rows}: peak live {report["peak_live_bytes"] / 2**20:.1f} MiB; region costs (MiB) '
          f'{json.dumps({k: round(v / 2**20, 1) for k, v in costs.items()})}; worst growth / region cost '
          f'{worst_growth:.2f}; worst start-live increase {worst_start:.2f} chunk units')
    assert worst_growth <= 1.3, 'a region grew more than one region graph: graphs of several chunks alive'
    assert worst_start <= 2.0, 'live memory accumulated across regions: recomputed tensors were not freed'
    assert min(s['growth'] for s in segments if s['region'].startswith('hyak-recompute/up_ffn')) \
        >= 0.5 * costs['up_ffn'], 'recomputation was expected to rebuild the region graph'


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', required=True)
    parser.add_argument('measure', choices=['rss', 'profile'])
    parser.add_argument('mode', choices=['pinned', 'chunked'])
    parser.add_argument('rows', type=int)
    parser.add_argument('vocab', type=int)
    parser.add_argument('width', type=int)
    args = parser.parse_args(argv)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if args.measure == 'rss':
        result = _worker_rss(args.mode, args.rows, args.vocab, args.width)
    else:
        result = _worker_profile(args.rows, args.vocab, args.width)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    _main()
