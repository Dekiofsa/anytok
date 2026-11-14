"""The GPU memory probe, exercised on the CPU (no CUDA here).

* the update function follows the trainer (T:542-584) with a tiny stand-in LM and the real
  helpers of ``runtime/anytok_pool_20k`` (``make_lm_microbatch``, ``input_lexical_loss``; the
  trainer's ``make_optimizer_groups`` / ``set_learning_rates`` executed from their pinned source);
* chunked and pinned updates agree when dropout is off;
* an out-of-memory error in one configuration is recorded, memory is released and the next
  configuration still runs;
* verdict / summary / fit logic, the on-device equivalence check, and an end-to-end
  ``main()`` run on a tiny saved Llama model with the real SmolLM2 tokenizer.
"""
from __future__ import annotations

import copy
import json
import platform
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from hyak.tests.test_memory_common import (  # noqa: F401
    PINNED_CODE, _no_leftover_patch, build_hn,
)
from hyak.memory import chunked_hypernet as ch
from hyak.memory import memory_model as mm
from hyak.memory import probe
from hf_hypernet.modeling_hypernet import ZettHypernet

SOURCE_VOCAB = 300
WIDTH = 64
CPU = torch.device('cpu')


class TinyLM(torch.nn.Module):
    """Stand-in for SmolLM2 with the call signature the trainer uses (T:559-561)."""

    def __init__(self, source):
        super().__init__()
        vocab, width = source.shape
        self.mix = torch.nn.Linear(width, width)
        self.head = torch.nn.Linear(width, vocab, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(source)

    def forward(self, inputs_embeds, attention_mask, position_ids, labels, use_cache):
        assert use_cache is False and inputs_embeds.dtype == torch.bfloat16
        hidden = torch.tanh(self.mix(inputs_embeds)) * attention_mask[..., None]
        hidden = hidden.cumsum(1) / (position_ids[..., None] + 1)   # causal mixing
        logits = self.head(hidden)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


@pytest.fixture(scope='module')
def code():
    return probe.load_code(PINNED_CODE, 'source')


def _models(code, *, seed=51, dropout=True):
    hn, source = build_hn(SOURCE_VOCAB, width=WIDTH, layers=2, heads=4, seed=seed, dtype=torch.float32)
    code.freeze_rescalers(hn)
    if not dropout:
        for module in hn.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    torch.manual_seed(seed)
    lm = TinyLM(source)
    hn_params = [p for p in hn.parameters() if p.requires_grad]
    lm_params = [p for p in lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(code.make_optimizer_groups(hn, 0.0) + code.make_optimizer_groups(lm, 0.0),
                                  betas=(0.9, 0.95), eps=1e-8, fused=True)
    fallback = {f'sym{i}': SOURCE_VOCAB + 1 + i for i in range(3)}
    return SimpleNamespace(lm=lm, hn=hn, source=source, optimizer=optimizer, hn_params=hn_params,
                           lm_params=lm_params, fallback=fallback, source_vocab=SOURCE_VOCAB, width=WIDTH)


def _clone_models(code, models):
    clone = copy.deepcopy(SimpleNamespace(lm=models.lm, hn=models.hn))
    hn_params = [p for p in clone.hn.parameters() if p.requires_grad]
    lm_params = [p for p in clone.lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(code.make_optimizer_groups(clone.hn, 0.0) + code.make_optimizer_groups(clone.lm, 0.0),
                                  betas=(0.9, 0.95), eps=1e-8, fused=True)
    return SimpleNamespace(lm=clone.lm, hn=clone.hn, source=models.source, optimizer=optimizer, hn_params=hn_params,
                           lm_params=lm_params, fallback=models.fallback, source_vocab=SOURCE_VOCAB, width=WIDTH)


# --- helpers taken from the pinned trainer -----------------------------------------------------------
def test_trainer_helpers_from_source_behave_like_the_trainer(code):
    assert code.helpers == 'source'
    trainer = (PINNED_CODE / probe.TRAINER_REL).read_bytes()
    import hashlib
    assert code.files[str(probe.TRAINER_REL)] == hashlib.sha256(trainer).hexdigest() \
        == 'c6406d263fb9cbc8638492d8c77327ce249e215eb74dc43e10335bc95a11ed52'
    assert code.files['hf_hypernet/hyper_roberta.py'] == ch.PINNED_SOURCES['hyper_roberta.py']
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(4)]
    optimizer = torch.optim.SGD([{'params': [p]} for p in params], lr=1.0)
    assert code.set_learning_rates(optimizer, 0) == (0.0, 0.0)
    hn_lr, lm_lr = code.set_learning_rates(optimizer, 1)
    assert hn_lr == pytest.approx(6e-9, rel=1e-12) and lm_lr == pytest.approx(1e-10, rel=1e-12)
    assert [g['lr'] for g in optimizer.param_groups] == [hn_lr, hn_lr, lm_lr, lm_lr]
    assert code.set_learning_rates(optimizer, 19999)[0] == pytest.approx(5.958989115705565e-05, rel=1e-12)
    module = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3))
    module[0].bias.requires_grad_(False)
    decay, no_decay = code.make_optimizer_groups(module, 0.5)
    assert [tuple(p.shape) for p in decay['params']] == [(3, 3)] and decay['weight_decay'] == 0.01
    assert [tuple(p.shape) for p in no_decay['params']] == [(3,), (3,)] and no_decay['weight_decay'] == 0.0
    assert decay['lr'] == no_decay['lr'] == 0.5


def test_trainer_helpers_with_unexpected_globals_are_refused(tmp_path):
    fake = tmp_path / 'trainer.py'
    fake.write_text('def make_optimizer_groups(module, learning_rate):\n    return os.getcwd()\n\n'
                    'def set_learning_rates(optimizer, update):\n    return zett_main_learning_rate(update, 1.0)\n')
    with pytest.raises(probe.ProbeError, match="unexpected globals \\['os'\\]"):
        probe._trainer_helpers_from_source(fake, {'zett_main_learning_rate': None})


def test_load_code_refuses_incomplete_trees_and_reports_rust_failures(tmp_path):
    with pytest.raises(probe.ProbeError, match='incomplete'):
        probe.load_code(tmp_path, 'source')
    with pytest.raises(probe.ProbeError, match='helpers'):
        probe.load_code(PINNED_CODE, 'bogus')
    linux_x86 = sys.platform.startswith('linux') and platform.machine() == 'x86_64'
    if not linux_x86:
        # The committed librust_utils.so is a Linux x86-64 ELF: importing the trainer fails loudly here,
        # and the half-finished import is undone.
        rust_before, path_before = sys.modules.get('rust_utils'), list(sys.path)
        with pytest.raises(probe.ProbeError, match='--helpers source'):
            probe.load_code(PINNED_CODE, 'import')
        assert 'hyak_probe_staged_trainer' not in sys.modules
        assert sys.modules.get('rust_utils') is rust_before and sys.path == path_before


def test_load_code_refuses_a_second_hf_hypernet(tmp_path):
    other = tmp_path / 'tree'
    shutil.copytree(PINNED_CODE / 'hf_hypernet', other / 'hf_hypernet')
    (other / 'scripts').mkdir()
    shutil.copy(PINNED_CODE / probe.TRAINER_REL, other / probe.TRAINER_REL)
    with pytest.raises(probe.ProbeError, match='already imported'):
        probe.load_code(other, 'source')


# --- synthetic inputs and the update --------------------------------------------------------------------
def test_synthetic_inputs_shape_and_content():
    forms, priors, examples = probe.synthetic_inputs(500, source_vocab=SOURCE_VOCAB, fallback_ids=[301, 302, 303],
                                                     seed=3, positions=24)
    pad = SOURCE_VOCAB
    assert forms.shape == (500, 10) and forms.dtype == torch.int64
    assert forms[0, 0] == 0 and bool((forms[0, 1:] == pad).all())
    assert bool((forms[:, 0] != pad).all()) and int(forms.max()) <= 303
    assert bool((forms > pad).any()), 'byte-fallback ids expected'
    single = (forms[:, 1:] == pad).all(dim=1)
    assert 30 < int(single.sum()) < 130
    assert bool((priors['zero'] == 0).all()) and bool((priors['unigram'] < 0).all())
    assert len(examples) == 128
    for prefix, continuation, nbytes in examples:
        assert 1 + len(prefix) + len(continuation) == 24 and max(prefix) < 500 and min(continuation) >= 1
        assert nbytes == 4 * len(continuation)
    again = probe.synthetic_inputs(500, source_vocab=SOURCE_VOCAB, fallback_ids=[301, 302, 303], seed=3,
                                   positions=24)
    assert torch.equal(forms, again[0]) and examples == again[2]


def test_training_update_runs_the_trainer_sequence_on_cpu(code):
    models = _models(code)
    forms, priors, examples = probe.synthetic_inputs(157, source_vocab=SOURCE_VOCAB,
                                                     fallback_ids=sorted(models.fallback.values()), seed=4,
                                                     positions=20)
    before = {n: p.detach().clone() for n, p in models.hn.named_parameters()}
    with ch.patched(ZettHypernet, 40, min_chunk_rows=1):
        metrics = probe.training_update(models, code, forms, priors['unigram'], examples, 5000,
                                        device_type='cpu', microbatch_size=32)
    assert metrics['hn_lr'] == pytest.approx(3e-5) and metrics['lm_lr'] == pytest.approx(5e-7)
    assert metrics['scored_tokens'] == 128 * 10 and metrics['tokenizer_vocab_size'] == 157
    assert metrics['lm_phase_max_allocated_bytes'] is None          # CUDA only
    for key in ('loss_nats_per_token', 'input_lexical_loss', 'gradient_norm_before_clip'):
        assert metrics[key] > 0 and torch.isfinite(torch.tensor(metrics[key]))
    assert all(p.grad is None for p in models.hn.parameters()), 'zero_grad(set_to_none=True) after the step'
    moved = [n for n, p in models.hn.named_parameters() if not torch.equal(p.detach(), before[n])]
    assert 'model.layers.0.intermediate.weight' in moved and 'in_scaler.w' not in moved
    with pytest.raises(ValueError, match='128 examples'):
        probe.training_update(models, code, forms, priors['zero'], examples[:64], 1, device_type='cpu')


def test_chunked_and_pinned_updates_agree_without_dropout(code):
    reference = _models(code, dropout=False)
    candidate = _clone_models(code, reference)
    forms, priors, examples = probe.synthetic_inputs(157, source_vocab=SOURCE_VOCAB,
                                                     fallback_ids=sorted(reference.fallback.values()), seed=5,
                                                     positions=20)
    with ch.patched(ZettHypernet, 0):
        pinned = probe.training_update(reference, code, forms, priors['unigram'], examples, 5000, device_type='cpu')
    with ch.patched(ZettHypernet, 33, min_chunk_rows=1):
        chunked = probe.training_update(candidate, code, forms, priors['unigram'], examples, 5000, device_type='cpu')
    for key in ('loss_nats_per_token', 'input_lexical_loss', 'gradient_norm_before_clip'):
        assert chunked[key] == pytest.approx(pinned[key], rel=2e-2), key
    # The first AdamW moment is (1 - beta1) * clipped gradient: compare them parameter by parameter.
    pairs = [(p, q) for p, q in zip(reference.hn_params, candidate.hn_params)
             if 'exp_avg' in reference.optimizer.state[p]]   # parameters without gradient have no state
    assert {('exp_avg' in candidate.optimizer.state[q]) for _, q in pairs} == {True}
    assert len(pairs) == len(reference.hn_params) - 2, 'only word_embeddings and lang_embedding get no gradient'
    scale = max(float(reference.optimizer.state[p]['exp_avg'].norm()) for p, _ in pairs)
    worst = 0.0
    for p_ref, p_new in pairs:
        ref = reference.optimizer.state[p_ref]['exp_avg']
        new = candidate.optimizer.state[p_new]['exp_avg']
        if float(ref.norm()) > 1e-3 * scale:
            worst = max(worst, float((new - ref).norm() / ref.norm()))
    assert worst <= 5e-2, f'AdamW first moments differ by {worst:.3e} (bf16 autocast)'


# --- configurations, OOM, verdicts --------------------------------------------------------------------
def test_out_of_memory_in_one_configuration_is_recorded_and_released(code, monkeypatch):
    models = _models(code)
    released = []
    real_update, real_release = probe.training_update, probe.release_memory

    def update(models_, code_, forms, *args, **kwargs):
        if forms.shape[0] == 200 and args[-1] == 1:  # the second update at V=200
            raise torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 GiB')
        return real_update(models_, code_, forms, *args, **kwargs)

    monkeypatch.setattr(probe, 'training_update', update)
    monkeypatch.setattr(probe, 'release_memory', lambda m, d: (released.append(d), real_release(m, d)))
    common = dict(updates=2, device=CPU, seed=1, positions=12, microbatch_size=32)
    failed = probe.run_configuration(models, code, vocab=200, chunk_rows=64, **common)
    assert failed['oom'].startswith('OutOfMemoryError in update 2: CUDA out of memory')
    assert failed['error'] is None and len(failed['updates']) == 1
    assert not ch.is_installed(ZettHypernet) and released == [CPU]
    assert all(p.grad is None for p in models.hn.parameters())
    fine = probe.run_configuration(models, code, vocab=150, chunk_rows=0, **common)
    assert fine['oom'] is None and fine['error'] is None and len(fine['updates']) == 2
    assert [u['priors'] for u in fine['updates']] == ['zero', 'unigram']
    judged = probe.verdict(failed, capacity_bytes=0, min_headroom_gib=8, device_type='cpu')
    assert judged['verdict'] == 'DOES NOT FIT' and judged['line'].startswith('VERDICT V=200 chunk_rows=64: DOES NOT FIT')


def test_errors_other_than_oom_are_reported_as_errors(code, monkeypatch):
    models = _models(code)

    def broken(*args, **kwargs):
        raise FloatingPointError('Non-finite LM loss')

    monkeypatch.setattr(probe, 'training_update', broken)
    result = probe.run_configuration(models, code, vocab=100, chunk_rows=32, updates=2, device=CPU, seed=0,
                                     positions=12, microbatch_size=32)
    assert result['oom'] is None and result['error'].startswith('FloatingPointError in update 1')
    assert probe.verdict(result, capacity_bytes=0, min_headroom_gib=8, device_type='cuda')['verdict'] == 'ERROR'
    assert probe.is_out_of_memory(RuntimeError('CUDA error: out of memory'))
    assert probe.is_out_of_memory(RuntimeError('CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate'))
    assert not probe.is_out_of_memory(RuntimeError('shape mismatch'))


def _cuda_result(vocab, rows, reserved_gib, allocated_gib, seconds=5.0, lm_gib=None, first_seconds=None):
    lm = int((lm_gib if lm_gib is not None else allocated_gib) * probe.GIB)
    updates = [{'seconds': seconds if i else (first_seconds or seconds),
                'max_reserved_bytes': int(reserved_gib * probe.GIB) - 1000 * i,
                'max_allocated_bytes': int(allocated_gib * probe.GIB) - 1000 * i,
                'lm_phase_max_allocated_bytes': lm - 1000 * i} for i in range(2)]
    return {'vocab': vocab, 'chunk_rows': rows, 'label': f'V={vocab} ' + (f'chunk_rows={rows}' if rows else 'pinned'),
            'updates': updates, 'oom': None, 'error': None, 'prediction': mm.predict_peak_gib(vocab, rows)}


def test_prediction_takes_the_larger_of_the_two_phases():
    pinned = mm.predict_peak_gib(32768, 0)
    assert pinned['gib'] == 72.2 and pinned['phase'] == mm.PHASE_HYPERNETWORK      # the B300 anchor
    assert pinned['lm_gib'] < pinned['hypernetwork_gib']
    chunked = mm.predict_peak_gib(128000, 16384)
    assert chunked['phase'] == mm.PHASE_LM and chunked['gib'] == chunked['lm_gib'] == 90.9
    assert chunked['hypernetwork_gib'] == 82.7
    assert [mm.predict_peak_gib(v, 16384)['gib'] for v in (32768, 64000, 96000)] == [58.2, 68.9, 79.9]
    # smaller chunks do not lower an LM-bound peak
    assert mm.predict_peak_gib(128000, 8192)['gib'] == chunked['gib']
    assert mm.predict_peak_gib(128000, 16384, first_update=True)['gib'] == pytest.approx(90.9 - 13.82, abs=0.11)
    assert chunked['low_gib'] < chunked['gib'] < chunked['high_gib']
    with pytest.raises(ValueError):
        mm.predict_peak_gib(0, 16384)
    with pytest.raises(ValueError):
        mm.predict_peak_gib(1000, -1)
    assert {row['chunk_rows'] for row in mm.prediction_table()} == {0, 32768, 16384, 8192}


def test_verdicts_summary_and_fit():
    capacity = int(139.8 * probe.GIB)
    fits = probe.verdict(_cuda_result(128000, 16384, 84.0, 80.0, first_seconds=17.4, lm_gib=80.0),
                         capacity_bytes=capacity, min_headroom_gib=8, device_type='cuda')
    assert fits['verdict'] == 'FITS' and fits['headroom_gib'] == pytest.approx(55.8, abs=0.01)
    assert fits['line'].startswith('VERDICT V=128000 chunk_rows=16384: FITS (peak reserved 84.0 GiB')
    assert 'predicted 90.9 GiB (LM micro-batches)' in fits['line']
    # the last update's time, not the slowest (the first includes CUDA start-up)
    assert fits['seconds_per_update'] == 5.0 and fits['first_update_seconds'] == 17.4
    assert '5.0 s/update (first update 17.4 s)' in fits['line']
    assert fits['peak_phase'] == mm.PHASE_LM and 'peak set by the LM micro-batches' in fits['line']
    backward = probe.verdict(_cuda_result(128000, 16384, 84.0, 80.0, lm_gib=70.0), capacity_bytes=capacity,
                             min_headroom_gib=8, device_type='cuda')
    assert backward['peak_phase'] == mm.PHASE_HYPERNETWORK
    tight = probe.verdict(_cuda_result(128000, 16384, 135.0, 130.0), capacity_bytes=capacity,
                          min_headroom_gib=8, device_type='cuda')
    assert tight['verdict'] == 'DOES NOT FIT' and 'below the required 8 GiB' in tight['line']
    results = [_cuda_result(v, 16384, 30 + 0.41 * v / 1000, 29 + 0.4 * v / 1000) for v in (32768, 64000, 96000)]
    report = {'verdicts': [probe.verdict(r, capacity_bytes=capacity, min_headroom_gib=8, device_type='cuda')
                           for r in results],
              'equivalence': {'passed': True}}
    summary = probe._summarize(report, 16384, capacity, 8, 'cuda')
    assert summary['ok'] is True
    assert summary['fit']['gib_per_1000_rows'] == pytest.approx(0.41, rel=1e-3)
    assert summary['fit']['largest_fitting_vocab'] == pytest.approx((139.8 - 8 - 30) / 0.41 * 1000, rel=1e-3)
    assert summary['summary'].startswith('SUMMARY chunk_rows=16384: FITS for every measured vocabulary up to 96000')
    report['equivalence'] = {'passed': False}
    assert probe._summarize(report, 16384, capacity, 8, 'cuda')['ok'] is False
    report['equivalence'] = None
    report['verdicts'].append(tight)
    failing = probe._summarize(report, 16384, capacity, 8, 'cuda')
    assert failing['ok'] is False and 'DOES NOT FIT or failed at V=128000' in failing['summary']
    assert 'HYAK_HN_CHUNK_ROWS does not change' in failing['summary']      # the LM phase set that peak
    tight_backward = probe.verdict(_cuda_result(128000, 16384, 135.0, 130.0, lm_gib=100.0), capacity_bytes=capacity,
                                   min_headroom_gib=8, device_type='cuda')
    report['verdicts'][-1] = tight_backward
    failing = probe._summarize(report, 16384, capacity, 8, 'cuda')
    assert 'lower HYAK_HN_CHUNK_ROWS (not below 4096)' in failing['summary']
    assert probe.fit_linear([(1, 2.0)]) is None


def test_only_the_configured_setting_and_the_equivalence_decide():
    """memory-5: a failing pinned comparison is information only."""
    capacity = int(139.8 * probe.GIB)
    configured = [probe.verdict(_cuda_result(v, 16384, 60 + v / 4000, 58), capacity_bytes=capacity,
                                min_headroom_gib=8, device_type='cuda') for v in (32768, 64000, 96000, 128000)]
    pinned_error = {'vocab': 64000, 'chunk_rows': 0, 'label': 'V=64000 pinned', 'updates': [], 'oom': None,
                    'error': 'RuntimeError in update 1: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED', 'prediction': None}
    pinned_oom = dict(pinned_error, vocab=96000, label='V=96000 pinned', error=None,
                      oom='OutOfMemoryError in update 1: CUDA out of memory')
    others = [probe.verdict(r, capacity_bytes=capacity, min_headroom_gib=8, device_type='cuda')
              for r in (pinned_error, pinned_oom)]
    report = {'verdicts': configured + others, 'equivalence': {'passed': True}}
    summary = probe._summarize(report, 16384, capacity, 8, 'cuda')
    assert summary['ok'] is True, summary
    assert 'FITS for every measured vocabulary up to 128000' in summary['summary']
    assert '1 comparison measurement(s) raised errors (information only): V=64000 pinned' in summary['summary']
    configured[1] = probe.verdict(dict(_cuda_result(64000, 16384, 70, 68), updates=[],
                                       error='RuntimeError in update 1: CUBLAS_STATUS_EXECUTION_FAILED'),
                                  capacity_bytes=capacity, min_headroom_gib=8, device_type='cuda')
    report['verdicts'] = configured + others
    assert probe._summarize(report, 16384, capacity, 8, 'cuda')['ok'] is False


def test_configured_setting_is_measured_first_and_equivalence_uses_its_chunk_count():
    assert probe.equivalence_chunk_rows(4096, 16384, [32768, 128000]) == 512        # 8 chunks, as at 128,000
    assert probe.equivalence_chunk_rows(4096, 0, [128000]) == 1366                  # pinned: 3 chunks
    assert probe.equivalence_chunk_rows(4096, 65536, [32768]) == 1366               # at least 3 chunks
    assert probe.equivalence_chunk_rows(200, 128, [200, 400]) == 50   # 4 chunks at 400


def test_compare_skips_analytically_zero_key_bias_gradients():
    reference = (torch.ones(4, 2), None, {'a.weight': torch.ones(3) * 10, 'a.key.bias': torch.full((3,), 1e-6),
                                          'tiny.bias': torch.full((3,), 1e-3)})
    candidate = (torch.ones(4, 2), None, {'a.weight': torch.ones(3) * 10.01, 'a.key.bias': torch.full((3,), -1e-6),
                                          'tiny.bias': torch.full((3,), 2e-3)})
    result = probe._compare(reference, candidate)
    assert result['output'] == 0.0 and result['worst_parameter'] == 'a.weight'
    assert result['gradient_worst'] == pytest.approx(1e-3, rel=1e-3)
    assert result['gradient'] == pytest.approx(float(torch.tensor(0.01 * 3 ** 0.5)) / (300 + 6e-12 + 3e-6) ** 0.5,
                                               rel=1e-2)
    assert probe._compare(reference, (candidate[0], None, {'a.weight': torch.ones(3)}))['gradient'] is None
    assert not probe._within({'output': None, 'gradient': 0.0}, {'output': 1.0, 'gradient': 1.0})
    assert not probe._within({'output': float('nan')}, {'output': 1.0})


def test_jsonable_replaces_non_finite_numbers():
    assert probe.jsonable({'a': [1.0, float('nan'), {'b': float('inf')}], 'c': 'x'}) == \
        {'a': [1.0, 'nan', {'b': 'inf'}], 'c': 'x'}


# --- equivalence on the device (CPU here) ------------------------------------------------------------
@pytest.mark.parametrize('seed', [61, 62, 63])
def test_equivalence_check_passes_on_cpu_and_catches_a_broken_mirror(code, monkeypatch, seed):
    models = _models(code, seed=seed)
    result = probe.equivalence_check(models, code, rows=157, chunk_rows=40, device=CPU, seed=seed)
    assert result['passed'], result
    assert result['float32']['output'] < 1e-5 and result['float32']['gradient'] < 1e-5
    assert result['float32']['gradient_worst'] < 1e-4
    assert not result['float32']['worst_parameter'].endswith('.key.bias')
    for key, limit in result['bf16_limits'].items():
        assert result['bfloat16'][key] <= limit / 2, f'bf16 {key} has less than a 2x margin: {result}'
    assert result['dropout_forward_reproducible'] and result['rng_continues_after_backward']
    real = ch.chunked_call

    def broken(*args, **kwargs):
        first, second, third = real(*args, **kwargs)
        return first * 1.01, second, third

    monkeypatch.setattr(ch, 'chunked_call', broken)
    result = probe.equivalence_check(models, code, rows=157, chunk_rows=40, device=CPU, seed=3)
    assert not result['passed'] and result['float32']['output'] > 5e-3


def test_equivalence_check_catches_wrong_masks_in_recomputation(code, monkeypatch):
    """memory-3: recomputation that redraws other dropout masks must fail the check."""
    models = _models(code, seed=64)
    clean = probe.equivalence_check(models, code, rows=157, chunk_rows=40, device=CPU, seed=64)
    checked = clean['dropout_checkpoint_vs_direct']
    assert clean['passed'] and checked['passed'], clean
    for kind in ('float32', 'bfloat16'):
        assert checked[kind]['outputs_equal'] and checked[kind]['gradient'] <= 1e-6, checked

    def reseeding(stage):
        def wrapper(*args, **kwargs):
            if torch._C._current_graph_task_id() != -1:      # inside backward: recomputation
                torch.manual_seed(12345)
            return stage(*args, **kwargs)
        return wrapper

    for name in ('_stage_embed', '_stage_attention', '_stage_up_ffn'):
        monkeypatch.setattr(ch, name, reseeding(getattr(ch, name)))
    broken = probe.equivalence_check(models, code, rows=157, chunk_rows=40, device=CPU, seed=64)
    assert broken['dropout_forward_reproducible'] and broken['rng_continues_after_backward']   # the old checks pass
    assert not broken['dropout_checkpoint_vs_direct']['passed'] and not broken['passed']
    assert broken['dropout_checkpoint_vs_direct']['float32']['gradient'] > 1e-2


# --- end to end on a tiny saved model --------------------------------------------------------------------
@pytest.fixture
def tiny_inputs(tmp_path, scratch_assets):
    from transformers import LlamaConfig, LlamaForCausalLM
    from hf_hypernet.identity_warmup import build_input_hypernet, initialize_rescalers
    tokenizer = scratch_assets / 'full' / 'tokenizer.json'
    if not tokenizer.is_file():
        pytest.skip(f'{tokenizer} not found')
    model_dir = tmp_path / 'model'
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=49152, hidden_size=WIDTH, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                         tie_word_embeddings=True, bos_token_id=0, eos_token_id=0)
    lm = LlamaForCausalLM(config)
    lm.save_pretrained(model_dir, safe_serialization=True)
    shutil.copy(tokenizer, model_dir / 'tokenizer.json')
    hn = build_input_hypernet(49152, WIDTH, attention_heads=4)
    initialize_rescalers(hn, lm.get_input_embeddings().weight.detach())
    hn.save_pretrained(tmp_path / 'hypernet', safe_serialization=True)
    return model_dir, tmp_path / 'hypernet'


@pytest.mark.slow
def test_probe_main_end_to_end_on_cpu(tiny_inputs, make_tmp_config, capsys):
    model_dir, hypernet_dir = tiny_inputs
    tmp = make_tmp_config(HYAK_PLATFORM='test platform')
    report_path = tmp.base / 'report.json'
    code = probe.load_code(PINNED_CODE, 'source')  # already imported from PINNED_CODE in this process
    assert code.ZettHypernet is ZettHypernet
    threads = torch.get_num_threads()
    try:
        exit_code = probe.main(['--model-dir', str(model_dir), '--hypernet', str(hypernet_dir),
                                '--code-dir', str(PINNED_CODE), '--device', 'cpu', '--helpers', 'source',
                                '--vocab', '200', '400', '--chunk-rows', '128', '--also-pinned',
                                '--equivalence-rows', '200', '--positions', '8', '--report', str(report_path)])
        assert torch.get_num_threads() == 1, 'the probe sets one CPU thread like the trainer (T:367)'
    finally:
        torch.set_num_threads(threads)
    output = capsys.readouterr().out
    assert exit_code == 0, output
    lines = [line for line in output.splitlines() if 'VERDICT' in line]
    assert len(lines) == 4, output
    assert '[probe] VERDICT V=200 chunk_rows=128: MEASURED' in output
    assert '[probe] VERDICT V=400 pinned: MEASURED' in output
    assert '[probe] EQUIVALENCE PASS' in output and '[probe] SUMMARY chunk_rows=128: MEASURED' in output
    assert 'WARNING: chunk_rows=128 is below the floor of 4096 rows' in output
    report = json.loads(report_path.read_text())
    assert report['ok'] is True and report['settings']['chunk_settings'] == [128, 0]
    assert report['equivalence']['passed'] is True and report['equivalence']['chunk_rows'] == 50
    assert report['equivalence']['dropout_checkpoint_vs_direct']['passed'] is True
    assert [m['label'] for m in report['measurements']] == ['V=200 chunk_rows=128', 'V=400 chunk_rows=128',
                                                             'V=200 pinned', 'V=400 pinned']
    assert all(len(m['updates']) == 2 for m in report['measurements'])
    assert report['chunked_module']['module_sha256'] == ch.module_sha256()
    archive = list(tmp.paths.probe_dir.glob('probe-report-*.json'))
    assert len(archive) == 1 and json.loads(archive[0].read_text()) == report
    status = json.loads((tmp.paths.monitor_dir('probe') / 'status.json').read_text())
    assert status['state'] == 'complete' and status['stage'] == 'complete'
    assert status['completed'] == status['total'] == 5 and status['platform'] == 'test platform'
    events = [json.loads(line) for line in (tmp.paths.monitor_dir('probe') / 'events.jsonl').read_text().splitlines()]
    assert sum(1 for e in events if e.get('event') == 'measurement') == 4
    assert sum(1 for e in events if e.get('event') == 'equivalence') == 1
    assert not ch.is_installed(ZettHypernet)


def test_probe_main_reports_setup_errors(make_tmp_config, tmp_path, capsys):
    tmp = make_tmp_config()
    exit_code = probe.main(['--model-dir', str(tmp_path / 'missing'), '--hypernet', str(tmp_path / 'missing'),
                            '--code-dir', str(PINNED_CODE), '--device', 'cpu', '--helpers', 'source',
                            '--vocab', '100'])
    assert exit_code == 2
    assert 'lacks' in capsys.readouterr().err
    status = json.loads((tmp.paths.monitor_dir('probe') / 'status.json').read_text())
    assert status['state'] == 'failed' and 'lacks' in status['error']
    if not torch.cuda.is_available():
        assert probe.main(['--device', 'cuda', '--vocab', '100']) == 2


def _canned_report(verdicts):
    return {'ok': False, 'summary': 'SUMMARY chunk_rows=16384: DOES NOT FIT or failed at V=127000',
            'equivalence': None, 'measurements': [], 'verdicts': verdicts}


@pytest.mark.parametrize('verdicts, stage', [
    ([{'label': 'V=127000 chunk_rows=16384', 'verdict': 'DOES NOT FIT', 'peak_reserved_gib': 139.0,
       'headroom_gib': 0.8, 'seconds_per_update': 3.0, 'chunk_rows': 16384, 'vocab': 127000}], 'does not fit'),
    ([], 'failed'),
])
def test_probe_main_exits_4_when_the_configuration_does_not_fit(make_tmp_config, monkeypatch, verdicts, stage):
    """tests-7 / exit-code contract: does not fit or equivalence failure = 4."""
    tmp = make_tmp_config()
    monkeypatch.setattr(probe, 'run_probe', lambda args, **kwargs: _canned_report(verdicts))
    assert probe.main(['--device', 'cpu']) == 4
    status = json.loads((tmp.paths.monitor_dir('probe') / 'status.json').read_text())
    assert status['state'] == 'failed' and status['stage'] == stage
    assert 'DOES NOT FIT' in status['error']


def test_probe_main_reports_unexpected_loading_errors_as_setup_errors(make_tmp_config, tmp_path, monkeypatch, capsys):
    """memory-v1: an unexpected exception while loading exits 2 (setup) with a traceback, not 4."""
    tmp = make_tmp_config()

    def broken(*args, **kwargs):
        raise RuntimeError()                                 # an empty message must not break the report

    monkeypatch.setattr(probe, 'load_models', broken)
    exit_code = probe.main(['--model-dir', str(tmp_path), '--hypernet', str(tmp_path), '--code-dir', str(PINNED_CODE),
                            '--device', 'cpu', '--helpers', 'source', '--vocab', '100', '--equivalence-rows', '0'])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert 'Traceback' in err and 'unexpected' in err
    status = json.loads((tmp.paths.monitor_dir('probe') / 'status.json').read_text())
    assert status['state'] == 'failed' and status['stage'] == 'failed' and 'RuntimeError' in status['error']


def test_probe_refuses_a_configured_chunk_below_the_floor_on_cuda(make_tmp_config, monkeypatch):
    tmp = make_tmp_config()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(probe, 'load_code', lambda *a, **k: pytest.fail('must stop before loading'))
    assert probe.main(['--device', 'cuda', '--chunk-rows', '1024', '--vocab', '100']) == 2
    status = json.loads((tmp.paths.monitor_dir('probe') / 'status.json').read_text())
    assert 'below the floor of 4096' in status['error']


def test_load_code_refuses_a_tree_that_differs_from_the_pins(tmp_path):
    """env-10: an edited staged tree is refused before anything is imported."""
    tree = tmp_path / 'tree'
    shutil.copytree(PINNED_CODE / 'hf_hypernet', tree / 'hf_hypernet')
    (tree / 'scripts').mkdir()
    shutil.copy(PINNED_CODE / probe.TRAINER_REL, tree / probe.TRAINER_REL)
    target = tree / 'hf_hypernet' / 'joint_training.py'
    target.write_text(target.read_text() + '\nTAMPERED = True\n')
    with pytest.raises(probe.ProbeError, match='differs from the pinned snapshot.*joint_training.py'):
        probe.load_code(tree, 'source')
    assert 'TAMPERED' not in vars(sys.modules.get('hf_hypernet.joint_training', object()))
    pins = probe.pinned_hashes()
    assert pins[str(probe.TRAINER_REL)] == 'c6406d263fb9cbc8638492d8c77327ce249e215eb74dc43e10335bc95a11ed52'
    assert all(f'hf_hypernet/{name}' in pins for name in probe.HF_FILES)


def test_failed_backward_on_cpu_releases_the_chunk_graph(code, monkeypatch):
    """memory-9: after a failure inside one region's recomputation no checkpoint frame stays alive."""
    import gc
    models = _models(code, seed=65)
    real = ch._stage_up_ffn

    def failing(layer, deterministic, index, chunk, *args):
        if torch._C._current_graph_task_id() != -1 and index == 1 and chunk == 1:
            raise torch.OutOfMemoryError('simulated out of memory during recomputation')
        return real(layer, deterministic, index, chunk, *args)

    monkeypatch.setattr(ch, '_stage_up_ffn', failing)
    result = probe.run_configuration(models, code, vocab=200, chunk_rows=40, updates=1, device=CPU, seed=2,
                                     positions=12, microbatch_size=32)
    assert result['oom'] and result['oom'].startswith('OutOfMemoryError in update 1')
    gc.collect()
    frames = [o for o in gc.get_objects() if type(o).__name__ == '_CheckpointFrame']
    assert frames == [], f'{len(frames)} checkpoint frames still alive'
