"""The class-level patch and its refusals (DESIGN.md 5.D test 3)."""
from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import sys

import pytest
import torch

from hyak.tests.test_memory_common import (  # noqa: F401
    PINNED_CODE, _no_leftover_patch, build_hn, make_forms, make_priors, pinned_call,
)
from hyak.memory import chunked_hypernet as ch
from hf_hypernet import hyper_roberta
from hf_hypernet.modeling_hypernet import ZettHypernet

SOURCE_VOCAB = 200
V = 33


@pytest.fixture
def small():
    hn, source = build_hn(SOURCE_VOCAB, width=32, layers=2, heads=2, seed=31, dtype=torch.float32)
    forms = make_forms(V, SOURCE_VOCAB, seed=32)
    priors = make_priors(V, 'unigram')
    return hn, source, forms, priors


@pytest.fixture
def spy(monkeypatch):
    """Replace chunked_call by a recorder that delegates to the real function."""
    calls = []
    real = ch.chunked_call

    def recorder(*args, **kwargs):
        calls.append(kwargs.get('chunk_rows'))
        return real(*args, **kwargs)

    monkeypatch.setattr(ch, 'chunked_call', recorder)
    return calls


def test_install_uninstall_idempotence_and_original_kept():
    original = ZettHypernet.__dict__['__call__']
    assert not ch.is_installed(ZettHypernet)
    assert ch.uninstall(ZettHypernet) is False
    ch.install(ZettHypernet, 16, min_chunk_rows=1)
    try:
        patched_call = ZettHypernet.__dict__['__call__']
        assert ch.is_installed(ZettHypernet) and patched_call is not original
        assert patched_call.__wrapped__ is original
        ch.install(ZettHypernet, 16, min_chunk_rows=1)  # identical settings: no-op
        assert ZettHypernet.__dict__['__call__'] is patched_call
        with pytest.raises(ch.ChunkedHypernetError, match='already patched'):
            ch.install(ZettHypernet, 32, min_chunk_rows=1)
        with pytest.raises(ch.ChunkedHypernetError, match='already patched'):
            ch.install(ZettHypernet, 16, use_checkpoint=False, min_chunk_rows=1)
        assert pinned_call() is original
    finally:
        assert ch.uninstall(ZettHypernet) is True
    assert ZettHypernet.__dict__['__call__'] is original and not ch.is_installed(ZettHypernet)
    assert ch.uninstall(ZettHypernet) is False


def test_install_validates_arguments():
    for bad in (-1, 1.5, True, '16', None):
        with pytest.raises((ValueError, TypeError)):
            ch.install(ZettHypernet, bad)
    with pytest.raises(ValueError):
        ch.install(ZettHypernet, 8, use_checkpoint=1, min_chunk_rows=1)
    with pytest.raises(TypeError):
        ch.install(object(), 8)
    assert not ch.is_installed(ZettHypernet)


def test_patch_is_class_level_and_used_with_grad(small, spy):
    hn, source, forms, priors = small
    other, _ = build_hn(SOURCE_VOCAB, width=32, layers=2, heads=2, seed=33, dtype=torch.float32)
    ch.install(ZettHypernet, 8, min_chunk_rows=1)
    try:
        assert type(hn).__call__ is ZettHypernet.__dict__['__call__']
        hn(forms, priors, source, None, deterministic=False)
        other(forms, priors, source, None, deterministic=True)
        hn(forms, priors, source, None, True)  # positional deterministic, as init_rescaler calls it
    finally:
        ch.uninstall(ZettHypernet)
    assert spy == [8, 8, 8]


@pytest.mark.parametrize('context', ['no_grad', 'inference_mode'])
def test_no_grad_and_inference_mode_call_the_original(small, spy, context):
    hn, source, forms, priors = small
    manager = torch.no_grad if context == 'no_grad' else torch.inference_mode
    with manager():
        reference, _, _ = pinned_call()(hn, forms, priors, source, None, True)
    ch.install(ZettHypernet, 8, min_chunk_rows=1)
    try:
        with manager():
            out, second, bias = hn(forms, priors, source, None, deterministic=True)
    finally:
        ch.uninstall(ZettHypernet)
    assert spy == []
    assert second is None and torch.equal(out, reference)


def test_chunk_rows_zero_is_a_pass_through(small, spy):
    hn, source, forms, priors = small
    ch.install(ZettHypernet, 0)
    try:
        hn(forms, priors, source, None, deterministic=False)
        assert ch.settings()['chunk_rows'] == 0
    finally:
        ch.uninstall(ZettHypernet)
    assert spy == []


def test_patched_context_restores_previous_state(small, spy):
    hn, source, forms, priors = small
    with ch.patched(ZettHypernet, 5, min_chunk_rows=1):
        hn(forms, priors, source, None, deterministic=True)
        with ch.patched(ZettHypernet, 0):
            hn(forms, priors, source, None, deterministic=True)
        assert ch.settings()['chunk_rows'] == 5
        hn(forms, priors, source, None, deterministic=True)
    assert not ch.is_installed(ZettHypernet)
    assert spy == [5, 5]


def test_settings_for_manifests():
    expected = ch.module_sha256()
    import hashlib
    assert expected == hashlib.sha256(open(ch.__file__, 'rb').read()).hexdigest()
    off = ch.settings()
    assert off['chunk_rows'] == 0 and off['installed'] is False and off['module_sha256'] == expected
    ch.install(ZettHypernet, 16384)
    try:
        on = ch.settings()
        assert on['chunk_rows'] == 16384 and on['use_checkpoint'] is True and on['installed'] is True
        assert on['module_sha256'] == expected and on['algorithm_version'] == ch.ALGORITHM_VERSION
        assert on['classes'] == ['hf_hypernet.modeling_hypernet.ZettHypernet']
        assert on['pinned_sources'] == ch.PINNED_SOURCES
        assert ch.settings(ZettHypernet) == on
    finally:
        ch.uninstall(ZettHypernet)
    assert ch.settings()['chunk_rows'] == 0


def test_pinned_source_hashes_are_the_repository_files():
    found = ch.verify_pinned_sources(ZettHypernet)
    assert found == ch.PINNED_SOURCES


# --- refusals --------------------------------------------------------------------------------------
def _call(hn, forms, priors, source, lang_index=None):
    return ch.chunked_call(hn, forms, priors, source, lang_index, True, chunk_rows=8)


@pytest.mark.parametrize('flag, value, message', [
    ('hn_embed_lang_id', True, 'hn_embed_lang_id'),
    ('hn_predict_bias', True, 'hn_predict_bias'),
    ('hn_concat_last_hidden_state', True, 'hn_concat_last_hidden_state'),
    ('hn_single_head', True, 'hn_single_head'),
    ('hn_rescale_embeddings', False, 'hn_rescale_embeddings'),
    ('hn_language_adapter_bottleneck_dim', 4, 'hn_language_adapter_bottleneck_dim'),
    ('hn_embed_target_priors', False, 'hn_embed_target_priors'),
    ('hn_add_inter_token_attention', False, 'hn_add_inter_token_attention'),
    ('hn_embed_using_source_embeddings', False, 'hn_embed_using_source_embeddings'),
    ('hn_model_type', 't5', 'hn_model_type'),
])
def test_unsupported_configuration_flags_are_refused(small, flag, value, message):
    hn, source, forms, priors = small
    setattr(hn.config, flag, value)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match=message):
        _call(hn, forms, priors, source)


def test_other_unsupported_calls_are_refused(small):
    hn, source, forms, priors = small
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='lang_index'):
        _call(hn, forms, priors, source, lang_index=torch.tensor(0))
    with pytest.raises(ValueError, match='source_embeddings are required'):
        _call(hn, forms, priors, None)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='target_priors are required'):
        _call(hn, forms, None, source)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='target_priors must have shape'):
        _call(hn, forms, priors[:-1], source)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='2-D integer'):
        _call(hn, forms.float(), priors, source)
    with pytest.raises(ValueError, match='chunk_rows'):
        ch.chunked_call(hn, forms, priors, source, None, True, chunk_rows=0)
    hn.has_separate_out_embeddings = True
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='separate_out_embeddings'):
        _call(hn, forms, priors, source)


def test_hooks_on_bypassed_modules_are_refused(small):
    hn, source, forms, priors = small
    handle = hn.model.register_forward_hook(lambda *args: None)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='HyperRobertaModel'):
        _call(hn, forms, priors, source)
    handle.remove()
    handle = hn.model.layers[1].register_forward_pre_hook(lambda *args: None)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='layer 1'):
        _call(hn, forms, priors, source)
    handle.remove()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        with pytest.raises(ch.UnsupportedHypernetConfiguration, match='global nn.Module hooks'):
            _call(hn, forms, priors, source)
    finally:
        handle.remove()
    _call(hn, forms, priors, source)  # hooks gone: accepted again


def test_replaced_layer_classes_and_methods_are_refused(small, monkeypatch):
    hn, source, forms, priors = small

    class OtherLayer(hyper_roberta.HyperLayer):
        pass

    layer = hn.model.layers[0]
    layer.__class__ = OtherLayer
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='layer 0 must be exactly HyperLayer'):
        _call(hn, forms, priors, source)
    layer.__class__ = hyper_roberta.HyperLayer
    layer.adapter_width = 2
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='language adapter'):
        _call(hn, forms, priors, source)
    layer.adapter_width = 0
    monkeypatch.setattr(hyper_roberta.HyperLayer, 'forward', lambda self, *a, **k: None)
    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='HyperLayer.forward has been replaced'):
        _call(hn, forms, priors, source)


def test_classes_without_own_call_or_with_other_signatures_are_refused():
    class Sub(ZettHypernet):
        pass

    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='does not define __call__'):
        ch.install(Sub, 8, min_chunk_rows=1)

    class Other:
        def __call__(self, x):
            return x

    with pytest.raises(ch.UnsupportedHypernetConfiguration, match='parameters'):
        ch.install(Other, 8, min_chunk_rows=1)
    assert not ch.is_installed(Other)


def test_modified_pinned_sources_are_refused(tmp_path, monkeypatch):
    package = tmp_path / 'hf_hypernet_modified'
    shutil.copytree(PINNED_CODE / 'hf_hypernet', package)
    roberta = package / 'hyper_roberta.py'
    roberta.write_text(roberta.read_text() + '\n# local edit\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        modeling = importlib.import_module('hf_hypernet_modified.modeling_hypernet')
        with pytest.raises(ch.UnsupportedHypernetConfiguration, match='SHA-256'):
            ch.install(modeling.ZettHypernet, 8, min_chunk_rows=1)
        assert not ch.is_installed(modeling.ZettHypernet)
    finally:
        for name in [n for n in sys.modules if n.startswith('hf_hypernet_modified')]:
            del sys.modules[name]


# --- chunk-size floor (memory-2) ------------------------------------------------------------------
def test_install_refuses_chunks_below_the_floor(monkeypatch):
    assert ch.MIN_CHUNK_ROWS == 4096
    for rows in (1, 1024, 4095):
        with pytest.raises(ValueError, match='below the floor of 4096 rows'):
            ch.install(ZettHypernet, rows)
        assert not ch.is_installed(ZettHypernet)
    with pytest.raises(ValueError, match='below the floor'):
        with ch.patched(ZettHypernet, 2048):
            pass
    for rows in (0, 4096, 16384):
        ch.install(ZettHypernet, rows)
        assert ch.settings()['chunk_rows'] == rows
        ch.uninstall(ZettHypernet)
    ch.install(ZettHypernet, 8, min_chunk_rows=1)       # explicit override (tests, probe smoke runs)
    ch.uninstall(ZettHypernet)
    monkeypatch.setattr(ch, 'MIN_CHUNK_ROWS', 1)          # a test harness may lower the module floor
    ch.install(ZettHypernet, 8)
    ch.uninstall(ZettHypernet)


def test_patched_restores_a_previous_small_patch():
    ch.install(ZettHypernet, 8, min_chunk_rows=1)
    try:
        with ch.patched(ZettHypernet, 4096):
            assert ch.settings()['chunk_rows'] == 4096
        assert ch.settings()['chunk_rows'] == 8
    finally:
        ch.uninstall(ZettHypernet)


# --- source files are read once (memory-4) -----------------------------------------------------------
def test_pinned_sources_are_read_once_not_per_update(small, monkeypatch):
    hn, source, forms, priors = small
    ch.install(ZettHypernet, 8, min_chunk_rows=1)
    try:
        hn(forms, priors, source, None, deterministic=True)[0].sum().backward()

        def broken(self, *args, **kwargs):
            raise OSError(f'simulated file-system failure on {self}')

        monkeypatch.setattr(Path, 'stat', broken)
        monkeypatch.setattr(Path, 'read_bytes', broken)
        out, _, _ = hn(forms, priors, source, None, deterministic=False)     # a later update still runs
        out.sum().backward()
        assert ch.settings()['module_sha256'] == ch.module_sha256()
    finally:
        monkeypatch.undo()
        ch.uninstall(ZettHypernet)


def test_module_sha256_is_the_imported_file():
    import hashlib
    assert ch.module_sha256() == hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest()
    assert 'PREDICTION' not in vars(ch) and 'predict_peak_gib' not in vars(ch), \
        'the prediction lives in the unhashed hyak.memory.memory_model'


def test_unverified_class_from_a_modified_copy_is_refused_at_its_first_call(tmp_path, monkeypatch):
    package = tmp_path / 'hf_hypernet_edited'
    shutil.copytree(PINNED_CODE / 'hf_hypernet', package)
    modeling_file = package / 'modeling_hypernet.py'
    modeling_file.write_text(modeling_file.read_text() + '\n# local edit\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        modeling = importlib.import_module('hf_hypernet_edited.modeling_hypernet')
        hn, source = build_hn(SOURCE_VOCAB, width=32, layers=2, heads=2, seed=31, dtype=torch.float32)
        hn.__class__ = modeling.ZettHypernet
        with pytest.raises(ch.UnsupportedHypernetConfiguration, match='SHA-256'):
            ch.chunked_call(hn, make_forms(V, SOURCE_VOCAB, seed=32), make_priors(V, 'unigram'), source, None,
                            True, chunk_rows=8)
    finally:
        for name in [n for n in sys.modules if n.startswith('hf_hypernet_edited')]:
            del sys.modules[name]


# --- sys.modules is only a cache (tests-5) -------------------------------------------------------
def test_works_after_the_source_modules_left_sys_modules(small, monkeypatch):
    """A harness that purges hf_hypernet* from sys.modules (a new 'process') must not break the patch."""
    hn, source, forms, priors = small
    reference = ch.chunked_call(hn, forms, priors, source, None, True, chunk_rows=8)[0].detach()
    for name in [n for n in sys.modules if n == 'hf_hypernet' or n.startswith('hf_hypernet.')]:
        monkeypatch.delitem(sys.modules, name)
    out = ch.chunked_call(hn, forms, priors, source, None, True, chunk_rows=8)[0]
    assert torch.equal(out.detach(), reference)
    ch.install(ZettHypernet, 8, min_chunk_rows=1)
    try:
        assert hn(forms, priors, source, None, deterministic=True)[0].shape == reference.shape
        assert ch.verify_pinned_sources(ZettHypernet) == ch.PINNED_SOURCES
    finally:
        ch.uninstall(ZettHypernet)
