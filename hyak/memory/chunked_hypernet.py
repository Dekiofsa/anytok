"""Vocabulary-chunked, checkpointed ``ZettHypernet.__call__`` for training (DESIGN.md D2, 5.D).

Why
---
Every training update embeds the whole target vocabulary (32,000 to 128,000
rows) with the hypernetwork and back-propagates through it once.  The pinned
forward checkpoints one whole encoder layer at a time, so the backward pass
rebuilds the graph of one layer over *all* rows: about 31.6 hidden-size
tensors ``[V, 11, 2048]`` (bf16) are alive at the peak, i.e. 72.2 GiB at
V = 32,768 (measured on the original B300 run) and about 190 GiB at
V = 128,000 -- more than an H200 holds.  This module computes the same
function with the vocabulary split into chunks, so the backward pass rebuilds
the graph of one chunk of one stage at a time.

Public API
----------
``chunked_call(hn, forms, priors, source, lang_index=None, deterministic=True, *, chunk_rows, use_checkpoint=True)``
    The chunked forward.  Returns the pinned triple ``(embeddings, None, zeros)``.
``install(hypernet_class, chunk_rows, *, use_checkpoint=True, min_chunk_rows=None)``
    Class-level patch of ``__call__`` (dunder lookup is on the type).  The
    original callable is kept and still used when ``chunk_rows == 0`` or when
    gradients are disabled (``torch.no_grad()``, ``torch.inference_mode()``:
    the trainer's ``evaluate()``, ``init_rescaler`` and all scoring).
    Idempotent for identical settings; different settings require ``uninstall``.
    Refuses classes whose source files differ from the pinned snapshot (the
    files are read once per class: at ``install`` or at the first chunked call)
    and chunk sizes below ``MIN_CHUNK_ROWS`` (4,096; see "Memory" below).
``uninstall(hypernet_class)``, ``is_installed(hypernet_class)``, ``patched(...)`` (context manager).
``settings()``
    ``{"chunk_rows", "use_checkpoint", "module_sha256", "algorithm_version", ...}``
    for run manifests (``hyak_segment.json``); compare ``chunk_rows``,
    ``use_checkpoint``, ``module_sha256`` and ``algorithm_version`` on resume.
``check_supported(hn, ...)``
    Raises :class:`UnsupportedHypernetConfiguration` listing every difference
    from the configuration this mirror is exact for.

Correspondence with the pinned forward (P = hf_hypernet/modeling_hypernet.py,
R = hf_hypernet/hyper_roberta.py, both pinned by SHA-256 below).  dtypes are
those under the trainer's ``torch.autocast("cuda", dtype=torch.bfloat16)``
(CUDA autocast runs linear/matmul in bf16 and layer_norm/softmax in fp32;
embedding, gelu, dropout, add and cat keep their input dtype)::

    pinned operation (in execution order)                  chunked stage   dtype out
    ---------------------------------------------------    -------------   ---------
    P:266-275 inputs_embeds = _project_source_inputs(...)   stage 0, chunk  bf16 [n,10,W]
      P:247-249 fallback mask / clamped ids                   (same method)
      P:250 F.embedding(main_ids, source)                                     fp32
      P:251-252 in_scaler (frozen w*x+b)                                      fp32
      P:253-255 torch.where(fallback, fallback_embeddings, .)                 fp32
      P:256 input_projection: Linear -> ProjectorBlock                        bf16
            (dense1, gelu-tanh, dense2, gelu-tanh, ln(h+x) fp32, .to(bf16))
    P:278 attention_mask = forms != pad                     per chunk       bool [n,10]
    P:308-311 position_ids = broadcast(arange(10))          stage 0, chunk  int64
    R:121-122 token_type_ids = zeros_like(input_ids)        stage 0, chunk  int64
    R:128-130 inputs_embeds + pos_emb.to(dt) + type_emb.to(dt)  stage 0      bf16
    R:134 cat(hidden, target_priors_projection(priors.to(dt)))  stage 0      bf16 [n,11,W]
    R:135 attention_mask = F.pad(mask, (0, 1), value=1)     per chunk       bool [n,11]
    R:142 F.dropout(LayerNorm(hidden).to(dt), p, not det)   stage 0         bf16  = h0
    R:145-155 for each layer (pinned: one checkpoint per layer over all rows):
      R:80 attended = layer.attention(h, mask, None, det)   stage (a), chunk bf16 [n,11,W]
           (R:36-38 q/k/v bf16, R:39 scores bf16, R:41 masked_fill(finfo.min),
            R:44 softmax fp32 + dropout, R:47 matmul bf16, R:48-49 dense,
            dropout, LayerNorm fp32, .to(bf16))
      R:82 bias = priors * scale if prior_bias else None    (b), all rows   None here
      R:83-84 blocks = down(latents[None], keys=attended[None, :, 0], bias)[0]
                                                            (b), all rows,  fp32 [16,W]
           keys = cat of every chunk's attended[:, 0]       NOT chunked,
           (scores [1,H,16,V] bf16, softmax fp32 over all  NOT checkpointed
            V rows, residual on the fp32 latents -> fp32)
      R:85-87 attended = up(attended.reshape(1,-1,W), keys=blocks[None]).reshape_as(attended)
                                                            stage (c), chunk bf16
      R:88-89 h = output_norm(attended + dropout(output_dense(act(intermediate(attended))))).to(dt)
                                                            stage (c), chunk bf16 = next h
      R:90-93 language adapter                              refused (width must be 0)
    P:327-332 last_hidden_state[:, 0]                       inside the last layer's (c) region
                                                            (returns h[:, 0]); cat  bf16 [V,W]
    P:334 output_projection (ProjectorBlock, Linear)        all rows        bf16 [V,W]
    P:336-350 single head off, no separate output           (refused otherwise) None
    P:352-353 scaler (frozen fp32 w*x+b)                    all rows        fp32 [V,W]
    P:358-363 predicted_bias = zeros_like(forms[..., 0], dtype=self.dtype)  float32 [V]
    P:365 return (predicted_in, None, predicted_bias)

Every tensor operation is a call of the same pinned submodule or method with
the same arguments, casts and ``F.dropout(..., training=not deterministic)``
calls; only the row range differs.  Modules that the chunked path bypasses
(``HyperRobertaModel.__call__``, ``HyperLayer.__call__``) must carry no hooks.

Why this is exact.  Rows interact only through ``blocks`` (the 16 latents of
the down attention attend over position 0 of *all* rows, softmax over V).  That
step runs once over all rows.  Everything else is row-wise: the self attention
is over the 11 positions of one row, the up attention has one query per
position over the 16 blocks, LayerNorm/Linear/GELU/dropout/embedding are
element- or row-wise.  So with dropout off (``deterministic=True``) outputs and
all parameter gradients equal the pinned ones up to floating-point summation
order (tested to <= 1e-9 relative in float64); under bf16 autocast they are
equal up to bf16 rounding (the two paths can take different kernels, for example
a strided versus a contiguous ``keys`` input to a Linear with a non-zero bias).
With dropout on, masks are drawn per chunk: same distribution, different random
stream than the pinned forward (a single chunk, ``chunk_rows >= V``, draws
exactly the pinned stream).

torch.utils.checkpoint (non-reentrant), as read in torch 2.15.0 ``torch/utils/checkpoint.py``
------------------------------------------------------------------------------------------
* Saved inputs: ``_CheckpointFrame.save_inputs`` keeps the tensor arguments as
  ``SavedTensor`` objects; no input needs ``requires_grad`` (stage 0 has none:
  forms, source and priors are constants).  Gradients reach the parameters the
  stage function closes over because the *original* forward builds the full
  autograd graph; only its saved tensors are replaced by holders (pack hook).
* Recompute: the first unpack of a saved tensor of a region during backward
  reruns the whole region (``recompute_fn``) under ``enable_grad``, storing the
  recomputed tensors in a ``WeakKeyDictionary`` keyed by per-holder handles,
  stopping as soon as all of them exist (early stop, default on).  Each unpack
  drops the handle (``holder.handles[gid] = None``), so a recomputed tensor is
  freed as soon as the backward node that needed it has run; the frame and the
  saved inputs die with the last node of the region.
* RNG: ``preserve_rng_state=True`` stashes ``torch.get_rng_state()`` and, when
  CUDA is initialised, the RNG state of every CUDA device of the tensor
  arguments (all stage arguments are on the training GPU).  ``recompute_fn``
  runs under ``torch.random.fork_rng`` after restoring those states, so the
  recomputation draws the forward's dropout masks and the global generators
  are unchanged afterwards: after ``backward()`` the RNG continues where the
  forward left it (tested).
* Autocast: ``_get_autocast_kwargs`` captures the device (and CPU) autocast
  state at forward time and re-enters it for the recomputation.
* ``torch.is_grad_enabled()`` False: checkpoint runs the function plainly
  (this module never gets there: it hands such calls to the pinned forward).
* Order: the autograd engine executes ready nodes by decreasing sequence number
  (creation order), so regions are back-propagated strictly one after another
  in reverse creation order: for each layer from the last, stage (c) chunks
  C-1..0, then the unchunked down attention, then stage (a) chunks C-1..0, and
  finally stage-0 chunks C-1..0.  At most one region's recomputed graph is
  alive at any time (tested by recording recompute order and peak memory).

Memory
------
The peak-memory model and the predicted peaks live in :mod:`hyak.memory.memory_model`
(deliberately outside this file: the SHA-256 of this file is part of every run's identity,
so it must not change between the uniform bootstrap and update 20,000 of both runs).
``python -m hyak.memory.probe`` measures the real peak of whole updates on the GPU.

Chunk-size floor.  ``install`` refuses ``0 < chunk_rows < MIN_CHUNK_ROWS`` (4,096).  Under
autocast every fp32 weight is cast to bf16 once per autocast region (the weight cache), so all
chunks share one cast node and their weight-gradient contributions are summed in bf16; the
error grows with the number of chunks (measured: negligible at 8 to 32 chunks, several percent
at hundreds).  Smaller chunks also do not lower the peak: with 16,384-row chunks the LM
micro-batches already set it.  ``chunked_call`` itself has no floor (tests and the probe's equivalence check
use small chunks on small models).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

#: Bump when the chunked algorithm changes in a way that changes results (part of ``settings()``).
ALGORITHM_VERSION = 1

#: SHA-256 of the pinned files this mirror was derived from (runtime/anytok_pool_20k/hf_hypernet/;
#: the RL code tree stages the same bytes).  A different file means the mirror may be wrong.
PINNED_SOURCES = {
    'modeling_hypernet.py': '9314094415250b093040f27948434f949811913d5a82b77c5676681361031541',
    'hyper_roberta.py': '5c385150184fd0a55fb45a289a4eed2b6b7756880849f080c4f55b25355b8512',
}

#: Smallest accepted ``chunk_rows`` of ``install`` (0 = pinned forward is always accepted).  Read at
#: call time, so a test harness may lower it for tiny vocabularies; training never does.
MIN_CHUNK_ROWS = 4096

#: Parameter names of the pinned ``ZettHypernet.__call__`` (P:258-265).
PINNED_CALL_PARAMETERS = ('self', 'target_surface_forms', 'target_priors', 'source_embeddings',
                          'lang_index', 'deterministic')

_MODULE_PATH = Path(__file__).resolve()
#: SHA-256 of this file as imported (a later edit on disk does not change what runs in this process).
_MODULE_SHA256 = hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()
_OBSERVERS: list = []
#: Every class patched by ``install`` in this process (entries stay until ``uninstall``).
_REGISTRY: dict = {}
#: Classes whose pinned source files were verified in this process (the imported code cannot change
#: afterwards, so the files are read once, at ``install`` or at the first chunked call, not per update).
_VERIFIED: dict = {}

_REMEDY = ('The chunked forward is exact only for the AnyTok input hypernetwork built by '
           'hf_hypernet.identity_warmup.build_input_hypernet.  Use the pinned forward instead '
           '(HYAK_HN_CHUNK_ROWS=0, needs about 200 GiB of GPU memory for 128,000-row vocabularies) '
           'or extend hyak/memory/chunked_hypernet.py with a new equivalence test.')


class ChunkedHypernetError(RuntimeError):
    """Misuse of the patch (conflicting install, unknown class layout)."""


class UnsupportedHypernetConfiguration(ValueError):
    """The hypernetwork or call differs from the configuration the chunked mirror is exact for."""


# --- helpers ------------------------------------------------------------------------------------
def _sha256_file(path) -> str:
    path = Path(path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ChunkedHypernetError(f'cannot read {path}: {exc}') from exc


def module_sha256() -> str:
    """SHA-256 of this file as imported (recorded in run manifests next to ``chunk_rows``)."""
    return _MODULE_SHA256


def _validate_rows(chunk_rows, *, allow_zero: bool) -> int:
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int):
        raise ValueError(f'chunk_rows must be an integer, got {chunk_rows!r}')
    if chunk_rows < 0 or (chunk_rows == 0 and not allow_zero):
        raise ValueError(f'chunk_rows must be {"non-negative" if allow_zero else "positive"}, '
                         f'got {chunk_rows}')
    return chunk_rows


def chunk_bounds(total: int, chunk_rows: int) -> list:
    """Consecutive ``(start, stop)`` row ranges of at most ``chunk_rows`` rows covering ``0..total-1``."""
    total = int(total)
    rows = _validate_rows(chunk_rows, allow_zero=False)
    if total < 1:
        raise ValueError(f'the vocabulary must have at least one row, got {total}')
    return [(start, min(start + rows, total)) for start in range(0, total, rows)]


class _Namespace:
    """Attribute view of a module's globals (used when the module is no longer in ``sys.modules``)."""

    __slots__ = ('_globals',)

    def __init__(self, namespace):
        self._globals = namespace

    def __getattr__(self, name):
        try:
            return self._globals[name]
        except KeyError:
            raise AttributeError(name) from None


def _defining_module(cls):
    """The module that defined ``cls``: its own globals, found through a function defined in the class.

    ``sys.modules`` is only a cache: a test harness (or a re-import) may have dropped or replaced the
    entry while the class and its module globals live on, so the globals are authoritative.
    """
    if cls is None:
        return None
    for value in vars(cls).values():
        function = getattr(value, '__func__', value)
        if inspect.isfunction(function) and function.__module__ == cls.__module__ \
                and not hasattr(function, '_hyak_chunked_patch'):
            namespace = function.__globals__
            module = sys.modules.get(cls.__module__)
            return module if getattr(module, '__dict__', None) is namespace else _Namespace(namespace)
    return sys.modules.get(cls.__module__)


def _source_modules(hypernet_class):
    """The pinned modules (modeling_hypernet, hyper_roberta) that define ``hypernet_class``."""
    modeling = _defining_module(hypernet_class)
    roberta_class = getattr(modeling, 'HyperRobertaModel', None)
    roberta = _defining_module(roberta_class) if isinstance(roberta_class, type) else None
    if modeling is None or roberta is None or not getattr(modeling, '__file__', None) \
            or not getattr(roberta, '__file__', None):
        raise ChunkedHypernetError(
            f'cannot locate the source modules of {hypernet_class.__module__}.{hypernet_class.__qualname__}; '
            'import it from the staged hf_hypernet package (hf_hypernet.modeling_hypernet).')
    return modeling, roberta


def verify_pinned_sources(hypernet_class) -> dict:
    """Check that ``hypernet_class`` comes from the pinned modeling_hypernet.py / hyper_roberta.py.

    Returns ``{file name: sha256}``; raises :class:`UnsupportedHypernetConfiguration` on a mismatch.
    """
    modeling, roberta = _source_modules(hypernet_class)
    found = {}
    problems = []
    for module in (modeling, roberta):
        name = Path(module.__file__).name
        digest = _sha256_file(module.__file__)
        found[name] = digest
        expected = PINNED_SOURCES.get(name)
        if expected is None:
            problems.append(f'unexpected source file {module.__file__}')
        elif digest != expected:
            problems.append(f'{module.__file__} has SHA-256 {digest}, the mirror was derived from {expected}')
    if set(found) != set(PINNED_SOURCES):
        problems.append(f'expected source files {sorted(PINNED_SOURCES)}, found {sorted(found)}')
    if problems:
        raise UnsupportedHypernetConfiguration(
            'hyak.memory.chunked_hypernet mirrors a specific pinned hypernetwork: ' + '; '.join(problems)
            + '. ' + _REMEDY)
    _VERIFIED[hypernet_class] = dict(found)
    return found


def _has_hooks(module) -> list:
    names = ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')
    return [name for name in names if getattr(module, name, None)]


def _global_hooks() -> list:
    from torch.nn.modules import module as nn_module
    names = ('_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks',
             '_global_backward_pre_hooks')
    return [name for name in names if getattr(nn_module, name, None)]


def check_supported(hn, target_surface_forms=None, target_priors=None, lang_index=None) -> None:
    """Raise :class:`UnsupportedHypernetConfiguration` unless the chunked mirror is exact for this call.

    Required (DESIGN.md 5.D): the HyperRoberta backbone with source embeddings, target priors and
    inter-token attention; no language id, no language adapters, no concatenated hidden states,
    no bias head, no separate output embeddings, no single head, rescaled embeddings;
    ``lang_index is None``; pinned source files and classes; no hooks on bypassed modules.
    """
    cls = type(hn)
    if cls not in _VERIFIED:          # files read once per class and process (install or first call)
        verify_pinned_sources(cls)
    modeling, roberta = _source_modules(cls)
    config = hn.config
    problems = []

    def need(condition, text):
        if not condition:
            problems.append(text)

    need(config.hn_model_type == 'roberta', f'hn_model_type must be "roberta", got {config.hn_model_type!r}')
    need(bool(getattr(hn, 'use_hyper_roberta', False)), 'the HyperRoberta backbone must be in use')
    need(bool(config.hn_embed_using_source_embeddings), 'hn_embed_using_source_embeddings must be true')
    need(bool(config.hn_embed_target_priors), 'hn_embed_target_priors must be true')
    need(bool(config.hn_add_inter_token_attention), 'hn_add_inter_token_attention must be true')
    need(not config.hn_embed_lang_id, 'hn_embed_lang_id must be false')
    need(not config.hn_language_adapter_bottleneck_dim, 'hn_language_adapter_bottleneck_dim must be 0')
    need(not config.hn_concat_last_hidden_state, 'hn_concat_last_hidden_state must be false')
    need(not getattr(config, 'hn_predict_bias', False), 'hn_predict_bias must be false')
    need(not getattr(hn, 'has_separate_out_embeddings', False), 'separate_out_embeddings must be false')
    need(not config.hn_single_head, 'hn_single_head must be false')
    need(bool(config.hn_rescale_embeddings), 'hn_rescale_embeddings must be true')
    need(lang_index is None, 'lang_index must be None')
    model = getattr(hn, 'model', None)
    need(type(model) is modeling.HyperRobertaModel,
         f'hn.model must be exactly {modeling.HyperRobertaModel.__module__}.HyperRobertaModel, '
         f'got {type(model).__module__}.{type(model).__qualname__}')
    if type(model) is modeling.HyperRobertaModel:
        mcfg = model.config
        need(bool(mcfg.embed_target_priors), 'backbone embed_target_priors must be true')
        need(bool(mcfg.add_inter_token_attention), 'backbone add_inter_token_attention must be true')
        need(not mcfg.embed_lang_id, 'backbone embed_lang_id must be false')
        need(len(model.layers) >= 1, 'the backbone must have at least one layer')
        for index, layer in enumerate(model.layers):
            need(type(layer) is roberta.HyperLayer, f'layer {index} must be exactly HyperLayer')
            for name in ('attention', 'inter_token_attention_down', 'inter_token_attention_up'):
                sub = getattr(layer, name, None)
                need(type(sub) is roberta.HyperAttention, f'layer {index}.{name} must be exactly HyperAttention')
            need(bool(getattr(layer, 'inter_token_attention', False)), f'layer {index} must use inter-token attention')
            need(not getattr(layer, 'adapter_width', 0), f'layer {index} must have no language adapter')
            hooks = _has_hooks(layer)
            need(not hooks, f'layer {index} (HyperLayer, bypassed by the chunked forward) has hooks {hooks}')
        hooks = _has_hooks(model)
        need(not hooks, f'hn.model (HyperRobertaModel, bypassed by the chunked forward) has hooks {hooks}')
        for owner, method in ((roberta.HyperLayer, 'forward'), (roberta.HyperRobertaModel, 'forward'),
                              (roberta.HyperAttention, 'forward')):
            function = owner.__dict__.get(method)
            need(function is not None and getattr(function, '__module__', None) == roberta.__name__
                 and function.__qualname__ == f'{owner.__name__}.{method}',
                 f'{owner.__name__}.{method} has been replaced; the mirror follows the pinned method')
    method = cls.__dict__.get('_project_source_inputs')
    need(method is not None and method.__qualname__ == f'{cls.__name__}._project_source_inputs'
         and method.__module__ == modeling.__name__,
         '_project_source_inputs must be the pinned method')
    global_hooks = _global_hooks()
    need(not global_hooks, f'global nn.Module hooks are registered ({global_hooks}); they would not fire '
                           'for the modules the chunked forward bypasses')
    if target_surface_forms is not None:
        forms = target_surface_forms
        need(isinstance(forms, torch.Tensor) and forms.ndim == 2 and not forms.is_floating_point()
             and not forms.is_complex() and forms.dtype != torch.bool,
             'target_surface_forms must be a 2-D integer tensor [V, pieces]')
        if isinstance(forms, torch.Tensor) and forms.ndim == 2:
            need(forms.shape[0] >= 1, 'target_surface_forms must have at least one row')
            if target_priors is None:
                problems.append('target_priors are required when embed_target_priors=True')
            elif not (isinstance(target_priors, torch.Tensor) and tuple(target_priors.shape) == (forms.shape[0],)):
                problems.append(f'target_priors must have shape ({forms.shape[0]},), '
                                f'got {tuple(getattr(target_priors, "shape", ()))}')
    if problems:
        raise UnsupportedHypernetConfiguration(
            'chunked hypernetwork forward refused: ' + '; '.join(problems) + '. ' + _REMEDY)


# --- chunk stages (each is one checkpoint region per chunk) ---------------------------------------
def _notify(region, layer, chunk):
    if _OBSERVERS:
        recompute = torch._C._current_graph_task_id() != -1
        for callback in list(_OBSERVERS):
            callback(region, layer, chunk, recompute)


def _stage_embed(hn, deterministic, chunk, forms_c, source, priors_c):
    """Stage 0 for one chunk: P:270-275 (input projection) + R:121-142 (embedding stage)."""
    _notify('embed', None, chunk)
    model = hn.model
    inputs_embeds = hn._project_source_inputs(forms_c, source)                       # P:270-275
    attention_mask = forms_c != hn.pad_token_id                                       # P:278
    position_ids = torch.broadcast_to(                                                # P:308-311
        torch.arange(torch.atleast_2d(attention_mask).shape[-1], device=hn.device),
        attention_mask.shape,
    )
    token_type_ids = torch.zeros_like(forms_c)                                        # R:121-122
    hidden = (inputs_embeds                                                           # R:128-130
              + model.position_embeddings(position_ids).to(inputs_embeds.dtype)
              + model.token_type_embeddings(token_type_ids).to(inputs_embeds.dtype))
    hidden = torch.cat((hidden, model.target_priors_projection(                       # R:134
        priors_c[:, None, None].to(hidden.dtype))), dim=1)
    return F.dropout(model.LayerNorm(hidden).to(hidden.dtype), p=model.dropout.p,     # R:142
                     training=not deterministic)


def _stage_attention(layer, deterministic, index, chunk, hidden_c, mask_c):
    """Stage (a) for one chunk: R:80 (self attention over the 11 positions of each row)."""
    _notify('attention', index, chunk)
    return layer.attention(hidden_c, mask=mask_c, head_mask=None, deterministic=deterministic)


def _stage_up_ffn(layer, deterministic, index, chunk, pool, attended_c, blocks):
    """Stage (c) for one chunk: R:85-89 (up attention to the 16 blocks, then the feed-forward block).

    For the last layer (``pool=True``) the region returns only position 0 (P:332), so the
    zero-filled ``[n, 11, W]`` gradient of the select exists inside one region's backward at a
    time instead of for every chunk at once (one full hidden unit less at the peak).
    """
    _notify('up_ffn', index, chunk)
    attended = layer.inter_token_attention_up(                                        # R:85-87
        attended_c.reshape(1, -1, attended_c.shape[-1]), keys=blocks[None], deterministic=deterministic
    ).reshape_as(attended_c)
    hidden = layer.output_norm(attended + F.dropout(                                  # R:88-89
        layer.output_dense(layer.act(layer.intermediate(attended))),
        p=layer.output_dropout.p, training=not deterministic)).to(attended.dtype)
    return hidden[:, 0] if pool else hidden                                           # P:332


def _run(function, args, use_checkpoint):
    if use_checkpoint:
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)
    return function(*args)


def chunked_call(hn, target_surface_forms, target_priors=None, source_embeddings=None, lang_index=None,
                 deterministic=True, *, chunk_rows, use_checkpoint=True):
    """The pinned ``ZettHypernet.__call__`` (P:258-365) computed in vocabulary chunks.

    Args mirror the pinned call; ``chunk_rows`` (> 0) is the number of vocabulary rows per chunk
    (the last chunk is shorter); ``use_checkpoint=False`` runs the identical chunk loop without
    activation checkpointing (reference for tests, no memory saving).

    The hypernetwork's own flags ``model.activation_checkpointing`` and
    ``activation_checkpoint_input`` are not consulted: every chunk stage is its own checkpoint.
    Returns ``(predicted_embeddings_in, None, predicted_bias)`` exactly like the pinned call.
    """
    rows = _validate_rows(chunk_rows, allow_zero=False)
    if source_embeddings is None:                                                     # P:266-268
        raise ValueError("source_embeddings are required when hn_embed_using_source_embeddings=True")
    check_supported(hn, target_surface_forms, target_priors, lang_index)
    model = hn.model
    forms = target_surface_forms
    bounds = chunk_bounds(forms.shape[0], rows)

    # Stage 0 (per chunk): input projection and embedding stage.  P:266-311, R:117-142.
    hidden, masks = [], []
    for chunk, (start, stop) in enumerate(bounds):
        forms_c = forms[start:stop]
        masks.append(F.pad(forms_c != hn.pad_token_id, (0, 1), value=1))              # P:278, R:135
        hidden.append(_run(functools.partial(_stage_embed, hn, deterministic, chunk),
                           (forms_c, source_embeddings, target_priors[start:stop]), use_checkpoint))

    # Encoder layers.  R:145-155 -> R:79-94.
    last = len(model.layers) - 1
    for index, layer in enumerate(model.layers):
        attended = [_run(functools.partial(_stage_attention, layer, deterministic, index, chunk),
                         (hidden_c, mask_c), use_checkpoint)                              # R:80
                    for chunk, (hidden_c, mask_c) in enumerate(zip(hidden, masks))]
        del hidden
        keys = torch.cat([attended_c[:, 0] for attended_c in attended])                 # R:83 keys
        bias = target_priors * layer.prior_bias_scale if layer.prior_bias else None     # R:82
        blocks = layer.inter_token_attention_down(model.inter_token_blocks[None],       # R:83-84
                                                  keys=keys[None], bias=bias,
                                                  deterministic=deterministic)[0]
        del keys, bias
        hidden = [_run(functools.partial(_stage_up_ffn, layer, deterministic, index, chunk, index == last),
                       (attended_c, blocks), use_checkpoint)                            # R:85-89
                  for chunk, attended_c in enumerate(attended)]
        del attended, blocks

    pooled = torch.cat(hidden)               # P:332: the last layer's regions returned position 0
    del hidden, masks
    predicted_embeddings = hn.output_projection(pooled)                                 # P:334
    del pooled
    predicted_embeddings_in = hn.scaler(predicted_embeddings)                          # P:346, P:352-353
    predicted_bias = torch.zeros_like(target_surface_forms[..., 0], dtype=hn.dtype)    # P:361-363
    return predicted_embeddings_in, None, predicted_bias                               # P:365


# --- observation (tests and the probe) -----------------------------------------------------------
@contextlib.contextmanager
def observe(callback):
    """Call ``callback(region, layer, chunk, recompute)`` at every chunk-stage call (forward and recompute).

    ``region`` is ``'embed'``, ``'attention'`` or ``'up_ffn'``; ``recompute`` is True when the call
    runs inside a backward pass (checkpoint recomputation).  Observation never changes results.
    """
    _OBSERVERS.append(callback)
    try:
        yield callback
    finally:
        _OBSERVERS.remove(callback)


# --- the class-level patch ---------------------------------------------------------------------------
class _Patch:
    __slots__ = ('cls', 'original', 'chunk_rows', 'use_checkpoint', 'function')

    def __init__(self, cls, original, chunk_rows, use_checkpoint):
        self.cls = cls
        self.original = original
        self.chunk_rows = chunk_rows
        self.use_checkpoint = use_checkpoint
        self.function = None


def _current_patch(hypernet_class):
    return getattr(hypernet_class.__dict__.get('__call__'), '_hyak_chunked_patch', None)


def is_installed(hypernet_class) -> bool:
    """True when ``hypernet_class.__call__`` is the chunked patch."""
    return _current_patch(hypernet_class) is not None


def original_call(hypernet_class):
    """The pinned ``__call__`` of ``hypernet_class``, whether or not the patch is installed."""
    patch = _current_patch(hypernet_class)
    return patch.original if patch is not None else hypernet_class.__dict__['__call__']


def install(hypernet_class, chunk_rows: int, *, use_checkpoint: bool = True,
            min_chunk_rows: int | None = None) -> None:
    """Replace ``hypernet_class.__call__`` by the chunked forward (class level; the original is kept).

    ``chunk_rows == 0`` installs a pass-through to the original (patch disabled; negative values are
    refused).  ``0 < chunk_rows < min_chunk_rows`` (default :data:`MIN_CHUNK_ROWS`, 4,096) is refused:
    many chunks sum the weight gradients in bf16 (module docstring).  Calling again with the same
    settings does nothing; different settings raise :class:`ChunkedHypernetError`.
    Raises :class:`UnsupportedHypernetConfiguration` when the class does not come from the pinned
    sources or its ``__call__`` signature differs.
    """
    if not isinstance(hypernet_class, type):
        raise TypeError(f'install() expects the hypernetwork class, got {hypernet_class!r}')
    rows = _validate_rows(chunk_rows, allow_zero=True)
    floor = MIN_CHUNK_ROWS if min_chunk_rows is None else _validate_rows(min_chunk_rows, allow_zero=False)
    if 0 < rows < floor:
        raise ValueError(f'chunk_rows={rows} is below the floor of {floor} rows: with more, smaller chunks '
                         'autocast sums the weight gradients of all chunks in bf16, and smaller chunks do not '
                         'lower the peak memory (the LM micro-batches dominate). Use the largest value that '
                         'fits (default 16384), or 0 for the pinned forward')
    if not isinstance(use_checkpoint, bool):
        raise ValueError(f'use_checkpoint must be a bool, got {use_checkpoint!r}')
    existing = _current_patch(hypernet_class)
    if existing is not None:
        if existing.chunk_rows == rows and existing.use_checkpoint == use_checkpoint:
            return
        raise ChunkedHypernetError(
            f'{hypernet_class.__qualname__} is already patched with chunk_rows={existing.chunk_rows}, '
            f'use_checkpoint={existing.use_checkpoint}; call uninstall() first (a run must not change '
            'its chunk setting)')
    original = hypernet_class.__dict__.get('__call__')
    if original is None:
        raise UnsupportedHypernetConfiguration(
            f'{hypernet_class.__qualname__} does not define __call__ itself; patch the class that does '
            '(hf_hypernet.modeling_hypernet.ZettHypernet)')
    names = tuple(inspect.signature(original).parameters)
    if names != PINNED_CALL_PARAMETERS:
        raise UnsupportedHypernetConfiguration(
            f'{hypernet_class.__qualname__}.__call__ has parameters {names}, expected the pinned '
            f'{PINNED_CALL_PARAMETERS}. ' + _REMEDY)
    verify_pinned_sources(hypernet_class)
    patch = _Patch(hypernet_class, original, rows, use_checkpoint)

    def __call__(self, target_surface_forms, target_priors=None, source_embeddings=None,
                 lang_index=None, deterministic: bool = True):
        if patch.chunk_rows <= 0 or not torch.is_grad_enabled():
            return patch.original(self, target_surface_forms, target_priors, source_embeddings,
                                  lang_index, deterministic)
        return chunked_call(self, target_surface_forms, target_priors, source_embeddings, lang_index,
                            deterministic, chunk_rows=patch.chunk_rows, use_checkpoint=patch.use_checkpoint)

    __call__.__qualname__ = f'{hypernet_class.__qualname__}.__call__'
    __call__.__doc__ = (f'hyak.memory.chunked_hypernet patch (chunk_rows={rows}, '
                        f'use_checkpoint={use_checkpoint}); the pinned __call__ is __wrapped__.')
    __call__.__wrapped__ = original
    __call__._hyak_chunked_patch = patch
    patch.function = __call__
    type.__setattr__(hypernet_class, '__call__', __call__)
    _REGISTRY[hypernet_class] = patch


def uninstall(hypernet_class) -> bool:
    """Restore the original ``__call__``.  Returns True if a patch was removed, False if none was installed."""
    patch = _current_patch(hypernet_class)
    if patch is None:
        return False
    type.__setattr__(hypernet_class, '__call__', patch.original)
    _REGISTRY.pop(hypernet_class, None)
    return True


@contextlib.contextmanager
def patched(hypernet_class, chunk_rows: int, *, use_checkpoint: bool = True,
            min_chunk_rows: int | None = None):
    """Temporarily install the patch with these settings (``chunk_rows=0``: pinned forward), then restore."""
    previous = _current_patch(hypernet_class)
    uninstall(hypernet_class)
    try:
        install(hypernet_class, chunk_rows, use_checkpoint=use_checkpoint, min_chunk_rows=min_chunk_rows)
        yield
    finally:
        uninstall(hypernet_class)
        if previous is not None:
            install(hypernet_class, previous.chunk_rows, use_checkpoint=previous.use_checkpoint, min_chunk_rows=1)


def settings(hypernet_class=None) -> dict:
    """Patch settings for run manifests.

    ``chunk_rows`` is 0 when no patch is installed (or the patch is a pass-through).  With
    ``hypernet_class=None`` every patched class known to this process is considered (they must agree).
    """
    if hypernet_class is not None:
        patches = [p for p in [_current_patch(hypernet_class)] if p is not None]
    else:
        patches = [p for cls, p in list(_REGISTRY.items()) if _current_patch(cls) is p]
    if len({(p.chunk_rows, p.use_checkpoint) for p in patches}) > 1:
        raise ChunkedHypernetError('patched classes disagree on chunk settings: '
                                   + ', '.join(f'{p.cls.__module__}.{p.cls.__qualname__}={p.chunk_rows}'
                                               for p in patches))
    patch = patches[0] if patches else None
    return {
        'chunk_rows': patch.chunk_rows if patch else 0,
        'use_checkpoint': patch.use_checkpoint if patch else False,
        'installed': patch is not None,
        'classes': sorted(f'{p.cls.__module__}.{p.cls.__qualname__}' for p in patches),
        'module': __name__,
        'module_sha256': module_sha256(),
        'algorithm_version': ALGORITHM_VERSION,
        'pinned_sources': dict(PINNED_SOURCES),
        'torch_version': torch.__version__,
    }
