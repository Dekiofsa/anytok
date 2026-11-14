"""H200 memory and speed probe for training updates (DESIGN.md 5.C; dashboard id ``hyak-memory-probe``).

    python -m hyak.memory.probe [--model-dir DIR] [--hypernet DIR] [--code-dir DIR]
        [--vocab 32768 64000 96000 128000] [--chunk-rows N [N ...]] [--also-pinned]
        [--equivalence-rows 4096] [--equivalence-chunk-rows N] [--updates 2]
        [--microbatch-size 32] [--positions 128] [--min-headroom-gib 8]
        [--device cuda] [--helpers import|source] [--seed 42] [--report PATH] [--config FILE]

Defaults come from ``hyak/config.env``: ``input/model``, ``input/hypernet_init``,
``code/uniform`` and ``HYAK_HN_CHUNK_ROWS``.  Needs only the environment, the
model, the random-init hypernetwork and the staged code (no corpus, no pool).

What it does, on one GPU:

1. Loads the LM, tokenizer and hypernetwork exactly like the pinned trainer
   (T = runtime/anytok_pool_20k/scripts/train_joint_smollm_input.py, lines
   421-455: float32 LM, untied frozen tables, byte fallbacks, frozen rescalers,
   checkpoint flags, the fused AdamW of ``make_optimizer_groups``).  The helper
   functions come from the staged code: ``--helpers import`` (default) imports
   the staged trainer module itself (this also loads its Rust library);
   ``--helpers source`` executes the two trainer helpers' pinned source text
   (``make_optimizer_groups``, ``set_learning_rates``) for machines where the
   Rust library cannot load.
   The staged files are compared with the pinned SHA-256 values
   (``hyak.code.pinned``) before anything is imported.
2. Deterministic equivalence on the GPU at ``--equivalence-rows`` rows: chunked
   versus pinned forward and all parameter gradients, in float32 (strict) and
   under bf16 autocast (relative to the bf16 rounding of the pinned path), plus
   the dropout checks (forward reproducible from a seed; CUDA RNG after
   backward equals the state after the forward; with dropout on, gradients
   with and without checkpointing are equal, in float32 and under bf16
   autocast, which proves that recomputation redraws the forward's masks).
   The equivalence uses as many chunks as the configured chunk size gives at
   the largest ``--vocab`` (at least 3), unless ``--equivalence-chunk-rows``.
3. For every vocabulary size the configured chunk size first, then the other
   settings (``0`` = pinned forward, added with ``--also-pinned``; information
   only): ``--updates`` training-shaped updates with synthetic
   inputs that follow T:542-584 operation for operation (bf16 hypernetwork
   forward with dropout, detached float32 leaf, four LM micro-batches of 32
   sequences with backward, lexical loss, the combined backward, joint clip
   0.1, fused AdamW step).  Records ``max_memory_allocated`` (also at the end of
   the LM micro-batches, to name the phase that set the peak),
   ``max_memory_reserved``, the memory allocated before the first update and
   seconds per update; CUDA out-of-memory is caught per configuration, memory
   is released before the next one.
4. Prints one ``VERDICT`` line per vocabulary size (``FITS`` / ``DOES NOT FIT``
   with the headroom = device memory - peak ``max_memory_reserved``, which must be
   at least ``--min-headroom-gib``; ``MEASURED`` on ``--device cpu``; ``ERROR`` for
   failures other than out-of-memory) for every chunk setting and a final ``SUMMARY`` line for
   the configured chunk size (the first ``--chunk-rows``), writes a JSON report
   (``<repo>/modal_results/hyak_tillicum/probe/probe-report.json`` and a per-job
   copy ``$ANYTOK_RUNS/probe/probe-report-<job>.json``) and status/events for
   the dashboard.

Exit codes (project contract): 0 every configured-chunk measurement FITS (or
MEASURED on ``--device cpu``) and the equivalence passed; 4 a configured-chunk
measurement does not fit or failed, or the equivalence failed; 2 setup or
unexpected error (configuration, missing or modified files, no CUDA, a chunk
size below the 4,096-row floor on CUDA; tracebacks are printed).  Measurements
of the other settings (the pinned comparison, extra ``--chunk-rows``) are
information only and never change the exit code.

``--device cpu`` is a smoke run.  On the CPU, a failure inside one chunk region
during backward leaves the other regions' pending autograd tasks queued on the
calling thread until the next backward; the probe runs one trivial backward
after a caught failure so the graph is released (CUDA drains them at once).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import gc
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from hyak.common import slurm
from hyak.common.config import ConfigError, int_value, load_config
from hyak.common.io import atomic_write_json, utc_now
from hyak.common.paths import STAGE_PROBE, Paths
from hyak.common.status import StatusWriter
from hyak.memory import chunked_hypernet as ch
from hyak.memory import memory_model as mm

DEFAULT_VOCABS = (32768, 64000, 96000, 128000)
#: Examples per update (T:553 ``range(0, 128, args.microbatch_size)``).
EXAMPLES_PER_UPDATE = 128
#: Position limit of an example (joint_training.py:148-149 ``max_positions=128``).
MAX_POSITIONS = 128
#: SmolLM2's end-of-text id (T:48-49).
SPECIAL_TOKEN = '<|endoftext|>'
SPECIAL_ID = 0
#: Surface-form width (T:446 ``hn_surface_maxlen == 10``).
PIECES = 10
TRAINER_REL = Path('scripts') / 'train_joint_smollm_input.py'
HF_FILES = ('configuration_hypernet.py', 'modeling_hypernet.py', 'hyper_roberta.py', 'identity_warmup.py',
            'tokenizer_inputs.py', 'joint_training.py', 'smollm_setup.py')
#: Trainer helpers executed from source in ``--helpers source`` mode, and the globals they may use.
TRAINER_HELPERS = ('make_optimizer_groups', 'set_learning_rates')
TRAINER_HELPER_GLOBALS = {'zett_main_learning_rate'}
GIB = 2 ** 30
EXIT_OK, EXIT_SETUP, EXIT_CHECK_FAILED = 0, 2, 4


class ProbeError(RuntimeError):
    """Setup problem (files, code tree, device); the message names the remedy."""


def _log(message):
    print(f'[probe] {message}', flush=True)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    """``value`` with every non-finite float replaced by its string (strict JSON for reports and events)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


# --- code from the staged tree ------------------------------------------------------------------------
def load_code(code_dir, helpers: str = 'import') -> SimpleNamespace:
    """Import the pinned hypernetwork package and trainer helpers from ``code_dir``.

    ``helpers='import'`` imports the staged trainer module (and therefore its Rust library) the
    way the training wrapper does; ``'source'`` executes only the pinned source text of
    ``make_optimizer_groups`` and ``set_learning_rates`` (no Rust library needed).
    """
    code_dir = Path(code_dir).absolute()
    trainer_path = code_dir / TRAINER_REL
    missing = [str(p) for p in [trainer_path] + [code_dir / 'hf_hypernet' / f for f in HF_FILES] if not p.is_file()]
    if missing:
        raise ProbeError(f'staged code tree {code_dir} is incomplete (missing {missing}); '
                         'stage it with python -m hyak.code.stage_code or pass --code-dir')
    if helpers not in ('import', 'source'):
        raise ProbeError(f'--helpers must be import or source, got {helpers!r}')
    files = {str(TRAINER_REL): _sha256(trainer_path)}
    files.update({f'hf_hypernet/{f}': _sha256(code_dir / 'hf_hypernet' / f) for f in HF_FILES})
    pins = pinned_hashes()
    changed = [f'{rel} (SHA-256 {digest[:16]}..., pinned {pins.get(rel, "none")[:16]}...)'
               for rel, digest in files.items() if pins.get(rel) != digest]
    if changed:
        raise ProbeError(f'staged code tree {code_dir} differs from the pinned snapshot: {"; ".join(changed)}. '
                         'Remedy: python -m hyak.code.stage_code --verify (re-stage with python -m '
                         'hyak.code.stage_code if it fails), or pass --code-dir of an unmodified tree')
    loaded = sys.modules.get('hf_hypernet')
    if loaded is not None:
        locations = [Path(p).resolve() for p in getattr(loaded, '__path__', [])]
        if (code_dir / 'hf_hypernet').resolve() not in locations:
            raise ProbeError(f'hf_hypernet is already imported from {locations}, not from {code_dir}; '
                             'run the probe in a fresh process')
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    modeling = importlib.import_module('hf_hypernet.modeling_hypernet')
    identity = importlib.import_module('hf_hypernet.identity_warmup')
    tokenizer_inputs = importlib.import_module('hf_hypernet.tokenizer_inputs')
    smollm = importlib.import_module('hf_hypernet.smollm_setup')
    if helpers == 'import':
        name = 'hyak_probe_staged_trainer'
        spec = importlib.util.spec_from_file_location(name, trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        previous_rust, previous_path = sys.modules.get('rust_utils'), list(sys.path)
        sys.modules[name] = trainer
        try:
            spec.loader.exec_module(trainer)   # T:22-46: inserts the tree on sys.path, loads the Rust library
        except (ImportError, OSError) as exc:
            # Undo the half-finished import (T:27-30 registers sys.modules['rust_utils'] before loading it).
            del sys.modules[name]
            if previous_rust is None:
                sys.modules.pop('rust_utils', None)
            else:
                sys.modules['rust_utils'] = previous_rust
            sys.path[:] = previous_path
            raise ProbeError(f'importing the staged trainer {trainer_path} failed ({exc}); check the Rust '
                             'library with check_env.py, or use --helpers source') from exc
        joint = importlib.import_module('hf_hypernet.joint_training')
        functions = {helper: getattr(trainer, helper) for helper in TRAINER_HELPERS}
        if trainer.ZettHypernet is not modeling.ZettHypernet:
            raise ProbeError('the staged trainer imported a different ZettHypernet class')
    else:
        joint = importlib.import_module('hf_hypernet.joint_training')  # rust_utils: empty namespace package
        functions = _trainer_helpers_from_source(trainer_path, {'zett_main_learning_rate':
                                                                joint.zett_main_learning_rate})
    return SimpleNamespace(
        code_dir=str(code_dir), helpers=helpers, files=files,
        ZettHypernet=modeling.ZettHypernet,
        enable_byte_fallbacks=tokenizer_inputs.enable_byte_fallbacks,
        freeze_rescalers=identity.freeze_rescalers,
        separate_input_embeddings_and_freeze_output=smollm.separate_input_embeddings_and_freeze_output,
        make_lm_microbatch=joint.make_lm_microbatch,
        input_lexical_loss=joint.input_lexical_loss,
        make_optimizer_groups=functions['make_optimizer_groups'],
        set_learning_rates=functions['set_learning_rates'],
    )


def pinned_hashes() -> dict:
    """``{path inside the uniform code tree: pinned SHA-256}`` (``hyak.code.pinned``)."""
    from hyak.code import pinned
    return {rel: pinned.SNAPSHOT_FILES[source][0] for rel, source in pinned.tree_layout('uniform').items()}


def _trainer_helpers_from_source(trainer_path, namespace_globals) -> dict:
    """Execute the pinned source text of the trainer helpers (T:54-74) without importing the trainer."""
    source = Path(trainer_path).read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    namespace = {'__builtins__': builtins, **namespace_globals}
    for helper in TRAINER_HELPERS:
        node = nodes.get(helper)
        if node is None:
            raise ProbeError(f'{trainer_path} has no function {helper}')
        local = {arg.arg for arg in node.args.args}
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assigned = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        free = used - local - assigned - set(dir(builtins))
        if not free <= TRAINER_HELPER_GLOBALS:
            raise ProbeError(f'{helper} in {trainer_path} uses unexpected globals {sorted(free)}')
        exec(compile(ast.get_source_segment(source, node), str(trainer_path), 'exec'), namespace)  # noqa: S102
    return {helper: namespace[helper] for helper in TRAINER_HELPERS}


# --- models (T:421-455) -------------------------------------------------------------------------------
def load_models(code, model_dir, hypernet_dir, device, *, seed=42) -> SimpleNamespace:
    """LM, tokenizer, hypernetwork and optimizer prepared exactly as the trainer does (T:367-369, 421-455).

    Like the trainer this sets ``torch.set_num_threads(1)`` for the rest of the process.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir, hypernet_dir = Path(model_dir), Path(hypernet_dir)
    for directory, files in ((model_dir, ('config.json', 'model.safetensors', 'tokenizer.json')),
                             (hypernet_dir, ('config.json', 'model.safetensors'))):
        missing = [f for f in files if not (directory / f).is_file()]
        if missing:
            raise ProbeError(f'{directory} lacks {missing}; run the fetch/init stages first or pass the directory')
    torch.set_num_threads(1)                                                           # T:367
    torch.manual_seed(seed)                                                            # T:368
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)                                               # T:369
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)   # T:421
    if tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN) != SPECIAL_ID:                    # T:422-423
        raise ProbeError("This runner expects SmolLM2's end-of-text token at ID zero")
    lm = AutoModelForCausalLM.from_pretrained(                                         # T:424-427
        str(model_dir), local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True,
    ).to(device)
    if lm.config.eos_token_id != SPECIAL_ID or lm.config.bos_token_id != SPECIAL_ID:    # T:428-429
        raise ProbeError('SmolLM2 model special IDs differ from the tokenizer')
    code.separate_input_embeddings_and_freeze_output(lm)                               # T:430
    hn = code.ZettHypernet.from_pretrained(str(hypernet_dir), local_files_only=True).to(device)  # T:436-439
    fallback = code.enable_byte_fallbacks(hn, tokenizer)                               # T:440
    code.freeze_rescalers(hn)                                                          # T:441
    hn.model.activation_checkpointing = True                                           # T:442
    hn.activation_checkpoint_input = True                                              # T:443
    if not hn.config.hn_add_inter_token_attention or not hn.config.hn_embed_target_priors:  # T:444-445
        raise ProbeError('Expected vocabulary attention and target priors')
    if hn.config.hn_surface_maxlen != PIECES:                                           # T:446
        raise ProbeError(f'hn_surface_maxlen must be {PIECES}, got {hn.config.hn_surface_maxlen}')
    if lm.get_input_embeddings().weight.requires_grad or lm.get_output_embeddings().weight.requires_grad:
        raise ProbeError('the LM embedding tables must be frozen')                     # T:447-448
    source = lm.get_input_embeddings().weight.detach()                                 # T:449
    hn_params = [p for p in hn.parameters() if p.requires_grad]                        # T:450
    lm_params = [p for p in lm.parameters() if p.requires_grad]                        # T:451
    optimizer = torch.optim.AdamW(                                                     # T:452-455
        code.make_optimizer_groups(hn, 0.0) + code.make_optimizer_groups(lm, 0.0),
        betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    lm.train()                                                                         # T:499
    hn.train()                                                                         # T:500
    return SimpleNamespace(lm=lm, hn=hn, tokenizer=tokenizer, source=source, fallback=fallback,
                           hn_params=hn_params, lm_params=lm_params, optimizer=optimizer,
                           source_vocab=len(tokenizer), width=source.shape[1])


# --- synthetic inputs ------------------------------------------------------------------------------------
def synthetic_inputs(vocab, *, source_vocab, fallback_ids=(), seed=0, positions=MAX_POSITIONS,
                     examples=EXAMPLES_PER_UPDATE, device='cpu'):
    """Surface forms ``[vocab, 10]``, both prior kinds and ``examples`` training examples.

    Forms look like ``build_surface_forms`` output: row 0 is end-of-text ``[0, pad...]``, about
    15 % of rows are one canonical piece (lexical-loss overlap), the rest 2..10 pieces with right
    padding (pad id = ``source_vocab``) and about 1 % byte-fallback ids.  Priors: zeros (BPE and
    WordPiece pool members) and Unigram-like log-probabilities.  Every example has exactly
    ``positions`` positions (BOS + prefix + continuation; 128 is the trainer's maximum and the
    worst case for LM memory) as ``(prefix_ids, continuation_ids, continuation_bytes)``.
    """
    if not 3 <= positions <= MAX_POSITIONS:
        raise ValueError(f'positions must be in 3..{MAX_POSITIONS}, got {positions}')
    generator = torch.Generator().manual_seed(seed)
    pad = source_vocab
    forms = torch.randint(1, source_vocab, (vocab, PIECES), generator=generator)
    lengths = torch.randint(2, PIECES + 1, (vocab,), generator=generator)
    single = torch.rand(vocab, generator=generator) < 0.15
    lengths[single] = 1
    forms[torch.arange(PIECES)[None, :] >= lengths[:, None]] = pad
    fallback_ids = list(fallback_ids)
    if fallback_ids:
        chosen = (torch.rand(vocab, PIECES, generator=generator) < 0.01) & (forms != pad)
        chosen[single] = False
        ids = torch.tensor(fallback_ids)[torch.randint(0, len(fallback_ids), (vocab, PIECES), generator=generator)]
        forms[chosen] = ids[chosen]
    forms[0] = pad
    forms[0, 0] = SPECIAL_ID
    priors = {
        'zero': torch.zeros(vocab, dtype=torch.float32, device=device),
        'unigram': (-(torch.rand(vocab, generator=generator) * 12 + 2)).to(device),
    }
    prefix_len = (positions - 1) // 2
    continuation_len = positions - 1 - prefix_len
    batch = []
    for _ in range(examples):
        prefix = torch.randint(0, vocab, (prefix_len,), generator=generator).tolist()
        continuation = torch.randint(1, source_vocab, (continuation_len,), generator=generator).tolist()
        batch.append((prefix, continuation, 4 * continuation_len))
    return forms.to(device), priors, batch


# --- one training update (T:542-584) ------------------------------------------------------------------
def training_update(models, code, forms, priors, examples, update, *, device_type, microbatch_size=32):
    """One update exactly as the pinned trainer runs it (T:542-584), on ``device_type``.

    Returns the trainer's ``train`` metrics (T:586-596) that do not need the corpus or pool.
    """
    lm, hn, optimizer, source = models.lm, models.hn, models.optimizer, models.source
    if len(examples) != EXAMPLES_PER_UPDATE:
        raise ValueError(f'an update has {EXAMPLES_PER_UPDATE} examples, got {len(examples)}')
    if microbatch_size < 1 or EXAMPLES_PER_UPDATE % microbatch_size:
        raise ValueError('--microbatch-size must divide the 128-example update')
    total_scored = sum(len(x[1]) for x in examples)                                    # T:542
    hn_lr, lm_lr = code.set_learning_rates(optimizer, update)                          # T:543
    optimizer.zero_grad(set_to_none=True)                                              # T:544
    with torch.autocast(device_type, dtype=torch.bfloat16):                            # T:545-548
        generated, output_embeddings, _ = hn(
            forms, priors, source, None, deterministic=False
        )
    if output_embeddings is not None or not torch.isfinite(generated).all():           # T:549-550
        raise FloatingPointError("Hypernetwork produced invalid input embeddings")
    leaf = generated.detach().float().requires_grad_(True)                             # T:551
    loss_nats = 0.0                                                                    # T:552
    for start in range(0, 128, microbatch_size):                                       # T:553
        micro = examples[start:start + microbatch_size]                                # T:554
        embeds, mask, labels, positions = code.make_lm_microbatch(                     # T:555-557
            micro, leaf, source, device_type
        )
        with torch.autocast(device_type, dtype=torch.bfloat16):                        # T:558-561
            loss = lm(inputs_embeds=embeds, attention_mask=mask,
                      position_ids=positions, labels=labels,
                      use_cache=False).loss
        if not torch.isfinite(loss):                                                   # T:562-563
            raise FloatingPointError("Non-finite LM loss")
        count = sum(len(x[1]) for x in micro)                                          # T:564
        loss_nats += float(loss.detach()) * count                                      # T:565
        (loss * count / total_scored).backward()                                       # T:566
        del embeds, mask, labels, positions, loss                                      # T:567
    lm_phase_peak = torch.cuda.max_memory_allocated() if device_type == 'cuda' else None  # probe only
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():                       # T:568-569
        raise FloatingPointError("Non-finite accumulated embedding gradient")
    lexical = code.input_lexical_loss(                                                 # T:570-573
        generated, forms, source, hn.config.pad_token_id,
        SPECIAL_ID,
    )
    if not torch.isfinite(lexical):                                                    # T:574-575
        raise FloatingPointError("Non-finite input lexical loss")
    lexical_value = float(lexical.detach())                                            # T:576
    ((generated.float() * leaf.grad.detach()).sum() + 0.5 * lexical).backward()        # T:577
    grad_norm = torch.nn.utils.clip_grad_norm_(models.hn_params + models.lm_params, 0.1,  # T:578-579
                                               foreach=False)
    if not torch.isfinite(grad_norm):                                                  # T:580-581
        raise FloatingPointError("Non-finite global gradient norm")
    optimizer.step()                                                                   # T:582
    optimizer.zero_grad(set_to_none=True)                                              # T:583
    del generated, leaf, lexical                                                       # T:584
    return {'loss_nats_per_token': loss_nats / total_scored, 'input_lexical_loss': lexical_value,
            'scored_tokens': total_scored, 'hn_lr': hn_lr, 'lm_lr': lm_lr,
            'gradient_norm_before_clip': float(grad_norm), 'tokenizer_vocab_size': int(forms.shape[0]),
            'lm_phase_max_allocated_bytes': lm_phase_peak}


def is_out_of_memory(exc) -> bool:
    """CUDA (or host) allocation failure, as opposed to a real error."""
    if isinstance(exc, getattr(torch, 'OutOfMemoryError', ())) or isinstance(exc, MemoryError):
        return True
    text = str(exc)
    return isinstance(exc, RuntimeError) and ('out of memory' in text.lower() or 'CUBLAS_STATUS_ALLOC_FAILED' in text)


def release_memory(models, device):
    """Drop gradients and cached blocks between configurations (the optimizer state is kept)."""
    models.optimizer.zero_grad(set_to_none=True)
    models.hn.zero_grad(set_to_none=True)
    models.lm.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:  # accelerate 1.14.0 requires psutil, so it is installed; RSS stays optional
        return None


def measure_update(function, device) -> dict:
    """Run ``function()`` (one update) and add seconds and memory peaks (CUDA) or RSS (CPU)."""
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        stats_before = torch.cuda.memory_stats(device)
    started = time.perf_counter()
    metrics = function()
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    record = {'seconds': round(seconds, 3), **metrics}
    if cuda:
        stats = torch.cuda.memory_stats(device)
        record.update(
            max_allocated_bytes=torch.cuda.max_memory_allocated(device),
            max_reserved_bytes=torch.cuda.max_memory_reserved(device),
            allocated_after_bytes=torch.cuda.memory_allocated(device),
            alloc_retries=stats.get('num_alloc_retries', 0) - stats_before.get('num_alloc_retries', 0),
        )
    else:
        record['rss_after_bytes'] = _rss_bytes()
    return record


def run_configuration(models, code, *, vocab, chunk_rows, updates, device, seed, positions,
                      microbatch_size) -> dict:
    """``updates`` training updates at one vocabulary size and chunk setting; never raises on OOM.

    ``prediction`` (``chunked_hypernet.predict_peak_gib``, steady state with the AdamW moments
    allocated) is given only for the real width (2048) on a GPU.
    """
    label = f'V={vocab} ' + (f'chunk_rows={chunk_rows}' if chunk_rows > 0 else 'pinned')
    real_size = device.type == 'cuda' and models.width == mm.PREDICTION['width']
    result = {'vocab': vocab, 'chunk_rows': chunk_rows, 'label': label, 'updates': [], 'oom': None,
              'error': None, 'prediction': mm.predict_peak_gib(vocab, chunk_rows) if real_size else None,
              'allocated_before_bytes': torch.cuda.memory_allocated(device) if device.type == 'cuda' else None}
    forms, priors, examples = synthetic_inputs(
        vocab, source_vocab=models.source_vocab, fallback_ids=sorted(models.fallback.values()), seed=seed + vocab,
        positions=positions, device=device)
    kinds = ('zero', 'unigram')
    try:
        # the floor is checked by run_probe (training refuses small chunks; a CPU smoke run may use them)
        with ch.patched(code.ZettHypernet, chunk_rows, min_chunk_rows=1):
            for update in range(updates):
                kind = kinds[update % 2]
                record = measure_update(
                    lambda: training_update(models, code, forms, priors[kind], examples, update,
                                            device_type=device.type, microbatch_size=microbatch_size),
                    device)
                record.update(update=update, priors=kind)
                result['updates'].append(record)
    except Exception as exc:  # noqa: BLE001 - OOM and errors are results here; KeyboardInterrupt propagates
        kind = 'oom' if is_out_of_memory(exc) else 'error'
        first_line = (str(exc).splitlines() or [''])[0][:400]
        result[kind] = f'{type(exc).__name__} in update {len(result["updates"]) + 1}: {first_line}'
        exc = None  # drop the traceback (and the frames holding GPU tensors) before releasing memory
        if device.type == 'cpu':
            # CPU only: a failure inside one chunk region during backward leaves the other regions' tasks
            # queued on this thread (keeping the graph alive) until the next backward runs.
            torch.ones(1, requires_grad=True).sum().backward()
    finally:
        del forms, priors, examples
        release_memory(models, device)
    return result


# --- verdicts -----------------------------------------------------------------------------------------
def verdict(result, *, capacity_bytes, min_headroom_gib, device_type) -> dict:
    """``{'verdict', 'peak_reserved_gib', 'peak_allocated_gib', 'headroom_gib', 'seconds_per_update', 'line'}``.

    ``seconds_per_update`` is the last update's time (the first includes CUDA start-up and allocator
    growth; it is reported as ``first_update_seconds``).  ``peak_phase`` names the phase of the update
    that set the peak (``hypernetwork backward`` or ``LM micro-batches``, which includes the forward).
    """
    out = {'vocab': result['vocab'], 'chunk_rows': result['chunk_rows'], 'label': result['label'],
           'peak_reserved_gib': None, 'peak_allocated_gib': None, 'headroom_gib': None,
           'seconds_per_update': None, 'first_update_seconds': None, 'peak_phase': None}
    prediction = result.get('prediction') or {}
    predicted = prediction.get('gib')
    if result['updates']:
        out['seconds_per_update'] = round(result['updates'][-1]['seconds'], 2)
        out['first_update_seconds'] = round(result['updates'][0]['seconds'], 2)
    if result['updates'] and device_type == 'cuda':
        reserved = max(u['max_reserved_bytes'] for u in result['updates'])
        peak = max(result['updates'], key=lambda u: u['max_allocated_bytes'])
        allocated = peak['max_allocated_bytes']
        out['peak_reserved_gib'] = round(reserved / GIB, 2)
        out['peak_allocated_gib'] = round(allocated / GIB, 2)
        out['headroom_gib'] = round((capacity_bytes - reserved) / GIB, 2)
        lm_peak = peak.get('lm_phase_max_allocated_bytes')
        if lm_peak is not None:
            out['peak_phase'] = mm.PHASE_HYPERNETWORK if allocated > lm_peak else mm.PHASE_LM
    if result['error']:
        out['verdict'] = 'ERROR'
        detail = result['error']
    elif result['oom']:
        out['verdict'] = 'DOES NOT FIT'
        detail = f'out of memory: {result["oom"]}'
    elif device_type != 'cuda':
        out['verdict'] = 'MEASURED'
        detail = (f'{device_type} run, no GPU memory verdict; {out["seconds_per_update"]} s/update '
                  f'(first update {out["first_update_seconds"]} s)')
    else:
        fits = out['headroom_gib'] >= min_headroom_gib
        out['verdict'] = 'FITS' if fits else 'DOES NOT FIT'
        detail = (f'peak reserved {out["peak_reserved_gib"]} GiB, allocated {out["peak_allocated_gib"]} GiB of '
                  f'{capacity_bytes / GIB:.1f} GiB; headroom {out["headroom_gib"]} GiB'
                  + ('' if fits else f' (below the required {min_headroom_gib} GiB)')
                  + (f'; peak set by the {out["peak_phase"]}' if out['peak_phase'] else '')
                  + f'; {out["seconds_per_update"]} s/update (first update {out["first_update_seconds"]} s)')
    if predicted is not None:
        detail += f'; predicted {predicted} GiB ({prediction.get("phase")})'
    setting = f'chunk_rows={result["chunk_rows"]}' if result['chunk_rows'] > 0 else 'pinned'
    out['line'] = f'VERDICT V={result["vocab"]} {setting}: {out["verdict"]} ({detail})'
    return out


def fit_linear(points):
    """Least-squares ``peak_gib = intercept + slope * vocab`` over ``[(vocab, peak_gib)]`` (>= 2 points)."""
    if len(points) < 2 or len({v for v, _ in points}) < 2:
        return None
    n = len(points)
    mean_v = sum(v for v, _ in points) / n
    mean_p = sum(p for _, p in points) / n
    slope = sum((v - mean_v) * (p - mean_p) for v, p in points) / sum((v - mean_v) ** 2 for v, _ in points)
    return {'intercept_gib': round(mean_p - slope * mean_v, 3), 'gib_per_1000_rows': round(slope * 1000, 4),
            'points': n}


# --- equivalence on the device ------------------------------------------------------------------------
def _relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def _deterministic_pass(models, code, call, forms, priors, cotangent, autocast):
    hn, source = models.hn, models.source
    hn.zero_grad(set_to_none=True)
    with torch.autocast(source.device.type, dtype=torch.bfloat16, enabled=autocast):
        generated, second, _ = call(hn, forms, priors, source, None, True)
    lexical = code.input_lexical_loss(generated, forms, source, hn.config.pad_token_id, SPECIAL_ID)
    ((generated.float() * cotangent).sum() + 0.5 * lexical).backward()
    grads = {name: p.grad.detach().float().cpu().clone() for name, p in hn.named_parameters() if p.grad is not None}
    out = generated.detach().float().cpu().clone()
    hn.zero_grad(set_to_none=True)
    return out, second, grads


#: Gradients that are analytically zero (softmax is invariant to the per-query shift a key bias adds),
#: so their values are pure round-off and excluded from the per-parameter comparison.
ANALYTICALLY_ZERO_GRADIENTS = ('.key.bias',)


def _compare(reference, candidate, floor_fraction=1e-2):
    """Relative errors: output (Frobenius), all gradients together, and the worst single parameter.

    The per-parameter error is taken over parameters whose reference gradient norm is at least
    ``floor_fraction`` of the largest one, excluding ``ANALYTICALLY_ZERO_GRADIENTS``.
    """
    out_ref, _, grads_ref = reference
    out, _, grads = candidate
    if set(grads_ref) != set(grads):
        return {'output': None, 'gradient': None, 'gradient_worst': None, 'worst_parameter': 'gradient sets differ'}
    norms = {name: float(g.norm()) for name, g in grads_ref.items()}
    scale = max(norms.values(), default=0.0)
    total_diff = total_ref = 0.0
    worst, worst_name = 0.0, None
    for name, ref in grads_ref.items():
        diff = float((grads[name].double() - ref.double()).norm())
        total_diff += diff ** 2
        total_ref += norms[name] ** 2
        if name.endswith(ANALYTICALLY_ZERO_GRADIENTS) or norms[name] < floor_fraction * scale:
            continue
        if diff / norms[name] > worst:
            worst, worst_name = diff / norms[name], name
    return {'output': _relative(out, out_ref), 'gradient': math.sqrt(total_diff / max(total_ref, 1e-300)),
            'gradient_worst': worst, 'worst_parameter': worst_name}


#: Dropout on: checkpointed versus not checkpointed gradients (same seeds).  Equal masks give at most
#: non-deterministic CUDA kernels' noise (about 1e-7); masks redrawn differently give about 0.7.
CHECKPOINT_GRADIENT_LIMIT = 1e-2


def _seed_all(device, seed):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _dropout_checkpoint_pass(models, forms, priors, cotangent, *, chunk_rows, use_checkpoint, autocast, seed):
    """Seeded dropout-on chunked forward + backward; returns ``(output, {name: grad})`` on the CPU."""
    hn, device = models.hn, models.source.device
    hn.zero_grad(set_to_none=True)
    _seed_all(device, seed)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        out, _, _ = ch.chunked_call(hn, forms, priors, models.source, None, False, chunk_rows=chunk_rows,
                                    use_checkpoint=use_checkpoint)
    (out.float() * cotangent).sum().backward()
    grads = {name: p.grad.detach().float().cpu().clone() for name, p in hn.named_parameters() if p.grad is not None}
    out = out.detach().float().cpu().clone()
    hn.zero_grad(set_to_none=True)
    return out, grads


def checkpoint_dropout_check(models, forms, priors, cotangent, *, chunk_rows, device, seed) -> dict:
    """Dropout on: gradients with and without checkpointing, fp32 and bf16 autocast (recomputed masks)."""
    report = {}
    for name, autocast in (('float32', False), ('bfloat16', True)):
        out_ckpt, grads_ckpt = _dropout_checkpoint_pass(models, forms, priors, cotangent, chunk_rows=chunk_rows,
                                                        use_checkpoint=True, autocast=autocast, seed=seed)
        out_plain, grads_plain = _dropout_checkpoint_pass(models, forms, priors, cotangent, chunk_rows=chunk_rows,
                                                          use_checkpoint=False, autocast=autocast, seed=seed)
        compared = _compare((out_plain, None, grads_plain), (out_ckpt, None, grads_ckpt))
        report[name] = {'outputs_equal': bool(torch.equal(out_ckpt, out_plain)), 'gradient': compared['gradient'],
                        'gradient_worst': compared['gradient_worst'], 'worst_parameter': compared['worst_parameter']}
        del out_ckpt, grads_ckpt, out_plain, grads_plain
    report['limit'] = CHECKPOINT_GRADIENT_LIMIT
    report['passed'] = all(report[k]['outputs_equal'] and isinstance(report[k]['gradient'], float)
                           and math.isfinite(report[k]['gradient']) and report[k]['gradient'] <= CHECKPOINT_GRADIENT_LIMIT
                           for k in ('float32', 'bfloat16'))
    return report


def equivalence_check(models, code, *, rows, chunk_rows, device, seed=0) -> dict:
    """Chunked versus pinned on the device (deterministic), and the dropout checks (RNG, recomputed masks)."""
    hn = models.hn
    forms, priors, _ = synthetic_inputs(rows, source_vocab=models.source_vocab,
                                        fallback_ids=sorted(models.fallback.values()), seed=seed, examples=0,
                                        device=device)
    priors = priors['unigram']
    generator = torch.Generator().manual_seed(seed + 1)
    cotangent = torch.randn(rows, models.width, generator=generator).to(device) * 1e-2
    original = ch.original_call(code.ZettHypernet)

    def chunked(hn_, forms_, priors_, source_, lang_, det_):
        return ch.chunked_call(hn_, forms_, priors_, source_, lang_, det_, chunk_rows=chunk_rows)

    pinned32 = _deterministic_pass(models, code, original, forms, priors, cotangent, autocast=False)
    chunked32 = _deterministic_pass(models, code, chunked, forms, priors, cotangent, autocast=False)
    pinned16 = _deterministic_pass(models, code, original, forms, priors, cotangent, autocast=True)
    chunked16 = _deterministic_pass(models, code, chunked, forms, priors, cotangent, autocast=True)
    fp32 = _compare(pinned32, chunked32)
    bf16 = _compare(pinned16, chunked16)
    floor = _compare(pinned32, pinned16)
    del pinned32, chunked32, pinned16, chunked16
    fp32_limits = {'output': 1e-4, 'gradient': 1e-4, 'gradient_worst': 1e-3}
    fp32_ok = _within(fp32, fp32_limits)
    # bf16: chunking may not add more error than bf16 rounding itself (pinned bf16 versus pinned fp32).
    minimum = {'output': 1e-2, 'gradient': 1e-2, 'gradient_worst': 5e-2}
    bf16_limits = {key: max(3 * floor[key], low) if isinstance(floor[key], float) and math.isfinite(floor[key])
                   else low for key, low in minimum.items()}
    bf16_ok = _within(bf16, bf16_limits)
    # Dropout on: reproducible forward from a seed, and the RNG continues where the forward left it.
    rng = _rng_state(device)
    _seed_all(device, seed + 2)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        first, _, _ = ch.chunked_call(hn, forms, priors, models.source, None, False, chunk_rows=chunk_rows)
    after_forward = _rng_state(device)
    (first.float() * cotangent).sum().backward()
    after_backward = _rng_state(device)
    hn.zero_grad(set_to_none=True)
    _seed_all(device, seed + 2)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        again, _, _ = ch.chunked_call(hn, forms, priors, models.source, None, False, chunk_rows=chunk_rows)
    reproducible = bool(torch.equal(first.detach(), again.detach()))
    continuity = all(torch.equal(a, b) for a, b in zip(after_forward, after_backward))
    del first, again
    hn.zero_grad(set_to_none=True)
    checkpointed = checkpoint_dropout_check(models, forms, priors, cotangent, chunk_rows=chunk_rows, device=device,
                                            seed=seed + 3)
    _restore_rng(device, rng)
    del forms, priors, cotangent
    release_memory(models, device)
    passed = fp32_ok and bf16_ok and reproducible and continuity and checkpointed['passed']
    return {'rows': rows, 'chunk_rows': chunk_rows, 'float32': fp32, 'bfloat16': bf16,
            'bf16_rounding_floor': floor, 'bf16_limits': bf16_limits, 'float32_limits': fp32_limits,
            'dropout_forward_reproducible': reproducible, 'rng_continues_after_backward': continuity,
            'dropout_checkpoint_vs_direct': checkpointed, 'passed': passed}


def _within(errors, limits) -> bool:
    """True when every error is a finite number within its limit (NaN and None fail)."""
    return all(isinstance(errors[key], float) and math.isfinite(errors[key]) and errors[key] <= limit
               for key, limit in limits.items())


def _rng_state(device):
    states = [torch.get_rng_state()]
    if device.type == 'cuda':
        states.append(torch.cuda.get_rng_state(device))
    return states


def _restore_rng(device, states):
    torch.set_rng_state(states[0])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(states[1], device)


# --- the run ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m hyak.memory.probe', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, help='default: $ANYTOK_ROOT/input/model')
    parser.add_argument('--hypernet', type=Path, help='default: $ANYTOK_ROOT/input/hypernet_init')
    parser.add_argument('--code-dir', type=Path, help='default: $ANYTOK_ROOT/code/uniform')
    parser.add_argument('--vocab', type=int, nargs='+', default=list(DEFAULT_VOCABS))
    parser.add_argument('--chunk-rows', type=int, nargs='+',
                        help='rows per chunk (0 = pinned forward); the first is the configured setting '
                             f'(default HYAK_HN_CHUNK_ROWS); training refuses 1..{ch.MIN_CHUNK_ROWS - 1}')
    parser.add_argument('--also-pinned', action='store_true',
                        help='also measure the pinned forward, after the configured setting (information only)')
    parser.add_argument('--equivalence-rows', type=int, default=4096, help='0 skips the equivalence check')
    parser.add_argument('--equivalence-chunk-rows', type=int,
                        help='default: as many chunks as the configured chunk size gives at the largest --vocab '
                             '(at least 3); ceil(rows / 3) when the configured setting is the pinned forward')
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--microbatch-size', type=int, default=32)
    parser.add_argument('--positions', type=int, default=MAX_POSITIONS,
                        help='positions per synthetic example (128 = trainer maximum, worst case)')
    parser.add_argument('--min-headroom-gib', type=float, default=8.0)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--helpers', default='import', choices=['import', 'source'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', type=Path, help='default: <repo>/modal_results/hyak_tillicum/probe/probe-report.json')
    parser.add_argument('--config', type=Path, help='config file (default $HYAK_CONFIG or hyak/config.env)')
    args = parser.parse_args(argv)
    if any(v < 1 for v in args.vocab):
        parser.error('--vocab values must be positive')
    if args.chunk_rows is not None and any(r < 0 for r in args.chunk_rows):
        parser.error('--chunk-rows values must be >= 0')
    if args.updates < 1:
        parser.error('--updates must be >= 1')
    if args.microbatch_size < 1 or EXAMPLES_PER_UPDATE % args.microbatch_size:
        parser.error('--microbatch-size must divide 128')
    if not 3 <= args.positions <= MAX_POSITIONS:
        parser.error(f'--positions must be in 3..{MAX_POSITIONS}')
    if args.equivalence_rows < 0 or (args.equivalence_chunk_rows is not None and args.equivalence_chunk_rows < 1):
        parser.error('--equivalence-rows must be >= 0 and --equivalence-chunk-rows >= 1')
    return args


def _device_info(device):
    info = {'type': device.type, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        info.update(name=properties.name, total_memory_bytes=properties.total_memory,
                    free_at_start_bytes=free, total_reported_bytes=total,
                    capability=f'{properties.major}.{properties.minor}')
    return info


def run_probe(args, *, cfg, paths, writer=None) -> dict:
    """Run every measurement; returns the report dict (also written to disk)."""
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ProbeError('CUDA is not available: run the probe inside a GPU job (submit.sh probe) or pass '
                         '--device cpu for a local smoke run')
    chunk_settings = list(dict.fromkeys(args.chunk_rows if args.chunk_rows else
                                        [int_value(cfg, 'HYAK_HN_CHUNK_ROWS')]))
    configured = chunk_settings[0]
    if args.also_pinned and 0 not in chunk_settings:
        chunk_settings.append(0)
    small = [rows for rows in chunk_settings if 0 < rows < ch.MIN_CHUNK_ROWS]
    if configured in small and device.type == 'cuda':
        raise ProbeError(f'the configured chunk size {configured} is below the floor of {ch.MIN_CHUNK_ROWS} rows '
                         'that training enforces (more, smaller chunks sum the weight gradients in bf16 and do not '
                         'lower the peak). Remedy: set HYAK_HN_CHUNK_ROWS (or the first --chunk-rows) to the '
                         f'largest value that fits, at least {ch.MIN_CHUNK_ROWS} (default 16384), or 0')
    for rows in small:
        _log(f'WARNING: chunk_rows={rows} is below the floor of {ch.MIN_CHUNK_ROWS} rows; training refuses it '
             '(measured anyway, information only)')
    model_dir = args.model_dir or paths.model_dir
    hypernet_dir = args.hypernet or paths.hypernet_init
    code_dir = args.code_dir or paths.code_dir('uniform')
    eq_rows = args.equivalence_rows
    eq_chunk = (args.equivalence_chunk_rows or equivalence_chunk_rows(eq_rows, configured, args.vocab)) if eq_rows else None
    # the configured setting for every vocabulary first (it decides the exit code), then the comparisons
    configs = [(vocab, rows) for rows in chunk_settings for vocab in args.vocab]
    total = len(configs) + (1 if eq_rows else 0)
    done = 0
    started = time.time()

    def status(stage, **extra):
        if writer is not None:
            writer.update(done, stage=stage, total=total, elapsed_seconds=round(time.time() - started, 1), **extra)

    status('loading models')
    _log(f'code {code_dir} (helpers: {args.helpers}); model {model_dir}; hypernetwork {hypernet_dir}; '
         f'device {device}; chunk settings {chunk_settings} (configured {configured})')
    code = load_code(code_dir, args.helpers)
    models = load_models(code, model_dir, hypernet_dir, device, seed=args.seed)
    device_info = _device_info(device)
    capacity = device_info.get('total_memory_bytes', 0)
    _log(f'device: {json.dumps(device_info, sort_keys=True)}')
    if device.type == 'cuda':
        _log(f'allocated after loading: {torch.cuda.memory_allocated(device) / GIB:.2f} GiB')
    report = {
        'version': 1, 'started_at': utc_now(), 'job_id': slurm.job_id(), 'host': slurm.node_name(),
        'device': device_info, 'code': {'dir': str(code_dir), 'helpers': args.helpers, 'files': code.files},
        'model_dir': str(model_dir), 'hypernet_dir': str(hypernet_dir),
        'settings': {'vocab': args.vocab, 'chunk_settings': chunk_settings, 'configured_chunk_rows': configured,
                     'updates': args.updates, 'microbatch_size': args.microbatch_size, 'positions': args.positions,
                     'min_headroom_gib': args.min_headroom_gib, 'seed': args.seed},
        'chunked_module': {**{key: value for key, value in ch.settings().items()
                              if key in ('module', 'module_sha256', 'algorithm_version', 'pinned_sources',
                                         'torch_version')}, 'min_chunk_rows': ch.MIN_CHUNK_ROWS},
        'equivalence': None, 'measurements': [], 'verdicts': [],
    }

    if eq_rows:
        status('equivalence')
        _log(f'equivalence check: {eq_rows} rows, chunks of {eq_chunk}')
        try:
            report['equivalence'] = equivalence_check(models, code, rows=eq_rows, chunk_rows=eq_chunk,
                                                      device=device, seed=args.seed)
        except Exception as exc:  # noqa: BLE001 - reported, the measurements still run
            report['equivalence'] = {'passed': False, 'error': f'{type(exc).__name__}: {str(exc)[:400]}'}
            if device.type == 'cpu':   # release a graph left queued by a failed chunked backward (CPU only)
                torch.ones(1, requires_grad=True).sum().backward()
            exc = None
            release_memory(models, device)
        equivalence = report['equivalence']
        _log(f'EQUIVALENCE {"PASS" if equivalence["passed"] else "FAIL"}: '
             + json.dumps({k: equivalence.get(k) for k in ('float32', 'bfloat16', 'bf16_rounding_floor',
                                                           'dropout_forward_reproducible',
                                                           'rng_continues_after_backward',
                                                           'dropout_checkpoint_vs_direct', 'error')
                           if k in equivalence}, sort_keys=True))
        done += 1
        if writer is not None:
            writer.event(jsonable({'event': 'equivalence', **equivalence}))

    for vocab, rows in configs:
        label = f'V={vocab} ' + (f'chunk_rows={rows}' if rows > 0 else 'pinned')
        status(label)
        _log(f'measuring {label}: {args.updates} updates')
        result = run_configuration(models, code, vocab=vocab, chunk_rows=rows, updates=args.updates,
                                   device=device, seed=args.seed, positions=args.positions,
                                   microbatch_size=args.microbatch_size)
        judged = verdict(result, capacity_bytes=capacity, min_headroom_gib=args.min_headroom_gib,
                         device_type=device.type)
        report['measurements'].append(result)
        report['verdicts'].append(judged)
        _log(judged['line'])
        done += 1
        if writer is not None:
            writer.event(jsonable({'event': 'measurement', 'verdict': judged, 'updates': result['updates'],
                                   'oom': result['oom'], 'error': result['error']}))

    report.update(_summarize(report, configured, capacity, args.min_headroom_gib, device.type))
    report['finished_at'] = utc_now()
    _log(report['summary'])
    return report


def equivalence_chunk_rows(rows: int, configured: int, vocabs) -> int:
    """Chunk size of the equivalence check: as many chunks as ``configured`` gives at the largest vocabulary.

    At least 3 chunks; ``ceil(rows / 3)`` when the configured setting is the pinned forward (0).
    """
    chunks = 3 if configured <= 0 else max(3, math.ceil(max(vocabs) / configured))
    return max(1, math.ceil(rows / chunks))


def _summarize(report, configured, capacity, min_headroom_gib, device_type) -> dict:
    """SUMMARY line and ``ok``.  Only the configured setting and the equivalence decide ``ok``."""
    mine = [v for v in report['verdicts'] if v['chunk_rows'] == configured]
    fits = [v for v in mine if v['verdict'] in ('FITS', 'MEASURED')]
    failed = [v for v in mine if v not in fits]
    errors = [v for v in report['verdicts'] if v['verdict'] == 'ERROR' and v['chunk_rows'] != configured]
    equivalence = report['equivalence']
    eq_ok = equivalence is None or equivalence.get('passed') is True
    setting = f'chunk_rows={configured}' if configured > 0 else 'pinned forward'
    fit = None
    if device_type == 'cuda':
        points = [(v['vocab'], v['peak_reserved_gib']) for v in mine
                  if v['peak_reserved_gib'] is not None and (configured <= 0 or v['vocab'] >= configured)]
        fit = fit_linear(points)
        if fit and fit['gib_per_1000_rows'] > 0:
            usable = capacity / GIB - min_headroom_gib
            fit['largest_fitting_vocab'] = int((usable - fit['intercept_gib']) / fit['gib_per_1000_rows'] * 1000)
    parts = []
    if not failed and mine:
        largest = max(v['vocab'] for v in mine)
        headrooms = [v['headroom_gib'] for v in mine if v['headroom_gib'] is not None]
        parts.append(f'{setting}: {"FITS" if device_type == "cuda" else "MEASURED"} for every measured '
                     f'vocabulary up to {largest}'
                     + (f' (smallest headroom {min(headrooms)} GiB)' if headrooms else ''))
    elif mine:
        hypernetwork_bound = any(v.get('peak_phase') == mm.PHASE_HYPERNETWORK for v in failed)
        if configured > ch.MIN_CHUNK_ROWS and hypernetwork_bound:
            remedy = (f'remedy: lower HYAK_HN_CHUNK_ROWS (not below {ch.MIN_CHUNK_ROWS}) before the first training '
                      'segment and run the probe again')
        elif configured == 0:
            remedy = 'remedy: set HYAK_HN_CHUNK_ROWS=16384 (the pinned forward needs about 200 GiB) and run the probe again'
        elif any(v.get('peak_phase') == mm.PHASE_LM for v in failed):
            remedy = ('the LM micro-batches set the peak, which HYAK_HN_CHUNK_ROWS does not change: this GPU '
                      'cannot run the configuration; stop and decide with the report')
        else:
            remedy = 'see the VERDICT lines and the report; do not start training'
        parts.append(f'{setting}: DOES NOT FIT or failed at V=' + ', '.join(str(v['vocab']) for v in failed)
                     + '; ' + remedy)
    else:
        parts.append(f'{setting}: nothing measured')
    if errors:
        parts.append(f'{len(errors)} comparison measurement(s) raised errors (information only): '
                     + '; '.join(f'{v["label"]}' for v in errors))
    parts.append('equivalence ' + ('skipped' if equivalence is None else 'PASS' if eq_ok else 'FAIL'))
    if fit and 'largest_fitting_vocab' in fit:
        parts.append(f'linear fit {fit["intercept_gib"]} GiB + {fit["gib_per_1000_rows"]} GiB per 1,000 rows '
                     f'-> largest fitting vocabulary about {fit["largest_fitting_vocab"]}')
    ok = bool(mine) and not failed and eq_ok
    return {'ok': ok, 'fit': fit, 'summary': 'SUMMARY ' + '; '.join(parts)}


def main(argv=None) -> int:
    """Exit codes: 0 fits and equivalence passed; 4 does not fit or equivalence failed; 2 setup/unexpected."""
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
    except (ConfigError, ValueError) as exc:
        print(f'[probe] ERROR: {exc}', file=sys.stderr, flush=True)
        return EXIT_SETUP
    writer = StatusWriter(STAGE_PROBE, None, 'measurements', 'completed', cfg=cfg)
    report_path = args.report or paths.monitor_dir(STAGE_PROBE) / 'probe-report.json'
    try:
        report = run_probe(args, cfg=cfg, paths=paths, writer=writer)
    except (ProbeError, ConfigError) as exc:
        print(f'[probe] ERROR: {exc}', file=sys.stderr, flush=True)
        writer.update(0, stage='failed', state='failed', error=str(exc)[:2000])
        return EXIT_SETUP
    except Exception as exc:  # noqa: BLE001 - loading/driver/library failures are setup errors (contract: 2)
        import traceback
        traceback.print_exc()
        message = f'{type(exc).__name__}: {(str(exc).splitlines() or [""])[0][:1500]}'
        print(f'[probe] ERROR (unexpected, traceback above): {message}. Remedy: check the environment '
              '(python -m hyak.env.check_env), the model and hypernetwork directories and the GPU', file=sys.stderr,
              flush=True)
        writer.update(0, stage='failed', state='failed', error=message[:2000])
        return EXIT_SETUP
    report = jsonable(report)
    atomic_write_json(report_path, report)
    job = slurm.worker_id()
    archive = paths.probe_dir / f'probe-report-{job}.json'
    try:
        archive.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(archive, report)
    except OSError as exc:
        print(f'[probe] WARNING: could not write {archive}: {exc}', file=sys.stderr, flush=True)
    _log(f'report: {report_path} (copy: {archive})')
    total = len(report['verdicts']) + (1 if report['equivalence'] is not None else 0)
    verdicts = [{k: v.get(k) for k in ('label', 'verdict', 'peak_reserved_gib', 'headroom_gib', 'seconds_per_update')}
                for v in report['verdicts']]
    if report['ok']:
        writer.update(total, stage='complete', state='complete', total=total, summary=report['summary'],
                      verdicts=verdicts, report=str(report_path))
        return EXIT_OK
    writer.update(total, stage='does not fit' if report['verdicts'] else 'failed', state='failed', total=total,
                  error=report['summary'], verdicts=verdicts, report=str(report_path))
    return EXIT_CHECK_FAILED


if __name__ == '__main__':
    sys.exit(main())
