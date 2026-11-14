"""Shared helpers for the hyak/tests/test_eval_*.py tests (no tests of its own).

Builds small tokenizers in the pool's file format with the stock ``tokenizers``
library (same model/pre-tokenizer/decoder construction as
scripts/tokenizer_pool.py:144-171, without the patched noise options), a fake
pool directory accepted by the trainer's ``TokenizerPool`` reader, an evaluation
layout rooted in a temporary directory, and a fake GPU scorer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from hyak.common.paths import Paths

REPO = Path(__file__).resolve().parents[2]

#: (name, algorithm, pretokenizer, vocab size) of the synthetic tokenizers used at pool indices 0, 1, 2, ...
TOKENIZERS = (('bpe-regex', 'bpe', 'regex', 6000), ('wp-line', 'wordpiece', 'line256', 6000),
              ('uni-noregex', 'unigram', 'no-regex', 6000))


def asset_dir() -> Path:
    """SmolLM2 test assets as in conftest.scratch_assets ($HYAK_TEST_ASSETS, else <venv parent>/smollm2)."""
    value = os.environ.get('HYAK_TEST_ASSETS')
    return Path(value) if value else Path(os.environ.get('HYAK_TEST_SCRATCH') or Path(sys.prefix).parent) / 'smollm2'


def asset_tokenizer() -> Path:
    """The pinned SmolLM2 tokenizer.json of the test assets; skips the calling test when absent."""
    import pytest
    path = asset_dir() / 'full' / 'tokenizer.json'
    if not path.is_file():
        pytest.skip(f'SmolLM2 tokenizer.json not found at {path} (set HYAK_TEST_ASSETS)')
    return path


def train_tokenizer(algorithm, pretokenizer, vocab, texts, out, normalizer=None) -> int:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    common = dict(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    specials = ['<|endoftext|>'] if algorithm == 'bpe' else ['<|unk|>', '<|endoftext|>']
    common['special_tokens'] = specials
    if algorithm == 'bpe':
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(**common, min_frequency=2, max_token_length=64)
    elif algorithm == 'wordpiece':
        tokenizer = Tokenizer(models.WordPiece(unk_token='<|unk|>', max_input_chars_per_word=1_000_000))
        trainer = trainers.WordPieceTrainer(**common, min_frequency=2, continuing_subword_prefix='')
    else:
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(**common, max_piece_length=64, unk_token='<|unk|>')
    if pretokenizer == 'line256':
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r'[^\n]{1,256}\n?|\n'), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=pretokenizer == 'regex')
    tokenizer.decoder = decoders.ByteLevel()
    if normalizer is not None:
        tokenizer.normalizer = normalizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    return tokenizer.get_vocab_size()


def make_layout(base: Path) -> Paths:
    base = Path(base)
    return Paths(repo=REPO, root=base / 'root', runs=base / 'runs', monitor_root=base / 'monitor')


def write_pool(paths: Paths, tokenizer_files: list) -> None:
    """A 1,000-entry pool index whose first entries are the given ``(path, algorithm, pretokenizer)`` files.

    Other entries carry their plan configs with a dummy hash (never loaded).
    """
    from tokenizers import Tokenizer
    plan = json.loads((REPO / 'plans/tokenizer-pool-1000-v1.json').read_text())
    configs, entries = [], []
    for i in range(1000):
        config = dict(plan['configs'][i])
        sha = '0' * 64
        if i < len(tokenizer_files):
            source, algorithm, pretokenizer = tokenizer_files[i]
            target = paths.pool_tokenizer_dir(i) / 'tokenizer.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            config.update(algorithm=algorithm, pretokenizer=pretokenizer,
                          vocab_size=Tokenizer.from_file(str(target)).get_vocab_size())
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
        configs.append(config)
        entries.append({'config': config, 'status': 'complete', 'vocab_size_actual': config['vocab_size'],
                        'tokenizer_sha256': sha})
    paths.pool_dir.mkdir(parents=True, exist_ok=True)
    (paths.pool_dir / 'plan.json').write_text(json.dumps({'plan_sha256': 'synthetic-test-plan', 'configs': configs}))
    (paths.pool_dir / 'full-results.json').write_text(json.dumps(
        {'plan_sha256': 'synthetic-test-plan', 'completed': 1000, 'tokenizers': entries}))


def install_model_tokenizer(paths: Paths, scratch_assets: Path) -> Path:
    target = paths.model_dir / 'tokenizer.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(scratch_assets) / 'full' / 'tokenizer.json', target)
    return target


def install_rebuilt(layout, panel: str, rows: list) -> None:
    from hyak.common.io import atomic_write_json
    atomic_write_json(layout.rebuilt(panel), rows)


def fake_checkpoint(directory: Path, *, with_manifest: bool = True, step: int = 20000,
                    lm: bytes = b'lm weights') -> Path:
    """Tiny stand-ins for lm.pt and hypernet/{config.json,model.safetensors} plus a training-style manifest."""
    from hyak.common.io import atomic_write_json, file_record
    directory = Path(directory)
    (directory / 'hypernet').mkdir(parents=True, exist_ok=True)
    (directory / 'lm.pt').write_bytes(lm)
    (directory / 'hypernet/config.json').write_text('{"hn_surface_maxlen": 10}')
    (directory / 'hypernet/model.safetensors').write_bytes(b'hypernet weights')
    if with_manifest:
        files = {name: file_record(directory / name) for name in ('lm.pt', 'hypernet/config.json',
                                                                  'hypernet/model.safetensors')}
        atomic_write_json(directory / 'manifest.json', {'step': step, 'kind': 'full_restart_checkpoint',
                                                        'files': files, 'trainer_sha256': 'x'})
    return directory


class FakeScorer:
    """Stands in for TorchScorer: deterministic non-negative losses; checks the feature arrays it is given."""

    def __init__(self, fallbacks: dict, gpu_name='Fake GPU'):
        self.fallbacks = fallbacks
        self.surface_maxlen = 10
        self.gpu_name = gpu_name
        self.control_calls = 0
        self.transfer_calls = []

    def controls(self, rows):
        self.control_calls += 1
        canonical = [[0.01 * (1 + (j + t) % 7) for t in range(32)] for j in range(len(rows))]
        no_prefix = [[0.02 * (1 + (j + t) % 5) for t in range(32)] for j in range(len(rows))]
        return canonical, no_prefix

    def transferred(self, rows, tokenizer, features):
        import numpy as np
        forms = np.load(Path(features) / 'forms.npy')
        if forms.shape[1] != self.surface_maxlen:
            raise ValueError('Surface feature lengths differ between models')
        self.transfer_calls.append(str(features))
        lengths = [len(tokenizer.encode(r['prefix'], add_special_tokens=False).ids) for r in rows]
        return [[0.001 * (n % 13) + 0.01 * (t % 3) for t in range(32)] for n in lengths], {'peak_gib': 0.0}


# --- CPU rehearsal of the GPU scorer (test only) -----------------------------------------------------------
def install_cuda_shim() -> None:
    """Make code that hard-codes ``'cuda'`` run on the CPU: device 'cuda' -> 'cpu', autocast('cuda') -> 'cpu'.

    Only for a separate test process (it patches torch globally).  It lets the real
    ``TorchScorer`` and ``natural_transfer.score_rows`` (bf16 autocast, 1e-5 loss check)
    run on a tiny model without a GPU.
    """
    import functools
    import torch

    def device(value):
        if isinstance(value, str) and value.startswith('cuda'):
            return 'cpu'
        if isinstance(value, torch.device) and value.type == 'cuda':
            return torch.device('cpu')
        return value

    def factory(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if 'device' in kwargs:
                kwargs['device'] = device(kwargs['device'])
            return fn(*args, **kwargs)
        return wrapper

    for name in ('tensor', 'arange', 'full', 'ones', 'zeros', 'empty', 'as_tensor'):
        setattr(torch, name, factory(getattr(torch, name)))
    tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to

    def to_tensor(self, *args, **kwargs):
        if 'device' in kwargs:
            kwargs['device'] = device(kwargs['device'])
        return tensor_to(self, *(device(a) for a in args), **kwargs)

    def to_module(self, *args, **kwargs):
        if 'device' in kwargs:
            kwargs['device'] = device(kwargs['device'])
        return module_to(self, *(device(a) for a in args), **kwargs)

    torch.Tensor.to = to_tensor
    torch.nn.Module.to = to_module
    real_autocast = torch.autocast

    class autocast(real_autocast):
        def __init__(self, device_type, *args, **kwargs):
            super().__init__('cpu' if device_type == 'cuda' else device_type, *args, **kwargs)

    torch.autocast = autocast
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda *a: 'CPU stand-in (test shim)'
    torch.cuda.reset_peak_memory_stats = lambda *a: None
    torch.cuda.max_memory_allocated = lambda *a: 0
    torch.cuda.empty_cache = lambda: None


def tiny_checkpoint(work: Path, tokenizer: Path):
    """A 2-layer, width-64 Llama with SmolLM2's tokenizer (tied tables, like SmolLM2) and a matching
    "trained" checkpoint: untied lm.pt with a perturbed backbone and an expanded random hypernetwork."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from hf_hypernet.identity_warmup import build_input_hypernet, initialize_rescalers
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    work = Path(work)
    model_dir, checkpoint = work / 'model', work / 'step-020000'
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=49152, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                         rms_norm_eps=1e-5, rope_theta=130000, tie_word_embeddings=True, bos_token_id=0,
                         eos_token_id=0)
    LlamaForCausalLM(config).save_pretrained(model_dir, safe_serialization=True)
    shutil.copyfile(tokenizer, model_dir / 'tokenizer.json')
    lm = LlamaForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    separate_input_embeddings_and_freeze_output(lm)
    with torch.no_grad():
        lm.model.layers[0].mlp.up_proj.weight.add_(0.01)
    (checkpoint / 'hypernet').mkdir(parents=True)
    torch.save(lm.state_dict(), checkpoint / 'lm.pt')
    hypernet = build_input_hypernet(49152, 64, layers=1)
    initialize_rescalers(hypernet, lm.get_input_embeddings().weight.detach())
    enable_byte_fallbacks(hypernet, AutoTokenizer.from_pretrained(model_dir, local_files_only=True))
    hypernet.save_pretrained(checkpoint / 'hypernet', safe_serialization=True)
    return model_dir, checkpoint


def rehearsal_main(argv=None) -> int:
    """``python -c 'from hyak.tests.test_eval_support import rehearsal_main; rehearsal_main()' <json args>``."""
    args = json.loads((argv or sys.argv)[1])
    install_cuda_shim()
    from hyak.eval import setup_imports
    setup_imports()
    from uniform_trajectory import load_candidate
    from hyak.common.io import read_json
    from hyak.eval.score import TorchScorer, check_features
    import numpy as np
    import torch
    from natural_transfer import score_rows
    model_dir, checkpoint = tiny_checkpoint(Path(args['work']), Path(args['tokenizer']))
    scorer = TorchScorer(model_dir, checkpoint)
    rows = read_json(args['panel'])[:args['rows']]
    # Property 1: the trained LM (lm.pt) is exactly what the scorer scores with (strict load happened).
    saved = torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True)
    state = scorer.model.state_dict()
    lm_differences = sorted(set(saved) ^ set(state))
    lm_differences += [name for name, value in saved.items()
                       if name in state and not torch.equal(state[name].cpu(), value)]
    canonical, no_prefix = scorer.controls(rows)
    out = {'gpu': scorer.gpu_name, 'fallbacks': scorer.fallbacks, 'canonical': canonical, 'no_prefix': no_prefix,
           'transferred': {}, 'reference': {}, 'lm_differences': lm_differences,
           # Property 2: the BOS-only control is not the canonical control.
           'controls_differ': canonical != no_prefix}
    for index, (features, tokenizer) in args['tokenizers'].items():
        check_features(Path(features), scorer.fallbacks, args['sha256'][index])
        native, _, _, _, _ = load_candidate(tokenizer, 'pool')
        out['transferred'][index] = scorer.transferred(rows, native, Path(features))[0]
        # Property 3: an independent recomputation with the original worker's lines
        # (scripts/modal_random_pool_50.py:192-200): deterministic hypernetwork over the full native vocabulary.
        forms = torch.from_numpy(np.load(Path(features) / 'forms.npy').astype(np.int64)).to('cuda')
        priors = torch.from_numpy(np.load(Path(features) / 'priors.npy')).to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated, _, _ = scorer.hypernet(forms, priors, scorer.source, None, deterministic=True)
        out['reference'][index] = score_rows(scorer.model, generated, scorer.source, rows, native)
    Path(args['out']).write_text(json.dumps(out))
    return 0
