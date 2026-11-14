"""Score one final checkpoint on a validated panel with the corrected protocol (GPU; claims; resumable).

    python -m hyak.eval.score --run uniform|rl --panel pool|natural
                              [--checkpoint DIR] [--indices ...] [--compute-manifest]
                              [--min-remaining-minutes 10] [--claim-timeout-minutes 60]
                              [--controls-wait-minutes 60] [--verify-only]
                              [--allow-non-final-checkpoint] [--config FILE] [--code-dir DIR]

Port of the worker ``evaluate`` of scripts/modal_random_pool_50.py:101-213 (the
function every corrected-protocol run used: the complete-pool reward table via
scripts/modal_frozen_policy_cache.py:40-41, and the natural-panel model
comparison, scripts/modal_natural_model_comparison.py:76-181, whose body is the
same).  Scoring itself is ``natural_transfer.score_rows`` / ``validate_scores``
(imported): BOS + prefix + exactly 32 canonical continuation targets, at most 128
positions, bf16 autocast forward, float32 cross-entropy, batches of 16, the 1e-5
agreement check against the Hugging Face loss.  The hypernetwork runs
``deterministic=True`` over the complete native vocabulary on the pinned forward
(no chunk patch; inference mode).

What changed versus Modal (no score depends on it): storage paths
(``$ANYTOK_ROOT/eval``), no volume reload/commit, the plan holds this run's
checkpoint manifest (read from the training wrapper's ``<checkpoint>/manifest.json``),
model input hashes come from plans/zett-modal-20k-inputs.json (the original
``spec['input_hashes']`` held the same values), the pool index is
``input/pool/full-results.json``, and work is shared by claims instead of two
fixed 500-tokenizer shards.  The canonical and BOS-only controls are computed once
per checkpoint and panel (claim ``controls``) before any tokenizer is scored; other
jobs wait for them.  Results: ``$ANYTOK_ROOT/eval/scores/<run>/<panel>/``.

A job keeps working until every index of its work list has a result: after a pass it
waits (``--claim-timeout-minutes`` plus one poll without any new result at most) for
indices that other jobs hold and takes over claims whose owner is gone (squeue) or
that are older than the claim timeout.  The checkpoint is identified by the SHA-256 of
its training manifest, not by its path, so an archived copy passed with ``--checkpoint``
continues an existing scores directory (the path is recorded in every result file).
The plan also records the SHA-256 of ``hyak/eval/__init__.py`` and ``score.py``; the
panel plan those of the pinned hypernetwork files and the imported scripts.  Any
difference refuses the job ("Comparison plan changed" naming the files).

Exit codes: 0 every index of the work list is scored, or the job stopped on a signal
or the time margin (resubmit to continue); 75 indices remain in claims of other live
jobs after the bounded wait (not complete yet; the stage is not complete); 2 usage,
configuration or input error (the message names the remedy); 1 crash (traceback).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

from hyak.common.config import ConfigError
from hyak.common.io import file_record, read_json, sha256_file, utc_now
from hyak.common.paths import STAGE_EVAL_RL, STAGE_EVAL_UNIFORM, RUNS, pool_index_name, validate_run
from hyak.eval import (CHECKPOINT_FILES, EXIT_LATER, EXIT_OK, EXIT_USAGE, FINAL_CHECKPOINT, MODEL_NAMES, PANELS,
                       Claims, EvalError, EvalLayout, check_plan_code, check_tokenizer_file, code_changes, crashed,
                       create_once, digest, load_pool_spec, log, parse_indices, remaining_seconds, setup_imports,
                       verify_model_inputs)

EVAL_STAGES = {'uniform': STAGE_EVAL_UNIFORM, 'rl': STAGE_EVAL_RL}


# --- pure preparation (CPU; tested) -----------------------------------------------------------------
def load_panel(layout: EvalLayout, panel: str) -> tuple:
    """``(panel_plan, rows)``; ``digest(rows) == plan['panel_sha256']`` and every row jointly encodable.

    scripts/modal_random_pool_50.py:120-124.
    """
    plan_path, panel_path = layout.plan(panel), layout.panel(panel)
    if not plan_path.is_file() or not panel_path.is_file():
        raise EvalError(f'{plan_path} or {panel_path} is missing; run hyak.eval.rebuild_panel and '
                        f'hyak.eval.validate_prefixes --panel {panel} first')
    plan = read_json(plan_path)
    rows = read_json(panel_path)
    if digest(rows) != plan['panel_sha256'] or not all(r['canonical_joint_encoding_matches'] for r in rows):
        raise EvalError('Corrected panel changed')
    if plan.get('models'):
        raise EvalError(f'{plan_path} is not a panel plan (it names models)')
    return plan, rows


def checkpoint_manifest(checkpoint: Path, *, compute_missing: bool = False) -> dict:
    """The training-side manifest of a checkpoint, with the three inference files verified.

    The original plan embedded the checkpoint's manifest and the worker compared
    size and SHA-256 of ``lm.pt`` and ``hypernet/{config.json,model.safetensors}``
    with it (scripts/modal_random_pool_50.py:125-130).  Here the manifest is the
    ``manifest.json`` the training wrapper writes into each checkpoint (the Modal
    launchers' save hook did the same).  ``compute_missing`` hashes the files when
    that manifest is absent (recorded as ``source: computed``).
    """
    checkpoint = Path(checkpoint)
    path = checkpoint / 'manifest.json'
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file():
            raise EvalError(f'{checkpoint / name} is missing; is {checkpoint} a finished checkpoint?')
    if path.is_file():
        manifest = read_json(path)
        files = manifest.get('files', {})
        missing = [name for name in CHECKPOINT_FILES if name not in files]
        if missing:
            raise EvalError(f'{path} lists no entry for {missing}')
        manifest = {**manifest, 'source': 'training manifest.json'}
    elif compute_missing:
        manifest = {'source': 'computed by hyak.eval.score (no training manifest.json)',
                    'files': {name: file_record(checkpoint / name) for name in CHECKPOINT_FILES}}
        return manifest
    else:
        raise EvalError(f'{path} is missing; the training wrapper writes it for every checkpoint. If you are sure '
                        'the checkpoint is complete, pass --compute-manifest to hash its files instead')
    for name in CHECKPOINT_FILES:
        expected = manifest['files'][name]
        size = (checkpoint / name).stat().st_size
        if size != expected['bytes'] or sha256_file(checkpoint / name) != expected['sha256']:
            raise EvalError(f'Checkpoint changed: {checkpoint / name} differs from {path}')
    return manifest


#: Files of this package whose code produces the scores (their SHA-256 is part of the scoring plan).
SCORING_CODE = ('__init__.py', 'score.py')


def scoring_code_hashes() -> dict:
    """``{'hyak/eval/<name>': sha256}`` of :data:`SCORING_CODE`."""
    here = Path(__file__).resolve().parent
    return {f'hyak/eval/{name}': sha256_file(here / name) for name in SCORING_CODE}


def scoring_plan(panel_plan: dict, run: str, checkpoint: Path, manifest: dict) -> dict:
    """Panel plan plus this run's model entry (the shape of plan['models'] in plans/frozen-policy-search-v1.json).

    The checkpoint is identified by its manifest (``manifest_sha256`` = digest of the training
    manifest), never by its location: the original plans named it without a path, and an archived
    copy must continue the same comparison.  ``checkpoint`` is only logged and written into the
    result files.  ``scoring_code_sha256`` holds the hashes of :data:`SCORING_CODE`.
    """
    return {**panel_plan, 'models': [{'name': MODEL_NAMES[validate_run(run)], 'run': run,
                                      'manifest_sha256': digest(manifest), 'manifest': manifest}],
            'scoring_code_sha256': scoring_code_hashes()}


def plan_differences(existing: dict, plan: dict) -> list:
    """Human-readable list of what differs between two scoring plans."""
    out = []
    old_models, new_models = existing.get('models') or [{}], plan.get('models') or [{}]
    if old_models[0].get('manifest_sha256') != new_models[0].get('manifest_sha256'):
        out.append('checkpoint (training manifest sha256 '
                   f'{old_models[0].get("manifest_sha256")} != {new_models[0].get("manifest_sha256")})')
    elif old_models != new_models:
        out.append('model entry')
    changed = code_changes(existing.get('scoring_code_sha256'), plan.get('scoring_code_sha256') or {})
    if changed:
        out.append('scoring code: ' + ', '.join(changed))
    rest = sorted(k for k in set(existing) | set(plan) if k not in ('models', 'scoring_code_sha256')
                  and existing.get(k) != plan.get(k))
    if rest:
        out.append('panel plan keys: ' + ', '.join(rest))
    return out


def ensure_plan(out_dir: Path, plan: dict, *, create: bool = True) -> dict:
    """Create ``plan.json`` once; later jobs must find exactly the same plan ("Comparison plan changed").

    ``create=False`` (``--verify-only``) only compares with an existing plan and writes nothing, so a
    verification of another checkpoint cannot bind the scores directory to it.
    """
    path = Path(out_dir) / 'plan.json'
    if create and create_once(path, plan):
        log(f'created {path} (plan sha256 {digest(plan)})')
    if not create and not path.is_file():
        return plan
    existing = read_json(path)
    if existing != plan:
        raise EvalError(f'Comparison plan changed: {path} was made for another panel, pool, checkpoint or scoring '
                        f'code (existing sha256 {digest(existing)}, now {digest(plan)}; differs in: '
                        f'{"; ".join(plan_differences(existing, plan)) or "?"}). Score the checkpoint (any copy with '
                        'the same manifest.json) with the code it was started with (git checkout the listed files), '
                        'or move the scores directory away to start a new comparison')
    return existing


def check_final_step(checkpoint: Path, manifest: dict) -> None:
    """The corrected evaluation scores the step-20,000 checkpoints; anything else needs an explicit flag."""
    step = manifest.get('step')
    final = int(FINAL_CHECKPOINT.split('-')[1])
    if (step is not None and int(step) != final) or (step is None and Path(checkpoint).name != FINAL_CHECKPOINT):
        raise EvalError(f'{checkpoint} is not a final checkpoint (manifest step {step}, expected {final}); '
                        'pass --allow-non-final-checkpoint to score it anyway (for tests and diagnostics only)')


def check_native_prefixes(rows: list, tokenizer, forbidden: set, index: int) -> None:
    """Every stored native length still holds (scripts/modal_random_pool_50.py:180-184)."""
    for row in rows:
        ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
        if (len(ids) != row['native_prefix_lengths'][str(index)] or forbidden.intersection(ids) or 0 in ids
                or tokenizer.decode(ids, skip_special_tokens=False) != row['prefix']):
            raise EvalError(f'Native prefix changed: tokenizer {index}, row {row["id"]}')


def check_features(features: Path, fallbacks: dict, tokenizer_sha256: str) -> dict:
    """Manifest mapping and array hashes (scripts/modal_random_pool_50.py:185-191).  Returns the manifest."""
    features = Path(features)
    if not (features / 'manifest.json').is_file():
        raise EvalError(f'{features}/manifest.json is missing; run python -m hyak.eval.prepare_features')
    manifest = read_json(features / 'manifest.json')
    if manifest['fallbacks'] != fallbacks or manifest['tokenizer_sha256'] != tokenizer_sha256:
        raise EvalError(f'Hypernetwork feature mapping changed: {features}')
    for name in ['forms', 'priors']:
        if sha256_file(features / (name + '.npy')) != manifest[name + '_sha256']:
            raise EvalError(f'Feature arrays changed: {features}/{name}.npy')
    return manifest


# --- the GPU part (mirrors scripts/modal_random_pool_50.py line for line) ------------------------------
class TorchScorer:
    """Model loading (lines 141-154), controls (162-163) and per-tokenizer scoring (192-206)."""

    def __init__(self, model_dir: Path, checkpoint: Path):
        import torch
        if not torch.cuda.is_available():
            raise EvalError('CUDA is not available; score.py must run in a GPU job (the scorer hard-codes cuda, '
                            'as scripts/natural_transfer.py:236-237 does)')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from hf_hypernet.modeling_hypernet import ZettHypernet
        from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
        from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
        torch.set_num_threads(1)                                                    # line 111
        model_dir, checkpoint = Path(model_dir), Path(checkpoint)
        canonical = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)  # line 141
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
            dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')                 # lines 142-143
        separate_input_embeddings_and_freeze_output(model)                          # line 144
        original = model.get_input_embeddings().weight.detach().clone()             # line 145
        model.load_state_dict(torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True))  # line 146
        model.eval().requires_grad_(False)                                          # line 147
        source = model.get_input_embeddings().weight.detach()                       # line 148
        if not torch.equal(source, original) or not torch.equal(model.get_output_embeddings().weight, original):
            raise EvalError('Canonical tables changed')                             # lines 149-150
        del original
        hypernet = ZettHypernet.from_pretrained(checkpoint / 'hypernet', local_files_only=True).to('cuda').eval()
        fallbacks = enable_byte_fallbacks(hypernet, canonical)                      # line 153
        hypernet.requires_grad_(False)                                              # line 154
        self.torch, self.model, self.source, self.hypernet = torch, model, source, hypernet
        self.fallbacks = fallbacks
        self.surface_maxlen = hypernet.config.hn_surface_maxlen
        self.gpu_name = torch.cuda.get_device_name()

    def controls(self, rows):
        """Canonical and BOS-only controls (lines 162-163)."""
        from natural_transfer import score_rows
        return (score_rows(self.model, self.source, self.source, rows),
                score_rows(self.model, self.source, self.source, rows, empty_prefix=True))

    def transferred(self, rows, tokenizer, features: Path):
        """Hypernetwork embeddings for the full native vocabulary, then the transferred scores (lines 192-206)."""
        import numpy as np
        from natural_transfer import score_rows, validate_scores
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        forms = torch.from_numpy(np.load(features / 'forms.npy').astype(np.int64)).to('cuda')
        priors = torch.from_numpy(np.load(features / 'priors.npy')).to('cuda')
        if forms.shape[1] != self.hypernet.config.hn_surface_maxlen:
            raise EvalError('Surface feature lengths differ between models')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated, output, _ = self.hypernet(forms, priors, self.source, None, deterministic=True)
        if output is not None or not torch.isfinite(generated).all():
            raise EvalError('Invalid full-vocabulary embeddings')
        transferred = score_rows(self.model, generated, self.source, rows, tokenizer)
        validate_scores(rows, transferred)
        del forms, priors, generated
        torch.cuda.empty_cache()
        return transferred, {'peak_gib': torch.cuda.max_memory_allocated() / 2 ** 30}


# --- orchestration (claims, resume, status; tested with a fake scorer) ----------------------------------
class Stop:
    """SIGTERM/SIGINT flag: finish the current tokenizer, claim nothing new, exit 0."""

    def __init__(self):
        self.requested = False
        self.reason = None

    def __call__(self, signum, frame):
        self.requested = True
        self.reason = signal.Signals(signum).name


class ScoreJob:
    """One scoring process for one run and panel.  ``scorer_factory()`` returns a TorchScorer-like object."""

    def __init__(self, layout: EvalLayout, run: str, panel: str, spec: dict, pool_dir: Path, checkpoint: Path, *,
                 indices=None, status=None, claim_timeout: float = 3600.0, controls_wait: float = 3600.0,
                 min_remaining: float = 600.0, poll_seconds: float = 15.0, stop: Stop | None = None,
                 compute_manifest: bool = False, sleep=time.sleep, verify_only: bool = False,
                 allow_non_final: bool = False):
        self.layout, self.run_name, self.panel = layout, validate_run(run), panel
        self.spec, self.pool_dir, self.checkpoint = spec, Path(pool_dir), Path(checkpoint)
        self.out = layout.scores_dir(run, panel)
        self.status = status
        self.claims = Claims(layout.claims(f'scores-{run}-{panel}'), claim_timeout)
        self.claim_timeout = float(claim_timeout)
        self.controls_wait, self.min_remaining, self.poll = controls_wait, min_remaining, poll_seconds
        self.stop = stop or Stop()
        self.sleep = sleep
        panel_plan, self.rows = load_panel(layout, panel)
        if panel_plan['hyak']['pool_index_sha256'] != spec['index_sha256']:
            raise EvalError(f'The {panel} panel was validated for another pool index '
                            f'({panel_plan["hyak"]["pool_index_sha256"]} != {spec["index_sha256"]}); rerun '
                            'hyak.eval.validate_prefixes')
        check_plan_code(panel_plan, layout.plan(panel))
        manifest = checkpoint_manifest(self.checkpoint, compute_missing=compute_manifest)
        if not allow_non_final:
            check_final_step(self.checkpoint, manifest)
        self.plan = ensure_plan(self.out, scoring_plan(panel_plan, run, self.checkpoint, manifest),
                                create=not verify_only)
        self.plan_sha256 = digest(self.plan)
        self.model_name = self.plan['models'][0]['name']
        self.work = parse_indices(indices, self.plan['indices'])
        self.durations = []
        self.begin = time.monotonic()

    # paths
    def result_path(self, index: int) -> Path:
        return self.out / f'{pool_index_name(index)}.json'

    def done(self, indices=None) -> list:
        return [i for i in (self.work if indices is None else indices) if self.result_path(i).is_file()]

    def _emit(self, stage, state=None, **extra):
        """Dashboard status: progress over all plan indices (every job writes the same global count)."""
        if self.status is None:
            return
        total = len(self.plan['indices'])
        done = len(self.done(self.plan['indices']))
        # Other JOBS sharing the work (this job's own 'starting' row, written by the batch script, is not one).
        others = len({str(row.get('job_id', row['id'])) for row in self.status.other_rows()} - {str(self.status.job)})
        rate = sum(self.durations) / len(self.durations) if self.durations else None
        eta = rate * (total - done) / (1 + others) if rate is not None else None
        self.status.update(done, stage=stage, state=state, total=total, eta_seconds=eta,
                           elapsed_seconds=time.monotonic() - self.begin, panel=self.panel, model=self.model_name,
                           plan_sha256=self.plan_sha256, workers=[] if state == 'complete' else None, **extra)

    # controls
    def _read_controls(self, path):
        from natural_transfer import validate_scores
        controls = read_json(path)
        if controls['plan_sha256'] != self.plan_sha256:
            raise EvalError('Canonical controls belong to another comparison')  # lines 159-160
        for values in [controls['canonical'], controls['no_prefix']]:
            validate_scores(self.rows, values)                                  # lines 165-166
        return controls

    def controls(self, scorer):
        """The controls of this plan (computed here, or written by another job); None after a stop signal
        that arrived while another job was computing them (a graceful stop, like any other: exit 0)."""
        path = self.out / 'canonical.json'
        waited = 0.0
        while True:
            if path.is_file():
                return self._read_controls(path)
            if self.claims.try_acquire('controls', task='controls'):
                try:
                    if path.is_file():
                        return self._read_controls(path)
                    self._emit('scoring model-specific canonical controls')    # line 155
                    begin = time.monotonic()
                    canonical, no_prefix = scorer.controls(self.rows)
                    controls = dict(plan_sha256=self.plan_sha256, canonical=canonical, no_prefix=no_prefix,
                                    gpu=scorer.gpu_name, elapsed_seconds=time.monotonic() - begin,
                                    created_at=utc_now(), job_id=self.claims.job, host=self.claims.host,
                                    checkpoint_path=str(self.checkpoint))
                    from natural_transfer import validate_scores
                    for values in [canonical, no_prefix]:
                        validate_scores(self.rows, values)
                    if not create_once(path, controls, indent=None):
                        raise EvalError(f'{path} appeared while this job held the controls claim')
                    log(f'controls written ({controls["elapsed_seconds"]:.1f} s on {scorer.gpu_name})')
                    return self._read_controls(path)
                finally:
                    self.claims.release_after_work('controls')
            if waited >= self.controls_wait:
                raise EvalError(f'Waited {waited:.0f} s for {path} (claimed by {self.claims.owner("controls")}); '
                                'resubmit when that job has finished or failed')
            if self.stop.requested:
                log(f'stopped ({self.stop.reason}) while waiting for {path} from another job')
                return None
            self._emit('waiting for canonical controls from another job')
            self.sleep(self.poll)
            waited += self.poll

    # one tokenizer
    def score_one(self, scorer, index: int) -> dict:
        from uniform_trajectory import load_candidate
        begin = time.monotonic()                                                # line 175
        token_path = check_tokenizer_file(self.pool_dir, self.spec, index)     # lines 176-178
        tokenizer, _, _, _, forbidden = load_candidate(token_path, 'pool')     # line 179
        check_native_prefixes(self.rows, tokenizer, forbidden, index)          # lines 180-184
        features = self.layout.feature_dir(index)
        check_features(features, scorer.fallbacks, self.spec['tokenizers'][index]['sha256'])  # lines 185-191
        transferred, extra = scorer.transferred(self.rows, tokenizer, features)  # lines 192-201
        entry = self.spec['tokenizers'][index]
        return dict(model=self.model_name, index=index, config=entry['config'], transferred=transferred,
                    plan_sha256=self.plan_sha256, elapsed_seconds=time.monotonic() - begin,   # lines 202-203
                    tokenizer_sha256=entry['sha256'], gpu=scorer.gpu_name, created_at=utc_now(),
                    job_id=self.claims.job, host=self.claims.host, checkpoint_path=str(self.checkpoint), **extra)

    def check_existing(self, index: int) -> dict:
        """An existing result must belong to this plan (scripts/modal_random_pool_50.py:169-174)."""
        result = read_json(self.result_path(index))
        if result.get('plan_sha256') != self.plan_sha256 or result.get('index') != index:
            raise EvalError('Existing scores belong to another comparison: '
                            f'{self.result_path(index)}')                         # lines 171-173
        return result

    def run(self, scorer_factory) -> dict:
        """Score every index of this job's work list, sharing the work with other jobs through claims.

        After each pass over the unscored indices the job waits (``poll_seconds``) for indices that
        other jobs hold and tries them again (a claim whose owner is gone, or older than the claim
        timeout, is taken over).  It ends when every index has a result, on a stop signal or the
        time margin (``counts['stopped']``), or when ``claim_timeout + poll_seconds`` passed without
        any new result (``counts['incomplete']`` lists the indices still held).  Returns counts.
        """
        counts = {'scored': 0, 'existing': 0, 'claimed_elsewhere': 0, 'stopped': None, 'incomplete': [],
                  'waited_seconds': 0.0}
        if len(self.done()) == len(self.work):
            log(f'{self.run_name}/{self.panel}: all {len(self.work)} tokenizers already scored')
            return self.finish(counts)
        self._emit('loading model and hypernetwork')
        scorer = scorer_factory()
        log(f'{self.run_name}/{self.panel}: model loaded on {scorer.gpu_name}; {len(self.done())}/{len(self.work)} '
            f'done; checkpoint {self.checkpoint}')
        if self.controls(scorer) is None:
            counts['stopped'] = self.stop.reason
            return self.finish(counts)
        todo = list(self.work)
        last_done, idle = len(self.done()), 0.0
        while True:
            held = self._pass(scorer, todo, counts)
            if counts['stopped']:
                break
            todo = [i for i in held if not self.result_path(i).is_file()]
            counts['claimed_elsewhere'] = len(todo)
            if not todo:
                break
            done = len(self.work) - len(todo)
            if done > last_done:
                last_done, idle = done, 0.0
            elif idle >= self.claim_timeout + self.poll:
                counts['incomplete'] = todo
                log(f'{self.run_name}/{self.panel}: {len(todo)} tokenizers are still held by other jobs after '
                    f'{idle:.0f} s without a new result (first {todo[:10]}); resubmit when those jobs have ended')
                break
            if self.stop.requested:
                counts['stopped'] = self.stop.reason
                break
            if remaining_seconds() < self.min_remaining:
                counts['stopped'] = 'job time limit approaching'
                break
            self._emit(f'waiting for {len(todo)} tokenizers claimed by other jobs')
            self.sleep(self.poll)
            idle += self.poll
            counts['waited_seconds'] += self.poll
        return self.finish(counts)

    def _pass(self, scorer, indices, counts) -> list:
        """One pass over ``indices``; returns those another job held (or that were left after a stop)."""
        held = []
        for position, index in enumerate(indices):
            if self.stop.requested:
                counts['stopped'] = self.stop.reason
                return held + list(indices[position:])
            if remaining_seconds() < self.min_remaining:
                counts['stopped'] = 'job time limit approaching'
                return held + list(indices[position:])
            name = pool_index_name(index)
            if self.result_path(index).is_file():
                self.check_existing(index)
                counts['existing'] += 1
                continue
            if not self.claims.try_acquire(name, task=f'score {self.run_name}/{self.panel}'):
                held.append(index)
                continue
            try:
                if self.result_path(index).is_file():
                    self.check_existing(index)
                    counts['existing'] += 1
                    continue
                result = self.score_one(scorer, index)
                if not create_once(self.result_path(index), result, indent=None):
                    self.check_existing(index)      # another job finished it after a takeover; keep the first
                    log(f'{name}: result already written by another job; kept the existing one')
                else:
                    counts['scored'] += 1
                    self.durations.append(result['elapsed_seconds'])
                    log(f'{name}: scored in {result["elapsed_seconds"]:.1f} s'
                        + (f' (peak {result["peak_gib"]:.1f} GiB)' if 'peak_gib' in result else ''))
            finally:
                self.claims.release_after_work(name)
            self._emit('scoring tokenizers', latest_index=index)
        return held

    def pending_panels(self) -> list:
        """Other panels of this run without a ``complete.json`` for the same checkpoint (the shared status)."""
        pending = []
        for panel in PANELS:
            if panel == self.panel:
                continue
            folder = self.layout.scores_dir(self.run_name, panel)
            try:
                marker, plan = read_json(folder / 'complete.json'), read_json(folder / 'plan.json')
            except (OSError, ValueError):
                pending.append(panel)
                continue
            if marker.get('plan_sha256') != digest(plan) or (plan.get('models') or [None])[0] != self.plan['models'][0]:
                pending.append(panel)
        return pending

    def finish(self, counts: dict) -> dict:
        """Write ``complete.json`` once every plan index has a valid result; set the status accordingly."""
        counts['takeovers'] = self.claims.takeovers
        counts['lost_claims'] = self.claims.lost
        all_indices = list(self.plan['indices'])
        present = self.done(all_indices)
        counts['done'] = len(present)
        counts['total'] = len(all_indices)
        if len(present) == len(all_indices):
            from natural_transfer import validate_scores
            gpus = set()
            for index in all_indices:
                result = self.check_existing(index)
                validate_scores(self.rows, result['transferred'])
                gpus.add(result.get('gpu') or '?')
            gpus = sorted(gpus)
            marker = dict(plan_sha256=self.plan_sha256, model=self.model_name, panel=self.panel,
                          panel_sha256=self.plan['panel_sha256'], tokenizers=len(all_indices), gpus=gpus,
                          checkpoint_path=str(self.checkpoint), completed_at=utc_now())
            create_once(self.out / 'complete.json', marker)
            counts['complete'] = True
            pending = self.pending_panels()
            if pending:
                # The eval-<run> stage covers both panels (D11): it is complete only when every panel is.
                self._emit(f'{self.panel} panel complete; {" and ".join(pending)} panel not complete yet')
            else:
                self._emit('complete', state='complete')
            log(f'{self.run_name}/{self.panel}: complete ({len(all_indices)} tokenizers)'
                + (f'; still to score: {" and ".join(pending)} panel' if pending else ''))
        else:
            counts['complete'] = False
            if counts.get('stopped'):
                stage = f'stopped: {counts["stopped"]}'
            elif counts.get('incomplete'):
                stage = (f'not complete: {len(counts["incomplete"])} tokenizers held by other jobs; '
                         'resubmit when they have ended')
            else:
                stage = 'scoring tokenizers'
            self._emit(stage)
            log(f'{self.run_name}/{self.panel}: {len(present)}/{len(all_indices)} tokenizers scored so far')
        return counts


def default_checkpoint(paths, run: str) -> Path:
    return paths.run_dir(run) / FINAL_CHECKPOINT


def main(argv=None, *, pool_reference_plan=True, model_hashes=None) -> int:
    """Exit codes: see the module docstring.  ``pool_reference_plan`` and ``model_hashes`` are test parameters
    only (a miniature world's TEST pool plan and tiny model; see load_pool_spec and verify_model_inputs);
    production uses the repository pins."""
    parser = argparse.ArgumentParser(prog='python -m hyak.eval.score', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', choices=RUNS, required=True)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--checkpoint', type=Path, help=f'default: $ANYTOK_RUNS/<run>/{FINAL_CHECKPOINT}')
    parser.add_argument('--indices', nargs='+', help='subset of the panel plan indices for this job (pilots)')
    parser.add_argument('--compute-manifest', action='store_true',
                        help='hash the checkpoint files when the training manifest.json is missing')
    parser.add_argument('--min-remaining-minutes', type=float, default=10.0)
    parser.add_argument('--claim-timeout-minutes', type=float, default=60.0)
    parser.add_argument('--controls-wait-minutes', type=float, default=60.0)
    parser.add_argument('--verify-only', action='store_true',
                        help='check panel, plan, checkpoint, model inputs, pool and features; load no model, '
                             'write nothing')
    parser.add_argument('--allow-non-final-checkpoint', action='store_true',
                        help=f'score a checkpoint whose manifest step is not {int(FINAL_CHECKPOINT.split("-")[1])} '
                             '(tests and diagnostics; the comparison uses the final checkpoints)')
    parser.add_argument('--no-status', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--code-dir', type=Path, help='pinned code tree (default: <repo>/runtime/anytok_pool_20k)')
    args = parser.parse_args(argv)
    stop = Stop()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = None
    try:
        setup_imports(args.code_dir)
        from hyak.common.config import load_config
        from hyak.common.paths import Paths
        from hyak.common.status import StatusWriter
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
        layout = EvalLayout(paths)
        # Absolute, normalised, symlinks kept; only recorded (the plan identifies the checkpoint by its manifest).
        checkpoint = Path(os.path.abspath(args.checkpoint)) if args.checkpoint else default_checkpoint(paths, args.run)
        if not args.no_status:
            status = StatusWriter(EVAL_STAGES[args.run], None, 'tokenizers', cfg=cfg)
        verify_model_inputs(paths.model_dir, table=model_hashes)                # lines 131-133
        spec = load_pool_spec(paths.pool_dir, reference_plan=pool_reference_plan)
        job = ScoreJob(layout, args.run, args.panel, spec, paths.pool_dir, checkpoint, indices=args.indices,
                       status=status, claim_timeout=args.claim_timeout_minutes * 60,
                       controls_wait=args.controls_wait_minutes * 60, min_remaining=args.min_remaining_minutes * 60,
                       stop=stop, compute_manifest=args.compute_manifest, verify_only=args.verify_only,
                       allow_non_final=args.allow_non_final_checkpoint)
        if args.verify_only:
            from hyak.eval.prepare_features import expected_fallbacks, feature_problem
            from hyak.eval import load_canonical
            canonical, _ = load_canonical(paths.model_dir / 'tokenizer.json')
            fallbacks = expected_fallbacks(canonical)
            bad = [i for i in job.work if feature_problem(layout.feature_dir(i), spec['tokenizers'][i], fallbacks,
                                                          deep=True) is not None]
            for index in job.work:
                check_tokenizer_file(paths.pool_dir, spec, index)
            if bad:
                raise EvalError(f'{len(bad)} feature directories are missing or invalid (first {bad[:10]}); '
                                'run python -m hyak.eval.prepare_features')
            log(f'{args.run}/{args.panel}: verified plan {job.plan_sha256}, checkpoint {checkpoint}, '
                f'{len(job.work)} tokenizers and their features')
            return EXIT_OK
        counts = job.run(lambda: TorchScorer(paths.model_dir, checkpoint))
        log(f'{args.run}/{args.panel}: {counts}')
        if status is not None:
            status.event({'event': 'scoring_job_end', 'panel': args.panel, 'checkpoint_path': str(checkpoint),
                          **{k: v for k, v in counts.items() if k not in ('takeovers', 'lost_claims', 'incomplete')},
                          'takeovers': len(counts['takeovers']), 'lost_claims': len(counts['lost_claims']),
                          'incomplete': len(counts['incomplete'])})
        return exit_code(job, counts)
    except (EvalError, ConfigError, OSError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        if status is not None and status.last is not None:
            status.update(status.last[status.completed_field], state='failed', error=str(exc)[:2000])
        return EXIT_USAGE
    except Exception:
        return crashed(status)


def exit_code(job: ScoreJob, counts: dict) -> int:
    """0 when the work list is scored or the job stopped (signal or time margin; resubmit to continue);
    75 (:data:`hyak.eval.EXIT_LATER`) when indices remain in claims of other live jobs after the bounded wait."""
    if counts.get('stopped') or len(job.done()) == len(job.work):
        return EXIT_OK
    print(f'NOT COMPLETE: {len(job.work) - len(job.done())} of {len(job.work)} tokenizers of {job.run_name}/'
          f'{job.panel} are still held by other jobs (first {counts.get("incomplete", [])[:10]}); resubmit when '
          'those jobs have ended', file=sys.stderr)
    return EXIT_LATER


if __name__ == '__main__':
    sys.exit(main())
