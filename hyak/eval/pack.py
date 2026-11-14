"""Pack per-tokenizer scores into the original table formats and summarise them.

    python -m hyak.eval.pack --panel pool    [--runs uniform rl] [--out-dir DIR] [--partial] [--config FILE]
    python -m hyak.eval.pack --panel natural [--runs uniform rl] [--out-dir DIR] [--partial] [--config FILE]

Inputs: ``$ANYTOK_ROOT/eval/scores/<run>/<panel>/`` (written by hyak.eval.score)
and ``$ANYTOK_ROOT/eval/panels/<panel>/preparation.json`` (hyak.eval.validate_prefixes).
Outputs (default ``$ANYTOK_ROOT/eval/reports/<panel>/``; never the repository's ``reports/``):

pool panel, per run
    ``<run>/reward_table.json.gz``  ``{plan, rows, canonical_nats, transferred_nats, configs,
    tokenizer_hashes}`` exactly as scripts/frozen_policy_search.py:78-80 packs it and
    serialised as at :230, so ``python scripts/frozen_policy_search.py --cache <file>``
    can load it.  ``tokenizer_hashes`` are the hashes of the regenerated tokenizers
    that were scored (the original listed the original pool's hashes).
    ``<run>/summary.json``  per split: nine-bucket balanced canonical / BOS-only / per-tokenizer
    BPB, gaps and ranks (``frozen_policy_search.balanced_scores`` / ``reference``, imported).
    ``<run>/rankings.csv``
natural panel
    ``results.json.gz``  ``{plan, metadata, models, tokenizers, score_definition}`` like
    scripts/analyze_natural_model_comparison.py:54-56, models keyed ``uniform20k-hyak`` /
    ``rl20k-straight-entropy0-hyak`` (each also carries ``no_prefix_nats``).
    ``summary.json``  five-domain balanced BPB (``analyze_natural_transfer.bootstrap_balanced``,
    imported) per model, split, regime (all / frozen context subset) and cohort, plus
    RL-minus-uniform paired effects (formulas of analyze_natural_model_comparison.py:80-135).
both
    ``comparison.json`` / ``comparison.csv`` (pool) uniform versus RL paired by tokenizer, and
    the original numbers recomputed from the stored bundles (``reference_original``), labelled as a
    reference from a different run, never a reproduction target: the stored ``rl20k`` used entropy 0.01,
    the raw reward, the original cuts and the warmup initialization (D8).

Every summary (``pool/<run>/summary.json``, ``natural/summary.json``) and ``pool/comparison.json``
carries a ``panel`` block: panel hash, stored hash, ``equals_stored``, ``trimmed_rows``,
``trimmed_row_ids``, ``trimmed_chars`` and ``accepted_valid_context_trims``.  The natural panel
must cover the 38 indices and 2,032 rows of plans/natural-model-comparison-v1.json unless
``--partial`` (then the summary says ``partial: true`` and lists ``missing_indices``).

Exit codes: 0 every requested run complete and packed; 75 at least one requested run is
not complete for this panel (possibly the one just scored; complete runs were packed);
2 usage or input error (e.g. the panel is not validated); 1 crash (traceback).

The analysis tools that write into the repository (``frozen_policy_search.py``
main, ``analyze_natural_model_comparison.py`` main) are never called from here.
Running ``scripts/frozen_policy_search.py --cache <file>`` itself rewrites
``reports/frozen-policy-search-20k/`` and ``plots/`` in the checkout it runs in.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import sys

from hyak.common.config import ConfigError
from hyak.common.io import atomic_write_bytes, atomic_write_json, atomic_write_text, read_json, utc_now
from hyak.common.paths import RUNS, pool_index_name
from hyak.eval import (EXIT_LATER, EXIT_OK, EXIT_USAGE, MODEL_NAMES, ORIGINAL_RL_REFERENCE_NOTE, PANEL_NATURAL,
                       PANEL_POOL, PANEL_SPECS, PANELS,
                       POSITION_CAP, SUFFIX_TOKENS, EvalError, EvalLayout, check_plan_code, crashed, digest, log,
                       read_bundle, repo_root, setup_imports)

SCORE_DEFINITION = ('Each number is the float64 sum of 32 saved per-token natural-log losses '
                    '(natural_transfer.score_rows). Full per-token scores remain in '
                    '$ANYTOK_ROOT/eval/scores/<run>/<panel>/NNNN.json.')
#: Original complete-pool reference (reports/frozen-policy-search-20k; plans/frozen-policy-search-v1.md:77-80).
SPLITS = ('screen', 'confirm')


# --- loading ----------------------------------------------------------------------------------------
def load_prepared(layout: EvalLayout, panel: str) -> tuple:
    """``(plan, metadata rows)`` of the validated panel, cross-checked against ``panel.json``.

    The panel plan's hypernetwork and script hashes must equal the code this process imported.
    """
    if not layout.preparation(panel).is_file() or not layout.panel(panel).is_file():
        raise EvalError(f'{layout.preparation(panel)} or {layout.panel(panel)} is missing: the {panel} panel is not '
                        f'validated; run hyak/slurm/submit.sh eval-prepare (python -m hyak.eval.validate_prefixes '
                        f'--panel {panel}) and score it before packing')
    prepared = read_json(layout.preparation(panel))
    plan = prepared['plan']
    check_plan_code(plan, layout.preparation(panel))
    rows = read_json(layout.panel(panel))
    if digest(rows) != plan['panel_sha256'] or prepared['panel_sha256'] != plan['panel_sha256']:
        raise EvalError(f'{layout.panel(panel)} does not match its plan; rerun hyak.eval.validate_prefixes')
    metadata = prepared['metadata']
    if metadata != [{k: v for k, v in r.items() if k != 'prefix'} for r in rows]:
        raise EvalError(f'{layout.preparation(panel)} metadata differ from {layout.panel(panel)}')
    return plan, metadata


def load_run_scores(scores_dir: Path, rows: list, panel_plan: dict, *, partial: bool = False) -> dict:
    """Controls and per-tokenizer losses of one run as float64 arrays (validated like the worker did).

    Returns ``{'plan', 'model', 'indices', 'canonical' (P,32), 'no_prefix' (P,32), 'transferred' (K,P,32),
    'configs', 'tokenizer_sha256', 'elapsed_seconds', 'gpus', 'missing'}``.
    """
    import numpy as np
    from natural_transfer import validate_scores
    scores_dir = Path(scores_dir)
    plan = read_json(scores_dir / 'plan.json')
    expected = digest(plan)
    own = ('models', 'scoring_code_sha256')     # the scoring plan's additions to the panel plan
    if {k: v for k, v in plan.items() if k not in own} != {k: v for k, v in panel_plan.items() if k not in own}:
        raise EvalError(f'{scores_dir}/plan.json was made for another panel plan; rescore or re-validate')
    if len(plan['models']) != 1:
        raise EvalError(f'{scores_dir}/plan.json must name exactly one model')
    model = plan['models'][0]['name']
    controls = read_json(scores_dir / 'canonical.json')
    if controls['plan_sha256'] != expected:
        raise EvalError('Canonical controls belong to another comparison')  # modal_random_pool_50.py:159-160
    canonical = validate_scores(rows, controls['canonical'])
    no_prefix = validate_scores(rows, controls['no_prefix'])
    indices = list(plan['indices'])
    present, missing = [], []
    for index in indices:
        (present if (scores_dir / f'{pool_index_name(index)}.json').is_file() else missing).append(index)
    if missing and not partial:
        raise EvalError(f'{len(missing)} of {len(indices)} tokenizers have no result in {scores_dir} '
                        f'(first: {missing[:10]}); finish scoring or pass --partial')
    transferred = np.zeros((len(present), len(rows), SUFFIX_TOKENS), dtype=float)
    configs, hashes, elapsed, gpus = [], [], [], set()
    for k, index in enumerate(present):
        result = read_json(scores_dir / f'{pool_index_name(index)}.json')
        if result['plan_sha256'] != expected:
            raise EvalError('Existing scores belong to another comparison')  # modal_random_pool_50.py:172-173
        if result['index'] != index or result['model'] != model:
            raise EvalError(f'{scores_dir}/{pool_index_name(index)}.json is for index {result["index"]} / '
                            f'model {result["model"]}')
        transferred[k] = validate_scores(rows, result['transferred'])
        configs.append(result['config'])
        hashes.append(result['tokenizer_sha256'])
        elapsed.append(result['elapsed_seconds'])
        gpus.add(result.get('gpu'))
    tokenizer_table = plan['hyak']['tokenizer_sha256']
    if any(tokenizer_table[str(i)] != h for i, h in zip(present, hashes)):
        raise EvalError('A result was scored with a tokenizer other than the validated one')
    return dict(plan=plan, plan_sha256=expected, model=model, indices=present, missing=missing,
                canonical=canonical, no_prefix=no_prefix, transferred=transferred, configs=configs,
                tokenizer_sha256=hashes, elapsed_seconds=elapsed, gpus=sorted(g for g in gpus if g),
                controls_gpu=controls.get('gpu'))


# --- pool panel --------------------------------------------------------------------------------------
def check_pool_rows(rows: list, plan: dict, repo: Path | None = None) -> None:
    """Checks of scripts/frozen_policy_search.py:68-77 (original spec order, shared encoding validation)."""
    repo = Path(repo) if repo is not None else repo_root()
    source = read_json(repo / 'reports/rl-pool-sweep-20k/spec.json')
    if digest(source) != plan['source_spec_sha256']:
        raise EvalError('Original evaluation specification changed')
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split]
        original = source['panels'][split]
        if [r['original_id'] for r in selected] != original['document_ids'] or \
                [r['domain'] for r in selected] != original['domains']:
            raise EvalError('Original source IDs, domain membership or split order changed')
    count = len(plan['indices'])
    if any(not r['canonical_joint_encoding_matches'] or r['maximum_positions'] > POSITION_CAP
           or len(r['canonical_continuation_ids']) != SUFFIX_TOKENS or len(r['native_prefix_lengths']) != count
           for r in rows):
        raise EvalError('Incomplete shared encoding validation')


def reward_table(scores: dict, rows: list) -> dict:
    """The packed table of scripts/frozen_policy_search.py:78-80 for one complete run.

    ``canonical_nats = base.sum(1)``, ``transferred_nats = losses.sum(2)``; the
    shapes and coverage checks of :51-67 apply (all 1,000 tokenizers, 576 x 32).
    """
    import numpy as np
    plan = scores['plan']
    if plan['indices'] != list(range(1000)) or scores['indices'] != list(range(1000)):
        raise EvalError('Full 1000-tokenizer coverage required')  # frozen_policy_search.py:51-52
    base, losses = scores['canonical'], scores['transferred']
    if base.shape != (576, 32) or losses.shape != (1000, 576, 32):
        raise EvalError('All576 passages and32 scored targets are required for every tokenizer')
    if not np.isfinite(base).all() or not np.isfinite(losses).all() or (base < 0).any() or (losses < 0).any():
        raise EvalError('Invalid raw loss values')
    return dict(plan=plan, rows=rows, canonical_nats=base.sum(1).tolist(),
                transferred_nats=losses.sum(2).tolist(), configs=scores['configs'],
                tokenizer_hashes=scores['tokenizer_sha256'])


def reward_table_bytes(cache: dict) -> bytes:
    """Serialisation of scripts/frozen_policy_search.py:230."""
    return gzip.compress(json.dumps(cache, sort_keys=True, separators=(',', ':')).encode(), mtime=0)


def pool_summary(cache: dict, no_prefix_nats=None) -> dict:
    """Per split balanced BPB, gaps and ranks, as scripts/frozen_policy_search.py:263-275 reports them.

    Uses ``frozen_policy_search.reference`` (imported), i.e. ``balanced_scores``
    per split.  ``uniform_bpb`` is the mean over tokenizers (uniform selection);
    ``uniform_mean_gap_bpb`` = uniform_bpb - canonical_bpb.
    """
    import numpy as np
    from frozen_policy_search import balanced_scores, reference
    refs = reference(cache)
    configs = cache['configs']
    indices = cache['plan']['indices']
    out = {}
    for split, ref in refs.items():
        bpb, canonical = ref['bpb'], ref['canonical_bpb']
        gaps = bpb - canonical
        best = int(bpb.argmax())
        entry = dict(canonical_bpb=canonical, uniform_bpb=float(bpb.mean()),
                     uniform_mean_gap_bpb=float(bpb.mean() - canonical), median_gap_bpb=float(np.median(gaps)),
                     best_index=indices[best], best_bpb=float(bpb.max()), best_gap_bpb=float(gaps[best]),
                     easiest_index=indices[int(bpb.argmin())], easiest_gap_bpb=float(gaps.min()),
                     tokenizers=len(bpb),
                     ranking=[dict(index=indices[i], rank=int(ref['rank'][i]), bpb=float(bpb[i]),
                                   gap_bpb=float(gaps[i]), algorithm=configs[i]['algorithm'],
                                   pretokenizer=configs[i]['pretokenizer'], vocab_size=configs[i]['vocab_size'])
                              for i in np.argsort(-bpb, kind='stable')])
        if no_prefix_nats is not None:
            ids = [i for i, r in enumerate(cache['rows']) if r['split'] == split]
            rows = [cache['rows'][i] for i in ids]
            entry['bos_only_bpb'] = float(balanced_scores(rows, np.asarray(no_prefix_nats)[ids]))
        by_group = {}
        for key in ('algorithm', 'pretokenizer'):
            for value in sorted({c[key] for c in configs}):
                mask = np.array([c[key] == value for c in configs])
                by_group[f'{key}={value}'] = dict(tokenizers=int(mask.sum()), mean_gap_bpb=float(gaps[mask].mean()),
                                                  median_gap_bpb=float(np.median(gaps[mask])))
        entry['groups'] = by_group
        out[split] = entry
    return out


def rankings_csv(summary: dict) -> str:
    """One row per tokenizer: rank, BPB and gap on both splits (as reports/frozen-policy-search-20k/rankings.csv)."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['index', 'algorithm', 'pretokenizer', 'vocab_size',
                     'screen_rank', 'screen_bpb', 'screen_gap', 'confirm_rank', 'confirm_bpb', 'confirm_gap'])
    by_split = {s: {r['index']: r for r in summary[s]['ranking']} for s in SPLITS}
    for index in sorted(by_split['screen']):
        a, b = by_split['screen'][index], by_split['confirm'][index]
        writer.writerow([index, a['algorithm'], a['pretokenizer'], a['vocab_size'], a['rank'], repr(a['bpb']),
                         repr(a['gap_bpb']), b['rank'], repr(b['bpb']), repr(b['gap_bpb'])])
    return stream.getvalue()


def pool_comparison(summaries: dict) -> dict:
    """Uniform versus RL on the same 1,000 tokenizers and passages (paired by tokenizer index)."""
    import numpy as np
    uniform, rl = summaries['uniform'], summaries['rl']
    result = {}
    for split in SPLITS:
        u = {r['index']: r for r in uniform[split]['ranking']}
        r_ = {r['index']: r for r in rl[split]['ranking']}
        if set(u) != set(r_):
            raise EvalError('Uniform and RL were scored on different tokenizer sets')
        order = sorted(u)
        gu = np.array([u[i]['gap_bpb'] for i in order])
        gr = np.array([r_[i]['gap_bpb'] for i in order])
        bu = np.array([u[i]['bpb'] for i in order])
        br = np.array([r_[i]['bpb'] for i in order])
        diff = gr - gu
        result[split] = dict(
            tokenizers=len(order),
            canonical_bpb=dict(uniform=uniform[split]['canonical_bpb'], rl=rl[split]['canonical_bpb']),
            uniform_mean_gap_bpb=dict(uniform=uniform[split]['uniform_mean_gap_bpb'], rl=rl[split]['uniform_mean_gap_bpb']),
            mean_gap_change_rl_minus_uniform=float(diff.mean()),
            median_gap_change_rl_minus_uniform=float(np.median(diff)),
            mean_absolute_bpb_change_rl_minus_uniform=float((br - bu).mean()),
            fraction_rl_gap_lower=float((diff < 0).mean()),
            worst_index=dict(uniform=uniform[split]['best_index'], rl=rl[split]['best_index']),
            worst_gap_bpb=dict(uniform=uniform[split]['best_gap_bpb'], rl=rl[split]['best_gap_bpb']),
            per_tokenizer=[dict(index=i, uniform_gap_bpb=float(a), rl_gap_bpb=float(b), rl_minus_uniform=float(b - a))
                           for i, a, b in zip(order, gu, gr)])
    return result


def original_pool_reference(repo: Path | None = None) -> dict:
    """The original RL20k complete-pool numbers recomputed from the stored reward table: a reference from a
    different run (entropy 0.01, raw reward, original cuts, warmup initialization), never a reproduction target."""
    repo = Path(repo) if repo is not None else repo_root()
    cache = read_bundle(repo / PANEL_SPECS[PANEL_POOL].source_rel)
    summary = pool_summary(cache)
    return {'model': 'rl20k (original, B300, original pool)', 'source': PANEL_SPECS[PANEL_POOL].source_rel,
            'relation': 'different_run', 'reproduction_target': False,
            'compared_with': MODEL_NAMES['rl'],
            'splits': {s: {k: v for k, v in summary[s].items() if k != 'ranking'} for s in SPLITS},
            'note': f'Original checkpoint and original tokenizers: {ORIGINAL_RL_REFERENCE_NOTE}. '
                    'The uniform checkpoint has no original complete-pool table.'}


# --- natural panel -----------------------------------------------------------------------------------
def natural_bundle(scores_by_model: dict, rows: list, plan: dict) -> dict:
    """Packed natural-panel data in the shape of scripts/analyze_natural_model_comparison.py:54-56."""
    first = next(iter(scores_by_model.values()))
    models = {}
    for name, scores in scores_by_model.items():
        if scores['indices'] != plan['indices']:
            raise EvalError(f'{name}: missing tokenizer results')
        models[name] = dict(canonical_nats=scores['canonical'].sum(1).tolist(),
                            transferred_nats=scores['transferred'].sum(2).tolist(),
                            no_prefix_nats=scores['no_prefix'].sum(1).tolist())
    return dict(plan=plan, metadata=rows, models=models,
                tokenizers=[dict(index=i, config=c) for i, c in zip(first['indices'], first['configs'])],
                score_definition=SCORE_DEFINITION)


def natural_summary(data: dict, model_names: list, replicates: int | None = None) -> dict:
    """Five-domain balanced BPB per model / regime / split / tokenizer and cohort summaries.

    Point estimates and paired cluster-bootstrap intervals from
    ``analyze_natural_transfer.bootstrap_balanced`` (imported).  The regimes,
    cohorts and paired comparisons follow scripts/analyze_natural_model_comparison.py:80-135
    for an arbitrary list of models ('rl-minus-<other>' for every other model when
    an RL model is present).
    """
    import numpy as np
    from analyze_natural_transfer import bootstrap_balanced
    plan, rows = data['plan'], data['metadata']
    replicates = replicates or plan['bootstrap_replicates']
    if digest(plan['context_selected_ids']) != plan['context_selected_ids_sha256']:
        raise EvalError('Frozen stress-subset membership changed')
    tokenizers = data['tokenizers']
    m_count, n = len(model_names), len(tokenizers)
    base = np.array([data['models'][m]['canonical_nats'] for m in model_names])
    targets = np.array([data['models'][m]['transferred_nats'] for m in model_names])
    if base.shape != (m_count, len(rows)) or targets.shape != (m_count, n, len(rows)):
        raise EvalError('Missing model/document losses')
    selected = set(plan['context_selected_ids'])
    cohorts = {}
    representative = set(plan['representative_indices'])
    cohorts['representative36'] = [k for k, r in enumerate(tokenizers) if r['index'] in representative]
    for label, pre in [('regex', ['regex']), ('line256', ['line256']), ('no-regex', ['no-regex']),
                       ('weak_pretokenization', ['line256', 'no-regex'])]:
        cohorts[label] = [k for k in cohorts['representative36'] if tokenizers[k]['config']['pretokenizer'] in pre]
    cohorts['large_vocab_weak'] = [k for k in cohorts['weak_pretokenization']
                                   if tokenizers[k]['config']['vocab_size'] >= plan['large_vocab_minimum']]
    regimes = dict(all=np.ones(len(rows), dtype=bool), context=np.array([r['id'] in selected for r in rows]))
    metrics, summaries, comparisons = [], [], []
    rl_names = [m for m in model_names if m.startswith('rl')]
    for regime, mask in regimes.items():
        for split in SPLITS:
            ids = np.flatnonzero(mask & np.array([r['split'] == split for r in rows]))
            subset = [rows[i] for i in ids]
            if not subset:
                raise EvalError('Empty regime')
            stack = np.concatenate([base[:, ids], targets[:, :, ids].reshape(m_count * n, len(ids))])
            points, draws = bootstrap_balanced(subset, stack, replicates)
            canonical, trans = points[:m_count], points[m_count:].reshape(m_count, n)
            canonical_draws = draws[:, :m_count]
            trans_draws = draws[:, m_count:].reshape(len(draws), m_count, n)
            gaps, gap_draws = trans - canonical[:, None], trans_draws - canonical_draws[:, :, None]
            for m, name in enumerate(model_names):
                for k, tokenizer in enumerate(tokenizers):
                    metrics.append(dict(model=name, index=tokenizer['index'], regime=regime, split=split,
                                        canonical_bpb=float(canonical[m]), transferred_bpb=float(trans[m, k]),
                                        gap_bpb=float(gaps[m, k]),
                                        gap_ci95=np.quantile(gap_draws[:, m, k], [.025, .975]).tolist()))
                for cohort, ks in cohorts.items():
                    if not ks:
                        continue
                    median_draws = np.median(gap_draws[:, m, ks], axis=1)
                    summaries.append(dict(model=name, regime=regime, split=split, cohort=cohort, tokenizers=len(ks),
                                          canonical_bpb=float(canonical[m]),
                                          mean_transferred_bpb=float(trans[m, ks].mean()),
                                          median_transferred_bpb=float(np.median(trans[m, ks])),
                                          mean_gap_bpb=float(gaps[m, ks].mean()),
                                          median_gap_bpb=float(np.median(gaps[m, ks])),
                                          median_gap_ci95=np.quantile(median_draws, [.025, .975]).tolist(),
                                          min_gap_bpb=float(gaps[m, ks].min()), max_gap_bpb=float(gaps[m, ks].max())))
            primary = [k for k, r in enumerate(tokenizers) if r['index'] == plan['primary_tokenizer']]
            selections = {**cohorts, 'fixed%04d' % plan['primary_tokenizer']: primary}
            for rl_name in rl_names:
                a = model_names.index(rl_name)
                for other in model_names:
                    if other == rl_name:
                        continue
                    b = model_names.index(other)
                    for cohort, ks in selections.items():
                        if not ks:
                            continue
                        absolute = trans[a, ks] - trans[b, ks]
                        gap = gaps[a, ks] - gaps[b, ks]
                        absolute_draws = (trans_draws[:, a, ks] - trans_draws[:, b, ks]).mean(1)
                        difference_draws = (gap_draws[:, a, ks] - gap_draws[:, b, ks]).mean(1)
                        comparisons.append(dict(
                            comparison=f'{rl_name}-minus-{other}', regime=regime, split=split, cohort=cohort,
                            tokenizers=len(ks), mean_absolute_bpb_change=float(absolute.mean()),
                            mean_absolute_bpb_change_ci95=np.quantile(absolute_draws, [.025, .975]).tolist(),
                            mean_gap_change_bpb=float(gap.mean()),
                            mean_gap_change_ci95=np.quantile(difference_draws, [.025, .975]).tolist(),
                            canonical_bpb_change=float(canonical[a] - canonical[b])))
    return dict(models=model_names, metrics=metrics, summaries=summaries, paired_comparisons=comparisons,
                cohort_indices={label: [tokenizers[k]['index'] for k in ks] for label, ks in cohorts.items()},
                bootstrap_replicates=replicates)


def original_natural_reference(repo: Path | None = None, replicates: int | None = None) -> dict:
    """The three original models' natural-panel summaries recomputed from the stored bundle."""
    repo = Path(repo) if repo is not None else repo_root()
    data = read_bundle(repo / PANEL_SPECS[PANEL_NATURAL].twin_rel)
    summary = natural_summary(data, ['zett20k', 'uniform20k', 'rl20k'], replicates)
    keep = [s for s in summary['summaries'] if s['cohort'] in ('weak_pretokenization', 'representative36', 'regex')]
    return {'source': PANEL_SPECS[PANEL_NATURAL].twin_rel, 'summaries': keep,
            'paired_comparisons': [c for c in summary['paired_comparisons']
                                   if c['cohort'] in ('weak_pretokenization', 'fixed0771')],
            'relation': 'different_runs', 'reproduction_target': False,
            'note': 'Original checkpoints (B300, original pool) on the stored panel; a plausibility reference. '
                    f"Its 'rl20k' model is {ORIGINAL_RL_REFERENCE_NOTE}; its 'uniform20k' model is the original "
                    'uniform run (warmup initialization, original pool).'}


# --- command line ----------------------------------------------------------------------------------
def _refuse_repo_reports(out_dir: Path) -> None:
    reports = (repo_root() / 'reports').resolve()
    target = Path(out_dir).resolve()
    if target == reports or reports in target.parents:
        raise EvalError(f'{out_dir} is inside the repository reports/ directory; choose another --out-dir')


def panel_block(plan: dict) -> dict:
    """How the scored panel differs from the stored one (D11: extra trimming is reported, never hidden)."""
    hyak = plan.get('hyak', {})
    trimmed_ids = list(hyak.get('trimmed_row_ids', []))
    return dict(panel=hyak.get('panel'), panel_sha256=plan['panel_sha256'],
                stored_panel_sha256=hyak.get('stored_panel_sha256'),
                equals_stored=plan['panel_sha256'] == hyak.get('stored_panel_sha256'),
                trimmed_rows=int(hyak.get('trimmed_rows', len(trimmed_ids))), trimmed_row_ids=trimmed_ids,
                trimmed_chars=int(hyak.get('trimmed_chars', 0)),
                accepted_valid_context_trims=list(hyak.get('accepted_valid_context_trims', [])),
                validation_rounds=hyak.get('validation_rounds'), pool_index_sha256=hyak.get('pool_index_sha256'),
                indices=len(plan.get('indices', [])),
                note='Rows of the stored panel whose prefix lost characters on the left for the regenerated pool '
                     '(targets and scored bytes unchanged); see eval/panels/<panel>/validation-report.json')


def log_panel(name: str, block: dict) -> None:
    log(f'{name} panel: {block["trimmed_rows"]} rows trimmed ({block["trimmed_chars"]} characters) relative to the '
        f'stored panel; equal to the stored panel: {block["equals_stored"]}'
        + (f'; {len(block["accepted_valid_context_trims"])} accepted trims of originally valid contexts'
           if block['accepted_valid_context_trims'] else ''))


def scores_state(scores_dir: Path) -> str:
    """'not scored', 'incomplete (k/N tokenizers)' or 'complete' (cheap: plan.json + file existence)."""
    scores_dir = Path(scores_dir)
    if not (scores_dir / 'plan.json').is_file():
        return 'not scored'
    indices = read_json(scores_dir / 'plan.json')['indices']
    present = sum((scores_dir / f'{pool_index_name(i)}.json').is_file() for i in indices)
    if present < len(indices) or not (scores_dir / 'canonical.json').is_file():
        return f'incomplete ({present}/{len(indices)} tokenizers)'
    return 'complete'


def pack_pool(layout: EvalLayout, runs, out_dir: Path, *, partial: bool = False) -> dict:
    """Pack every complete run (``partial``: summarise incomplete ones too).  Returns ``{run: state}``."""
    plan, rows = load_prepared(layout, PANEL_POOL)
    panel = panel_block(plan)
    log_panel(PANEL_POOL, panel)
    summaries, states = {}, {}
    for run in runs:
        scores_dir = layout.scores_dir(run, PANEL_POOL)
        states[run] = scores_state(scores_dir)
        if states[run] == 'not scored' or (states[run] != 'complete' and not partial):
            log(f'pool/{run}: {states[run]} in {scores_dir}; skipped')
            continue
        scores = load_run_scores(scores_dir, rows, plan, partial=partial)
        folder = Path(out_dir) / run
        info = dict(model=scores['model'], scores_dir=str(scores_dir), plan_sha256=scores['plan_sha256'],
                    checkpoint=scores['plan']['models'][0], gpus=scores['gpus'], controls_gpu=scores['controls_gpu'],
                    tokenizers=len(scores['indices']), missing=scores['missing'], panel=panel,
                    mean_elapsed_seconds=(sum(scores['elapsed_seconds']) / len(scores['elapsed_seconds'])
                                          if scores['elapsed_seconds'] else None),
                    created_at=utc_now())
        if scores['missing']:
            partial_cache = dict(plan={**plan, 'indices': scores['indices']}, rows=rows,
                                 canonical_nats=scores['canonical'].sum(1).tolist(),
                                 transferred_nats=scores['transferred'].sum(2).tolist(), configs=scores['configs'])
            summary = pool_summary(partial_cache, scores['no_prefix'].sum(1))
            atomic_write_json(folder / 'summary.json', {**info, 'partial': True, 'splits': summary})
            log(f'pool/{run}: PARTIAL summary of {len(scores["indices"])} tokenizers written; no reward table')
            continue
        check_pool_rows(rows, scores['plan'])
        cache = reward_table(scores, rows)
        atomic_write_bytes(folder / 'reward_table.json.gz', reward_table_bytes(cache))
        summary = pool_summary(cache, scores['no_prefix'].sum(1))
        atomic_write_json(folder / 'summary.json', {**info, 'partial': False, 'splits': summary})
        atomic_write_text(folder / 'rankings.csv', rankings_csv(summary))
        summaries[run] = summary
        for split in SPLITS:
            s = summary[split]
            log(f'pool/{run} {split}: canonical {s["canonical_bpb"]:.5f}  uniform-mean gap '
                f'{s["uniform_mean_gap_bpb"]:+.5f}  worst #{s["best_index"]:04d} {s["best_gap_bpb"]:+.5f}')
    reference = original_pool_reference()
    comparison = {'reference_original': reference, 'created_at': utc_now(), 'runs': states, 'panel': panel}
    if {'uniform', 'rl'} <= set(summaries):
        comparison['uniform_vs_rl'] = pool_comparison(summaries)
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['split', 'index', 'uniform_gap_bpb', 'rl_gap_bpb', 'rl_minus_uniform'])
        for split in SPLITS:
            for row in comparison['uniform_vs_rl'][split]['per_tokenizer']:
                writer.writerow([split, row['index'], repr(row['uniform_gap_bpb']), repr(row['rl_gap_bpb']),
                                 repr(row['rl_minus_uniform'])])
        atomic_write_text(Path(out_dir) / 'comparison.csv', stream.getvalue())
    atomic_write_json(Path(out_dir) / 'comparison.json', comparison)
    return states


def pack_natural(layout: EvalLayout, runs, out_dir: Path, *, partial: bool = False) -> dict:
    """Pack the complete runs into one natural-panel bundle and summary.  Returns ``{run: state}``.

    The five-domain summaries need every tokenizer of a run, so incomplete runs are
    skipped even with ``partial``.
    """
    plan, rows = load_prepared(layout, PANEL_NATURAL)
    panel = panel_block(plan)
    log_panel(PANEL_NATURAL, panel)
    expected = list(read_json(repo_root() / PANEL_SPECS[PANEL_NATURAL].plan_rel)['indices'])
    missing_indices = [i for i in expected if i not in plan['indices']]
    if sorted(plan['indices']) != sorted(expected) or len(rows) != PANEL_SPECS[PANEL_NATURAL].rows:
        message = (f'the natural panel was validated for {len(plan["indices"])} of the {len(expected)} plan indices '
                   f'(missing {missing_indices[:10]}) with {len(rows)} rows')
        if not partial:
            raise EvalError(f'{message}; validate it for all of them (python -m hyak.eval.validate_prefixes '
                            '--panel natural --force) or pass --partial for a summary marked partial')
        log(f'natural: PARTIAL: {message}')
    scores_by_model, states = {}, {}
    for run in runs:
        scores_dir = layout.scores_dir(run, PANEL_NATURAL)
        states[run] = scores_state(scores_dir)
        if states[run] != 'complete':
            log(f'natural/{run}: {states[run]} in {scores_dir}; skipped')
            continue
        scores_by_model[MODEL_NAMES[run]] = load_run_scores(scores_dir, rows, plan)
    if not scores_by_model:
        # Not complete yet (exit 1 in main), like the pool panel: 09_eval_score.sbatch runs pack after every
        # scoring step that ended without a stop signal, also when it stopped claiming before the time limit.
        log(f'natural: no run has complete natural-panel scores yet ({states}); nothing packed')
        return states
    data = natural_bundle(scores_by_model, rows, plan)
    atomic_write_bytes(Path(out_dir) / 'results.json.gz', gzip.compress(
        json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))
    names = [MODEL_NAMES[r] for r in RUNS if MODEL_NAMES[r] in scores_by_model]
    summary = natural_summary(data, names)
    summary['checkpoints'] = {name: s['plan']['models'][0] for name, s in scores_by_model.items()}
    summary['reference_original'] = original_natural_reference()
    summary['runs'] = states
    summary['panel'] = panel
    summary['partial'] = bool(missing_indices)
    if missing_indices:
        summary['missing_indices'] = missing_indices
    summary['created_at'] = utc_now()
    atomic_write_json(Path(out_dir) / 'summary.json', summary)
    for s in summary['summaries']:
        if s['regime'] == 'context' and s['split'] == 'confirm' and s['cohort'] == 'weak_pretokenization':
            log(f'natural {s["model"]}: confirm context weak-pretokenization median gap {s["median_gap_bpb"]:+.5f}')
    return states


def main(argv=None) -> int:
    """Exit codes: 0 every requested run is complete and packed; 75 at least one requested run (possibly the
    one just scored) is not complete for this panel (complete runs were packed); 2 usage or input error;
    1 crash (traceback)."""
    parser = argparse.ArgumentParser(prog='python -m hyak.eval.pack', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--runs', nargs='+', choices=RUNS, default=list(RUNS))
    parser.add_argument('--out-dir', type=Path, help='default: $ANYTOK_ROOT/eval/reports/<panel>')
    parser.add_argument('--partial', action='store_true',
                        help='summarise incomplete score sets (pool: summary only, never a reward table)')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--code-dir', type=Path, help='pinned code tree (default: <repo>/runtime/anytok_pool_20k)')
    args = parser.parse_args(argv)
    try:
        setup_imports(args.code_dir)
        from hyak.common.config import load_config
        from hyak.common.paths import Paths
        layout = EvalLayout(Paths.from_config(load_config(args.config)))
        out_dir = args.out_dir or layout.report_dir(args.panel)
        _refuse_repo_reports(out_dir)
        if args.panel == PANEL_POOL:
            states = pack_pool(layout, args.runs, out_dir, partial=args.partial)
        else:
            states = pack_natural(layout, args.runs, out_dir, partial=args.partial)
        if any(state == 'complete' for state in states.values()) or args.partial:
            log(f'{args.panel}: outputs in {out_dir}; runs {states}')
        else:
            log(f'{args.panel}: nothing packed (no run is complete); runs {states}')
        if any(state != 'complete' for state in states.values()):
            print(f'NOT COMPLETE: {args.panel} panel runs {states}; finish scoring (python -m hyak.eval.score) '
                  'and pack again', file=sys.stderr)
            return EXIT_LATER
        return EXIT_OK
    except (EvalError, ConfigError, OSError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return EXIT_USAGE
    except Exception:
        return crashed()


if __name__ == '__main__':
    sys.exit(main())
