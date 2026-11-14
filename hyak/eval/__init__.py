"""Corrected-protocol evaluation of the two final checkpoints (DESIGN.md D11, section 5.H).

Entry points (run from the repository root with the ``train`` environment)::

    python -m hyak.eval.rebuild_panel     --panel pool|natural|all     rebuild both stored panels (CPU, seconds)
    python -m hyak.eval.prepare_features  [--indices ...]              hypernetwork inputs per tokenizer (CPU, claims)
    python -m hyak.eval.validate_prefixes --panel pool|natural         native prefix lengths / joint trim (CPU)
    python -m hyak.eval.score --run uniform|rl --panel pool|natural    GPU scoring (claims, resumable)
    python -m hyak.eval.pack --panel pool|natural [--run ...]          reward table, BPB summaries, comparison

Two panels, both stored in the repository as canonical SmolLM2 token ids plus hashes:

* ``pool``: the corrected 576-passage panel (288 screen + 288 confirm; nine
  buckets x 32 per split) of the complete-pool reward table,
  ``reports/frozen-policy-search-20k/reward_table.json.gz`` (whole-panel hash
  ``30ee5e0d...``), scored on all 1,000 pool tokenizers.
* ``natural``: the 2,032-passage natural-text panel (five domains),
  ``reports/natural-transfer-20k/results.json.gz`` ``preparation.metadata``
  (identical to ``reports/natural-model-comparison-20k/results.json.gz``
  ``metadata``; hash ``5c8ad1c6...``), scored on the 38 indices of
  ``plans/natural-model-comparison-v1.json``.

Storage below ``$ANYTOK_ROOT/eval`` (``hyak.common.paths``: eval_panels,
eval_features, eval_scores, eval_reports; the layout below those belongs to this
package, see :class:`EvalLayout`)::

    panels/<panel>/rebuilt.json             rebuild_panel: stored rows + decoded 'prefix'
    panels/<panel>/rebuild-report.json
    panels/<panel>/panel.json               validate_prefixes: validated rows (the scorer's panel)
    panels/<panel>/plan.json                derived plan (panel_sha256, indices, pool identity)
    panels/<panel>/preparation.json         {plan, panel_sha256, metadata (rows without prefix), files}
    panels/<panel>/validation-report.json   rounds, trimmed rows, rows unchanged
    features/NNNN/{forms.npy,priors.npy,manifest.json}
    scores/<run>/<panel>/plan.json          scoring plan = panel plan + checkpoint manifest
    scores/<run>/<panel>/canonical.json     canonical and BOS-only controls (computed once)
    scores/<run>/<panel>/NNNN.json          per-tokenizer per-token losses
    scores/<run>/<panel>/complete.json      written when every index has a result
    claims/<task>/<name>/owner.json         atomic-mkdir work claims (features, scores, controls)
    reports/<panel>/...                     pack outputs (never under the repository's reports/)

Scoring code is imported from the repository's Modal-free modules with
``<repo>/scripts`` and the pinned code tree (default ``<repo>/runtime/anytok_pool_20k``,
exactly the tree the Modal images mounted at /opt/zett) on ``sys.path``; see
:func:`setup_imports`.  Logic that lived in Modal-importing modules is copied
with ``file:line`` references.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import uuid

from hyak.common import slurm
from hyak.common.io import atomic_write_json, read_json, sha256_file, utc_now

#: Final checkpoint directory name of both runs (20,000 updates).
FINAL_CHECKPOINT = 'step-020000'
#: Model names in score results and packed bundles.  The uniform run reruns the original uniform20k protocol; the
#: RL run is the straight-BPB-reward counterpart of the joint regret RL20k (entropy 0, corrected paired cuts, D8),
#: not a rerun of the original 'rl20k' (entropy 0.01, raw reward, original cuts, warmup initialization).
MODEL_NAMES = {'uniform': 'uniform20k-hyak', 'rl': 'rl20k-straight-entropy0-hyak'}
#: How the stored original reports relate to the RL run of this port (labels in every packed comparison).
ORIGINAL_RL_REFERENCE_NOTE = ("a reference from a different run, never a reproduction target: the stored original "
                              "'rl20k' used entropy 0.01, the raw training-BPB reward, the original example cuts and "
                              "the private warmup initialization (and the original pool and hardware); "
                              f"{MODEL_NAMES['rl']} uses entropy 0, the straight paired transferred-BPB reward, the "
                              "corrected paired cuts and a random hypernetwork initialization")
#: Inference files of a checkpoint (scripts/modal_random_pool_50.py:127).
CHECKPOINT_FILES = ('lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors')
#: Panel names.
PANEL_POOL = 'pool'
PANEL_NATURAL = 'natural'
PANELS = (PANEL_POOL, PANEL_NATURAL)
#: Positions / targets of the corrected protocol (natural_transfer.score_rows; plans *.json position_cap/suffix_tokens).
POSITION_CAP = 128
SUFFIX_TOKENS = 32
#: Native prefix token cap (pool_regime.required_start: len(native) > 95 rejected).
NATIVE_PREFIX_CAP = 95
#: Canonical prefix budget (pool_regime.required_start: 16 <= len(source) <= 64).
CANONICAL_PREFIX_MIN = 16
CANONICAL_PREFIX_MAX = 64
#: Surface-form pieces (modal_eval_rl_pool.py:159 build_surface_forms(..., 10, ...)).
MAX_PIECES = 10
#: Maximum validation rounds (modal_frozen_policy_cache.py:181 range(5); modal_random_pool_50.py:69).
MAX_ROUNDS = 5
#: Repository-relative path of the pinned trainer/hypernetwork tree the scorers mounted (BASE in the Modal scripts).
CODE_TREE_REL = 'runtime/anytok_pool_20k'
#: SHA-256 of the pinned SmolLM2 tokenizer.json (plans/zett-modal-20k-inputs.json files['model/tokenizer.json']).
CANONICAL_TOKENIZER_SHA256 = '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'
#: Model input hash table (files['model/<name>'] = {bytes, sha256}).
MODEL_INPUTS_REL = 'plans/zett-modal-20k-inputs.json'
#: Pool plan (byte-copied to input/pool/plan.json by hyak.pool.collect).
POOL_PLAN_REL = 'plans/tokenizer-pool-1000-v1.json'
#: Shared-settings policy source (frozen_policy_search checks cache['plan']['source_policy_sha256'] against it).
POLICY_REL = 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'


@dataclass(frozen=True)
class PanelSpec:
    """Where a stored panel lives in the repository and what it must hash to."""

    name: str
    source_rel: str          # bundle holding the rows (canonical ids, no prefix text)
    source_sha256: str       # SHA-256 of that bundle file
    twin_rel: str | None     # second bundle with the identical rows (natural panel only)
    twin_sha256: str | None
    plan_rel: str            # the original frozen plan the panel belongs to
    rows: int
    panel_sha256: str        # digest(rows with 'prefix') of the original panel
    splits: tuple            # ((split, rows), ...)


PANEL_SPECS = {
    PANEL_POOL: PanelSpec(
        name=PANEL_POOL,
        source_rel='reports/frozen-policy-search-20k/reward_table.json.gz',
        source_sha256='370e78868f1eb6284c5dd975146c0597ea8e5344df56bc622471cc64a0668151',
        twin_rel=None, twin_sha256=None,
        plan_rel='plans/frozen-policy-search-v1.json',
        rows=576,
        panel_sha256='30ee5e0d25b2c77fd3d743d8a1a0635670572bf086c3adeef22bf192834e7fc9',
        splits=(('screen', 288), ('confirm', 288)),
    ),
    PANEL_NATURAL: PanelSpec(
        name=PANEL_NATURAL,
        source_rel='reports/natural-transfer-20k/results.json.gz',
        source_sha256='4b5f65f7afa2619672b96aade8b35a2eb8a273c5dc5aae7f68969db66180b423',
        twin_rel='reports/natural-model-comparison-20k/results.json.gz',
        twin_sha256='705e8fd90da9e2cfb03d04db51b1e6c6ce189d73fb6ae6cf6a60a18ccf8c3a75',
        plan_rel='plans/natural-model-comparison-v1.json',
        rows=2032,
        panel_sha256='5c8ad1c63adf655cd9e2610500cf5ec58e9cc9190af3e16142fd7868ccec2f64',
        splits=(('screen', 1016), ('confirm', 1016)),
    ),
}


class EvalError(RuntimeError):
    """A loud, expected failure of an evaluation step; the message names the remedy."""


#: Exit codes of the evaluation tools (project-wide contract read by the job scripts).
EXIT_OK = 0            # success or nothing left to do
EXIT_CRASH = 1         # unexpected error; the traceback is in the job log
EXIT_USAGE = 2         # usage, configuration, input or environment error (message names the remedy)
EXIT_PROTOCOL = 3      # protocol stop that needs the user's decision (validate_prefixes)
EXIT_CHECK = 4         # a check failed (rebuild_panel: a stored panel did not rebuild exactly)
EXIT_LATER = 75        # not complete yet, try later (score: work left in claims of live jobs; pack: NOT COMPLETE)


def log(message: str) -> None:
    """Timestamped progress line on stdout (the sbatch log)."""
    print(f'[hyak eval {utc_now()}] {message}', flush=True)


def crashed(status=None) -> int:
    """Print the traceback of the exception being handled, mark the status failed, return :data:`EXIT_CRASH`."""
    import traceback
    traceback.print_exc()
    kind, value = sys.exc_info()[:2]
    message = f'crashed ({kind.__name__ if kind else "error"}: {value}); traceback in the job log'
    print(f'ERROR: {message}', file=sys.stderr, flush=True)
    if status is not None:
        try:
            last = status.last[status.completed_field] if status.last else 0
            status.update(last, state='failed', error=message[:2000])
        except Exception:                     # the status must never hide the original error
            traceback.print_exc()
    return EXIT_CRASH


def repo_root() -> Path:
    from hyak.common.config import repo_root as _root
    return _root()


def validate_panel_name(panel: str) -> str:
    if panel not in PANELS:
        raise EvalError(f'Unknown panel {panel!r}; expected one of {", ".join(PANELS)}')
    return panel


# --- imports of the repository's own scoring code ---------------------------------------------
_IMPORTED = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: Repository scripts whose functions the evaluation imports (scoring, trimming, packing); their SHA-256 is
#: recorded in every panel plan and compared by hyak.eval.score and hyak.eval.pack (they are not pinned).
SCORING_SCRIPTS = ('natural_transfer.py', 'pool_regime.py', 'uniform_trajectory.py', 'uniform_weakness.py',
                   'pool_generalization.py', 'frozen_policy_search.py', 'analyze_natural_transfer.py')


def verify_code_tree(code_dir: Path, repo: Path | None = None) -> dict:
    """Check that ``code_dir/hf_hypernet`` holds exactly the pinned hypernetwork files (size and SHA-256).

    The scorers of the original runs mounted ``runtime/anytok_pool_20k/hf_hypernet``
    (``BASE / 'hf_hypernet'`` in scripts/modal_random_pool_50.py:22,32).  Every file of
    ``hyak.code.pinned.HF_HYPERNET_FILES`` is compared with its pin in
    ``hyak.code.pinned.SNAPSHOT_FILES`` (not with the repository file, which may itself have
    been edited), so the default tree and a staged tree (``$ANYTOK_ROOT/code/uniform`` or
    ``code/rl``) are held to the same bytes.  Returns ``{'hf_hypernet/<name>': sha256}``.
    """
    from hyak.code import pinned
    repo = Path(repo) if repo is not None else repo_root()
    code_dir = Path(code_dir)
    hashes = {}
    for name in pinned.HF_HYPERNET_FILES:
        expected, size = pinned.SNAPSHOT_FILES[f'{pinned.UNIFORM_SNAPSHOT}/hf_hypernet/{name}']
        staged = code_dir / 'hf_hypernet' / name
        if not staged.is_file():
            raise EvalError(f'{staged} is missing; --code-dir must be a staged code tree '
                            f'(e.g. $ANYTOK_ROOT/code/uniform) or {repo / CODE_TREE_REL}')
        actual = _sha(staged)
        if actual != expected or staged.stat().st_size != size:
            raise EvalError(f'{staged} is not the pinned file (sha256 {actual} != {expected}, '
                            f'{staged.stat().st_size} bytes, pinned {size}); restore it (git checkout '
                            f'{pinned.UNIFORM_SNAPSHOT}/hf_hypernet/{name}) or restage the code tree '
                            '(python -m hyak.code.stage_code)')
        hashes[f'hf_hypernet/{name}'] = actual
    return hashes


def scripts_hashes(repo: Path | None = None) -> dict:
    """``{'scripts/<name>': sha256}`` of :data:`SCORING_SCRIPTS`."""
    repo = Path(repo) if repo is not None else repo_root()
    hashes = {}
    for name in SCORING_SCRIPTS:
        path = repo / 'scripts' / name
        if not path.is_file():
            raise EvalError(f'{path} is missing; the evaluation imports it')
        hashes[f'scripts/{name}'] = _sha(path)
    return hashes


def current_code_hashes() -> dict:
    """``{'hf_hypernet': {...}, 'scripts': {...}}`` of the code this process imported (see setup_imports)."""
    info = next(iter(_IMPORTED.values())) if _IMPORTED else setup_imports()
    return {'hf_hypernet': info['hf_hypernet_sha256'], 'scripts': info['scripts_sha256']}


def code_changes(recorded: dict | None, current: dict) -> list:
    """Files whose recorded SHA-256 differs from the current one (missing on either side counts)."""
    recorded = recorded or {}
    return sorted(name for name in set(recorded) | set(current) if recorded.get(name) != current.get(name))


def check_plan_code(plan: dict, where: str) -> None:
    """Refuse a panel plan validated with other hypernetwork or scoring-script code than this process imports."""
    hyak = plan.get('hyak', {})
    current = current_code_hashes()
    changed = code_changes(hyak.get('code_sha256'), current['hf_hypernet'])
    if 'scripts_sha256' not in hyak:
        changed.append('scripts/* (not recorded: the panel was validated by an older hyak.eval)')
    else:
        changed += code_changes(hyak['scripts_sha256'], current['scripts'])
    if changed:
        raise EvalError(f'{where} was validated with other evaluation code; changed: {", ".join(changed)}. '
                        'Restore those files (git checkout) or re-validate the panel '
                        '(python -m hyak.eval.validate_prefixes --force; scores of the old panel are then refused)')


def setup_imports(code_dir=None, repo: Path | None = None) -> dict:
    """Put ``<repo>/scripts`` and the pinned code tree at the front of ``sys.path``.

    Mirrors the Modal images: ``add_local_python_source('natural_transfer',
    'pool_generalization', 'uniform_trajectory', 'uniform_weakness', ...)`` and
    ``sys.path.insert(0, '/opt/zett')`` with ``/opt/zett/hf_hypernet`` =
    ``runtime/anytok_pool_20k/hf_hypernet`` (scripts/modal_random_pool_50.py:27-35,102).
    In that tree ``rust_utils/`` has no ``__init__.py`` and resolves as an empty
    namespace package, which is all ``hf_hypernet.joint_training`` needs for scoring
    (its ``import rust_utils`` at line 13; the scorers never call the sampler).

    Returns ``{'code_dir', 'scripts_dir', 'hf_hypernet_sha256', 'scripts_sha256'}``.  Idempotent.
    """
    repo = Path(repo) if repo is not None else repo_root()
    code_dir = Path(code_dir) if code_dir is not None else repo / CODE_TREE_REL
    key = (str(repo), str(code_dir))
    if key in _IMPORTED:
        return _IMPORTED[key]
    if _IMPORTED:
        previous = next(iter(_IMPORTED.values()))
        raise EvalError(f'setup_imports was already called for {previous["code_dir"]}; '
                        'one process may only use one code tree')
    hashes = verify_code_tree(code_dir, repo)
    scripts = repo / 'scripts'
    for path in (str(code_dir), str(scripts)):
        while path in sys.path:
            sys.path.remove(path)
    # scripts first (natural_transfer, pool_regime, ...), then the code tree (hf_hypernet, rust_utils).
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(scripts))
    loaded = sys.modules.get('hf_hypernet')
    if loaded is not None:
        origin = Path(getattr(loaded, '__file__', '') or '').parent
        if origin != code_dir / 'hf_hypernet':
            # Already imported from another tree in this process: acceptable only if it holds the
            # same pinned files (e.g. the RL snapshot's identical hf_hypernet); otherwise refuse.
            try:
                verify_code_tree(origin.parent, repo)
            except EvalError as exc:
                raise EvalError(f'hf_hypernet was already imported from {origin}, which is not the pinned code '
                                f'({exc})') from exc
    info = {'code_dir': str(code_dir), 'scripts_dir': str(scripts), 'hf_hypernet_sha256': hashes,
            'scripts_sha256': scripts_hashes(repo)}
    _IMPORTED[key] = info
    return info


def digest(value) -> str:
    """The repository's object hash (scripts/pool_generalization.py:15-16), imported from the original module."""
    from pool_generalization import digest as _digest
    return _digest(value)


def read_bundle(path: Path):
    """Parse a ``.json.gz`` or ``.json`` file."""
    import gzip
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


# --- inputs ------------------------------------------------------------------------------------
def model_input_hashes(repo: Path | None = None) -> dict:
    """``{'config.json': {...}, 'model.safetensors': {...}, 'tokenizer.json': {...}}`` from plans/zett-modal-20k-inputs.json."""
    repo = Path(repo) if repo is not None else repo_root()
    files = read_json(repo / MODEL_INPUTS_REL)['files']
    table = {name.split('/', 1)[1]: record for name, record in files.items() if name.startswith('model/')}
    if set(table) != {'config.json', 'model.safetensors', 'tokenizer.json'}:
        raise EvalError(f'{MODEL_INPUTS_REL} does not list exactly the three model files')
    if table['tokenizer.json']['sha256'] != CANONICAL_TOKENIZER_SHA256:
        raise EvalError(f'{MODEL_INPUTS_REL} changed: tokenizer.json hash differs from the pinned constant')
    return table


def verify_model_inputs(model_dir: Path, names=('tokenizer.json', 'model.safetensors', 'config.json'),
                        repo: Path | None = None, *, table: dict | None = None) -> dict:
    """Size + SHA-256 of ``input/model`` files against plans/zett-modal-20k-inputs.json.

    Port of the scorer check ``file_sha(root/'input'/name) != spec['input_hashes'][name]``
    (scripts/modal_random_pool_50.py:131-133); ``spec['input_hashes']`` of the
    original equals the model entries of plans/zett-modal-20k-inputs.json.
    ``table`` (``{name: {bytes, sha256}}``) is a test parameter only (miniature worlds
    with a tiny model); production always uses the pinned table.
    """
    table = model_input_hashes(repo) if table is None else table
    checked = {}
    for name in names:
        path = Path(model_dir) / name
        if not path.is_file():
            raise EvalError(f'{path} is missing; stage the model with python -m hyak.model.fetch_model')
        expected = table[name]
        size = path.stat().st_size
        if size != expected['bytes']:
            raise EvalError(f'{path} has {size} bytes, expected {expected["bytes"]} (wrong revision?); re-fetch it')
        actual = sha256_file(path)
        if actual != expected['sha256']:
            raise EvalError(f'{path} sha256 {actual} != {expected["sha256"]}; re-fetch the pinned SmolLM2 files')
        checked[name] = {'bytes': size, 'sha256': actual}
    return checked


def load_canonical(tokenizer_path: Path, *, require_pinned: bool = True):
    """The canonical tokenizer exactly as the panel builders and CPU workers loaded it.

    ``uniform_trajectory.CanonicalTokenizer`` (scripts/uniform_trajectory.py:131-144):
    ``tokenizers.Tokenizer.from_file``; ``decode`` passes only
    ``skip_special_tokens`` (default False) and ignores
    ``clean_up_tokenization_spaces``.  Used by original_pool_check.correct_panel
    (via modal_frozen_policy_cache.py:73), pool_regime.shorten (reduce_panel,
    modal_frozen_policy_cache.py:112) and natural_transfer.align_panel
    (modal_natural_transfer.py:91).
    """
    from uniform_trajectory import CanonicalTokenizer
    path = Path(tokenizer_path)
    if not path.is_file():
        raise EvalError(f'Canonical tokenizer {path} not found')
    actual = sha256_file(path)
    if require_pinned and actual != CANONICAL_TOKENIZER_SHA256:
        raise EvalError(f'{path} sha256 {actual} is not the pinned SmolLM2 tokenizer.json '
                        f'({CANONICAL_TOKENIZER_SHA256}); use input/model/tokenizer.json')
    canonical = CanonicalTokenizer(path)
    return canonical, actual


def load_pool_spec(pool_dir: Path, *, reference_plan: Path | None | bool = True, repo: Path | None = None) -> dict:
    """Index of the regenerated pool, in the shape of the original ``spec.json['tokenizers']``.

    The pool is read with the trainer's own reader
    ``hf_hypernet.pool_sampling.TokenizerPool`` (imported; it enforces 1,000
    complete entries whose configs equal ``plan.json`` and whose vocabulary sizes
    match).  Each entry becomes ``{'index', 'sha256', 'vocab_size', 'config'}`` as
    in ``reports/rl-pool-sweep-20k/spec.json`` (built at
    scripts/modal_eval_rl_pool.py:334-342), the fields the workers read.

    ``reference_plan`` (default: ``plans/tokenizer-pool-1000-v1.json``) must be
    byte-identical to ``<pool>/plan.json``; pass ``None`` (tests only) to skip.
    """
    from hf_hypernet.pool_sampling import TokenizerPool
    repo = Path(repo) if repo is not None else repo_root()
    pool_dir = Path(pool_dir)
    for name in ('full-results.json', 'plan.json'):
        if not (pool_dir / name).is_file():
            raise EvalError(f'{pool_dir / name} is missing; finish the pool and run python -m hyak.pool.collect')
    if reference_plan is True:
        reference_plan = repo / POOL_PLAN_REL
    if reference_plan:
        if _sha(pool_dir / 'plan.json') != _sha(reference_plan):
            raise EvalError(f'{pool_dir / "plan.json"} is not a byte copy of {reference_plan}')
    try:
        pool = TokenizerPool(pool_dir)
    except (ValueError, KeyError) as exc:
        raise EvalError(f'{pool_dir}: the trainer rejects this pool index ({exc}); rerun hyak.pool.collect') from exc
    tokenizers = [{'index': m['config']['index'], 'sha256': m['tokenizer_sha256'],
                   'vocab_size': m['vocab_size_actual'], 'config': m['config']} for m in pool.entries]
    return {'pool_dir': str(pool_dir), 'index_sha256': pool.index_hash, 'plan_sha256': pool.plan_hash,
            'tokenizers': tokenizers}


def tokenizer_path(pool_dir: Path, index: int) -> Path:
    from hyak.common.paths import pool_index_name
    return Path(pool_dir) / 'tokenizers' / pool_index_name(index) / 'tokenizer.json'


def check_tokenizer_file(pool_dir: Path, spec: dict, index: int) -> Path:
    """``file_sha(path) != spec['tokenizers'][index]['sha256']`` -> error (scripts/modal_random_pool_50.py:176-178)."""
    path = tokenizer_path(pool_dir, index)
    if not path.is_file():
        raise EvalError(f'{path} is missing (pool tokenizer {index})')
    expected = spec['tokenizers'][index]['sha256']
    actual = sha256_file(path)
    if actual != expected:
        raise EvalError(f'Native tokenizer changed: {path} sha256 {actual} != index {expected}')
    return path


def parse_indices(values, allowed) -> list:
    """Duplicate-free subset of ``allowed`` (scripts/modal_random_pool_50.py:114-116), in the given order."""
    if values is None:
        return list(allowed)
    work = []
    for value in values:
        for part in str(value).split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                low, high = (int(x) for x in part.split('-', 1))
                work.extend(range(low, high + 1))
            else:
                work.append(int(part))
    if len(set(work)) != len(work) or not set(work) <= set(allowed):
        raise EvalError('Invalid index list: indices must be unique and belong to the plan '
                        f'({len(allowed)} allowed)')
    return work


# --- storage layout -------------------------------------------------------------------------------
class EvalLayout:
    """Files of this package below ``$ANYTOK_ROOT/eval`` (see the module docstring)."""

    def __init__(self, paths):
        self.paths = paths

    @property
    def root(self) -> Path:
        return self.paths.eval_root

    def panel_dir(self, panel: str) -> Path:
        return self.paths.eval_panels / validate_panel_name(panel)

    def rebuilt(self, panel):
        return self.panel_dir(panel) / 'rebuilt.json'

    def rebuild_report(self, panel):
        return self.panel_dir(panel) / 'rebuild-report.json'

    def panel(self, panel):
        return self.panel_dir(panel) / 'panel.json'

    def plan(self, panel):
        return self.panel_dir(panel) / 'plan.json'

    def preparation(self, panel):
        return self.panel_dir(panel) / 'preparation.json'

    def validation_report(self, panel):
        return self.panel_dir(panel) / 'validation-report.json'

    def feature_dir(self, index: int) -> Path:
        return self.paths.eval_feature_dir(index)

    def scores_dir(self, run: str, panel: str) -> Path:
        return self.paths.eval_scores_dir(run) / validate_panel_name(panel)

    def claims(self, task: str) -> Path:
        return self.root / 'claims' / task

    def report_dir(self, panel: str) -> Path:
        return self.paths.eval_reports / validate_panel_name(panel)


# --- atomic-mkdir claims ------------------------------------------------------------------------------
def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


class Claims:
    """Work claims as atomic directories ``<root>/<name>/`` with ``owner.json``.

    Same semantics as the pool workers (DESIGN.md 5.F): a claim is created with
    ``mkdir`` (atomic on POSIX and GPFS); the owner file records job id, host,
    pid and start time.  A claim may be taken over when its owner is gone:
    the pid no longer exists, judged only for a claim of the same job on the same
    host or of a non-Slurm ``local-`` process on the same host (other Slurm jobs may
    run in their own PID namespace, so their pids are never tested); or a Slurm job
    that ``squeue`` no longer lists as alive (``hyak.common.slurm.is_job_alive`` is
    False); or the claim is older than ``timeout_seconds`` (also when liveness is unknown).

    Takeover happens under a per-name lock directory ``.takeover-<name>`` (``mkdir``):
    inside it the owner is read again and the claim is renamed away only if it is
    still the one judged stale; the renamed copy is checked once more and put back if
    it is not.  So of several jobs that find the same stale claim at the same time,
    at most one removes it and no live claim is ever renamed away.  Then the taker
    claims afresh with ``mkdir`` like anyone else.
    """

    OWNER = 'owner.json'
    #: A takeover lock older than this (seconds) is left over from a killed process and removed.
    TAKEOVER_LOCK_SECONDS = 600.0

    def __init__(self, root: Path, timeout_seconds: float, *, job=None, host=None, pid=None,
                 alive=slurm.is_job_alive, clock=time.time):
        self.root = Path(root)
        self.timeout = float(timeout_seconds)
        self.job = str(job) if job is not None else slurm.worker_id()
        self.host = host or slurm.node_name() or socket.gethostname()
        self.pid = int(pid) if pid is not None else os.getpid()
        self._alive = alive
        self._clock = clock
        self.takeovers = []
        self.lost = []          # names whose claim another job had taken over by the time we released it

    def path(self, name: str) -> Path:
        return self.root / name

    def owner(self, name: str):
        try:
            return read_json(self.path(name) / self.OWNER)
        except (OSError, ValueError):
            return None

    def _write_owner(self, name, task):
        info = {'job_id': self.job, 'host': self.host, 'pid': self.pid, 'task': task,
                'claimed_at': utc_now(), 'claimed_unix': self._clock()}
        atomic_write_json(self.path(name) / self.OWNER, info)
        return info

    def is_mine(self, name: str) -> bool:
        info = self.owner(name)
        return bool(info) and info.get('job_id') == self.job and info.get('host') == self.host \
            and info.get('pid') == self.pid

    _UNREAD = object()

    def stale_reason(self, name: str, info=_UNREAD):
        """Why the existing claim may be taken over, or None if its owner may still be working.

        ``info`` is the owner record already read by the caller (default: read it here).
        """
        path = self.path(name)
        if info is Claims._UNREAD:
            info = self.owner(name)
        now = self._clock()
        if info is None:
            try:
                age = now - path.stat().st_mtime
            except FileNotFoundError:
                return 'vanished'
            # mkdir happened but owner.json is not written yet: give the owner a grace period.
            return f'no owner file for {age:.0f} s' if age > min(600.0, self.timeout) else None
        age = now - float(info.get('claimed_unix') or 0)
        job = str(info.get('job_id', ''))
        # The pid test only where pids are comparable: this job's own claims, or non-Slurm processes, on this host.
        same_namespace = job == self.job or job.startswith('local-')
        if same_namespace and info.get('host') == self.host and not _pid_alive(info.get('pid')):
            return f'owner pid {info.get("pid")} on {self.host} is gone'
        if job and not job.startswith('local-') and job != self.job:
            alive = self._alive(job)
            if alive is False:
                return f'owner job {job} is no longer alive (squeue)'
        if age > self.timeout:
            return f'claim is {age:.0f} s old (timeout {self.timeout:.0f} s)'
        return None

    def _take_over(self, name: str, info, reason: str) -> bool:
        """Remove the stale claim ``info`` of ``name`` under the takeover lock.  True if it is gone now."""
        lock = self.root / f'.takeover-{name}'
        try:
            os.mkdir(lock)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > self.TAKEOVER_LOCK_SECONDS:
                    os.rmdir(lock)          # left by a killed process; the next attempt may take over
            except OSError:
                pass
            return False
        try:
            if self.owner(name) != info:
                return False                # changed since it was judged stale (another taker, or a new owner)
            graveyard = self.root / f'.{name}.stale-{uuid.uuid4().hex}'
            try:
                os.rename(self.path(name), graveyard)
            except FileNotFoundError:
                return True                 # already gone
            except OSError:
                return False
            previous = self._read_dir(graveyard)
            if previous != info:
                try:                        # not the claim judged stale: put it back (best effort)
                    os.rename(graveyard, self.path(name))
                except OSError:
                    pass
                return False
            self.takeovers.append({'name': name, 'reason': reason, 'previous': previous})
            shutil.rmtree(graveyard, ignore_errors=True)
            return True
        finally:
            try:
                os.rmdir(lock)
            except OSError:
                pass

    def try_acquire(self, name: str, task: str = '') -> bool:
        """Claim ``name``; True on success.  Takes over a stale claim once."""
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                os.mkdir(self.path(name))
            except FileExistsError:
                info = self.owner(name)
                reason = self.stale_reason(name, info)
                if reason is None:
                    return False
                if reason != 'vanished' and not self._take_over(name, info, reason):
                    return False
                continue
            try:
                self._write_owner(name, task)
            except FileNotFoundError:
                return False                # the fresh directory was removed before its owner file was written
            return True
        return False

    @staticmethod
    def _read_dir(path):
        try:
            return read_json(Path(path) / Claims.OWNER)
        except (OSError, ValueError):
            return None

    def release(self, name: str) -> None:
        """Remove our claim (only if we still own it)."""
        path = self.path(name)
        if not path.exists():
            return
        if not self.is_mine(name):
            raise EvalError(f'claim {path} is no longer ours (owner {self.owner(name)}); '
                            'another job took it over, so two jobs may have done the same work')
        shutil.rmtree(path)

    def release_after_work(self, name: str) -> bool:
        """Release once the claimed work is safely written: a lost claim is only logged (True if released).

        Results are written with :func:`create_once`, so the first result stays; a claim that another
        job took over meanwhile means duplicated work, not wrong work, and must not fail this job.
        """
        try:
            self.release(name)
            return True
        except EvalError as exc:
            log(f'WARNING: {exc}')
            self.lost.append(name)
            return False

    def live_names(self) -> list:
        try:
            return sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        except FileNotFoundError:
            return []


def create_once(path: Path, obj, *, indent=2) -> bool:
    """Write ``obj`` as JSON at ``path`` only if it does not exist yet (atomic, race-safe).

    Returns True if this call created it.  Uses a fsynced temporary file plus
    ``os.link`` (fails with FileExistsError if another process won), so readers
    never see a partial file and an existing file is never replaced.
    """
    from hyak.common.io import fsync_dir, json_text
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        stream.write(json_text(obj, indent=indent))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        fsync_dir(path.parent)
        return True
    except FileExistsError:
        return False
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def remaining_seconds(default=float('inf')) -> float:
    """Seconds until this Slurm job's limit (inf outside Slurm)."""
    value = slurm.seconds_remaining(default)
    return float('inf') if value is None else float(value)
