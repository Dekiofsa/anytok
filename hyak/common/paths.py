"""Single source of truth for storage locations, stage names and dashboard run ids.

No other module hard-codes these names: build a :class:`Paths` with
``Paths.from_config(load_config())`` and use its attributes.

Layout (DESIGN.md section 3)::

    $ANYTOK_ROOT/                         persistent project storage
      env/train, env/data                 conda prefixes (+ env/<name>.freeze.txt)
      toolchain/{rustup,cargo}            rustup + cargo homes
      build/{tokenizers-src,rust_utils}   out-of-repo build trees
      wheels/                             patched tokenizers wheel + provenance.json
      hf-cache/                           HF_HOME
      code/uniform, code/rl               staged code trees; code/code_manifest.json
      corpus/<size>-v1/<bucket>/          size in full, extra, specialty
      input/recipe.json, input/pool-recipe.json
      input/corpus/<bucket>/train.jsonl.gz + input/corpus/manifest.json
      input/eval/<bucket>.jsonl.gz
      input/model/, input/hypernet_init/
      input/pool/{plan.json,full-results.json,tokenizers/NNNN/,claims/NNNN/}
      input/uniform_run_config.json, input/uniform_heldout_tokenizers.json, input/pairing.json
      input/manifest.json
      eval/{panels,features,scores,reports}/
    $ANYTOK_RUNS/
      uniform/ rl/                        trainer --output directories
      probe/ smoke/<run>/                 scratch
    <repo>/modal_results/hyak_tillicum/<stage>/
      status.json events.jsonl slurm-<jobid>.out latest-slurm.out <jobid>-train.log workers/

Paths are normalised with ``os.path.normpath`` but symlinks are never
resolved: the trainer compares path strings on resume, so the same configured
string must give the same path in every job.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

# --- Stage names (also the monitor directory names and submit.sh stage names) ---
STAGE_CHECK_ENV = 'check-env'
STAGE_PROBE = 'probe'
STAGE_CORPUS = 'corpus'
STAGE_WHEEL = 'wheel'
STAGE_POOL = 'pool'
STAGE_STAGE_INPUTS = 'stage-inputs'
STAGE_INIT = 'init'
STAGE_TRAIN_UNIFORM = 'train-uniform'
STAGE_TRAIN_RL = 'train-rl'
STAGE_EVAL_PREPARE = 'eval-prepare'
STAGE_EVAL_UNIFORM = 'eval-uniform'
STAGE_EVAL_RL = 'eval-rl'
STAGE_REPLAY_CHECK = 'replay-check'
STAGE_SMOKE_TRAIN = 'smoke-train'
#: The raw-BPB handoff run (Qingyun's launcher scripts/run_raw_bpb_training.py on his input bundle; README 0).
STAGE_SMOKE_RAWBPB = 'smoke-rawbpb'
STAGE_TRAIN_RAWBPB = 'train-rawbpb'

#: Every stage, in pipeline order.  Must equal ``HYAK_STAGES`` in hyak/slurm/lib.sh (tested).
STAGES = (
    STAGE_CHECK_ENV, STAGE_PROBE, STAGE_CORPUS, STAGE_WHEEL, STAGE_POOL, STAGE_STAGE_INPUTS,
    STAGE_INIT, STAGE_TRAIN_UNIFORM, STAGE_TRAIN_RL, STAGE_EVAL_PREPARE, STAGE_EVAL_UNIFORM,
    STAGE_EVAL_RL, STAGE_REPLAY_CHECK, STAGE_SMOKE_TRAIN, STAGE_SMOKE_RAWBPB, STAGE_TRAIN_RAWBPB,
)

#: Stages submitted to the debug QoS and debug time limit by default (Tillicum debug: <= 1 h, 1 GPU).
DEBUG_STAGES = (STAGE_CHECK_ENV, STAGE_PROBE, STAGE_SMOKE_RAWBPB)

#: Stages chained with ``--dependency=singleton`` (D14): only one job per name runs at a time.
SINGLETON_STAGES = (STAGE_TRAIN_UNIFORM, STAGE_TRAIN_RL, STAGE_TRAIN_RAWBPB)

#: Stages for which ``submit.sh --count N`` with N > 1 is allowed: claim-based work
#: distribution (pool, eval-*) or singleton chains (train-*).  Every other stage is one-shot.
MULTI_JOB_STAGES = (STAGE_POOL, STAGE_EVAL_PREPARE, STAGE_EVAL_UNIFORM, STAGE_EVAL_RL,
                    STAGE_TRAIN_UNIFORM, STAGE_TRAIN_RL, STAGE_TRAIN_RAWBPB)

#: Training runs (also run directory names under $ANYTOK_RUNS and code trees under code/).
RUN_UNIFORM = 'uniform'
RUN_RL = 'rl'
RUNS = (RUN_UNIFORM, RUN_RL)

#: Stages bound to one run; submit.sh exports HYAK_RUN=<run> for them.
STAGE_RUN = {
    STAGE_TRAIN_UNIFORM: RUN_UNIFORM, STAGE_TRAIN_RL: RUN_RL,
    STAGE_EVAL_UNIFORM: RUN_UNIFORM, STAGE_EVAL_RL: RUN_RL,
}

#: sbatch script of each stage, relative to hyak/slurm/.  Must equal the table in submit.sh (tested).
SBATCH_FILES = {
    STAGE_CHECK_ENV: '00_check_env.sbatch',
    STAGE_PROBE: '01_probe.sbatch',
    STAGE_CORPUS: '02_corpus.sbatch',
    STAGE_WHEEL: '03_wheel.sbatch',
    STAGE_POOL: '04_pool.sbatch',
    STAGE_STAGE_INPUTS: '05_stage_inputs.sbatch',
    STAGE_INIT: '06_init.sbatch',
    STAGE_TRAIN_UNIFORM: '07_train.sbatch',
    STAGE_TRAIN_RL: '07_train.sbatch',
    STAGE_EVAL_PREPARE: '08_eval_prepare.sbatch',
    STAGE_EVAL_UNIFORM: '09_eval_score.sbatch',
    STAGE_EVAL_RL: '09_eval_score.sbatch',
    STAGE_REPLAY_CHECK: '10_replay_check.sbatch',
    STAGE_SMOKE_TRAIN: '11_smoke_train.sbatch',
    STAGE_SMOKE_RAWBPB: '12_smoke_rawbpb.sbatch',
    STAGE_TRAIN_RAWBPB: '13_train_rawbpb.sbatch',
}

#: Dashboard registry id of each long-running stage (D13).  check-env, wheel, stage-inputs,
#: init, replay-check and smoke-train are short or diagnostic: they write status.json and
#: events.jsonl in their monitor directory but get no registry entry.  eval-prepare (feature
#: preparation for all 1,000 tokenizers, several claim-based jobs) is evaluation work and is
#: registered because AGENTS.md requires long evaluation jobs to be visible.
#: The raw-BPB handoff stages (train-rawbpb, smoke-rawbpb) are registered first among the managed entries:
#: their status is the launcher's own OUTPUT/status.json, read through the link <monitor>/<stage>/run.
DASHBOARD_RUN_IDS = {
    STAGE_TRAIN_RAWBPB: 'hyak-raw-bpb-training',
    STAGE_SMOKE_RAWBPB: 'hyak-raw-bpb-smoke',
    STAGE_TRAIN_UNIFORM: 'hyak-uniform-training',
    STAGE_TRAIN_RL: 'hyak-rl-training',
    STAGE_EVAL_UNIFORM: 'hyak-eval-uniform',
    STAGE_EVAL_RL: 'hyak-eval-rl',
    STAGE_EVAL_PREPARE: 'hyak-eval-prepare',
    STAGE_POOL: 'hyak-tokenizer-pool',
    STAGE_CORPUS: 'hyak-corpus',
    STAGE_PROBE: 'hyak-memory-probe',
}

#: Number of training updates of each run (unchanged from the original runs).
FINAL_STEP = 20000

#: Progress meta of each stage's status.json: the key holding the numerator, the fixed total
#: (None = the stage writes ``total`` itself) and the unit.  Used by StatusWriter defaults and
#: by the dashboard registry entries.
STAGE_PROGRESS = {
    STAGE_CHECK_ENV: {'completed_field': 'completed', 'total': None, 'unit': 'checks'},
    STAGE_PROBE: {'completed_field': 'completed', 'total': None, 'unit': 'measurements'},
    STAGE_CORPUS: {'completed_field': 'completed', 'total': 23, 'unit': 'buckets'},
    STAGE_WHEEL: {'completed_field': 'completed', 'total': None, 'unit': 'steps'},
    STAGE_POOL: {'completed_field': 'completed', 'total': 1000, 'unit': 'tokenizers'},
    STAGE_STAGE_INPUTS: {'completed_field': 'completed', 'total': None, 'unit': 'files'},
    STAGE_INIT: {'completed_field': 'completed', 'total': None, 'unit': 'steps'},
    STAGE_TRAIN_UNIFORM: {'completed_field': 'step', 'total': FINAL_STEP, 'unit': 'steps'},
    STAGE_TRAIN_RL: {'completed_field': 'step', 'total': FINAL_STEP, 'unit': 'steps'},
    STAGE_EVAL_PREPARE: {'completed_field': 'completed', 'total': None, 'unit': 'tokenizers'},
    STAGE_EVAL_UNIFORM: {'completed_field': 'completed', 'total': None, 'unit': 'tokenizers'},
    STAGE_EVAL_RL: {'completed_field': 'completed', 'total': None, 'unit': 'tokenizers'},
    STAGE_REPLAY_CHECK: {'completed_field': 'completed', 'total': None, 'unit': 'updates'},
    STAGE_SMOKE_TRAIN: {'completed_field': 'step', 'total': None, 'unit': 'steps'},
    STAGE_SMOKE_RAWBPB: {'completed_field': 'step', 'total': 3, 'unit': 'steps'},
    STAGE_TRAIN_RAWBPB: {'completed_field': 'step', 'total': FINAL_STEP, 'unit': 'steps'},
}

#: Corpus pool sizes (directory corpus/<size>-v1).
CORPUS_POOL_SIZES = ('full', 'extra', 'specialty')

#: Monitor directory root relative to the repository (the dashboard's globs must be repo-relative).
MONITOR_REL = 'modal_results/hyak_tillicum'

#: Environment variable that redirects monitor directories (tests only; the dashboard reads
#: only <repo>/modal_results/hyak_tillicum).
MONITOR_ROOT_ENV = 'HYAK_MONITOR_ROOT'

_BUCKET_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
_JOB_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.+-]*')


def validate_stage(stage: str) -> str:
    """Return ``stage`` if it is a known stage name, else raise ValueError naming the valid ones."""
    if stage not in STAGES:
        raise ValueError(f'Unknown stage {stage!r}; expected one of: {", ".join(STAGES)}')
    return stage


def validate_run(run: str) -> str:
    """Return ``run`` if it is ``'uniform'`` or ``'rl'``, else raise ValueError."""
    if run not in RUNS:
        raise ValueError(f'Unknown run {run!r}; expected one of: {", ".join(RUNS)}')
    return run


def stage_suffix(stage: str) -> str:
    """Config-key suffix of a stage: ``'train-uniform'`` -> ``'TRAIN_UNIFORM'``."""
    return validate_stage(stage).upper().replace('-', '_')


def pool_index_name(index: int) -> str:
    """Four-digit directory name of a pool index (``7`` -> ``'0007'``); index must be 0..9999."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 9999:
        raise ValueError(f'Pool index must be an integer in 0..9999, got {index!r}')
    return f'{index:04d}'


def _bucket(name: str) -> str:
    if not isinstance(name, str) or not _BUCKET_RE.fullmatch(name):
        raise ValueError(f'Invalid bucket name {name!r}')
    return name


def _absolute(value: str, key: str) -> Path:
    if not value or not os.path.isabs(value):
        raise ValueError(f'{key} must be an absolute path, got {value!r}; edit hyak/config.env')
    return Path(os.path.normpath(value))


def monitor_rel(stage: str) -> str:
    """Repo-relative POSIX path of a stage's monitor directory, for dashboard registry entries."""
    return f'{MONITOR_REL}/{validate_stage(stage)}'


def default_monitor_root(repo: Path | None = None) -> Path:
    """Monitor root: ``$HYAK_MONITOR_ROOT`` if set (tests only), else ``<repo>/modal_results/hyak_tillicum``."""
    override = os.environ.get(MONITOR_ROOT_ENV)
    if override:
        return _absolute(override, MONITOR_ROOT_ENV)
    if repo is None:
        from hyak.common.config import repo_root
        repo = repo_root()
    return Path(repo) / MONITOR_REL


def default_monitor_dir(stage: str, repo: Path | None = None) -> Path:
    """``<monitor root>/<stage>`` without needing a loaded config."""
    return default_monitor_root(repo) / validate_stage(stage)


@dataclass(frozen=True)
class Paths:
    """Every directory and file of the cluster layout (see the module docstring).

    Fields: ``repo`` (source checkout), ``root`` (``$ANYTOK_ROOT``), ``runs``
    (``$ANYTOK_RUNS``), ``monitor_root`` (dashboard-visible status files).
    All other locations are properties or methods derived from these.
    """

    repo: Path
    root: Path
    runs: Path
    monitor_root: Path

    @classmethod
    def from_config(cls, cfg: dict, *, repo: Path | None = None, monitor_root: Path | None = None) -> 'Paths':
        """Build from a ``load_config()`` dict.

        ``repo`` defaults to ``repo_root()``; ``monitor_root`` defaults to
        ``$HYAK_MONITOR_ROOT`` (tests) or ``<repo>/modal_results/hyak_tillicum``.
        Raises ValueError if ANYTOK_ROOT / ANYTOK_RUNS are missing or relative.
        """
        if repo is None:
            from hyak.common.config import repo_root
            repo = repo_root()
        repo = Path(repo)
        root = _absolute(cfg.get('ANYTOK_ROOT', ''), 'ANYTOK_ROOT')
        runs = _absolute(cfg.get('ANYTOK_RUNS', ''), 'ANYTOK_RUNS')
        monitor = Path(monitor_root) if monitor_root is not None else default_monitor_root(repo)
        return cls(repo=repo, root=root, runs=runs, monitor_root=monitor)

    # --- environments and toolchains --------------------------------------------
    @property
    def env_root(self) -> Path:
        """``$ANYTOK_ROOT/env``."""
        return self.root / 'env'

    @property
    def env_train(self) -> Path:
        """Conda prefix of the training/evaluation environment (six pins + pytest)."""
        return self.env_root / 'train'

    @property
    def env_data(self) -> Path:
        """Conda prefix of the data environment (datasets, maturin, patched tokenizers)."""
        return self.env_root / 'data'

    def env_prefix(self, which: str) -> Path:
        """``env/train`` or ``env/data``."""
        if which not in ('train', 'data'):
            raise ValueError(f"Environment must be 'train' or 'data', got {which!r}")
        return self.env_root / which

    def env_freeze(self, which: str) -> Path:
        """``env/<which>.freeze.txt`` (pip freeze written by the setup scripts)."""
        return self.env_root / f'{self.env_prefix(which).name}.freeze.txt'

    @property
    def toolchain(self) -> Path:
        """``$ANYTOK_ROOT/toolchain`` (rustup + cargo homes)."""
        return self.root / 'toolchain'

    @property
    def rustup_home(self) -> Path:
        """``toolchain/rustup`` (RUSTUP_HOME)."""
        return self.toolchain / 'rustup'

    @property
    def cargo_home(self) -> Path:
        """``toolchain/cargo`` (CARGO_HOME)."""
        return self.toolchain / 'cargo'

    @property
    def build_root(self) -> Path:
        """``$ANYTOK_ROOT/build`` (out-of-repo build trees; never write target/ inside the repo)."""
        return self.root / 'build'

    @property
    def tokenizers_src(self) -> Path:
        """``build/tokenizers-src`` (patched huggingface/tokenizers checkout)."""
        return self.build_root / 'tokenizers-src'

    @property
    def rust_utils_build(self) -> Path:
        """``build/rust_utils`` (fallback rebuild of librust_utils.so)."""
        return self.build_root / 'rust_utils'

    @property
    def wheels(self) -> Path:
        """``$ANYTOK_ROOT/wheels`` (patched tokenizers wheel)."""
        return self.root / 'wheels'

    @property
    def wheel_provenance(self) -> Path:
        """``wheels/provenance.json``."""
        return self.wheels / 'provenance.json'

    @property
    def hf_cache(self) -> Path:
        """``$ANYTOK_ROOT/hf-cache`` (HF_HOME)."""
        return self.root / 'hf-cache'

    # --- staged code ----------------------------------------------------------------
    @property
    def code_root(self) -> Path:
        """``$ANYTOK_ROOT/code``."""
        return self.root / 'code'

    def code_dir(self, run: str) -> Path:
        """Staged code tree of a run: ``code/uniform`` or ``code/rl``."""
        return self.code_root / validate_run(run)

    @property
    def code_manifest(self) -> Path:
        """``code/code_manifest.json``."""
        return self.code_root / 'code_manifest.json'

    # --- corpus pools -----------------------------------------------------------------
    @property
    def corpus_root(self) -> Path:
        """``$ANYTOK_ROOT/corpus``."""
        return self.root / 'corpus'

    def corpus_pool_dir(self, size: str) -> Path:
        """``corpus/<size>-v1`` for size in full, extra, specialty."""
        if size not in CORPUS_POOL_SIZES:
            raise ValueError(f'Corpus size must be one of {CORPUS_POOL_SIZES}, got {size!r}')
        return self.corpus_root / f'{size}-v1'

    def corpus_bucket_dir(self, size: str, bucket: str) -> Path:
        """``corpus/<size>-v1/<bucket>`` (train.jsonl.gz, eval.jsonl.gz, manifest.json)."""
        return self.corpus_pool_dir(size) / _bucket(bucket)

    # --- staged inputs ----------------------------------------------------------------
    @property
    def input(self) -> Path:
        """``$ANYTOK_ROOT/input``."""
        return self.root / 'input'

    @property
    def recipe(self) -> Path:
        """``input/recipe.json`` (byte copy of data/balanced-full-v3.json)."""
        return self.input / 'recipe.json'

    @property
    def pool_recipe(self) -> Path:
        """``input/pool-recipe.json`` (recipe with source_path rewritten for tokenizer generation)."""
        return self.input / 'pool-recipe.json'

    @property
    def input_corpus(self) -> Path:
        """``input/corpus`` (the trainer's --data-dir)."""
        return self.input / 'corpus'

    def input_corpus_bucket(self, bucket: str) -> Path:
        """``input/corpus/<bucket>``."""
        return self.input_corpus / _bucket(bucket)

    def input_corpus_train(self, bucket: str) -> Path:
        """``input/corpus/<bucket>/train.jsonl.gz``."""
        return self.input_corpus_bucket(bucket) / 'train.jsonl.gz'

    @property
    def input_corpus_manifest(self) -> Path:
        """``input/corpus/manifest.json`` (exactly the 23 bucket keys; the trainer's --data-manifest)."""
        return self.input_corpus / 'manifest.json'

    @property
    def input_eval(self) -> Path:
        """``input/eval`` (the trainer's --eval-dir)."""
        return self.input / 'eval'

    def input_eval_file(self, bucket: str) -> Path:
        """``input/eval/<bucket>.jsonl.gz``."""
        return self.input_eval / f'{_bucket(bucket)}.jsonl.gz'

    @property
    def model_dir(self) -> Path:
        """``input/model`` (config.json, model.safetensors, tokenizer.json of SmolLM2-1.7B)."""
        return self.input / 'model'

    @property
    def hypernet_init(self) -> Path:
        """``input/hypernet_init`` (random-init hypernetwork; --warmup-checkpoint of both runs)."""
        return self.input / 'hypernet_init'

    @property
    def hypernet_init_manifest(self) -> Path:
        """``input/hypernet_init/init_manifest.json``."""
        return self.hypernet_init / 'init_manifest.json'

    @property
    def pool_dir(self) -> Path:
        """``input/pool`` (the trainer's --tokenizer-pool)."""
        return self.input / 'pool'

    @property
    def pool_plan(self) -> Path:
        """``input/pool/plan.json`` (byte copy of plans/tokenizer-pool-1000-v1.json)."""
        return self.pool_dir / 'plan.json'

    @property
    def pool_results(self) -> Path:
        """``input/pool/full-results.json`` (written once by hyak.pool.collect)."""
        return self.pool_dir / 'full-results.json'

    @property
    def pool_tokenizers(self) -> Path:
        """``input/pool/tokenizers``."""
        return self.pool_dir / 'tokenizers'

    def pool_tokenizer_dir(self, index: int) -> Path:
        """``input/pool/tokenizers/NNNN`` (tokenizer.json, manifest.json)."""
        return self.pool_tokenizers / pool_index_name(index)

    @property
    def pool_claims(self) -> Path:
        """``input/pool/claims``."""
        return self.pool_dir / 'claims'

    def pool_claim_dir(self, index: int) -> Path:
        """``input/pool/claims/NNNN`` (atomic mkdir claim)."""
        return self.pool_claims / pool_index_name(index)

    @property
    def uniform_run_config(self) -> Path:
        """``input/uniform_run_config.json`` (frozen copy; the RL trainer reads <pool>/../uniform_run_config.json)."""
        return self.input / 'uniform_run_config.json'

    @property
    def uniform_heldout(self) -> Path:
        """``input/uniform_heldout_tokenizers.json`` (the RL run's --eval-tokenizers)."""
        return self.input / 'uniform_heldout_tokenizers.json'

    @property
    def pairing(self) -> Path:
        """``input/pairing.json``."""
        return self.input / 'pairing.json'

    @property
    def input_manifest(self) -> Path:
        """``input/manifest.json`` (bytes + sha256 of every staged input)."""
        return self.input / 'manifest.json'

    # --- evaluation ---------------------------------------------------------------------
    @property
    def eval_root(self) -> Path:
        """``$ANYTOK_ROOT/eval``."""
        return self.root / 'eval'

    @property
    def eval_panels(self) -> Path:
        """``eval/panels``."""
        return self.eval_root / 'panels'

    @property
    def eval_features(self) -> Path:
        """``eval/features``."""
        return self.eval_root / 'features'

    def eval_feature_dir(self, index: int) -> Path:
        """``eval/features/NNNN`` (forms.npy, priors.npy, manifest.json)."""
        return self.eval_features / pool_index_name(index)

    @property
    def eval_scores(self) -> Path:
        """``eval/scores``."""
        return self.eval_root / 'scores'

    def eval_scores_dir(self, run: str) -> Path:
        """``eval/scores/<run>`` (layout below it belongs to hyak.eval)."""
        return self.eval_scores / validate_run(run)

    @property
    def eval_reports(self) -> Path:
        """``eval/reports``."""
        return self.eval_root / 'reports'

    # --- run directories ------------------------------------------------------------------
    def run_dir(self, run: str) -> Path:
        """Trainer --output directory: ``$ANYTOK_RUNS/uniform`` or ``$ANYTOK_RUNS/rl``."""
        return self.runs / validate_run(run)

    @property
    def probe_dir(self) -> Path:
        """``$ANYTOK_RUNS/probe`` (memory probe scratch)."""
        return self.runs / 'probe'

    @property
    def smoke_root(self) -> Path:
        """``$ANYTOK_RUNS/smoke``."""
        return self.runs / 'smoke'

    def smoke_dir(self, run: str) -> Path:
        """``$ANYTOK_RUNS/smoke/<run>`` (``run_segment --smoke`` output)."""
        return self.smoke_root / validate_run(run)

    # --- the raw-BPB handoff (README section 0; hyak/handoff) ------------------------------------
    @property
    def handoff_dir(self) -> Path:
        """``$ANYTOK_ROOT/handoff``: the downloaded bundle archive, its .sha256, its receipt, and
        ``inputs-verified.json`` (written by hyak/handoff/fetch_inputs.sh after the full verification)."""
        return self.root / 'handoff'

    @property
    def handoff_receipt(self) -> Path:
        """``handoff/inputs-verified.json``: the record that ``$ANYTOK_ROOT/input`` is the verified bundle."""
        return self.handoff_dir / 'inputs-verified.json'

    @property
    def bundle_manifest(self) -> Path:
        """``$ANYTOK_ROOT/BUNDLE_MANIFEST.json`` (extracted next to ``input/`` from the bundle archive)."""
        return self.root / 'BUNDLE_MANIFEST.json'

    # --- dashboard-visible monitor files ----------------------------------------------------
    def monitor_dir(self, stage: str) -> Path:
        """``<repo>/modal_results/hyak_tillicum/<stage>`` (git-ignored)."""
        return self.monitor_root / validate_stage(stage)

    @staticmethod
    def monitor_rel(stage: str) -> str:
        """Repo-relative twin of :meth:`monitor_dir` for registry entries."""
        return monitor_rel(stage)

    def status_file(self, stage: str) -> Path:
        """``<monitor_dir>/status.json`` (atomic; read by the progress_json adapter)."""
        return self.monitor_dir(stage) / 'status.json'

    def events_file(self, stage: str) -> Path:
        """``<monitor_dir>/events.jsonl`` (timestamped JSON lines)."""
        return self.monitor_dir(stage) / 'events.jsonl'

    def workers_dir(self, stage: str) -> Path:
        """``<monitor_dir>/workers`` (one file per live job; merged into status.json ``workers``)."""
        return self.monitor_dir(stage) / 'workers'

    def slurm_output_pattern(self, stage: str) -> Path:
        """``<monitor_dir>/slurm-%j.out`` (sbatch --output, set by submit.sh)."""
        return self.monitor_dir(stage) / 'slurm-%j.out'

    def slurm_log(self, stage: str, job_id: str) -> Path:
        """``<monitor_dir>/slurm-<jobid>.out``."""
        return self.monitor_dir(stage) / f'slurm-{_job(job_id)}.out'

    def latest_slurm_log(self, stage: str) -> Path:
        """``<monitor_dir>/latest-slurm.out`` (symlink kept by lib.sh hyak_log_dir)."""
        return self.monitor_dir(stage) / 'latest-slurm.out'

    def train_log(self, stage: str, job_id: str) -> Path:
        """``<monitor_dir>/<jobid>-train.log``: trainer stdout of one job.

        The learning panel globs ``*-train.log`` in the train-rl / train-uniform
        monitor directories, so only trainer JSON lines belong in such files.
        """
        return self.monitor_dir(stage) / f'{_job(job_id)}-train.log'


def _job(job_id: str) -> str:
    job_id = str(job_id)
    if not _JOB_RE.fullmatch(job_id):
        raise ValueError(f'Invalid job id {job_id!r}')
    return job_id
