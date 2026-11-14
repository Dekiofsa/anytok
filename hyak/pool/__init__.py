"""The 1,000-tokenizer pool on Slurm (DESIGN.md D4, D9 and section 5.F).

Entry points:

* ``hyak/pool/build_wheel.sh fetch|build|install`` -- the patched ``tokenizers`` 0.22.2 wheel.
* ``python -m hyak.pool.worker``    -- claim-based generation of the 1,000 tokenizers.
* ``python -m hyak.pool.collect``   -- the one-time pool index ``full-results.json`` + ``plan.json``.
* ``python -m hyak.pool.scorecard`` -- three-level fingerprint against the original pool.

This module holds the pinned identities and the checks shared by those tools
(standard library only, importable on any Python >= 3.9).  The plan is
always read with Python's ``json`` (its seeds exceed 2**53, so any parser that
uses doubles would corrupt them).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# --- upstream source and patches (scripts/build_fast_tokenizers.py:17,29-30) ---------------
UPSTREAM_URL = 'https://github.com/huggingface/tokenizers.git'
#: scripts/build_fast_tokenizers.py:17.  Verified 2026-11-13 with ``git ls-remote``: this is
#: the commit of the lightweight tag ``v0.22.2`` (2026-01-05, "fix max build?").
UPSTREAM_REVISION = 'f383101a26663708484cac0727792aad74f78234'
UPSTREAM_TAG = 'v0.22.2'
#: Applied in this order (scripts/build_fast_tokenizers.py:29).  Never merge-noise.
PATCH_NAMES = ('all-candidate-noise', 'unigram-literal-unk', 'unigram-stack-drop')
PATCH_FILES = tuple(f'tokenizers-0.22.2-{name}.patch' for name in PATCH_NAMES)
PATCH_SHA256 = {
    'tokenizers-0.22.2-all-candidate-noise.patch': '715378fe851b44bfb94effbc3ce5f2a960960fd0d441c122e984325f58355f2e',
    'tokenizers-0.22.2-unigram-literal-unk.patch': 'e33d0882298dd6cdefd5f1bad30d5bfa1b301d4fdc79d38df90d2b0e08504fbb',
    'tokenizers-0.22.2-unigram-stack-drop.patch': 'f833d852a4e83054978e6a30a740617ce7132bcbf7097064bc60add5bee9cabb',
}
#: Earlier top-32 experiment; edits the same hunks as all-candidate-noise.  Must never be applied.
FORBIDDEN_PATCH = 'tokenizers-0.22.2-merge-noise.patch'
#: Full git blob ids of the four upstream files the patches modify, at UPSTREAM_REVISION
#: (the patches' ``index`` lines give the 7-digit prefixes 53415ff, cda6aea, 955ac7b, 2897bb3;
#: NOTES/gaps.md section 3.1).  Read from the upstream repository on 2026-11-13.
UPSTREAM_BLOBS = {
    'bindings/python/src/trainers.rs': '53415fff02cb5d2651dd92cdb760f944bbe77ef3',
    'tokenizers/src/models/bpe/trainer.rs': 'cda6aea65436842277d4ed7f4ab53f6447157af4',
    'tokenizers/src/models/unigram/model.rs': '955ac7b5ce831597cdaf8b9099140bfb964f4ed0',
    'tokenizers/src/models/unigram/lattice.rs': '2897bb3767ff20c017e0087871fc68a9f7b2ff01',
}
#: Rust toolchain for the wheel (NOTES/gaps.md 3.1: the compiler of the committed librust_utils.so).
#: Upstream ships ``rust-toolchain`` files containing "stable", so the build sets RUSTUP_TOOLCHAIN.
RUST_TOOLCHAIN = '1.96.0'
TOKENIZERS_VERSION = '0.22.2'

# --- the plan (plans/tokenizer-pool-1000-v1.json; never written by this port) --------------
PLAN_REL = 'plans/tokenizer-pool-1000-v1.json'
PLAN_FILE_SHA256 = '4bbd89c6d40993a036ec33072a8300acf9f50a222e8b258c9ec2e0b24fe43b8e'
PLAN_FILE_BYTES = 1230775
PLAN_SHA256 = '1f5d0be379e321015d1ab7b35bb93bcf82e8ce32f949411de1d7272dfa3cb1bb'
PLAN_COUNT = 1000
PILOT_INDICES = (413, 522, 262, 374, 802, 329, 86, 492, 217, 398, 663, 79)
#: The seven tokenizers with exactly 95 native tokens (the cap of scripts/pool_regime.py:12) on a passage of the
#: pool panel that was valid in the original (original_joint_encoding_matches): one more token for any of them
#: makes hyak.eval.validate_prefixes --panel pool stop (exit 3).  No pilot is among them (review readme-2).
BINDING_INDICES = (64, 100, 632, 687, 728, 821, 914)
#: Words of --indices besides integers and ranges.
INDEX_PRESETS = {'pilots': 'the 12 pilot indices', 'binding': 'the seven BINDING_INDICES',
                 'pilots+binding': 'the pilots, then the seven BINDING_INDICES'}
#: configs[0]['seed'] of the saved plan; > 2**53, so it proves the plan was parsed exactly.
PLAN_SEED_0 = 7566336434293423836
TOTAL_BYTES_PER_TOKENIZER = 200_000_000
#: scripts/run_tokenizer_pool_farm.py:190-191: up to 3 bytes per bucket may stay unused.
MAX_SLACK_PER_BUCKET = 3

# --- the unchanged per-tokenizer worker ---------------------------------------------------------
WORKER_REL = 'scripts/tokenizer_pool.py'
WORKER_SHA256 = 'b194b104a6dc4843f52f939d5f7bf9826b05e7b007bd2c03aef12f04be588a83'
#: RAYON_NUM_THREADS of the original pool (scripts/modal_generate_tokenizer_pool.py:47).
RAYON_THREADS = 4
#: Names recorded in the original provenance (scripts/modal_generate_tokenizer_pool.py:196-197).
UNIGRAM_PATCHES = ['tokenizers-0.22.2-unigram-literal-unk.patch', 'tokenizers-0.22.2-unigram-stack-drop.patch']

# --- consumers (pinned snapshot files) ------------------------------------------------------------
POOL_SAMPLING_REL = 'runtime/anytok_pool_20k/hf_hypernet/pool_sampling.py'
POOL_SAMPLING_SHA256 = '846446a0ddb8b833801bb74b7939452a3195760a98e9bab2a3dd2e73d57f1b57'
TOKENIZER_INPUTS_REL = 'runtime/anytok_pool_20k/hf_hypernet/tokenizer_inputs.py'
TOKENIZER_INPUTS_SHA256 = '70e99d9edb56fead0597a2d593dc992a2ad3b85d688c09bfe953bcc7d42077c8'
SETTINGS_POLICY_REL = 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
SETTINGS_POLICY_SHA256 = '6dbfcbf8c6552ef98b56de08c09ae59bd35a2d7d1a75e7e295cfcf5346a90b8c'
#: policy.feature_sha256 of the original RL run (NOTES/trainer-rl.md, launch-verification.json).
FEATURE_SHA256 = '9fe7f9e5ace73c04001db96c1953cdc52e62287f1b0a64210e247b0e1f290c23'
FEATURE_SHAPE = (1000, 18)
#: First action of SettingsTokenizerBandit(seed=43) over the 1,000 configs (rl.csv update 1).
FIRST_POLICY_ACTION = 652

# --- original-pool anchors (scorecard) ---------------------------------------------------------------
SPEC_REL = 'reports/rl-pool-sweep-20k/spec.json'
SPEC_SHA256 = '754aab7d55d78d9f37986d9ec0eb2208be99f1b95721fb6a19b006664c42cc0a'
UNIFORM_CSV_REL = 'reports/rl-tokenizer-selection-20k/data/training/uniform.csv'
UNIFORM_CSV_SHA256 = 'fce2fb316d2de2e2a75d3582ecd8cd7b1e2bc34cc917b0678dfd33de30262eda'
REWARD_TABLE_REL = 'reports/frozen-policy-search-20k/reward_table.json.gz'
REWARD_TABLE_SHA256 = '370e78868f1eb6284c5dd975146c0597ea8e5344df56bc622471cc64a0668151'
#: SHA-256 of the original full-results.json (spec.json pool_index_sha256); a regenerated index differs.
ORIGINAL_INDEX_SHA256 = '7cdc2c0bb4b300605d152255ed6156f598154fb3dbe402e272c83276941353a1'

ALGORITHMS = ('bpe', 'wordpiece', 'unigram')          # scripts/tokenizer_pool.py:26
PRETOKENIZERS = ('regex', 'line256', 'no-regex')      # scripts/tokenizer_pool.py:27
MODEL_TYPES = {'bpe': 'BPE', 'wordpiece': 'WordPiece', 'unigram': 'Unigram'}
LINE256_PATTERN = r'[^\n]{1,256}\n?|\n'               # scripts/tokenizer_pool.py:166

#: Provenance keys that must be identical for every tokenizer of one pool (collect.py refuses a
#: mixed pool; the worker refuses to reuse a tokenizer whose identity differs).
IDENTITY_KEYS = ('plan_sha256', 'plan_file_sha256', 'production_plan', 'worker_sha256', 'rayon_threads',
                 'runtime_wheel_sha256', 'upstream_revision', 'patches', 'recipe_sha256', 'corpus')


#: Recipe fields that a generation recipe may rewrite, with the number of trailing path components kept.
REWRITABLE_PATHS = {'source_path': 2, 'eval_path': 3}


class PoolError(RuntimeError):
    """A pool input or output is invalid; the message says what to do."""


def digest(value) -> str:
    """Canonical JSON SHA-256, verbatim from scripts/tokenizer_pool.py:30-31."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def family(config: dict) -> str:
    """``'<algorithm>/<pretokenizer>'`` of a plan configuration (e.g. ``'bpe/no-regex'``)."""
    return f"{config['algorithm']}/{config['pretokenizer']}"


def index_name(index: int) -> str:
    return f'{index:04d}'


def manifest_text(manifest: dict) -> str:
    """Serialization of a per-tokenizer manifest, as the original wrote it
    (scripts/tokenizer_pool.py:213 and scripts/modal_generate_tokenizer_pool.py:91)."""
    return json.dumps(manifest, indent=2) + "\n"


def max_slack(plan: dict) -> int:
    """``3 * number of buckets`` (69 for the production recipe); scripts/run_tokenizer_pool_farm.py:190-191."""
    return MAX_SLACK_PER_BUCKET * len(plan['recipe']['buckets'])


# --- plan ------------------------------------------------------------------------------------------
def load_plan(path, *, allow_test_plan: bool = False):
    """Read and verify a pool plan.  Returns ``(plan, info)``.

    ``info`` = ``{"path", "file_sha256", "bytes", "production"}``.  The production plan must be
    the repository file byte for byte (PLAN_FILE_SHA256).  Any other file is rejected unless
    ``allow_test_plan`` (tests and local smoke runs only).  Every plan is checked for internal
    consistency: ``digest(plan without plan_sha256) == plan_sha256``
    (scripts/modal_generate_tokenizer_pool.py:51-55), ``configs[i]['index'] == i``, every
    ``config_sha256`` (scripts/tokenizer_pool.py:71), integer seeds in ``[0, 2**63)``, quotas
    summing to ``total_bytes_per_tokenizer`` over exactly the recipe's buckets.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PoolError(f'Cannot read the pool plan {path}: {exc}') from exc
    file_sha = sha256_bytes(raw)
    production = file_sha == PLAN_FILE_SHA256
    if not production and not allow_test_plan:
        raise PoolError(f'{path} is not the saved production plan (SHA-256 {file_sha}, expected '
                        f'{PLAN_FILE_SHA256}). Use the checked-in {PLAN_REL} unchanged; never regenerate it.')
    try:
        plan = json.loads(raw)
    except ValueError as exc:
        raise PoolError(f'{path}: invalid JSON ({exc})') from exc
    problems = plan_problems(plan, production=production)
    if problems:
        raise PoolError(f'{path}: invalid pool plan: ' + '; '.join(problems[:10])
                        + (f' (+{len(problems) - 10} more)' if len(problems) > 10 else ''))
    return plan, {'path': str(path), 'file_sha256': file_sha, 'bytes': len(raw), 'production': production}


def plan_problems(plan, *, production: bool) -> list:
    """Every reason ``plan`` is not a consistent plan (empty list = valid)."""
    problems = []
    if not isinstance(plan, dict):
        return ['plan is not a JSON object']
    for key in ('plan_sha256', 'count', 'configs', 'recipe', 'total_bytes_per_tokenizer', 'pilot_indices'):
        if key not in plan:
            problems.append(f'missing key {key}')
    if problems:
        return problems
    payload = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    if digest(payload) != plan['plan_sha256']:
        problems.append('plan_sha256 does not match the plan contents')
    if production:
        if plan['plan_sha256'] != PLAN_SHA256:
            problems.append(f'plan_sha256 {plan["plan_sha256"]} != {PLAN_SHA256}')
        if plan['count'] != PLAN_COUNT or tuple(plan['pilot_indices']) != PILOT_INDICES:
            problems.append('count or pilot_indices differ from the saved plan')
        if plan['configs'] and plan['configs'][0].get('seed') != PLAN_SEED_0:
            problems.append('configs[0].seed was not parsed exactly (use Python json)')
    configs = plan['configs']
    if not isinstance(configs, list) or len(configs) != plan['count']:
        problems.append(f'expected {plan["count"]} configs, found {len(configs) if isinstance(configs, list) else "none"}')
        return problems
    try:
        buckets = [b['name'] for b in plan['recipe']['buckets']]
    except (KeyError, TypeError):
        return problems + ['recipe has no bucket names']
    total = plan['total_bytes_per_tokenizer']
    for position, config in enumerate(configs):
        if not isinstance(config, dict) or config.get('index') != position:
            problems.append(f'configs[{position}] has index {config.get("index") if isinstance(config, dict) else None}')
            continue
        body = {k: v for k, v in config.items() if k != 'config_sha256'}
        if digest(body) != config.get('config_sha256'):
            problems.append(f'config {position}: config_sha256 mismatch')
        seed = config.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
            problems.append(f'config {position}: seed {seed!r} is not an integer in [0, 2**63)')
        if config.get('algorithm') not in ALGORITHMS or config.get('pretokenizer') not in PRETOKENIZERS:
            problems.append(f'config {position}: unknown algorithm/pretokenizer')
        quota = config.get('bucket_bytes') or {}
        if list(quota) != buckets and set(quota) != set(buckets):
            problems.append(f'config {position}: bucket_bytes keys differ from the recipe buckets')
        elif sum(quota.values()) != total:
            problems.append(f'config {position}: bucket quotas sum to {sum(quota.values())}, not {total}')
        vocab = config.get('vocab_size')
        if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab <= 0:
            problems.append(f'config {position}: invalid vocab_size {vocab!r}')
    for index in plan['pilot_indices']:
        if not isinstance(index, int) or not 0 <= index < len(configs):
            problems.append(f'pilot index {index!r} out of range')
    return problems


def recipe_problems(pool_recipe, plan_recipe) -> list:
    """Differences between the generation recipe and the plan's recipe that could change a tokenizer.

    The generation recipe (``input/pool-recipe.json``, written by hyak.corpus.stage_inputs) may
    differ from ``plan['recipe']`` only in each bucket's cluster paths ``source_path`` and
    ``eval_path`` (the precedent is scripts/modal_generate_fast_tokenizers.py:58).  The worker reads
    only ``name`` and ``source_path`` (scripts/tokenizer_pool.py:104-111), in recipe order.  A
    rewritten path must be absolute and keep the trailing ``<pool>-v1/<bucket>`` components of the
    original ``/corpus/<pool>-v1/<bucket>`` (``.../eval.jsonl.gz`` for eval_path), so a swapped
    bucket cannot slip through.
    """
    problems = []
    try:
        mine, theirs = pool_recipe['buckets'], plan_recipe['buckets']
    except (KeyError, TypeError):
        return ['recipe has no "buckets" list']
    if [b.get('name') for b in mine] != [b['name'] for b in theirs]:
        return ['bucket names or order differ from the plan recipe']
    for new, old in zip(mine, theirs):
        name = old['name']
        if ({k: v for k, v in new.items() if k not in REWRITABLE_PATHS}
                != {k: v for k, v in old.items() if k not in REWRITABLE_PATHS}):
            problems.append(f'{name}: fields other than {"/".join(REWRITABLE_PATHS)} differ from the plan recipe')
        for key, keep in REWRITABLE_PATHS.items():
            if key not in old and key not in new:
                continue
            path, original = new.get(key), old.get(key)
            if not isinstance(path, str) or not path.startswith('/'):
                problems.append(f'{name}: {key} {path!r} is not absolute')
            elif not isinstance(original, str) or Path(path).parts[-keep:] != Path(original).parts[-keep:]:
                problems.append(f'{name}: {key} {path} does not end with '
                                f'{"/".join(Path(str(original)).parts[-keep:])}')
    return problems


def recipe_identity(pool_recipe) -> str:
    """``recipe_sha256`` of the pool identity: the generation recipe with every rewritable path reduced to the
    trailing components :func:`recipe_problems` pins (``full-v1/web-en``, ``full-v1/web-en/eval.jsonl.gz``), so
    the identity does not depend on where this cluster keeps the corpus.  Canonical JSON digest."""
    normal = json.loads(json.dumps(pool_recipe))
    for bucket in normal.get('buckets', []):
        for key, keep in REWRITABLE_PATHS.items():
            if isinstance(bucket.get(key), str):
                bucket[key] = '/'.join(Path(bucket[key]).parts[-keep:])
    return digest(normal)


# --- per-tokenizer outputs -------------------------------------------------------------------------------
def tokenizer_json_problems(raw: bytes, config: dict, vocab_size: int) -> list:
    """Structural checks of a saved ``tokenizer.json`` (JSON level, no ``tokenizers`` import).

    Mirrors what the consumers require: contiguous ids ``0..vocab_size-1`` and finite Unigram
    scores with ``<|endoftext|>`` present (runtime/anytok_pool_20k/hf_hypernet/pool_sampling.py:39-51),
    a byte-level decoder (scripts/uniform_weakness.py:21-22), and the lossless shape of
    scripts/tokenizer_speed.py:40-58 for the configured algorithm and pretokenizer
    (scripts/tokenizer_pool.py:150-172).
    """
    problems = []
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return [f'tokenizer.json is not valid JSON ({exc})']
    model = data.get('model') or {}
    expected_type = MODEL_TYPES.get(config.get('algorithm'))
    if model.get('type') != expected_type:
        problems.append(f'model type {model.get("type")!r}, expected {expected_type!r}')
    if data.get('normalizer') is not None or data.get('post_processor') is not None:
        problems.append('normalizer or post_processor is set')
    if data.get('truncation') is not None or data.get('padding') is not None:
        problems.append('truncation or padding is set')
    if (data.get('decoder') or {}).get('type') != 'ByteLevel':
        problems.append('decoder is not ByteLevel')
    pre = data.get('pre_tokenizer') or {}
    pretokenizer = config.get('pretokenizer')
    if pretokenizer == 'line256':
        steps = pre.get('pretokenizers') or []
        if (pre.get('type') != 'Sequence' or len(steps) != 2 or steps[0].get('type') != 'Split'
                or steps[0].get('pattern') != {'Regex': LINE256_PATTERN} or steps[0].get('behavior') != 'Isolated'
                or steps[0].get('invert')):
            problems.append('pre_tokenizer is not the line256 Split + ByteLevel sequence')
        pre = steps[1] if len(steps) == 2 else {}
        if pre.get('use_regex') is not False:
            problems.append('line256 ByteLevel step must have use_regex false')
    elif pre.get('use_regex') is not (pretokenizer == 'regex'):
        problems.append(f'ByteLevel use_regex {pre.get("use_regex")!r} does not match pretokenizer {pretokenizer!r}')
    if pre.get('type') != 'ByteLevel' or pre.get('add_prefix_space'):
        problems.append('pre_tokenizer is not ByteLevel without add_prefix_space')
    if any(t.get('lstrip') or t.get('rstrip') for t in data.get('added_tokens') or []):
        problems.append('an added token strips whitespace')
    vocab = model.get('vocab')
    ids = []
    tokens = set()
    if expected_type == 'Unigram':
        if not isinstance(vocab, list):
            return problems + ['Unigram vocab is not a list']
        for position, row in enumerate(vocab):
            try:
                piece, score = row
                score = float(score)
            except (TypeError, ValueError):
                problems.append(f'Unigram vocab row {position} is malformed')
                break
            if not math.isfinite(score):
                problems.append(f'Unigram score of row {position} is not finite')
                break
            ids.append(position)
            tokens.add(piece)
    elif isinstance(vocab, dict):
        ids = list(vocab.values())
        tokens = set(vocab)
    else:
        return problems + ['vocab is missing']
    for token in data.get('added_tokens') or []:
        ids.append(token.get('id'))
        tokens.add(token.get('content'))
    if set(ids) != set(range(vocab_size)) or len(tokens) != vocab_size:
        problems.append(f'token ids are not exactly 0..{vocab_size - 1} for {len(tokens)} distinct tokens')
    if '<|endoftext|>' not in tokens:
        problems.append('<|endoftext|> is missing')
    return problems


def manifest_problems(manifest, config: dict, plan: dict, tokenizer_path=None, *, check_hash: bool = True,
                      identity=None, require_provenance: bool = True) -> list:
    """Reasons ``manifest`` is not a verified complete tokenizer of ``config`` (empty list = complete).

    Completion marker (DESIGN.md 5.F): status "complete", config equal to the plan's, vocabulary
    size as configured, corpus slack ``0 <= total - used <= 3 * buckets``
    (scripts/run_tokenizer_pool_farm.py:185-192), tokenizer bytes and SHA-256 as recorded
    (scripts/modal_generate_tokenizer_pool.py:71-77), and a provenance block whose identity equals
    ``identity`` when given.
    """
    if not isinstance(manifest, dict):
        return ['manifest is not a JSON object']
    problems = []
    if manifest.get('status') != 'complete':
        problems.append(f'status is {manifest.get("status")!r}')
    if manifest.get('config') != config:
        problems.append('config differs from the plan')
    if manifest.get('vocab_size_actual') != config.get('vocab_size'):
        problems.append(f'vocab_size_actual {manifest.get("vocab_size_actual")} != {config.get("vocab_size")}')
    used = manifest.get('total_used_bytes')
    if not isinstance(used, int) or not 0 <= plan['total_bytes_per_tokenizer'] - used <= max_slack(plan):
        problems.append(f'total_used_bytes {used!r} is outside {plan["total_bytes_per_tokenizer"]} - [0, {max_slack(plan)}]')
    stats = manifest.get('source_stats')
    if not isinstance(stats, dict) or set(stats) != {b['name'] for b in plan['recipe']['buckets']}:
        problems.append('source_stats does not cover exactly the recipe buckets')
    if require_provenance or identity is not None:
        provenance = manifest.get('provenance')
        if not isinstance(provenance, dict):
            problems.append('provenance block is missing')
        elif identity is not None:
            mine = identity_of(provenance)
            changed = sorted(k for k in IDENTITY_KEYS if mine.get(k) != identity.get(k))
            if changed:
                problems.append('provenance differs from this pool in: ' + ', '.join(changed))
    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        try:
            size = path.stat().st_size
        except OSError:
            problems.append(f'{path.name} is missing')
        else:
            if size != manifest.get('tokenizer_bytes'):
                problems.append(f'{path.name} has {size} bytes, manifest says {manifest.get("tokenizer_bytes")}')
            elif check_hash:
                actual = file_sha256(path)
                if actual != manifest.get('tokenizer_sha256'):
                    problems.append(f'{path.name} SHA-256 {actual} != manifest {manifest.get("tokenizer_sha256")}')
    return problems


def identity_of(provenance: dict) -> dict:
    """The pool-identity projection of a provenance block (IDENTITY_KEYS)."""
    return {key: provenance.get(key) for key in IDENTITY_KEYS}


def file_sha256(path, chunk: int = 8 << 20) -> str:
    digest_ = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(chunk), b''):
            digest_.update(block)
    return digest_.hexdigest()


def parse_indices(values, count: int, pilots=PILOT_INDICES) -> list:
    """Parse ``--indices`` values: integers, ``a-b`` ranges, comma lists, or a preset word
    (``pilots``, ``binding``, ``pilots+binding``; see INDEX_PRESETS).

    Order is preserved and duplicates are dropped; every index must be in ``0..count-1``.
    """
    result = []
    for value in values or []:
        for part in str(value).replace(',', ' ').split():
            if part in INDEX_PRESETS:
                if part in ('pilots', 'pilots+binding'):
                    result.extend(pilots)
                if part in ('binding', 'pilots+binding'):
                    result.extend(BINDING_INDICES)
                continue
            if '-' in part:
                low, _, high = part.partition('-')
                if not (low.isdigit() and high.isdigit()) or int(low) > int(high):
                    raise PoolError(f'Invalid index range {part!r}')
                result.extend(range(int(low), int(high) + 1))
            elif part.isdigit():
                result.append(int(part))
            else:
                raise PoolError(f'Invalid index {part!r} (expected integers, a-b ranges or one of '
                                f'{", ".join(INDEX_PRESETS)})')
    bad = [i for i in result if not 0 <= i < count]
    if bad:
        raise PoolError(f'Indices out of range 0..{count - 1}: {bad[:10]}')
    return list(dict.fromkeys(result))
