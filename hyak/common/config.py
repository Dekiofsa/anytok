"""Cluster settings: ``hyak/config.env`` on top of the defaults in ``hyak/config.env.example``.

Resolution of every key (identical in hyak/slurm/lib.sh ``hyak_load_config``; tested):

1. an environment variable of the same name, if it is set (even to the empty string);
2. else the value in the config file (``$HYAK_CONFIG`` or ``<repo>/hyak/config.env``);
3. else the default written in ``hyak/config.env.example``.

Per-stage overrides ``HYAK_<FLAG>_<STAGE>`` (see ``STAGE_OVERRIDE_FLAGS``) come
from the file or the environment and apply when non-empty (:func:`stage_value`).
Unknown keys in the file, malformed lines, duplicate keys, ``CHANGE_ME``
placeholders and invalid values are errors (:class:`ConfigError`) whose message
names the file, line and remedy.

``ANYTOK_ROOT`` and ``ANYTOK_RUNS`` must be normalised absolute paths (no trailing slash, ``//``, ``.`` or
``..`` component: conda normalises the prefix it is given, so a different spelling makes every job fail
at activation) and must lie outside the repository checkout (environments, corpus and checkpoints would
end up in the git working tree).  An empty ``HYAK_MEM`` / ``HYAK_MEM_<STAGE>`` means "no --mem flag": the
cluster's per-GPU default memory applies.  ``HYAK_HN_CHUNK_ROWS`` must be at least 4096.

File grammar (a strict subset of shell assignment syntax, so ``source`` and this
parser agree): ``KEY=value`` with an unquoted value of ``[A-Za-z0-9_./:@%+,=-]``,
a double-quoted value without ``" \\ $ ` ``, or a single-quoted value without
``'``; optional leading ``export``; blank lines; ``#`` comments (after a value
only when preceded by whitespace).

Command line (works with any Python >= 3.9, no third-party packages)::

    python -m hyak.common.config --check              # validate; exit 2 with the reason
    python -m hyak.common.config --print              # resolved values (HF_TOKEN redacted)
    python -m hyak.common.config --get KEY [--stage S]
    python -m hyak.common.config --resources STAGE [--debug]   # JSON, as submit.sh resolves them
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

from hyak.common.paths import STAGES, stage_suffix
from hyak.common.slurm import parse_time_limit

CONFIG_ENV = 'HYAK_CONFIG'
REPO_ENV = 'HYAK_REPO'
PLACEHOLDER = 'CHANGE_ME'
REQUIRED_PATH_KEYS = ('ANYTOK_ROOT', 'ANYTOK_RUNS')
#: Optional storage paths (empty = a default under $ANYTOK_RUNS; hyak.handoff.common).
OPTIONAL_PATH_KEYS = ('HYAK_RAWBPB_OUTPUT', 'HYAK_RAWBPB_SMOKE_OUTPUT')
#: Smallest accepted HYAK_HN_CHUNK_ROWS (hypernetwork chunk rows; hyak.memory enforces the same floor).
MIN_HN_CHUNK_ROWS = 4096
SECRET_KEYS = ('HF_TOKEN',)

#: Flags that may be overridden per stage as HYAK_<FLAG>_<STAGE>.
STAGE_OVERRIDE_FLAGS = ('PARTITION', 'QOS', 'ACCOUNT', 'GRES', 'CPUS', 'MEM', 'TIME',
                        'SEGMENT_MARGIN_MINUTES', 'PLATFORM')
#: Flags whose sbatch option is omitted when the resolved value is empty or ``none``.
OMITTABLE_FLAGS = ('PARTITION', 'QOS', 'ACCOUNT', 'GRES', 'CPUS', 'MEM')

_BLANK_RE = re.compile(r'[ \t]*(?:#.*)?')
_LINE_RE = re.compile(r'[ \t]*(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)')
_DQ_RE = re.compile(r'"([^"\\$`]*)"(?:[ \t]+(?:#.*)?)?')
_SQ_RE = re.compile(r"'([^']*)'(?:[ \t]+(?:#.*)?)?")
_UQ_RE = re.compile(r'([A-Za-z0-9_./:@%+,=-]*)(?:[ \t]+(?:#.*)?)?')

_NAME_RE = re.compile(r'[A-Za-z0-9_.,-]+')
_GRES_RE = re.compile(r'[A-Za-z0-9_]+(?::[A-Za-z0-9_.-]+)*(?:,[A-Za-z0-9_]+(?::[A-Za-z0-9_.-]+)*)*')
_MEM_RE = re.compile(r'\d+[KMGT]?')
_MODULES_RE = re.compile(r'[A-Za-z0-9_./+ -]*')

STAGE_OVERRIDE_KEYS = tuple(f'HYAK_{flag}_{stage_suffix(stage)}'
                            for flag in STAGE_OVERRIDE_FLAGS for stage in STAGES)
_STAGE_KEY_SET = frozenset(STAGE_OVERRIDE_KEYS)


class ConfigError(RuntimeError):
    """Invalid or incomplete configuration; the message says what to change."""


def repo_root() -> Path:
    """Repository root: ``$HYAK_REPO`` if set (must be an existing absolute directory), else two levels above this file."""
    value = os.environ.get(REPO_ENV, '')
    if value:
        path = Path(value)
        if not path.is_absolute() or not path.is_dir():
            raise ConfigError(f'{REPO_ENV}={value!r} is not an existing absolute directory; '
                              'unset it or point it at the repository checkout')
        return path
    return Path(__file__).resolve().parents[2]


def example_path() -> Path:
    """``<repo>/hyak/config.env.example`` (tracked; also the defaults)."""
    return repo_root() / 'hyak' / 'config.env.example'


def config_path(environ=None) -> Path:
    """Config file in use: ``$HYAK_CONFIG`` if set, else ``<repo>/hyak/config.env``."""
    environ = os.environ if environ is None else environ
    value = environ.get(CONFIG_ENV, '')
    return Path(value) if value else repo_root() / 'hyak' / 'config.env'


def is_stage_override_key(key: str) -> bool:
    """True for ``HYAK_<FLAG>_<STAGE>`` keys with a known flag and stage."""
    return key in _STAGE_KEY_SET


def parse_env_file(path) -> dict:
    """Parse a config file (grammar in the module docstring) into an ordered ``{key: value}`` dict.

    Raises ConfigError naming the file and line for unreadable files, invalid
    UTF-8, carriage returns, malformed lines and duplicate keys.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f'Cannot read config file {path}: {exc}') from exc
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ConfigError(f'{path}: not valid UTF-8 ({exc})') from exc
    values = {}
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    for number, line in enumerate(lines, 1):
        where = f'{path}:{number}'
        if '\r' in line:
            raise ConfigError(f"{where}: carriage return found (Windows line endings); "
                              f"convert the file with: sed -i 's/\\r$//' {path}")
        if '\0' in line:
            raise ConfigError(f'{where}: NUL byte in line')
        if _BLANK_RE.fullmatch(line):
            continue
        match = _LINE_RE.fullmatch(line)
        if not match:
            raise ConfigError(f'{where}: expected KEY=value, got {line!r}')
        key, rest = match.groups()
        for pattern in (_DQ_RE, _SQ_RE, _UQ_RE):
            value_match = pattern.fullmatch(rest)
            if value_match:
                value = value_match.group(1)
                break
        else:
            raise ConfigError(
                f'{where}: value of {key} must be unquoted [A-Za-z0-9_./:@%+,=-], double-quoted '
                'without " \\ $ or backtick, or single-quoted without \' (a trailing comment '
                'needs whitespace before #)')
        if key in values:
            raise ConfigError(f'{where}: duplicate key {key}')
        values[key] = value
    return values


def load_config(path=None, *, environ=None) -> dict:
    """Resolved settings as ``{key: str}`` (every key of config.env.example plus any per-stage overrides).

    ``path`` overrides the search order (``$HYAK_CONFIG``, then
    ``<repo>/hyak/config.env``).  ``environ`` defaults to ``os.environ``.
    Raises ConfigError with a remedy if the file is missing, malformed,
    contains unknown keys, still contains ``CHANGE_ME``, or has invalid values.
    """
    environ = os.environ if environ is None else environ
    example = example_path()
    defaults = parse_env_file(example)
    file = Path(path) if path is not None else config_path(environ)
    if not file.is_file():
        raise ConfigError(f'Config file {file} not found. Remedy: cp {example} '
                          f'{repo_root() / "hyak" / "config.env"} and set ANYTOK_ROOT and ANYTOK_RUNS '
                          f'(or point {CONFIG_ENV} at your file)')
    values = parse_env_file(file)
    unknown = [k for k in values if k not in defaults and not is_stage_override_key(k)]
    if unknown:
        raise ConfigError(f'{file}: unknown key(s) {", ".join(unknown)} (typo?). Known keys: '
                          f'{", ".join(defaults)}; per-stage overrides HYAK_<FLAG>_<STAGE> with FLAG in '
                          f'{" ".join(STAGE_OVERRIDE_FLAGS)}')
    cfg = {}
    for key, default in defaults.items():
        cfg[key] = environ[key] if key in environ else values.get(key, default)
    extra = sorted({k for k in values if k not in defaults} | {k for k in environ if k in _STAGE_KEY_SET})
    for key in extra:
        cfg[key] = environ[key] if key in environ else values[key]
    validate_config(cfg, source=file)
    return cfg


def _checkout_problems(key, value):
    """Problems of a storage path that is, or lies inside, the repository checkout (compared as strings)."""
    try:
        checkout = os.path.normpath(str(repo_root()))
    except ConfigError:
        return []  # the invalid HYAK_REPO is reported where it is read
    if value == checkout or value.startswith(checkout.rstrip('/') + '/'):
        return [f'{key}={value!r} is inside the repository checkout {checkout}; environments, corpus and '
                f'checkpoints would land in the git working tree. Choose a directory outside it, e.g. '
                f'/gpfs/projects/<group>/anytok-data']
    return []


def validate_config(cfg: dict, source='config') -> None:
    """Raise ConfigError listing every problem in a resolved config dict (placeholders, paths, numbers, times)."""
    problems = []
    placeholders = sorted(k for k, v in cfg.items() if PLACEHOLDER in v)
    if placeholders:
        problems.append(f'{", ".join(placeholders)} still contain {PLACEHOLDER}')
    for key in REQUIRED_PATH_KEYS:
        value = cfg.get(key, '')
        if not value:
            problems.append(f'{key} is empty')
        elif not value.startswith('/'):
            problems.append(f'{key}={value!r} must be an absolute path')
        elif value != '/' and (value != os.path.normpath(value) or value.startswith('//')):
            problems.append(f'{key}={value!r} must be a normalised path (no trailing slash, no //, no . or .. '
                            f'components); write {os.path.normpath(value)!r}')
        else:
            problems.extend(_checkout_problems(key, value))

    def integer(key, minimum, allow_none=False):
        value = cfg.get(key, '')
        if allow_none and value.strip().lower() in ('', 'none'):
            return
        if not re.fullmatch(r'\d+', value) or int(value) < minimum:
            problems.append(f'{key}={value!r} must be an integer >= {minimum}'
                            + (' (or none)' if allow_none else ''))

    def pattern(key, regex, what, allow_none=True):
        value = cfg.get(key, '')
        if allow_none and value.strip().lower() in ('', 'none'):
            return
        if not regex.fullmatch(value):
            problems.append(f'{key}={value!r} is not a valid {what}')

    def slurm_time(key, allow_empty=False):
        value = cfg.get(key, '')
        if allow_empty and value == '':
            return
        try:
            seconds = parse_time_limit(value)
        except ValueError:
            problems.append(f'{key}={value!r} is not a Slurm time limit (e.g. 24:00:00, 1-00:00:00)')
            return
        if not 0 < seconds < float('inf'):
            problems.append(f'{key}={value!r} must be a finite, non-zero time limit')

    outputs = {}
    for key in OPTIONAL_PATH_KEYS:
        value = cfg.get(key, '')
        if not value:
            continue
        if not value.startswith('/'):
            problems.append(f'{key}={value!r} must be an absolute path (or empty for the default)')
        elif value == '/' or value != os.path.normpath(value) or value.startswith('//'):
            problems.append(f'{key}={value!r} must be a normalised path (no trailing slash, no //, no . or .. '
                            f'components); write {os.path.normpath(value)!r}')
        else:
            problems.extend(_checkout_problems(key, value))
            outputs[key] = value
    if len(outputs) == 2 and len(set(outputs.values())) == 1:
        problems.append('HYAK_RAWBPB_OUTPUT and HYAK_RAWBPB_SMOKE_OUTPUT must be different directories')
    integer('HYAK_CPUS', 1, allow_none=True)
    integer('HYAK_SEGMENT_MARGIN_MINUTES', 0)
    integer('HYAK_HN_CHUNK_ROWS', MIN_HN_CHUNK_ROWS)
    integer('HYAK_POOL_WORKERS', 1)
    integer('HYAK_POOL_THREADS', 1)
    integer('HYAK_CORPUS_WORKERS', 1)
    slurm_time('HYAK_TIME')
    slurm_time('HYAK_TIME_DEBUG')
    pattern('HYAK_MEM', _MEM_RE, 'memory size (e.g. 160G)')
    pattern('HYAK_GRES', _GRES_RE, 'GRES request (e.g. gpu:1)')
    for key in ('HYAK_PARTITION', 'HYAK_QOS', 'HYAK_QOS_DEBUG', 'HYAK_ACCOUNT'):
        pattern(key, _NAME_RE, 'Slurm name')
    if not _MODULES_RE.fullmatch(cfg.get('HYAK_MODULES', '')):
        problems.append(f"HYAK_MODULES={cfg.get('HYAK_MODULES')!r} must be space-separated module names")
    if not cfg.get('HYAK_PLATFORM', '').strip():
        problems.append('HYAK_PLATFORM must not be empty')
    url = cfg.get('HYAK_TORCH_INDEX_URL', '')
    if url and not re.fullmatch(r'https?://\S+', url):
        problems.append(f'HYAK_TORCH_INDEX_URL={url!r} must be an http(s) URL or empty')
    for stage in STAGES:
        suffix = stage_suffix(stage)
        for flag in STAGE_OVERRIDE_FLAGS:
            key = f'HYAK_{flag}_{suffix}'
            if cfg.get(key, '') == '':
                continue
            if flag == 'CPUS':
                integer(key, 1, allow_none=True)
            elif flag == 'SEGMENT_MARGIN_MINUTES':
                integer(key, 0)
            elif flag == 'TIME':
                slurm_time(key)
            elif flag == 'MEM':
                pattern(key, _MEM_RE, 'memory size (e.g. 160G)')
            elif flag == 'GRES':
                pattern(key, _GRES_RE, 'GRES request (e.g. gpu:1)')
            elif flag in ('PARTITION', 'QOS', 'ACCOUNT'):
                pattern(key, _NAME_RE, 'Slurm name')
    if problems:
        raise ConfigError(f'Invalid configuration ({source}): ' + '; '.join(problems)
                          + '. Remedy: edit the config file or unset the overriding environment variables.')


def stage_value(cfg: dict, key: str, stage: str) -> str:
    """Per-stage value of an overridable key: non-empty ``<key>_<STAGE>`` if present, else ``cfg[key]``.

    ``key`` is the global key, e.g. ``'HYAK_PLATFORM'`` or ``'HYAK_SEGMENT_MARGIN_MINUTES'``.
    Mirrors ``hyak_stage_value`` in lib.sh.  (For QoS/time with debug semantics use
    :func:`hyak.common.slurm.sbatch_resources`.)
    """
    flag = key[len('HYAK_'):] if key.startswith('HYAK_') else ''
    if flag not in STAGE_OVERRIDE_FLAGS:
        raise ValueError(f'{key} has no per-stage override; overridable: '
                         + ', '.join(f'HYAK_{f}' for f in STAGE_OVERRIDE_FLAGS))
    override = cfg.get(f'{key}_{stage_suffix(stage)}', '')
    return override if override != '' else cfg[key]


def int_value(cfg: dict, key: str, stage=None) -> int:
    """Integer value of a key (per-stage resolved when ``stage`` is given and the key is overridable)."""
    value = stage_value(cfg, key, stage) if stage is not None else cfg[key]
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f'{key}={value!r} is not an integer') from exc


def redacted(cfg: dict) -> dict:
    """Copy of ``cfg`` with secrets replaced by ``***`` (empty secrets stay empty); safe to print or log."""
    return {k: ('***' if k in SECRET_KEYS and v else v) for k, v in cfg.items()}


def redact_text(text: str, cfg=None) -> str:
    """Replace any secret value (from ``cfg`` and the environment) occurring in ``text`` with ``***``."""
    secrets = set()
    for key in SECRET_KEYS:
        for source in (cfg or {}, os.environ):
            value = source.get(key, '')
            if len(value) >= 4:
                secrets.add(value)
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, '***')
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.common.config', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='validate and print OK')
    group.add_argument('--print', action='store_true', dest='print_all', help='print resolved KEY=VALUE lines')
    group.add_argument('--get', metavar='KEY', help='print one resolved value')
    group.add_argument('--resources', metavar='STAGE', help='print sbatch resources of a stage as JSON')
    parser.add_argument('--stage', help='resolve --get per stage')
    parser.add_argument('--debug', action='store_true', help='with --resources: as submit.sh --debug')
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.check:
            print(f'OK {args.config or config_path()}')
        elif args.print_all:
            for key, value in sorted(redacted(cfg).items()):
                print(f'{key}={value}')
        elif args.get:
            if args.stage:
                print(stage_value(cfg, args.get, args.stage))
            elif args.get in cfg:
                print('***' if args.get in SECRET_KEYS and cfg[args.get] else cfg[args.get])
            else:
                raise ConfigError(f'Unknown key {args.get}')
        else:
            from hyak.common.slurm import sbatch_resources
            print(json.dumps(sbatch_resources(cfg, args.resources, args.debug), sort_keys=True))
    except (ConfigError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
