"""Insert or update the ``hyak-*`` entries of the dashboard registry (``plans/run-dashboard.json``).

    python -m hyak.tools.register_dashboard [--registry PATH] [--dry-run | --check] [--config FILE]

* Idempotent: the managed entries (``hyak.common.paths.DASHBOARD_RUN_IDS``) are
  placed, in a fixed order, AFTER every other entry (so the dashboard's default view, the first entry,
  does not change); running it again with the same config changes nothing (the file is not rewritten
  when unchanged).  Every managed entry has ``completion_stage: complete``: the dashboard shows
  complete only when the stage itself says so, not when a counter reaches its total (a training run
  at 20,000 steps is still ``auditing`` until its final audit passes).
* Every other entry, and everything outside the ``runs`` array, is kept
  byte-for-byte and in order: the new file is spliced from the original text
  (entries are located with the JSON scanner), not re-serialised.  Other ``hyak-*``
  ids that are not managed here are kept where they are.
* The result is validated with ``monitoring.state.registry`` and written
  atomically.  All paths are repo-relative (the learning panel's globs must be).
* ``--dry-run`` prints the managed entries and whether the file would change;
  ``--check`` exits 4 when the registry is not up to date (writes nothing).  Exit status: 0 done, 2 usage,
  configuration or input error, 4 ``--check`` failed.

Only the platform text comes from the config (``HYAK_PLATFORM`` and its per-stage
overrides), so the registry contains no cluster paths.  Run it on the cluster
(the sbatch scripts do); do not commit placeholder output from elsewhere.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from hyak.common.config import ConfigError, load_config, repo_root, stage_value
from hyak.common.io import atomic_write_text
from hyak.common.paths import (DASHBOARD_RUN_IDS, FINAL_STEP, STAGE_PROGRESS, STAGE_SMOKE_RAWBPB, STAGE_TRAIN_RAWBPB,
                               STAGE_TRAIN_RL, default_monitor_root, monitor_rel)

#: Display names of the managed entries.
NAMES = {
    'train-rawbpb': 'AnyTok · Raw-BPB RL · entropy weight 0 · 20k steps · Hyak',
    'smoke-rawbpb': 'AnyTok · Raw-BPB RL · startup and restart check · Hyak',
    'train-uniform': 'AnyTok · Uniform pool 20k · Hyak',
    'train-rl': 'AnyTok · Joint RL 20k · straight BPB reward · entropy 0 · Hyak',
    'eval-uniform': 'AnyTok · Evaluation · uniform 20k checkpoint · Hyak',
    'eval-rl': 'AnyTok · Evaluation · joint RL 20k (straight BPB reward, entropy 0) checkpoint · Hyak',
    'eval-prepare': 'AnyTok · Evaluation features and prefixes · Hyak',
    'pool': 'AnyTok · 1,000 tokenizer pool · Hyak',
    'corpus': 'AnyTok · Corpus · 23 buckets · Hyak',
    'probe': 'AnyTok · H200 memory probe · Hyak',
}

#: Plan whose configs give algorithm / pretokenizer of each pool index (learning panel).
POOL_PLAN = 'plans/tokenizer-pool-1000-v1.json'
#: The RL run's sanity panel uses the corrected paired cuts (D8), the uniform run's the original ones: like
#: Qingyun's anytok-pool-regret-rl-training entry, the uniform sanity scores are not overlaid (a baseline glob
#: that matches nothing, baseline_required false) and the panel says why.
RL_BASELINE_LOGS = 'no-old-protocol-baseline-*.log'
RL_PROTOCOL_NOTE = ('Straight transferred-BPB policy reward, entropy 0, corrected paired boundaries on this run. '
                    'The Hyak uniform run\'s sanity scores used the original cuts and are not overlaid.')

#: The raw-BPB handoff stages (README section 0) are modelled on Qingyun's example entry in
#: docs/raw-bpb-handoff/README.md: the progress_json adapter on the launcher's own OUTPUT/status.json, completed field
#: ``step``, its OUTPUT/events.jsonl in the logs.  The job links <monitor>/<stage>/run -> OUTPUT, so the registry keeps
#: repository-relative paths only.
RAWBPB_STAGES = (STAGE_TRAIN_RAWBPB, STAGE_SMOKE_RAWBPB)
RAWBPB_LINK = 'run'
DESCRIPTIONS = {
    STAGE_TRAIN_RAWBPB: ('Fresh raw-BPB comparison with corrected boundaries and the GPU-only memory fix (Qingyun\'s '
                         'launcher and input bundle on Tillicum); the baseline is the original uniform run.'),
    STAGE_SMOKE_RAWBPB: 'Two updates from scratch, then a resume to update 3, with the raw-BPB receipt checks.',
}
RAWBPB_PROTOCOL_NOTE = ('Raw transferred-BPB policy reward, entropy 0, corrected paired boundaries on this run. '
                        'Historical uniform sanity scores used different cuts and are not overlaid.')

_WS = re.compile(r'[ \t\n\r]*')


def _rel(stage, name):
    return f'{monitor_rel(stage)}/{name}'


def build_entries(cfg: dict) -> list:
    """The managed registry entries, in registry order, for a ``load_config()`` dict."""
    entries = []
    for stage, run_id in DASHBOARD_RUN_IDS.items():
        progress = STAGE_PROGRESS[stage]
        rawbpb = stage in RAWBPB_STAGES
        entry = {
            'id': run_id,
            'name': NAMES[stage],
            'adapter': 'progress_json',
            'platform': stage_value(cfg, 'HYAK_PLATFORM', stage),
            'status': _rel(stage, f'{RAWBPB_LINK}/status.json' if rawbpb else 'status.json'),
            'completed_field': progress['completed_field'],
        }
        entry['completion_stage'] = 'complete'
        if progress['total'] is None:
            entry['total_field'] = 'total'
        else:
            entry['total'] = progress['total']
        entry['unit'] = progress['unit']
        logs = [_rel(stage, 'events.jsonl'), _rel(stage, 'latest-slurm.out')]
        if rawbpb:
            logs = [_rel(stage, f'{RAWBPB_LINK}/events.jsonl')] + logs
        if stage == STAGE_TRAIN_RAWBPB:
            entry['learning'] = {
                'logs': _rel(stage, '*-train.log'),
                'baseline_logs': _rel(stage, RL_BASELINE_LOGS),
                'plan': POOL_PLAN,
                'snapshots': _rel(stage, 'policy-monitor.json'),
                'protocol_note': RAWBPB_PROTOCOL_NOTE,
                'baseline_required': False,
            }
            logs.append(_rel(stage, 'policy-monitor.log'))
        if stage == STAGE_TRAIN_RL:
            entry['learning'] = {
                'logs': _rel(STAGE_TRAIN_RL, '*-train.log'),
                'baseline_logs': _rel(STAGE_TRAIN_RL, RL_BASELINE_LOGS),
                'plan': POOL_PLAN,
                'snapshots': _rel(STAGE_TRAIN_RL, 'policy-monitor.json'),
                'protocol_note': RL_PROTOCOL_NOTE,
                'baseline_required': False,
            }
            logs.append(_rel(STAGE_TRAIN_RL, 'policy-monitor.log'))
        entry['profiles'] = []
        entry['app_names'] = []
        entry['logs'] = logs
        if rawbpb:
            entry['category'] = 'Training'
            entry['description'] = DESCRIPTIONS[stage]
        entries.append(entry)
    for entry in entries:
        smoke = entry['id'] == DASHBOARD_RUN_IDS[STAGE_SMOKE_RAWBPB]      # a 3-update check, not a 20k run
        if entry['completed_field'] == 'step' and not smoke and entry.get('total') != FINAL_STEP:
            raise RuntimeError(f"training entry {entry['id']} must have total {FINAL_STEP}")
    return entries


def _skip(text, index):
    return _WS.match(text, index).end()


def scan_runs(text: str):
    """Locate the ``runs`` array of a registry text.

    Returns ``(elements, separator, indent)`` where ``elements`` is a list of
    ``(start, end, value)`` character spans of each entry.  Raises ValueError for
    anything that is not a JSON object with exactly one ``runs`` array.
    """
    decoder = json.JSONDecoder()
    index = _skip(text, 0)
    if text[index:index + 1] != '{':
        raise ValueError('registry must be a JSON object')
    index = _skip(text, index + 1)
    found = None
    if text[index:index + 1] == '}':
        raise ValueError('registry has no "runs" array')
    while True:
        if text[index:index + 1] != '"':
            raise ValueError(f'expected a key at character {index}')
        key, index = json.decoder.scanstring(text, index + 1)
        index = _skip(text, index)
        if text[index:index + 1] != ':':
            raise ValueError(f'expected ":" at character {index}')
        index = _skip(text, index + 1)
        if key == 'runs':
            if found is not None:
                raise ValueError('registry has more than one "runs" key')
            if text[index:index + 1] != '[':
                raise ValueError('"runs" must be an array')
            open_at = index
            index = _skip(text, index + 1)
            elements = []
            if text[index:index + 1] != ']':
                while True:
                    start = index
                    value, index = decoder.raw_decode(text, index)
                    elements.append((start, index, value))
                    index = _skip(text, index)
                    char = text[index:index + 1]
                    if char == ',':
                        index = _skip(text, index + 1)
                        continue
                    if char == ']':
                        break
                    raise ValueError(f'expected "," or "]" at character {index}')
            found = (open_at, index, elements)
            index += 1
        else:
            _, index = decoder.raw_decode(text, index)
        index = _skip(text, index)
        char = text[index:index + 1]
        if char == ',':
            index = _skip(text, index + 1)
            continue
        if char == '}':
            break
        raise ValueError(f'expected "," or "}}" at character {index}')
    if text[index + 1:].strip():
        raise ValueError('trailing data after the registry object')
    if found is None:
        raise ValueError('registry has no "runs" array')
    return found


def _entry_text(entry, indent):
    return json.dumps(entry, indent=2, ensure_ascii=False).replace('\n', '\n' + indent)


def render(text: str, entries: list) -> str:
    """New registry text: every other original entry verbatim, then ``entries`` (stale managed ones removed)."""
    open_at, close_at, elements = scan_runs(text)
    managed = {e['id'] for e in entries}
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('managed entries have duplicate ids')
    stale = [e for e in elements if isinstance(e[2], dict) and e[2].get('id') in managed]
    if elements:
        first, last = elements[0][0], elements[-1][1]
        line_start = text.rfind('\n', 0, first) + 1
        indent = text[line_start:first] if not text[line_start:first].strip() else '    '
        separator = text[elements[0][1]:elements[1][0]] if len(elements) > 1 else ',\n' + indent
        head, tail = text[:first], text[last:]
    else:
        indent = '    '
        separator = ',\n' + indent
        head = text[:open_at + 1] + '\n' + indent
        tail = '\n  ' + text[close_at:]
    kept = [text[s:e] for s, e, value in elements
            if not (isinstance(value, dict) and value.get('id') in managed)]
    pieces = kept + [_entry_text(e, indent) for e in entries]
    if elements and not stale:
        # nothing to replace: append after the last entry, everything before it stays byte for byte
        result = text[:last] + separator + separator.join(_entry_text(e, indent) for e in entries) + tail
        return _checked(text, result, entries, elements, managed)
    result = head + separator.join(pieces) + tail
    return _checked(text, result, entries, elements, managed)


def _checked(text, result, entries, elements, managed):
    """Return ``result`` after proving that it parses to the original with the managed entries last."""
    original = json.loads(text)
    expected = dict(original)
    expected['runs'] = [v for _, _, v in elements
                        if not (isinstance(v, dict) and v.get('id') in managed)] + list(entries)
    if json.loads(result) != expected:
        raise RuntimeError('internal error: spliced registry does not parse to the expected content; nothing written')
    return result


def validate(text: str, root: Path) -> list:
    """Run ``monitoring.state.registry`` on ``text`` (via a temporary file). Returns the runs."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from monitoring.state import registry
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'run-dashboard.json'
        path.write_text(text, encoding='utf-8')
        return registry(root, path)


def register(registry_path, entries, *, dry_run=False, check=False, lock_dir=None) -> bool:
    """Bring ``registry_path`` up to date with ``entries``. Returns True if it changed (or would change).

    Writes nothing when ``dry_run`` or ``check`` is set.  Holds an advisory
    ``flock`` on ``<lock_dir>/.registry.lock`` while reading and writing (best
    effort: concurrent writers produce identical content anyway).
    """
    registry_path = Path(registry_path)
    root = repo_root()
    lock = None
    if not (dry_run or check) and lock_dir is not None:
        try:
            Path(lock_dir).mkdir(parents=True, exist_ok=True)
            lock = open(Path(lock_dir) / '.registry.lock', 'a')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            print(f'[register_dashboard] WARNING: no lock ({exc}); continuing', file=sys.stderr)
            lock = None
    try:
        text = registry_path.read_text(encoding='utf-8')
        new = render(text, entries)
        runs = validate(new, root)
        ids = [r['id'] for r in runs]
        if len(set(ids)) != len(ids):
            raise RuntimeError('duplicate ids after registration')
        changed = new != text
        if changed and not (dry_run or check):
            atomic_write_text(registry_path, new)
        return changed
    finally:
        if lock is not None:
            lock.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.tools.register_dashboard', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--registry', type=Path, help='registry file (default: <repo>/plans/run-dashboard.json)')
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='print the managed entries; write nothing')
    mode.add_argument('--check', action='store_true', help='exit 4 if the registry is not up to date')
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        entries = build_entries(cfg)
        registry_path = args.registry or repo_root() / 'plans' / 'run-dashboard.json'
        lock_dir = default_monitor_root()
        changed = register(registry_path, entries, dry_run=args.dry_run, check=args.check, lock_dir=lock_dir)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f'{registry_path}: {"would change" if changed else "up to date"}', file=sys.stderr)
        return 0
    if args.check:
        print(f'{registry_path}: {"NOT up to date" if changed else "up to date"}')
        return 4 if changed else 0
    print(f'{registry_path}: {"updated" if changed else "unchanged"} ({len(entries)} hyak entries)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
