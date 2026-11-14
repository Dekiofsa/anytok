"""Fidelity scorecard of a regenerated corpus against every anchor stored in the repository.

The original pools cannot be rebuilt byte for byte (gzip headers carry the build time), so
content is compared with what the repository recorded about them (NOTES corpus.md section 7,
gaps.md N3).  All anchors are read at run time from tracked files; nothing is copied here:

Exact numbers
  * ``data/balanced-full-v3.json``: ``train_available_bytes`` of all 23 buckets, i.e. the
    original ``counts.train_bytes`` (scripts/modal_prepare_corpus.py:517-522).  The overshoot past
    the target depends on the exact document sequence, so equality is strong evidence of
    identical training text.  When all 23 manifests exist, the recipe is also regenerated with
    the ported ``publish_recipes`` and compared byte for byte with the checked-in file.
  * ``plans/zett-modal-20k-inputs.json``: compressed size of each original ``eval/<bucket>.jsonl.gz``
    (equal only for identical content written by the same zlib at level 9 with the same
    temporary file name; the SHA-256 cannot match because of the gzip timestamp).
  * ``reports/natural-transfer-20k/results.json.gz`` ``preparation.corpora``: number of documents in
    five original held-out files (web-en, code, math-en, medical-en, legal-en).

Held-out documents (bucket, document id; optionally text SHA-256 and length bounds)
  * ``reports/frozen-policy-search-20k/reward_table.json.gz`` and ``reports/corrected-pool-check``
    (576 passages, 576 documents, ``document_sha256`` = SHA-256 of the full document text,
    scripts/original_pool_check.py:58; window end = min(len(text), 4096));
  * ``reports/natural-model-comparison-20k/results.json.gz``, ``reports/natural-transfer-20k/
    {results,initial-results}.json.gz`` (1,016 documents with ``document_sha256``,
    scripts/natural_transfer.py:57,100) and ``reports/random-pool-50/results.json.gz``;
  * ``initial-results ... preparation.rejected`` (documents shorter than 1,024 characters,
    scripts/natural_transfer.py:64);
  * ids only: ``scripts/extract_corpus_samples.py`` EXCERPTS (with the excerpt's end offset as a
    minimum length), ``reports/rl-pool-sweep-20k/spec.json`` panels, ``reports/rl-tokenizer-
    selection-20k/data/evaluation/{panel-spec,broad-screen,rl-per-document}.json`` (window ids
    ``<id>::chars-N`` mean the document has at least N + 256 characters,
    scripts/uniform_trajectory.py:153-156).

Text excerpts (must occur verbatim in the document)
  * ``reports/natural-transfer-20k/examples.json`` (prefix + continuation at ``absolute_cut_chars``),
    ``reports/rl-pool-passages-20k/passages.json`` and ``reports/context-eval-explainer/example.json``.

The text the trainer read (``plans/zett-modal-20k-inputs.json``; corpus-v1/readme-v1 of the 2026-11-13 review)
  * ``corpus/<bucket>/train.jsonl.gz`` (size of the 23 ORIGINAL staged training files) and
    ``corpus/manifest.json`` (size and SHA-256).  :func:`joint_file` re-creates each staged file from the
    regenerated pool with the original compact-copy rule (scripts/modal_tokenizer_pool_transfer.py:102-110
    and scripts/tokenizer_pool.py:112-131): text-only records ``{"text": ...}`` in pool order up to the
    recipe quota, the crossing document cut with ``encoded[:remaining].decode('utf-8', errors='ignore')``
    (the byte counter follows the written text, so after a split multi-byte character a few bytes come from
    later documents; empty cuts are skipped), gzip text mode, level 1, header name ``train.jsonl.partial``.
    The rebuilt original ``corpus/manifest.json`` (document counts and raw bytes of all 23 files, plus the
    sizes and SHA-256 of the original files) must hash to the pinned SHA-256: a hash anchor of every
    bucket's document count that does not depend on zlib.  The compressed size of each re-created file must
    equal the original size (depends on the deflate implementation: compared only when the zlib known-answer
    test passes).
  * ``eval/<bucket>.jsonl.gz`` SHA-256: a regenerated held-out file with the original gzip header time
    (``hyak/corpus/reference/original_gzip_times.json``) written into bytes 4-7 must be the original file
    byte for byte (zlib-dependent, like the sizes).

Reference content hashes (``hyak/corpus/reference/reference_content_hashes.json``)
  * SHA-256 and line count of the DECOMPRESSED ``train.jsonl.gz`` and ``eval.jsonl.gz`` of all 23 buckets of
    the full-size build made off-cluster on 2026-11-13 that matched every anchor above (independent of
    gzip time and zlib).

Verdict: ``exact`` (``content-identical to every recorded anchor``) needs every comparable anchor; anchors
that depend on the zlib build are reported as "not comparable" when the known-answer test fails.  Anything
else is a stop before stage-inputs and the pool (``hyak.corpus.stage_inputs`` refuses with exit 3 unless
``--accept-corpus-scorecard "<reason>"`` is given).

Not covered here (other stages): ``scored_tokens`` of ``uniform.csv`` (hyak.train.replay_check)
and tokenizer fingerprints (hyak.pool.scorecard); the deleted plan text of gaps.md N3 records
that the original pool's ``documents_sha256`` matched, but the hash values themselves are not in
the repository.

Usage::

    python -m hyak.corpus.verify [--corpus-root DIR] [--output-dir DIR] [--workers N] [--skip-train-files]

Writes ``scorecard.json`` and ``scorecard.txt`` (default: the corpus root, plus a copy in the
corpus monitor directory, a ``scorecard`` event and a ``scorecard`` field in the corpus status) and
prints the text.  Reading every training file takes about 10 core-minutes for the full corpus
(``--workers`` buckets in parallel; ``--skip-train-files`` skips that part, and the verdict is then never
exact).  It only reports: the exit code is 0 whenever the scorecard was written, also when anchors do not
match; 2 for usage/configuration errors.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import zlib

from hyak.common.config import ConfigError, load_config, repo_root
from hyak.common.io import atomic_write_json, atomic_write_text, read_json, sha256_file, utc_now
from hyak.common.paths import STAGE_CORPUS, STAGE_PROGRESS, Paths
from hyak.corpus import sources

#: Anchor files (repo-relative).  Missing files are reported, not fatal.
RECIPE_FILE = 'data/balanced-full-v3.json'
INPUTS_FILE = 'plans/zett-modal-20k-inputs.json'
EXCERPTS_FILE = 'scripts/extract_corpus_samples.py'
REWARD_TABLE = 'reports/frozen-policy-search-20k/reward_table.json.gz'
CORRECTED_CHECK = 'reports/corrected-pool-check/results.json.gz'
NATURAL_COMPARISON = 'reports/natural-model-comparison-20k/results.json.gz'
NATURAL_RESULTS = 'reports/natural-transfer-20k/results.json.gz'
NATURAL_INITIAL = 'reports/natural-transfer-20k/initial-results.json.gz'
RANDOM_POOL = 'reports/random-pool-50/results.json.gz'
SWEEP_SPEC = 'reports/rl-pool-sweep-20k/spec.json'
PANEL_SPEC = 'reports/rl-tokenizer-selection-20k/data/evaluation/panel-spec.json'
BROAD_SCREEN = 'reports/rl-tokenizer-selection-20k/data/evaluation/broad-screen.json'
RL_PER_DOCUMENT = 'reports/rl-tokenizer-selection-20k/data/evaluation/rl-per-document.json'
NATURAL_EXAMPLES = 'reports/natural-transfer-20k/examples.json'
PASSAGES = 'reports/rl-pool-passages-20k/passages.json'
EXPLAINER = 'reports/context-eval-explainer/example.json'
#: Reference anchors kept next to this module (see the module docstring).
REFERENCE_DIR = Path(__file__).resolve().parent / 'reference'
GZIP_TIMES_FILE = 'hyak/corpus/reference/original_gzip_times.json'
CONTENT_HASHES_FILE = 'hyak/corpus/reference/reference_content_hashes.json'
#: Name stored in the gzip header of the original staged training files (the temporary name of
#: scripts/modal_tokenizer_pool_transfer.py:103-107).
JOINT_HEADER_NAME = 'train.jsonl.partial'
#: Format version of scorecard.json (hyak.corpus.stage_inputs reads it).
SCORECARD_FORMAT = 2
#: Verdict prefix of an exact scorecard (job scripts and the run guide look for it).
EXACT_PREFIX = 'content-identical to every recorded anchor'
#: SHA-256 of the raw deflate streams of :func:`_kat_sample` (zlib 1.2.12 and pyarrow 26.0.1's bundled zlib,
#: SCRATCH/review/corpus-verify/zlib_kat.out): levels 9 and 1, and level 1 closed as gzip text mode closes.
ZLIB_KNOWN_ANSWERS = {
    'level9': '1b69e4e372fed0ad4bb5480be1d3065e5eeb4078e03f3b629add0b80fac5bc98',
    'level1': '45f922c746cb57a223b59933742938cd43f7e6fd164973e871330910d2131063',
    'level1_sync_finish': '1d3dc8829080f88b6103f0e96cf870e2eeda3401f4ef2a0988c5b93c0e957b34',
}
KAT_SAMPLE_SHA256 = '4250cb6367c721d2b14f36d33668cae82ec8a972548ac28b27768c57f9228030'

_WINDOW_RE = re.compile(r'(.+)::chars-(\d+)')
_SHORT_REASON = 'less than1024 characters for two disjoint windows'
_MAX_EXAMPLES = 8


# --- anchors ------------------------------------------------------------------------------------------
class Anchors:
    """Everything the repository records about the original pools (see the module docstring)."""

    def __init__(self):
        self.available = {}              # bucket -> original counts.train_bytes
        self.eval_files = {}             # bucket -> {"bytes", "sha256"} of the original eval file
        self.eval_docs = {}              # bucket -> original number of held-out documents
        self.documents = defaultdict(dict)  # bucket -> id -> {"sha256": set, "min_chars", "max_chars", "exact_chars", "sources": set}
        self.excerpts = []               # {"bucket", "id", "text", "offset" (or None), "source"}
        self.files = {}                  # repo-relative anchor file -> sha256 or None (missing)
        self.problems = []               # inconsistencies inside the anchor files themselves
        self.quota = {}                  # bucket -> recipe train_target_bytes (size of the staged training text)
        self.train_files = {}            # bucket -> {"bytes", "sha256"} of the ORIGINAL staged corpus/<bucket>/train.jsonl.gz
        self.joint_manifest = None       # {"bytes", "sha256"} of the original corpus/manifest.json
        self.eval_gzip_mtime = {}        # bucket -> gzip header time of the original eval file
        self.content = {}                # bucket -> {"train": {...}, "eval": {...}} decompressed fingerprints

    def document(self, bucket, identifier, source):
        entry = self.documents[bucket].setdefault(identifier, {'sha256': set(), 'min_chars': 0, 'max_chars': None,
                                                               'exact_chars': None, 'sources': set()})
        entry['sources'].add(source)
        return entry

    def add(self, bucket, identifier, source, *, sha256=None, min_chars=0, max_chars=None, exact_chars=None):
        if bucket not in sources.bucket_specs():
            self.problems.append(f'{source}: unknown bucket {bucket!r} for id {identifier!r}')
            return
        if not isinstance(identifier, str) or not identifier:
            self.problems.append(f'{source}: empty document id in bucket {bucket}')
            return
        entry = self.document(bucket, identifier, source)
        if sha256:
            entry['sha256'].add(sha256)
            if len(entry['sha256']) > 1:
                self.problems.append(f'{bucket}/{identifier}: conflicting document_sha256 values in the anchor files')
        entry['min_chars'] = max(entry['min_chars'], int(min_chars or 0))
        if max_chars is not None:
            entry['max_chars'] = max_chars if entry['max_chars'] is None else min(entry['max_chars'], max_chars)
        if exact_chars is not None:
            if entry['exact_chars'] not in (None, exact_chars):
                self.problems.append(f'{bucket}/{identifier}: conflicting document lengths in the anchor files')
            entry['exact_chars'] = exact_chars

    def counts(self) -> dict:
        return {'buckets_with_document_anchors': len(self.documents),
                'document_ids': sum(len(v) for v in self.documents.values()),
                'document_sha256': sum(1 for v in self.documents.values() for e in v.values() if e['sha256']),
                'excerpts': len(self.excerpts), 'train_available_bytes': len(self.available),
                'eval_file_sizes': len(self.eval_files), 'eval_document_counts': len(self.eval_docs),
                'original_train_files': len(self.train_files), 'original_corpus_manifest': int(bool(self.joint_manifest)),
                'eval_gzip_times': len(self.eval_gzip_mtime), 'reference_content_hashes': len(self.content)}


def _load(repo: Path, relative: str, anchors: Anchors):
    path = repo / relative
    if not path.is_file():
        anchors.files[relative] = None
        anchors.problems.append(f'anchor file {relative} is missing')
        return None
    anchors.files[relative] = sha256_file(path)
    if relative.endswith('.py'):
        return path.read_text(encoding='utf-8')
    with (gzip.open(path, 'rt', encoding='utf-8') if relative.endswith('.gz') else open(path, encoding='utf-8')) as handle:
        return json.load(handle)


def _window_id(identifier: str):
    """``'<id>::chars-N'`` -> ``(<id>, N + 256)`` (scripts/uniform_trajectory.py:153-156); else ``(id, 0)``."""
    match = _WINDOW_RE.fullmatch(identifier)
    if match:
        return match.group(1), int(match.group(2)) + 256
    return identifier, 0


def _panel_rows(anchors, rows, source, exact_window=False):
    for row in rows or []:
        end = int(row.get('source_window_end') or 0)
        if exact_window:  # original_pool_check.py: window = text[:4096]; source_window_end = len(window)
            anchors.add(row['domain'], row['source_document_id'], source, sha256=row.get('document_sha256'),
                        min_chars=end, exact_chars=end if end < 4096 else None)
        else:
            anchors.add(row['domain'], row['source_document_id'], source, sha256=row.get('document_sha256'),
                        min_chars=end)


def load_anchors(repo=None) -> Anchors:
    """Read every anchor from the repository (see the module docstring for the list)."""
    repo = Path(repo or repo_root())
    anchors = Anchors()
    recipe = _load(repo, RECIPE_FILE, anchors)
    if recipe:
        anchors.available = {b['name']: int(b['train_available_bytes']) for b in recipe['buckets']}
        anchors.quota = {b['name']: int(b['train_target_bytes']) for b in recipe['buckets']}
    inputs = _load(repo, INPUTS_FILE, anchors)
    if inputs:
        for bucket in sources.bucket_specs():
            item = inputs['files'].get(f'eval/{bucket}.jsonl.gz')
            if item:
                anchors.eval_files[bucket] = {'bytes': int(item['bytes']), 'sha256': item['sha256']}
            item = inputs['files'].get(f'corpus/{bucket}/train.jsonl.gz')
            if item:
                anchors.train_files[bucket] = {'bytes': int(item['bytes']), 'sha256': item['sha256']}
        item = inputs['files'].get('corpus/manifest.json')
        if item:
            anchors.joint_manifest = {'bytes': int(item['bytes']), 'sha256': item['sha256']}
    data = _load(repo, GZIP_TIMES_FILE, anchors)
    if data:
        anchors.eval_gzip_mtime = {b: int(t) for b, t in data['times'].items() if b in sources.bucket_specs()}
    data = _load(repo, CONTENT_HASHES_FILE, anchors)
    if data:
        anchors.content = {b: {split: dict(row[split]) for split in ('train', 'eval')}
                           for b, row in data['buckets'].items() if b in sources.bucket_specs()}
    text = _load(repo, EXCERPTS_FILE, anchors)
    if text:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'EXCERPTS' for t in node.targets):
                for bucket, (identifier, _start, end) in ast.literal_eval(node.value).items():
                    anchors.add(bucket, identifier, EXCERPTS_FILE, min_chars=end)
    for relative, exact in ((REWARD_TABLE, True), (CORRECTED_CHECK, True)):
        data = _load(repo, relative, anchors)
        if data:
            rows = data['rows'] if 'rows' in data else data['preparation']['metadata']
            _panel_rows(anchors, rows, relative, exact_window=exact)
            if relative == CORRECTED_CHECK:
                for split in ('screen', 'confirm'):
                    old = data.get('old', {}).get(split, {}).get('metadata', {})
                    for bucket, identifier in zip(old.get('domains', []), old.get('document_ids', [])):
                        anchors.add(bucket, identifier, relative + ':old')
    data = _load(repo, NATURAL_COMPARISON, anchors)
    if data:
        _panel_rows(anchors, data['metadata'], NATURAL_COMPARISON)
    for relative in (NATURAL_RESULTS, NATURAL_INITIAL):
        data = _load(repo, relative, anchors)
        if not data:
            continue
        preparation = data.get('preparation', {})
        _panel_rows(anchors, preparation.get('metadata'), relative)
        for row in preparation.get('rejected', []) or []:
            anchors.add(row['domain'], row['id'], relative + ':rejected',
                        max_chars=1023 if row.get('reason') == _SHORT_REASON else None)
        for bucket, corpus in (preparation.get('corpora') or {}).items():
            count = int(corpus['source_documents'])
            if anchors.eval_docs.get(bucket, count) != count:
                anchors.problems.append(f'{relative}: conflicting held-out document count for {bucket}')
            anchors.eval_docs[bucket] = count
        del data
    data = _load(repo, RANDOM_POOL, anchors)
    if data:
        _panel_rows(anchors, data['preparation']['metadata'], RANDOM_POOL)
    data = _load(repo, SWEEP_SPEC, anchors)
    if data:
        for split in ('screen', 'confirm'):
            panel = data['panels'][split]
            for bucket, identifier in zip(panel['domains'], panel['document_ids']):
                anchors.add(bucket, identifier, SWEEP_SPEC)
    data = _load(repo, PANEL_SPEC, anchors)
    if data:
        for case in data['cases']:
            for identifier in list(case.get('document_ids', [])) + list(case.get('cluster_ids', [])):
                base, minimum = _window_id(identifier)
                anchors.add(case['domain'], base, PANEL_SPEC, min_chars=minimum)
    data = _load(repo, BROAD_SCREEN, anchors)
    if data:
        for row in data['rows']:
            if 'domain' in row:
                for identifier in row.get('document_ids', []):
                    base, minimum = _window_id(identifier)
                    anchors.add(row['domain'], base, BROAD_SCREEN, min_chars=minimum)
    data = _load(repo, RL_PER_DOCUMENT, anchors)
    if data:
        for key, case in data['cases'].items():
            bucket = key.rsplit('--', 1)[-1]
            for identifier in (case.get('input_audit') or {}).get('document_ids', []):
                base, minimum = _window_id(identifier)
                anchors.add(bucket, base, RL_PER_DOCUMENT, min_chars=minimum)
    data = _load(repo, NATURAL_EXAMPLES, anchors)
    for row in data or []:
        text = row['prefix'] + row['continuation']
        anchors.excerpts.append({'bucket': row['domain'], 'id': row['source_document_id'], 'text': text,
                                 'offset': int(row['absolute_cut_chars']) - len(row['prefix']),
                                 'source': NATURAL_EXAMPLES})
        anchors.add(row['domain'], row['source_document_id'], NATURAL_EXAMPLES,
                    min_chars=int(row['absolute_cut_chars']) + len(row['continuation']))
    data = _load(repo, PASSAGES, anchors)
    for row in (data or {}).get('examples', []):
        anchors.excerpts.append({'bucket': row['domain'], 'id': row['id'], 'text': row['prefix'] + row['continuation'],
                                 'offset': None, 'source': PASSAGES})
        anchors.add(row['domain'], row['id'], PASSAGES)
    data = _load(repo, EXPLAINER, anchors)
    if data:
        bucket, rest = data['id'].split('/', 1)
        identifier = rest.rsplit('/window-', 1)[0]
        anchors.excerpts.append({'bucket': bucket, 'id': identifier, 'text': data['prefix'] + data['suffix'],
                                 'offset': None, 'source': EXPLAINER})
        anchors.add(bucket, identifier, EXPLAINER)
    # Sanity of the anchors themselves: hash-held-out buckets must satisfy the held-out rule.
    for bucket, documents in anchors.documents.items():
        size, spec = sources.bucket_specs()[bucket]
        if sources.source_info(spec)[2] in sources.SEPARATE_EVAL_POLICIES:
            continue
        for identifier in documents:
            try:
                key = sources.split_key_of(spec['kind'], identifier)
            except ValueError as exc:
                anchors.problems.append(f'{bucket}: {exc}')
                continue
            if not sources.held_out(key):
                anchors.problems.append(f'{bucket}/{identifier}: anchor id fails held_out(); anchor extraction is wrong')
    return anchors




# --- zlib known-answer test ------------------------------------------------------------------------------
def _kat_sample() -> bytes:
    """A fixed 3.97 MB buffer of JSON lines (ASCII and multi-byte UTF-8); SCRATCH/review/corpus-verify/zlib_kat.py."""
    words = ['alpha', 'beta', 'gamma', 'δέλτα', 'эпсилон', '中文维基百科', 'العربية', 'हिन्दी', '日本語のテキスト', 'café', 'naïve',
             'tokenizer', 'hypernetwork', '0123456789', 'The quick brown fox jumps over the lazy dog.']
    lines = []
    state = 12345
    for i in range(30000):
        parts = []
        for _ in range(8):
            state = (state * 1103515245 + 12345) % (1 << 31)
            parts.append(words[state % len(words)])
        lines.append(json.dumps({'text': ' '.join(parts) + f' #{i}'}, ensure_ascii=False) + '\n')
    return ''.join(lines).encode('utf-8')


def _deflate(data: bytes, level: int, text_mode_close: bool = False) -> bytes:
    compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
    out = compressor.compress(data)
    if text_mode_close:
        out += compressor.flush(zlib.Z_SYNC_FLUSH)
    return out + compressor.flush(zlib.Z_FINISH)


def zlib_known_answer() -> dict:
    """Whether this Python's zlib writes the deflate streams of the zlib that wrote the original files.

    Compressed sizes and the held-out byte identity depend on the deflate implementation; they are only
    compared when this test passes (``comparable``).  Content anchors never depend on it.
    """
    data = _kat_sample()
    digests = {'level9': hashlib.sha256(_deflate(data, 9)).hexdigest(),
               'level1': hashlib.sha256(_deflate(data, 1)).hexdigest(),
               'level1_sync_finish': hashlib.sha256(_deflate(data, 1, True)).hexdigest()}
    sample = hashlib.sha256(data).hexdigest()
    return {'zlib_version': zlib.ZLIB_VERSION, 'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION, 'sample_sha256': sample,
            'digests': digests, 'comparable': sample == KAT_SAMPLE_SHA256 and digests == ZLIB_KNOWN_ANSWERS}


# --- the original staged training file ------------------------------------------------------------------------
class _ByteCounter(io.RawIOBase):
    """A write-only sink that only counts bytes (the re-created file is never stored)."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def writable(self):
        return True

    def write(self, data):
        self.count += len(data)
        return len(data)


def joint_file(pool_path, quota: int) -> dict:
    """Re-create the original staged ``corpus/<bucket>/train.jsonl.gz`` of one bucket from its pool file.

    Rule of the original compact copy (module docstring): records ``{"text": ...}`` in pool order until
    ``quota`` raw text bytes; the crossing document is cut at the byte quota with ``errors='ignore'``;
    the byte counter follows the written text, so a cut inside a multi-byte character continues with the
    next documents (their cuts are often empty and are skipped); gzip text mode, level 1, header name
    ``train.jsonl.partial``.  Only the size of the gzip stream is kept.  The same pass computes the SHA-256
    and line count of the whole decompressed pool file (reference content anchor).

    Returns ``{"complete", "documents", "raw_bytes", "compressed_bytes", "empty_cuts_skipped",
    "documents_after_crossing", "pool_lines", "pool_decompressed_sha256"}``; ``complete`` is False when the
    pool file holds less text than the quota.
    """
    sink = _ByteCounter()
    binary = gzip.GzipFile(filename=JOINT_HEADER_NAME, mode='wb', compresslevel=1, fileobj=sink, mtime=0)
    writer = io.TextIOWrapper(binary, encoding='utf-8', newline='\n')
    digest = hashlib.sha256()
    used = documents = skipped = later = lines = 0
    crossed = complete = False
    try:
        with gzip.open(pool_path, 'rb') as source:
            for raw in source:
                digest.update(raw)
                lines += 1
                if complete:
                    continue
                text = json.loads(raw)['text']
                remaining = quota - used
                encoded = text.encode('utf-8')
                if len(encoded) > remaining:
                    text = encoded[:remaining].decode('utf-8', errors='ignore')
                    encoded = text.encode('utf-8')
                    later += crossed
                    crossed = True
                if text:
                    writer.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    documents += 1
                else:
                    skipped += 1
                used += len(encoded)
                if used >= quota:
                    complete = True
                    writer.close()
    finally:
        if not writer.closed:
            writer.close()
    return {'complete': complete, 'documents': documents, 'raw_bytes': used, 'compressed_bytes': sink.count,
            'empty_cuts_skipped': skipped, 'documents_after_crossing': later, 'pool_lines': lines,
            'pool_decompressed_sha256': digest.hexdigest()}


def rebuild_joint_manifest(joint: dict, anchors: Anchors) -> dict:
    """The original ``corpus/manifest.json`` rebuilt from the re-created files, compared with its pinned hash.

    Entry per bucket: ``{compressed_bytes, documents, name, quota_bytes, raw_bytes, sha256}`` with the size and
    SHA-256 of the ORIGINAL file (from the inputs plan) and the document count and raw bytes of the
    re-created one; ``json.dumps(indent=2, sort_keys=True) + "\\n"``.  Equal SHA-256 proves every bucket's
    document count and that every original file held exactly its quota.
    """
    if anchors.joint_manifest is None:
        return {'rebuilt': False, 'reason': f'{INPUTS_FILE} has no corpus/manifest.json entry'}
    entries = {}
    for name, quota in anchors.quota.items():
        row, original = joint.get(name), anchors.train_files.get(name)
        if row is None or original is None:
            return {'rebuilt': False, 'reason': f'{name}: training file not read' if row is None
                    else f'{name}: no original training-file record'}
        if not row['complete']:
            return {'rebuilt': False, 'reason': f'{name}: the pool file holds only {row["raw_bytes"]:,} of the '
                                                f'{quota:,} quota bytes'}
        entries[name] = {'compressed_bytes': original['bytes'], 'documents': row['documents'], 'name': name,
                         'quota_bytes': quota, 'raw_bytes': row['raw_bytes'], 'sha256': original['sha256']}
    data = (json.dumps(entries, indent=2, sort_keys=True) + '\n').encode('utf-8')
    digest = hashlib.sha256(data).hexdigest()
    return {'rebuilt': True, 'bytes': len(data), 'sha256': digest, 'original_bytes': anchors.joint_manifest['bytes'],
            'original_sha256': anchors.joint_manifest['sha256'],
            'identical': (len(data), digest) == (anchors.joint_manifest['bytes'], anchors.joint_manifest['sha256'])}


def eval_fingerprint(path, gzip_mtime=None, original=None) -> dict:
    """Decompressed SHA-256 and lines of a held-out file; with the original gzip time, byte identity."""
    data = Path(path).read_bytes()
    plain = gzip.decompress(data)
    result = {'compressed_bytes': len(data), 'decompressed_sha256': hashlib.sha256(plain).hexdigest(),
              'lines': plain.count(b'\n'), 'gzip_mtime': struct.unpack('<I', data[4:8])[0] if len(data) >= 8 else None}
    if gzip_mtime is not None and original is not None:
        patched = bytearray(data)
        if len(patched) >= 8:
            patched[4:8] = struct.pack('<I', gzip_mtime)
        result['identical_with_original_time'] = (len(patched) == original['bytes']
                                                  and hashlib.sha256(patched).hexdigest() == original['sha256'])
    return result


def measure_bucket(directory, quota, gzip_mtime=None, original_eval=None, train_files=True) -> dict:
    """The expensive per-bucket measurements (a process-pool task): held-out fingerprint, re-created training file."""
    directory = Path(directory)
    result = {}
    eval_path = directory / 'eval.jsonl.gz'
    if eval_path.is_file():
        result['eval'] = eval_fingerprint(eval_path, gzip_mtime, original_eval)
    train_path = directory / 'train.jsonl.gz'
    if train_files and quota and train_path.is_file():
        result['joint'] = joint_file(train_path, quota)
    return result


# --- the regenerated corpus ----------------------------------------------------------------------------
def index_eval(path: Path, keep_text=()):
    """``{id: [(sha256(text), len(text)), ...]}`` of a held-out file plus the texts of ids in ``keep_text``."""
    index, texts, docs = defaultdict(list), {}, 0
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record['text']
            index[record['id']].append((hashlib.sha256(text.encode()).hexdigest(), len(text)))
            if record['id'] in keep_text:
                texts[record['id']] = text
            docs += 1
    return index, texts, docs


def score_bucket(corpus_root: Path, bucket: str, anchors: Anchors, measured=None) -> dict:
    size, spec = sources.bucket_specs()[bucket]
    directory = Path(corpus_root) / f'{size}-v1' / bucket
    row = {'size': size, 'directory': str(directory)}
    try:
        manifest = read_json(directory / 'manifest.json')
        row['manifest_sha256'] = sha256_file(directory / 'manifest.json')
    except FileNotFoundError:
        row['state'] = 'missing'
        return row
    except (OSError, ValueError) as exc:
        row.update(state='unreadable', error=str(exc))
        return row
    counts = manifest.get('counts', {})
    row['state'] = 'test-scale' if manifest.get('test_scale') else 'built'
    row['zlib'] = (manifest.get('hyak') or {}).get('zlib')
    available = anchors.available.get(bucket)
    row['train_bytes'] = counts.get('train_bytes')
    row['train_available_bytes'] = available
    row['train_bytes_equal'] = None if available is None else counts.get('train_bytes') == available
    row['train_bytes_delta'] = None if available is None else (counts.get('train_bytes') or 0) - available
    eval_path = directory / 'eval.jsonl.gz'
    original = anchors.eval_files.get(bucket)
    row['eval_compressed_bytes'] = eval_path.stat().st_size if eval_path.is_file() else None
    row['original_eval_compressed_bytes'] = original['bytes'] if original else None
    row['eval_size_equal'] = (None if original is None or row['eval_compressed_bytes'] is None
                              else row['eval_compressed_bytes'] == original['bytes'])
    row['eval_bytes'] = counts.get('eval_bytes')
    if not eval_path.is_file():
        row['state'] = 'incomplete'
        return row
    if measured is not None and row['state'] == 'built':
        reference = anchors.content.get(bucket) or {}
        fingerprint = measured.get('eval')
        if fingerprint is not None:
            expected = reference.get('eval')
            row['eval_content_equal'] = None if expected is None else (
                fingerprint['decompressed_sha256'] == expected['decompressed_sha256'] and fingerprint['lines'] == expected['lines'])
            row['eval_identical'] = fingerprint.get('identical_with_original_time')
            row['eval_gzip_mtime'] = fingerprint.get('gzip_mtime')
        joint = measured.get('joint')
        if joint is not None:
            expected = reference.get('train')
            row['train_content_equal'] = None if expected is None else (
                joint['pool_decompressed_sha256'] == expected['decompressed_sha256'] and joint['pool_lines'] == expected['lines'])
            original_file = anchors.train_files.get(bucket)
            row['joint'] = {key: joint[key] for key in ('complete', 'documents', 'raw_bytes', 'compressed_bytes',
                                                        'empty_cuts_skipped', 'documents_after_crossing')}
            row['joint']['original_bytes'] = original_file['bytes'] if original_file else None
            row['train_file_size_equal'] = (None if original_file is None
                                            else joint['complete'] and joint['compressed_bytes'] == original_file['bytes'])
    wanted = {e['id'] for e in anchors.excerpts if e['bucket'] == bucket}
    index, texts, docs = index_eval(eval_path, keep_text=wanted)
    row['eval_docs'] = docs
    row['original_eval_docs'] = anchors.eval_docs.get(bucket)
    row['eval_docs_equal'] = None if bucket not in anchors.eval_docs else docs == anchors.eval_docs[bucket]
    documents = anchors.documents.get(bucket, {})
    found = once = sha_total = sha_equal = length_bad = 0
    missing, sha_bad, length_examples, duplicated = [], [], [], []
    for identifier, entry in sorted(documents.items()):
        hits = index.get(identifier, [])
        if not hits:
            missing.append(identifier)
            if entry['sha256']:
                sha_total += 1
            continue
        found += 1
        if len(hits) == 1:
            once += 1
        else:
            duplicated.append(identifier)
        digest, chars = hits[0]
        if entry['sha256']:
            sha_total += 1
            if digest in entry['sha256']:
                sha_equal += 1
            else:
                sha_bad.append(identifier)
        bad = (chars < entry['min_chars'] or (entry['max_chars'] is not None and chars > entry['max_chars'])
               or (entry['exact_chars'] is not None and chars != entry['exact_chars']))
        if bad:
            length_bad += 1
            length_examples.append(f'{identifier} ({chars} chars)')
    excerpts = [e for e in anchors.excerpts if e['bucket'] == bucket]
    excerpt_equal = excerpt_offset_equal = 0
    excerpt_bad = []
    for excerpt in excerpts:
        text = texts.get(excerpt['id'])
        if text is not None and excerpt['text'] in text:
            excerpt_equal += 1
            if excerpt['offset'] is not None and text[excerpt['offset']:excerpt['offset'] + len(excerpt['text'])] == excerpt['text']:
                excerpt_offset_equal += 1
        else:
            excerpt_bad.append(excerpt['id'])
    row['anchors'] = {
        'ids_total': len(documents), 'ids_found': found, 'ids_found_once': once,
        'sha256_total': sha_total, 'sha256_equal': sha_equal, 'length_violations': length_bad,
        'excerpts_total': len(excerpts), 'excerpts_equal': excerpt_equal,
        'excerpt_offsets_total': sum(e['offset'] is not None for e in excerpts),
        'excerpt_offsets_equal': excerpt_offset_equal,
        'missing_examples': missing[:_MAX_EXAMPLES], 'sha256_mismatch_examples': sha_bad[:_MAX_EXAMPLES],
        'length_violation_examples': length_examples[:_MAX_EXAMPLES],
        'duplicated_examples': duplicated[:_MAX_EXAMPLES], 'excerpt_mismatch_examples': excerpt_bad[:_MAX_EXAMPLES],
    }
    return row


def recipe_check(corpus_root: Path, repo: Path) -> dict:
    """Regenerate ``balanced-full-v3`` from the manifests with the ported ``publish_recipes`` and compare."""
    checked_in = (repo / RECIPE_FILE).read_bytes()
    try:
        recipe, text = sources.publish_recipe_v3(corpus_root)
    except FileNotFoundError as exc:
        return {'regenerated': False, 'reason': f'missing manifest: {exc.filename}'}
    except RuntimeError as exc:
        return {'regenerated': False, 'reason': str(exc)}
    original = json.loads(checked_in)
    differences = []
    for new, old in zip(recipe['buckets'], original['buckets']):
        for key in old:
            if new.get(key) != old[key]:
                differences.append(f'{old["name"]}.{key}: {new.get(key)!r} vs {old[key]!r}')
    return {'regenerated': True, 'byte_identical': text.encode('utf-8') == checked_in,
            'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'checked_in_sha256': hashlib.sha256(checked_in).hexdigest(), 'differences': differences}


def _measure_all(corpus_root: Path, anchors: Anchors, train_files: bool, workers: int) -> dict:
    """``{bucket: measure_bucket(...)}`` for every production bucket with a manifest (``workers`` in parallel)."""
    jobs = {}
    for name, (size, _spec) in sources.bucket_specs().items():
        directory = corpus_root / f'{size}-v1' / name
        try:
            manifest = read_json(directory / 'manifest.json')
        except (OSError, ValueError):
            continue
        if manifest.get('test_scale') is not None:
            continue
        jobs[name] = (str(directory), anchors.quota.get(name), anchors.eval_gzip_mtime.get(name),
                      anchors.eval_files.get(name), train_files)
    if workers <= 1 or len(jobs) <= 1:
        return {name: measure_bucket(*arguments) for name, arguments in jobs.items()}
    # Largest training files first, so the slowest bucket does not start last.
    order = sorted(jobs, key=lambda n: -(anchors.quota.get(n) or 0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {name: pool.submit(measure_bucket, *jobs[name]) for name in order}
        return {name: futures[name].result() for name in jobs}


def scorecard(corpus_root, repo=None, anchors=None, *, train_files=True, workers=1) -> dict:
    repo = Path(repo or repo_root())
    anchors = anchors or load_anchors(repo)
    corpus_root = Path(corpus_root)
    kat = zlib_known_answer()
    measured = _measure_all(corpus_root, anchors, train_files, workers)
    buckets = {name: score_bucket(corpus_root, name, anchors, measured.get(name)) for name in sources.bucket_specs()}
    built = [r for r in buckets.values() if r['state'] in ('built', 'test-scale')]
    production = [r for r in built if r['state'] == 'built']
    totals = {
        'buckets_built': len(built),
        'test_scale_buckets': sum(r['state'] == 'test-scale' for r in buckets.values()),
        'train_bytes_equal': sum(bool(r.get('train_bytes_equal')) for r in built),
        'eval_size_equal': sum(bool(r.get('eval_size_equal')) for r in built),
        'eval_docs_equal': sum(bool(r.get('eval_docs_equal')) for r in built),
        'eval_docs_anchors': sum(r.get('eval_docs_equal') is not None for r in built),
        'eval_identical': sum(bool(r.get('eval_identical')) for r in production),
        'eval_content_equal': sum(bool(r.get('eval_content_equal')) for r in production),
        'train_content_equal': sum(bool(r.get('train_content_equal')) for r in production),
        'train_file_size_equal': sum(bool(r.get('train_file_size_equal')) for r in production),
        'train_files_read': sum('joint' in r for r in production),
    }
    for key in ('ids_total', 'ids_found', 'ids_found_once', 'sha256_total', 'sha256_equal', 'length_violations',
                'excerpts_total', 'excerpts_equal', 'excerpt_offsets_total', 'excerpt_offsets_equal'):
        totals[key] = sum(r.get('anchors', {}).get(key, 0) for r in built)
    joint = {name: measured[name]['joint'] for name in measured if 'joint' in measured[name]}
    report = {
        'format': SCORECARD_FORMAT, 'at_utc': utc_now(), 'corpus_root': str(corpus_root), 'repo': str(repo),
        'anchor_files': anchors.files, 'anchor_counts': anchors.counts(), 'anchor_problems': anchors.problems,
        'recipe': recipe_check(corpus_root, repo) if len(built) == 23 else {'regenerated': False,
                                                                            'reason': f'{len(built)}/23 buckets built'},
        'zlib': kat, 'train_files_checked': bool(train_files),
        'original_corpus_manifest': (rebuild_joint_manifest(joint, anchors) if train_files else
                                     {'rebuilt': False, 'reason': 'skipped (--skip-train-files)'}),
        'buckets': buckets, 'totals': totals,
        'bucket_manifests': {name: row.get('manifest_sha256') for name, row in buckets.items()},
    }
    report['checks'] = checks(report)
    report['verdict'] = verdict(report)
    report['exact'] = report['verdict'].startswith(EXACT_PREFIX)
    return report


def checks(report) -> dict:
    """``{anchor: {"label", "equal", "total", "comparable", "skipped"}}`` behind the verdict.

    ``comparable`` is False for the zlib-dependent anchors when the known-answer test fails; ``skipped``
    marks anchors not measured (``--skip-train-files``).  ``exact`` needs ``equal == total`` for every
    comparable anchor and nothing skipped.
    """
    t = report['totals']
    zlib_ok = bool((report.get('zlib') or {}).get('comparable', True))
    train_read = bool(report.get('train_files_checked', True))
    joint = report.get('original_corpus_manifest') or {}

    def item(label, equal, total, comparable=True, skipped=False):
        return {'label': label, 'equal': int(equal), 'total': int(total), 'comparable': comparable, 'skipped': skipped}

    return {
        'train_bytes': item('train bytes (counts.train_bytes)', t['train_bytes_equal'], 23),
        'recipe': item('regenerated recipe byte-identical', bool((report.get('recipe') or {}).get('byte_identical')), 1),
        'held_out_ids': item('held-out ids found', t['ids_found'], t['ids_total']),
        'document_sha256': item('held-out document SHA-256', t['sha256_equal'], t['sha256_total']),
        'excerpts': item('text excerpts', t['excerpts_equal'], t['excerpts_total']),
        'document_lengths': item('held-out document lengths', t['ids_found'] - t['length_violations'], t['ids_found']),
        'eval_document_counts': item('held-out document counts', t['eval_docs_equal'], t['eval_docs_anchors']),
        'eval_content': item('held-out content (reference hash)', t.get('eval_content_equal', 0), 23),
        'train_content': item('training content (reference hash)', t.get('train_content_equal', 0), 23,
                              skipped=not train_read),
        'original_corpus_manifest': item('original corpus/manifest.json rebuilt', bool(joint.get('identical')), 1,
                                         skipped=not train_read),
        'original_train_file_sizes': item('original training-file sizes', t.get('train_file_size_equal', 0), 23,
                                          comparable=zlib_ok, skipped=not train_read),
        'eval_file_sizes': item('held-out file sizes', t['eval_size_equal'], 23, comparable=zlib_ok),
        'eval_files_identical': item('held-out files identical (original gzip time)', t.get('eval_identical', 0), 23,
                                     comparable=zlib_ok),
    }


def verdict(report) -> str:
    t = report['totals']
    if t['buckets_built'] < 23:
        return f'incomplete: {t["buckets_built"]}/23 buckets built'
    if t['test_scale_buckets']:
        return 'test-scale corpus: byte counts cannot match; only the first held-out documents can be compared'
    found = report.get('checks') or checks(report)
    failing = [c for c in found.values() if c['comparable'] and not c['skipped'] and c['equal'] != c['total']]
    skipped = [c['label'] for c in found.values() if c['skipped']]
    if failing:
        return 'differs from the original: ' + '; '.join(f'{c["label"]} {c["equal"]:,}/{c["total"]:,}' for c in failing)
    if skipped:
        return ('not checked completely: ' + ', '.join(skipped) + ' skipped (--skip-train-files); rerun '
                'python -m hyak.corpus.verify without it')
    not_comparable = [c['label'] for c in found.values() if not c['comparable']]
    return (f'{EXACT_PREFIX} (train bytes and training content of all 23 buckets, the original corpus manifest, '
            'regenerated recipe, all held-out documents, hashes and excerpts'
            + (f'; not comparable with this zlib build: {", ".join(not_comparable)}' if not_comparable else '') + ')')


def _yes(value) -> str:
    return '-' if value is None else ('yes' if value else 'NO')


def render(report) -> str:
    t = report['totals']
    lines = [f'Corpus fidelity scorecard  {report["at_utc"]}', f'corpus root: {report["corpus_root"]}',
             f'verdict: {report["verdict"]}',
             f'expected: {EXACT_PREFIX} (reproduced at full scale off-cluster on 2026-11-13 with datasets 5.2.0, '
             'pyarrow 26.0.1, huggingface-hub 2.4.0, hf-xet 1.8.0, fsspec 2026.9.0, requests 2.34.2, urllib3 2.8.0)',
             ('anything else: STOP before stage-inputs and the pool (hyak/README.md section 10); stage-inputs refuses '
              'it with exit 3 unless --accept-corpus-scorecard "<reason>" is given') if not report.get('exact') else
             'OK: stage-inputs may run', '',
             f'{"bucket":11s} {"state":10s} {"train bytes":>14s} {"original":>14s} {"delta":>9s} '
             f'{"eval gz":>9s} {"orig gz":>9s} {"docs":>6s} {"orig":>6s}  ids found  sha equal  excerpts']
    for name, r in report['buckets'].items():
        if r['state'] in ('missing', 'unreadable', 'incomplete'):
            lines.append(f'{name:11s} {r["state"]:10s}')
            continue
        a = r.get('anchors', {})

        def fmt(value, width):
            return f'{value:>{width},}' if isinstance(value, int) and not isinstance(value, bool) else f'{"-":>{width}}'
        lines.append(
            f'{name:11s} {r["state"]:10s} {fmt(r["train_bytes"], 14)} {fmt(r["train_available_bytes"], 14)} '
            f'{fmt(r["train_bytes_delta"], 9)} {fmt(r["eval_compressed_bytes"], 9)} '
            f'{fmt(r["original_eval_compressed_bytes"], 9)} {fmt(r.get("eval_docs"), 6)} {fmt(r.get("original_eval_docs"), 6)}'
            f'  {a.get("ids_found", 0):>4}/{a.get("ids_total", 0):<4} {a.get("sha256_equal", 0):>4}/{a.get("sha256_total", 0):<4}'
            f'  {a.get("excerpts_equal", 0)}/{a.get("excerpts_total", 0)}')
    lines += ['', f'{"bucket":11s} {"train text":>10s} {"eval text":>9s} {"eval file":>9s} {"staged docs":>11s} '
                  f'{"staged gz":>12s} {"original gz":>12s} {"size":>5s}']
    for name, r in report['buckets'].items():
        if r['state'] != 'built':
            continue
        joint = r.get('joint') or {}
        lines.append(f'{name:11s} {_yes(r.get("train_content_equal")):>10s} {_yes(r.get("eval_content_equal")):>9s} '
                     f'{_yes(r.get("eval_identical")):>9s} {joint.get("documents", "-"):>11} '
                     f'{joint.get("compressed_bytes", "-"):>12} {joint.get("original_bytes", "-"):>12} '
                     f'{_yes(r.get("train_file_size_equal")):>5s}')
    lines += ['', f'totals: train bytes equal {t["train_bytes_equal"]}/23; eval file size equal {t["eval_size_equal"]}/23; '
              f'eval document counts equal {t["eval_docs_equal"]}/{t["eval_docs_anchors"]}; held-out ids found '
              f'{t["ids_found"]}/{t["ids_total"]} (exactly once {t["ids_found_once"]}); document SHA-256 equal '
              f'{t["sha256_equal"]}/{t["sha256_total"]}; length violations {t["length_violations"]}; excerpts '
              f'{t["excerpts_equal"]}/{t["excerpts_total"]} (at the recorded offset {t["excerpt_offsets_equal"]}/'
              f'{t["excerpt_offsets_total"]})',
              f'        training content equal {t.get("train_content_equal", 0)}/23; held-out content equal '
              f'{t.get("eval_content_equal", 0)}/23; held-out files identical with the original gzip time '
              f'{t.get("eval_identical", 0)}/23; original training-file sizes equal {t.get("train_file_size_equal", 0)}/23']
    joint = report.get('original_corpus_manifest') or {}
    if joint.get('rebuilt'):
        lines.append(f'original corpus/manifest.json rebuilt: SHA-256 {joint["sha256"]} '
                     f'{"IDENTICAL to" if joint["identical"] else "DIFFERS from"} the pinned {joint["original_sha256"]}')
    else:
        lines.append(f'original corpus/manifest.json not rebuilt: {joint.get("reason")}')
    kat = report.get('zlib') or {}
    lines.append(f'zlib {kat.get("zlib_runtime")}: known-answer test '
                 + ('passed (compressed sizes are comparable)' if kat.get('comparable') else
                    'FAILED: another deflate implementation; compressed sizes and file identity are reported as not '
                    'comparable (content anchors still apply)'))
    recipe = report['recipe']
    if recipe.get('regenerated'):
        lines.append(f'regenerated recipe byte-identical to {RECIPE_FILE}: {recipe["byte_identical"]}'
                     + (f' ({len(recipe["differences"])} differing fields, e.g. {recipe["differences"][:3]})'
                        if recipe['differences'] else ''))
    else:
        lines.append(f'recipe not regenerated: {recipe.get("reason")}')
    if report['anchor_problems']:
        lines.append(f'anchor problems ({len(report["anchor_problems"])}): {report["anchor_problems"][:5]}')
    lines += ['', 'How to read this (equal values are strong evidence, not proof; hashes exist for 9 of the 23 buckets '
              'in the stored panels, so the training-content and corpus-manifest anchors carry most of the weight):',
              '* train bytes, training content, staged docs or the original corpus manifest differ in a bucket: its '
              'training text differs from the original. Stop.',
              '* a missing held-out id or a different document hash in a hash-held-out bucket (web-en, wiki-*, code, '
              'math-en, qa-en, books-en, medical-en, legal-en): the stream differs near its start. Stop.',
              '* only the held-out anchors of an upstream-split bucket (web-es/fr/ru/zh/ja/ar/hi, papers-en) differ: '
              'only its held-out file differs; delete that bucket and rebuild it.',
              '* only compressed sizes or file identity differ, in every bucket: another zlib build, not content.',
              '* remedy for a non-exact verdict: compare $ANYTOK_ROOT/env/data.freeze.txt (pip freeze) with the '
              'validated set above, delete the bucket directories whose row differs, resubmit the corpus stage and '
              'read this scorecard again; if it still differs, report it: continuing needs the user\'s approval '
              '(AGENTS.md).',
              'A test-scale corpus can only match the first held-out documents.']
    return '\n'.join(lines) + '\n'


def record_status(report, *, cfg=None, monitor_dir=None) -> None:
    """Append a ``scorecard`` event to the corpus stage and add a ``scorecard`` field to its status.json.

    The status keeps its state, progress and rows; only a terminal status is rewritten (a running build
    owns its status).  Never raises.
    """
    summary = {'verdict': report['verdict'], 'exact': report['exact'],
               'checks': {k: f'{c["equal"]}/{c["total"]}' + ('' if c['comparable'] else ' (not comparable)')
                          + (' (skipped)' if c['skipped'] else '') for k, c in report['checks'].items()},
               'at_utc': report['at_utc'], 'corpus_root': report['corpus_root']}
    try:
        from hyak.common.status import TERMINAL_STATES, StatusWriter
        meta = STAGE_PROGRESS[STAGE_CORPUS]
        writer = StatusWriter(STAGE_CORPUS, meta['total'], meta['unit'], meta['completed_field'], cfg=cfg,
                              monitor_dir=monitor_dir)
        writer.event({'event': 'scorecard', **summary, 'totals': report['totals']})
        current = read_json(writer.status_path, None)
        if isinstance(current, dict) and current.get('state') in TERMINAL_STATES:
            carry = {k: current[k] for k in ('buckets', 'corpus_root', 'test_scale', 'total') if k in current}
            writer.update(current.get(meta['completed_field']) or 0, stage=current.get('stage'),
                          state=current['state'], error=current.get('error'), workers=[], append=False,
                          scorecard=summary, **carry)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'[hyak corpus verify] WARNING: cannot record the scorecard in the corpus status: {exc}', file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.corpus.verify', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-root', type=Path, help='default: $ANYTOK_ROOT/corpus')
    parser.add_argument('--output-dir', type=Path, help='default: the corpus root (plus a copy in the monitor dir)')
    parser.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)),
                        help='buckets measured in parallel (default: min(4, cores))')
    parser.add_argument('--skip-train-files', action='store_true',
                        help='do not read the training files (fast; the verdict is then never exact)')
    parser.add_argument('--config', type=Path, help='config file (default $HYAK_CONFIG or hyak/config.env)')
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise ValueError('--workers must be >= 1')
        cfg = load_config(args.config)
        paths = Paths.from_config(cfg)
    except (ConfigError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    corpus_root = Path(os.path.normpath(str(args.corpus_root))) if args.corpus_root else paths.corpus_root
    report = scorecard(corpus_root, train_files=not args.skip_train_files, workers=args.workers)
    text = render(report)
    outputs = [args.output_dir or corpus_root]
    if args.output_dir is None:
        outputs.append(paths.monitor_dir(STAGE_CORPUS))
    for directory in outputs:
        try:
            atomic_write_json(Path(directory) / 'scorecard.json', report)
            atomic_write_text(Path(directory) / 'scorecard.txt', text)
            print(f'[hyak corpus verify] wrote {Path(directory) / "scorecard.json"} and scorecard.txt', file=sys.stderr)
        except OSError as exc:
            print(f'[hyak corpus verify] WARNING: cannot write the scorecard to {directory}: {exc}', file=sys.stderr)
    if args.output_dir is None:
        record_status(report, cfg=cfg)
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
