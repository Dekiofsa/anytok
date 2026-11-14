"""Modal-free port of ``scripts/modal_prepare_corpus.py`` (repository commit c2b9c40).

Source file SHA-256 ``b8ae4523...781c`` (``ORIGINAL_SOURCE_SHA256``; a test pins it).  Every
function below is a textual copy of the original with the line range cited next to it.  The
original module imports ``modal`` at load time, so it cannot be imported here; its pure
functions are copied instead (DESIGN.md section 0 rule 6).

What is identical to the original, operation for operation: the constants and pinned
``SOURCE_REVISIONS``; the plans of the ``full`` / ``extra`` / ``specialty`` pools and the recipe
byte targets; the ten source adapters (dataset, config, split, file names, shard orders and
seeds, per-shard byte cuts, filters, Gutenberg chunking); ``record_fields``; the held-out rule
``held_out``; the record writer ``_write_record`` (``{"id", "text"}`` JSON lines,
``ensure_ascii=False``, gzip written by ``gzip.open(path, "wb")`` into ``train.jsonl.gz.tmp`` /
``eval.jsonl.gz.tmp`` so even the file name stored in the gzip header is the same); the bucket
builder ``prepare_bucket`` (document order, size filters, split decisions, stopping rules,
counts); ``verify_corpus``; ``publish_recipes`` (``balanced-full-v3`` part); ``probe_sources``.

Port additions, none of which can change which documents are written or their order:

* ``_open_url`` replaces the bare ``requests.get(url, stream=True, timeout=...)`` of the three
  ``requests`` adapters.  It adds ``Authorization: Bearer $HF_TOKEN`` only when ``HF_TOKEN`` is
  set (the original build used no token), and retries **the request itself** a bounded number of
  times on connection errors and HTTP 429/5xx.  A retry happens strictly before the response
  body is given to the caller, i.e. before any line of that shard has been read, so the shard is
  always read from its first byte exactly once.  A failure after the body was handed out
  propagates unchanged: the bucket fails, its temporary files are deleted, and the bucket is
  rebuilt from zero (what the original did; ``hyak.corpus.build`` may do that automatically).
  HTTP 401/403/404 raise :class:`SourceUnavailable` with a remedy.
* ``prepare_bucket`` closes the record iterators explicitly when it leaves them (datasets 5.x
  streaming can otherwise hang at interpreter exit; measured on this port's test machine), calls
  an optional ``progress`` callback, writes ``manifest.json`` atomically (same bytes as the
  original's ``write_text``), and can merge extra top-level manifest keys (``hyak`` provenance,
  ``test_scale``) that never replace an original key.
* ``verify_corpus`` / ``publish_recipe_v3`` take the corpus root as an argument instead of the
  Modal mount ``/corpus``; ``publish_recipe_v3`` still renders ``/corpus/...`` paths so the
  regenerated recipe can be compared byte for byte with ``data/balanced-full-v3.json``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, Iterator, Optional

#: SHA-256 of the ported file ``scripts/modal_prepare_corpus.py`` at commit c2b9c40.
ORIGINAL_SOURCE = 'scripts/modal_prepare_corpus.py'
ORIGINAL_SOURCE_SHA256 = 'b8ae45230f4df06f630ea279c13c1deed2416d0f50d391102565aa55fbef781c'

# --- constants: scripts/modal_prepare_corpus.py:29-59 (Modal app/volume/image lines 61-65 dropped) ---
VOLUME_NAME = "anytok-corpus"
#: The Modal mount of the corpus volume.  Here it is only used to render recipe paths
#: (``source_path`` / ``eval_path`` are ``/corpus/<size>-v1/<bucket>`` in the checked-in recipe).
ROOT = Path("/corpus")
SOURCE_REVISIONS = {
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "HuggingFaceFW/fineweb-2": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "allenai/peS2o": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
    "codeparrot/github-code-clean": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
    "HuggingFaceTB/finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594",
    "common-pile/stackexchange_filtered": "c0ac7373830c688a43fc12d1988c4b19ccd884ab",
    "common-pile/project_gutenberg_filtered": "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7",
    "common-pile/pubmed": "648b8cfc93953ca0663a9c96a8d842a91b98fb64",
    "common-pile/caselaw_access_project_filtered": "50e1961a5ed8fdddcee64c9e66e1338de8953b63",
}
NON_ENGLISH = {
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "zh": "cmn_Hani",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
}
WIKI_LANGUAGES = ("en", *NON_ENGLISH)
PILOT_BYTES = 2_000_000_000
FULL_BYTES = 10_000_000_000
EXTRA_BYTES = 2_000_000_000
EVAL_BYTES_PER_BUCKET = 5_000_000
MAX_DOCUMENT_BYTES = 1_000_000
PERMISSIVE_CODE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
CODE_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Java", "GO", "Rust", "C", "C++", "Shell"}

#: Pool sizes of the balanced-full-v3 recipe, in the order README.md:45-49 says to build them.
POOL_SIZES = ("full", "extra", "specialty")
#: Split policies whose held-out text comes from a separate upstream split (scripts/modal_prepare_corpus.py:349).
SEPARATE_EVAL_POLICIES = {"upstream-test", "upstream-validation"}

# --- port additions: request retries (see module docstring) ---------------------------------
#: Attempts of one shard request (only before its body is read).
OPEN_ATTEMPTS = 5
#: Seconds to wait before attempt 2, 3, ... (doubling, capped).
OPEN_BACKOFF_SECONDS = 15.0
OPEN_BACKOFF_CAP_SECONDS = 240.0
#: HTTP statuses worth repeating the identical request for.
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class SourceUnavailable(RuntimeError):
    """An upstream file cannot be read (gated, deleted, renamed, forbidden).  Not retried."""


def _auth_headers() -> dict:
    """``{"Authorization": "Bearer <HF_TOKEN>"}`` when ``HF_TOKEN`` is set and non-empty, else ``{}``."""
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _open_url(url: str, timeout):
    """``requests.get(url, stream=True, timeout=timeout)`` with the port additions of the module docstring.

    Returns the open ``requests.Response`` (use it as a context manager, as the original did).
    Only the request is retried; nothing of the body has been read when a retry happens.
    """
    import requests

    headers = _auth_headers()
    delay = OPEN_BACKOFF_SECONDS
    for attempt in range(1, OPEN_ATTEMPTS + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == OPEN_ATTEMPTS:
                raise
            _note_retry(url, attempt, f"{type(exc).__name__}: {exc}", delay)
            time.sleep(delay)
            delay = min(delay * 2, OPEN_BACKOFF_CAP_SECONDS)
            continue
        status = response.status_code
        if status in RETRY_STATUS and attempt < OPEN_ATTEMPTS:
            response.close()
            _note_retry(url, attempt, f"HTTP {status}", delay)
            time.sleep(delay)
            delay = min(delay * 2, OPEN_BACKOFF_CAP_SECONDS)
            continue
        if status in (401, 403, 404, 410):
            response.close()
            hint = ("the file or the pinned revision no longer exists upstream; the pinned source cannot "
                    "be rebuilt (do not substitute another revision without the user's approval)"
                    if status in (404, 410) else
                    "the dataset is gated or private now; accept its terms on huggingface.co and set "
                    "HF_TOKEN in hyak/config.env")
            raise SourceUnavailable(f"HTTP {status} for {url}: {hint}")
        return response
    raise AssertionError("unreachable")


def _note_retry(url, attempt, reason, delay):
    print(f"[hyak corpus] request attempt {attempt}/{OPEN_ATTEMPTS} for {url} failed ({reason}); "
          f"repeating the identical request in {delay:.0f} s (no data of this shard was read yet)",
          file=sys.stderr, flush=True)


# --- plans and recipe targets: scripts/modal_prepare_corpus.py:68-154 ------------------------
def bucket_plan(total_bytes: int) -> list[dict]:
    # scripts/modal_prepare_corpus.py:68-84
    specs = [{"name": "web-en", "kind": "fineweb", "train_target": total_bytes * 25 // 100}]
    specs.extend(
        {"name": f"web-{language}", "kind": "fineweb2", "language": language,
         "config": config, "train_target": total_bytes * 45 // 100 // len(NON_ENGLISH)}
        for language, config in NON_ENGLISH.items()
    )
    specs.extend(
        {"name": f"wiki-{language}", "kind": "wiki", "language": language,
         "train_target": total_bytes * 15 // 100 // len(WIKI_LANGUAGES)}
        for language in WIKI_LANGUAGES
    )
    specs.append({"name": "code", "kind": "code", "train_target": total_bytes * 10 // 100})
    specs.append({"name": "papers-en", "kind": "papers", "train_target": total_bytes * 5 // 100})
    for spec in specs:
        spec["eval_target"] = EVAL_BYTES_PER_BUCKET
    return specs


def extra_plan() -> list[dict]:
    """Additive pools for a 10 GB recipe with math, Q&A, and books."""
    # scripts/modal_prepare_corpus.py:87-96
    return [
        {"name": "math-en", "kind": "math", "train_target": 1_000_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
        {"name": "qa-en", "kind": "qa", "train_target": 500_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
        {"name": "books-en", "kind": "books", "train_target": 500_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
    ]


def specialty_plan() -> list[dict]:
    """Pools large enough for a 30% share in a 2 GB tokenizer run."""
    # scripts/modal_prepare_corpus.py:99-106
    return [
        {"name": "medical-en", "kind": "medical", "train_target": 800_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
        {"name": "legal-en", "kind": "legal", "train_target": 800_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
    ]


def balanced_recipe_targets(total_bytes: int) -> dict[str, int]:
    """Byte quotas for web 55%, wiki 10%, code 10%, papers 5%, and extras 20%."""
    # scripts/modal_prepare_corpus.py:109-126
    web_total = total_bytes * 55 // 100
    english_web = web_total * 25 // 70
    other_web = web_total - english_web
    targets = {"web-en": english_web}
    for index, language in enumerate(NON_ENGLISH):
        targets[f"web-{language}"] = other_web // 7 + (index < other_web % 7)
    wiki_total = total_bytes * 10 // 100
    for index, language in enumerate(WIKI_LANGUAGES):
        targets[f"wiki-{language}"] = wiki_total // 8 + (index < wiki_total % 8)
    targets.update({"code": total_bytes * 10 // 100,
                    "papers-en": total_bytes * 5 // 100,
                    "math-en": total_bytes * 10 // 100,
                    "qa-en": total_bytes * 5 // 100,
                    "books-en": total_bytes * 5 // 100})
    assert sum(targets.values()) == total_bytes
    return targets


def specialty_recipe_targets(total_bytes: int) -> dict[str, int]:
    """Keep prior non-web shares; allocate 5% each to medical and legal."""
    # scripts/modal_prepare_corpus.py:129-142
    targets = balanced_recipe_targets(total_bytes)
    web_names = [name for name in targets if name.startswith("web-")]
    web_total = sum(targets[name] for name in web_names)
    new_web = total_bytes * 45 // 100
    web_allocations = [new_web * targets[name] // web_total for name in web_names]
    for index in range(new_web - sum(web_allocations)):
        web_allocations[index] += 1
    targets.update(zip(web_names, web_allocations))
    targets["medical-en"] = total_bytes * 5 // 100
    targets["legal-en"] = total_bytes * 5 // 100
    assert sum(targets.values()) == total_bytes
    return targets


def plan_for_size(size: str) -> list[dict]:
    # scripts/modal_prepare_corpus.py:145-154
    if size == "pilot":
        return bucket_plan(PILOT_BYTES)
    if size == "full":
        return bucket_plan(FULL_BYTES)
    if size == "extra":
        return extra_plan()
    if size == "specialty":
        return specialty_plan()
    raise ValueError("size must be 'pilot', 'full', 'extra', or 'specialty'")


def source_info(spec: dict) -> tuple[str, str, str]:
    # scripts/modal_prepare_corpus.py:157-177
    kind = spec["kind"]
    if kind == "fineweb":
        return "HuggingFaceFW/fineweb", "sample-10BT", "hash-heldout"
    if kind == "fineweb2":
        return "HuggingFaceFW/fineweb-2", spec["config"], "upstream-test"
    if kind == "wiki":
        return "wikimedia/wikipedia", f"20231101.{spec['language']}", "hash-heldout"
    if kind == "code":
        return "codeparrot/github-code-clean", "permissive-selected-languages", "repository-hash-heldout"
    if kind == "math":
        return "HuggingFaceTB/finemath", "finemath-4plus", "url-hash-heldout"
    if kind == "qa":
        return "common-pile/stackexchange_filtered", "default", "document-hash-heldout"
    if kind == "books":
        return "common-pile/project_gutenberg_filtered", "default", "book-hash-heldout"
    if kind == "medical":
        return "common-pile/pubmed", "default", "article-hash-heldout"
    if kind == "legal":
        return "common-pile/caselaw_access_project_filtered", "default", "opinion-hash-heldout"
    return "allenai/peS2o", "v2", "upstream-validation"


def recipe_pool(name: str) -> str:
    """Pool size holding a bucket in ``balanced-full-v3`` (scripts/modal_prepare_corpus.py:514-515)."""
    return ("specialty" if name in {"medical-en", "legal-en"} else
            "extra" if name in {"math-en", "qa-en", "books-en"} else "full")


def bucket_specs() -> dict[str, tuple[str, dict]]:
    """``{bucket: (size, spec)}`` for the 23 buckets of the full/extra/specialty pools, in build order."""
    result = {}
    for size in POOL_SIZES:
        for spec in plan_for_size(size):
            result[spec["name"]] = (size, spec)
    return result


# --- source adapters: scripts/modal_prepare_corpus.py:180-301 --------------------------------
def _hf_records(repo: str, config: str, split: str) -> Iterator[dict]:
    # scripts/modal_prepare_corpus.py:180-186
    from datasets import load_dataset

    yield from load_dataset(
        repo, name=config, split=split, streaming=True,
        revision=SOURCE_REVISIONS[repo],
    )


def _paper_records(split: str) -> Iterator[dict]:
    # scripts/modal_prepare_corpus.py:189-200 (requests.get -> _open_url, see module docstring)
    revision = SOURCE_REVISIONS["allenai/peS2o"]
    name = "train-00000-of-00020.json.gz" if split == "train" else "validation-00000-of-00002.json.gz"
    url = f"https://huggingface.co/datasets/allenai/peS2o/resolve/{revision}/data/v2/{name}"
    with _open_url(url, timeout=(30, 120)) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for line in stream:
                if line:
                    yield json.loads(line)


def _code_records() -> Iterator[dict]:
    # scripts/modal_prepare_corpus.py:203-213
    from datasets import load_dataset

    revision = SOURCE_REVISIONS["codeparrot/github-code-clean"]
    base = f"https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/{revision}/data"
    for index in range(32):
        url = f"{base}/train-{index:05d}-of-00880.parquet"
        dataset = load_dataset("parquet", data_files=url, split="train", streaming=True)
        for record in dataset:
            if record.get("license", "").lower() in PERMISSIVE_CODE_LICENSES and record.get("language") in CODE_LANGUAGES:
                yield record


def _common_pile_records(repo: str, prefix: str, shard_count: int) -> Iterator[dict]:
    """Stream bounded slices across shards to avoid one site's or book range's bias."""
    # scripts/modal_prepare_corpus.py:216-235 (requests.get -> _open_url, see module docstring)
    indices = list(range(shard_count))
    random.Random(20261022).shuffle(indices)
    revision = SOURCE_REVISIONS[repo]
    for index in indices:
        name = f"{prefix}-{index:04d}.json.gz"
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
        shard_bytes = 0
        with _open_url(url, timeout=(30, 120)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for line in stream:
                    record = json.loads(line)
                    shard_bytes += len(record.get("text", "").encode("utf-8"))
                    yield record
                    if shard_bytes >= 90_000_000:
                        break


def _book_records() -> Iterator[dict]:
    # scripts/modal_prepare_corpus.py:238-251
    for book in _common_pile_records(
        "common-pile/project_gutenberg_filtered", "project_gutenberg-dolma", 15
    ):
        if book.get("metadata", {}).get("language") != "en":
            continue
        book_id = str(book.get("id", ""))
        if not book_id:
            continue
        content = book.get("text", "")
        # Keep long books while assigning every chunk of a book to one split.
        for offset in range(0, len(content), 50_000):
            yield {"id": f"gutenberg:{book_id}:{offset // 50_000}",
                   "split_key": f"gutenberg:{book_id}", "text": content[offset:offset + 50_000]}


def _specialty_records(kind: str) -> Iterator[dict]:
    """Stream a deterministic, bounded slice of each shard from HF."""
    # scripts/modal_prepare_corpus.py:254-277 (requests.get -> _open_url, see module docstring)
    if kind == "medical":
        repo, count = "common-pile/pubmed", 17
        filename = lambda i: f"data/{i:05d}_pubmedcentral.jsonl.gz"  # noqa: E731 (verbatim)
    else:
        repo, count = "common-pile/caselaw_access_project_filtered", 27
        filename = lambda i: f"CAP-Dolma-{i:04d}.json.gz"  # noqa: E731 (verbatim)
    indices = list(range(count))
    random.Random(20261023).shuffle(indices)
    for index in indices:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{SOURCE_REVISIONS[repo]}/{filename(index)}"
        shard_bytes = 0
        with _open_url(url, timeout=(30, 180)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for line in stream:
                    record = json.loads(line)
                    shard_bytes += len(record.get("text", "").encode("utf-8"))
                    yield record
                    if shard_bytes >= 100_000_000:
                        break


def records(spec: dict, split: str) -> Iterator[dict]:
    # scripts/modal_prepare_corpus.py:280-301
    kind = spec["kind"]
    if kind == "fineweb":
        yield from _hf_records("HuggingFaceFW/fineweb", "sample-10BT", "train")
    elif kind == "fineweb2":
        yield from _hf_records("HuggingFaceFW/fineweb-2", spec["config"], split)
    elif kind == "wiki":
        yield from _hf_records("wikimedia/wikipedia", f"20231101.{spec['language']}", "train")
    elif kind == "code":
        yield from _code_records()
    elif kind == "math":
        yield from _hf_records("HuggingFaceTB/finemath", "finemath-4plus", "train")
    elif kind == "qa":
        yield from _common_pile_records(
            "common-pile/stackexchange_filtered", "stackexchange-dolma", 44
        )
    elif kind == "books":
        yield from _book_records()
    elif kind in {"medical", "legal"}:
        yield from _specialty_records(kind)
    else:
        yield from _paper_records(split)


def record_fields(spec: dict, record: dict) -> tuple[str, str, str]:
    # scripts/modal_prepare_corpus.py:304-314
    if spec["kind"] == "code":
        identifier = f"{record.get('repo_name', '')}/{record.get('path', '')}"
        return record.get("code", ""), identifier, record.get("repo_name", "")
    if spec["kind"] == "qa":
        identifier = str(record.get("metadata", {}).get("provenance") or record.get("id") or "")
        return record.get("text", ""), identifier, identifier
    if spec["kind"] == "books":
        return record["text"], record["id"], record["split_key"]
    identifier = str(record.get("id") or record.get("url") or "")
    return record.get("text", ""), identifier, identifier


def held_out(key: str) -> bool:
    # scripts/modal_prepare_corpus.py:317-318
    return hashlib.sha256(key.encode("utf-8")).digest()[0] < 26


def _write_record(handle, text: str, identifier: str) -> None:
    # scripts/modal_prepare_corpus.py:321-323
    handle.write(json.dumps({"id": identifier, "text": text}, ensure_ascii=False).encode("utf-8"))
    handle.write(b"\n")


def _sha256(path: Path) -> str:
    # scripts/modal_prepare_corpus.py:326-331
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_key_of(kind: str, identifier: str) -> str:
    """The held-out split key of a *written* record id (inverse of ``record_fields`` per kind).

    code: id ``<owner>/<repo>/<path>`` -> ``<owner>/<repo>`` (``repo_name``); books:
    ``gutenberg:<book>:<chunk>`` -> ``gutenberg:<book>``; every other kind: the id itself.
    Used only by verification (the builder uses ``record_fields``).
    """
    if kind == "code":
        parts = identifier.split("/")
        if len(parts) < 3:
            raise ValueError(f"code record id {identifier!r} is not <owner>/<repo>/<path>")
        return "/".join(parts[:2])
    if kind == "books":
        head, sep, chunk = identifier.rpartition(":")
        if not sep or not head.startswith("gutenberg:") or not chunk.isdigit():
            raise ValueError(f"books record id {identifier!r} is not gutenberg:<book>:<chunk>")
        return head
    return identifier


def _close(stream) -> None:
    """Close a record iterator now (port addition; generators propagate the close to datasets/requests)."""
    close = getattr(stream, "close", None)
    if close is not None:
        close()


# --- bucket builder: scripts/modal_prepare_corpus.py:334-406 ---------------------------------
#: Seconds between two progress callbacks.
PROGRESS_SECONDS = 10.0


def prepare_bucket(root: Path, spec: dict, *, progress: Optional[Callable[[dict, str], None]] = None,
                   extra_manifest: Optional[dict] = None) -> dict:
    """Build ``root/<bucket>/{train.jsonl.gz,eval.jsonl.gz,manifest.json}`` exactly as the original.

    Port additions (module docstring): explicit iterator close, ``progress(counts, phase)``
    every ``PROGRESS_SECONDS`` (phase ``'train'`` or ``'eval'``), atomic manifest write,
    ``extra_manifest`` keys merged into the manifest (they may not replace original keys).
    """
    # scripts/modal_prepare_corpus.py:334-406
    from hyak.common.io import atomic_write_text

    extra_manifest = dict(extra_manifest or {})
    reserved = {"spec", "dataset", "config", "revision", "split_policy", "counts", "files"}
    if reserved & set(extra_manifest):
        raise ValueError(f"extra_manifest may not replace original manifest keys: {sorted(reserved & set(extra_manifest))}")
    repo, config, split_policy = source_info(spec)
    destination = root / spec["name"]
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec or manifest["revision"] != SOURCE_REVISIONS[repo]:
            raise RuntimeError(f"Existing bucket has different settings: {destination}")
        print(f"Skipping completed {spec['name']}", flush=True)
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    train_tmp = destination / "train.jsonl.gz.tmp"
    eval_tmp = destination / "eval.jsonl.gz.tmp"
    counts = {"train_bytes": 0, "eval_bytes": 0, "train_docs": 0, "eval_docs": 0,
              "seen_train_rows": 0, "seen_eval_rows": 0, "skipped_docs": 0}
    separate_eval = split_policy in {"upstream-test", "upstream-validation"}
    last_report = time.monotonic()

    def report(phase):  # port addition
        nonlocal last_report
        now = time.monotonic()
        if progress is not None and now - last_report >= PROGRESS_SECONDS:
            last_report = now
            progress(dict(counts), phase)

    try:
        with gzip.open(train_tmp, "wb") as train_file, gzip.open(eval_tmp, "wb") as eval_file:
            stream = records(spec, "train")
            try:
                for record in stream:
                    counts["seen_train_rows"] += 1
                    report("train")
                    text, identifier, split_key = record_fields(spec, record)
                    if not text or not identifier:
                        counts["skipped_docs"] += 1
                        continue
                    byte_count = len(text.encode("utf-8"))
                    if byte_count < 100 or byte_count > MAX_DOCUMENT_BYTES:
                        counts["skipped_docs"] += 1
                        continue
                    use_eval = not separate_eval and held_out(split_key)
                    if use_eval and counts["eval_bytes"] < spec["eval_target"]:
                        _write_record(eval_file, text, identifier)
                        counts["eval_bytes"] += byte_count
                        counts["eval_docs"] += 1
                    elif not use_eval and counts["train_bytes"] < spec["train_target"]:
                        _write_record(train_file, text, identifier)
                        counts["train_bytes"] += byte_count
                        counts["train_docs"] += 1
                    if counts["train_bytes"] >= spec["train_target"] and (separate_eval or counts["eval_bytes"] >= spec["eval_target"]):
                        break
            finally:
                _close(stream)  # port addition
            if separate_eval:
                eval_split = "test" if split_policy == "upstream-test" else "validation"
                stream = records(spec, eval_split)
                try:
                    for record in stream:
                        counts["seen_eval_rows"] += 1
                        report("eval")
                        text, identifier, _ = record_fields(spec, record)
                        if not text or not identifier:
                            counts["skipped_docs"] += 1
                            continue
                        byte_count = len(text.encode("utf-8"))
                        if byte_count < 100 or byte_count > MAX_DOCUMENT_BYTES:
                            counts["skipped_docs"] += 1
                            continue
                        _write_record(eval_file, text, identifier)
                        counts["eval_bytes"] += byte_count
                        counts["eval_docs"] += 1
                        if counts["eval_bytes"] >= spec["eval_target"]:
                            break
                finally:
                    _close(stream)  # port addition
        if counts["train_bytes"] < spec["train_target"] or counts["eval_bytes"] < spec["eval_target"]:
            raise RuntimeError(f"Source exhausted before target for {spec['name']}: {counts}")
        train_path = destination / "train.jsonl.gz"
        eval_path = destination / "eval.jsonl.gz"
        os.replace(train_tmp, train_path)
        os.replace(eval_tmp, eval_path)
        manifest = {
            "spec": spec, "dataset": repo, "config": config,
            "revision": SOURCE_REVISIONS[repo], "split_policy": split_policy,
            "counts": counts,
            "files": {"train.jsonl.gz": _sha256(train_path), "eval.jsonl.gz": _sha256(eval_path)},
        }
        manifest.update(extra_manifest)  # port addition (never replaces an original key, checked above)
        # Original: manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n").
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    finally:
        train_tmp.unlink(missing_ok=True)
        eval_tmp.unlink(missing_ok=True)


# --- pool summary: scripts/modal_prepare_corpus.py:419-422 -----------------------------------
def summary_text(size: str, manifests: list[dict]) -> str:
    """``summary.json`` text of a pool size, exactly as ``prepare`` wrote it (indent=2, no key sort)."""
    # scripts/modal_prepare_corpus.py:419-422
    summary = {"size": size, "total_train_bytes": sum(m["counts"]["train_bytes"] for m in manifests),
               "total_eval_bytes": sum(m["counts"]["eval_bytes"] for m in manifests),
               "buckets": [m["spec"]["name"] for m in manifests]}
    return json.dumps(summary, indent=2) + "\n"


# --- verify_corpus: scripts/modal_prepare_corpus.py:454-485 ----------------------------------
def verify_corpus(corpus_root: Path, size: str) -> dict:
    """Check committed file hashes and every document count and byte total."""
    # scripts/modal_prepare_corpus.py:454-485 (ROOT -> corpus_root argument)
    root = Path(corpus_root) / f"{size}-v1"
    summary = json.loads((root / "summary.json").read_text())
    verified = []
    totals = {"train": 0, "eval": 0}
    for name in summary["buckets"]:
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text())
        counts = manifest["counts"]
        for split in ("train", "eval"):
            filename = f"{split}.jsonl.gz"
            path = directory / filename
            if _sha256(path) != manifest["files"][filename]:
                raise RuntimeError(f"Checksum mismatch: {path}")
            actual_bytes = actual_docs = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    actual_bytes += len(record["text"].encode("utf-8"))
                    actual_docs += 1
            if actual_bytes != counts[f"{split}_bytes"] or actual_docs != counts[f"{split}_docs"]:
                raise RuntimeError(f"Manifest count mismatch: {path}")
            totals[split] += actual_bytes
        verified.append(name)
        print(f"Verified {name}", flush=True)
    for split in ("train", "eval"):
        if totals[split] != summary[f"total_{split}_bytes"]:
            raise RuntimeError(f"Corpus summary {split}-byte mismatch")
    return {"size": size, "verified_buckets": len(verified),
            "train_bytes": summary["total_train_bytes"], "eval_bytes": summary["total_eval_bytes"]}


# --- publish_recipes (balanced-full-v3 part): scripts/modal_prepare_corpus.py:511-530 --------
def publish_recipe_v3(corpus_root: Path, display_root: Path = ROOT) -> tuple[dict, str]:
    """Regenerate ``balanced-full-v3`` from the bucket manifests under ``corpus_root``.

    Paths in the recipe are rendered under ``display_root`` (default ``/corpus``, the original
    mount), so for a faithful rebuild the returned text equals ``data/balanced-full-v3.json``
    byte for byte.  Returns ``(recipe, text)``; writes nothing.  The two ``balanced-*-v2``
    recipes of the original function need the unused 2 GB pilot pool and are not ported.
    """
    # scripts/modal_prepare_corpus.py:511-530 (reads manifests below corpus_root instead of ROOT)
    size, total = "full", FULL_BYTES
    buckets = []
    for name, target in specialty_recipe_targets(total).items():
        pool = ("specialty" if name in {"medical-en", "legal-en"} else
                "extra" if name in {"math-en", "qa-en", "books-en"} else size)
        directory = Path(display_root) / f"{pool}-v1" / name
        manifest = json.loads((Path(corpus_root) / f"{pool}-v1" / name / "manifest.json").read_text())
        available = manifest["counts"]["train_bytes"]
        if available < target:
            raise RuntimeError(f"Insufficient data in {directory}: {available} < {target}")
        buckets.append({"name": name, "source_path": str(directory),
                        "train_target_bytes": target, "train_available_bytes": available,
                        "eval_path": str(directory / "eval.jsonl.gz"),
                        "dataset": manifest["dataset"], "revision": manifest["revision"]})
    recipe = {"name": "balanced-full-v3", "profile": "group-cron",
              "volume": VOLUME_NAME, "unit": "raw UTF-8 text bytes",
              "total_train_target_bytes": total, "buckets": buckets}
    return recipe, json.dumps(recipe, indent=2) + "\n"


# --- probe_sources: scripts/modal_prepare_corpus.py:428-451 ----------------------------------
#: One spec per source kind (scripts/modal_prepare_corpus.py:431-442).
PROBE_EXAMPLES = (
    {"name": "web-en", "kind": "fineweb"},
    {"name": "web-es", "kind": "fineweb2", "config": "spa_Latn"},
    {"name": "wiki-en", "kind": "wiki", "language": "en"},
    {"name": "code", "kind": "code"},
    {"name": "papers-en", "kind": "papers"},
    {"name": "math-en", "kind": "math"},
    {"name": "qa-en", "kind": "qa"},
    {"name": "books-en", "kind": "books"},
    {"name": "medical-en", "kind": "medical"},
    {"name": "legal-en", "kind": "legal"},
)


def probe_selection(size: str = "full") -> list[dict]:
    """Specs probed for a size, as the original selected them (scripts/modal_prepare_corpus.py:444-445)."""
    examples = [dict(spec) for spec in PROBE_EXAMPLES]
    return (examples[-2:] if size == "specialty" else
            examples[-5:-2] if size == "extra" else examples)


def probe_one(spec: dict) -> dict:
    """First record of one source: ``{"id", "bytes"}`` (scripts/modal_prepare_corpus.py:447-449)."""
    stream = records(spec, "train")
    try:
        record = next(stream)
    finally:
        _close(stream)  # port addition
    content, identifier, _ = record_fields(spec, record)
    return {"id": identifier, "bytes": len(content.encode("utf-8"))}


def probe_sources(size: str = "full") -> dict[str, dict]:
    """Verify source adapters (first record of each source kind); raises on the first failure."""
    # scripts/modal_prepare_corpus.py:428-451
    result = {}
    for spec in probe_selection(size):
        result[spec["name"]] = probe_one(spec)
        print(f"Probed {spec['name']}: {result[spec['name']]['bytes']} bytes", flush=True)
    return result
