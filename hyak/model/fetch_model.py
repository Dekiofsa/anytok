"""Fetch the pinned SmolLM2-1.7B files into ``$ANYTOK_ROOT/input/model`` and verify them (DESIGN.md 5.A).

    python -m hyak.model.fetch_model [--dest DIR] [--method auto|hub|https] [--config FILE]
    python -m hyak.model.fetch_model --from-dir DIR      # bring your own copy (e.g. an HF snapshot)
    python -m hyak.model.fetch_model --verify-only

Exactly ``config.json``, ``model.safetensors`` and ``tokenizer.json`` of
``HuggingFaceTB/SmolLM2-1.7B`` at revision ``effd688a12921b4cc83e3312b6feb579f70f9c71`` (the
files and revision of ``scripts/modal_anytok_pool_20k.py:26-27,71-72``).  Bytes and SHA-256 are
checked against ``plans/zett-modal-20k-inputs.json`` (``files["model/<name>"]``), whose own
SHA-256 is pinned in ``hyak.code.pinned``.  The destination ends up holding exactly those
three files, like the original ``input/model`` (NOTES/modal-launchers.md risk 19: extra Hub
files such as ``tokenizer_config.json`` could change tokenizer defaults).

Downloads go to a staging directory next to the destination (``<dest>.download``, same
filesystem) and are resumable: ``huggingface_hub.hf_hub_download(local_dir=...)`` keeps its
partial files there, the ``https`` method resumes with HTTP Range requests.  A file is moved
into the destination (atomic rename) only after its size and SHA-256 match.  The tool never
deletes or overwrites anything in the destination: wrong or extra files there are reported
with the command that removes them.  ``HF_TOKEN`` (optional, config or environment) is
passed to the Hub and never printed.  ``HF_HOME`` defaults to ``$ANYTOK_ROOT/hf-cache``.

Run it on the login node (compute nodes may have no internet), in the train environment
(``huggingface_hub`` comes with transformers); ``--method https`` needs only the standard library.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from hyak.code import pinned
from hyak.common.io import fsync_dir, read_json, sha256_file

MODEL_REPO = 'HuggingFaceTB/SmolLM2-1.7B'
MODEL_REVISION = 'effd688a12921b4cc83e3312b6feb579f70f9c71'
MODEL_FILES = ('config.json', 'model.safetensors', 'tokenizer.json')
HF_ENDPOINT = 'https://huggingface.co'
CHUNK = 8 * 1024 * 1024
USER_AGENT = 'hyak-anytok-fetch/1 (python-urllib)'


class FetchError(RuntimeError):
    """Download or verification failed; the message says what to do."""


def expected_files(repo=None) -> dict:
    """``{name: {"bytes", "sha256"}}`` of the three model files from ``plans/zett-modal-20k-inputs.json``.

    The manifest itself must match its pinned SHA-256 (a changed reference would make every
    check meaningless).
    """
    if repo is None:
        from hyak.common.config import repo_root
        repo = repo_root()
    path = Path(repo) / pinned.INPUTS_MANIFEST
    problems = pinned.check_files(repo, {pinned.INPUTS_MANIFEST: pinned.REFERENCE_FILES[pinned.INPUTS_MANIFEST]})
    if problems:
        raise FetchError(f'{path} does not match its pin ({"; ".join(problems)}). Remedy: '
                         f'{pinned.remedy(problems)}')
    files = read_json(path)['files']
    expected = {}
    for name in MODEL_FILES:
        entry = files[f'model/{name}']
        expected[name] = {'bytes': int(entry['bytes']), 'sha256': str(entry['sha256'])}
    return expected


def verify_file(path, expected: dict, *, check_hash: bool = True) -> list:
    """Problems of one file against ``{"bytes", "sha256"}`` (empty list = verified)."""
    path = Path(path)
    if path.is_symlink():
        return [f'{path}: is a symlink; a real copy is required']
    if not path.is_file():
        return [f'{path}: missing']
    size = path.stat().st_size
    if size != expected['bytes']:
        return [f'{path}: {size} bytes, expected {expected["bytes"]}']
    if check_hash:
        digest = sha256_file(path)
        if digest != expected['sha256']:
            return [f'{path}: sha256 {digest}, expected {expected["sha256"]}']
    return []


def verify_model_dir(dest, expected=None, *, check_hash: bool = True) -> list:
    """Problems of a model directory: exactly the three files, each with the pinned bytes and SHA-256."""
    dest = Path(dest)
    expected = expected_files() if expected is None else expected
    if not dest.is_dir():
        return [f'{dest}: directory missing']
    problems = []
    extra = sorted(p.name for p in dest.iterdir() if p.name not in expected)
    if extra:
        problems.append(f'{dest}: unexpected entries {extra} (the model directory must hold exactly '
                        f'{", ".join(sorted(expected))})')
    for name in sorted(expected):
        problems += verify_file(dest / name, expected[name], check_hash=check_hash)
    return problems


# --- downloads -------------------------------------------------------------------------------------
def _token(cfg):
    value = (cfg or {}).get('HF_TOKEN') or os.environ.get('HF_TOKEN') or ''
    return value or None


def _hub_download(name, staging: Path, token, log) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FetchError('huggingface_hub is not importable; run this in the train environment '
                         '($ANYTOK_ROOT/env/train/bin/python) or use --method https') from exc
    log(f'downloading {name} with huggingface_hub (resumable) into {staging}')
    path = hf_hub_download(repo_id=MODEL_REPO, filename=name, revision=MODEL_REVISION,
                           local_dir=str(staging), token=token)
    return Path(path)


class _NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects but never forward the Hub token to another host (the file CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            for key in [k for k in new.headers if k.lower() == 'authorization']:
                del new.headers[key]
            new.unredirected_hdrs.pop('Authorization', None)
        return new


_OPENER = urllib.request.build_opener(_NoAuthRedirect)


def _header(response, name):
    """A response header value (``None`` when the response has no headers, like the test doubles)."""
    headers = getattr(response, 'headers', None)
    if headers is None:
        return None
    value = headers.get(name)
    return None if value is None else str(value)


def _declared_size_problem(response, status, have, expected_bytes):
    """Why the response cannot be the pinned file (``None`` when its declared size is fine or not declared).

    A 206 carries ``Content-Range: bytes a-b/total`` (total must be the pinned size); a 200 carries
    ``Content-Length`` (the whole file: must be the pinned size).  Checked before a byte is written, so a block
    page or another file served with a 200 is refused at once instead of being downloaded.
    """
    if status == 206:
        content_range = _header(response, 'Content-Range')
        if content_range and '/' in content_range:
            total = content_range.rsplit('/', 1)[1].strip()
            if total.isdigit() and int(total) != expected_bytes:
                return f'Content-Range says the file has {total} bytes, expected {expected_bytes}'
        return None
    length = _header(response, 'Content-Length')
    if length and length.strip().isdigit() and int(length) != expected_bytes:
        return f'Content-Length is {length} bytes for the whole file, expected {expected_bytes}'
    return None


def _https_download(name, staging: Path, expected: dict, token, log, *, endpoint=HF_ENDPOINT,
                    opener=_OPENER.open, retries=5) -> Path:
    """Resumable plain-HTTPS download of one file to ``staging/<name>`` (Range requests).

    Every kind of failed round counts as an attempt (an exception, a short answer, an answer of the wrong or
    larger size) and ends in :class:`FetchError` after ``retries`` of them, so a proxy that answers every
    request with a block page cannot make the tool download for ever.
    """
    final = staging / name
    partial = staging / f'{name}.partial'
    url = f'{endpoint}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
    attempt = 0

    def failed_round(reason):
        nonlocal attempt
        attempt += 1
        if attempt > retries:
            raise FetchError(f'{url}: {reason}; gave up after {attempt} attempts (the source does not serve the '
                             f'pinned file of revision {MODEL_REVISION}); rerun to resume')
        wait = min(60, 2 ** attempt)
        log(f'{reason}; retrying in {wait} s (attempt {attempt}/{retries})')
        time.sleep(wait)

    while True:
        have = partial.stat().st_size if partial.exists() else 0
        if have > expected['bytes']:
            partial.unlink()
            failed_round(f'{partial} was larger than the expected {expected["bytes"]:,} bytes; restarted it')
            have = 0
        if have == expected['bytes']:
            break
        headers = {'User-Agent': USER_AGENT}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        if have:
            headers['Range'] = f'bytes={have}-'
        log(f'downloading {name} over HTTPS from byte {have:,} of {expected["bytes"]:,}')
        oversize = None
        try:
            with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
                status = getattr(response, 'status', None) or response.getcode()
                mode = 'ab'
                if have and status != 206:
                    log('server ignored the Range request; restarting the file from byte 0')
                    mode = 'wb'
                    have = 0
                oversize = _declared_size_problem(response, status, have, expected['bytes'])
                if oversize is None:
                    with open(partial, mode) as stream:
                        written = have
                        while True:
                            block = response.read(CHUNK)
                            if not block:
                                break
                            written += len(block)
                            if written > expected['bytes']:
                                oversize = f'the response is larger than the expected {expected["bytes"]:,} bytes'
                                break
                            stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            attempt += 1
            if attempt > retries:
                raise FetchError(f'HTTPS download of {url} failed {attempt} times ({exc}); rerun to resume') from exc
            wait = min(60, 2 ** attempt)
            log(f'download interrupted ({exc}); retrying in {wait} s (attempt {attempt}/{retries})')
            time.sleep(wait)
            continue
        if oversize is not None:
            if partial.exists():
                partial.unlink()                 # whatever was kept is not (all of) the pinned file
            failed_round(oversize)
            continue
        if partial.stat().st_size < expected['bytes']:
            attempt += 1
            if attempt > retries:
                raise FetchError(f'{url}: connection kept ending early; rerun to resume')
    os.replace(partial, final)
    return final


def _copy_from(source_dir: Path, name, staging: Path, log) -> Path:
    source = source_dir / name
    if not source.exists():
        raise FetchError(f'--from-dir {source_dir} has no {name}')
    log(f'copying {source} (links resolved) into {staging}')
    target = staging / name
    shutil.copyfile(source, target)   # follows symlinks: an HF snapshot links into blobs/
    return target


def fetch(dest, *, method='auto', from_dir=None, cfg=None, repo=None, log=print) -> dict:
    """Make ``dest`` hold exactly the three verified model files.  Returns ``{name: "present"|"fetched"}``.

    Raises FetchError for wrong/extra files already in ``dest`` (never removes them), failed
    downloads and hash mismatches.  Partial downloads stay in ``<dest>.download`` for a resume.
    """
    dest = Path(dest)
    expected = expected_files(repo)
    dest.mkdir(parents=True, exist_ok=True)
    extra = sorted(p.name for p in dest.iterdir() if p.name not in expected)
    if extra:
        raise FetchError(f'{dest} contains unexpected entries {extra}; it must hold only {", ".join(MODEL_FILES)}. '
                         f'Remedy: move them away, e.g. mkdir -p {dest}.extra && mv '
                         + ' '.join(str(dest / e) for e in extra) + f' {dest}.extra/')
    status, todo = {}, []
    for name in MODEL_FILES:
        path = dest / name
        if path.exists() or path.is_symlink():
            problems = verify_file(path, expected[name])
            if problems:
                raise FetchError(f'{problems[0]}. The file in the destination is not the pinned one. '
                                 f'Remedy: rm {path} and rerun')
            status[name] = 'present'
            log(f'{name}: present and verified')
        else:
            todo.append(name)
    if not todo:
        return status
    if method == 'auto':
        try:
            import huggingface_hub  # noqa: F401
            method = 'hub'
        except ImportError:
            method = 'https'
    if from_dir is None and method not in ('hub', 'https'):
        raise FetchError(f'unknown method {method!r}')
    staging = dest.parent / f'{dest.name}.download'
    staging.mkdir(parents=True, exist_ok=True)
    token = _token(cfg)
    for name in todo:
        if from_dir is not None:
            downloaded = _copy_from(Path(from_dir), name, staging, log)
        elif method == 'hub':
            downloaded = _hub_download(name, staging, token, log)
        else:
            downloaded = _https_download(name, staging, expected[name], token, log)
        if downloaded.is_symlink():
            real = staging / f'{name}.real'
            shutil.copyfile(downloaded, real)
            downloaded.unlink()
            os.replace(real, downloaded)
        problems = verify_file(downloaded, expected[name])
        if problems:
            bad = staging / f'{name}.bad'
            os.replace(downloaded, bad)
            raise FetchError(f'{problems[0]} (kept as {bad} for inspection). The source does not serve the '
                             f'pinned file of revision {MODEL_REVISION}; do not continue with other files')
        os.replace(downloaded, dest / name)
        fsync_dir(dest)
        status[name] = 'fetched'
        log(f'{name}: fetched and verified ({expected[name]["bytes"]:,} bytes)')
    problems = verify_model_dir(dest, expected, check_hash=False)
    if problems:
        raise FetchError('; '.join(problems))
    shutil.rmtree(staging, ignore_errors=True)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.model.fetch_model', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    parser.add_argument('--dest', type=Path, help='target directory (default: $ANYTOK_ROOT/input/model)')
    parser.add_argument('--method', choices=('auto', 'hub', 'https'), default='auto',
                        help='hub = huggingface_hub.hf_hub_download, https = standard library (default auto)')
    parser.add_argument('--from-dir', type=Path, help='copy the three files from this directory instead')
    parser.add_argument('--verify-only', action='store_true', help='verify the destination; download nothing')
    args = parser.parse_args(argv)
    from hyak.common.config import ConfigError, load_config
    from hyak.common.paths import Paths
    cfg = None
    try:
        if args.dest is None or not args.verify_only:
            cfg = load_config(args.config)
        dest = args.dest or Paths.from_config(cfg).model_dir
        if cfg is not None:
            os.environ.setdefault('HF_HOME', str(Paths.from_config(cfg).hf_cache))
    except (ConfigError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    log = lambda message: print(f'[fetch_model] {message}', flush=True)  # noqa: E731
    try:
        if args.verify_only:
            problems = verify_model_dir(dest)
            if problems:
                for line in problems:
                    print(f'PROBLEM {line}', file=sys.stderr)
                print(f'ERROR: {dest} is not the pinned SmolLM2-1.7B model. Remedy: python -m hyak.model.fetch_model',
                      file=sys.stderr)
                return 1
            print(f'OK {dest}: {", ".join(MODEL_FILES)} match plans/zett-modal-20k-inputs.json')
            return 0
        status = fetch(dest, method=args.method, from_dir=args.from_dir, cfg=cfg, log=log)
    except FetchError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({'stage': 'model_fetched', 'dest': str(dest), 'repo': MODEL_REPO,
                      'revision': MODEL_REVISION, 'files': status}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
