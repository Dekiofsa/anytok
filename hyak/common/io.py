"""Hashing, atomic file writes and JSON-lines appends (standard library only).

Atomic writes go to a temporary file in the destination directory, are flushed
and fsynced, get the destination's existing mode (or 0666 & ~umask), and are
moved into place with ``os.replace``; the directory is fsynced afterwards
(best effort).  A reader therefore sees either the old or the new content.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

CHUNK = 8 * 1024 * 1024

_UMASK = os.umask(0o022)
os.umask(_UMASK)


def utc_now() -> str:
    """Current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix, e.g. ``2026-11-13T20:00:00.123Z``.

    Parsed by the dashboard (``monitoring.state.stamp``) on Python >= 3.7.
    """
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def sha256_file(path, chunk: int = CHUNK) -> str:
    """Hex SHA-256 of a file, read in ``chunk``-byte blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def file_record(path, chunk: int = CHUNK) -> dict:
    """``{"bytes": <size>, "sha256": <hex>}`` of a file (the manifest entry format used across the port)."""
    path = Path(path)
    size = path.stat().st_size
    digest = sha256_file(path, chunk)
    if path.stat().st_size != size:
        raise RuntimeError(f'{path} changed size while it was being hashed; retry when no job writes it')
    return {'bytes': size, 'sha256': digest}


def _target_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o7777
    except FileNotFoundError:
        return 0o666 & ~_UMASK


def fsync_dir(path) -> None:
    """fsync a directory so a rename inside it is durable (best effort; ignored where unsupported)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path, data: bytes, *, mode=None) -> Path:
    """Atomically replace ``path`` with ``data`` (parent directories are created). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_mode = _target_mode(path) if mode is None else mode
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, final_mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    fsync_dir(path.parent)
    return path


def atomic_write_text(path, text: str, *, encoding: str = 'utf-8', mode=None) -> Path:
    """Atomically replace ``path`` with ``text`` (no newline translation). Returns the path."""
    return atomic_write_bytes(path, text.encode(encoding), mode=mode)


def json_text(obj, indent=2, sort_keys=True, allow_nan=False) -> str:
    """JSON serialisation used by every writer here: UTF-8 (``ensure_ascii=False``), trailing newline.

    With ``allow_nan=False`` (default) NaN/Infinity raise ValueError instead of
    producing invalid JSON.
    """
    return json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False,
                      allow_nan=allow_nan) + '\n'


def finite_json(value):
    """Deep copy of a JSON-like value in which every non-finite float is ``None``.

    The raw value is kept as a string next to it: in a dict ``{"loss": nan}`` becomes
    ``{"loss": None, "loss_raw": "nan"}`` (an existing ``loss_raw`` key is left alone); in a list the
    element is simply ``None``.  Tuples become lists; everything else is returned unchanged, so values
    json.dumps cannot serialise still fail later.  Status and event writers use it so that one NaN from a
    trainer or a scorer never stops a status update (``allow_nan=False`` stays on everywhere else).
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out[key] = finite_json(item)
            if isinstance(item, float) and not math.isfinite(item) and isinstance(key, str):
                out.setdefault(f'{key}_raw', repr(item))
        return out
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def atomic_write_json(path, obj, indent=2, sort_keys=True, *, allow_nan=False) -> Path:
    """Atomically write ``obj`` as JSON (see :func:`json_text`). Returns the path."""
    return atomic_write_text(path, json_text(obj, indent=indent, sort_keys=sort_keys, allow_nan=allow_nan))


def read_json(path, default=...):
    """Parse a JSON file.  With ``default`` given, a missing file returns it.

    Raises ValueError naming the file for invalid JSON (never returns partial data).
    """
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        if default is not ...:
            return default
        raise
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f'{path}: invalid JSON ({exc})') from exc


def append_jsonl(path, obj: dict, *, allow_nan=False) -> dict:
    """Append ``obj`` as one JSON line with a single ``write`` (O_APPEND); adds ``at_utc`` if absent.

    The input dict is not modified; the written dict is returned.  Keys keep
    their insertion order with ``at_utc`` first when it was added.
    """
    if not isinstance(obj, dict):
        raise TypeError(f'append_jsonl needs a dict, got {type(obj).__name__}')
    row = obj if 'at_utc' in obj else {'at_utc': utc_now(), **obj}
    line = (json.dumps(row, ensure_ascii=False, allow_nan=allow_nan) + '\n').encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        written = os.write(fd, line)
        if written != len(line):
            raise OSError(f'Short write to {path}: {written} of {len(line)} bytes')
    finally:
        os.close(fd)
    return row


def atomic_copy(source, destination, *, chunk: int = CHUNK) -> dict:
    """Copy ``source`` to ``destination`` atomically and verify the copy by re-reading it.

    Returns ``{"bytes", "sha256"}`` of the copy; raises RuntimeError if the copy's
    hash differs from the source bytes read during the copy.  Meant for small and
    medium files (it reads the data twice).
    """
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(destination.parent), prefix=f'.{destination.name}.', suffix='.tmp')
    digest = hashlib.sha256()
    size = 0
    try:
        with open(source, 'rb') as reader, os.fdopen(fd, 'wb') as writer:
            while True:
                block = reader.read(chunk)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.chmod(temporary, _target_mode(destination))
        copied = sha256_file(temporary, chunk)
        if copied != digest.hexdigest():
            raise RuntimeError(f'Copy of {source} to {destination} is corrupt (hash mismatch); retry')
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    fsync_dir(destination.parent)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def tail_lines(path, lines: int = 20, max_bytes: int = 65536) -> list:
    """Last ``lines`` lines of a text file (reads at most ``max_bytes``); [] if unreadable."""
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            data = stream.read().decode('utf-8', errors='replace')
    except OSError:
        return []
    rows = data.splitlines()
    if size > max_bytes and rows:
        rows = rows[1:]
    return rows[-lines:]
