"""Download, check, extract and verify Qingyun's input bundle (run through hyak/handoff/fetch_inputs.sh).

    python -m hyak.handoff.fetch [--dry-run] [--force] [--remove-archive] [--no-download | --download-only]
                                 [--from-dir DIR] [--reverify] [--reference-dir DIR] [--config FILE]

1. ``$ANYTOK_ROOT/input`` already holds the verified bundle (``handoff/inputs-verified.json`` matches every
   file's size and modification time, and ``BUNDLE_MANIFEST.json`` equals the repository's copy): nothing to
   do (``--reverify`` hashes every file again instead).  A tree without that record is verified in full; one
   that does not verify is refused (exit 2) unless ``--force``, which MOVES it (and BUNDLE_MANIFEST.json) to
   ``input.replaced-<UTC>`` (nothing is deleted).
2. Download the archive, its ``.sha256`` and ``bundle-receipt.json`` at the pinned revision of
   ``docs/raw-bpb-handoff/bundle-reference.json`` with ``huggingface_hub.hf_hub_download(..., token=False)``
   into ``$ANYTOK_ROOT/handoff/`` (an interrupted download resumes; a complete file of the right size is not
   downloaded again).  ``--from-dir DIR`` copies the three files from a directory on the cluster instead (for
   example a copy someone already downloaded there; no laptop in between).
3. Check the archive's size and SHA-256 against the repository pin (authoritative), the downloaded ``.sha256``
   and the downloaded receipt against the same pin (and the receipt against the repository's receipt).
4. Extract into a staging directory next to ``input/`` (only ``input/...`` regular files and directories and
   ``BUNDLE_MANIFEST.json`` are accepted; tarfile's ``data`` filter), then move ``input/`` and
   ``BUNDLE_MANIFEST.json`` into ``$ANYTOK_ROOT`` in one rename each: a reaped extraction never leaves a
   half-extracted ``input/``.  Extraction may be moved into a debug job if the login node reaps it
   (``--no-download`` there; it needs no internet).
5. Run ``python scripts/verify_training_inputs.py --root $ANYTOK_ROOT`` (Qingyun's verifier: every file's size
   and SHA-256) and require 1,062 files, 1,000 tokenizers and 23 corpus buckets; the extracted
   ``BUNDLE_MANIFEST.json`` must equal ``docs/raw-bpb-handoff/BUNDLE_MANIFEST.json``.  Then write
   ``handoff/inputs-verified.json``.  ``--remove-archive`` deletes the 9.3 GB archive after that (default:
   kept).

``--reference-dir`` replaces ``docs/raw-bpb-handoff`` (tests only).  Exit status: 0 done (or nothing to do),
2 usage, configuration, space, download or existing-tree problem (with a remedy), 4 a size, checksum or
verification failure, 1 a crash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from hyak.handoff import common as C

EXIT_OK, EXIT_USAGE, EXIT_CHECK = 0, 2, 4
SHA_SUFFIX = '.sha256'
#: Slack on top of the byte counts when free space is checked (GB = 1e9).
SPACE_SLACK = 2_000_000_000


class Stop(Exception):
    """A refusal with an exit code and a remedy."""

    def __init__(self, code, message, remedy=None):
        super().__init__(message)
        self.code, self.remedy = code, remedy


def say(message: str) -> None:
    print(f'[fetch_inputs] {message}', flush=True)


def _stamp() -> str:
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def sha256_with_progress(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    done, next_report = 0, 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
            done += len(block)
            if size and done >= next_report:
                say(f'hashing {label}: {done / 1e9:.1f} of {size / 1e9:.1f} GB')
                next_report += max(size // 10, 1)
    return digest.hexdigest()


# --- the extracted tree ----------------------------------------------------------------------------------
def tree_listing(root: Path) -> dict:
    """``{relative path: [bytes, mtime_ns]}`` of every file under ``root/input``."""
    listing = {}
    base = root / 'input'
    for path in sorted(base.rglob('*')):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            listing[path.relative_to(root).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return listing


def manifest_matches(root: Path, reference_manifest: Path) -> bool:
    path = root / C.MANIFEST_NAME
    try:
        return json.loads(path.read_text()) == json.loads(Path(reference_manifest).read_text())
    except (OSError, ValueError):
        return False


def quick_verified(root: Path, receipt_path: Path, ref: dict) -> bool:
    """True when the receipt of an earlier full verification still describes the tree exactly."""
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError):
        return False
    return (receipt.get('archive_sha256') == ref['reference']['sha256']
            and manifest_matches(root, ref['manifest_path'])
            and receipt.get('files') == tree_listing(root))


def run_verifier(repo: Path, root: Path, python: str) -> dict:
    """Qingyun's scripts/verify_training_inputs.py, as it is; its JSON line, or Stop(4)."""
    script = repo / C.VERIFY_INPUTS_REL
    say(f'verifying every file: {python} {script} --root {root}')
    proc = subprocess.run([python, str(script), '--root', str(root)], capture_output=True, text=True,
                          errors='replace', cwd=str(repo))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise Stop(EXIT_CHECK, f'{C.VERIFY_INPUTS_REL} failed (exit {proc.returncode}): {" | ".join(tail)}',
                   f'the extracted tree does not match the bundle manifest; move {root}/input away '
                   '(or rerun with --force, which moves it) and extract again')
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise Stop(EXIT_CHECK, f'{C.VERIFY_INPUTS_REL} printed no result: {proc.stdout[-300:]}') from exc


def full_verification(repo: Path, root: Path, ref: dict, python: str) -> dict:
    result = run_verifier(repo, root, python)
    expected = {'verified_files': ref['reference']['input_files'], 'tokenizers': C.TOKENIZERS,
                'corpus_buckets': C.CORPUS_BUCKETS, 'input_bytes': ref['receipt']['input_bytes']}
    wrong = {k: (result.get(k), v) for k, v in expected.items() if result.get(k) != v}
    if wrong:
        raise Stop(EXIT_CHECK, f'verification result differs from the pin: {wrong} (found, expected)',
                   'the bundle is not the pinned one; do not train on it')
    if not manifest_matches(root, ref['manifest_path']):
        raise Stop(EXIT_CHECK, f'{root / C.MANIFEST_NAME} differs from the repository copy {ref["manifest_path"]}',
                   'the archive is not the pinned bundle; do not train on it')
    return result


def write_receipt(paths, ref: dict, result: dict) -> Path:
    from hyak.common.io import atomic_write_json, sha256_file, utc_now
    record = {'kind': 'hyak_rawbpb_inputs_verified', 'at_utc': utc_now(), 'root': str(paths.root),
              'repo_id': ref['reference']['repo_id'], 'revision': ref['reference']['revision'],
              'archive': ref['archive'], 'archive_sha256': ref['reference']['sha256'],
              'archive_bytes': ref['reference']['bytes'],
              'bundle_manifest_sha256': sha256_file(paths.bundle_manifest), 'verification': result,
              'files': tree_listing(paths.root)}
    paths.handoff_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.handoff_receipt, record)
    return paths.handoff_receipt


# --- download --------------------------------------------------------------------------------------------
def _free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def need_space(path: Path, wanted: int, what: str) -> None:
    free = _free(path)
    if free < wanted + SPACE_SLACK:
        raise Stop(EXIT_USAGE, f'{what} needs {wanted / 1e9:.1f} GB (+{SPACE_SLACK / 1e9:.0f} GB slack) on the file '
                   f'system of {path}; {free / 1e9:.1f} GB are free (a file-system view, not your quota)',
                   'free space or choose another ANYTOK_ROOT (check your quota with hyakstorage)')


def download(ref: dict, dest: Path, from_dir=None) -> None:
    names = [ref['archive'], ref['archive'] + SHA_SUFFIX, C.RECEIPT_NAME]
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ref['archive']
    if archive.is_file() and archive.stat().st_size == ref['reference']['bytes']:
        say(f'{archive} is complete ({archive.stat().st_size:,} bytes); not downloaded again')
        names.remove(ref['archive'])
    else:
        have = archive.stat().st_size if archive.is_file() else 0
        need_space(dest, ref['reference']['bytes'] - have, 'the archive download')
    if from_dir is not None:
        source = Path(from_dir)
        for name in names:
            if not (source / name).is_file():
                raise Stop(EXIT_USAGE, f'--from-dir {source}: {name} is missing', 'name the directory holding all '
                           f'three files ({", ".join(names)})')
            partial = dest / (name + '.partial')
            say(f'copying {source / name} -> {dest / name}')
            shutil.copyfile(source / name, partial)
            os.replace(partial, dest / name)
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise Stop(EXIT_USAGE, f'huggingface_hub is not importable ({exc})',
                   'run this with the train environment (bash hyak/env/setup_train_env.sh installs huggingface_hub==1.33.0)') from exc
    reference = ref['reference']
    for name in names:
        say(f'downloading {name} from {reference["repo_id"]} (dataset) at revision {reference["revision"]} '
            f'into {dest} (resumable; rerun after an interruption)')
        try:
            hf_hub_download(repo_id=reference['repo_id'], repo_type='dataset', revision=reference['revision'],
                            filename=name, local_dir=str(dest), token=False)
        except Exception as exc:  # noqa: BLE001 - network, HTTP and file errors all mean "try again"
            raise Stop(EXIT_USAGE, f'download of {name} failed: {type(exc).__name__}: {exc}',
                       'check the login node\'s internet access and rerun bash hyak/handoff/fetch_inputs.sh '
                       '(the download resumes)') from exc


def check_downloads(ref: dict, dest: Path, repo_receipt: dict) -> None:
    reference = ref['reference']
    archive = dest / ref['archive']
    if not archive.is_file():
        raise Stop(EXIT_USAGE, f'{archive} does not exist', 'run without --no-download first')
    size = archive.stat().st_size
    if size != reference['bytes']:
        raise Stop(EXIT_CHECK, f'{archive} has {size:,} bytes; the pin (docs/raw-bpb-handoff/bundle-reference.json) '
                   f'says {reference["bytes"]:,}', f'remove it (rm {archive}) and run again')
    digest = sha256_with_progress(archive, archive.name)
    if digest != reference['sha256']:
        raise Stop(EXIT_CHECK, f'{archive}: SHA-256 {digest} differs from the pin {reference["sha256"]}',
                   f'remove it (rm {archive}) and run again; do not extract it')
    say(f'archive size and SHA-256 equal the repository pin ({reference["sha256"]})')
    sha_file = dest / (ref['archive'] + SHA_SUFFIX)
    words = sha_file.read_text().split() if sha_file.is_file() else []
    if len(words) < 2 or words[0] != reference['sha256'] or words[1].lstrip('*') != ref['archive']:
        raise Stop(EXIT_CHECK, f'{sha_file} does not name {reference["sha256"]} for {ref["archive"]} ({words[:2]})',
                   f'remove it (rm {sha_file}) and run again')
    receipt_path = dest / C.RECEIPT_NAME
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise Stop(EXIT_CHECK, f'{receipt_path}: {exc}', f'remove it (rm {receipt_path}) and run again') from exc
    if receipt != repo_receipt:
        raise Stop(EXIT_CHECK, f'{receipt_path} differs from the repository copy of {C.RECEIPT_NAME}',
                   'the dataset revision does not match the pin; do not use it')
    say(f'{sha_file.name} and {receipt_path.name} agree with the pin')


# --- extraction ------------------------------------------------------------------------------------------
def _member_ok(member: tarfile.TarInfo) -> bool:
    name = member.name
    while name.startswith('./'):
        name = name[2:]
    parts = Path(name).parts
    if not parts or name.startswith('/') or '..' in parts:
        return False
    if not (member.isfile() or member.isdir()):
        return False
    if name == C.MANIFEST_NAME:
        return member.isfile()
    return parts[0] == 'input'


def extract(archive: Path, root: Path, input_bytes: int) -> None:
    need_space(root, input_bytes, 'the extraction')
    staging = root / '.handoff-extract'
    if staging.exists():
        say(f'removing the staging directory of an interrupted extraction: {staging}')
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    say(f'extracting {archive} into {staging} (13.4 GB; several minutes)')
    count = 0
    with tarfile.open(archive, mode='r|gz') as stream:
        for member in stream:
            if member.name in ('.', './'):
                continue
            if not _member_ok(member):
                raise Stop(EXIT_CHECK, f'{archive}: unexpected member {member.name!r} (type {member.type!r})',
                           f'the archive is not the pinned bundle; remove it (rm {archive})')
            stream.extract(member, staging, filter='data')
            count += 1
            if count % 200 == 0:
                say(f'extracted {count} members')
    for name in ('input', C.MANIFEST_NAME):
        if not (staging / name).exists():
            raise Stop(EXIT_CHECK, f'{archive} holds no {name}', f'the archive is not the bundle; remove it (rm {archive})')
    os.rename(staging / 'input', root / 'input')
    os.replace(staging / C.MANIFEST_NAME, root / C.MANIFEST_NAME)
    staging.rmdir()
    say(f'extracted {count} members into {root}/input and {root}/{C.MANIFEST_NAME}')


# --- main ------------------------------------------------------------------------------------------------
def plan_text(paths, ref: dict, args) -> list:
    reference = ref['reference']
    archive = paths.handoff_dir / ref['archive']
    lines = [f'bundle: {reference["repo_id"]} (dataset) at revision {reference["revision"]}',
             f'archive: {reference["archive_url"]} ({reference["bytes"]:,} bytes, SHA-256 {reference["sha256"]})',
             f'download into: {paths.handoff_dir} ({"copy from " + str(args.from_dir) if args.from_dir else "huggingface_hub, token=False"})',
             f'archive present: {archive.is_file()} ({archive.stat().st_size if archive.is_file() else 0:,} bytes)',
             f'extract into: {paths.root} -> {paths.input} and {paths.bundle_manifest} '
             f'({reference["input_files"]} files, {ref["receipt"]["input_bytes"]:,} bytes)',
             f'input/ exists: {paths.input.exists()}; verified record: {paths.handoff_receipt.is_file()}',
             f'verify: scripts/verify_training_inputs.py --root {paths.root}; receipt {paths.handoff_receipt}',
             f'archive afterwards: {"removed (--remove-archive)" if args.remove_archive else "kept"}']
    return lines


def run(args) -> int:
    from hyak.common.config import load_config, repo_root
    from hyak.common.paths import Paths
    repo = repo_root()
    paths = Paths.from_config(load_config(args.config))
    ref = C.load_reference(args.reference_dir or C.docs_dir(repo))
    repo_receipt = ref['receipt']
    python = sys.executable
    if args.dry_run:
        for line in plan_text(paths, ref, args):
            say(f'dry run: {line}')
        say('dry run: nothing downloaded, extracted or written')
        return EXIT_OK
    root = paths.root
    archive = paths.handoff_dir / ref['archive']

    def finish(result=None) -> int:
        if result is not None:
            say(f'wrote {write_receipt(paths, ref, result)}')
        if args.remove_archive and archive.is_file():
            archive.unlink()
            say(f'removed {archive} (--remove-archive)')
        say(f'OK: {root}/input is the verified bundle (1,062 files, 1,000 tokenizers, 23 buckets). '
            'Next: hyak/slurm/submit.sh smoke-rawbpb')
        return EXIT_OK

    if paths.input.exists():
        if not args.reverify and quick_verified(root, paths.handoff_receipt, ref):
            say(f'{paths.input} was verified in full at '
                f'{json.loads(paths.handoff_receipt.read_text()).get("at_utc")} and is unchanged (sizes and times of '
                'all files); nothing to do (--reverify hashes every file again)')
            return finish()
        say(f'{paths.input} exists; verifying it before anything is downloaded')
        try:
            return finish(full_verification(repo, root, ref, python))
        except Stop as exc:
            if not args.force:
                raise Stop(EXIT_USAGE, f'{paths.input} exists but is not the verified bundle ({exc})',
                           f'move it away yourself, or rerun with --force (moves {paths.input} to '
                           f'{paths.input}.replaced-<UTC>; nothing is deleted)') from exc
            aside = root / f'input.replaced-{_stamp()}'
            say(f'--force: moving {paths.input} -> {aside}')
            os.rename(paths.input, aside)
            if paths.bundle_manifest.exists():
                os.rename(paths.bundle_manifest, root / f'{aside.name}.{C.MANIFEST_NAME}')
            if paths.handoff_receipt.exists():
                os.rename(paths.handoff_receipt, paths.handoff_receipt.with_name(f'inputs-verified.replaced-{_stamp()}.json'))
    if not args.no_download:
        download(ref, paths.handoff_dir, args.from_dir)
    check_downloads(ref, paths.handoff_dir, repo_receipt)
    if args.download_only:
        say(f'--download-only: {archive} checked; extract with bash hyak/handoff/fetch_inputs.sh --no-download')
        return EXIT_OK
    if paths.bundle_manifest.exists():
        aside = root / f'{C.MANIFEST_NAME}.replaced-{_stamp()}'
        say(f'moving a stale {paths.bundle_manifest} -> {aside}')
        os.rename(paths.bundle_manifest, aside)
    extract(archive, root, ref['receipt']['input_bytes'])
    return finish(full_verification(repo, root, ref, python))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m hyak.handoff.fetch', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='print the plan; no network, no writes')
    parser.add_argument('--force', action='store_true', help='move an existing input/ that does not verify aside')
    parser.add_argument('--remove-archive', action='store_true', help='delete the archive after the verification')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--no-download', action='store_true', help='use the archive already in $ANYTOK_ROOT/handoff')
    mode.add_argument('--download-only', action='store_true', help='download and check; do not extract')
    parser.add_argument('--from-dir', type=Path, help='copy the three files from this directory instead of the Hub')
    parser.add_argument('--reverify', action='store_true', help='hash every file of a verified tree again')
    parser.add_argument('--reference-dir', type=Path, help='replaces docs/raw-bpb-handoff (tests only)')
    parser.add_argument('--config', type=Path, help='config file (default: $HYAK_CONFIG or hyak/config.env)')
    args = parser.parse_args(argv)
    from hyak.common.config import ConfigError
    try:
        return run(args)
    except Stop as exc:
        say(f'ERROR: {exc}')
        if exc.remedy:
            say(f'remedy: {exc.remedy}')
        return exc.code
    except (ConfigError, ValueError, OSError) as exc:
        say(f'ERROR: {type(exc).__name__}: {exc}')
        say('remedy: check hyak/config.env (ANYTOK_ROOT) and the paths named above')
        return EXIT_USAGE


if __name__ == '__main__':
    sys.exit(main())
