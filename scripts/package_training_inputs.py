"""Verify the complete saved training input tree and build a portable archive.

Run beside the cloud copy of the artifacts. No model checkpoints from a
partially trained run, credentials or local caller files enter this archive.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import os


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def emit(path, stage, **fields):
    value = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, **fields)
    atomic_json(path, value)
    with Path(path).with_suffix('.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def input_manifest(inputs, status):
    audit_path = inputs / 'rl-input-audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['tokenizers'] != 1000:
        raise ValueError('The complete 1,000-tokenizer pool is required')
    expected = {}
    for item in audit['files']:
        name = Path(item['path'])
        if name.is_absolute() or '..' in name.parts or name.parts[0] != 'input':
            raise ValueError('Invalid original input path')
        name = name.relative_to('input').as_posix()
        if name in expected:
            raise ValueError('Duplicate input audit entry')
        expected[name] = {k: item[k] for k in ['bytes', 'sha256']}
    expected['rl-input-audit.json'] = dict(bytes=audit_path.stat().st_size, sha256=sha256(audit_path))
    actual = {p.relative_to(inputs).as_posix() for p in inputs.rglob('*') if p.is_file()}
    if actual != set(expected) or any(p.is_symlink() for p in inputs.rglob('*')):
        raise ValueError('Input tree differs from its original complete file audit')
    completed_bytes = 0
    for number, (name, metadata) in enumerate(sorted(expected.items()), 1):
        path = inputs / name
        if path.stat().st_size != metadata['bytes'] or sha256(path) != metadata['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
        completed_bytes += metadata['bytes']
        if number % 100 == 0 or number == len(expected):
            emit(status, 'verifying_inputs', completed=number, total=len(expected), completed_bytes=completed_bytes)
    index = json.loads((inputs / 'pool/full-results.json').read_text())
    if index['completed'] != 1000 or len(index['tokenizers']) != 1000:
        raise ValueError('Incomplete original pool index')
    manifest = dict(schema_version=1, source=dict(profile='queues', volume='anytok-pool-rl-20k', prefix='input'),
                    tokenizers=1000, corpus_buckets=23,
                    pool_plan_sha256=audit['pool_plan_sha256'], pool_index_sha256=audit['pool_index_sha256'],
                    files={'input/' + name: metadata for name, metadata in sorted(expected.items())},
                    total_bytes=completed_bytes)
    return manifest


def verify_archive(path, manifest, status):
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for entry in archive:
            if not entry.isfile() or entry.name in seen:
                raise ValueError('Unexpected or duplicate archive member')
            seen.add(entry.name)
            stream = archive.extractfile(entry)
            if entry.name == 'BUNDLE_MANIFEST.json':
                if json.load(stream) != manifest:
                    raise ValueError('Embedded manifest differs')
            else:
                expected = manifest['files'][entry.name]
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
                if entry.size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                    raise ValueError(f'Archive checksum mismatch: {entry.name}')
            if len(seen) % 100 == 0:
                emit(status, 'verifying_archive', completed=len(seen), total=len(manifest['files']) + 1)
    if seen != set(manifest['files']) | {'BUNDLE_MANIFEST.json'}:
        raise ValueError('Archive omitted original input files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new versioned archive path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = input_manifest(args.inputs, args.status)
    sidecar = args.output.with_name('BUNDLE_MANIFEST.json')
    atomic_json(sidecar, manifest)
    temporary = args.output.with_suffix('.partial')
    def stable(entry):
        entry.uid = entry.gid = entry.mtime = 0
        entry.uname = entry.gname = ''
        entry.mode = 0o644
        return entry
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                archive.add(sidecar, arcname='BUNDLE_MANIFEST.json', filter=stable)
                for number, name in enumerate(manifest['files'], 1):
                    archive.add(args.inputs.parent / name, arcname=name, recursive=False, filter=stable)
                    if number % 100 == 0 or number == len(manifest['files']):
                        emit(args.status, 'packing', completed=number, total=len(manifest['files']), archive_bytes=raw.tell())
    verify_archive(temporary, manifest, args.status)
    temporary.replace(args.output)
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(f'{digest}  {args.output.name}\n')
    receipt = dict(archive=args.output.name, bytes=args.output.stat().st_size, sha256=digest,
                   input_files=len(manifest['files']), input_bytes=manifest['total_bytes'], tokenizers=1000,
                   verified_archive_members=len(manifest['files']) + 1, manifest='BUNDLE_MANIFEST.json')
    atomic_json(args.output.with_name('bundle-receipt.json'), receipt)
    emit(args.status, 'complete', completed=1, total=1, **receipt)


if __name__ == '__main__':
    main()
