"""Verify every extracted bundle file before launching training (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'BUNDLE_MANIFEST.json').read_text())
    if manifest['tokenizers'] != 1000 or manifest['corpus_buckets'] != 23:
        raise ValueError('Wrong complete training bundle')
    expected = manifest['files']
    actual = set()
    for path in (root / 'input').rglob('*'):
        if path.is_symlink():
            raise ValueError('Unexpected symlink in extracted input tree')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError('Missing or additional input files')
    for name, item in expected.items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('input/'):
            raise ValueError('Invalid manifest path')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError('Checksum mismatch: ' + name)
    index = json.loads((root / 'input/pool/full-results.json').read_text())
    if index['completed'] != 1000 or len(index['tokenizers']) != 1000:
        raise ValueError('Incomplete tokenizer pool index')
    for row in index['tokenizers']:
        name = f"input/pool/tokenizers/{row['config']['index']:04d}/tokenizer.json"
        if expected[name] != dict(bytes=row['tokenizer_bytes'], sha256=row['tokenizer_sha256']):
            raise ValueError('Tokenizer index and bundle manifest disagree')
    if sum(v['bytes'] for v in expected.values()) != manifest['total_bytes']:
        raise ValueError('Total bundle size mismatch')
    return dict(verified_files=len(expected), tokenizers=1000, corpus_buckets=23,
                input_bytes=manifest['total_bytes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Extracted directory containing input/ and BUNDLE_MANIFEST.json')
    print(json.dumps(verify(parser.parse_args().root)))
