#!/usr/bin/env python3
"""Write manifest.json: the size and SHA-256 of every other file in this report folder.

Standard library only:  python3 code/write_manifest.py   (run it last, after any other change)
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = {}
for path in sorted(ROOT.rglob('*')):
    rel = path.relative_to(ROOT).as_posix()
    if not path.is_file() or rel == 'manifest.json' or path.name == '.DS_Store' or '__pycache__' in path.parts:
        continue
    data = path.read_bytes()
    files[rel] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
(ROOT / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(f'manifest.json: {len(files)} files')
