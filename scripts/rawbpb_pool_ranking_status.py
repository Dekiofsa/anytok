"""Collect small status records for the separate full-pool evaluation.

scripts/pool_ranking_status.py at 5f431ea with the state directory as --state and
the Slurm submission record (submissions.json, when present) under "submissions".
"""
import argparse
import gzip
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    STATE = parser.parse_args().state
    statuses = {str(i): read(STATE / 'status' / f'{i}.json') for i in range(4)}
    paths = sorted((STATE / 'scores').glob('*.json'))
    complete = len(paths) == 1000 and all(s and s['stage'] == 'complete' for s in statuses.values())
    if complete and not (STATE / 'all-scores.json.gz').exists():
        temporary = STATE / 'all-scores.tmp.json.gz'
        with gzip.open(temporary, 'wt') as stream:
            json.dump([json.loads(p.read_text()) for p in paths], stream)
        temporary.replace(STATE / 'all-scores.json.gz')
    print(json.dumps(dict(preparation=read(STATE / 'preparation-status.json'),
                          workers=statuses, completed=len(paths), complete=complete,
                          submissions=read(STATE / 'submissions.json'))))
