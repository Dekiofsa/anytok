"""Collect small status records for the separate full-pool evaluation."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_pool_ranking'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


if __name__ == '__main__':
    statuses = {str(i): read(STATE / 'status' / f'{i}.json') for i in range(4)}
    paths = sorted((STATE / 'scores').glob('*.json'))
    complete = len(paths) == 1000 and all(s and s['stage'] == 'complete' for s in statuses.values())
    if complete and not (STATE / 'all-scores.json.gz').exists():
        temporary = STATE / 'all-scores.tmp.json.gz'
        with gzip.open(temporary, 'wt') as stream:
            json.dump([json.loads(p.read_text()) for p in paths], stream)
        temporary.replace(STATE / 'all-scores.json.gz')
    print(json.dumps(dict(preparation=read(STATE / 'preparation-status.json'),
                          workers=statuses, completed=len(paths), complete=complete)))
