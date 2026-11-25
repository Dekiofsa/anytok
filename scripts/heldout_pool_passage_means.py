"""Per-passage means over the 1,000 pool tokenizers of the held-out pool scores.

Reads SCORES/<label>/ for every label that has a complete.json receipt: canonical.json and
0000.json ... 0999.json, as written by Qingyun's run_rift_pool_sweep.py score (protocol
heldout-pool-4992, 948787e). Prints one JSON object keyed by label. Each entry holds the plan
hash, the canonical per-passage losses and, for every passage, the mean over all 1,000
tokenizers of the transferred loss (nats summed over the passage's 32 scored tokens, as in
his files). The pool-mean gap is linear in these numbers, so the report can split it by
domain and bootstrap passages from them alone.

Read-only and stdlib only, so it runs on Tillicum's system Python 3.9:
    python3 -I - SCORES < heldout_pool_passage_means.py > passage-means.json
"""
import json
import math
import sys
from pathlib import Path

TOKENIZERS = 1000


def passage_means(folder):
    receipt = json.loads((folder / 'complete.json').read_text())
    canonical = json.loads((folder / 'canonical.json').read_text())
    key = dict(plan_sha256=receipt['plan_sha256'], checkpoint=receipt['checkpoint'])
    if (receipt['checkpoint'] != folder.name or receipt.get('completed') != TOKENIZERS
            or any(canonical.get(k) != v for k, v in key.items())):
        raise ValueError(f'{folder.name}: the receipt and the canonical scores disagree')
    columns = [[] for _ in canonical['nats']]
    for index in range(TOKENIZERS):
        result = json.loads((folder / f'{index:04d}.json').read_text())
        if (any(result.get(k) != v for k, v in key.items()) or result.get('index') != index
                or len(result['nats']) != len(columns)):
            raise ValueError(f'{folder.name}/{index:04d}.json does not match its receipt')
        for column, value in zip(columns, result['nats']):
            column.append(value)
    return dict(key, tokenizers=TOKENIZERS, canonical_nats=canonical['nats'],
                mean_transferred_nats=[math.fsum(column) / TOKENIZERS for column in columns])


def main():
    scores = Path(sys.argv[1])
    folders = sorted(p for p in scores.iterdir() if (p / 'complete.json').is_file())
    json.dump({folder.name: passage_means(folder) for folder in folders}, sys.stdout, sort_keys=True)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
