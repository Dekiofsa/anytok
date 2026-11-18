"""Join step: add Qingyun's tokenizer difficulty ranks and the code itself to the run's tables.

Reads tables.json (the output of extract_results.py, run on Tillicum) and the ranking that Qingyun's report
regret-pool-ranking-6500 published in the anytok repository (branch zett-modal-20k), then writes results.json:
{tables: {..., difficulty}, meta: [...]}. The ranks describe his own run's model at update 6,500, not this run:
rank 1 is the tokenizer that model found hardest among the 1,000.
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = sys.argv[1]
REF = 'origin/zett-modal-20k'
RANKING = 'reports/regret-pool-ranking-6500/tokenizer-ranking.csv'

data = json.loads((HERE / 'tables.json').read_text())
commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', REF], capture_output=True, text=True, check=True).stdout.strip()
text = subprocess.run(['git', '-C', REPO, 'show', f'{REF}:{RANKING}'], capture_output=True, text=True, check=True).stdout
ranks = [{'tokenizer': int(r['index']), 'difficulty_rank': int(r['rank'])} for r in csv.DictReader(io.StringIO(text))]
assert sorted(r['tokenizer'] for r in ranks) == list(range(1000)), 'ranking does not cover the 1,000 pool tokenizers'
by_index = {t['tokenizer']: t for t in data['tables']['tokenizers']}
for row in csv.DictReader(io.StringIO(text)):  # the ranking must describe the same pool
    t = by_index[int(row['index'])]
    assert int(row['vocab_size']) == t['vocab_size'], row['index']
data['tables']['difficulty'] = sorted(ranks, key=lambda r: r['tokenizer'])
data['meta'] = [
    {'label': 'extract_results.py', 'value': (HERE / 'extract_results.py').read_text(), 'type': 'code', 'lang': 'python'},
    {'label': 'add_difficulty.py', 'value': Path(__file__).read_text(), 'type': 'code', 'lang': 'python'},
    {'label': 'Input', 'value': 'The finished run on Tillicum project storage (events.jsonl, status.json, heldout_tokenizers.json), '
                                'the input bundle\'s tokenizer pool (pool/plan.json), Slurm accounting (sacct) and the cluster cost '
                                'report (hyakusage -p), read on 2026-11-17.'},
    {'label': 'Difficulty ranks', 'value': f'{RANKING} at {REF} ({commit}): Qingyun\'s measurement of his own run at update 6,500.'},
]
(HERE / 'results.json').write_text(json.dumps(data, separators=(',', ':')) + '\n')
print('wrote', HERE / 'results.json', 'tables:', sorted(data['tables']), 'commit', commit)
