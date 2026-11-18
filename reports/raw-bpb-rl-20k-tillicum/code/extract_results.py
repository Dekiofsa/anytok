"""Read-only: summarize the finished raw-BPB run for the results page.

Runs on the Tillicum login node with the run's own Python. It reads the run's event log, status file, tokenizer
pool and held-out tokenizer list, plus Slurm's accounting (sacct) and the cluster's cost report (hyakusage -p), and
prints one JSON object {tables: {...}} to stdout. It writes nothing anywhere.

Conventions:
  * events.jsonl holds updates 4,001-5,000 twice (the crashed first job, then the resumed run); the last row for
    each step is the one the finished run kept, so every table uses the last row per (stage, step).
  * "Score" is bits per byte (BPB) on the run's built-in check: two tokenizers held out of training, about
    10,000 bytes of text each. unseen_bpb uses the held-out tokenizer; own_bpb scores the same bytes with the
    model's original tokenizer.
"""
import collections
import csv
import io
import json
import os
import subprocess
import sys

USER = os.environ['USER']
RUN = f'/gpfs/projects/stf/{USER}/anytok-runs/raw-bpb-20k'
POOL = f'/gpfs/projects/stf/{USER}/anytok-data/input/pool/plan.json'
METHOD = {'bpe': 'BPE', 'unigram': 'Unigram', 'wordpiece': 'WordPiece'}
SPLIT = {'no-regex': 'none', 'regex': 'words', 'line256': 'lines'}


rows =[json.loads(line) for line in open(f'{RUN}/events.jsonl') if line.strip()]
last = {}
for row in rows:
    last[(row.get('stage'), row.get('step'))] = row
train = {step: row for (stage, step), row in last.items() if stage == 'train'}
steps = sorted(train)
assert steps == list(range(1, 20001)), 'train steps are not exactly 1..20000'

status = json.load(open(f'{RUN}/status.json'))
run = [{'steps_done': status['step'], 'steps_total': status['total'], 'state': status.get('stage'),
        'finished_utc': status.get('at_utc')}]

evals = []
initial = [row for row in rows if row.get('stage') == 'initial_eval'][-1]
for step, row in [(0, initial)] + sorted(((s, r) for (st, s), r in last.items() if st == 'eval'), key=lambda x: x[0]):
    for i, result in enumerate(row['results'], start=1):
        evals.append({'step': step, 'check_tokenizer': i, 'unseen_bpb': round(result['generated_bpb'], 5),
                      'own_bpb': round(result['canonical_bpb'], 5), 'passages': result['examples'],
                      'bytes': result['bytes']})

focus = [{'step': s, 'favorite_chance': round(train[s]['policy_max_probability'], 4),
          'effective_choices': round(train[s]['policy_effective_tokenizers'], 2)}
         for s in [1] + list(range(50, 20001, 50))]

plan = json.load(open(POOL))
tokenizers = []
for position, cfg in enumerate(plan['configs']):
    assert cfg['index'] == position, (position, cfg['index'])
    tokenizers.append({'tokenizer': cfg['index'], 'method': METHOD[cfg['algorithm']],
                       'pre_split': SPLIT[cfg['pretokenizer']], 'vocab_size': cfg['vocab_size']})
assert len(tokenizers) == 1000, len(tokenizers)
for s in (1, 20000):  # the trainer's pool index must name the same tokenizer as this table
    assert train[s]['tokenizer_vocab_size'] == tokenizers[train[s]['tokenizer_pool_index']]['vocab_size'], s

segments, method_picks = [], []
for end in range(500, 20001, 500):
    picks = collections.Counter(train[s]['tokenizer_pool_index'] for s in range(end - 499, end + 1))
    favorite, count = picks.most_common(1)[0]
    segments.append({'from_step': end - 499, 'to_step': end, 'favorite_tokenizer': favorite, 'favorite_picks': count,
                     'different_tokenizers': len(picks)})
    by_method = collections.Counter()
    for index, n in picks.items():
        by_method[tokenizers[index]['method']] += n
    for method in ('BPE', 'Unigram', 'WordPiece'):
        method_picks.append({'from_step': end - 499, 'to_step': end, 'method': method, 'picks': by_method[method]})

# The two check tokenizers are vocabularies of (piece, score) pairs that the baseline run also used
# (the trainer copies the baseline's fixed panel); none of them is in the 1,000-tokenizer pool.
held = json.load(open(f'{RUN}/heldout_tokenizers.json'))
assert isinstance(held, list) and len(held) == 2
check_tokenizers = [{'check_tokenizer': i, 'vocab_size': len(vocab)} for i, vocab in enumerate(held, start=1)]

sacct = subprocess.run(['sacct', '-u', USER, '-S', '2026-11-14', '-E', '2026-11-17T23:59', '-X', '-P', '-n',
                        '-o', 'JobID,JobName,State,Start,End,ElapsedRaw,AllocTRES'],
                       capture_output=True, text=True, check=True).stdout
parts = {'473379': ('Smoke test', 'A 3-step test before the real run'),
         '473452': ('First try, job 1', 'Steps 1-5,000; crashed while saving step 5,000'),
         '473453': ('First try, job 2', 'Stopped at once because job 1 had failed'),
         '473454': ('First try, job 3', 'Stopped at once because job 1 had failed'),
         '481232': ('Restart, job 1', 'Steps 4,001-16,409, resumed from the step-4,000 save'),
         '481233': ('Restart, job 2', 'Steps 16,410-20,000'),
         '481234': ('Restart, job 3', 'Spare job, cancelled before it started')}
jobs = []
for line in sacct.splitlines():
    jobid, name, state, start, end, elapsed, tres = line.split('|')
    if jobid not in parts:
        continue
    gpus = 0
    for item in tres.split(','):
        if item.startswith('gres/gpu='):
            gpus = int(item.split('=')[1])
    jobs.append({'job': jobid, 'part': parts[jobid][0], 'what_it_did': parts[jobid][1], 'state': state.split()[0],
                 'started': None if start in ('None', 'Unknown') else start, 'ended': None if end in ('None', 'Unknown') else end,
                 'gpu_hours': round(int(elapsed) * gpus / 3600, 3)})
assert {j['job'] for j in jobs} == set(parts), sorted(j['job'] for j in jobs)

usage = subprocess.run(['hyakusage', '-p', '-a', 'stf', '-s', '2026-10-26'], capture_output=True, text=True, check=True).stdout
budget = [{'account': r['account'], 'budget_start': r['budget_start'], 'budget_end': r['budget_end'],
           'budget_usd': float(r['budget_usd']), 'used_usd': float(r['usage_usd'])}
          for r in csv.DictReader(io.StringIO(usage)) if r['user'] == USER]
assert len(budget) == 1, budget

json.dump({'tables': {'run': run, 'evals': evals, 'check_tokenizers': check_tokenizers, 'focus': focus,
                      'segments': segments, 'method_picks': method_picks, 'tokenizers': tokenizers, 'jobs': jobs,
                      'budget': budget}}, sys.stdout, indent=None, separators=(',', ':'))
