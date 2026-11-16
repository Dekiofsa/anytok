"""Measured finite reward table plus explicitly synthetic changing environments."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('fixed', 'gradual', 'abrupt', 'practice_tokenizer', 'practice_family')
METHODS = ('uniform', 'reinforce', 'predictor')
LABELS = dict(fixed='Fixed measured rewards', gradual='Gradual observed-size change',
    abrupt='Abrupt change · 2× observed size', practice_tokenizer='Practice improves selected tokenizer',
    practice_family='Practice improves its algorithm family')


def load_inputs():
    reference = ROOT/'reports/regret-pool-ranking-6500'
    with gzip.open(reference/'all-scores.json.gz', 'rt') as stream:
        scores = json.load(stream)
    # Entries are explicitly indexed; never infer tokenizer identity from rank.
    entries = json.loads((reference/'pool-index.json').read_text())['tokenizers']
    configs = sorted([row['config'] for row in entries], key=lambda row: row['index'])
    if [row['index'] for row in configs] != list(range(1000)):
        raise ValueError('The complete original tokenizer pool is required')
    rewards = np.full((1000, 32), np.nan)
    seen = []
    for row in scores:
        index = row['index']; seen.append(index)
        if sorted(batch['index'] for batch in row['batches']) != list(range(32)):
            raise ValueError('Every tokenizer requires all 32 measured batches')
        for batch in row['batches']:
            if batch['valid_examples'] != 128 or batch['maximum_positions'] > 128:
                raise ValueError('Source scoring protocol changed')
            rewards[index, batch['index']] = batch['gap_bpb']
    if sorted(seen) != list(range(1000)) or not np.isfinite(rewards).all():
        raise ValueError('Incomplete reward table')
    with (reference/'tokenizer-ranking.csv').open() as stream:
        for row in csv.DictReader(stream):
            if abs(float(row['gap_bpb']) - rewards[int(row['index'])].mean()) > 1e-12:
                raise ValueError('Reward table and published means differ')
    with (ROOT/'reports/regret-favorite-trajectory/learning-curve.csv').open() as stream:
        trajectory = {int(row['step']): row for row in csv.DictReader(stream)}
    before, after = trajectory[6500], trajectory[8000]
    start, end = float(before['gap_bpb']), float(after['gap_bpb'])
    selections = int(after['cumulative_selections']) - int(before['cumulative_selections'])
    calibration = dict(start_step=6500, end_step=8000, start_gap=start, end_gap=end,
        observed_drop=start-end, elapsed_steps=1500, selected_visits=selections,
        decay_per_visit=float(-np.log(end/start)/selections),
        causal=False, note='Observed joint-training change; not a causal effect per selection.')
    return configs, rewards, calibration


class SimulatedRewards:
    """The learner receives only observe(action, step); means are evaluator-only."""
    def __init__(self, rewards, configs, calibration, scenario, seed, steps=20000):
        if scenario not in SCENARIOS:
            raise ValueError('Unknown simulation')
        self.base = np.asarray(rewards, dtype=np.float64).mean(axis=1)
        self.residuals = np.asarray(rewards, dtype=np.float64)-self.base[:, None]
        self.algorithms = np.array([row['algorithm'] for row in configs])
        self.family_ids = np.array([('bpe', 'wordpiece', 'unigram').index(name) for name in self.algorithms])
        self.hard_family = self.algorithms == 'unigram'
        self.counts = np.zeros(len(configs), dtype=np.int64)
        self.scenario, self.calibration = scenario, calibration
        # The same seed gives every method the same sequence of text-batch draws.
        self.batches = np.random.default_rng(10000+seed).integers(rewards.shape[1], size=steps)

    def means(self, step):
        if self.scenario == 'fixed':
            return self.base.copy()
        if self.scenario in ('gradual', 'abrupt'):
            progress = (np.clip((step-10000)/self.calibration['elapsed_steps'], 0., 1.)
                        if self.scenario == 'gradual' else 2.*float(step >= 10000))
            return self.base-self.hard_family*self.calibration['observed_drop']*progress
        visits = self.counts.astype(float)
        if self.scenario == 'practice_family':
            visits = np.bincount(self.family_ids, weights=self.counts, minlength=3)[self.family_ids]
        return self.base*np.exp(-self.calibration['decay_per_visit']*visits)

    def observe(self, action, step, means=None):
        if not 0 <= action < len(self.base):
            raise ValueError('Action outside full pool')
        means = self.means(step) if means is None else means
        reward = float(means[action]+self.residuals[action, self.batches[step-1]])
        self.counts[action] += 1
        return reward
