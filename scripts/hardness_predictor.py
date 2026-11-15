"""Settings-to-regret regression with one coverage choice in every four steps."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

FEATURE_SOURCE = Path(__file__).resolve().parents[1] / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
spec = importlib.util.spec_from_file_location('hardness_settings_features', FEATURE_SOURCE)
_features = importlib.util.module_from_spec(spec); spec.loader.exec_module(_features)


class CoverageSchedule:
    def __init__(self, size=1000, seed=43):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(size)
        self.cursor = 0
        self.steps = 0

    def choose(self, predictions):
        if len(predictions) != self.size or not np.isfinite(predictions).all():
            raise ValueError('One finite prediction per pool action required')
        explore = self.steps % 4 == 0
        if explore:
            if self.cursor == self.size:
                self.order = self.rng.permutation(self.size); self.cursor = 0
            index = int(self.order[self.cursor]); self.cursor += 1
        else:
            # Random tie breaking preserves symmetry before rewards distinguish settings.
            ties = np.flatnonzero(predictions == predictions.max())
            index = int(ties[0] if len(ties) == 1 else self.rng.choice(ties))
        self.steps += 1
        return index, 'explore' if explore else 'exploit'

    def state_dict(self):
        return dict(size=self.size, rng=copy.deepcopy(self.rng.bit_generator.state),
                    order=self.order.tolist(), cursor=self.cursor, steps=self.steps)

    def load_state_dict(self, state):
        if (state['size'] != self.size or sorted(state['order']) != list(range(self.size))
                or not 0 <= state['cursor'] <= self.size or state['steps'] < 0):
            raise ValueError('Invalid exploration-cycle state')
        self.rng.bit_generator.state = state['rng']
        self.order = np.asarray(state['order'], dtype=int)
        self.cursor = state['cursor']; self.steps = state['steps']


class HardnessPredictor:
    def __init__(self, configs, seed=43):
        if len(configs) != 1000:
            raise ValueError('The complete 1,000-tokenizer pool is required')
        self.features = torch.from_numpy(_features.settings_features(configs))
        self.feature_sha256 = hashlib.sha256(self.features.numpy().tobytes()).hexdigest()
        # Reuse the original architecture, initialization and optimizer settings.
        original = _features.SettingsTokenizerBandit(configs, seed=seed,
                    learning_rate=.001, entropy_weight=0.)
        self.network = original.network
        self.optimizer = original.optimizer
        self.settings = dict(kind='online_hardness_regression', hidden_sizes=[64,64],
            learning_rate=.001, loss='single_observation_squared_error', seed=seed,
            exploration_period=4, warmup_updates=0, replay=False,
            feature_sha256=self.feature_sha256,
            feature_source_sha256=hashlib.sha256(FEATURE_SOURCE.read_bytes()).hexdigest())
        self.schedule = CoverageSchedule(1000, seed)
        self.counts = torch.zeros(1000, dtype=torch.int64)
        self.exploration_counts = torch.zeros(1000, dtype=torch.int64)
        self.updates = 0
        self.pending = None

    def predictions(self):
        with torch.no_grad():
            values = self.network(self.features).squeeze(-1).numpy().copy()
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite hardness predictions')
        return values

    def select(self):
        if self.pending is not None:
            raise RuntimeError('Observe the real reward before selecting again')
        values = self.predictions()
        index, mode = self.schedule.choose(values)
        self.pending = dict(index=index, mode=mode, predicted_gap=float(values[index]))
        return dict(self.pending)

    def observe(self, index, reward):
        if self.pending is None or self.pending['index'] != index or not math.isfinite(reward):
            raise ValueError('Expected the selected action and its finite real reward')
        value = self.network(self.features[index:index+1]).squeeze()
        loss = (value - float(reward)).square()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
            raise FloatingPointError('Nonfinite predictor gradient')
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.network.parameters()):
            raise FloatingPointError('Nonfinite predictor parameters')
        self.counts[index] += 1
        if self.pending['mode'] == 'explore':
            self.exploration_counts[index] += 1
        self.updates += 1; self.pending = None
        return float(loss.detach())

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError('Checkpoint only after the selected reward is observed')
        return dict(settings=self.settings, network=self.network.state_dict(),
            optimizer=self.optimizer.state_dict(), schedule=self.schedule.state_dict(),
            counts=self.counts.clone(), exploration_counts=self.exploration_counts.clone(), updates=self.updates)

    def load_state_dict(self, state):
        if (state['settings'] != self.settings or state['schedule']['steps'] != state['updates']
                or state['counts'].shape != (1000,) or state['exploration_counts'].shape != (1000,)
                or int(state['counts'].sum()) != state['updates']
                or int(state['exploration_counts'].sum()) != (state['updates'] + 3) // 4
                or (state['counts'] < state['exploration_counts']).any()
                or (state['exploration_counts'] < 0).any()):
            raise ValueError('Predictor checkpoint counters or settings changed')
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.schedule.load_state_dict(state['schedule'])
        self.counts.copy_(state['counts']); self.exploration_counts.copy_(state['exploration_counts'])
        self.updates = state['updates']; self.pending = None
