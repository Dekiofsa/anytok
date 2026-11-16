"""Trainer interface for the unchanged online predictor used in the frozen pilot."""
from collections import deque
import math

import numpy as np
import torch

from hardness_predictor import HardnessPredictor


class SettingsTokenizerBandit(HardnessPredictor):
    """Compatibility name only: single-observation regression, never REINFORCE."""
    def __init__(self, configs, seed=43, learning_rate=.001, entropy_weight=0., baseline_decay=.99):
        if learning_rate!=.001 or entropy_weight!=0. or baseline_decay!=.99:
            raise ValueError('This run uses exactly the approved predictor settings')
        super().__init__(configs, seed)
        self.recent=deque(maxlen=500)
        self._conditional=None

    def sample(self):
        action=self.select()
        p=np.zeros(1000)
        if action['mode']=='explore':
            p[action['index']]=1.
        else:
            values=self.predictions();ties=values==values.max()
            p[ties]=1./int(ties.sum())
        self._conditional=p
        return action['index']

    def probabilities(self):
        if self.pending is None or self._conditional is None:
            raise RuntimeError('Conditional choice probabilities exist only for the pending choice')
        return self._conditional.copy()

    def update(self,index,reward_bpb):
        action=dict(self.pending) if self.pending is not None else None
        error=self.observe(index,float(reward_bpb))
        self._conditional=None;self.recent.append(index)
        counts=np.bincount(self.recent,minlength=1000)
        p=counts[counts>0]/len(self.recent)
        entropy=float(-(p*np.log(p)).sum())
        return dict(selector_mode=action['mode'],selector_prediction_bpb=action['predicted_gap'],
            selector_reward_bpb=float(reward_bpb),selector_squared_error=error,
            selector_updates=self.updates,selector_unique_tokenizers=int((self.counts>0).sum()),
            selector_exploration_visits=int(self.exploration_counts.sum()),
            selector_recent_steps=len(self.recent),selector_recent_entropy=entropy,
            selector_recent_effective_tokenizers=math.exp(entropy),
            selector_recent_largest_share=float(p.max()))

    def weights_dict(self):
        if self.pending is not None:raise RuntimeError('Save only after the reward update')
        return dict(settings=self.settings,network=self.network.state_dict(),
            predictions=torch.from_numpy(self.predictions()),counts=self.counts.clone(),
            exploration_counts=self.exploration_counts.clone(),updates=self.updates,
            recent_counts=torch.from_numpy(np.bincount(self.recent,minlength=1000)),
            recent_steps=len(self.recent),kind='regret_predictor_with_shuffled_coverage')

    def state_dict(self):
        return {**super().state_dict(),'recent_indices':list(self.recent)}

    def load_state_dict(self,state):
        recent=state['recent_indices']
        if len(recent)!=min(500,state['updates']) or any(not isinstance(i,int) or not 0<=i<1000 for i in recent):
            raise ValueError('Invalid recent-choice metadata in restart')
        super().load_state_dict(state)
        self.recent=deque(recent,maxlen=500);self._conditional=None
