"""Plateauing synthetic students and a controlled REINFORCE coverage intervention."""
import copy
import math

import numpy as np
import torch

from hardness_predictor import _features

SCENARIOS=('frozen','individual','family')
INITIALIZATIONS=('fresh','collapsed')
COVERAGES=(0.,.25)


class SyntheticStudent:
    """Observed gap = learnable deficit + floor + unchanged empirical batch noise."""
    def __init__(self,rewards,configs,calibration,scenario,seed,steps=20000,floor_fraction=.5):
        values=np.asarray(rewards,dtype=np.float64)
        if scenario not in SCENARIOS or values.shape!=(len(configs),32):
            raise ValueError('Unknown student rule or incomplete measured rewards')
        self.base=values.mean(1)
        if not np.isfinite(values).all() or (self.base<=0).any() or not 0<=floor_fraction<1:
            raise ValueError('Require finite measured rewards and positive initial mean gaps')
        self.floor=self.base*floor_fraction
        self.noise=values-self.base[:,None]
        self.family=np.array([('bpe','wordpiece','unigram').index(c['algorithm']) for c in configs])
        self.visits=np.zeros(len(configs),dtype=np.int64)
        self.scenario=scenario
        ratio=calibration['end_gap']/calibration['start_gap']
        if not floor_fraction<ratio<1:
            raise ValueError('Observed calibration cannot be represented by this floor')
        self.rate=-math.log((ratio-floor_fraction)/(1-floor_fraction))/calibration['selected_visits']
        self.batches=np.random.default_rng(10000+seed).integers(32,size=steps)

    def means(self):
        if self.scenario=='frozen':return self.base.copy()
        practices=self.visits.astype(float)
        if self.scenario=='family':
            practices=np.bincount(self.family,weights=practices,minlength=3)[self.family]
        return self.floor+(self.base-self.floor)*np.exp(-self.rate*practices)

    def observe(self,action,step):
        if not 0<=action<len(self.base) or not 1<=step<=len(self.batches):
            raise ValueError('Invalid action or observation step')
        reward=float(self.means()[action]+self.noise[action,self.batches[step-1]])
        self.visits[action]+=1
        return reward


def mixed_probabilities(policy,coverage):
    if not 0<=coverage<1:raise ValueError('Invalid uniform mixture')
    p=policy.probabilities()
    return p,(1-coverage)*p+coverage/len(p)


def update_policy(policy,action,reward,coverage):
    """Differentiate the actual mixed sampling policy; no off-policy log-p shortcut."""
    if coverage==0:return policy.update(action,reward)
    if not 0<coverage<1 or not math.isfinite(reward):raise ValueError('Invalid observation')
    advantage=0. if policy.baseline is None else float(reward)-policy.baseline
    logp=policy.logits.log_softmax(0)
    logq=torch.logaddexp(logp+math.log1p(-coverage),
        torch.full_like(logp,math.log(coverage/len(logp))))
    loss=-advantage*logq[action]
    policy.optimizer.zero_grad(set_to_none=True);loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.network.parameters()):
        raise FloatingPointError('Nonfinite mixed-policy gradient')
    policy.optimizer.step()
    if any(not torch.isfinite(p).all() for p in policy.network.parameters()):
        raise FloatingPointError('Nonfinite policy weights')
    decay=policy.settings['baseline_decay']
    policy.baseline=float(reward) if policy.baseline is None else decay*policy.baseline+(1-decay)*float(reward)
    policy.counts[action]+=1;policy.updates+=1
    return dict(policy_advantage_bpb=advantage,policy_selected_probability=float(policy.probabilities()[action]))


def restore(configs,state,continuation_seed):
    policy=_features.SettingsTokenizerBandit(configs,seed=43,entropy_weight=0.)
    policy.load_state_dict(copy.deepcopy(state))
    # All conditions retain the same fitted weights/Adam/EMA state. Only future
    # sampling and batch streams vary across the five continuation repetitions.
    policy.rng=np.random.default_rng(continuation_seed)
    return policy
