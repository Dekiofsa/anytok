"""Separate sharing, learning speed, noise, and practice budget in synthetic students."""
import math

import numpy as np

SHARING=(0.,.01,.1,1.)
SPEEDS=(.01,.25,1.,4.,64.,256.)
METHODS=('reinforce','coverage','uniform','current_hardest')
SEEDS=(48,49,50,51,52)
STEPS=20000


def reference_rate(calibration):
    ratio=calibration['end_gap']/calibration['start_gap']
    return -math.log((ratio-.5)/.5)/calibration['selected_visits']


def group_ids(configs,kind):
    algorithm=np.array([('bpe','wordpiece','unigram').index(c['algorithm']) for c in configs])
    if kind=='algorithm':return algorithm
    if kind=='fine':
        return 3*algorithm+np.array([('regex','line256','no-regex').index(c['pretokenizer']) for c in configs])
    if kind=='random':return np.random.default_rng(20261118).permutation(algorithm)
    if kind=='global':return np.zeros(len(configs),dtype=np.int64)
    raise ValueError('Unknown skill grouping')


def worlds():
    base=dict(sharing=1.,speed=1.,budget='extra',grouping='algorithm',floor=.5,
              noise=1.,curve='exponential',starting_gaps='measured')
    result=[];seen={}
    def add(study,label,**changes):
        config=dict(base,**changes)
        if config['speed']==0:
            config.update(sharing=0.,budget='extra',grouping='algorithm',floor=.5,curve='exponential')
        if config['sharing']==0:
            config.update(budget='extra',grouping='algorithm')
        key=tuple(sorted(config.items()))
        if key in seen:return
        row=dict(id=f'world-{len(result):03d}',study=study,label=label,**config)
        result.append(row);seen[key]=row['id']
    add('map','Student stays unchanged',speed=0.)
    for budget in ['extra','shared']:
        for sharing in SHARING:
            for speed in SPEEDS:
                add('map',f'Sharing {sharing:g}, speed {speed:g}, {budget} practice',
                    sharing=sharing,speed=speed,budget=budget)
    for sharing in [.1,1.]:
        for noise in [0.,2.]:add('checks',f'{sharing:g} sharing; noise {noise:g}×',sharing=sharing,noise=noise)
        for floor in [0.,.9]:add('checks',f'{sharing:g} sharing; plateau at {floor:.0%}',sharing=sharing,floor=floor)
        for grouping,label in [('fine','smaller skill groups'),('random','shuffled groups'),('global','all skills')]:
            add('checks',f'{sharing:g} sharing; {label}',sharing=sharing,grouping=grouping)
        for curve in ['linear','inverse']:
            add('checks',f'{sharing:g} sharing; {curve} learning',sharing=sharing,curve=curve)
        for layout in ['flat','shuffled']:
            add('checks',f'{sharing:g} sharing; {layout} starting difficulty',sharing=sharing,starting_gaps=layout)
    for noise in [0.,2.]:add('controls',f'Frozen; noise {noise:g}×',speed=0.,noise=noise)
    for layout in ['flat','shuffled']:
        add('controls',f'Frozen; {layout} starting difficulty',speed=0.,starting_gaps=layout)
    return result


class GeneralizedStudent:
    def __init__(self,rewards,configs,calibration,world,seed,steps=STEPS):
        values=np.asarray(rewards,dtype=np.float64)
        if values.shape!=(len(configs),32) or not np.isfinite(values).all():
            raise ValueError('All measured batch rewards required')
        self.base=values.mean(1)
        self.residuals=values-self.base[:,None]
        self.world=world
        if world['starting_gaps']=='flat':self.base=np.full_like(self.base,self.base.mean())
        elif world['starting_gaps']=='shuffled':self.base=np.random.default_rng(20261118).permutation(self.base)
        elif world['starting_gaps']!='measured':raise ValueError('Unknown starting difficulty')
        if not 0<=world['sharing']<=1 or world['speed']<0 or world['noise']<0 or not 0<=world['floor']<1:
            raise ValueError('Invalid student parameters')
        if world['budget'] not in ['extra','shared'] or world['curve'] not in ['exponential','linear','inverse']:
            raise ValueError('Unknown learning rule')
        self.groups=group_ids(configs,world['grouping'])
        self.sizes=np.bincount(self.groups)
        self.counts=np.zeros(len(configs),dtype=np.int64)
        self.group_counts=np.zeros(len(self.sizes),dtype=np.int64)
        self.floor=world['floor']*self.base
        # Always retain the same reference rate: changing the plateau must not
        # silently change the learning speed too.
        self.rate=reference_rate(calibration)*world['speed']
        self.batches=np.random.default_rng(10000+seed).integers(32,size=steps)

    def practices(self):
        sharing=self.world['sharing']
        effective=(1-sharing)*self.counts+sharing*self.group_counts[self.groups]
        if self.world['budget']=='shared':
            effective=effective/(1+sharing*(self.sizes[self.groups]-1))
        return effective

    def means(self):
        if self.rate==0:return self.base.copy()
        progress=self.rate*self.practices()
        curve=self.world['curve']
        remaining=(np.exp(-progress) if curve=='exponential' else
                   np.maximum(0.,1-progress) if curve=='linear' else 1/(1+progress))
        return self.floor+(self.base-self.floor)*remaining

    def observe(self,action,step,means=None):
        if not 0<=action<len(self.base) or not 1<=step<=len(self.batches):raise ValueError('Invalid observation')
        values=self.means() if means is None else means
        reward=float(values[action]+self.world['noise']*self.residuals[action,self.batches[step-1]])
        self.counts[action]+=1;self.group_counts[self.groups[action]]+=1
        return reward
