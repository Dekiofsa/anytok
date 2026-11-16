"""Read-only view of every action and world in the completed selector simulations."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from scripts.selector_simulation import load_inputs, SCENARIOS, METHODS, LABELS
from scripts.package_training_inputs import sha256


class SelectorExplorer:
    def __init__(self,root):
        self.root=Path(root)
        self.configs,self.rewards,self.calibration=load_inputs()
        self.base=self.rewards.mean(axis=1)
        self.families=np.array([('bpe','wordpiece','unigram').index(row['algorithm']) for row in self.configs])
        self.frames=np.r_[1,np.arange(25,20001,25)]

    @lru_cache(maxsize=12)
    def trace(self,scenario,method,seed):
        path=self.root/'reports/selector-simulations/traces'/f'{scenario}-{method}-{seed}.npz'
        with np.load(path) as file:return {key:file[key] for key in file.files}

    @lru_cache(maxsize=12)
    def beliefs(self,scenario,method,seed):
        folder=self.root/'modal_results/selector_playback_25'/f'{scenario}-{method}-{seed}'
        if not (folder/'complete.json').exists():raise FileNotFoundError('Exact selector beliefs are still being reconstructed')
        receipt=json.loads((folder/'complete.json').read_text())
        if not receipt['actions_match_every_step'] or sha256(folder/'values.npz')!=receipt['values_sha256']:
            raise ValueError('Playback belief verification failed')
        source=self.root/'reports/selector-simulations/traces'/f'{scenario}-{method}-{seed}.npz'
        if receipt['source']['trace_sha256']!=sha256(source):
            raise ValueError('Playback beliefs refer to a different simulation trace')
        with np.load(folder/'values.npz') as file:
            np.testing.assert_array_equal(file['steps'],self.frames)
            values=file['values']
            if values.shape!=(len(self.frames),1000) or not np.isfinite(values).all():
                raise ValueError('Incomplete selector estimates')
            if method!='predictor' and (not ((values>=0)&(values<=1)).all() or not np.allclose(values.sum(axis=1),1.,rtol=0,atol=1e-12)):
                raise ValueError('Invalid cached selection probabilities')
            return values

    def metadata(self):
        scenarios=[dict(id=name,name=LABELS[name]) for name in SCENARIOS]
        return dict(scenarios=scenarios,methods=list(METHODS),seeds=list(range(43,48)),steps=self.frames.tolist(),
            tokenizers=[dict(index=row['index'],algorithm=row['algorithm'],pretokenizer=row['pretokenizer'],
                vocab_size=row['vocab_size'],noise_per_million=row['noise_per_million'],
                domain_bytes=row['domain_bytes'],initial_mean=float(self.base[row['index']]),
                measured_noise_sd=float(self.rewards[row['index']].std(ddof=1))) for row in self.configs],
            calibration=self.calibration,
            order=sorted(range(1000),key=lambda i:(self.families[i],self.configs[i]['vocab_size'],i)))

    def frame(self,scenario,seed,frame):
        if scenario not in SCENARIOS or seed not in range(43,48) or not 0<=frame<len(self.frames):
            raise ValueError('Unknown scenario, seed, or slider frame')
        step=int(self.frames[frame]);methods={}
        for method in METHODS:
            trace=self.trace(scenario,method,seed);past=trace['actions'][:step-1]
            counts=np.bincount(past,minlength=1000)
            if scenario in ('gradual','abrupt'):
                progress=(np.clip((step-10000)/1500,0.,1.) if scenario=='gradual' else 2.*float(step>=10000))
                means=self.base-(self.families==2)*self.calibration['observed_drop']*progress
            elif scenario in ('practice_tokenizer','practice_family'):
                visits=(np.bincount(self.families,weights=counts,minlength=3)[self.families] if scenario=='practice_family' else counts)
                means=self.base*np.exp(-self.calibration['decay_per_visit']*visits)
            else:means=self.base.copy()
            action=int(trace['actions'][step-1]);best=int(means.argmax())
            if abs(means[action]-float(trace['chosen_mean'][step-1]))>1e-12 or abs(means[best]-float(trace['oracle_mean'][step-1]))>1e-12:
                raise ValueError('Interactive world differs from audited simulation')
            if int(trace['selected_rank'][step-1])!=1+int((means>means[action]).sum()):
                raise ValueError('Interactive rank differs from original simulation')
            try:
                beliefs=(np.full(1000,.001) if method=='uniform' else self.beliefs(scenario,method,seed)[frame])
                belief_error=None
            except FileNotFoundError as error:
                beliefs=None;belief_error=str(error)
            recent=trace['actions'][max(0,step-1000):step]
            recent_counts=np.bincount(recent,minlength=1000)
            leaders=np.argsort(-recent_counts,kind='stable')[:5]
            fraction=np.array([np.mean(self.families[recent]==family) for family in range(3)])
            # Reveal this observed outcome, but use beliefs BEFORE that outcome.
            observed=float(trace['rewards'][step-1]);mean=float(means[action])
            # Curves cover every saved observation, not just slider positions.
            shortfall=trace['oracle_mean']-trace['chosen_mean']
            curve=shortfall.reshape(200,100).mean(axis=1)
            methods[method]=dict(means=means.tolist(),beliefs=beliefs.tolist() if beliefs is not None else None,
                belief_error=belief_error,counts=counts.tolist(),selected=action,best=best,
                chosen_mean=mean,best_mean=float(means[best]),observed_reward=observed,noise=observed-mean,
                missed_hardness=float(means[best]-mean),rank=int(trace['selected_rank'][step-1]),
                mode=('coverage exploration' if trace['exploration'][step-1] else 'highest predicted gap') if method=='predictor' else 'sampled softmax' if method=='reinforce' else 'uniform random',
                visited_before_step=int((counts>0).sum()),largest_recent_share=float(recent_counts.max()/len(recent)),
                recent_family_shares=fraction.tolist(),recent_top=[dict(index=int(i),count=int(recent_counts[i])) for i in leaders if recent_counts[i]],
                recent_steps=len(recent),curve=curve.tolist(),
                post_step_changes=int((self.families==self.families[action]).sum()) if scenario=='practice_family' else 1 if scenario=='practice_tokenizer' else 0)
        return dict(scenario=scenario,seed=seed,frame=frame,step=step,methods=methods,
            timing='World and selector estimates are before this step; selected action and reward are revealed from the saved trace.')
