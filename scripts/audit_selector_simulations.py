"""Independently reconstruct every simulated reward and current oracle value."""
import heapq
import itertools
import json

import numpy as np

from package_training_inputs import atomic_json, sha256
from selector_simulation import load_inputs, ROOT, SCENARIOS, METHODS


def main():
    state=ROOT/'modal_results/selector_simulations'
    report=ROOT/'reports/selector-simulations'
    plan=json.loads((state/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('Pinned source changed: '+name)
    configs,rewards,calibration=load_inputs()
    base=rewards.mean(axis=1);residuals=rewards-base[:,None]
    families=np.array([('bpe','wordpiece','unigram').index(row['algorithm']) for row in configs])
    maxima=np.array([base[families==group].max() for group in range(3)])
    checked=0;steps=plan['steps'];rate=calibration['decay_per_visit']
    for scenario,method,seed in itertools.product(SCENARIOS,METHODS,plan['seeds']):
        folder=state/'cells'/f'{scenario}-{method}-{seed}'
        receipt=json.loads((folder/'complete.json').read_text())
        if receipt['steps']!=steps or receipt['trace_sha256']!=sha256(folder/'trace.npz'):
            raise ValueError('Missing or changed completed trace')
        with np.load(folder/'trace.npz') as trace:
            assert all(trace[key].shape==(steps,) for key in trace.files)
            action=trace['actions'];assert ((action>=0)&(action<1000)).all()
            assert ((trace['selected_rank']>=1)&(trace['selected_rank']<=1000)).all()
            if method=='predictor':
                exploration=trace['exploration'].astype(bool)
                np.testing.assert_array_equal(np.flatnonzero(exploration),np.arange(0,steps,4))
                np.testing.assert_array_equal(np.bincount(action[exploration],minlength=1000),np.full(1000,5))
            if scenario in ('fixed','gradual','abrupt'):
                progress=(np.clip((np.arange(1,steps+1)-10000)/1500,0.,1.) if scenario=='gradual' else
                          2.*(np.arange(1,steps+1)>=10000) if scenario=='abrupt' else np.zeros(steps))
                shift=progress*calibration['observed_drop']
                selected=base[action]-(families[action]==2)*shift
                oracle=np.maximum(np.maximum(maxima[0],maxima[1]),maxima[2]-shift)
            elif scenario=='practice_family':
                pre_counts=np.array([np.r_[0,np.cumsum(families[action]==group)[:-1]] for group in range(3)])
                factors=np.exp(-rate*pre_counts)
                selected=base[action]*factors[families[action],np.arange(steps)]
                oracle=(maxima[:,None]*factors).max(axis=0)
            else:
                counts=np.zeros(1000,dtype=int);queue=[(-float(value),index,0) for index,value in enumerate(base)]
                heapq.heapify(queue);selected=np.empty(steps);oracle=np.empty(steps)
                for step,index in enumerate(action):
                    while queue[0][2]!=counts[queue[0][1]]:heapq.heappop(queue)
                    oracle[step]=-queue[0][0]
                    selected[step]=base[index]*np.exp(-rate*counts[index])
                    counts[index]+=1
                    heapq.heappush(queue,(-float(base[index]*np.exp(-rate*counts[index])),int(index),int(counts[index])))
            batches=np.random.default_rng(10000+seed).integers(32,size=steps)
            expected=selected+residuals[action,batches]
            np.testing.assert_allclose(trace['chosen_mean'],selected,rtol=0,atol=1e-14)
            np.testing.assert_allclose(trace['oracle_mean'],oracle,rtol=0,atol=1e-14)
            np.testing.assert_allclose(trace['rewards'],expected,rtol=0,atol=1e-14)
            assert (oracle-selected>=-1e-14).all()
            checked+=steps
    result=dict(complete_cells=75,observations_reconstructed=checked,tokenizers=1000,measured_batches_per_tokenizer=32,
        all_pinned_source_hashes_match=True,all_trace_hashes_match=True,
        every_observed_reward_and_current_oracle_reconstructed=True,
        predictor_explores_every_tokenizer_exactly_five_times_in_every_cell=True,
        real_lm_or_hypernetwork_updates=0)
    atomic_json(state/'completion-audit.json',result)
    atomic_json(report/'completion-audit.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
