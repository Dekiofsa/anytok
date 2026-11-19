"""Audit and plot matched predictor/uniform checkpoint evaluations."""
import csv
import gzip
import json
import math
from pathlib import Path
import shutil

import numpy as np
from analyze_natural_transfer import bootstrap_balanced
from package_training_inputs import atomic_json, sha256
from pool_generalization import digest
from report_four_way_8k import group_members, plotting_style, savefig
from uniform_trajectory import PROFILE_DOMAINS

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/predictor_trajectory'
REPORT=ROOT/'reports/predictor-uniform-trajectory'
STEPS=[2000,4000,6000,8000]
NAMES=['uniform','predictor']


def audit(data):
    plan=data['plan']; parent=plan['parent']; rows=data['rows']; candidates=parent['candidates']
    if digest(parent)!=plan['parent_plan_sha256'] or data['preparation']['plan_sha256']!=digest(plan):
        raise ValueError('Evaluation plan changed')
    if len(candidates)!=62 or len({c['id'] for c in candidates})!=62 or sum(c['kind']=='pool' for c in candidates)!=50:
        raise ValueError('All 62 original tokenizers required')
    if len(rows)!=576 or digest([[r['id'],r['prefix_sha256'],r['canonical_continuation_ids'],r['continuation_bytes']] for r in rows])!=parent['panel_audit_sha256']:
        raise ValueError('Document IDs, prefixes or scored targets changed')
    for split,bytes_ in [('screen',27039),('confirm',26668)]:
        sub=[r for r in rows if r['split']==split]
        if len(sub)!=288 or sum(r['continuation_bytes'] for r in sub)!=bytes_ or any(sum(r['domain']==d for r in sub)!=32 for d in PROFILE_DOMAINS):
            raise ValueError('Original split or domain counts changed')
    for r in rows:
        if (not r['canonical_joint_encoding_matches'] or r['maximum_positions']>128
            or len(r['canonical_continuation_ids'])!=32 or set(r['native_prefix_lengths'])!={c['id'] for c in candidates}):
            raise ValueError('Shared scoring protocol changed')
    expected={(s,n) for s in STEPS for n in NAMES}
    index={(m['step'],m['name']):m for m in data['models']}
    provenance={(m['step'],m['name']):m for m in plan['models']}
    if len(data['models'])!=8 or set(index)!=expected or set(provenance)!=expected:
        raise ValueError('All eight actual checkpoints required')
    bases=np.empty((4,2,576)); native=np.empty((4,2,62,576))
    for si,step in enumerate(STEPS):
        for ni,name in enumerate(NAMES):
            m=index[step,name]; record=provenance[step,name]
            if record['manifest']['step']!=step or bool(m['cached'])!=(step==8000):
                raise ValueError('Checkpoint step or cached provenance mismatch')
            if step==8000:
                original=next(r for r in parent['models'] if r['name']==name)
                if record['manifest']!=original['manifest']:
                    raise ValueError('Cached 8k checkpoint identity changed')
            key=dict(step=step,model=name,panel_sha256=plan['panel_sha256'],
                plan_sha256=digest(parent) if step==8000 else digest(plan))
            def losses(row,field):
                if any(row.get(k)!=v for k,v in key.items()):
                    raise ValueError('Score provenance mismatch')
                values=np.asarray(row[field],dtype=float)
                if values.shape!=(576,32) or not np.isfinite(values).all() or (values<0).any():
                    raise ValueError('Missing or invalid complete per-token scores')
                return values.sum(-1)
            bases[si,ni]=losses(m['canonical'],'canonical')
            if len(m['tokenizers'])!=62:
                raise ValueError('Incomplete tokenizer coverage')
            for i,(c,r) in enumerate(zip(candidates,m['tokenizers'])):
                if r.get('tokenizer')!=c['id'] or r.get('tokenizer_sha256')!=c['sha256']:
                    raise ValueError('Native tokenizer provenance mismatch')
                native[si,ni,i]=losses(r,'transferred')
    return bases,native


def ci(values):
    return np.quantile(values,[.025,.975],axis=0).tolist()


def analyze(data,replicates=2000):
    base,native=audit(data); rows=data['rows']; candidates=data['plan']['parent']['candidates']
    pool=np.array([i for i,c in enumerate(candidates) if c['kind']=='pool'])
    held=np.array([i for i,c in enumerate(candidates) if c['kind']!='pool'])
    groups=group_members(candidates)
    draws=np.random.default_rng(20261119).integers(len(pool),size=(replicates,len(pool)))
    fpc=math.sqrt((1000-50)/(1000-1))
    result=dict(steps=STEPS,models=NAMES,tokenizers=candidates,splits={},
        audit=dict(verified=True,comparisons=496,new_comparisons=372,cached_comparisons=124,
            documents_per_pair=576,tokens_per_pair=18432,bytes_per_pair=53707,
            parent_plan_sha256=data['plan']['parent_plan_sha256']),
        improvement_definition='Uniform transfer gap minus Predictor transfer gap; positive favors Predictor.',
        uncertainty='Paired source-document cluster bootstrap within domains and across steps. Pool mean adds paired finite-population tokenizer resampling. One trained seed per method.',
        limitations='Actual-run comparison, including different training boundaries. Largest 8k winners are a retrospective display, not an independently selected confirmation group.')
    for split in ['screen','confirm']:
        pos=[i for i,r in enumerate(rows) if r['split']==split]; sub=[rows[i] for i in pos]
        values=np.concatenate([base[:,:,pos].reshape(8,len(pos)),native[:,:,:,pos].reshape(496,len(pos))])
        point,boot=bootstrap_balanced(sub,values,replicates,domains=PROFILE_DOMAINS)
        canonical=point[:8].reshape(4,2); absolute=point[8:].reshape(4,2,62)
        gaps=absolute-canonical[:,:,None]
        cb=boot[:,:8].reshape(replicates,4,2)
        ab=boot[:,8:].reshape(replicates,4,2,62); gb=ab-cb[:,:,:,None]
        gain=gaps[:,0]-gaps[:,1]; gain_boot=gb[:,:,0]-gb[:,:,1]
        def mean_boot(indices,values):
            selected=values[:,:,indices]; average=selected.mean(-1)
            if np.array_equal(indices,pool):
                resampled=np.take_along_axis(selected,draws[:,None,:],axis=2).mean(-1)
                return average+fpc*(resampled-average)
            return average
        pool_gain_boot=mean_boot(pool,gain_boot)
        held_gain_boot=mean_boot(held,gain_boot)
        info=dict(canonical_bpb=canonical.tolist(),absolute_bpb=absolute.tolist(),gap_bpb=gaps.tolist(),
            improvement_bpb=gain.tolist(),improvement_ci95=ci(gain_boot),
            absolute_improvement_bpb=(absolute[:,0]-absolute[:,1]).tolist(),
            pool_mean_gap=gaps[:,:,pool].mean(-1).tolist(),heldout_mean_gap=gaps[:,:,held].mean(-1).tolist(),
            pool_mean_absolute=absolute[:,:,pool].mean(-1).tolist(),heldout_mean_absolute=absolute[:,:,held].mean(-1).tolist(),
            pool_mean_improvement=gain[:,pool].mean(-1).tolist(),pool_mean_improvement_ci95=ci(pool_gain_boot),
            heldout_mean_improvement=gain[:,held].mean(-1).tolist(),heldout_mean_improvement_ci95=ci(held_gain_boot),
            pool_gain_minus_8k=(gain[:,pool].mean(-1)-gain[-1,pool].mean()).tolist(),
            pool_gain_minus_8k_ci95=ci(pool_gain_boot-pool_gain_boot[:,-1:]),
            heldout_gain_minus_8k=(gain[:,held].mean(-1)-gain[-1,held].mean()).tolist(),
            heldout_gain_minus_8k_ci95=ci(held_gain_boot-held_gain_boot[:,-1:]),groups={},domains={})
        for label,indices in groups.items():
            info['groups'][label]=dict(count=len(indices),mean_improvement=gain[:,indices].mean(-1).tolist(),
                mean_improvement_ci95=ci(gain_boot[:,:,indices].mean(-1)))
        for d in PROFILE_DOMAINS:
            di=[i for i,r in enumerate(rows) if r['split']==split and r['domain']==d]
            denom=sum(rows[i]['continuation_bytes'] for i in di)*math.log(2)
            domain_gap=native[:,:,:,di].sum(-1)/denom-base[:,:,di].sum(-1)[:,:,None]/denom
            info['domains'][d]=dict(pool_mean_improvement=(domain_gap[:,0,pool]-domain_gap[:,1,pool]).mean(-1).tolist(),
                heldout_mean_improvement=(domain_gap[:,0,held]-domain_gap[:,1,held]).mean(-1).tolist())
        result['splits'][split]=info
    gain=np.array(result['splits']['confirm']['improvement_bpb'])[-1]
    result['retrospective_top10_at_8k']=[candidates[i]['id'] for i in np.argsort(-gain)[:10]]
    return result


def plots(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotting_style(); output.mkdir(parents=True,exist_ok=True)
    steps=np.array(STEPS)/1000; a=report['splits']['screen']; b=report['splits']['confirm']
    colors=['#2F8FD0','#E78042','#8657BC','#36967B','#B95A79']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0]
    for key,label,color in [('pool','Random 50 pool','#2F8FD0'),('heldout','12 held-out','#E78042')]:
        v=np.array(b[key+'_mean_improvement']); bounds=np.array(b[key+'_mean_improvement_ci95'])
        ax.plot(steps,v,'o-',color=color,label=label+' · sample B')
        ax.fill_between(steps,bounds[0],bounds[1],color=color,alpha=.12)
        ax.plot(steps,a[key+'_mean_improvement'],'--',color=color,alpha=.6)
    ax.set_title('A · Does the average advantage fade?'); ax.set_ylabel('Improvement in transfer gap (BPB)')
    ax.legend(frameon=False,fontsize=7)
    ax=axes[0,1]; lookup={c['id']:i for i,c in enumerate(report['tokenizers'])}
    for id_,color in zip(report['retrospective_top10_at_8k'][:5],colors):
        i=lookup[id_]; c=report['tokenizers'][i]['config']
        label=f'#{int(id_.split("-")[-1])} · {c["algorithm"]} · {c["vocab_size"]//1000}k'
        ax.plot(steps,np.array(b['improvement_bpb'])[:,i],'o-',color=color,label=label)
        ax.plot(steps,np.array(a['improvement_bpb'])[:,i],'--',color=color,alpha=.55)
    ax.set_title('B · Five largest 8k winners, followed backward');ax.set_ylabel('Improvement in transfer gap (BPB)')
    ax.legend(frameon=False,fontsize=6.8)
    ax=axes[1,0]
    for name,color in [('wordpiece','#8657BC'),('bpe','#2F8FD0'),('unigram','#36967B')]:
        for pre,style in [('no-regex','-'),('regex','--')]:
            group=b['groups'].get(name+'/'+pre)
            if group: ax.plot(steps,group['mean_improvement'],marker='o',ls=style,color=color,label=name+' / '+pre)
    ax.set_title('C · Which tokenizer settings benefit? · sample B'); ax.set_ylabel('Mean improvement in transfer gap (BPB)')
    ax.legend(frameon=False,fontsize=6.5,ncol=2)
    ax=axes[1,1]
    for ni,label,color in [(0,'Uniform','#2F8FD0'),(1,'Predictor','#36967B')]:
        ax.plot(steps,np.array(b['pool_mean_absolute'])[:,ni],'o-',color=color,label=label+' · transferred')
        ax.plot(steps,np.array(b['canonical_bpb'])[:,ni],'--',color=color,label=label+' · canonical')
    ax.set_title('D · Actual LM loss · random 50 · sample B'); ax.set_ylabel('Absolute BPB (lower is better)')
    ax.legend(frameon=False,fontsize=7)
    for ax in axes.flat:
        ax.set_xticks(steps,[f'{s}k' for s in [2,4,6,8]]);ax.set_xlabel('Training step');ax.grid(alpha=.12)
    for ax in axes.flat[:3]: ax.axhline(0,color='#67717D',lw=.8)
    fig.suptitle('Predictor versus uniform · the same tokenizers and text at every checkpoint',fontsize=12)
    fig.supxlabel('Positive improvement favors Predictor. A/B: solid = text sample B; dashed = sample A. Bands in A: paired 95% intervals.',fontsize=8)
    savefig(fig,output/'predictor_trajectory')
    pool=[i for i,c in enumerate(report['tokenizers']) if c['kind']=='pool']
    held=[i for i,c in enumerate(report['tokenizers']) if c['kind']!='pool']
    order=pool+held
    matrices=[np.array(report['splits'][s]['improvement_bpb'])[:,order].T for s in ['screen','confirm']]
    vmax=max(np.abs(v).max() for v in matrices)
    fig,axes=plt.subplots(1,2,figsize=(10,12),layout='constrained')
    for ax,m,sample in zip(axes,matrices,['A','B']):
        image=ax.imshow(m,aspect='auto',cmap='RdBu',vmin=-vmax,vmax=vmax)
        ax.set_yticks(range(62),[report['tokenizers'][i]['id'] for i in order],fontsize=6.5)
        ax.set_xticks(range(4),['2k','4k','6k','8k']);ax.axhline(49.5,color='#333333',lw=1)
        ax.set_title('Text sample '+sample)
    fig.colorbar(image,ax=axes,label='Uniform gap − Predictor gap (BPB); blue favors Predictor',shrink=.7)
    fig.suptitle('All 62 tokenizers · predictor advantage throughout training',fontsize=12)
    savefig(fig,output/'predictor_trajectory_tokenizers')


def write_report(report,data):
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'analysis.json',report);atomic_json(REPORT/'plan.json',data['plan'])
    shutil.copy2(STATE/'results.json.gz',REPORT/'results.json.gz')
    with (REPORT/'tokenizer-scores.csv').open('w') as stream:
        fields=['sample','step','tokenizer','kind','uniform_canonical_bpb','predictor_canonical_bpb',
            'uniform_bpb','predictor_bpb','uniform_gap','predictor_gap','gap_improvement','absolute_improvement']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for split,v in report['splits'].items():
            for si,step in enumerate(STEPS):
                for i,c in enumerate(report['tokenizers']):
                    writer.writerow(dict(sample='A' if split=='screen' else 'B',step=step,tokenizer=c['id'],kind=c['kind'],
                        uniform_canonical_bpb=v['canonical_bpb'][si][0],predictor_canonical_bpb=v['canonical_bpb'][si][1],
                        uniform_bpb=v['absolute_bpb'][si][0][i],predictor_bpb=v['absolute_bpb'][si][1][i],
                        uniform_gap=v['gap_bpb'][si][0][i],predictor_gap=v['gap_bpb'][si][1][i],
                        gap_improvement=v['improvement_bpb'][si][i],absolute_improvement=v['absolute_improvement_bpb'][si][i]))
    lines=['# Does the predictor help more early in training?','',
        'Both actual training runs are evaluated at 2k, 4k, 6k and 8k on exactly the same 62 tokenizers and 576 broad-text passages. There are 50 randomly drawn training-pool tokenizers and 12 held-out tokenizers. No tokenizer or passage is dropped. The original 8k results are reused unchanged.','',
        '**Improvement = Uniform transfer gap − Predictor transfer gap. Positive favors Predictor.** Each transfer gap subtracts that checkpoint’s own canonical score. Absolute BPB and canonical BPB are saved separately, so changes in the LM are visible.','',
        'Scoring preserves the four-way evaluation: BOS + unscored prefix + 32 scored canonical continuation tokens, within 128 positions. Only prefix embeddings are transferred. Every pair scores 18,432 tokens and 53,707 bytes. Scores are byte-weighted within each of nine domains, then averaged equally over domains.','',
        '## Average improvement','',
        '| Step | Pool · sample A | Pool · sample B | Held-out · sample A | Held-out · sample B |','| --- | ---: | ---: | ---: | ---: |']
    a=report['splits']['screen'];b=report['splits']['confirm']
    for i,step in enumerate(STEPS):
        lines.append(f'| {step:,} | {a["pool_mean_improvement"][i]:+.5f} | {b["pool_mean_improvement"][i]:+.5f} | {a["heldout_mean_improvement"][i]:+.5f} | {b["heldout_mean_improvement"][i]:+.5f} |')
    lines+=['','## Earlier advantage versus 8k','',
        'These paired intervals compare each earlier advantage with the 8k advantage, using the same resampled documents and tokenizer IDs at both steps. Positive means the advantage was larger earlier. They cover evaluation sampling uncertainty, not retraining uncertainty.','',
        '| Step | Pool difference · sample B | 95% interval |','| --- | ---: | --- |']
    for i,step in enumerate(STEPS[:-1]):
        lo,hi=np.array(b['pool_gain_minus_8k_ci95'])[:,i]
        lines.append(f'| {step:,} versus 8k | {b["pool_gain_minus_8k"][i]:+.5f} | [{lo:+.5f}, {hi:+.5f}] |')
    lines+=['','![Matched checkpoint comparison](predictor_trajectory.png)','',
        '[Every tokenizer at every checkpoint](predictor_trajectory_tokenizers.png) · [Full scores](tokenizer-scores.csv)','',
        '## Interpretation limits','',
        'The five highlighted tokenizers are selected retrospectively for their largest improvements at 8k on sample B. Their earlier trajectories are descriptive; sample B is not an independent confirmation after selecting those winners. All 62 scores and both text samples are retained.','',
        'This compares the resulting models from one training seed per method. Uniform used the older training boundaries, while Predictor used corrected paired boundaries. It therefore does not isolate the selector as the sole cause. The fixed 12 held-out BPE tokenizers do not represent all unseen tokenizer algorithms.','',
        '## Reproduction and persistence','',
        '`python scripts/report_predictor_trajectory.py` audits the complete saved loss bundle and reproduces the analysis, CSV and figures. The plan pins checkpoint hashes, parent panel and evaluator sources. Model weights, corpora and tokenizer files stay in cloud storage; local outputs contain only losses and provenance. No training was resumed.','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    plots(report,REPORT)
    for p in REPORT.glob('*.png'):shutil.copy2(p,ROOT/'plots'/p.name)
    for p in REPORT.glob('*.pdf'):shutil.copy2(p,ROOT/'plots'/p.name)
    atomic_json(REPORT/'manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p))
        for p in REPORT.iterdir() if p.is_file() and p.name!='manifest.json'}))


def main():
    with gzip.open(STATE/'results.json.gz','rt') as stream:data=json.load(stream)
    report=analyze(data);write_report(report,data)
    print(json.dumps(dict(audit=report['audit'],sample_b_pool_improvement=report['splits']['confirm']['pool_mean_improvement'])),flush=True)


if __name__=='__main__':main()
