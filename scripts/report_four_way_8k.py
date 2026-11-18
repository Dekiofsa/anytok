"""Audit every pair, estimate paired uncertainty, and plot the four-way 8k comparison."""
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
from uniform_trajectory import PROFILE_DOMAINS

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/four_way_8k'
REPORT=ROOT/'reports/four-way-8k'
MODELS=['uniform','raw_bpb','regret','predictor']
LABELS=['Uniform','Raw-BPB RL','Regret RL','Predictor']
COLORS=['#2F8FD0','#E78042','#8657BC','#36967B']
DOMAIN_LABELS={'web-en':'English web','code':'Code','math-en':'Math','medical-en':'Medical',
    'legal-en':'Legal','web-zh':'Chinese','web-ja':'Japanese','web-ar':'Arabic','web-hi':'Hindi'}


def interval(values):
    return np.quantile(values,[.025,.975],axis=0).tolist()


def audit(data):
    plan=data['plan'];rows=data['rows'];key=dict(plan_sha256=digest(plan),panel_sha256=plan['panel_sha256'],step=8000)
    if data['preparation']['plan_sha256']!=digest(plan) or len(rows)!=576:
        raise ValueError('Frozen evaluation specification changed')
    proof=[[r['id'],r['prefix_sha256'],r['canonical_continuation_ids'],r['continuation_bytes']] for r in rows]
    if digest(proof)!=plan['panel_audit_sha256']:
        raise ValueError('Scored document IDs, prefixes or targets changed')
    candidates=plan['candidates'];ids=[c['id'] for c in candidates]
    if len(ids)!=62 or len(set(ids))!=62 or sum(c['kind']=='pool' for c in candidates)!=50:
        raise ValueError('All 62 original tokenizers required')
    if set(data['models'])!=set(MODELS):
        raise ValueError('All four checkpoints required')
    for split,byte_count in [('screen',27039),('confirm',26668)]:
        sub=[r for r in rows if r['split']==split]
        if len(sub)!=288 or sum(r['continuation_bytes'] for r in sub)!=byte_count:
            raise ValueError('Scored split counts changed')
        if any(sum(r['domain']==d for r in sub)!=32 for d in PROFILE_DOMAINS):
            raise ValueError('Missing domain coverage')
    for r in rows:
        if (not r['canonical_joint_encoding_matches'] or r['maximum_positions']>128
            or len(r['canonical_continuation_ids'])!=32
            or set(r['native_prefix_lengths'])!=set(ids)):
            raise ValueError('Invalid shared scoring proof')
    bases=[];transferred=[]
    def losses(record,field):
        if any(record.get(k)!=v for k,v in key.items()):
            raise ValueError('Results from another plan or checkpoint')
        values=np.asarray(record[field],dtype=float)
        if values.shape!=(576,32) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Missing or invalid per-token losses')
        return values.sum(axis=-1)
    for name in MODELS:
        model=data['models'][name]
        if model['canonical']['model']!=name or len(model['tokenizers'])!=62:
            raise ValueError('Wrong model or missing tokenizer result')
        bases.append(losses(model['canonical'],'canonical'))
        per=[]
        for c,record in zip(candidates,model['tokenizers']):
            if (record['model']!=name or record['tokenizer']!=c['id']
                or record['tokenizer_sha256']!=c['sha256']):
                raise ValueError('Tokenizer provenance mismatch')
            per.append(losses(record,'transferred'))
        transferred.append(per)
    return np.asarray(bases),np.asarray(transferred)


def group_members(candidates):
    result={}
    for algorithm in ['bpe','wordpiece','unigram']:
        for pretokenizer in ['regex','line256','no-regex']:
            result[algorithm+'/'+pretokenizer]=[i for i,c in enumerate(candidates)
                if c['kind']=='pool' and c['config']['algorithm']==algorithm
                and c['config']['pretokenizer']==pretokenizer]
    for label,field,bins in [('vocab','vocab_size',[(32000,64000),(64001,96000),(96001,128000)]),
                              ('noise','merge_noise',[(0,.25),(.2500000001,.5),(.5000000001,.75)])]:
        for lo,hi in bins:
            indices=[]
            for i,c in enumerate(candidates):
                if c['kind']!='pool':continue
                value=(c['config'].get('noise_per_million',0)/1_000_000
                       if field=='merge_noise' else c['config'].get(field))
                if value is not None and lo<=value<=hi:indices.append(i)
            result[f'{label} {lo:g}–{hi:g}']=indices
    return {k:v for k,v in result.items() if v}


def analyze(data,replicates=2000):
    base,native=audit(data);rows=data['rows'];candidates=data['plan']['candidates']
    pool=np.array([i for i,c in enumerate(candidates) if c['kind']=='pool'])
    held=np.array([i for i,c in enumerate(candidates) if c['kind']!='pool'])
    groups=group_members(candidates)
    rng=np.random.default_rng(20261118)
    tokenizer_draws=rng.integers(50,size=(replicates,50))
    fpc=math.sqrt((1000-50)/(1000-1))
    report=dict(step=8000,models=MODELS,labels=LABELS,tokenizers=candidates,splits={},
        audit=dict(comparisons=248,passages=576,continuation_tokens_per_pair=18432,
            continuation_bytes_per_pair=53707,all_pairs_verified=True,plan_sha256=digest(data['plan'])),
        uncertainty='95% paired document-cluster bootstrap within each domain. Full-pool mean intervals additionally resample the random 50 tokenizer IDs with finite-population correction. One training seed per method.',
        limitations='Actual-run comparison, not a selector-only ablation: Uniform and Raw-BPB used older training cuts; Raw-BPB entropy is 0.01, Regret RL entropy is 0. Held-out tokenizers are a fixed BPE panel, not a random population sample.')
    for split in ['screen','confirm']:
        positions=[i for i,r in enumerate(rows) if r['split']==split];sub=[rows[i] for i in positions]
        values=np.concatenate([base[:,positions],native[:,:,positions].reshape(248,len(positions))])
        point,boot=bootstrap_balanced(sub,values,replicates,domains=PROFILE_DOMAINS)
        canonical=point[:4];absolute=point[4:].reshape(4,62)
        gaps=absolute-canonical[:,None]
        canonical_boot=boot[:,:4];absolute_boot=boot[:,4:].reshape(replicates,4,62)
        gap_boot=absolute_boot-canonical_boot[:,:,None]
        pool_boot=gap_boot[:,:,pool]
        # Separate text uncertainty from centered finite-pool tokenizer uncertainty.
        sampled=pool_boot[np.arange(replicates)[:,None,None],np.arange(4)[None,:,None],tokenizer_draws[:,None,:]].mean(2)
        total_pool_boot=pool_boot.mean(2)+fpc*(sampled-pool_boot.mean(2))
        info=dict(canonical_bpb=canonical.tolist(),absolute_bpb=absolute.tolist(),gap_bpb=gaps.tolist(),
            per_tokenizer_gap_ci95=interval(gap_boot),canonical_ci95=interval(canonical_boot),
            pool_mean_gap=gaps[:,pool].mean(1).tolist(),pool_mean_gap_ci95=interval(total_pool_boot),
            pool_mean_delta_vs_uniform=(gaps[:,pool].mean(1)-gaps[0,pool].mean()).tolist(),
            pool_mean_delta_vs_uniform_ci95=interval(total_pool_boot-total_pool_boot[:,:1]),
            pool_median_gap=np.median(gaps[:,pool],axis=1).tolist(),
            pool_p90_gap=np.quantile(gaps[:,pool],.9,axis=1).tolist(),
            pool_worst_gap=gaps[:,pool].max(1).tolist(),
            pool_worst_tokenizer=[candidates[pool[i]]['id'] for i in gaps[:,pool].argmax(1)],
            pool_mean_absolute=absolute[:,pool].mean(1).tolist(),
            heldout_mean_gap=gaps[:,held].mean(1).tolist(),
            heldout_mean_gap_ci95=interval(gap_boot[:,:,held].mean(2)),
            heldout_mean_delta_vs_uniform_ci95=interval((gap_boot-gap_boot[:,:1])[:,:,held].mean(2)),
            heldout_mean_absolute=absolute[:,held].mean(1).tolist(),
            groups={},domains={})
        for label,indices in groups.items():
            info['groups'][label]=dict(count=len(indices),mean_gap=gaps[:,indices].mean(1).tolist(),
                delta_vs_uniform=(gaps[:,indices]-gaps[:1,indices]).mean(1).tolist(),
                delta_ci95=interval((gap_boot-gap_boot[:,:1])[:,:,indices].mean(2)))
        for domain in PROFILE_DOMAINS:
            j=[i for i,r in enumerate(sub) if r['domain']==domain]
            denom=sum(sub[i]['continuation_bytes'] for i in j)*math.log(2)
            b=base[:,np.asarray(positions)[j]].sum(1)/denom
            t=native[:,:,np.asarray(positions)[j]].sum(2)/denom
            info['domains'][domain]=dict(canonical_bpb=b.tolist(),
                pool_mean_gap=(t[:,pool]-b[:,None]).mean(1).tolist(),
                heldout_mean_gap=(t[:,held]-b[:,None]).mean(1).tolist(),
                pool_mean_absolute=t[:,pool].mean(1).tolist())
        report['splits'][split]=info
    return report


def plotting_style():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=ROOT/'monitoring/web/Manrope.ttf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Manrope','font.size':8.5,
        'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'axes.edgecolor':'#555B65','axes.labelcolor':'#3F4957',
        'xtick.color':'#3F4957','ytick.color':'#3F4957','text.color':'#3F4957',
        'figure.facecolor':'white','savefig.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})


def savefig(fig,path):
    import matplotlib.pyplot as plt
    for ext in ['png','pdf']:
        fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot(report,output):
    import matplotlib.pyplot as plt
    plotting_style();candidates=report['tokenizers']
    pool=[i for i,c in enumerate(candidates) if c['kind']=='pool']
    held=[i for i,c in enumerate(candidates) if c['kind']!='pool']
    for split in ['screen','confirm']:
        info=report['splits'][split];gaps=np.asarray(info['gap_bpb']);absolute=np.asarray(info['absolute_bpb'])
        sample='A' if split=='screen' else 'B'
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        ax=axes[0,0]
        for k,(label,color) in enumerate(zip(LABELS,COLORS)):
            jitter=np.linspace(-.18,.18,50)
            ax.scatter(k+jitter,gaps[k,pool],s=13,alpha=.55,color=color)
            ax.scatter(k,info['pool_mean_gap'][k],marker='D',s=45,color=color,edgecolor='white',zorder=5)
            ax.plot([k-.22,k+.22],[info['pool_p90_gap'][k]]*2,color=color,lw=2)
            ci=np.asarray(info['pool_mean_gap_ci95'])[:,k];v=info['pool_mean_gap'][k]
            ax.errorbar(k,v,yerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color='#3F4957',capsize=3,zorder=6)
        ax.set_xticks(range(4),LABELS);ax.set(title='A · Training-pool tokenizers: typical and difficult cases',ylabel='Transfer gap · BPB (lower is better)')
        ax.text(.01,.98,'50 dots per model · diamond = mean · bar = 90th percentile',transform=ax.transAxes,va='top',fontsize=7,color='#67717D')
        ax=axes[0,1]
        image=ax.imshow(gaps[:,held].T,aspect='auto',cmap='YlOrRd',vmin=min(0,float(gaps[:,held].min())),vmax=float(gaps[:,held].max()))
        ax.set_xticks(range(4),LABELS);ax.set_yticks(range(12),[candidates[i]['id'].removeprefix('ood48-') for i in held])
        ax.set_title('B · Held-out tokenizer files · transfer gap')
        for i in range(12):
            for k in range(4):
                value=gaps[k,held[i]]
                ax.text(k,i,f'{value:.3f}',ha='center',va='center',fontsize=7,color='white' if image.norm(value)>.65 else '#333333')
        fig.colorbar(image,ax=ax,pad=.02,label='BPB')
        ax=axes[1,0]
        groups=[(label,r) for label,r in info['groups'].items() if '/' in label]
        matrix=np.array([r['delta_vs_uniform'][1:] for label,r in groups])
        limit=max(.001,float(np.abs(matrix).max()))
        image=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks(range(3),LABELS[1:]);ax.set_yticks(range(len(groups)),[f'{label} (n={r["count"]})' for label,r in groups],fontsize=7)
        ax.set_title('C · Which tokenizer groups improved?')
        fig.colorbar(image,ax=ax,pad=.02,label='Gap change vs Uniform · negative = improvement')
        ax=axes[1,1]
        for k,color in enumerate(COLORS):
            ax.plot([k,k],[info['canonical_bpb'][k],info['pool_mean_absolute'][k]],color=color,alpha=.35,lw=3)
            ax.scatter(k,info['pool_mean_absolute'][k],color=color,s=45,marker='o')
            ax.scatter(k,info['heldout_mean_absolute'][k],color=color,s=45,marker='s')
            ax.scatter(k,info['canonical_bpb'][k],color=color,s=60,marker='_',linewidths=3)
        ax.scatter([],[],marker='o',color='#67717D',label='Mean of random 50')
        ax.scatter([],[],marker='s',color='#67717D',label='Mean of 12 held-out')
        ax.scatter([],[],marker='_',color='#67717D',label='Canonical LM')
        ax.set_xticks(range(4),LABELS);ax.set(title='D · Actual loss, including LM strength',ylabel='Absolute BPB (lower is better)');ax.legend(frameon=False,fontsize=7)
        fig.suptitle(f'Four training methods at 8k · broad text sample {sample}',fontsize=13)
        fig.get_layout_engine().set(h_pad=.15,w_pad=.13)
        name='four_way_8k' if split=='confirm' else 'four_way_8k_sample_a'
        savefig(fig,output/name)
    # Full per-tokenizer and per-domain views retain both text samples.
    for kind in ['tokenizers','domains']:
        if kind=='tokenizers':
            order=pool+held;labels=[candidates[i]['id'] for i in order]
            matrices=[np.asarray(report['splits'][s]['gap_bpb'])[:,order].T for s in ['screen','confirm']]
            title='Every tokenizer · transfer gap (BPB)';size=(10,13)
        else:
            labels=[DOMAIN_LABELS[d] for d in PROFILE_DOMAINS]
            matrices=[np.asarray([report['splits'][s]['domains'][d]['pool_mean_gap'] for d in PROFILE_DOMAINS]) for s in ['screen','confirm']]
            title='Text domains · mean training-pool transfer gap (BPB)';size=(10,4.5)
        vmax=max(float(m.max()) for m in matrices);vmin=min(0,min(float(m.min()) for m in matrices))
        fig,axes=plt.subplots(1,2,figsize=size,layout='constrained')
        for ax,matrix,sample in zip(axes,matrices,['A','B']):
            image=ax.imshow(matrix,aspect='auto',cmap='YlOrRd',vmin=vmin,vmax=vmax)
            ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.set_xticks(range(4),LABELS,rotation=20)
            ax.set_title('Text sample '+sample)
            if kind=='tokenizers':ax.axhline(49.5,color='#333333',lw=1)
        fig.colorbar(image,ax=axes,label='BPB',shrink=.7);fig.suptitle(title,fontsize=12)
        savefig(fig,output/('four_way_8k_'+kind))


def write_report(report,data):
    REPORT.mkdir(parents=True,exist_ok=True);atomic_json(REPORT/'analysis.json',report)
    atomic_json(REPORT/'plan.json',data['plan'])
    target=REPORT/'results.json.gz'
    if not target.exists() or sha256(target)!=sha256(STATE/'results.json.gz'):
        shutil.copy2(STATE/'results.json.gz',target)
    with (REPORT/'tokenizer-scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['sample','model','tokenizer','kind','canonical_bpb','transferred_bpb','gap_bpb'])
        writer.writeheader()
        for split,info in report['splits'].items():
            for k,name in enumerate(MODELS):
                for i,c in enumerate(report['tokenizers']):
                    writer.writerow(dict(sample='A' if split=='screen' else 'B',model=name,tokenizer=c['id'],kind=c['kind'],
                        canonical_bpb=info['canonical_bpb'][k],transferred_bpb=info['absolute_bpb'][k][i],gap_bpb=info['gap_bpb'][k][i]))
    lines=['# Four training methods at step 8,000','',
        'Every checkpoint is frozen. All four models score the same 50 randomly sampled training-pool tokenizer files and 12 held-out tokenizer files on the same 576 broad-text passages. No passages were selected for having high model loss.','',
        '**Scoring:** BOS + unscored prefix + 32 scored canonical continuation tokens, within 128 total positions. Only prefix embeddings are transferred. The canonical continuation embeddings and output head remain fixed. Every tokenizer uses the same prefix text and target bytes.','',
        '**Gap = transferred BPB − that checkpoint’s canonical BPB. Lower is better.** Absolute BPB is reported separately because a smaller transfer gap can accompany a weaker LM. Scores are byte-weighted within each domain, then averaged equally over nine domains.','',
        '## Results','']
    for split,info in report['splits'].items():
        sample='A' if split=='screen' else 'B'
        lines += [f'### Text sample {sample} · 288 passages','',
            '| Model | Canonical BPB | Random 50 mean BPB | Random 50 mean gap | 90th-percentile gap | Held-out mean gap |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for k,label in enumerate(LABELS):
            lines.append(f'| {label} | {info["canonical_bpb"][k]:.4f} | {info["pool_mean_absolute"][k]:.4f} | {info["pool_mean_gap"][k]:+.4f} | {info["pool_p90_gap"][k]:+.4f} | {info["heldout_mean_gap"][k]:+.4f} |')
        lines += ['']
    lines += ['![Four-way comparison on text sample B](four_way_8k.png)','',
        '[Text sample A](four_way_8k_sample_a.png) · [Every tokenizer](four_way_8k_tokenizers.png) · [Text domains](four_way_8k_domains.png)','',
        '## What this comparison can establish','',
        'Look for lower absolute transferred BPB, a lower average transfer gap, and improvements in the difficult tokenizer tail. Inspect domain and tokenizer groups to see whether improvements are broad or whether weaknesses moved elsewhere. The random 50 estimate typical performance in the original 1,000-tokenizer pool; they cannot establish its absolute worst tokenizer. The 12 held-out cases are a fixed BPE panel, not a random sample of all unseen tokenizer algorithms.','',
        '**This is not a selector-only ablation.** Uniform and Raw-BPB were trained using the older cuts. Raw-BPB used entropy weight 0.01, while Regret RL used 0. Both newer methods use corrected paired training boundaries. All four use the corrected shared evaluation boundaries here. There is one training seed per method.','',
        'The nine domains are English web, code, math, medical, legal, Chinese, Japanese, Arabic and Hindi. Text samples A and B are the original screen and confirmation documents; neither is a loss-selected stress subset. Domain-specific results are retained so that the aggregate does not hide where changes occur.','',
        '95% intervals use paired source-document-cluster bootstrap resampling within domains. Intervals for the random-50 population mean additionally resample tokenizer IDs with finite-population correction. They do not cover retraining uncertainty or guarantee a population-wide tail estimate. See analysis.json for intervals, all group sizes, domain scores and paired changes.','',
        '## Reproduce','',
        '`python scripts/report_four_way_8k.py` reproduces the audit, analysis and figures from `modal_results/four_way_8k/results.json.gz`. The same loss data are included here as results.json.gz. The frozen plan pins checkpoints, native files, complete vocabulary features, runtime and evaluator source hashes. Large model weights, tokenizer files and text remain in cloud storage. Training was not resumed.','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    plot(report,REPORT)
    (ROOT/'plots').mkdir(exist_ok=True)
    for p in REPORT.glob('*.png'):shutil.copy2(p,ROOT/'plots'/p.name)
    for p in REPORT.glob('*.pdf'):shutil.copy2(p,ROOT/'plots'/p.name)
    atomic_json(REPORT/'manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p))
        for p in REPORT.iterdir() if p.is_file() and p.name!='manifest.json'}))


def main():
    data=json.loads(gzip.decompress((STATE/'results.json.gz').read_bytes()))
    report=analyze(data);write_report(report,data)
    print(json.dumps(dict(stage='report_complete',path=str(REPORT),
        sample_b_mean_gaps=report['splits']['confirm']['pool_mean_gap'])),flush=True)


if __name__=='__main__':main()
