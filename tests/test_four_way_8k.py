import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from four_way_8k import common_panel


class Characters:
    def __call__(self,text,**kwargs):
        return SimpleNamespace(input_ids=[ord(c) for c in text])
    def encode(self,text,**kwargs):
        return SimpleNamespace(ids=[ord(c) for c in text])
    def decode(self,ids,**kwargs):
        return ''.join(map(chr,ids))


class ExpensivePrefix(Characters):
    def encode(self,text,**kwargs):
        return SimpleNamespace(ids=([1]*100 if text.startswith('!') else [ord(c) for c in text]))


def passage(prefix):
    return dict(id='source-1',split='screen',domain='web-en',prefix=prefix,
        canonical_continuation_ids=[ord('z')]*32,continuation_bytes=32)


def test_valid_prefix_and_targets_are_preserved_for_every_encoder():
    original=[passage('abcdefghijklmnop')]; saved=copy.deepcopy(original)
    result=common_panel(original,Characters(),{'one':Characters(),'two':Characters()}, {'one':set(),'two':set()})
    assert original==saved
    assert result[0]['prefix']==original[0]['prefix']
    assert result[0]['canonical_continuation_ids']==original[0]['canonical_continuation_ids']
    assert result[0]['additional_trim_chars']==0
    assert set(result[0]['native_prefix_lengths'])=={'one','two'}


def test_one_shared_trim_preserves_continuation_bytes_and_ids():
    original=[passage('!abcdefghijklmnop')]
    result=common_panel(original,Characters(),{'one':Characters(),'two':ExpensivePrefix()}, {'one':set(),'two':set()})
    assert result[0]['prefix']=='abcdefghijklmnop'
    assert result[0]['additional_trim_chars']==1
    for key in ['id','split','domain','canonical_continuation_ids','continuation_bytes']:
        assert result[0][key]==original[0][key]
    assert result[0]['native_prefix_lengths']=={'one':16,'two':16}


def test_unencodable_passage_fails_instead_of_being_removed():
    with pytest.raises(ValueError,match='no passage removed'):
        common_panel([passage('abcdefghijklmnop')],Characters(),{'one':Characters()}, {'one':{ord('a')}})


def test_boundary_disagreement_is_rejected():
    class InvalidBoundary(Characters):
        def __call__(self,text,**kwargs):
            ids=super().__call__(text).input_ids
            return SimpleNamespace(input_ids=[999]+ids if 'z' in text and len(text)>32 else ids)
    with pytest.raises(ValueError,match='no passage removed'):
        common_panel([passage('abcdefghijklmnop')],InvalidBoundary(),{'one':Characters()},{'one':set()})


@pytest.fixture
def scored_panel():
    import math
    from pool_generalization import digest
    from uniform_trajectory import PROFILE_DOMAINS
    from report_four_way_8k import MODELS
    candidates=[]
    for i in range(62):
        candidates.append(dict(id=f'case-{i}',kind='pool' if i<50 else 'saved-native',sha256=f'hash-{i}',
            config=dict(algorithm=['bpe','wordpiece','unigram'][i%3],
                pretokenizer=['regex','line256','no-regex'][(i//3)%3],
                vocab_size=32000+i*1000,noise_per_million=i*10000)))
    rows=[]
    for split,total in [('screen',27039),('confirm',26668)]:
        for i in range(288):
            rows.append(dict(id=f'{split}/{i}',cluster_id=f'{split}/document-{i//2}',split=split,
                domain=PROFILE_DOMAINS[i//32],continuation_bytes=total//288+(i<total%288),
                canonical_continuation_ids=[97]*32,prefix_sha256=f'prefix-{i}',
                maximum_positions=64,canonical_joint_encoding_matches=True,
                native_prefix_lengths={c['id']:31 for c in candidates}))
    plan=dict(candidates=candidates,panel_sha256='frozen-panel',
        panel_audit_sha256=digest([[r['id'],r['prefix_sha256'],r['canonical_continuation_ids'],r['continuation_bytes']] for r in rows]))
    key=dict(plan_sha256=digest(plan),panel_sha256=plan['panel_sha256'],step=8000)
    models={}
    def nats(bpb):return [[bpb*r['continuation_bytes']*math.log(2)/32]*32 for r in rows]
    for k,name in enumerate(MODELS):
        canonical=1+k*.1
        models[name]=dict(canonical=dict(**key,model=name,canonical=nats(canonical)),tokenizers=[])
        for i,c in enumerate(candidates):
            models[name]['tokenizers'].append(dict(**key,model=name,tokenizer=c['id'],
                tokenizer_sha256=c['sha256'],transferred=nats(canonical+.05-k*.01+i*.001)))
    return dict(plan=plan,rows=rows,models=models,preparation=dict(plan_sha256=digest(plan)))


def test_analysis_separates_lm_strength_from_transfer_and_keeps_pairs(scored_panel):
    import numpy as np
    from report_four_way_8k import analyze
    report=analyze(scored_panel,replicates=16)
    for split in ['screen','confirm']:
        info=report['splits'][split]
        np.testing.assert_allclose(info['canonical_bpb'],[1,1.1,1.2,1.3],atol=1e-12)
        np.testing.assert_allclose(info['pool_mean_gap'],[.0745,.0645,.0545,.0445],atol=1e-12)
        np.testing.assert_allclose(info['pool_mean_delta_vs_uniform'],[0,-.01,-.02,-.03],atol=1e-12)
        ci=np.asarray(info['pool_mean_delta_vs_uniform_ci95'])
        np.testing.assert_allclose(ci,[[0,-.01,-.02,-.03]]*2,atol=1e-12)
        assert info['pool_mean_absolute'][3]>info['pool_mean_absolute'][0]
        assert sum(r['count'] for k,r in info['groups'].items() if '/' in k)==50
        assert sum(r['count'] for k,r in info['groups'].items() if k.startswith('noise '))==50


def test_analysis_refuses_incomplete_or_wrong_tokenizer_scores(scored_panel):
    from report_four_way_8k import audit
    scored_panel['models']['predictor']['tokenizers'][17]['tokenizer_sha256']='wrong'
    with pytest.raises(ValueError,match='Tokenizer provenance'):
        audit(scored_panel)


def test_monitor_accepts_actual_workers_and_progress(tmp_path,monkeypatch):
    import json
    import run_four_way_8k
    monkeypatch.setattr(run_four_way_8k,'STATE',tmp_path)
    workers=[dict(id='predictor',state='scoring',completed=12,total=62)]
    run_four_way_8k.status('scoring',12,workers=workers)
    saved=json.loads((tmp_path/'status.json').read_text())
    assert saved['workers']==workers
    assert saved['completed']==12 and saved['total']==248
    assert json.loads((tmp_path/'events.jsonl').read_text())==saved
