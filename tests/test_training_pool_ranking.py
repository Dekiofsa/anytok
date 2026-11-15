"""Training-style reconstruction, replacements and batch-reward aggregation."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from tokenizers import Tokenizer, models, decoders
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from score_training_pool_ranking import pair_batch
from report_training_pool_ranking import analyze

spec = importlib.util.spec_from_file_location('pairing', ROOT / 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py')
pairing = importlib.util.module_from_spec(spec);spec.loader.exec_module(pairing)


def test_rejected_items_consume_distinct_replacements_without_dropping_examples():
    vocab = {'<eos>':0, 'a':1, 'b':2, 'c':3, ' ':4, 'ab':5}
    native = Tokenizer(models.BPE(vocab=vocab,merges=[]));native.decoder=decoders.Fuse()
    canonical = PreTrainedTokenizerFast(tokenizer_object=native,bos_token='<eos>',eos_token='<eos>')
    texts=['a','b']+['ab c '*20]*126
    replacements=['a b '*20,'b c '*20]+['ab c '*20]*126
    batch=dict(items=[dict(text=text,raw_sha256=hashlib.sha256(text.encode()).hexdigest(),domain='web-en') for text in texts],
               replacements={'web-en':replacements})
    examples, controls, proofs, rejected=pair_batch(batch,native,canonical,pairing.make_paired_prefix_example)
    assert len(examples)==len(controls)==len(proofs)==128 and rejected==2
    for index in [0,1]:
        reconstructed=canonical.decode(controls[index][0]+controls[index][1],clean_up_tokenization_spaces=False)
        assert reconstructed==replacements[index][:proofs[index]['window_chars']]
    assert all(a[1:]==b[1:] for a,b in zip(examples,controls))


def records():
    probabilities=np.zeros(1000);probabilities[0]=1
    rows=[]
    for index in range(1000):
        batches=[]
        for batch in range(32):
            gap=index/1000+(batch%2)*.01
            batches.append(dict(index=batch,valid_examples=128,maximum_positions=128,scored_bytes=100 if batch%2==0 else 10000,
                dropout_seed=2026111500+batch,split='screen' if batch<16 else 'confirm',
                raw_bpb=1+gap,canonical_bpb=1,gap_bpb=gap,rejected_examples=0))
        rows.append(dict(index=index,checkpoint_step=6500,panel_sha256='fixed',policy_probability=float(probabilities[index]),
            batches=batches,config=dict(algorithm='bpe',pretokenizer='regex'),vocabulary=32000))
    return rows,dict(probabilities=probabilities.tolist()),dict(panel_sha256='fixed',aggregation='mean batch regret')


def test_ranking_matches_mean_update_reward_not_pooled_continuation_bytes():
    result, table, gaps=analyze(*records())
    assert result['comparisons']['all']['dominant_rank']==1000
    assert result['comparisons']['all']['policy_expected_gap']==pytest.approx(.005)
    assert result['confirmation']['discovery_winner']==999
    assert result['confirmation']['mean_gap_difference']==pytest.approx(.999)
    assert result['confirmation']['paired_batch_bootstrap_95pct']==pytest.approx([.999,.999])
    assert table[0]['index']==999 and gaps.shape==(1000,32)


def test_incomplete_scoring_and_mismatched_targets_cannot_be_reported():
    rows,policy,manifest=records()
    with pytest.raises(ValueError):analyze(rows[:-1],policy,manifest)
    rows[7]['batches'][3]['valid_examples']=127
    with pytest.raises(ValueError):analyze(rows,policy,manifest)
