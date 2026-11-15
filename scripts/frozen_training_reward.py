"""Actual paired training-reward forwards, with all LM/HN parameters frozen."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

from package_training_inputs import sha256
from run_raw_bpb_training import prepare_code

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/hardness_predictor_6500'
INPUT = ROOT / 'modal_results/local_memory/artifacts/input'
REFERENCE = ROOT / 'reports/regret-pool-ranking-6500'


class FrozenTrainingReward:
    def __init__(self):
        import torch
        if torch.__version__.split('+')[0] != '2.15.0':
            raise ValueError('Use the pinned torch 2.15.0 .venv-memory environment')
        torch.set_num_threads(1)
        self.torch = torch
        receipt = json.loads((STATE/'artifacts.json').read_text())
        checkpoint = Path(receipt['checkpoint'])
        manifest = receipt['source']['manifest']
        plan = json.loads((REFERENCE/'panel-manifest.json').read_text())
        if manifest != plan['checkpoint_manifest']:
            raise ValueError('Model is not the original frozen step-6500 ranking checkpoint')
        for name in ['lm.pt','hypernet/config.json','hypernet/model.safetensors']:
            expected = manifest['files'][name]
            path = checkpoint/name
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError('Frozen checkpoint changed: '+name)
        code = prepare_code(STATE)
        original = json.loads((REFERENCE/'run-plan.json').read_text())
        if sha256(code/'source-manifest.json') != original['staged_source_manifest_sha256']:
            raise ValueError('Training helpers or memory implementation changed')
        sys.path[:0] = [str(code/'scripts'),str(code)]
        spec = importlib.util.spec_from_file_location('hardness_frozen_trainer',code/'scripts/train_joint_smollm_input.py')
        trainer = importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
        self.trainer = trainer
        self.canonical = trainer.AutoTokenizer.from_pretrained(INPUT/'model',local_files_only=True)
        self.pool = trainer.TokenizerPool(INPUT/'pool')
        if self.pool.index_hash != plan['pool_index_sha256'] or self.pool.plan_hash != plan['pool_plan_sha256']:
            raise ValueError('The complete original pool changed')
        self.lm = trainer.AutoModelForCausalLM.from_pretrained(INPUT/'model',local_files_only=True,
            dtype=torch.float32,low_cpu_mem_usage=True).to('cuda')
        trainer.separate_input_embeddings_and_freeze_output(self.lm)
        source_original = self.lm.get_input_embeddings().weight.detach().clone()
        self.lm.load_state_dict(torch.load(checkpoint/'lm.pt',map_location='cpu',weights_only=True,mmap=True))
        self.source = self.lm.get_input_embeddings().weight.detach()
        if (not torch.equal(source_original,self.source)
                or not torch.equal(source_original,self.lm.get_output_embeddings().weight)):
            raise ValueError('Canonical embedding/output tables changed')
        del source_original
        self.hn = trainer.ZettHypernet.from_pretrained(checkpoint/'hypernet',local_files_only=True).to('cuda')
        self.fallback = trainer.enable_byte_fallbacks(self.hn,self.canonical)
        trainer.freeze_rescalers(self.hn)
        from pool_training_memory import apply_gpu_memory_fix
        apply_gpu_memory_fix(self.hn)
        self.lm.requires_grad_(False);self.hn.requires_grad_(False)
        self.lm.train();self.hn.train()
        self.versions = self.parameter_versions()
        self.calls = 0

    def parameter_versions(self):
        return {prefix+name:p._version for prefix,model in [('lm.',self.lm),('hn.',self.hn)]
                for name,p in model.named_parameters()}

    def assert_frozen(self):
        if self.parameter_versions() != self.versions:
            raise ValueError('A frozen LM/hypernetwork parameter changed')
        if any(p.requires_grad or p.grad is not None for model in [self.lm,self.hn] for p in model.parameters()):
            raise ValueError('Frozen model has gradients or trainable parameters')

    def score(self,index,examples,controls,proofs,rejected,dropout_seed):
        import numpy as np
        torch = self.torch;trainer = self.trainer
        if (len(examples)!=128 or len(controls)!=128 or len(proofs)!=128
                or any(a[1:]!=b[1:] for a,b in zip(examples,controls))
                or any(proof['maximum_positions']>128 for proof in proofs)):
            raise ValueError('A complete matched 128-example training update is required')
        _,vocab = self.pool.load(index)
        forms_np,truncated = self.pool.forms(index,self.canonical,self.fallback)
        forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
        priors = torch.tensor([score for _,score in vocab],device='cuda',dtype=torch.float32)
        count = sum(len(example[1]) for example in examples)
        size = sum(example[2] for example in examples)
        torch.cuda.manual_seed(dropout_seed)
        with torch.inference_mode():
            base_nats = trainer.canonical_loss_nats(self.lm,controls,self.source,32,trainer.make_lm_microbatch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                generated,output,_ = self.hn(forms,priors,self.source,None,deterministic=False)
            if output is not None or not torch.isfinite(generated).all():
                raise ValueError('Invalid full native embedding table')
            leaf = generated.detach().float();nats = 0.
            for start in range(0,128,32):
                micro = examples[start:start+32]
                embeds,mask,labels,positions = trainer.make_lm_microbatch(micro,leaf,self.source,'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = self.lm(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,
                                   labels=labels,use_cache=False).loss
                nats += float(loss)*sum(len(example[1]) for example in micro)
                del loss,embeds,mask,labels,positions
            raw,base,gap = trainer.paired_gap_bpb(nats,base_nats,size)
            del leaf,generated
        self.calls += 1
        return dict(raw_bpb=raw,canonical_bpb=base,gap_bpb=gap,transferred_nats=nats,canonical_nats=base_nats,
            scored_tokens=count,scored_bytes=size,valid_examples=128,
            maximum_positions=max(proof['maximum_positions'] for proof in proofs),
            paired_batch_sha256=hashlib.sha256(json.dumps(proofs,sort_keys=True).encode()).hexdigest(),
            rejected_examples=rejected,dropout_seed=dropout_seed,vocabulary=len(vocab),truncated_forms=truncated,
            tokenizer_sha256=self.pool.entries[index]['tokenizer_sha256'])


class FreshRewardStream:
    def __init__(self,scorer,state=None):
        self.scorer = scorer
        self.rng = random.Random(42)
        self.corpus = scorer.trainer.BalancedTextStream(INPUT/'recipe.json',INPUT/'corpus',
                                                       state['corpus'] if state else None)
        if state:self.rng.setstate(state['python_rng'])
        # This reads raw window hashes only, never the ranking's reward labels.
        panel_path=ROOT/'modal_results/regret_favorite_trajectory/training-panel.json.gz'
        with gzip.open(panel_path,'rt') as stream:panel=json.load(stream)
        self.reserved = {item['raw_sha256'] for batch in panel for item in batch['items']}
        self.reserved.update(hashlib.sha256(raw.encode()).hexdigest() for batch in panel
            for bank in batch['replacements'].values() for raw in bank)
        self.reserved_skipped = state.get('reserved_skipped',0) if state else 0

    def raw(self,name):
        starting_epoch=self.corpus.epochs[name]
        while True:
            value = self.corpus.next_text(name,chars=2048)[:512]
            if hashlib.sha256(value.encode()).hexdigest() not in self.reserved:
                return value
            self.reserved_skipped += 1
            # A reference bank can reserve thousands of consecutive windows.
            # Advance through it completely; fail only after a whole corpus
            # cycle establishes that this bucket has no unreserved window.
            if self.corpus.epochs[name]>starting_epoch+1:
                raise ValueError('No unreserved reward window exists in '+name)

    def batch(self,index):
        names = self.rng.choices(self.corpus.names,weights=self.corpus.weights,k=128)
        raws = [self.raw(name) for name in names]
        native,_ = self.scorer.pool.load(index)
        examples=[];controls=[];proofs=[];rejected=0
        for name,raw in zip(names,raws):
            paired = self.scorer.trainer.make_paired_prefix_example(raw,native,self.scorer.canonical,min_chars=32)
            while paired is None and rejected<128:
                rejected+=1;raw=self.raw(name)
                paired=self.scorer.trainer.make_paired_prefix_example(raw,native,self.scorer.canonical,min_chars=32)
            if paired is None:
                raise ValueError('Original paired-example rejection limit exceeded')
            example,control,proof=paired
            examples.append(example);controls.append(control);proofs.append(proof)
        return examples,controls,proofs,rejected

    def state_dict(self):
        return dict(corpus=self.corpus.state_dict(),python_rng=self.rng.getstate(),reserved_skipped=self.reserved_skipped)

    def close(self):self.corpus.close()
