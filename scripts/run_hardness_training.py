"""Stage the approved fresh joint run; preserve the existing scorer and memory path."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from package_training_inputs import atomic_json,sha256
from run_raw_bpb_training import source_files,command as original_command

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tillicum_hardness'
INPUT=ROOT/'modal_results/local_memory/artifacts/input'


def prepared_sources():
    files={name:path.read_bytes() for name,path in source_files().items()}
    trainer=files['scripts/train_joint_smollm_input.py'].decode()
    edits={
        'from hf_hypernet.settings_policy import SettingsTokenizerBandit':'from hf_hypernet.hardness_policy import SettingsTokenizerBandit',
        '"adversarial_shared_settings_mlp_per_optimizer_update"':'"regret_predictor_argmax_with_one_in_four_shuffled_coverage"',
        "(REPO / 'hf_hypernet/settings_policy.py').read_bytes()":"(REPO / 'hf_hypernet/hardness_policy.py').read_bytes()",
    }
    for before,after in edits.items():
        if trainer.count(before)!=1:raise ValueError('Unexpected original trainer source: '+before)
        trainer=trainer.replace(before,after)
    anchor='              "fixed_eval_sha256":'
    if trainer.count(anchor)!=1:raise ValueError('Missing source provenance insertion point')
    trainer=trainer.replace(anchor,'              "predictor_source_sha256": hashlib.sha256((REPO / "scripts/hardness_predictor.py").read_bytes()).hexdigest(),\n'+anchor)
    files['scripts/train_joint_smollm_input.py']=trainer.encode()
    files['scripts/hardness_predictor.py']=(ROOT/'scripts/hardness_predictor.py').read_bytes()
    files['hf_hypernet/hardness_policy.py']=(ROOT/'scripts/joint_hardness_policy.py').read_bytes()
    # Preserve the exact feature implementation and initialization imported by the pilot.
    files['runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py']=(ROOT/'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py').read_bytes()
    return files


def prepare_code(parent):
    files=prepared_sources()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    code=parent/('.hardness-code-'+revision[:16]);code.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:raise ValueError('Pinned source changed: '+name)
        if not path.exists():path.write_bytes(data)
    atomic_json(code/'source-manifest.json',hashes)
    return code


def command(output,code,stop=20000,resume=False):
    return original_command(SimpleNamespace(inputs=INPUT,output=output,stop_at_update=stop,
        resume=resume,reward_objective='regret',memory_execution='optimized'),code)


def prepare():
    STATE.mkdir(parents=True,exist_ok=True)
    audit=json.loads((INPUT/'rl-input-audit.json').read_text())
    total=0
    for row in audit['files']:
        relative=Path(row['path'])
        if relative.parts[0]!='input' or '..' in relative.parts:raise ValueError('Unsafe input audit path')
        path=INPUT/relative.relative_to('input')
        if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Original input changed: '+row['path'])
        total+=row['bytes']
    code=prepare_code(STATE)
    plan=dict(steps=20000,initialization='original SmolLM2 + original identity-warmed HN; fresh predictor',
        lm_seed=42,predictor_seed=43,tokenizers=1000,batch_size=128,microbatch_size=32,
        max_positions=128,continuation='variable, corrected paired boundaries',reward='paired pre-update transferred BPB minus canonical BPB',
        selector='unchanged pilot predictor: one current-observation MSE update, Adam .001; one-in-four shuffled coverage',
        lr_schedule='original 190k schedule; not rescaled',weights_every=500,restart_every=2000,
        input_files_verified=len(audit['files']),input_bytes_verified=total,
        input_audit_sha256=sha256(INPUT/'rl-input-audit.json'),
        code=str(code),source_manifest_sha256=sha256(code/'source-manifest.json'),
        platform='Tillicum · one H200',account='zlab',explicit_cpu_or_memory_requests=False,
        checkpoint_repo='cminst/anytok-hardness-predictor-20k')
    path=STATE/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Fresh run settings changed')
    atomic_json(path,plan)
    print(json.dumps(dict(stage='prepared',**plan)),flush=True)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();prepare()
