"""Prepare Qingyun's held-out full-pool eval (948787e) for every saved checkpoint of the raw-BPB run.

His scorer (run_rift_pool_sweep.py), its Slurm script (heldout_pool_h200.sbatch)
and his restore and staging scripts run unchanged. They accept only the
checkpoints named in his plan.json, which are his runs'. This script builds a
second workspace from his restored and verified one (restore_rift_pool_archive.py)
in which only the checkpoint list differs:

  plan.json        his plan with exactly two keys replaced: checkpoints (this run's
                   labels, raw-bpb-NNNNNN, in step order) and checkpoint_hashes
                   (their lm.pt and hypernetwork SHA-256s). The passages, their
                   continuations, the code snapshot, the runtime pins, the
                   tokenizer index and every other key stay his.
  sources.json     his input-bundle record, and checkpoints pointing at
                   Dekiofsa/RLtokenizers at a pinned revision with only the three
                   files the scorer reads, so his staging script downloads nothing
                   else (no optimizer or trainer state).
  resume-reference/checkpoints/<label>/ready.json
                   the checkpoint's own manifest.json from Hugging Face: its step
                   and every file's size and SHA-256. His preflight checks the
                   staged files against it.
  checkpoints.txt  the labels, one per line, for the Slurm array.
  adaptation.json  what changed, and the hashes of everything copied.

The workspace also gets his top-level modules and inbox/ (the passages and the
code snapshot) byte for byte, and his two scorer files from this repository,
checked against their SHA-256 at 948787e, as his README copies them.

A new checkpoint list changes the plan's hash, so his panel audit runs again in
the new workspace (run_rift_pool_sweep.py audit). check-audit then requires that
audit to equal his in every field except plan_sha256: the same 156 passages, the
same native prefixes for all 1,000 tokenizers, the same trimmed prefixes and the
same 4,992 tokens and 14,429 bytes.

Every save of the run is required: steps 0 to 20,000 every 500, plus any extra
save (16,409). CPU only: reads small JSON files; no weights and no GPU.
Exit codes: 0 done, 2 refused (the message says why).
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from package_training_inputs import atomic_json, sha256
from pool_generalization import digest

HERE = Path(__file__).resolve().parent
REPO = 'Dekiofsa/RLtokenizers'
FOLDER = 'raw-bpb-rl-20k'
# main on 2026-11-23; it holds all 42 saves of the run.
REVISION = '60670ac8cea5b53695ec43db8160c86856c2b70b'
GRID = range(0, 20001, 500)
SCORED_FILES = ('lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors')
CHANGED_PLAN_KEYS = ('checkpoint_hashes', 'checkpoints')
PROTOCOL = dict(documents=156, scored_tokens=4992, scored_bytes=14429)
# SHA-256 of his scorer files at 948787e. They are copied into the workspace unchanged.
SCORER_FILES = {
    'run_rift_pool_sweep.py': 'd3644be1fcbcc499a81cd7286838c384ca6a7413c8685bbc6c7d85ca649d1687',
    'rift_eval_memory.py': '011c3c82caf7848391296aa38756590a23590f324ab844fe23c6fc4e86f432d6',
}
# Top-level files of his workspace that describe his run, not the protocol.
NOT_COPIED = {'plan.json', 'sources.json', 'resume-verified.json'}
STEP_FOLDER = re.compile(r'step-(\d{6})')
SHA256 = re.compile(r'[0-9a-f]{64}')


def read(path):
    return json.loads(Path(path).read_text())


def label(step):
    return f'raw-bpb-{step:06d}'


def check_reference(reference):
    """His restored workspace: the plan his verified audit covers, that audit, and his sources."""
    if not (reference / 'resume-verified.json').is_file():
        raise ValueError(f'{reference} is not a restored workspace: run restore_rift_pool_archive.py first')
    plan, audit = read(reference / 'plan.json'), read(reference / 'run/panel-audit.json')
    if audit.get('stage') != 'verified' or audit.get('plan_sha256') != digest(plan):
        raise ValueError("His plan.json is not the plan his verified panel audit covers")
    if read(reference / 'resume-verified.json').get('plan_sha256') != audit['plan_sha256']:
        raise ValueError('His restore receipt names another plan')
    if {key: audit.get(key) for key in PROTOCOL} != PROTOCOL:
        raise ValueError('His audit is not the 156-passage, 4,992-token protocol')
    sources = read(reference / 'sources.json')
    if 'input_bundle' not in sources:
        raise ValueError('His sources.json has no input-bundle record')
    return plan, audit, sources


def check_manifest(step, manifest):
    if manifest.get('step') != step:
        raise ValueError(f'The manifest in step-{step:06d} names step {manifest.get("step")}')
    files = manifest.get('files', {})
    for name in SCORED_FILES:
        proof = files.get(name) or {}
        if not (type(proof.get('bytes')) is int and proof['bytes'] > 0 and SHA256.fullmatch(str(proof.get('sha256')))):
            raise ValueError(f'The manifest in step-{step:06d} has no size and SHA-256 for {name}')
    return manifest


def manifests_from_folder(folder):
    """{step: (manifest, SHA-256 of the manifest file)} for every step-NNNNNN/manifest.json in folder."""
    found = {}
    for path in sorted(Path(folder).glob('step-*/manifest.json')):
        match = STEP_FOLDER.fullmatch(path.parent.name)
        if match:
            step = int(match.group(1))
            found[step] = (check_manifest(step, read(path)), sha256(path))
    return found


def manifests_from_hub(revision):
    """The same, read anonymously from the public repository at a pinned revision."""
    from huggingface_hub import HfApi, hf_hub_download

    entries = HfApi().list_repo_tree(REPO, path_in_repo=FOLDER, revision=revision, token=False)
    names = sorted(name for name in (entry.path.rsplit('/', 1)[-1] for entry in entries) if STEP_FOLDER.fullmatch(name))
    with tempfile.TemporaryDirectory() as scratch:
        for name in names:
            hf_hub_download(REPO, f'{FOLDER}/{name}/manifest.json', revision=revision, local_dir=scratch, token=False)
        return manifests_from_folder(Path(scratch) / FOLDER)


def check_coverage(steps):
    missing = sorted(set(GRID) - set(steps))
    if missing:
        raise ValueError(f'Every save of the run is required; missing steps {missing}')
    outside = sorted(step for step in steps if not GRID.start <= step <= GRID.stop - 1)
    if outside:
        raise ValueError(f'Steps outside the run: {outside}')


def adapted_plan(plan, manifests):
    """His plan with only the checkpoint list and its hashes replaced by this run's."""
    steps = sorted(manifests)
    check_coverage(steps)
    adapted = copy.deepcopy(plan)
    adapted['checkpoints'] = [label(step) for step in steps]
    adapted['checkpoint_hashes'] = {label(step): {name: manifests[step][0]['files'][name]['sha256'] for name in SCORED_FILES}
                                    for step in steps}
    changed = sorted(key for key in set(plan) | set(adapted) if plan.get(key) != adapted.get(key))
    if not set(changed) <= set(CHANGED_PLAN_KEYS):
        raise AssertionError(f'Unexpected plan changes: {changed}')
    return adapted


def tree_hashes(root):
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob('*')) if path.is_file()}


def prepare(reference, workspace, manifests, manifest_source, revision):
    plan, audit, sources = check_reference(reference)
    if workspace.exists():
        raise FileExistsError(f'Choose an empty destination. {workspace} already exists')
    adapted = adapted_plan(plan, manifests)
    steps = sorted(manifests)
    for name, expected in SCORER_FILES.items():
        if sha256(HERE / name) != expected:
            raise ValueError(f'scripts/{name} differs from his file at 948787e')
    workspace.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workspace.parent) as scratch:
        staged = Path(scratch) / 'workspace'
        staged.mkdir()
        copied = {}
        for path in sorted(reference.iterdir()):
            if path.is_file() and path.name not in NOT_COPIED:
                shutil.copy2(path, staged / path.name)
                copied[path.name] = sha256(path)
        shutil.copytree(reference / 'inbox', staged / 'inbox')
        archived_scorer = {name: copied[name] for name in SCORER_FILES if name in copied}
        for name in SCORER_FILES:
            shutil.copy2(HERE / name, staged / name)
        added = []
        if not (staged / 'four_way_8k.py').is_file() and not (staged / 'inbox/code/four_way_8k.py').is_file():
            # Imported by his audit only when a shared prefix needs trimming.
            shutil.copy2(HERE / 'four_way_8k.py', staged / 'four_way_8k.py')
            added.append(dict(file='four_way_8k.py', sha256=sha256(HERE / 'four_way_8k.py')))
        atomic_json(staged / 'plan.json', adapted)
        items = {label(step): dict(prefix=f'{FOLDER}/step-{step:06d}', step=step,
                                   files={name: manifests[step][0]['files'][name] for name in SCORED_FILES})
                 for step in steps}
        atomic_json(staged / 'sources.json', dict(input_bundle=sources['input_bundle'],
                                                  checkpoints=dict(repo=REPO, revision=revision, items=items)))
        for step in steps:
            atomic_json(staged / 'resume-reference/checkpoints' / label(step) / 'ready.json', manifests[step][0])
        (staged / 'checkpoints.txt').write_text(''.join(label(step) + '\n' for step in steps))
        result = dict(
            protocol='heldout-pool-4992', scorer_commit='948787e', reference_plan_sha256=audit['plan_sha256'],
            plan_sha256=digest(adapted), changed_plan_keys=list(CHANGED_PLAN_KEYS), plan_keys=sorted(adapted),
            reference_audit={key: audit.get(key) for key in ('panel_sha256', 'native_prefixes_sha256', *PROTOCOL)},
            checkpoints=dict(repo=REPO, revision=revision, folder=FOLDER, manifests_from=manifest_source,
                             count=len(steps), steps=steps, scored_files=list(SCORED_FILES),
                             manifest_sha256={label(step): manifests[step][1] for step in steps}),
            copied_from_reference=copied, inbox=tree_hashes(staged / 'inbox'), scorer_files=SCORER_FILES,
            archived_scorer_files=archived_scorer, added_files=added,
            not_copied_directories=sorted(p.name for p in reference.iterdir() if p.is_dir() and p.name != 'inbox'))
        atomic_json(staged / 'adaptation.json', result)
        staged.rename(workspace)
    return dict(workspace=str(workspace), checkpoints=len(steps), plan_sha256=result['plan_sha256'],
                reference_plan_sha256=result['reference_plan_sha256'])


def check_audit(reference, workspace):
    """The audit re-run in this workspace must equal his in every field but the plan hash."""
    _, reference_audit, _ = check_reference(reference)
    plan, audit = read(workspace / 'plan.json'), read(workspace / 'run/panel-audit.json')
    if digest(plan) != read(workspace / 'adaptation.json')['plan_sha256']:
        raise ValueError('plan.json changed after preparation')
    if audit.get('plan_sha256') != digest(plan):
        raise ValueError("The audit does not cover this workspace's plan: run run_rift_pool_sweep.py audit here")
    differ = sorted(key for key in set(reference_audit) | set(audit)
                    if key != 'plan_sha256' and reference_audit.get(key) != audit.get(key))
    if differ:
        raise ValueError(f'The re-run audit differs from his in {differ}')
    if (digest(read(workspace / 'run/scoring-panel.json')) != audit['panel_sha256']
            or digest(read(workspace / 'run/native-prefixes.json')) != audit['native_prefixes_sha256']):
        raise ValueError('The audited panel files changed after the audit')
    result = dict(matches_reference=True, plan_sha256=audit['plan_sha256'],
                  reference_plan_sha256=reference_audit['plan_sha256'],
                  compared_fields=sorted(set(audit) - {'plan_sha256'}),
                  **{key: audit[key] for key in ('panel_sha256', 'native_prefixes_sha256', *PROTOCOL)})
    atomic_json(workspace / 'run/audit-matches-reference.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest='action', required=True)
    build = actions.add_parser('prepare', help='Build the raw-BPB workspace from his restored one')
    check = actions.add_parser('check-audit', help='Compare the re-run panel audit with his')
    for sub in (build, check):
        sub.add_argument('--reference', type=Path, required=True, help='His restored, verified workspace')
        sub.add_argument('--workspace', type=Path, required=True, help='The raw-BPB workspace (new for prepare)')
    build.add_argument('--revision', default=REVISION, help='Pinned revision of Dekiofsa/RLtokenizers')
    build.add_argument('--manifests', type=Path,
                       help='Folder of step-NNNNNN/manifest.json to use instead of reading them from Hugging Face')
    args = parser.parse_args(argv)
    reference, workspace = args.reference.resolve(), args.workspace.resolve()
    try:
        if args.action == 'prepare':
            if args.manifests:
                manifests, source = manifests_from_folder(args.manifests), 'local folder'
            else:
                manifests, source = manifests_from_hub(args.revision), 'huggingface'
            result = prepare(reference, workspace, manifests, source, args.revision)
        else:
            result = check_audit(reference, workspace)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
        print(json.dumps(dict(stage='refused', error=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
