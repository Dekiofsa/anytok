"""Download weight-only diagnostic copies, score locally and refresh the plot."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_favorite_trajectory'
REPO = 'cminst/anytok-regret-rl-20k'
PYTHON = ROOT / '.venv-memory/bin/python'
WEIGHTS = ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors', 'policy.pt']
STARTED = time.monotonic()


def modal_bytes(volume, path):
    # The SDK streaming reader can wait indefinitely on V2 files. Reuse the
    # existing bounded block API, including for small checkpoint metadata.
    from download_local_memory_artifacts import block_urls
    import requests
    urls, expected = block_urls(volume, path)
    chunks = []
    for url in urls:
        try:
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status(); chunks.append(response.content)
        except requests.RequestException:
            raise RuntimeError('Cloud metadata transfer failed: ' + path) from None
    payload = b''.join(chunks)
    if len(payload) != expected:
        raise ValueError('Incomplete Modal metadata download: ' + path)
    return payload


def emit(stage, completed=0, total=18, **fields):
    row = dict(stage=stage, completed=completed, total=total,
        platform='Local · RTX PRO 6000 96 GB', at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - STARTED,
        workers=[] if stage in ['complete', 'failed'] else [dict(id='local-favorite', profile='local',
            platform='Local · RTX PRO 6000 96 GB', state=stage,
            task=f"Checkpoint {fields.get('step', 'preparation')} · tokenizer #771")], **fields)
    atomic_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def prepare():
    STATE.mkdir(parents=True, exist_ok=True)
    inventory = STATE / 'inventory.json'
    if inventory.exists():
        return json.loads(inventory.read_text())
    emit('preparing_exact_checkpoint_inventory')
    api = HfApi(); revision = api.model_info(REPO).sha
    prefix = 'diagnostics/policy-ranking-step-006500/'
    for remote, local in [('training-panel.json.gz', 'training-panel.json.gz'),
                          ('panel-manifest.json', 'panel-manifest.json'),
                          ('run-plan.json', 'ranking-run-plan.json')]:
        path = hf_hub_download(REPO, prefix + remote, revision=revision)
        shutil.copy2(path, STATE / local)
    manifest = json.loads((STATE / 'panel-manifest.json').read_text())
    if sha256(STATE / 'training-panel.json.gz') != manifest['panel_sha256']:
        raise ValueError('Panel download differs from the frozen ranking')
    events = hf_hub_download(REPO, 'metadata/events.jsonl', revision=revision)
    shutil.copy2(events, STATE / 'training-events.jsonl')
    from modal_tokenizer_pool_transfer import source_client
    import modal
    sources = {}
    for profile in ['queues', 'stars']:
        client, _ = source_client(profile)
        volume = modal.Volume.from_name('anytok-pool-rl-20k', version=2, client=client)
        for entry in volume.listdir('regret-run'):
            if '/step-' not in entry.path:
                continue
            step = int(entry.path.rsplit('step-', 1)[1])
            if step > 5288:
                continue
            emit('reading_checkpoint_manifest', step=step, source=profile)
            raw = modal_bytes(volume, entry.path + '/manifest.json')
            metadata = json.loads(raw)
            sources[step] = dict(step=step, kind='modal', profile=profile,
                                 volume='anytok-pool-rl-20k', prefix=entry.path,
                                 manifest=metadata, manifest_sha256=hashlib.sha256(raw).hexdigest())
    files = api.list_repo_files(REPO, revision=revision)
    for filename in files:
        if filename.startswith('checkpoints/step-') and filename.endswith('/manifest.json'):
            step = int(filename.split('/')[1].removeprefix('step-'))
            if 5288 <= step <= 8000:
                path = Path(hf_hub_download(REPO, filename, revision=revision))
                sources[step] = dict(step=step, kind='hf', repo=REPO, revision=revision,
                    prefix=filename.rsplit('/', 1)[0], manifest=json.loads(path.read_text()),
                    manifest_sha256=sha256(path))
    expected = sorted([*range(500, 8001, 500), 2912, 5288])
    if sorted(sources) != expected:
        raise ValueError('Missing original-trajectory checkpoints: ' + str(sorted(sources)))
    for step, row in sources.items():
        if row['manifest']['step'] != step or not set(WEIGHTS) <= row['manifest']['files'].keys():
            raise ValueError('Checkpoint lacks complete model/policy weights')
    data = dict(tokenizer=771, steps=expected, checkpoints=[sources[step] for step in expected],
        hf_revision=revision, panel_sha256=manifest['panel_sha256'],
        training_events_sha256=sha256(STATE / 'training-events.jsonl'),
        approval='User explicitly requested local evaluation of this tokenizer throughout training; cloud originals retained.',
        excluded='Discarded transformer Modal continuation after 5288; resumed Tillicum grid is used from 5500.')
    atomic_json(inventory, data)
    return data


def download(row, completed, total):
    step = row['step']
    local = ROOT / 'modal_results/local_memory/artifacts/checkpoint'
    # The existing, previously approved step-2000 diagnostic copy stays intact.
    if step == 2000:
        saved = json.loads((local / 'manifest.json').read_text())
        if saved['files'] != row['manifest']['files']:
            raise ValueError('Existing step-2000 copy differs from the actual trajectory')
        return local, False
    stage = STATE / 'weight-copy' / f'step-{step:06d}'
    stage.mkdir(parents=True, exist_ok=True)
    atomic_json(stage / 'manifest.json', row['manifest'])
    if row['kind'] == 'modal':
        import modal
        from modal_tokenizer_pool_transfer import source_client
        from download_local_memory_artifacts import resume_blocks
        client, _ = source_client(row['profile'])
        volume = modal.Volume.from_name(row['volume'], version=2, client=client)
    for name in WEIGHTS:
        target = stage / name; expected = row['manifest']['files'][name]
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target.exists() and target.stat().st_size == expected['bytes']
                and sha256(target) == expected['sha256']):
            continue
        emit('downloading_weight_copy', completed, total, step=step, file=name,
             file_bytes=expected['bytes'], source=row['kind'])
        if row['kind'] == 'hf':
            fetched = Path(hf_hub_download(row['repo'], row['prefix'] + '/' + name,
                revision=row['revision'], local_dir=stage / 'hf-download'))
            fetched.replace(target)
        else:
            partial = target.with_suffix(target.suffix + '.partial')
            if expected['bytes'] > 64 * 2**20:
                last_progress = [0.]
                def progress(done, missing, blocks):
                    now = time.monotonic()
                    if now - last_progress[0] >= 10 or done == missing:
                        last_progress[0] = now
                        emit('downloading_weight_copy', completed, total, step=step, file=name,
                             file_bytes=expected['bytes'], source=row['kind'],
                             file_bytes_downloaded=min(expected['bytes'], (blocks - missing + done) * 8 * 2**20))
                for attempt in range(3):
                    resume_blocks(volume, row['prefix'] + '/' + name, partial,
                                  expected['bytes'], force_all=attempt > 0,
                                  concurrency=32, progress=progress)
                    if partial.stat().st_size == expected['bytes'] and sha256(partial) == expected['sha256']:
                        break
                else:
                    raise ValueError('Downloaded Modal blocks failed their full checksum: ' + name)
            else:
                partial.write_bytes(modal_bytes(volume, row['prefix'] + '/' + name))
            partial.replace(target)
        if target.stat().st_size != expected['bytes'] or sha256(target) != expected['sha256']:
            raise ValueError('Downloaded weights changed: ' + name)
    return stage, True


def main():
    inventory = prepare(); total = len(inventory['steps'])
    # Preflight the complete workload on the existing exact step-2000 copy.
    priorities = {2000: 0, 500: 1, 8000: 2}
    ordered = sorted(inventory['checkpoints'], key=lambda row: (priorities.get(row['step'], 3), row['step']))
    for row in ordered:
        step = row['step']; output = STATE / 'scores' / f'{step:06d}.json'
        completed = len(list((STATE / 'scores').glob('*.json')))
        if not output.exists():
            checkpoint, owned = download(row, completed, total)
            emit('scoring', completed, total, step=step, batches_scored=0)
            command = [str(PYTHON), '-u', 'scripts/evaluate_regret_favorite.py',
                       '--checkpoint', str(checkpoint), '--step', str(step)]
            env = {**os.environ, 'TOKENIZERS_PARALLELISM':'false', 'PYTHONUNBUFFERED':'1',
                   'PYTORCH_ALLOC_CONF':'expandable_segments:True'}
            worker = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, env=env)
            for line in worker.stdout:
                print(line, end='', flush=True)
                try:
                    update = json.loads(line)
                except ValueError:
                    continue
                if isinstance(update, dict) and update.get('stage') == 'scoring':
                    emit('scoring', completed, total, step=step,
                         batches_scored=update['batches_scored'], batch_total=32)
            if worker.wait() or not output.exists():
                raise RuntimeError(f'Checkpoint {step} scoring failed; all native inputs and artifacts preserved')
            result = json.loads(output.read_text())
            if result['step'] != step:
                raise ValueError('Wrong step returned by worker')
            # Only this script's newly created diagnostic copy is removed.
            if owned:
                shutil.rmtree(checkpoint)
        subprocess.run([str(PYTHON), 'scripts/report_regret_favorite.py'], cwd=ROOT, check=True)
        completed = len(list((STATE / 'scores').glob('*.json')))
        emit('checkpoint_scored', completed, total, step=step)
    report = ROOT / 'reports/regret-favorite-trajectory'
    HfApi().upload_folder(repo_id=REPO, folder_path=report,
        path_in_repo='diagnostics/tokenizer-0771-trajectory/report',
        commit_message='Tokenizer 771 across all committed regret checkpoints through 8000')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'zett-modal-20k':
        raise RuntimeError('Report ready; branch changed, so automatic Git publication stopped')
    subprocess.run(['git', 'add', 'reports/regret-favorite-trajectory'], cwd=ROOT, check=True)
    staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(not name.startswith('reports/regret-favorite-trajectory/') for name in staged):
        raise RuntimeError('Report ready; unrelated staged files prevent automatic publication')
    subprocess.run(['git','commit','-m','Report tokenizer 771 learning across the paused regret run'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','zett-modal-20k'],cwd=ROOT,check=True)
    emit('complete', total, total, report='reports/regret-favorite-trajectory')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        count = len(list((STATE / 'scores').glob('*.json')))
        emit('failed', count, error=str(error))
        raise
