"""Read checkpoint metadata and sign direct Modal-to-Tillicum weight transfers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import modal
from download_local_memory_artifacts import block_urls
from modal_tokenizer_pool_transfer import source_client
from package_training_inputs import atomic_json
from pool_generalization import digest

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/predictor_trajectory'
REMOTE = '/gpfs/scrubbed/qxli2/anytok-hardness-20k'
STEPS = [2000, 4000, 6000, 8000]
FILES = ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    parent = json.loads((ROOT/'reports/four-way-8k/plan.json').read_text())
    uniform = next(m for m in parent['models'] if m['name'] == 'uniform')
    client, _ = source_client('gildmember')
    volume = modal.Volume.from_name('anytok-pool-20k', version=2, client=client)
    config = json.loads(b''.join(volume.read_file('run/run_config.json')))
    if config != uniform['config']:
        raise ValueError('Uniform training run changed')
    models, jobs = [], []
    for step in STEPS[:-1]:
        prefix = f'run/step-{step:06d}'
        manifest = json.loads(b''.join(volume.read_file(prefix+'/manifest.json')))
        if manifest['step'] != step:
            raise ValueError('Wrong source checkpoint')
        label = f'uniform-{step:06d}'
        models.append(dict(name='uniform', step=step, job=label, manifest=manifest,
            checkpoint=f'{REMOTE}/modal_results/predictor_trajectory/checkpoints/{label}',
            source=dict(profile='gildmember', volume='anytok-pool-20k', path=prefix)))
        for relative in FILES:
            expected = manifest['files'][relative]
            jobs.append(dict(remote=prefix+'/'+relative,
                path=f'checkpoints/{label}/{relative}', **expected))

    def signed(row):
        urls, size = block_urls(volume, row['remote'])
        if size != row['bytes']:
            raise ValueError('Source file size changed')
        return dict(row, urls=urls)

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(signed, jobs))
    atomic_json(STATE/'signed-files.json', dict(block_bytes=8*2**20, files=files))
    (STATE/'signed-files.json').chmod(0o600)
    source = dict(version=1, steps=STEPS, models=models,
        parent_plan_sha256=digest(parent), parent_panel_sha256=parent['panel_sha256'],
        comparisons=496, new_comparisons=372, cached_comparisons=124,
        training_resumed=False)
    atomic_json(STATE/'source-plan.json', source)
    print(json.dumps(dict(stage='signed_metadata_ready', files=len(files),
        bytes=sum(r['bytes'] for r in files))), flush=True)


if __name__ == '__main__':
    main()
