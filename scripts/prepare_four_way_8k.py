"""Prepare signed cloud-to-cloud transfers; never download artifact payloads locally."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import modal
from download_local_memory_artifacts import block_urls
from modal_tokenizer_pool_transfer import source_client
from package_training_inputs import atomic_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/four_way_8k'


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    sources = {}
    def volume(profile, name):
        key = profile, name
        if key not in sources:
            client, _ = source_client(profile)
            sources[key] = modal.Volume.from_name(name, version=2, client=client)
        return sources[key]
    def metadata(profile, name, path, label):
        raw = b''.join(volume(profile, name).read_file(path))
        value = json.loads(raw)
        atomic_json(STATE / (label + '.json'), value)
        return value
    shared = ('blackpantherx86', 'anytok-rl-all-pool-20k')
    spec = metadata(*shared, 'spec.json', 'original-spec')
    prepared = metadata(*shared, 'corrected-pool-check-v1/preparation.json', 'parent-preparation')
    native = json.loads((ROOT / 'plans/uniform-weakness-v1.json').read_text())
    draw = json.loads((ROOT / 'plans/corrected-pool-check-v1.json').read_text())
    models = []
    jobs = []
    for label, profile, name, prefix in [
        ('uniform', 'gildmember', 'anytok-pool-20k', 'run'),
        ('raw_bpb', 'neural-net-1411', 'anytok-pool-rl-20k', 'settings-run'),
    ]:
        checkpoint = prefix + '/step-008000'
        manifest = metadata(profile, name, checkpoint + '/manifest.json', label + '-manifest')
        config = metadata(profile, name, prefix + '/run_config.json', label + '-config')
        if manifest['step'] != 8000:
            raise ValueError('Wrong checkpoint')
        models.append(dict(name=label, manifest=manifest, config=config,
                           checkpoint='checkpoints/' + label, source=dict(profile=profile, volume=name, path=checkpoint)))
        for path in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
            jobs.append(dict(profile=profile, volume=name, remote=checkpoint + '/' + path,
                path='checkpoints/' + label + '/' + path, **manifest['files'][path]))
    candidates = []
    for index in sorted(draw['random_indices']):
        row = spec['tokenizers'][index]
        if row['index'] != index:
            raise ValueError('Pool index mapping changed')
        candidates.append(dict(id=f'pool-{index:04d}', kind='pool', index=index,
            sha256=row['sha256'], vocab_size=row['vocab_size'], config=row['config'],
            path=f'input/pool/tokenizers/{index:04d}/tokenizer.json'))
    pool_hashes = {r['sha256'] for r in spec['tokenizers']}
    for row in native['candidates']:
        if not (row['id'].startswith('ood48-') or row['id'] == 'base64'):
            continue
        if row['sha256'] in pool_hashes:
            raise ValueError('Held-out tokenizer is in the training pool')
        destination = 'heldout/' + row['id'] + '/tokenizer.json'
        candidates.append(dict(id=row['id'], kind='saved-native', sha256=row['sha256'],
            vocab_size=row['vocab_size'], config=row['settings'], path=destination))
        jobs.append(dict(profile='trenton', volume='anytok-pool-generalization',
            remote='weakness-v1/tokenizers/' + row['id'] + '/tokenizer.json',
            path=destination, sha256=row['sha256']))
    if len(candidates) != 62:
        raise ValueError('Require all 50 pool and 12 held-out tokenizers')
    for path, destination in [('corrected-pool-check-v1/panel.json', 'parent-panel.json')]:
        jobs.append(dict(profile=shared[0], volume=shared[1], remote=path, path=destination))
    def signed(row):
        urls, size = block_urls(volume(row['profile'], row['volume']), row['remote'])
        if 'bytes' in row and row['bytes'] != size:
            raise ValueError('Cloud source size changed: ' + row['remote'])
        return dict(**{k:v for k,v in row.items() if k != 'bytes'}, bytes=size, urls=urls)
    # Hydrate source handles before parallel RPCs.
    for row in jobs:
        volume(row['profile'], row['volume'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(signed, jobs))
    path = STATE / 'signed-files.json'
    atomic_json(path, dict(block_bytes=8 * 2**20, files=files))
    path.chmod(0o600)
    atomic_json(STATE / 'source-plan.json', dict(version=1, step=8000, candidates=candidates,
        models=models, parent_panel_sha256=prepared['panel_sha256'],
        random_indices=draw['random_indices'], random_indices_sha256=draw['random_indices_sha256'],
        input_hashes=spec['input_hashes'], position_cap=128, suffix_tokens=32,
        sources=[{k:v for k,v in row.items() if k != 'urls'} for row in files]))
    print(json.dumps(dict(stage='signed_metadata_ready', files=len(files), bytes=sum(r['bytes'] for r in files))), flush=True)


if __name__ == '__main__':
    main()
