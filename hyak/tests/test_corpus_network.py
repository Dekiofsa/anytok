"""Live miniature build of all 23 buckets against Hugging Face at the pinned revisions (network).

Run with the data environment (datasets + requests installed)::

    HYAK_TEST_NETWORK=1 PYTHONDONTWRITEBYTECODE=1 <data-env python> -m pytest -p no:cacheprovider \\
        hyak/tests/test_corpus_network.py -q

It proves that every one of the ten source adapters still works at its pinned revision with the
installed ``datasets`` / ``requests``: the probe reads the first record of each source, and the
build (``--test-scale``, default 200,000 train / 50,000 held-out bytes per bucket, override with
``HYAK_TEST_MINI_SCALE``) writes and verifies all 23 buckets through the production code path
(child processes, locks, verification).  The output goes to a temporary directory, or to
``$HYAK_TEST_MINI_CORPUS_OUT`` (kept; an existing complete miniature there is re-verified instead
of downloaded again).  Per-bucket results are written to ``<out>/.hyak/network-test-report.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hyak.corpus import build, sources, stage_inputs, verify
from hyak.tests.test_corpus_fixtures import REPO

pytestmark = pytest.mark.network


def _run(args, env, timeout):
    return subprocess.run([sys.executable, '-m', *args], capture_output=True, text=True, timeout=timeout,
                          cwd=str(REPO), env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1', **env))


def test_live_miniature_of_all_23_buckets(make_tmp_config, tmp_path):
    pytest.importorskip('datasets')
    pytest.importorskip('requests')
    tmp = make_tmp_config()
    env = {'HYAK_CONFIG': str(tmp.config_file), 'HYAK_MONITOR_ROOT': str(tmp.base / 'monitor'),
           'HF_HOME': os.environ.get('HF_HOME') or str(tmp.base / 'hf-home')}
    scale = os.environ.get('HYAK_TEST_MINI_SCALE', '200000:50000')
    out = Path(os.environ.get('HYAK_TEST_MINI_CORPUS_OUT') or tmp_path / 'mini-corpus')

    probe = _run(['hyak.corpus.build', '--probe'], env, timeout=1800)
    assert probe.returncode == 0, probe.stdout[-4000:] + probe.stderr[-4000:]
    report = json.loads((tmp.base / 'monitor' / 'corpus' / 'probe.json').read_text())
    assert report['failures'] == 0 and len(report['results']) == 10
    assert all(row['bytes'] > 0 and row['id'] for row in report['results'].values())

    result = _run(['hyak.corpus.build', '--corpus-root', str(out), '--test-scale', scale, '--workers', '2'],
                  env, timeout=3600)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-4000:]
    status = json.loads((tmp.base / 'monitor' / 'corpus' / 'status.json').read_text())
    assert status['state'] == 'complete' and status['completed'] == 23

    test_scale = build.parse_test_scale(scale)
    rows = {}
    for name, (size, spec) in sources.bucket_specs().items():
        outcome = json.loads((out / '.hyak' / 'results' / f'{size}__{name}.json').read_text())
        assert outcome['status'] in ('built', 'verified-existing'), (name, outcome)
        summary = build.verify_bucket(out / f'{size}-v1' / name, build.scaled_spec(spec, test_scale), test_scale)
        manifest = json.loads((out / f'{size}-v1' / name / 'manifest.json').read_text())
        rows[name] = {
            'size': size, 'status': outcome['status'], 'seconds': outcome.get('build_seconds', outcome.get('seconds')),
            'attempts': outcome.get('attempts'), 'counts': manifest['counts'],
            'first_train_id': summary['train']['first_id'], 'first_eval_id': summary['eval']['first_id'],
            'train_compressed_bytes': summary['train']['compressed_bytes'],
            'eval_compressed_bytes': summary['eval']['compressed_bytes'],
            'packages': manifest['hyak']['packages'], 'zlib': manifest['hyak']['zlib'],
        }
    (out / '.hyak' / 'network-test-report.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    for size in sources.POOL_SIZES:
        assert (out / f'{size}-v1' / 'summary.json').is_file()

    staged = stage_inputs.stage(tmp.paths, corpus_root=out, input_dir=tmp_path / 'input', allow_test_scale=True,
                                log=lambda *a: None)
    assert staged['buckets'] == 23 and staged['test_scale'] is True
    assert stage_inputs.check_contract(stage_inputs.InputLayout(tmp.paths, tmp_path / 'input'),
                                       stage_inputs.load_recipe()[1]) == []

    card = verify.scorecard(out, REPO)
    mismatches = {n: r['anchors']['sha256_mismatch_examples'] for n, r in card['buckets'].items()
                  if r['anchors']['sha256_mismatch_examples']}
    assert mismatches == {}, mismatches
    assert card['totals']['ids_found'] > 0 and card['totals']['length_violations'] == 0
