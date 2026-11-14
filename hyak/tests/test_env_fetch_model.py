"""Tests of hyak.model.fetch_model (DESIGN.md 5.A): verification against
plans/zett-modal-20k-inputs.json with the real SmolLM2 files, corrupted copies, the
no-delete/no-extra-files rules, resumable HTTPS, and the huggingface_hub call."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import urllib.error
import urllib.request

import pytest

fetch_model = importlib.import_module('hyak.model.fetch_model')
FetchError = fetch_model.FetchError

PINNED = {
    'config.json': {'bytes': 635, 'sha256': '33397f6af8090f7290f27ce6fb1cd23a95fd0915a5c648dfd10b6002aaf7fc48'},
    'model.safetensors': {'bytes': 3422777952,
                          'sha256': '1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0'},
    'tokenizer.json': {'bytes': 2104556, 'sha256': '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'},
}


def _record(data: bytes) -> dict:
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


@pytest.fixture
def small_model(scratch_assets, tmp_path, monkeypatch):
    """The real config.json + tokenizer.json and a small stand-in model.safetensors with its own pin."""
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('config.json', 'tokenizer.json'):
        shutil.copyfile(scratch_assets / 'full' / name, source / name)
    weights = b'stand-in safetensors bytes ' * 1000
    (source / 'model.safetensors').write_bytes(weights)
    expected = {'config.json': PINNED['config.json'], 'tokenizer.json': PINNED['tokenizer.json'],
                'model.safetensors': _record(weights)}
    monkeypatch.setattr(fetch_model, 'expected_files', lambda repo=None: dict(expected))
    return source, expected


def test_expected_files_come_from_the_pinned_inputs_manifest(repo_root):
    assert fetch_model.expected_files(repo_root) == PINNED
    assert fetch_model.MODEL_REVISION == 'effd688a12921b4cc83e3312b6feb579f70f9c71'
    assert fetch_model.MODEL_REPO == 'HuggingFaceTB/SmolLM2-1.7B'


def test_expected_files_refuse_a_changed_inputs_manifest(repo_root, tmp_path):
    fake = tmp_path / 'repo'
    (fake / 'plans').mkdir(parents=True)
    data = json.loads((repo_root / 'plans/zett-modal-20k-inputs.json').read_text())
    data['files']['model/config.json']['sha256'] = '0' * 64
    (fake / 'plans/zett-modal-20k-inputs.json').write_text(json.dumps(data))
    with pytest.raises(FetchError, match='does not match its pin'):
        fetch_model.expected_files(fake)


def test_real_local_files_verify(scratch_assets, repo_root):
    expected = fetch_model.expected_files(repo_root)
    for name in ('config.json', 'tokenizer.json'):
        assert fetch_model.verify_file(scratch_assets / 'full' / name, expected[name]) == []


def test_corrupted_truncated_and_linked_copies_fail(scratch_assets, tmp_path, repo_root):
    expected = fetch_model.expected_files(repo_root)
    data = bytearray((scratch_assets / 'full' / 'tokenizer.json').read_bytes())
    data[len(data) // 2] ^= 0x01                       # same size, one bit flipped
    corrupt = tmp_path / 'tokenizer.json'
    corrupt.write_bytes(bytes(data))
    problems = fetch_model.verify_file(corrupt, expected['tokenizer.json'])
    assert len(problems) == 1 and 'sha256' in problems[0] and expected['tokenizer.json']['sha256'] in problems[0]
    assert fetch_model.verify_file(corrupt, expected['tokenizer.json'], check_hash=False) == []
    truncated = tmp_path / 'config.json'
    truncated.write_bytes((scratch_assets / 'full' / 'config.json').read_bytes()[:-1])
    assert '634 bytes, expected 635' in fetch_model.verify_file(truncated, expected['config.json'])[0]
    link = tmp_path / 'linked.json'
    link.symlink_to(scratch_assets / 'full' / 'config.json')
    assert 'symlink' in fetch_model.verify_file(link, expected['config.json'])[0]
    assert 'missing' in fetch_model.verify_file(tmp_path / 'nope', expected['config.json'])[0]


def test_verify_model_dir_requires_exactly_the_three_files(small_model, tmp_path):
    source, expected = small_model
    assert fetch_model.verify_model_dir(source, expected) == []
    (source / 'tokenizer_config.json').write_text('{}')
    problems = fetch_model.verify_model_dir(source, expected)
    assert problems == [f"{source}: unexpected entries ['tokenizer_config.json'] (the model directory must hold "
                        'exactly config.json, model.safetensors, tokenizer.json)']
    assert 'directory missing' in fetch_model.verify_model_dir(tmp_path / 'absent', expected)[0]


def test_fetch_from_dir_copies_verifies_and_cleans_up(small_model, tmp_path):
    source, expected = small_model
    dest = tmp_path / 'input' / 'model'
    status = fetch_model.fetch(dest, from_dir=source, log=lambda m: None)
    assert status == {'config.json': 'fetched', 'model.safetensors': 'fetched', 'tokenizer.json': 'fetched'}
    assert sorted(p.name for p in dest.iterdir()) == ['config.json', 'model.safetensors', 'tokenizer.json']
    assert fetch_model.verify_model_dir(dest, expected) == []
    assert not (tmp_path / 'input' / 'model.download').exists()
    assert all(not (dest / n).is_symlink() for n in expected)
    again = fetch_model.fetch(dest, from_dir=source, log=lambda m: None)
    assert set(again.values()) == {'present'}


def test_fetch_follows_snapshot_symlinks_into_real_copies(small_model, tmp_path):
    source, _ = small_model
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        (snapshot / name).symlink_to(source / name)    # like an HF cache snapshot pointing into blobs/
    dest = tmp_path / 'model'
    fetch_model.fetch(dest, from_dir=snapshot, log=lambda m: None)
    assert all((dest / n).is_file() and not (dest / n).is_symlink() for n in ('config.json', 'tokenizer.json'))


def test_fetch_never_deletes_wrong_or_extra_files(small_model, tmp_path):
    source, _ = small_model
    dest = tmp_path / 'model'
    dest.mkdir()
    (dest / 'README.md').write_text('x')
    with pytest.raises(FetchError, match='unexpected entries'):
        fetch_model.fetch(dest, from_dir=source, log=lambda m: None)
    assert (dest / 'README.md').exists()
    (dest / 'README.md').unlink()
    (dest / 'config.json').write_text('{"wrong": true}')
    with pytest.raises(FetchError, match='rm '):
        fetch_model.fetch(dest, from_dir=source, log=lambda m: None)
    assert (dest / 'config.json').read_text() == '{"wrong": true}'


def test_fetch_rejects_a_source_with_the_wrong_bytes(small_model, tmp_path):
    source, _ = small_model
    (source / 'tokenizer.json').write_bytes(b'{}')
    dest = tmp_path / 'model'
    with pytest.raises(FetchError, match='does not serve the pinned file'):
        fetch_model.fetch(dest, from_dir=source, log=lambda m: None)
    assert not (dest / 'tokenizer.json').exists()
    assert (tmp_path / 'model.download' / 'tokenizer.json.bad').exists()


class _FakeResponse(io.BytesIO):
    def __init__(self, data, status, fail_after=None):
        super().__init__(data)
        self.status = status
        self._fail_after = fail_after

    def read(self, size=-1):
        if self._fail_after is not None and self.tell() >= self._fail_after:
            raise ConnectionResetError('simulated drop')
        if self._fail_after is not None and size and size > 0:
            size = min(size, self._fail_after - self.tell())
        return super().read(size)

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_https_download_resumes_with_range_requests(tmp_path, monkeypatch):
    data = os.urandom(300_000)
    expected = _record(data)
    calls = []

    def opener(request, timeout):
        calls.append(dict(request.header_items()))
        start = int(request.get_header('Range', 'bytes=0-')[6:-1])
        if len(calls) == 1:
            return _FakeResponse(data, 200, fail_after=100_000)       # connection drops mid-file
        return _FakeResponse(data[start:], 206 if start else 200)

    monkeypatch.setattr(fetch_model.time, 'sleep', lambda s: None)
    monkeypatch.setattr(fetch_model, 'CHUNK', 4096)
    path = fetch_model._https_download('model.safetensors', tmp_path, expected, None, lambda m: None, opener=opener)
    assert path.read_bytes() == data
    assert 'Range' not in calls[0] and calls[1]['Range'] == f'bytes={100_000}-'
    assert all('Authorization' not in c for c in calls)
    assert not (tmp_path / 'model.safetensors.partial').exists()


def test_https_download_restarts_when_range_is_ignored(tmp_path, monkeypatch):
    data = b'abcdefghij' * 1000
    (tmp_path / 'config.json.partial').write_bytes(b'XXXX')             # stale, wrong partial bytes
    monkeypatch.setattr(fetch_model.time, 'sleep', lambda s: None)
    path = fetch_model._https_download('config.json', tmp_path, _record(data), 'secret-token', lambda m: None,
                                       opener=lambda request, timeout: _FakeResponse(data, 200))
    assert path.read_bytes() == data


def test_https_download_gives_up_after_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_model.time, 'sleep', lambda s: None)

    def opener(request, timeout):
        raise urllib.error.URLError('no route to host')
    with pytest.raises(FetchError, match='rerun to resume'):
        fetch_model._https_download('config.json', tmp_path, {'bytes': 10, 'sha256': '0' * 64}, None,
                                    lambda m: None, opener=opener, retries=2)


class _HeaderResponse(_FakeResponse):
    def __init__(self, data, status, headers):
        super().__init__(data, status)
        self.headers = headers


def test_https_download_stops_when_the_server_keeps_sending_more_than_the_pinned_size(tmp_path, monkeypatch):
    """A proxy block page / another file with a 200 used to restart for ever without counting an attempt (env-4)."""
    expected = _record(b'x' * 1000)
    sleeps, requests = [], []
    monkeypatch.setattr(fetch_model.time, 'sleep', sleeps.append)

    def opener(request, timeout):
        requests.append(request)
        return _FakeResponse(b'<html>blocked</html>' * 500, 200)           # 10,000 bytes for a 1,000-byte file
    with pytest.raises(FetchError, match='gave up after 3 attempts'):
        fetch_model._https_download('config.json', tmp_path, expected, None, lambda m: None, opener=opener, retries=2)
    assert len(requests) == 3 and sleeps == [2, 4]        # bounded, with a growing pause before each retry
    assert not (tmp_path / 'config.json.partial').exists() and not (tmp_path / 'config.json').exists()


def test_https_download_refuses_a_declared_wrong_size_before_writing(tmp_path, monkeypatch):
    data = os.urandom(5000)
    expected = _record(data)
    monkeypatch.setattr(fetch_model.time, 'sleep', lambda s: None)
    written = []

    def opener(request, timeout):
        written.append((tmp_path / 'config.json.partial').exists())
        return _HeaderResponse(data, 200, {'Content-Length': '123456'})   # declares another size
    with pytest.raises(FetchError, match='Content-Length is 123456'):
        fetch_model._https_download('config.json', tmp_path, expected, None, lambda m: None, opener=opener, retries=1)
    assert not any(written[1:]) and not (tmp_path / 'config.json.partial').exists()
    # a 206 whose Content-Range total is wrong is refused the same way; the right totals are accepted
    (tmp_path / 'config.json.partial').write_bytes(data[:100])

    def ranged(request, timeout):
        return _HeaderResponse(data[100:], 206, {'Content-Range': f'bytes 100-4999/{len(data)}'})
    assert fetch_model._https_download('config.json', tmp_path, expected, None, lambda m: None,
                                       opener=ranged).read_bytes() == data


def test_https_download_oversize_partial_counts_an_attempt(tmp_path, monkeypatch):
    expected = _record(b'y' * 500)
    (tmp_path / 'config.json.partial').write_bytes(b'z' * 900)               # larger than the pinned file
    sleeps = []
    monkeypatch.setattr(fetch_model.time, 'sleep', sleeps.append)
    path = fetch_model._https_download('config.json', tmp_path, expected, None, lambda m: None,
                                       opener=lambda request, timeout: _FakeResponse(b'y' * 500, 200))
    assert path.read_bytes() == b'y' * 500 and len(sleeps) == 1


def test_token_is_not_forwarded_to_another_host():
    handler = fetch_model._NoAuthRedirect()
    request = urllib.request.Request('https://huggingface.co/a/resolve/r/model.safetensors',
                                     headers={'Authorization': 'Bearer x', 'User-Agent': 'u'})
    same = handler.redirect_request(request, None, 302, 'Found', {}, 'https://huggingface.co/other')
    assert same.get_header('Authorization') == 'Bearer x'
    other = handler.redirect_request(request, None, 302, 'Found', {}, 'https://cas-bridge.xethub.hf.co/x')
    assert other.get_header('Authorization') is None and other.get_header('User-agent') == 'u'


def test_hub_download_passes_repo_revision_file_and_local_dir(tmp_path, monkeypatch):
    calls = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs['local_dir']) / kwargs['filename']
        target.write_bytes(b'{}')
        cache = Path(kwargs['local_dir']) / '.cache' / 'huggingface'
        cache.mkdir(parents=True, exist_ok=True)
        return str(target)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=hf_hub_download))
    path = fetch_model._hub_download('config.json', tmp_path, 'tok', lambda m: None)
    assert path == tmp_path / 'config.json'
    assert calls == [{'repo_id': 'HuggingFaceTB/SmolLM2-1.7B', 'filename': 'config.json',
                      'revision': 'effd688a12921b4cc83e3312b6feb579f70f9c71', 'local_dir': str(tmp_path),
                      'token': 'tok'}]


def test_fetch_with_hub_leaves_exactly_three_files(small_model, tmp_path, monkeypatch):
    source, expected = small_model

    def hf_hub_download(repo_id, filename, revision, local_dir, token):
        cache = Path(local_dir) / '.cache' / 'huggingface' / 'download'
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f'{filename}.metadata').write_text(revision)
        shutil.copyfile(source / filename, Path(local_dir) / filename)
        return str(Path(local_dir) / filename)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=hf_hub_download))
    dest = tmp_path / 'model'
    fetch_model.fetch(dest, method='hub', log=lambda m: None)
    assert sorted(p.name for p in dest.iterdir()) == ['config.json', 'model.safetensors', 'tokenizer.json']
    assert not (tmp_path / 'model.download').exists()


def test_cli_verify_only(small_model, tmp_path, capsys, repo_root, monkeypatch):
    source, _ = small_model
    assert fetch_model.main(['--verify-only', '--dest', str(source)]) == 0
    assert 'OK' in capsys.readouterr().out
    (source / 'config.json').write_text('{}')
    assert fetch_model.main(['--verify-only', '--dest', str(source)]) == 1
    assert 'PROBLEM' in capsys.readouterr().err


def test_cli_verify_only_against_the_real_pins(scratch_assets, tmp_path, capsys):
    dest = tmp_path / 'model'
    dest.mkdir()
    for name in ('config.json', 'tokenizer.json'):
        shutil.copyfile(scratch_assets / 'full' / name, dest / name)
    assert fetch_model.main(['--verify-only', '--dest', str(dest)]) == 1
    err = capsys.readouterr().err
    assert 'model.safetensors: missing' in err and 'config.json' not in err.split('ERROR')[0].replace(
        'model.safetensors', '')


def test_cli_fetch_uses_the_configured_model_dir(small_model, tmp_config, capsys):
    source, expected = small_model
    assert fetch_model.main(['--from-dir', str(source)]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line['dest'] == str(tmp_config.paths.model_dir) and line['revision'] == fetch_model.MODEL_REVISION
    assert fetch_model.verify_model_dir(tmp_config.paths.model_dir, expected) == []


_LIVE = r'''
import json, sys
from pathlib import Path
from hyak.model import fetch_model
method, staging = sys.argv[1], Path(sys.argv[2])
expected = fetch_model.expected_files()["config.json"]
if method == "https":
    path = fetch_model._https_download("config.json", staging, expected, None, print)
else:
    path = fetch_model._hub_download("config.json", staging, None, print)
print(json.dumps(fetch_model.verify_file(path, expected)))
'''


@pytest.mark.network
@pytest.mark.parametrize('method', ['https', 'hub'])
def test_live_download_of_config_json(method, tmp_path, repo_root):
    """Downloads the 635-byte config.json of the pinned revision; HF_HOME stays in tmp_path."""
    if method == 'hub':
        pytest.importorskip('huggingface_hub')
    staging = tmp_path / 'staging'
    staging.mkdir()
    env = {**os.environ, 'HF_HOME': str(tmp_path / 'hf-home'), 'PYTHONPATH': str(repo_root),
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    out = subprocess.run([sys.executable, '-c', _LIVE, method, str(staging)], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
