import hashlib
from types import SimpleNamespace

import pytest
import requests

from ctsc.scripts.download_unified38_public_data import _download
from ctsc.scripts import fetch_revision_stage8a_point_robot as fisor


class Response:
    def __init__(self, data):
        self.data = data
        self.headers = {"content-length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def download(path, expected):
    return _download("https://upstream.example/file.hdf5", path, chunk_bytes=128,
                     retries=0, retry_wait_seconds=0, timeout_seconds=1,
                     expected_sha256=expected, recorded_size=4)


def test_dsrl_download_and_cached_file_are_hash_checked(tmp_path, monkeypatch):
    payload = b"data"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload))
    path = tmp_path / "task.hdf5"
    assert download(path, expected)["status"] == "downloaded"
    assert path.read_bytes() == payload
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: pytest.fail("Verified cache should avoid network access"))
    assert download(path, expected)["status"] == "cached"


def test_dsrl_rejects_same_size_wrong_hash_and_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "task.hdf5"
    path.write_bytes(b"keep")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(b"xxxx"))
    result = download(path, hashlib.sha256(b"data").hexdigest())
    assert result["status"] == "failed"
    assert "SHA256" in result["error"]
    assert path.read_bytes() == b"keep"


def test_fisor_rejects_wrong_hash_before_accepting_download(tmp_path, monkeypatch):
    class Session:
        trust_env = True

        def get(self, url, **kwargs):
            assert self.trust_env
            if "dirents" in url:
                return SimpleNamespace(raise_for_status=lambda: None,
                                       json=lambda: {"dirent_list": [{"file_name": fisor.NAME, "size": 4}]})
            return Response(b"xxxx")

        def head(self, *args, **kwargs):
            return Response(b"xxxx")

    monkeypatch.setattr(fisor, "EXPECTED_BYTES", 4)
    monkeypatch.setattr(fisor.requests, "Session", Session)
    monkeypatch.setattr(fisor.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    destination = tmp_path / "custom" / fisor.NAME
    with pytest.raises(RuntimeError, match="SHA256"):
        fisor.fetch(destination)
    assert not destination.exists()
