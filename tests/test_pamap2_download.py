"""Offline regression tests for interrupted large UCI PAMAP2 ZIP downloads."""
import http.client
import io
import zipfile
import pytest
from scripts import preprocess_pamap2 as pamap


def tiny_zip():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("PAMAP2_Dataset/Protocol/subject101.dat", "0 1 2\n")
    return out.getvalue()


class FakeResponse:
    def __init__(self, payload, status, headers, *, drop=False):
        self.payload, self.status, self.headers, self.drop = payload, status, headers, drop
        self.pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, amount):
        if self.drop and self.pos:
            raise http.client.IncompleteRead(b"")
        if self.pos >= len(self.payload):
            return b""
        limit = min(len(self.payload), self.pos + (17 if self.drop else amount))
        blob = self.payload[self.pos:limit]
        self.pos = limit
        return blob


def test_retries_exact_byte_range(monkeypatch, tmp_path):
    data, attempts = tiny_zip(), []
    def fake_urlopen(req, timeout):
        assert timeout > 0
        start = req.get_header("Range")
        attempts.append(start)
        if not start:
            return FakeResponse(data, 200, {"Content-Length": str(len(data))}, drop=True)
        assert start == "bytes=17-"
        return FakeResponse(data[17:], 206, {
            "Content-Range": f"bytes 17-{len(data)-1}/{len(data)}"
        })
    monkeypatch.setattr(pamap, "urlopen", fake_urlopen)
    monkeypatch.setattr(pamap.time, "sleep", lambda _: None)
    archive = pamap._download_pamap2_archive(tmp_path / "sample.zip", max_attempts=3)
    assert archive.read_bytes() == data
    assert attempts == [None, "bytes=17-"]
    assert not (tmp_path / "sample.zip.part").exists()


def test_range_ignored_does_not_corrupt_archive(monkeypatch, tmp_path):
    data, reqs = tiny_zip(), []
    def fake_urlopen(req, timeout):
        reqs.append(req.get_header("Range"))
        if len(reqs) > 1:
            return FakeResponse(data, 200, {"Content-Length": str(len(data))})
        return FakeResponse(data, 200, {"Content-Length": str(len(data))}, drop=True)
    monkeypatch.setattr(pamap, "urlopen", fake_urlopen)
    monkeypatch.setattr(pamap.time, "sleep", lambda _: None)
    archive = pamap._download_pamap2_archive(tmp_path / "sample.zip", max_attempts=3)
    assert archive.read_bytes() == data
    assert reqs == [None, "bytes=17-"]


def test_invalid_zip_never_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(pamap, "urlopen", lambda req, timeout:
                        FakeResponse(b"not a zip", 200, {"Content-Length": "9"}))
    with pytest.raises(RuntimeError, match="Cannot download valid"):
        pamap._download_pamap2_archive(tmp_path / "sample.zip", max_attempts=1)
    assert not (tmp_path / "sample.zip").exists()


def test_atomic_extraction_and_reuse(monkeypatch, tmp_path):
    data, count = tiny_zip(), [0]
    def fake_urlopen(req, timeout):
        count[0] += 1
        return FakeResponse(data, 200, {"Content-Length": str(len(data))})
    monkeypatch.setattr(pamap, "urlopen", fake_urlopen)
    ds = pamap._ensure_pamap2(tmp_path)
    assert (ds / "Protocol" / "subject101.dat").read_text() == "0 1 2\n"
    assert not (tmp_path / "PAMAP2_Dataset.zip").exists()
    assert pamap._ensure_pamap2(tmp_path) == ds
    assert count == [1]
