"""Tests for the downloader."""

import hashlib
import json

import httpx
import pytest

from warehouse.ingest import download
from warehouse.ingest.download import Source

CONTENT = b"id,name\n1,alpha\n2,beta\n"


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=CONTENT)


@pytest.fixture
def source() -> Source:
    return Source(name="example", url="https://example.test/data.csv", filename="example.csv")


def write_registry(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# ── registry ──────────────────────────────────────────────────


def test_load_registry_reads_valid_file(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: a\n    url: https://x.test/a.csv\n    filename: a.csv\n"
        "  - name: b\n    url: https://x.test/b.csv\n    filename: b.csv\n",
    )
    sources = download.load_registry(registry)
    assert [s.name for s in sources] == ["a", "b"]
    assert sources[0].url == "https://x.test/a.csv"


def test_load_registry_allows_an_empty_list(tmp_path):
    # An empty list is valid: it simply means there is nothing to download.
    registry = write_registry(tmp_path / "sources.yml", "sources: []\n")
    assert download.load_registry(registry) == []


def test_load_registry_rejects_missing_key(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml", "sources:\n  - name: a\n    url: https://x.test/a.csv\n"
    )
    with pytest.raises(ValueError, match="missing filename"):
        download.load_registry(registry)


def test_load_registry_rejects_duplicate_names(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: a\n    url: https://x.test/a.csv\n    filename: a.csv\n"
        "  - name: a\n    url: https://x.test/b.csv\n    filename: b.csv\n",
    )
    with pytest.raises(ValueError, match="duplicate source name"):
        download.load_registry(registry)


def test_load_registry_rejects_filename_with_path(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n  - name: a\n    url: https://x.test/a.csv\n    filename: ../escape.csv\n",
    )
    with pytest.raises(ValueError, match="must not contain a path"):
        download.load_registry(registry)


def test_load_registry_rejects_wrong_shape(tmp_path):
    registry = write_registry(tmp_path / "sources.yml", "- just\n- a list\n")
    with pytest.raises(TypeError, match="top-level 'sources' list"):
        download.load_registry(registry)


# ── downloading ───────────────────────────────────────────────


def test_download_writes_file_and_manifest_entry(tmp_path, source):
    manifest: dict = {}
    with make_client(ok_handler) as client:
        downloaded = download.download_one(client, source, tmp_path, manifest)

    assert downloaded is True
    assert (tmp_path / "example.csv").read_bytes() == CONTENT
    entry = manifest["example"]
    assert entry["bytes"] == len(CONTENT)
    assert entry["sha256"] == hashlib.sha256(CONTENT).hexdigest()
    assert entry["lines"] == 3  # header plus two data rows
    assert entry["url"] == source.url
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_when_already_present(tmp_path, source):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=CONTENT)

    manifest: dict = {}
    with make_client(handler) as client:
        download.download_one(client, source, tmp_path, manifest)
        second = download.download_one(client, source, tmp_path, manifest)

    assert second is False
    assert len(calls) == 1


def test_download_redownloads_when_size_differs_from_manifest(tmp_path, source):
    manifest: dict = {}
    with make_client(ok_handler) as client:
        download.download_one(client, source, tmp_path, manifest)
        (tmp_path / "example.csv").write_bytes(b"tampered")  # size no longer matches
        again = download.download_one(client, source, tmp_path, manifest)

    assert again is True
    assert (tmp_path / "example.csv").read_bytes() == CONTENT


def test_force_downloads_again(tmp_path, source):
    manifest: dict = {}
    with make_client(ok_handler) as client:
        download.download_one(client, source, tmp_path, manifest)
        forced = download.download_one(client, source, tmp_path, manifest, force=True)
    assert forced is True


def test_failed_download_keeps_existing_file_and_leaves_no_part_file(tmp_path, source):
    target = tmp_path / "example.csv"
    target.write_bytes(b"previous good copy")

    def handler(request):
        return httpx.Response(404)

    with make_client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        download.download_one(client, source, tmp_path, {}, force=True)

    assert target.read_bytes() == b"previous good copy"
    assert not list(tmp_path.glob("*.part"))


def test_truncated_download_is_rejected(tmp_path, source):
    def handler(request):
        return httpx.Response(200, content=b"abc", headers={"Content-Length": "999"})

    manifest: dict = {}
    with make_client(handler) as client, pytest.raises(RuntimeError, match="expected 999"):
        download.download_one(client, source, tmp_path, manifest)

    assert not (tmp_path / "example.csv").exists()
    assert not list(tmp_path.glob("*.part"))
    assert manifest == {}


# ── manifest ──────────────────────────────────────────────────


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    assert download.read_manifest(path) == {}
    download.write_manifest(path, {"a": {"bytes": 1}})
    assert download.read_manifest(path) == {"a": {"bytes": 1}}


def test_corrupt_manifest_gives_clear_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        download.read_manifest(path)


# ── run ───────────────────────────────────────────────────────


def test_run_downloads_everything_and_saves_manifest(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: a\n    url: https://x.test/a.csv\n    filename: a.csv\n"
        "  - name: b\n    url: https://x.test/b.csv\n    filename: b.csv\n",
    )
    dest = tmp_path / "raw"
    with make_client(ok_handler) as client:
        code = download.run(registry, dest, client=client)

    assert code == 0
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"a", "b"}


def test_run_only_unknown_source_is_an_error(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n  - name: a\n    url: https://x.test/a.csv\n    filename: a.csv\n",
    )
    with make_client(ok_handler) as client:
        code = download.run(registry, tmp_path / "raw", only=["nope"], client=client)
    assert code == 2


def test_run_reports_failure_but_still_downloads_the_rest(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: bad\n    url: https://x.test/missing.csv\n    filename: bad.csv\n"
        "  - name: good\n    url: https://x.test/good.csv\n    filename: good.csv\n",
    )

    def handler(request):
        if "missing" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=CONTENT)

    dest = tmp_path / "raw"
    with make_client(handler) as client:
        code = download.run(registry, dest, client=client)

    assert code == 1
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"good"}
    assert (dest / "good.csv").exists()
    assert not (dest / "bad.csv").exists()


def test_main_reports_a_missing_registry_cleanly(tmp_path, capsys):
    code = download.main(
        ["--registry", str(tmp_path / "nope.yml"), "--dest", str(tmp_path / "raw")]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err