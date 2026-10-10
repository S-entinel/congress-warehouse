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


# ── Dataverse sources ─────────────────────────────────────────

DOI = "10.7910/DVN/EXAMPLE"
HOUSE_PATTERN = r"^\d{4}-\d{4}-house\."


def dataverse_payload(files, version=(15, 0)):
    return {
        "status": "OK",
        "data": {
            "latestVersion": {
                "versionNumber": version[0],
                "versionMinorNumber": version[1],
                "files": [{"dataFile": item} for item in files],
            }
        },
    }


TABULAR_FILES = [
    {"id": 11, "filename": "1976-2024-house.tab", "originalFileFormat": "text/csv"},
    {"id": 12, "filename": "codebook-us-house.md"},
    {"id": 13, "filename": "sources-house.tab", "originalFileFormat": "text/csv"},
]


def dataverse_handler(files, requests_seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if requests_seen is not None:
            requests_seen.append(str(request.url))
        if "/api/datasets/" in request.url.path:
            return httpx.Response(200, json=dataverse_payload(files))
        return httpx.Response(200, content=CONTENT)

    return handler


@pytest.fixture
def dataverse_source() -> Source:
    return Source(name="house", filename="house.csv", doi=DOI, file_pattern=HOUSE_PATTERN)


def test_registry_accepts_a_dataverse_source(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: house\n"
        f"    doi: {DOI}\n"
        f"    file_pattern: '{HOUSE_PATTERN}'\n"
        "    filename: house.csv\n",
    )
    (source,) = download.load_registry(registry)
    assert source.doi == DOI
    assert source.file_pattern == HOUSE_PATTERN
    assert source.url is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("    url: https://x.test/a.csv\n    doi: 10.1/x\n    file_pattern: a\n", "only one of"),
        ("", "needs one of"),
        ("    doi: 10.1/x\n", "no 'file_pattern'"),
        ("    url: https://x.test/a.csv\n    file_pattern: a\n", "no 'doi'"),
        ("    doi: 10.1/x\n    file_pattern: '['\n", "invalid file_pattern"),
    ],
)
def test_registry_rejects_bad_source_combinations(tmp_path, fields, message):
    registry = write_registry(
        tmp_path / "sources.yml", f"sources:\n  - name: a\n    filename: a.csv\n{fields}"
    )
    with pytest.raises(ValueError, match=message):
        download.load_registry(registry)


def test_dataverse_download_picks_the_matching_file_as_original_csv(tmp_path, dataverse_source):
    seen: list[str] = []
    manifest: dict = {}
    with make_client(dataverse_handler(TABULAR_FILES, seen)) as client:
        downloaded = download.download_one(client, dataverse_source, tmp_path, manifest)

    assert downloaded is True
    assert (tmp_path / "house.csv").read_bytes() == CONTENT
    assert seen[-1].endswith("/api/access/datafile/11?format=original")
    entry = manifest["house"]
    assert entry["doi"] == DOI
    assert entry["dataset_version"] == "15.0"
    assert entry["source_file_id"] == 11
    assert entry["source_filename"] == "1976-2024-house.tab"


def test_dataverse_plain_files_are_downloaded_without_the_original_option(
    tmp_path, dataverse_source
):
    files = [{"id": 21, "filename": "1976-2024-house.csv"}]  # not converted by Dataverse
    seen: list[str] = []
    with make_client(dataverse_handler(files, seen)) as client:
        download.download_one(client, dataverse_source, tmp_path, {})
    assert seen[-1].endswith("/api/access/datafile/21")


def test_dataverse_no_matching_file_is_an_error(tmp_path, dataverse_source):
    files = [{"id": 12, "filename": "codebook-us-house.md"}]
    with (
        make_client(dataverse_handler(files)) as client,
        pytest.raises(ValueError, match="found 0"),
    ):
        download.download_one(client, dataverse_source, tmp_path, {})
    assert not (tmp_path / "house.csv").exists()


def test_dataverse_several_matching_files_is_an_error(tmp_path, dataverse_source):
    files = [
        {"id": 1, "filename": "1976-2022-house.tab"},
        {"id": 2, "filename": "1976-2024-house.tab"},
    ]
    with (
        make_client(dataverse_handler(files)) as client,
        pytest.raises(ValueError, match="found 2"),
    ):
        download.download_one(client, dataverse_source, tmp_path, {})


def test_dataverse_unexpected_response_is_an_error(tmp_path, dataverse_source):
    def handler(request):
        return httpx.Response(200, json={"status": "ERROR", "message": "nope"})

    with make_client(handler) as client, pytest.raises(ValueError, match="unexpected response"):
        download.download_one(client, dataverse_source, tmp_path, {})


def test_dataverse_skip_makes_no_requests(tmp_path, dataverse_source):
    seen: list[str] = []
    manifest: dict = {}
    with make_client(dataverse_handler(TABULAR_FILES, seen)) as client:
        download.download_one(client, dataverse_source, tmp_path, manifest)
        before = len(seen)
        skipped = download.download_one(client, dataverse_source, tmp_path, manifest)
    assert skipped is False
    assert len(seen) == before


def test_run_continues_after_a_dataverse_resolution_failure(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: house\n"
        f"    doi: {DOI}\n"
        f"    file_pattern: '{HOUSE_PATTERN}'\n"
        "    filename: house.csv\n"
        "  - name: plain\n    url: https://x.test/plain.csv\n    filename: plain.csv\n",
    )
    dest = tmp_path / "raw"
    # No matching file in the dataset, so "house" fails but "plain" must still download.
    with make_client(dataverse_handler([{"id": 12, "filename": "codebook.md"}])) as client:
        code = download.run(registry, dest, client=client)

    assert code == 1
    assert (dest / "plain.csv").exists()
    assert not (dest / "house.csv").exists()


# ── manual sources ────────────────────────────────────────────

MANUAL_PAGE = "https://doi.org/10.1234/example"


@pytest.fixture
def manual_source() -> Source:
    return Source(
        name="manual",
        filename="manual.tsv",
        manual_page=MANUAL_PAGE,
        instructions="Fill in the form,\n  then save the file.",
    )


def test_registry_accepts_a_manual_source(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: manual\n"
        f"    manual_page: {MANUAL_PAGE}\n"
        "    filename: manual.tsv\n"
        "    instructions: Save it by hand.\n",
    )
    (source,) = download.load_registry(registry)
    assert source.manual_page == MANUAL_PAGE
    assert source.instructions == "Save it by hand."
    assert source.url is None


def test_registry_rejects_manual_page_combined_with_url(tmp_path):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n  - name: a\n    filename: a.csv\n"
        f"    url: https://x.test/a.csv\n    manual_page: {MANUAL_PAGE}\n",
    )
    with pytest.raises(ValueError, match="only one of"):
        download.load_registry(registry)


def test_manual_file_missing_explains_how_to_get_it(tmp_path, manual_source):
    manifest: dict = {}
    with (
        make_client(ok_handler) as client,
        pytest.raises(download.ManualDownloadRequired) as caught,
    ):
        download.download_one(client, manual_source, tmp_path, manifest)

    message = str(caught.value)
    assert str(tmp_path / "manual.tsv") in message
    assert MANUAL_PAGE in message
    assert "Fill in the form, then save the file." in message  # whitespace tidied
    assert manifest == {}


def test_manual_file_is_registered_in_the_manifest(tmp_path, manual_source):
    (tmp_path / "manual.tsv").write_bytes(CONTENT)
    manifest: dict = {}
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    with make_client(handler) as client:
        registered = download.download_one(client, manual_source, tmp_path, manifest)

    assert registered is True
    assert calls == []  # never touches the network
    entry = manifest["manual"]
    assert entry["manual"] is True
    assert entry["source_page"] == MANUAL_PAGE
    assert entry["bytes"] == len(CONTENT)
    assert entry["sha256"] == hashlib.sha256(CONTENT).hexdigest()
    assert entry["lines"] == 3


def test_manual_file_already_registered_is_skipped(tmp_path, manual_source):
    (tmp_path / "manual.tsv").write_bytes(CONTENT)
    manifest: dict = {}
    with make_client(ok_handler) as client:
        download.download_one(client, manual_source, tmp_path, manifest)
        again = download.download_one(client, manual_source, tmp_path, manifest)
    assert again is False


def test_manual_file_that_changed_is_registered_again(tmp_path, manual_source):
    target = tmp_path / "manual.tsv"
    target.write_bytes(CONTENT)
    manifest: dict = {}
    with make_client(ok_handler) as client:
        download.download_one(client, manual_source, tmp_path, manifest)
        target.write_bytes(CONTENT + b"3,gamma\n")
        again = download.download_one(client, manual_source, tmp_path, manifest)
    assert again is True
    assert manifest["manual"]["lines"] == 4


def test_empty_manual_file_is_rejected(tmp_path, manual_source):
    (tmp_path / "manual.tsv").write_bytes(b"")
    with make_client(ok_handler) as client, pytest.raises(ValueError, match="is empty"):
        download.download_one(client, manual_source, tmp_path, {})


def test_run_reports_a_missing_manual_file_but_downloads_the_rest(tmp_path, capsys):
    registry = write_registry(
        tmp_path / "sources.yml",
        "sources:\n"
        "  - name: manual\n"
        f"    manual_page: {MANUAL_PAGE}\n"
        "    filename: manual.tsv\n"
        "  - name: plain\n    url: https://x.test/plain.csv\n    filename: plain.csv\n",
    )
    dest = tmp_path / "raw"
    with make_client(ok_handler) as client:
        code = download.run(registry, dest, client=client)

    assert code == 1
    assert (dest / "plain.csv").exists()
    assert MANUAL_PAGE in capsys.readouterr().err