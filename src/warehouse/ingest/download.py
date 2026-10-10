"""Download the raw source files listed in sources.yml and record a manifest.

Files are written to data/raw/. A file is skipped if it is already present and
its size matches the manifest, so repeated runs are cheap. Downloads go to a
temporary ".part" file first and are moved into place only once complete, so an
interrupted run can never leave a half-written file that looks finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

USER_AGENT = "congress-warehouse/0.1 (personal data engineering project)"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
CHUNK_SIZE = 1024 * 1024  # 1 MiB
REPORT_EVERY = 100 * 1024 * 1024  # print progress every 100 MiB
MAX_ATTEMPTS = 3

DEFAULT_REGISTRY = Path("sources.yml")
DEFAULT_DEST = Path("data/raw")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    url: str | None = None
    doi: str | None = None
    file_pattern: str | None = None
    manual_page: str | None = None
    instructions: str | None = None


class ManualDownloadRequired(Exception):
    """A source must be downloaded by hand and the file is not in place yet."""


@dataclass(frozen=True)
class ResolvedFile:
    """A Dataverse file that has been located through the dataset's file list."""

    url: str
    filename: str
    file_id: int
    dataset_version: str


# ── registry ──────────────────────────────────────────────────


def _parse_source(path: Path, position: int, entry: object) -> Source:
    if not isinstance(entry, dict):
        raise TypeError(f"{path}: source #{position} is not a mapping")

    where = f"{path}: source #{position}"
    missing = [key for key in ("name", "filename") if not entry.get(key)]
    if missing:
        raise ValueError(f"{where} is missing {', '.join(missing)}")

    url, doi, pattern = entry.get("url"), entry.get("doi"), entry.get("file_pattern")
    manual_page = entry.get("manual_page")
    kinds = [
        kind for kind, value in (("url", url), ("doi", doi), ("manual_page", manual_page)) if value
    ]
    if len(kinds) > 1:
        raise ValueError(f"{where} must have only one of 'url', 'doi' or 'manual_page'")
    if not kinds:
        raise ValueError(f"{where} needs one of 'url', 'doi' or 'manual_page'")
    if doi and not pattern:
        raise ValueError(f"{where} has a 'doi' but no 'file_pattern'")
    if pattern:
        if not doi:
            raise ValueError(f"{where} has a 'file_pattern' but no 'doi'")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{where} has an invalid file_pattern: {error}") from error

    filename = entry["filename"]
    if Path(filename).name != filename:
        raise ValueError(f"{where}: filename '{filename}' must not contain a path")

    return Source(
        name=entry["name"],
        filename=filename,
        url=url,
        doi=doi,
        file_pattern=pattern,
        manual_page=manual_page,
        instructions=entry.get("instructions"),
    )


def load_registry(path: Path) -> list[Source]:
    """Read and validate the source registry."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise TypeError(f"{path}: expected a top-level 'sources' list")

    sources: list[Source] = []
    names: set[str] = set()
    filenames: set[str] = set()

    for position, entry in enumerate(data["sources"], start=1):
        source = _parse_source(path, position, entry)
        if source.name in names:
            raise ValueError(f"{path}: duplicate source name '{source.name}'")
        if source.filename in filenames:
            raise ValueError(f"{path}: duplicate filename '{source.filename}'")
        names.add(source.name)
        filenames.add(source.filename)
        sources.append(source)

    return sources


# ── manifest ──────────────────────────────────────────────────


def read_manifest(path: Path) -> dict[str, dict]:
    """Return the manifest, or an empty one if it does not exist yet."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{path} is not valid JSON ({error}). Delete it and run again with --force."
        ) from error


def write_manifest(path: Path, manifest: dict[str, dict]) -> None:
    """Write the manifest atomically."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


# ── Dataverse ─────────────────────────────────────────────────


def resolve_dataverse(client: httpx.Client, source: Source) -> ResolvedFile:
    """Find the one file in a Dataverse dataset that matches the source's pattern."""
    doi, file_pattern = source.doi, source.file_pattern
    if doi is None or file_pattern is None:
        raise ValueError(f"{source.name}: not a Dataverse source")

    api_url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId=doi:{doi}"
    response = client.get(api_url)
    response.raise_for_status()

    try:
        version = response.json()["data"]["latestVersion"]
        files = [entry["dataFile"] for entry in version["files"]]
        dataset_version = f"{version['versionNumber']}.{version['versionMinorNumber']}"
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{source.name}: unexpected response from Dataverse for doi:{doi}"
        ) from error

    pattern = re.compile(file_pattern)
    matches = [item for item in files if pattern.search(item["filename"])]
    if len(matches) != 1:
        available = ", ".join(item["filename"] for item in files)
        raise ValueError(
            f"{source.name}: expected exactly one file matching '{file_pattern}' "
            f"in doi:{doi}, found {len(matches)} (files: {available})"
        )

    chosen = matches[0]
    url = f"{DATAVERSE_BASE}/api/access/datafile/{chosen['id']}"
    if chosen.get("originalFileFormat"):
        # Dataverse converts uploaded tables to its own tab-delimited format;
        # this asks for the file as originally uploaded.
        url += "?format=original"

    return ResolvedFile(
        url=url,
        filename=chosen["filename"],
        file_id=chosen["id"],
        dataset_version=dataset_version,
    )


# ── downloading ───────────────────────────────────────────────


def is_up_to_date(target: Path, entry: dict | None) -> bool:
    """True if the file exists and its size matches what the manifest recorded."""
    return entry is not None and target.exists() and target.stat().st_size == entry["bytes"]


def register_manual(source: Source, target: Path, manifest: dict[str, dict]) -> bool:
    """Record a hand-downloaded file in the manifest, or explain how to get it."""
    if not target.exists():
        message = [
            f"expected the file at {target}",
            f"  download it by hand from {source.manual_page}",
        ]
        if source.instructions:
            message.append("  " + " ".join(source.instructions.split()))
        raise ManualDownloadRequired("\n".join(message))

    digest = hashlib.sha256()
    total_bytes = 0
    line_count = 0
    with target.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            total_bytes += len(chunk)
            line_count += chunk.count(b"\n")
    if total_bytes == 0:
        raise ValueError(f"{target} is empty")

    manifest[source.name] = {
        "manual": True,
        "source_page": source.manual_page,
        "filename": source.filename,
        "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "lines": line_count,
    }
    print(f"register  {source.name}: {total_bytes / 1024 / 1024:,.1f} MB (downloaded by hand)")
    return True


def download_one(
    client: httpx.Client,
    source: Source,
    dest_dir: Path,
    manifest: dict[str, dict],
    force: bool = False,
) -> bool:
    """Download one source. Returns True if downloaded, False if skipped.

    Updates the manifest in memory; the caller is responsible for saving it.
    On any failure the temporary file is removed and any existing copy of the
    target file is left untouched.
    """
    target = dest_dir / source.filename
    if not force and is_up_to_date(target, manifest.get(source.name)):
        print(f"skip      {source.name} (already downloaded)")
        return False

    if source.manual_page is not None:
        return register_manual(source, target, manifest)

    print(f"download  {source.name}")
    resolved = resolve_dataverse(client, source) if source.doi else None
    download_url = resolved.url if resolved else source.url
    if download_url is None:
        raise ValueError(f"{source.name}: no download URL")

    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    total_bytes = 0
    line_count = 0
    next_report = REPORT_EVERY
    started = time.monotonic()

    try:
        with client.stream("GET", download_url) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    handle.write(chunk)
                    digest.update(chunk)
                    total_bytes += len(chunk)
                    line_count += chunk.count(b"\n")
                    if total_bytes >= next_report:
                        print(f"          {total_bytes / 1024 / 1024:,.0f} MB so far")
                        next_report += REPORT_EVERY

            # Detect truncated downloads. Content-Length describes the encoded body,
            # so it can only be compared when the server did not compress it.
            expected = response.headers.get("content-length")
            compressed = bool(response.headers.get("content-encoding"))
            if expected is not None and not compressed and total_bytes != int(expected):
                raise RuntimeError(
                    f"{source.name}: received {total_bytes} bytes, expected {expected}"
                )
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    os.replace(part, target)

    entry: dict = {
        "url": download_url,
        "filename": source.filename,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "lines": line_count,
    }
    if resolved is not None:
        entry.update(
            {
                "doi": source.doi,
                "dataset_version": resolved.dataset_version,
                "source_file_id": resolved.file_id,
                "source_filename": resolved.filename,
            }
        )
    manifest[source.name] = entry

    elapsed = time.monotonic() - started
    print(f"done      {source.name}: {total_bytes / 1024 / 1024:,.1f} MB in {elapsed:,.1f}s")
    return True


def _should_retry(error: Exception) -> bool:
    """Retry network problems and server errors, but not client errors such as 404."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code >= 500
    return isinstance(error, (httpx.TransportError, RuntimeError))


def download_with_retries(
    client: httpx.Client,
    source: Source,
    dest_dir: Path,
    manifest: dict[str, dict],
    force: bool = False,
) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return download_one(client, source, dest_dir, manifest, force)
        except Exception as error:
            if attempt == MAX_ATTEMPTS or not _should_retry(error):
                raise
            wait = 2**attempt
            print(f"retry     {source.name} in {wait}s ({error})")
            time.sleep(wait)
    raise AssertionError("unreachable")


# ── command line ──────────────────────────────────────────────


def run(
    registry_path: Path,
    dest_dir: Path,
    only: list[str] | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
) -> int:
    """Download every selected source. Returns a process exit code."""
    sources = load_registry(registry_path)

    if only:
        known = {source.name for source in sources}
        unknown = [name for name in only if name not in known]
        if unknown:
            print(f"error: unknown source(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(sorted(known))}", file=sys.stderr)
            return 2
        sources = [source for source in sources if source.name in only]

    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / MANIFEST_NAME
    manifest = read_manifest(manifest_path)

    owns_client = client is None
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=120.0),
        )

    failures: list[str] = []
    try:
        for source in sources:
            try:
                if download_with_retries(client, source, dest_dir, manifest, force):
                    write_manifest(manifest_path, manifest)  # save progress after each file
            except (
                httpx.HTTPError,
                RuntimeError,
                OSError,
                ValueError,
                ManualDownloadRequired,
            ) as error:
                print(f"FAILED    {source.name}: {error}", file=sys.stderr)
                failures.append(source.name)
    finally:
        if owns_client:
            client.close()

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw source files.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--only", nargs="+", metavar="NAME", help="download only these sources")
    parser.add_argument("--force", action="store_true", help="download even if already present")
    args = parser.parse_args(argv)
    try:
        return run(args.registry, args.dest, args.only, args.force)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())