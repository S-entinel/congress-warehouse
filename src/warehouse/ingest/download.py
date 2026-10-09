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
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

USER_AGENT = "congress-warehouse/0.1 (personal data engineering project)"
CHUNK_SIZE = 1024 * 1024  # 1 MiB
REPORT_EVERY = 100 * 1024 * 1024  # print progress every 100 MiB
MAX_ATTEMPTS = 3
REQUIRED_KEYS = ("name", "url", "filename")

DEFAULT_REGISTRY = Path("sources.yml")
DEFAULT_DEST = Path("data/raw")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str


# ── registry ──────────────────────────────────────────────────


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
        if not isinstance(entry, dict):
            raise TypeError(f"{path}: source #{position} is not a mapping")
        missing = [key for key in REQUIRED_KEYS if not entry.get(key)]
        if missing:
            raise ValueError(f"{path}: source #{position} is missing {', '.join(missing)}")

        name, url, filename = entry["name"], entry["url"], entry["filename"]
        if Path(filename).name != filename:
            raise ValueError(f"{path}: filename '{filename}' must not contain a path")
        if name in names:
            raise ValueError(f"{path}: duplicate source name '{name}'")
        if filename in filenames:
            raise ValueError(f"{path}: duplicate filename '{filename}'")

        names.add(name)
        filenames.add(filename)
        sources.append(Source(name=name, url=url, filename=filename))

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


# ── downloading ───────────────────────────────────────────────


def is_up_to_date(target: Path, entry: dict | None) -> bool:
    """True if the file exists and its size matches what the manifest recorded."""
    return entry is not None and target.exists() and target.stat().st_size == entry["bytes"]


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

    print(f"download  {source.name}")
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    total_bytes = 0
    line_count = 0
    next_report = REPORT_EVERY
    started = time.monotonic()

    try:
        with client.stream("GET", source.url) as response:
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

    manifest[source.name] = {
        "url": source.url,
        "filename": source.filename,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "lines": line_count,
    }
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
            except (httpx.HTTPError, RuntimeError, OSError) as error:
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