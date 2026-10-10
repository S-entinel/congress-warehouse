"""Load the downloaded raw files into DuckDB and write each one out as Parquet."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from warehouse.ingest.download import (
    DEFAULT_DEST,
    DEFAULT_REGISTRY,
    MANIFEST_NAME,
    Source,
    load_registry,
    read_manifest,
)

DEFAULT_DB = Path("data/warehouse.duckdb")
DEFAULT_PARQUET = Path("data/parquet")
TABLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class LoadResult:
    name: str
    rows: int
    columns: int
    parquet_bytes: int
    skipped: bool


def sql_string(value: str | Path) -> str:
    """Quote a value as a SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def prepare_database(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw._load_log (
            name VARCHAR PRIMARY KEY,
            sha256 VARCHAR,
            row_count BIGINT,
            loaded_at VARCHAR
        )
        """
    )


def table_counts(con: duckdb.DuckDBPyConnection, name: str) -> tuple[int, int]:
    """Return (row count, column count) for a raw table."""
    rows = con.execute(f"SELECT count(*) FROM raw.{name}").fetchone()[0]
    columns = len(con.execute(f"DESCRIBE raw.{name}").fetchall())
    return rows, columns


def is_loaded(con: duckdb.DuckDBPyConnection, name: str, sha256: str) -> bool:
    """True if this exact file (by checksum) is already loaded into its table."""
    found = con.execute(
        "SELECT count(*) FROM raw._load_log WHERE name = ? AND sha256 = ?", [name, sha256]
    ).fetchone()[0]
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'raw' AND table_name = ?",
        [name],
    ).fetchone()[0]
    return bool(found and exists)


def load_one(
    con: duckdb.DuckDBPyConnection,
    source: Source,
    raw_dir: Path,
    parquet_dir: Path,
    manifest: dict[str, dict],
    force: bool = False,
) -> LoadResult:
    """Load one raw file as text into raw.<name> and write it to Parquet."""
    name = source.name
    if not TABLE_NAME.match(name):
        raise ValueError(f"'{name}' is not a valid table name (use lower case, digits, _)")

    entry = manifest.get(name)
    csv_path = raw_dir / source.filename
    if entry is None or not csv_path.exists():
        raise FileNotFoundError(f"{name}: {csv_path} not found; run the download step first")
    if csv_path.stat().st_size != entry["bytes"]:
        raise ValueError(f"{name}: {csv_path} does not match the manifest; download it again")

    parquet_path = parquet_dir / f"{name}.parquet"
    if not force and parquet_path.exists() and is_loaded(con, name, entry["sha256"]):
        rows, columns = table_counts(con, name)
        return LoadResult(name, rows, columns, parquet_path.stat().st_size, skipped=True)

    # Everything is read as text: the raw layer keeps the data exactly as published,
    # and typing is done later in the dbt staging models.
    con.execute(
        f"CREATE OR REPLACE TABLE raw.{name} AS "
        "SELECT * FROM read_csv(?, header = true, all_varchar = true)",
        [str(csv_path)],
    )

    temporary = parquet_path.with_name(parquet_path.name + ".tmp")
    con.execute(f"COPY raw.{name} TO {sql_string(temporary)} (FORMAT parquet, COMPRESSION zstd)")
    os.replace(temporary, parquet_path)

    rows, columns = table_counts(con, name)
    con.execute(
        "INSERT OR REPLACE INTO raw._load_log VALUES (?, ?, ?, ?)",
        [name, entry["sha256"], rows, datetime.now(UTC).isoformat(timespec="seconds")],
    )
    return LoadResult(name, rows, columns, parquet_path.stat().st_size, skipped=False)


def run(
    registry_path: Path,
    raw_dir: Path,
    parquet_dir: Path,
    db_path: Path,
    only: list[str] | None = None,
    force: bool = False,
) -> int:
    """Load every selected source. Returns a process exit code."""
    sources = load_registry(registry_path)

    if only:
        known = {source.name for source in sources}
        unknown = [name for name in only if name not in known]
        if unknown:
            print(f"error: unknown source(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(sorted(known))}", file=sys.stderr)
            return 2
        sources = [source for source in sources if source.name in only]

    manifest = read_manifest(raw_dir / MANIFEST_NAME)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    with duckdb.connect(str(db_path)) as con:
        prepare_database(con)
        for source in sources:
            started = time.monotonic()
            try:
                result = load_one(con, source, raw_dir, parquet_dir, manifest, force)
            except (duckdb.Error, OSError, ValueError) as error:
                print(f"FAILED    {source.name}: {error}", file=sys.stderr)
                failures.append(source.name)
                continue

            if result.skipped:
                print(f"skip      {result.name} (already loaded, {result.rows:,} rows)")
            else:
                size = result.parquet_bytes / 1024 / 1024
                elapsed = time.monotonic() - started
                print(
                    f"loaded    {result.name}: {result.rows:,} rows, {result.columns} columns, "
                    f"parquet {size:,.1f} MB in {elapsed:,.1f}s"
                )

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load raw files into DuckDB and Parquet.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--only", nargs="+", metavar="NAME", help="load only these sources")
    parser.add_argument("--force", action="store_true", help="reload even if unchanged")
    args = parser.parse_args(argv)
    try:
        return run(args.registry, args.raw, args.parquet, args.db, args.only, args.force)
    except (duckdb.Error, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())