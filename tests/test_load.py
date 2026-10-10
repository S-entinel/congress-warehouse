"""Tests for the loader, using small fake raw files."""

import hashlib
import json

import duckdb
import pytest

from warehouse import load
from warehouse.ingest.download import MANIFEST_NAME, Source, write_manifest

PLAIN = b"id,code,amount\n1,007,1.0\n2,NA,\n"
TRICKY = (
    b"candidate,party\n"
    b'"BUBAR, BENJAMIN """"BEN""""",PROHIBITION\n'  # comma plus doubled-quote nickname
    b'"CARTER, JIMMY",DEMOCRAT\n'
    b'"multi\nline",OTHER\n'  # a quoted line break is one row, not two
)


@pytest.fixture
def workspace(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    return {"raw": raw, "parquet": tmp_path / "parquet", "db": tmp_path / "wh.duckdb"}


def register(workspace, name, content, filename=None):
    """Write a raw file and a manifest entry for it, as the downloader would."""
    filename = filename or f"{name}.csv"
    (workspace["raw"] / filename).write_bytes(content)
    manifest_path = workspace["raw"] / MANIFEST_NAME
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[name] = {
        "filename": filename,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    write_manifest(manifest_path, manifest)
    return Source(name=name, filename=filename, url="https://x.test/f.csv")


def load_source(workspace, source, force=False):
    manifest = load.read_manifest(workspace["raw"] / MANIFEST_NAME)
    workspace["parquet"].mkdir(exist_ok=True)
    with duckdb.connect(str(workspace["db"])) as con:
        load.prepare_database(con)
        return load.load_one(con, source, workspace["raw"], workspace["parquet"], manifest, force)


def query(workspace, sql):
    with duckdb.connect(str(workspace["db"])) as con:
        return con.execute(sql).fetchall()


def test_sql_string_escapes_quotes():
    assert load.sql_string("it's") == "'it''s'"


def test_loads_everything_as_text(workspace):
    source = register(workspace, "plain", PLAIN)
    result = load_source(workspace, source)

    assert (result.rows, result.columns, result.skipped) == (2, 3, False)
    rows = query(workspace, "SELECT id, code, amount FROM raw.plain ORDER BY id")
    # Leading zeros and the literal text NA survive; nothing is typed or nulled yet.
    assert rows == [("1", "007", "1.0"), ("2", "NA", None)]
    types = query(
        workspace,
        "SELECT DISTINCT data_type FROM information_schema.columns WHERE table_name = 'plain'",
    )
    assert types == [("VARCHAR",)]


def test_writes_a_readable_parquet_file(workspace):
    source = register(workspace, "plain", PLAIN)
    result = load_source(workspace, source)

    parquet = workspace["parquet"] / "plain.parquet"
    assert result.parquet_bytes == parquet.stat().st_size > 0
    assert not list(workspace["parquet"].glob("*.tmp"))
    count = duckdb.sql(f"SELECT count(*) FROM read_parquet({load.sql_string(parquet)})")
    assert count.fetchone()[0] == 2


def test_quoted_commas_quotes_and_line_breaks_are_kept(workspace):
    source = register(workspace, "tricky", TRICKY)
    result = load_source(workspace, source)

    assert result.rows == 3  # three records, although the file has four line breaks
    rows = dict(query(workspace, "SELECT candidate, party FROM raw.tricky"))
    assert rows['BUBAR, BENJAMIN ""BEN""'] == "PROHIBITION"
    assert rows["multi\nline"] == "OTHER"


def test_unchanged_file_is_skipped_and_changed_file_is_reloaded(workspace):
    source = register(workspace, "plain", PLAIN)
    load_source(workspace, source)

    again = load_source(workspace, source)
    assert again.skipped is True
    assert again.rows == 2

    bigger = PLAIN + b"3,009,2.5\n"
    source = register(workspace, "plain", bigger)
    reloaded = load_source(workspace, source)
    assert reloaded.skipped is False
    assert reloaded.rows == 3


def test_force_reloads_an_unchanged_file(workspace):
    source = register(workspace, "plain", PLAIN)
    load_source(workspace, source)
    forced = load_source(workspace, source, force=True)
    assert forced.skipped is False


def test_load_log_records_checksum_and_row_count(workspace):
    source = register(workspace, "plain", PLAIN)
    load_source(workspace, source)

    ((name, sha256, row_count),) = query(
        workspace, "SELECT name, sha256, row_count FROM raw._load_log"
    )
    assert name == "plain"
    assert sha256 == hashlib.sha256(PLAIN).hexdigest()
    assert row_count == 2


def test_missing_file_is_an_error(workspace):
    source = register(workspace, "plain", PLAIN)
    (workspace["raw"] / "plain.csv").unlink()
    with pytest.raises(FileNotFoundError, match="download step"):
        load_source(workspace, source)


def test_file_that_disagrees_with_the_manifest_is_an_error(workspace):
    source = register(workspace, "plain", PLAIN)
    (workspace["raw"] / "plain.csv").write_bytes(PLAIN + b"extra\n")
    with pytest.raises(ValueError, match="does not match the manifest"):
        load_source(workspace, source)


def test_source_without_a_manifest_entry_is_an_error(workspace):
    register(workspace, "plain", PLAIN)
    stranger = Source(name="stranger", filename="stranger.csv", url="https://x.test/s.csv")
    with pytest.raises(FileNotFoundError):
        load_source(workspace, stranger)


@pytest.mark.parametrize("name", ["Bad", "1abc", "has-dash", "x; DROP TABLE y"])
def test_invalid_table_names_are_rejected(workspace, name):
    source = Source(name=name, filename="f.csv", url="https://x.test/f.csv")
    with pytest.raises(ValueError, match="not a valid table name"):
        load_source(workspace, source)


def test_paths_containing_an_apostrophe_work(tmp_path):
    raw = tmp_path / "o'brien" / "raw"
    raw.mkdir(parents=True)
    workspace = {"raw": raw, "parquet": tmp_path / "o'brien" / "pq", "db": tmp_path / "wh.duckdb"}
    source = register(workspace, "plain", PLAIN)
    assert load_source(workspace, source).rows == 2


def write_registry(path, names):
    body = "sources:\n" + "".join(
        f"  - name: {n}\n    url: https://x.test/{n}.csv\n    filename: {n}.csv\n" for n in names
    )
    path.write_text(body, encoding="utf-8")
    return path


def test_run_loads_what_it_can_and_reports_failures(workspace, tmp_path, capsys):
    register(workspace, "good", PLAIN)
    registry = write_registry(tmp_path / "sources.yml", ["good", "absent"])

    code = load.run(registry, workspace["raw"], workspace["parquet"], workspace["db"])

    assert code == 1
    assert query(workspace, "SELECT count(*) FROM raw.good") == [(2,)]
    captured = capsys.readouterr()
    assert "loaded    good: 2 rows" in captured.out
    assert "FAILED    absent" in captured.err


def test_run_with_everything_present_succeeds_and_rerun_skips(workspace, tmp_path, capsys):
    register(workspace, "one", PLAIN)
    register(workspace, "two", TRICKY)
    registry = write_registry(tmp_path / "sources.yml", ["one", "two"])
    args = (registry, workspace["raw"], workspace["parquet"], workspace["db"])

    assert load.run(*args) == 0
    capsys.readouterr()
    assert load.run(*args) == 0
    assert capsys.readouterr().out.count("skip") == 2


def test_run_rejects_an_unknown_source(workspace, tmp_path):
    register(workspace, "one", PLAIN)
    registry = write_registry(tmp_path / "sources.yml", ["one"])
    code = load.run(
        registry, workspace["raw"], workspace["parquet"], workspace["db"], only=["nope"]
    )
    assert code == 2