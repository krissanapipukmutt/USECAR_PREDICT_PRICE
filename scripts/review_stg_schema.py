#!/usr/bin/env python3
"""Export SQL Server column metadata only; never reads table rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text


METADATA_SQL = """
SELECT
    ORDINAL_POSITION,
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = :schema_name
  AND TABLE_NAME = :table_name
ORDER BY ORDINAL_POSITION
""".strip()

FORBIDDEN_SQL_TOKENS = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE", "DENY",
)


def assert_metadata_query_is_read_only(statement: str) -> None:
    normalized = re.sub(r"\s+", " ", statement.strip()).upper()
    if not normalized.startswith("SELECT "):
        raise RuntimeError("Schema review query must start with SELECT.")
    if "FROM INFORMATION_SCHEMA.COLUMNS" not in normalized:
        raise RuntimeError("Schema review query must use INFORMATION_SCHEMA.COLUMNS.")
    for token in FORBIDDEN_SQL_TOKENS:
        if re.search(rf"\b{token}\b", normalized):
            raise RuntimeError(f"Forbidden SQL token in schema review query: {token}")


def required_environment() -> dict[str, str]:
    values = {
        "server": os.getenv("USED_CAR_DB_SERVER", "127.0.0.1"),
        "port": os.getenv("USED_CAR_DB_PORT", "1433"),
        "database": os.getenv("USED_CAR_DB_DATABASE", "USED_CAR_DB"),
        "user": os.getenv("USED_CAR_DB_USER", "sa"),
        "password": os.getenv("USED_CAR_DB_PASSWORD", ""),
        "driver": os.getenv("USED_CAR_DB_DRIVER", "ODBC Driver 18 for SQL Server"),
        "schema": os.getenv("USED_CAR_DB_SCHEMA", "dbo"),
        "table": os.getenv("USED_CAR_SOURCE_TABLE", "STG_USED_CAR"),
    }
    if not values["password"]:
        raise RuntimeError("USED_CAR_DB_PASSWORD is not set.")
    if not values["schema"] or not values["table"]:
        raise RuntimeError("Schema and table environment settings must not be empty.")
    return values


def build_engine(settings: dict[str, str]):
    odbc = (
        f"DRIVER={{{settings['driver']}}};"
        f"SERVER={settings['server']},{settings['port']};"
        f"DATABASE={settings['database']};"
        f"UID={settings['user']};"
        f"PWD={settings['password']};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return create_engine(
        "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc),
        pool_pre_ping=True,
        future=True,
    )


def default_output_path() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(__file__).resolve().parent.parent
    return project_dir / "output" / "analysis" / "schema_review" / f"actual_schema_{run_id}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read only dbo.STG_USED_CAR column metadata."
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_path = (args.output or default_output_path()).expanduser().resolve()
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing report: {output_path}")

    assert_metadata_query_is_read_only(METADATA_SQL)
    settings = required_environment()
    engine = None
    try:
        engine = build_engine(settings)
        with engine.connect() as connection:
            rows = connection.execute(
                text(METADATA_SQL),
                {"schema_name": settings["schema"], "table_name": settings["table"]},
            ).mappings().all()
    except Exception as exc:
        # Do not emit the driver message: it may repeat connection details.
        print(
            f"Schema metadata query failed ({type(exc).__name__}). "
            "No credentials or connection string were printed.",
            file=sys.stderr,
        )
        return 2
    finally:
        if engine is not None:
            engine.dispose()

    if not rows:
        print("No column metadata returned for the configured schema/table.", file=sys.stderr)
        return 3

    columns = [
        {
            "ordinal_position": int(row["ORDINAL_POSITION"]),
            "column_name": str(row["COLUMN_NAME"]),
            "sql_data_type": str(row["DATA_TYPE"]),
            "nullable": str(row["IS_NULLABLE"]),
        }
        for row in rows
    ]
    fingerprint_source = json.dumps(
        columns, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    report = {
        "report_type": "SQL_SERVER_COLUMN_METADATA_ONLY",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_name": settings["schema"],
        "table_name": settings["table"],
        "schema_fingerprint_sha256": hashlib.sha256(
            fingerprint_source.encode("utf-8")
        ).hexdigest(),
        "column_count": len(columns),
        "columns": columns,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Schema metadata report written: {output_path}")
    print(f"Columns: {len(columns)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
