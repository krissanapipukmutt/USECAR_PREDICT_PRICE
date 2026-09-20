#!/usr/bin/env python3
"""Validate a local metadata-only schema report against the feature registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SQL_TYPE_FAMILIES = {
    "text": {"char", "nchar", "varchar", "nvarchar", "text", "ntext", "xml"},
    "numeric": {
        "bigint", "int", "smallint", "tinyint", "decimal", "numeric",
        "float", "real", "money", "smallmoney",
    },
    "datetime": {
        "date", "datetime", "datetime2", "smalldatetime", "datetimeoffset", "time",
    },
    "boolean": {"bit"},
    "other": {"binary", "varbinary", "image", "uniqueidentifier", "sql_variant"},
}


def sql_schema_type(sql_data_type: str) -> str:
    value = str(sql_data_type).casefold()
    for family, members in SQL_TYPE_FAMILIES.items():
        if value in members:
            return family
    return "other"


def canonical_fingerprint(columns: list[dict]) -> str:
    payload = json.dumps(
        columns, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate(report: dict, registry: dict, registry_checksum: str) -> dict:
    issues: list[str] = []
    if report.get("report_type") != "SQL_SERVER_COLUMN_METADATA_ONLY":
        issues.append("Unexpected report_type.")
    if str(report.get("schema_name", "")).casefold() != "dbo":
        issues.append("Report schema_name is not dbo.")
    if str(report.get("table_name", "")).casefold() != "stg_used_car":
        issues.append("Report table_name is not STG_USED_CAR.")

    columns = report.get("columns")
    if not isinstance(columns, list):
        raise ValueError("Report columns must be a list.")
    if report.get("column_count") != len(columns):
        issues.append("Declared column_count differs from the columns list.")
    computed_fingerprint = canonical_fingerprint(columns)
    if report.get("schema_fingerprint_sha256") != computed_fingerprint:
        issues.append("Schema fingerprint does not match report columns.")

    actual: dict[str, dict] = {}
    ordinals: list[int] = []
    for column in columns:
        name = str(column.get("column_name", ""))
        key = name.casefold()
        if not name or key in actual:
            issues.append(f"Blank or duplicate report column: {name!r}.")
            continue
        actual[key] = column
        ordinals.append(int(column["ordinal_position"]))
    if ordinals != list(range(1, len(ordinals) + 1)):
        issues.append("Ordinal positions are not contiguous and ordered from 1.")

    entries = {str(x["name"]).casefold(): x for x in registry.get("columns", [])}
    new_columns = sorted(actual.keys() - entries.keys())
    missing_columns = sorted(entries.keys() - actual.keys())
    if new_columns:
        issues.append("New columns absent from registry: " + ", ".join(new_columns))
    if missing_columns:
        issues.append("Registry columns absent from report: " + ", ".join(missing_columns))

    comparisons: list[dict] = []
    approved_compatible: list[str] = []
    for key in sorted(actual.keys() & entries.keys(), key=lambda x: int(actual[x]["ordinal_position"])):
        column = actual[key]
        entry = entries[key]
        family = sql_schema_type(column["sql_data_type"])
        allowed = [str(x).casefold() for x in entry.get("allowed_schema_types", [])]
        compatible = family in allowed
        status = str(entry.get("status", "")).upper()
        authorized = status == "APPROVED" and compatible
        if status == "APPROVED" and compatible:
            approved_compatible.append(str(entry["name"]))
        if status == "APPROVED" and not compatible:
            issues.append(
                f"Approved feature {entry['name']} has incompatible SQL type "
                f"{column['sql_data_type']} ({family})."
            )
        comparisons.append({
            "ordinal_position": int(column["ordinal_position"]),
            "column_name": str(column["column_name"]),
            "sql_data_type": str(column["sql_data_type"]),
            "sql_schema_type": family,
            "nullable": str(column["nullable"]),
            "registry_status": status,
            "type_compatible": compatible,
            "predictor_authorized": authorized,
        })

    for mandatory, status in (("price", "TARGET"), ("pcs_date", "MANDATORY_METADATA")):
        if mandatory not in actual:
            issues.append(f"Mandatory column is missing: {mandatory}.")
        elif str(entries.get(mandatory, {}).get("status", "")).upper() != status:
            issues.append(f"Mandatory column {mandatory} is not classified as {status}.")

    return {
        "valid": not issues,
        "issues": issues,
        "registry_version": registry.get("registry_version"),
        "registry_checksum_sha256": registry_checksum,
        "initial_approval_status": registry.get("initial_approval_status"),
        "schema_fingerprint_sha256": computed_fingerprint,
        "new_columns": new_columns,
        "missing_columns": missing_columns,
        "approved_compatible_features": approved_compatible,
        "comparisons": comparisons,
    }


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_dir / "config" / "feature_approval_registry.json",
    )
    args = parser.parse_args()
    report_path = args.report.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes.decode("utf-8"))
    result = validate(
        report, registry, hashlib.sha256(registry_bytes).hexdigest()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
