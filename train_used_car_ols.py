#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =============================================================================
# 1) ENVIRONMENT CONFIG
# =============================================================================

DB_SERVER = os.getenv("USED_CAR_DB_SERVER", "127.0.0.1")
DB_PORT = int(os.getenv("USED_CAR_DB_PORT", "1433"))
DB_DATABASE = os.getenv("USED_CAR_DB_DATABASE", "USED_CAR_DB")
DB_USER = os.getenv("USED_CAR_DB_USER", "sa")
DB_PASSWORD = os.getenv(
    "USED_CAR_DB_PASSWORD",
    "",
)

DB_DRIVER = os.getenv(
    "USED_CAR_DB_DRIVER",
    "ODBC Driver 18 for SQL Server",
)

DB_SCHEMA = os.getenv("USED_CAR_DB_SCHEMA", "dbo")
SOURCE_TABLE = os.getenv("USED_CAR_SOURCE_TABLE", "STG_USED_CAR")

OUTPUT_DIR = Path(
    os.getenv(
        "USED_CAR_OUTPUT_DIR",
        "/Users/krissanap/Document/KMUTT/USECAR_PREDICT_PRICE/output/train",
    )
)
FEATURE_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "config" / "feature_approval_registry.json"
)

# =============================================================================
# 2) MODEL CONFIG
# =============================================================================

TARGET_COLUMN = "price"
PCS_DATE_COLUMN = "PCS_DATE"
MANDATORY_SOURCE_COLUMNS = (TARGET_COLUMN, PCS_DATE_COLUMN)

TOP_N_MODELS = 3
P_VALUE_THRESHOLD = 0.05

# Two-sided confidence intervals for the fitted OLS coefficients.
# Independent of P_VALUE_THRESHOLD (which controls feature selection).
CONFIDENCE_LEVEL = 0.95

CV_FOLDS = 5
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.20
HOLDOUT_FRACTION_TOLERANCE = 0.05
TARGET_PRICE_MIN_EXCLUSIVE_THB = 1_000.0
# Fixed business bands used only to balance deterministic splits and report error.
PRICE_BAND_EDGES = (-np.inf, 300_000, 500_000, 750_000, 1_000_000,
                    1_500_000, 2_000_000, 3_000_000, 5_000_000, np.inf)

MAX_CANDIDATES = 8  # Keep the original baseline candidate feature combinations.
# Add matched encoding experiments for the same source-feature sets (no new DB columns).
MAX_EXPANDED_EXPERIMENTS = 2
CANDIDATE_SUBSET_FRACTIONS = (0.90, 0.80, 0.70, 0.60, 0.50)
CANDIDATES_PER_FRACTION = 1

MAX_MISSING_RATIO = 0.95
MAX_CATEGORICAL_LEVELS = 60
MAX_CATEGORICAL_UNIQUE_RATIO = 0.50
MIN_CATEGORY_COUNT = 20

# Brand/model are explicit vehicle attributes, not technical IDs. Allow them
# through the initial schema checks even when distinct values exceed 60.
CORE_VEHICLE_FEATURES = ("brand", "model")
OPTIONAL_VEHICLE_FEATURES = ("sub_model",)
# Hard caps prevent thousands of dummy columns and prohibit blind 2,000-level OLS.
# The most frequent levels are retained; all remaining levels map to __OTHER__.
MAX_LEVELS_BY_SOURCE = {"brand": 25, "model": 35, "sub_model": 15}
MIN_COUNT_BY_SOURCE = {"brand": 20, "model": 30, "sub_model": 40}

# V3 experiment: compare against V2 without changing the target, other source
# columns, CV folds, output schema or prediction formula. These are TRAIN-only
# frequency thresholds, not prices used to decide which categories survive.
# Category levels for each fold are learned from that fold's TRAIN partition.
ENCODING_PROFILES = {
    "BASELINE": {
        "max_levels": MAX_LEVELS_BY_SOURCE,
        "min_count": MIN_COUNT_BY_SOURCE,
    },
    "EXPANDED_BRAND_MODEL": {
        "max_levels": {**MAX_LEVELS_BY_SOURCE, "brand": 80, "model": 110},
        "min_count": {**MIN_COUNT_BY_SOURCE, "brand": 5, "model": 10},
    },
}
HIGH_PRICE_MIN_THB = 3_000_000  # Reporting only; not used to select levels.

# Price-quality screening is FLAG-ONLY: NEVER remove rows on this basis.
# Review suspect prices with source listings before approving exclusion rules.
OUTLIER_FLAG_ONLY = True
OUTLIER_MIN_GROUP_SIZE = 15
OUTLIER_LOW_PRICE_REVIEW_THB = 1_000.0
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
OUTLIER_PRICE_RATIO_THRESHOLD = 3.0
OUTLIER_LOG_MAD_FLOOR = 0.10  # Prevent tiny/zero MAD from excessive flags.
# Broad brand+model groups are used only if their training years span <= 3.
OUTLIER_MAX_BROAD_GROUP_YEAR_SPAN = 3
OUTLIER_REPORT_BASE_DIR = OUTPUT_DIR.parent / "analysis"

# V4 target cohort. This uses the known target during training/evaluation only.
# It must never be used to gate prediction inputs, where actual price is unknown.
PRICE_FILTER_MODE = "V4_ELIGIBLE"
PRICE_FILTER_MAX_THB = TARGET_PRICE_MIN_EXCLUSIVE_THB
# Specify confirmed erroneous listings ONLY after checking the source listing.
# Strings preserve any leading zeros in listing_id.
CONFIRMED_BAD_LISTING_IDS: List[str] = []
PRICE_SELECTION_REPORT_BASE_DIR = OUTPUT_DIR.parent / "analysis"
# Even when brand/model are available, their statistical significance is still
# tested by the same source-feature group P-value rule as other predictors.
# Optional sub_model is evaluated in separate candidates, never forced.

NUMERIC_COERCE_THRESHOLD = 0.98

REQUIRE_SINGLE_PCS_DATE = True
MIN_SOURCE_FEATURES = 1
MODEL_NAME_PREFIX = "USED_CAR_OLS"


# =============================================================================
# 3) DYNAMIC SCHEMA / EXCLUSION CONFIG
# =============================================================================

EXCLUDE_COLUMNS_EXACT = {
    PCS_DATE_COLUMN.lower(),
    TARGET_COLUMN.lower(),
    "listing_id",
    "raw_price",
    "raw_mileage",
    "source_url",
    "seller_url",
    "image_url",
    "description",
    "scraped_at",
    "dealer_slug",
}

EXCLUDE_COLUMN_PATTERNS = [
    r"(^|_)url($|_)",
    r"(^|_)image($|_)",
    r"(^|_)description($|_)",
    r"(^|_)scraped(_|$)",
    r"(^|_)created(_|$)",
    r"(^|_)updated(_|$)",
    r"^raw_",
]

FORCE_INCLUDE_COLUMNS: List[str] = []
FORCE_EXCLUDE_COLUMNS: List[str] = []


# =============================================================================
# 4) OUTPUT DATA DICTIONARY
# =============================================================================

RESULT_COLUMNS = [
    "MODEL_ID",
    "MODEL_NAME",
    "TARGET_NAME",
    "TRAIN_PCS_DATE",
    "TRAIN_DATE",
    "N_OBSERVATION",
    "R_SQUARED",
    "ADJ_R_SQUARED",
    "MAE",
    "RMSE",
    "F_STATISTIC",
    "F_P_VALUE",
    "P_VALUE_THRESHOLD",
    "CONFIDENCE_LEVEL",
    "ACTIVE_FLAG",
    "APPROVED_BY",
    "APPROVED_DATE",
    "PCS_DATE",
]

COEFFICIENT_COLUMNS = [
    "MODEL_ID",
    "FEATURE_SEQ",
    "SOURCE_COLUMN",
    "ORIGINAL_VALUE",
    "FEATURE_NAME",
    "FEATURE_TYPE",
    "COEFFICIENT",
    "P_VALUE",
    "SOURCE_FEATURE_P_VALUE",
    "CI_LOWER",
    "CI_UPPER",
    "IS_REFERENCE",
    "PCS_DATE",
]


@dataclass
class PreprocessorState:
    numeric_features: List[str]
    categorical_features: List[str]
    numeric_medians: Dict[str, float]
    category_levels: Dict[str, List[str]]
    reference_categories: Dict[str, str]
    encoded_feature_names: List[str]
    feature_groups: Dict[str, List[str]]
    numeric_like_features: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateExperiment:
    source_features: List[str]
    encoding_profile: str


@dataclass(frozen=True)
class SchemaGateResult:
    registry_version: str
    registry_checksum: str
    schema_fingerprint: str
    initial_approval_status: str
    approved_features: List[str]
    missing_approved_features: List[str]
    incompatible_approved_features: List[str]
    records: List[dict]


@dataclass
class FinalCandidate:
    candidate_id: int
    initial_source_features: List[str]
    selected_source_features: List[str]
    selected_encoded_features: List[str]
    preprocessor: PreprocessorState
    ols_result: object
    cv_rmse: float
    cv_mae: float
    r_squared: float
    adj_r_squared: float
    f_statistic: Optional[float]
    f_p_value: Optional[float]
    source_feature_p_values: Dict[str, float]
    encoding_profile: str = "BASELINE"
    cv_high_price_mae: Optional[float] = None
    cv_negative_rate: Optional[float] = None
    cv_model_other_rate: Optional[float] = None
    cv_eligible_rmse: Optional[float] = None
    cv_eligible_mae: Optional[float] = None
    cv_eligible_count: int = 0
    cv_fold_rmse_mean: Optional[float] = None
    cv_fold_rmse_std: Optional[float] = None
    cv_fold_mae_mean: Optional[float] = None
    cv_fold_mae_std: Optional[float] = None
    cv_price_band_metrics: List[dict] = field(default_factory=list)
    cv_other_count: int = 0
    cv_other_denominator: int = 0


@dataclass
class CVEvaluation:
    rmse: float
    mae: float
    count: int
    fold_rmse_mean: float
    fold_rmse_std: float
    fold_mae_mean: float
    fold_mae_std: float
    high_price_mae: Optional[float]
    negative_rate: float
    negative_count: int
    other_rate: Optional[float]
    other_count: int
    other_denominator: int
    price_band_metrics: List[dict]
    oof_predictions: pd.DataFrame


def build_engine() -> Engine:
    odbc = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER},{DB_PORT};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    connection_url = "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc)
    return create_engine(connection_url, pool_pre_ping=True, future=True)


def quote_sql_identifier(name: str) -> str:
    return "[" + name.replace("]", "]]") + "]"


def load_source_data(engine: Engine) -> pd.DataFrame:
    full_name = (
        f"{quote_sql_identifier(DB_SCHEMA)}."
        f"{quote_sql_identifier(SOURCE_TABLE)}"
    )
    sql = text(f"SELECT * FROM {full_name}")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        raise RuntimeError(f"{DB_SCHEMA}.{SOURCE_TABLE} contains no rows.")
    return df


def find_column_case_insensitive(
    columns: Iterable[str],
    target_name: str,
) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in columns}
    return lookup.get(target_name.lower())


def normalize_required_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    by_normalized_name: Dict[str, List[str]] = {}
    for column in df.columns:
        by_normalized_name.setdefault(str(column).casefold(), []).append(str(column))
    ambiguous = {
        name: originals
        for name, originals in by_normalized_name.items()
        if len(originals) > 1
    }
    if ambiguous:
        raise RuntimeError(
            "Source schema contains duplicate case-insensitive column names: "
            + ", ".join(sorted(ambiguous))
        )

    target_actual = find_column_case_insensitive(df.columns, TARGET_COLUMN)
    pcs_actual = find_column_case_insensitive(df.columns, PCS_DATE_COLUMN)

    if target_actual is None:
        raise RuntimeError(
            f"Required target column '{TARGET_COLUMN}' was not found in "
            f"{DB_SCHEMA}.{SOURCE_TABLE}. Current columns: {list(df.columns)}"
        )

    if pcs_actual is None:
        raise RuntimeError(
            f"Required snapshot column '{PCS_DATE_COLUMN}' was not found in "
            f"{DB_SCHEMA}.{SOURCE_TABLE}. Current columns: {list(df.columns)}"
        )

    rename_map = {}
    if target_actual != TARGET_COLUMN:
        rename_map[target_actual] = TARGET_COLUMN
    if pcs_actual != PCS_DATE_COLUMN:
        rename_map[pcs_actual] = PCS_DATE_COLUMN

    return df.rename(columns=rename_map)


FEATURE_REGISTRY_STATUSES = {
    "APPROVED", "EXCLUDED", "PENDING_REVIEW",
    "MANDATORY_METADATA", "TARGET",
}


def load_feature_registry(path: Path = FEATURE_REGISTRY_PATH) -> Tuple[dict, str]:
    if not path.is_file():
        raise RuntimeError(f"Feature approval registry was not found: {path}")
    raw = path.read_bytes()
    try:
        registry = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Feature approval registry is invalid JSON: {path}") from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("columns"), list):
        raise RuntimeError("Feature approval registry must contain a columns list.")
    if not str(registry.get("registry_version", "")).strip():
        raise RuntimeError("Feature approval registry requires registry_version.")
    names: Dict[str, str] = {}
    for entry in registry["columns"]:
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise RuntimeError("Every feature registry entry requires a name.")
        name = str(entry["name"])
        key = name.casefold()
        if key in names:
            raise RuntimeError(
                f"Feature registry has duplicate case-insensitive names: {names[key]}, {name}"
            )
        names[key] = name
        status = str(entry.get("status", "")).upper()
        if status not in FEATURE_REGISTRY_STATUSES:
            raise RuntimeError(f"Feature registry status is invalid for {name}: {status}")
        allowed = entry.get("allowed_schema_types", [])
        if not isinstance(allowed, list) or not allowed:
            raise RuntimeError(f"Feature registry requires allowed_schema_types for {name}.")
    required_registry_status = {
        TARGET_COLUMN.casefold(): "TARGET",
        PCS_DATE_COLUMN.casefold(): "MANDATORY_METADATA",
    }
    by_name = {str(x["name"]).casefold(): str(x["status"]).upper()
               for x in registry["columns"]}
    for name, expected_status in required_registry_status.items():
        if by_name.get(name) != expected_status:
            raise RuntimeError(
                f"Feature registry must classify {name} as {expected_status}."
            )
    checksum = hashlib.sha256(raw).hexdigest()
    return registry, checksum


def schema_type_of(series: pd.Series) -> str:
    """Classify storage dtype without learning from values or Holdout targets."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "text"
    return "other"


def _possible_target_leakage(column_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.casefold()).strip("_")
    tokens = set(normalized.split("_"))
    return bool(tokens & {
        "price", "sold", "sale", "target", "actual", "revenue",
        "transaction", "closed", "final",
    })


def evaluate_feature_approval_gate(df: pd.DataFrame, registry: dict,
                                   registry_checksum: str) -> SchemaGateResult:
    entries = {str(x["name"]).casefold(): x for x in registry["columns"]}
    actual = {str(column).casefold(): str(column) for column in df.columns}
    schema_payload = [
        {"name": actual[key], "pandas_dtype": str(df[actual[key]].dtype),
         "schema_type": schema_type_of(df[actual[key]])}
        for key in sorted(actual)
    ]
    schema_fingerprint = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    records: List[dict] = []
    approved: List[str] = []
    incompatible: List[str] = []

    for key in sorted(actual):
        name = actual[key]
        observed_type = schema_type_of(df[name])
        entry = entries.get(key)
        if entry is None:
            leakage_risk = _possible_target_leakage(name)
            records.append({
                "column_name": name,
                "registry_status": "PENDING_REVIEW",
                "schema_state": "NEW",
                "observed_schema_type": observed_type,
                "allowed_schema_types": [],
                "training_authorized": False,
                "predict_availability": "NOT_REVIEWED",
                "target_leakage_risk": leakage_risk,
                "reason": (
                    "Unregistered column; possible target/post-outcome leakage."
                    if leakage_risk else
                    "Unregistered column; owner review is required."
                ),
            })
            continue
        status = str(entry["status"]).upper()
        allowed_types = [str(x).lower() for x in entry["allowed_schema_types"]]
        compatible = observed_type in allowed_types
        authorized = status == "APPROVED" and compatible
        if authorized:
            approved.append(name)
        elif status == "APPROVED" and not compatible:
            incompatible.append(name)
        records.append({
            "column_name": name,
            "registry_status": status,
            "schema_state": "UNCHANGED" if compatible else "TYPE_CHANGED",
            "observed_schema_type": observed_type,
            "allowed_schema_types": allowed_types,
            "training_authorized": authorized,
            "predict_availability": entry.get("predict_availability", "NOT_REVIEWED"),
            "target_leakage_risk": bool(entry.get("target_leakage_risk", False)),
            "reason": (
                str(entry.get("reason", ""))
                if compatible else
                "Observed schema type is not allowed by the registry; review required."
            ),
        })

    missing_approved: List[str] = []
    for key, entry in sorted(entries.items()):
        if key in actual:
            continue
        status = str(entry["status"]).upper()
        if status == "APPROVED":
            missing_approved.append(str(entry["name"]))
        records.append({
            "column_name": str(entry["name"]),
            "registry_status": status,
            "schema_state": "MISSING",
            "observed_schema_type": None,
            "allowed_schema_types": entry["allowed_schema_types"],
            "training_authorized": False,
            "predict_availability": entry.get("predict_availability", "NOT_REVIEWED"),
            "target_leakage_risk": bool(entry.get("target_leakage_risk", False)),
            "reason": "Column listed in the registry is absent from this source schema.",
        })

    return SchemaGateResult(
        registry_version=str(registry["registry_version"]),
        registry_checksum=registry_checksum,
        schema_fingerprint=schema_fingerprint,
        initial_approval_status=str(
            registry.get("initial_approval_status", "PENDING_OWNER_APPROVAL")
        ).upper(),
        approved_features=approved,
        missing_approved_features=missing_approved,
        incompatible_approved_features=incompatible,
        records=records,
    )


def assert_feature_registry_approved(gate: SchemaGateResult) -> None:
    if gate.initial_approval_status != "APPROVED":
        raise RuntimeError(
            "Initial Feature Approval Registry is not owner-approved; "
            "Full Training is blocked. Review the Schema Review Report."
        )
    if not gate.approved_features:
        raise RuntimeError("No present, type-compatible APPROVED predictors remain.")


def export_schema_review_report(path: Path, gate: SchemaGateResult,
                                pcs_date: pd.Timestamp, run_id: str) -> None:
    payload = {
        "report_version": "1.0",
        "run_id": run_id,
        "pcs_date": pcs_date.date().isoformat(),
        "registry_version": gate.registry_version,
        "registry_checksum": gate.registry_checksum,
        "registry_initial_approval_status": gate.initial_approval_status,
        "schema_fingerprint": gate.schema_fingerprint,
        "approved_features_present_and_compatible": gate.approved_features,
        "missing_approved_features": gate.missing_approved_features,
        "incompatible_approved_features": gate.incompatible_approved_features,
        "columns": gate.records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_single_pcs_date(df: pd.DataFrame) -> pd.Timestamp:
    pcs = pd.to_datetime(df[PCS_DATE_COLUMN], errors="coerce").dt.normalize()
    unique_dates = sorted(pcs.dropna().unique())

    if not unique_dates:
        raise RuntimeError(f"{PCS_DATE_COLUMN} contains no valid date.")

    if REQUIRE_SINGLE_PCS_DATE and len(unique_dates) != 1:
        raise RuntimeError(
            f"{DB_SCHEMA}.{SOURCE_TABLE} is expected to be a latest snapshot, "
            f"but {len(unique_dates)} PCS_DATE values were found: "
            f"{[pd.Timestamp(x).date().isoformat() for x in unique_dates[:20]]}"
        )

    return pd.Timestamp(unique_dates[-1])


def build_group_price_outlier_report(source_df: pd.DataFrame) -> pd.DataFrame:
    """Audit the STG snapshot without modifying it or choosing training rows.

    Reference-group statistics use positive prices > OUTLIER_LOW_PRICE_REVIEW_THB
    to avoid 1-THB placeholders distorting medians. This report is exploratory,
    not a validation metric or proof that any particular listing is wrong.
    """
    if not OUTLIER_FLAG_ONLY:
        raise ValueError("Only FLAG-ONLY price screening is implemented.")
    if OUTLIER_MIN_GROUP_SIZE < 3:
        raise ValueError("OUTLIER_MIN_GROUP_SIZE must be at least 3.")
    if OUTLIER_PRICE_RATIO_THRESHOLD <= 1 or OUTLIER_LOG_MAD_FLOOR <= 0:
        raise ValueError("Outlier ratio and MAD floor must be positive/valid.")

    n = len(source_df)
    report = pd.DataFrame(index=source_df.index)
    report["SOURCE_ROW_NUMBER"] = np.arange(1, n + 1, dtype=int)

    metadata = ["listing_id", "brand", "model", "sub_model",
                "model_year", "raw_price", "title", "source_url"]
    for label in metadata:
        actual = find_column_case_insensitive(source_df.columns, label)
        report[label] = source_df[actual] if actual is not None else pd.NA

    # Preserve the originally supplied target and DO NOT mutate source_df.
    raw_price = source_df[TARGET_COLUMN]
    if pd.api.types.is_object_dtype(raw_price) or pd.api.types.is_string_dtype(raw_price):
        raw_price = raw_price.astype("string").str.replace(",", "", regex=False).str.strip()
    price = pd.to_numeric(raw_price, errors="coerce").astype("float64")
    price = price.replace([np.inf, -np.inf], np.nan)
    report["price"] = price
    report["STATUS"] = "INSUFFICIENT_GROUP_DATA"
    report["REASON"] = "NO_RELIABLE_COMPARISON_GROUP"
    report["GROUP_USED"] = pd.NA
    report["GROUP_SIZE"] = pd.Series(pd.NA, index=report.index, dtype="Int64")
    for label in ["GROUP_MEDIAN_PRICE", "GROUP_LOG_MAD", "PRICE_TO_MEDIAN_RATIO",
                  "MODIFIED_Z_SCORE"]:
        report[label] = np.nan

    missing_or_invalid = price.isna() | (price <= 0)
    report.loc[missing_or_invalid, "STATUS"] = "SUSPECTED_OUTLIER"
    report.loc[missing_or_invalid, "REASON"] = "INVALID_OR_NONPOSITIVE_PRICE"
    low = (~missing_or_invalid) & (price <= OUTLIER_LOW_PRICE_REVIEW_THB)
    report.loc[low, "STATUS"] = "SUSPECTED_OUTLIER"
    report.loc[low, "REASON"] = "VERY_LOW_PRICE_REVIEW"

    keys = pd.DataFrame(index=source_df.index)
    for key in ("brand", "model", "sub_model"):
        actual = find_column_case_insensitive(source_df.columns, key)
        if actual is None:
            keys[key] = pd.Series(pd.NA, index=source_df.index, dtype="string")
        else:
            values = source_df[actual].astype("string").str.strip().str.casefold()
            keys[key] = values.mask(values.isin(["", "nan", "none", "null", "__missing__"]))
    year_actual = find_column_case_insensitive(source_df.columns, "model_year")
    if year_actual is not None:
        keys["model_year"] = pd.to_numeric(source_df[year_actual], errors="coerce")
    else:
        keys["model_year"] = np.nan

    remaining = (~missing_or_invalid) & (~low)
    if not remaining.any():
        return report

    ref = keys.copy()
    ref["__log_price"] = np.log(price.where(price > OUTLIER_LOW_PRICE_REVIEW_THB))
    ref = ref.loc[ref["__log_price"].notna()].copy()

    levels = [
        ("BRAND_MODEL_SUB_MODEL_YEAR", ["brand", "model", "sub_model", "model_year"]),
        ("BRAND_MODEL_YEAR", ["brand", "model", "model_year"]),
        ("BRAND_MODEL", ["brand", "model"]),
    ]
    for label, group_cols in levels:
        if not remaining.any() or ref.empty:
            break
        # Missing group fields cannot form a reliable peer cohort.
        eligible_ref = ref.dropna(subset=group_cols).copy()
        if eligible_ref.empty:
            continue
        grouped = eligible_ref.groupby(group_cols, dropna=True, sort=False)
        stats = grouped["__log_price"].agg(
            GROUP_SIZE="size", GROUP_LOG_MEDIAN="median"
        ).reset_index()
        eligible_ref = eligible_ref.merge(stats, on=group_cols, how="left", validate="many_to_one")
        eligible_ref["__abs_dev"] = (
            eligible_ref["__log_price"] - eligible_ref["GROUP_LOG_MEDIAN"]
        ).abs()
        deviations = (eligible_ref.groupby(group_cols, dropna=True, sort=False)["__abs_dev"]
                      .median().rename("GROUP_LOG_MAD").reset_index())
        stats = stats.merge(deviations, on=group_cols, how="left", validate="one_to_one")
        if label == "BRAND_MODEL":
            # A broad model group may include cars of radically different ages.
            year_span = (eligible_ref.groupby(group_cols, dropna=True, sort=False)["model_year"]
                         .agg(lambda x: x.max() - x.min() if x.notna().any() else np.nan)
                         .rename("YEAR_SPAN").reset_index())
            stats = stats.merge(year_span, on=group_cols, how="left", validate="one_to_one")
        candidates = keys.loc[remaining, group_cols].copy()
        candidates["__row_index"] = candidates.index
        candidates = candidates.dropna(subset=group_cols)
        if candidates.empty:
            continue
        candidates = candidates.merge(stats, on=group_cols, how="left", validate="many_to_one")
        valid = candidates["GROUP_SIZE"].ge(OUTLIER_MIN_GROUP_SIZE).fillna(False)
        if label == "BRAND_MODEL":
            valid &= candidates["YEAR_SPAN"].le(OUTLIER_MAX_BROAD_GROUP_YEAR_SPAN).fillna(False)
        matched = candidates.loc[valid].set_index("__row_index")
        if matched.empty:
            continue
        ix = matched.index
        median_log = matched["GROUP_LOG_MEDIAN"].to_numpy(float)
        observed = price.loc[ix].to_numpy(float)
        ratio = np.exp(np.log(observed) - median_log)
        z_score = (0.6745 * (np.log(observed) - median_log) /
                   np.maximum(matched["GROUP_LOG_MAD"].to_numpy(float), OUTLIER_LOG_MAD_FLOOR))
        suspicious_high = ((z_score > OUTLIER_MODIFIED_Z_THRESHOLD) &
                           (ratio >= OUTLIER_PRICE_RATIO_THRESHOLD))
        suspicious_low = ((z_score < -OUTLIER_MODIFIED_Z_THRESHOLD) &
                          (ratio <= 1.0 / OUTLIER_PRICE_RATIO_THRESHOLD))

        report.loc[ix, "STATUS"] = "NORMAL"
        report.loc[ix, "REASON"] = "WITHIN_PEER_GROUP_THRESHOLD"
        report.loc[ix, "GROUP_USED"] = label
        report.loc[ix, "GROUP_SIZE"] = matched["GROUP_SIZE"].astype("int64")
        report.loc[ix, "GROUP_MEDIAN_PRICE"] = np.exp(median_log)
        report.loc[ix, "GROUP_LOG_MAD"] = matched["GROUP_LOG_MAD"].to_numpy(float)
        report.loc[ix, "PRICE_TO_MEDIAN_RATIO"] = ratio
        report.loc[ix, "MODIFIED_Z_SCORE"] = z_score
        if suspicious_high.any():
            hi_ix = ix[suspicious_high]
            report.loc[hi_ix, "STATUS"] = "SUSPECTED_OUTLIER"
            report.loc[hi_ix, "REASON"] = "PRICE_HIGH_VS_PEERS"
        if suspicious_low.any():
            lo_ix = ix[suspicious_low]
            report.loc[lo_ix, "STATUS"] = "SUSPECTED_OUTLIER"
            report.loc[lo_ix, "REASON"] = "PRICE_LOW_VS_PEERS"
        remaining.loc[ix] = False

    return report


def export_outlier_flag_only_reports(
    source_df: pd.DataFrame, pcs_date: pd.Timestamp, run_id: str
) -> Tuple[Path, Path]:
    """Write audit files outside train/. Return all-rows and flagged-only paths."""
    report = build_group_price_outlier_report(source_df)
    output_dir = OUTLIER_REPORT_BASE_DIR / pcs_date.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / f"price_quality_all_{run_id}.csv"
    flagged_path = output_dir / f"price_outliers_for_review_{run_id}.csv"
    report.to_csv(all_path, index=False, encoding="utf-8-sig")
    flagged = report.loc[report["STATUS"].eq("SUSPECTED_OUTLIER")].copy()
    flagged.to_csv(flagged_path, index=False, encoding="utf-8-sig")
    print("[PRICE QUALITY] FLAG-ONLY screening (no training rows removed):")
    for status, count in report["STATUS"].value_counts(dropna=False).items():
        print(f"    {status}: {count:,}")
    print("[PRICE QUALITY] Reasons:")
    for reason, count in report["REASON"].value_counts(dropna=False).items():
        print(f"    {reason}: {count:,}")
    print(f"[PRICE QUALITY] Full audit: {all_path}")
    print(f"[PRICE QUALITY] Review list: {flagged_path}")
    return all_path, flagged_path


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    before = len(df)
    mask = (
        df[TARGET_COLUMN].notna()
        & np.isfinite(df[TARGET_COLUMN].astype(float))
        & (df[TARGET_COLUMN] > 0)
    )
    df = df.loc[mask].copy()

    removed = before - len(df)
    if removed:
        print(
            f"[INFO] Removed {removed:,} rows with invalid/non-positive "
            f"{TARGET_COLUMN}."
        )

    if len(df) < max(30, CV_FOLDS * 5):
        raise RuntimeError(
            f"Only {len(df):,} usable rows remain after target cleaning."
        )

    return df


def select_eligible_cohort(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select the V4 training/evaluation cohort from already-clean target rows."""
    prices = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    eligible = prices.gt(TARGET_PRICE_MIN_EXCLUSIVE_THB).fillna(False)
    reasons = pd.Series("", index=df.index, dtype="string")
    reasons.loc[~eligible] = (
        f"PRICE_LE_{TARGET_PRICE_MIN_EXCLUSIVE_THB:g}_THB_NOT_ELIGIBLE"
    )
    selected = df.loc[eligible].copy()
    if len(selected) < max(30, CV_FOLDS * 5):
        raise RuntimeError(
            f"Only {len(selected):,} rows have {TARGET_COLUMN} > "
            f"{TARGET_PRICE_MIN_EXCLUSIVE_THB:,.0f}."
        )
    return selected, reasons


def price_filter_exclusion_reason(df: pd.DataFrame) -> pd.Series:
    """Backward-compatible wrapper for the permanent V4 cohort rule."""
    return select_eligible_cohort(df)[1]


def select_train_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Backward-compatible wrapper; V4 applies the rule before any split."""
    return select_eligible_cohort(df)


def price_filter_eligible_validation_mask(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df[TARGET_COLUMN], errors="coerce").gt(
        TARGET_PRICE_MIN_EXCLUSIVE_THB
    ).fillna(False)


def _normalize_identity(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.casefold()
    return values.mask(values.isna() | values.isin(["", "nan", "none", "null"]))


def build_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Build transitive identity groups without exposing the identities."""
    n = len(df)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for requested in ("listing_id", "source_url"):
        actual = find_column_case_insensitive(df.columns, requested)
        if actual is None:
            continue
        first_seen: Dict[str, int] = {}
        for pos, value in enumerate(_normalize_identity(df[actual]).tolist()):
            if pd.isna(value):
                continue
            key = str(value)
            if key in first_seen:
                union(pos, first_seen[key])
            else:
                first_seen[key] = pos

    roots = [find(i) for i in range(n)]
    canonical = {root: seq for seq, root in enumerate(sorted(set(roots)))}
    return pd.Series(
        [f"G{canonical[root]:08d}" for root in roots],
        index=df.index,
        dtype="string",
        name="DUPLICATE_GROUP",
    )


def make_price_bands(prices: pd.Series, n_splits: int) -> pd.Series:
    bands = pd.cut(
        pd.to_numeric(prices, errors="raise"),
        bins=PRICE_BAND_EDGES,
        right=True,
        include_lowest=True,
        labels=False,
    ).astype("int64")
    counts = bands.value_counts()
    if counts.empty or int(counts.min()) < n_splits:
        return pd.Series(0, index=prices.index, dtype="int64")
    return bands


def fixed_price_bands(prices: pd.Series) -> pd.Series:
    """Business price bands for reporting, independent of split feasibility."""
    return pd.cut(
        pd.to_numeric(prices, errors="raise"),
        bins=PRICE_BAND_EDGES,
        right=True,
        include_lowest=True,
    )


def _assignment_checksum(assignments: Sequence[Tuple[int, str]]) -> str:
    payload = "\n".join(f"{row}:{label}" for row, label in assignments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_development_holdout(
    df: pd.DataFrame,
    groups: pd.Series,
    test_size: float = HOLDOUT_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not math.isclose(test_size, 0.20):
        raise ValueError("V4 requires a 20% holdout.")
    if len(df) != len(groups) or len(df) < CV_FOLDS * 2:
        raise ValueError("Invalid rows/groups for Development/Holdout split.")
    group_values = groups.loc[df.index].astype(str).to_numpy()
    if len(set(group_values)) < CV_FOLDS:
        raise ValueError("At least five duplicate groups are required.")
    bands = make_price_bands(df[TARGET_COLUMN], CV_FOLDS).to_numpy()
    has_duplicate_groups = len(set(group_values)) < len(group_values)
    if has_duplicate_groups:
        splitter = StratifiedGroupKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=random_state
        )
        candidates = list(splitter.split(df, bands, group_values))
        method = "STRATIFIED_GROUP_5_WAY_CLOSEST_HOLDOUT"
    else:
        splitter = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=random_state
        )
        candidates = list(splitter.split(df, bands))
        method = "STRATIFIED_5_WAY_CLOSEST_HOLDOUT"
    dev_pos, holdout_pos = min(
        candidates,
        key=lambda pair: abs((len(pair[1]) / len(df)) - test_size),
    )
    actual_fraction = len(holdout_pos) / len(df)
    if abs(actual_fraction - test_size) > HOLDOUT_FRACTION_TOLERANCE:
        raise ValueError(
            "No group-safe holdout is within the allowed fraction tolerance: "
            f"target={test_size:.3f}, actual={actual_fraction:.3f}, "
            f"tolerance={HOLDOUT_FRACTION_TOLERANCE:.3f}."
        )
    if set(group_values[dev_pos]) & set(group_values[holdout_pos]):
        raise RuntimeError("Duplicate identity crossed Development/Holdout.")
    dev = df.iloc[dev_pos].copy()
    holdout = df.iloc[holdout_pos].copy()
    labels = [(int(pos), "DEVELOPMENT") for pos in dev_pos]
    labels += [(int(pos), "HOLDOUT") for pos in holdout_pos]
    report_bands = fixed_price_bands(df[TARGET_COLUMN]).astype(str).to_numpy()
    metadata = {
        "method": method,
        "random_state": random_state,
        "development_rows": len(dev),
        "holdout_rows": len(holdout),
        "target_holdout_fraction": test_size,
        "actual_holdout_fraction": actual_fraction,
        "holdout_fraction_tolerance": HOLDOUT_FRACTION_TOLERANCE,
        "group_count": len(set(group_values)),
        "duplicate_group_count": int(pd.Series(group_values).value_counts().gt(1).sum()),
        "assignment_checksum": _assignment_checksum(sorted(labels)),
        "development_price_band_counts": {
            str(k): int(v) for k, v in pd.Series(report_bands[dev_pos]).value_counts().sort_index().items()
        },
        "holdout_price_band_counts": {
            str(k): int(v) for k, v in pd.Series(report_bands[holdout_pos]).value_counts().sort_index().items()
        },
    }
    metadata["checksum"] = metadata["assignment_checksum"]
    return dev, holdout, metadata


def build_common_cv_folds(
    dev_df: pd.DataFrame,
    groups: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], dict]:
    group_values = groups.loc[dev_df.index].astype(str).to_numpy()
    if len(set(group_values)) < n_splits:
        raise ValueError(f"At least {n_splits} Development groups are required.")
    bands = make_price_bands(dev_df[TARGET_COLUMN], n_splits).to_numpy()
    has_duplicate_groups = len(set(group_values)) < len(group_values)
    if has_duplicate_groups:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        raw_folds = splitter.split(dev_df, bands, group_values)
        method = "STRATIFIED_GROUP_KFOLD"
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        raw_folds = splitter.split(dev_df, bands)
        method = "STRATIFIED_KFOLD"
    folds = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
             for tr, va in raw_folds]
    seen: List[int] = []
    checksum_rows: List[Tuple[int, str]] = []
    for fold_id, (train_pos, valid_pos) in enumerate(folds, start=1):
        if set(train_pos) & set(valid_pos):
            raise RuntimeError(f"Fold {fold_id} train/validation overlap.")
        if set(group_values[train_pos]) & set(group_values[valid_pos]):
            raise RuntimeError(f"Duplicate identity crossed fold {fold_id}.")
        seen.extend(valid_pos.tolist())
        checksum_rows.extend((int(pos), f"FOLD_{fold_id}") for pos in valid_pos)
    if sorted(seen) != list(range(len(dev_df))):
        raise RuntimeError("Each Development row must validate exactly once.")
    report_bands = fixed_price_bands(dev_df[TARGET_COLUMN]).astype(str).to_numpy()
    metadata = {
        "method": method,
        "n_splits": n_splits,
        "random_state": random_state,
        "fold_sizes": [len(valid) for _, valid in folds],
        "assignment_checksum": _assignment_checksum(sorted(checksum_rows)),
        "fold_price_band_counts": [
            {str(k): int(v) for k, v in pd.Series(report_bands[valid]).value_counts().sort_index().items()}
            for _, valid in folds
        ],
    }
    metadata["checksum"] = metadata["assignment_checksum"]
    return folds, metadata


def export_train_row_selection_report(
    cleaned_df: pd.DataFrame, reasons: pd.Series,
    pcs_date: pd.Timestamp, run_id: str,
    split_assignments: Optional[pd.Series] = None,
) -> Path:
    """Audit V4 eligibility and evaluation assignment without changing STG."""
    report = pd.DataFrame(index=cleaned_df.index)
    report["SOURCE_ROW_NUMBER"] = cleaned_df.index.to_numpy() + 1
    listing_col = find_column_case_insensitive(cleaned_df.columns, "listing_id")
    report["listing_id"] = (
        cleaned_df[listing_col] if listing_col is not None else pd.NA
    )
    report["price"] = cleaned_df[TARGET_COLUMN]
    report["PRICE_FILTER_MODE"] = PRICE_FILTER_MODE
    report["ELIGIBLE_FOR_EVALUATION"] = np.where(reasons.eq(""), "Y", "N")
    report["USED_FOR_TRAIN"] = np.where(reasons.eq(""), "Y", "N")
    report["EXCLUSION_REASON"] = reasons.to_numpy()
    if split_assignments is None:
        report["EVALUATION_SPLIT"] = np.where(reasons.eq(""), "UNASSIGNED", "EXCLUDED")
    else:
        aligned = split_assignments.reindex(cleaned_df.index).fillna("EXCLUDED")
        report["EVALUATION_SPLIT"] = aligned.to_numpy()
    report["PCS_DATE"] = pcs_date.date().isoformat()
    report_dir = PRICE_SELECTION_REPORT_BASE_DIR / pcs_date.strftime("%Y%m%d")
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"train_row_selection_{run_id}.csv"
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[PRICE FILTER] Training row audit: {output_path}")
    return output_path


def maybe_convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col in {TARGET_COLUMN, PCS_DATE_COLUMN}:
            continue

        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            continue

        original = df[col]
        non_null = original.dropna()
        if non_null.empty:
            continue

        cleaned = (
            non_null.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
        )
        converted = pd.to_numeric(cleaned, errors="coerce")
        success_ratio = converted.notna().mean()

        if success_ratio >= NUMERIC_COERCE_THRESHOLD:
            full_cleaned = (
                original.astype("string")
                .str.strip()
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(full_cleaned, errors="coerce")

    return df


def numeric_conversion_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    cleaned = (
        non_null.astype(str).str.strip().str.replace(",", "", regex=False)
    )
    converted = pd.to_numeric(cleaned, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return float(converted.notna().mean())


def coerce_numeric_like(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string").str.strip().str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def matches_exclusion_pattern(column_name: str) -> bool:
    for pattern in EXCLUDE_COLUMN_PATTERNS:
        if re.search(pattern, column_name, flags=re.IGNORECASE):
            return True
    return False


def infer_eligible_features(df: pd.DataFrame) -> Tuple[List[str], List[Tuple[str, str]]]:
    force_include = {c.lower() for c in FORCE_INCLUDE_COLUMNS}
    force_exclude = {c.lower() for c in FORCE_EXCLUDE_COLUMNS}
    exact_exclude = {c.lower() for c in EXCLUDE_COLUMNS_EXACT}

    eligible = []
    excluded = []

    for col in df.columns:
        lower = col.lower()

        if lower in force_include:
            eligible.append(col)
            continue

        if lower in force_exclude:
            excluded.append((col, "FORCE_EXCLUDE_COLUMNS"))
            continue

        if lower in exact_exclude:
            excluded.append((col, "EXCLUDE_COLUMNS_EXACT"))
            continue

        if matches_exclusion_pattern(col):
            excluded.append((col, "EXCLUDE_COLUMN_PATTERNS"))
            continue

        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            excluded.append((col, "datetime"))
            continue

        missing_ratio = series.isna().mean()
        if missing_ratio > MAX_MISSING_RATIO:
            excluded.append((col, f"missing_ratio={missing_ratio:.3f}"))
            continue

        non_null = series.dropna()
        unique_count = non_null.nunique(dropna=True)

        if unique_count <= 1:
            excluded.append((col, "constant_or_empty"))
            continue

        if pd.api.types.is_numeric_dtype(series):
            eligible.append(col)
            continue

        if numeric_conversion_ratio(series) >= NUMERIC_COERCE_THRESHOLD:
            eligible.append(col)
            continue

        unique_ratio = unique_count / max(len(non_null), 1)

        if lower in CORE_VEHICLE_FEATURES or lower in OPTIONAL_VEHICLE_FEATURES:
            eligible.append(col)
            continue

        if unique_count > MAX_CATEGORICAL_LEVELS:
            excluded.append((col, f"high_cardinality_levels={unique_count}"))
            continue

        if unique_ratio > MAX_CATEGORICAL_UNIQUE_RATIO:
            excluded.append((col, f"high_cardinality_ratio={unique_ratio:.3f}"))
            continue

        eligible.append(col)

    if not eligible:
        raise RuntimeError(
            "No usable predictor columns remain after dynamic schema checks."
        )

    return eligible, excluded


def infer_candidate_universe(
    df: pd.DataFrame,
    approved_features: Sequence[str],
) -> List[str]:
    """Apply explicit authorization before any data-driven eligibility."""
    approved = {str(c).casefold() for c in approved_features}
    force_include = {c.lower() for c in FORCE_INCLUDE_COLUMNS}
    force_exclude = {c.lower() for c in FORCE_EXCLUDE_COLUMNS}
    exact_exclude = {c.lower() for c in EXCLUDE_COLUMNS_EXACT}
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if lower not in approved:
            continue
        if lower in force_include:
            candidates.append(col)
        elif lower in force_exclude or lower in exact_exclude:
            continue
        elif matches_exclusion_pattern(col):
            continue
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        else:
            candidates.append(col)
    if not candidates:
        raise RuntimeError("No candidate columns remain after schema exclusions.")
    return candidates


def normalize_category_series(series: pd.Series) -> pd.Series:
    result = series.astype("string").fillna("__MISSING__").str.strip()
    result = result.replace("", "__MISSING__")
    return result.astype(str)


def safe_feature_name(source_column: str, original_value: str) -> str:
    raw = f"{source_column}={original_value}"
    if len(raw) <= 255:
        return raw

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return raw[:240] + "_" + digest


def deterministic_reference(counts: pd.Series) -> str:
    pairs = [(str(level), int(count)) for level, count in counts.items()]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs[0][0]


def fit_preprocessor(
    df: pd.DataFrame,
    source_features: Sequence[str],
    encoding_profile: str = "BASELINE",
) -> PreprocessorState:
    if encoding_profile not in ENCODING_PROFILES:
        raise ValueError(f"Unknown encoding profile: {encoding_profile}")
    profile = ENCODING_PROFILES[encoding_profile]
    numeric_features = []
    numeric_like_features = []
    categorical_features = []
    numeric_medians: Dict[str, float] = {}
    category_levels: Dict[str, List[str]] = {}
    reference_categories: Dict[str, str] = {}
    feature_groups: Dict[str, List[str]] = {}

    for col in source_features:
        s = df[col]

        missing_ratio = s.isna().mean()
        if missing_ratio > MAX_MISSING_RATIO or s.dropna().nunique() <= 1:
            continue

        if pd.api.types.is_numeric_dtype(s):
            values = pd.to_numeric(s, errors="coerce").astype(float).replace(
                [np.inf, -np.inf], np.nan
            )
            median = values.median()
            if pd.isna(median):
                continue

            numeric_features.append(col)
            numeric_medians[col] = float(median)
            feature_groups[col] = [col]
            continue

        if numeric_conversion_ratio(s) >= NUMERIC_COERCE_THRESHOLD:
            values = coerce_numeric_like(s).astype(float)
            median = values.median()
            if pd.isna(median):
                continue
            numeric_features.append(col)
            numeric_like_features.append(col)
            numeric_medians[col] = float(median)
            feature_groups[col] = [col]
            continue

        values = normalize_category_series(s)
        counts = values.value_counts(dropna=False)

        unique_count = int(s.dropna().nunique())
        unique_ratio = unique_count / max(int(s.notna().sum()), 1)
        if (
            col.lower() not in CORE_VEHICLE_FEATURES + OPTIONAL_VEHICLE_FEATURES
            and (unique_count > MAX_CATEGORICAL_LEVELS
                 or unique_ratio > MAX_CATEGORICAL_UNIQUE_RATIO)
        ):
            continue

        # Learn frequency grouping ONLY on the training partition. Future
        # observations use the exact same stored category levels and OTHER rule.
        minimum = profile["min_count"].get(col.lower(), MIN_CATEGORY_COUNT)
        max_levels = profile["max_levels"].get(col.lower(), MAX_CATEGORICAL_LEVELS)
        frequent = counts[counts >= minimum]
        # Reserve a slot for __OTHER__, including new categories at prediction.
        retained = [str(v) for v in frequent.index if str(v) != "__OTHER__"][:max_levels - 1]
        if not retained:
            retained = [str(counts.index[0])]
        values = values.where(values.isin(retained), "__OTHER__")
        counts = values.value_counts(dropna=False)
        if "__OTHER__" not in counts.index:
            counts.loc["__OTHER__"] = 0
        levels = [str(x) for x in counts.index.tolist()]

        if len(levels) <= 1:
            continue

        categorical_features.append(col)
        reference = deterministic_reference(counts)
        non_reference = sorted([x for x in levels if x != reference])
        ordered_levels = [reference] + non_reference

        category_levels[col] = ordered_levels
        reference_categories[col] = reference

        group_cols = [
            safe_feature_name(col, level)
            for level in non_reference
        ]
        feature_groups[col] = group_cols

    numeric_features = [
        c for c in numeric_features
        if c in feature_groups and feature_groups[c]
    ]
    categorical_features = [
        c for c in categorical_features
        if c in feature_groups and feature_groups[c]
    ]

    encoded_feature_names = [
        name
        for source_col in feature_groups
        for name in feature_groups[source_col]
    ]

    return PreprocessorState(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_medians=numeric_medians,
        category_levels=category_levels,
        reference_categories=reference_categories,
        encoded_feature_names=encoded_feature_names,
        feature_groups=feature_groups,
        numeric_like_features=numeric_like_features,
    )


def transform_with_preprocessor(
    df: pd.DataFrame,
    state: PreprocessorState,
) -> pd.DataFrame:
    # Construct encoded columns in one pass to avoid highly fragmented Frames.
    # Preserve the V2 names, reference-category behavior and column ordering.
    encoded: Dict[str, pd.Series] = {}

    for col in state.numeric_features:
        # Apply one coercion path for numeric, string and categorical dtypes so
        # legacy consumers that reconstruct state without numeric_like_features
        # still transform prediction rows identically.
        values = coerce_numeric_like(df[col]).astype(float)
        encoded[col] = values.fillna(state.numeric_medians[col])

    for col in state.categorical_features:
        values = normalize_category_series(df[col])
        levels = state.category_levels[col]
        reference = state.reference_categories[col]
        fallback = "__OTHER__" if "__OTHER__" in levels else reference
        values = values.where(values.isin(levels), fallback)

        for level in levels:
            if level != reference:
                encoded[safe_feature_name(col, level)] = (values == level).astype(float)

    if not state.encoded_feature_names:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(
        {name: encoded.get(name, pd.Series(0.0, index=df.index))
         for name in state.encoded_feature_names},
        index=df.index,
    ).astype(float)


def fit_ols(X: pd.DataFrame, y: pd.Series):
    X_const = sm.add_constant(X.astype(float), has_constant="add")
    return sm.OLS(
        y.astype(float),
        X_const,
        missing="drop",
    ).fit()


def source_feature_p_value(
    model,
    encoded_columns: Sequence[str],
) -> float:
    if not encoded_columns:
        return 1.0

    param_names = list(model.params.index)
    param_lookup = {name: i for i, name in enumerate(param_names)}

    valid = [c for c in encoded_columns if c in param_lookup]
    if not valid:
        return 1.0

    restriction = np.zeros((len(valid), len(param_names)), dtype=float)

    for row_idx, feature_name in enumerate(valid):
        restriction[row_idx, param_lookup[feature_name]] = 1.0

    try:
        test = model.f_test(restriction)
        p = float(np.asarray(test.pvalue).reshape(-1)[0])
        if not np.isfinite(p):
            return 1.0
        return p
    except Exception:
        return 1.0


def backward_eliminate_source_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: Dict[str, List[str]],
) -> Tuple[object, List[str], List[str], Dict[str, float]]:
    active_groups = {
        source: [c for c in encoded if c in X.columns]
        for source, encoded in feature_groups.items()
    }
    active_groups = {
        source: encoded
        for source, encoded in active_groups.items()
        if encoded
    }

    if len(active_groups) < MIN_SOURCE_FEATURES:
        raise RuntimeError("Not enough usable source features for OLS.")

    while True:
        selected_encoded = [
            encoded_name
            for source in active_groups
            for encoded_name in active_groups[source]
        ]

        model = fit_ols(X[selected_encoded], y)

        group_pvalues = {
            source: source_feature_p_value(model, encoded)
            for source, encoded in active_groups.items()
        }

        worst_source = max(group_pvalues, key=group_pvalues.get)
        worst_p = group_pvalues[worst_source]

        if (
            worst_p <= P_VALUE_THRESHOLD
            or len(active_groups) <= MIN_SOURCE_FEATURES
        ):
            break

        del active_groups[worst_source]

    selected_source = list(active_groups.keys())
    selected_encoded = [
        encoded_name
        for source in active_groups
        for encoded_name in active_groups[source]
    ]

    final_model = fit_ols(X[selected_encoded], y)

    final_group_pvalues = {
        source: source_feature_p_value(
            final_model,
            active_groups[source],
        )
        for source in active_groups
    }

    return (
        final_model,
        selected_source,
        selected_encoded,
        final_group_pvalues,
    )


def generate_candidate_feature_sets(
    all_features: Sequence[str],
) -> List[List[str]]:
    """Compare vehicle feature combinations, not arbitrary core-feature drops.

    Sub-model is optional; candidate sets include brand + model, brand only,
    model only, and (when present) a version with sub-model. Other source
    features remain dynamically discovered from today's Alice schema.
    """
    features = list(dict.fromkeys(all_features))
    by_lower = {c.lower(): c for c in features}
    brand = by_lower.get("brand")
    model = by_lower.get("model")
    sub_model = by_lower.get("sub_model")
    core = [c for c in (brand, model) if c]
    optional = [sub_model] if sub_model else []
    other = [c for c in features if c not in core + optional]
    sets: List[List[str]] = []
    seen = set()

    def add(items):
        cols = list(dict.fromkeys(c for c in items if c))
        key = tuple(sorted(cols))
        if cols and key not in seen:
            seen.add(key)
            sets.append(cols)

    # The first four deliberately test the importance of brand and model.
    add(core + other)
    add(core + optional + other)
    if brand:
        add([brand] + other)
    if model:
        add([model] + other)
    # Test whether a smaller supplementary feature set generalizes better.
    rng = np.random.default_rng(RANDOM_STATE)
    for fraction in (0.85, 0.65, 0.45):
        k = max(1, int(math.ceil(len(other) * fraction))) if other else 0
        picked = rng.choice(other, size=k, replace=False).tolist() if k else []
        add(core + picked)
        add(core + optional + picked)
    return sets[:MAX_CANDIDATES]


def build_candidate_experiments(all_features: Sequence[str]) -> List[CandidateExperiment]:
    """Pair BASELINE and EXPANDED on exactly the same source feature sets.

    Keep the V2 baseline search (8 sets) and add expanded versions of the
    first two sets (brand+model, with/without optional sub_model). This allows
    direct, fold-matched encoding comparisons without changing price filtering.
    """
    baseline_sets = generate_candidate_feature_sets(all_features)
    experiments = [CandidateExperiment(features, "BASELINE") for features in baseline_sets]
    for features in baseline_sets[:MAX_EXPANDED_EXPERIMENTS]:
        experiments.append(CandidateExperiment(features, "EXPANDED_BRAND_MODEL"))
    return experiments


def evaluate_candidate_cv(
    df: pd.DataFrame,
    initial_source_features: Sequence[str],
    encoding_profile: str = "BASELINE",
    folds: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> CVEvaluation:
    """Evaluate one candidate on common eligible-Development folds."""
    if not df[TARGET_COLUMN].gt(TARGET_PRICE_MIN_EXCLUSIVE_THB).all():
        raise ValueError("CV input must contain only the V4 eligible cohort.")
    if folds is None:
        groups = build_duplicate_groups(df)
        folds, _ = build_common_cv_folds(df, groups)

    fold_rmses: List[float] = []
    fold_maes: List[float] = []
    records: List[pd.DataFrame] = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        fold_eligible, _ = infer_eligible_features(train_df)
        fold_features = [c for c in initial_source_features if c in fold_eligible]
        if len(fold_features) < MIN_SOURCE_FEATURES:
            raise RuntimeError(
                f"Fold {fold_id}: candidate has no usable fold-train features."
            )
        state = fit_preprocessor(train_df, fold_features, encoding_profile)
        X_train = transform_with_preprocessor(train_df, state)
        X_valid = transform_with_preprocessor(valid_df, state)
        if X_train.shape[1] == 0:
            raise RuntimeError(f"Fold {fold_id}: no encoded predictors.")
        model, _, selected_encoded, _ = backward_eliminate_source_features(
            X_train, train_df[TARGET_COLUMN].astype(float), state.feature_groups
        )
        X_valid_const = sm.add_constant(
            X_valid[selected_encoded], has_constant="add"
        ).reindex(columns=model.params.index, fill_value=0.0)
        pred = np.asarray(model.predict(X_valid_const), dtype=float)
        if len(pred) != len(valid_df) or not np.isfinite(pred).all():
            raise RuntimeError(f"Fold {fold_id}: invalid predictions.")
        actual = valid_df[TARGET_COLUMN].to_numpy(dtype=float)
        errors = actual - pred
        fold_rmses.append(float(math.sqrt(np.mean(np.square(errors)))))
        fold_maes.append(float(np.mean(np.abs(errors))))

        mapped_model = pd.Series(pd.NA, index=valid_df.index, dtype="string")
        model_col = next(
            (c for c in state.categorical_features if c.lower() == "model"), None
        )
        if model_col is not None:
            values = normalize_category_series(valid_df[model_col])
            levels = state.category_levels[model_col]
            fallback = (
                "__OTHER__" if "__OTHER__" in levels
                else state.reference_categories[model_col]
            )
            mapped_model = values.where(values.isin(levels), fallback).astype("string")
        records.append(pd.DataFrame({
            "ROW_POSITION": np.asarray(valid_idx, dtype=int),
            "FOLD": fold_id,
            "ACTUAL_PRICE": actual,
            "PREDICTED_PRICE": pred,
            "ERROR": errors,
            "ABS_ERROR": np.abs(errors),
            "MODEL_CATEGORY": mapped_model.to_numpy(),
        }))

    oof = pd.concat(records, ignore_index=True).sort_values("ROW_POSITION")
    if oof["ROW_POSITION"].tolist() != list(range(len(df))):
        raise RuntimeError("OOF predictions do not cover Development exactly once.")
    errors = oof["ERROR"].to_numpy(dtype=float)
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    negative_count = int(oof["PREDICTED_PRICE"].lt(0).sum())
    model_denominator = int(oof["MODEL_CATEGORY"].notna().sum())
    other_count = int(oof["MODEL_CATEGORY"].eq("__OTHER__").sum())
    other_rate = (
        other_count / model_denominator if model_denominator else None
    )
    tail = oof["ACTUAL_PRICE"].ge(HIGH_PRICE_MIN_THB)
    high_price_mae = (
        float(oof.loc[tail, "ABS_ERROR"].mean()) if tail.any() else None
    )
    banded = pd.cut(
        oof["ACTUAL_PRICE"], PRICE_BAND_EDGES, include_lowest=True
    )
    price_band_metrics = []
    for interval, part in oof.groupby(banded, observed=True):
        band_errors = part["ERROR"].to_numpy(dtype=float)
        price_band_metrics.append({
            "price_band": str(interval),
            "count": len(part),
            "rmse": float(math.sqrt(np.mean(np.square(band_errors)))),
            "mae": float(np.mean(np.abs(band_errors))),
        })
    return CVEvaluation(
        rmse=rmse,
        mae=mae,
        count=len(oof),
        fold_rmse_mean=float(np.mean(fold_rmses)),
        fold_rmse_std=float(np.std(fold_rmses, ddof=0)),
        fold_mae_mean=float(np.mean(fold_maes)),
        fold_mae_std=float(np.std(fold_maes, ddof=0)),
        high_price_mae=high_price_mae,
        negative_rate=negative_count / len(oof),
        negative_count=negative_count,
        other_rate=other_rate,
        other_count=other_count,
        other_denominator=model_denominator,
        price_band_metrics=price_band_metrics,
        oof_predictions=oof,
    )


def fit_final_candidate(
    df: pd.DataFrame,
    candidate_id: int,
    initial_source_features: Sequence[str],
    cv_rmse: float,
    cv_mae: float,
    encoding_profile: str = "BASELINE",
    cv_high_price_mae: Optional[float] = None,
    cv_negative_rate: Optional[float] = None,
    cv_model_other_rate: Optional[float] = None,
    cv_eligible_rmse: Optional[float] = None,
    cv_eligible_mae: Optional[float] = None,
    cv_eligible_count: int = 0,
    cv_evaluation: Optional[CVEvaluation] = None,
) -> FinalCandidate:
    if cv_evaluation is not None:
        cv_rmse = cv_evaluation.rmse
        cv_mae = cv_evaluation.mae
        cv_high_price_mae = cv_evaluation.high_price_mae
        cv_negative_rate = cv_evaluation.negative_rate
        cv_model_other_rate = cv_evaluation.other_rate
        cv_eligible_rmse = cv_evaluation.rmse
        cv_eligible_mae = cv_evaluation.mae
        cv_eligible_count = cv_evaluation.count
    state = fit_preprocessor(df, initial_source_features, encoding_profile)

    X = transform_with_preprocessor(df, state)
    y = df[TARGET_COLUMN].astype(float)

    model, selected_source, selected_encoded, source_feature_p_values = (
        backward_eliminate_source_features(
            X,
            y,
            state.feature_groups,
        )
    )

    f_stat = None
    f_p = None

    try:
        if model.fvalue is not None and np.isfinite(float(model.fvalue)):
            f_stat = float(model.fvalue)
    except Exception:
        pass

    try:
        if model.f_pvalue is not None and np.isfinite(float(model.f_pvalue)):
            f_p = float(model.f_pvalue)
    except Exception:
        pass

    return FinalCandidate(
        candidate_id=candidate_id,
        initial_source_features=list(initial_source_features),
        selected_source_features=selected_source,
        selected_encoded_features=selected_encoded,
        preprocessor=state,
        ols_result=model,
        cv_rmse=float(cv_rmse),
        cv_mae=float(cv_mae),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=f_stat,
        f_p_value=f_p,
        source_feature_p_values=source_feature_p_values,
        encoding_profile=encoding_profile,
        cv_high_price_mae=cv_high_price_mae,
        cv_negative_rate=cv_negative_rate,
        cv_model_other_rate=cv_model_other_rate,
        cv_eligible_rmse=cv_eligible_rmse,
        cv_eligible_mae=cv_eligible_mae,
        cv_eligible_count=cv_eligible_count,
        cv_fold_rmse_mean=(cv_evaluation.fold_rmse_mean if cv_evaluation else None),
        cv_fold_rmse_std=(cv_evaluation.fold_rmse_std if cv_evaluation else None),
        cv_fold_mae_mean=(cv_evaluation.fold_mae_mean if cv_evaluation else None),
        cv_fold_mae_std=(cv_evaluation.fold_mae_std if cv_evaluation else None),
        cv_price_band_metrics=(cv_evaluation.price_band_metrics if cv_evaluation else []),
        cv_other_count=(cv_evaluation.other_count if cv_evaluation else 0),
        cv_other_denominator=(cv_evaluation.other_denominator if cv_evaluation else 0),
    )


def refit_candidate_coefficients(
    checkpoint: FinalCandidate,
    full_df: pd.DataFrame,
) -> FinalCandidate:
    """Refit only OLS coefficients using the frozen Development design."""
    X = transform_with_preprocessor(full_df, checkpoint.preprocessor)
    selected = checkpoint.selected_encoded_features
    missing = [name for name in selected if name not in X.columns]
    if missing:
        raise RuntimeError(f"Frozen refit is missing encoded columns: {missing}")
    model = fit_ols(X[selected], full_df[TARGET_COLUMN].astype(float))
    if int(model.nobs) != len(full_df):
        raise RuntimeError(
            f"Full refit used {int(model.nobs):,}/{len(full_df):,} rows."
        )
    source_pvalues = {
        source: source_feature_p_value(
            model,
            [c for c in checkpoint.preprocessor.feature_groups[source]
             if c in selected],
        )
        for source in checkpoint.selected_source_features
    }
    f_stat = float(model.fvalue) if model.fvalue is not None and np.isfinite(model.fvalue) else None
    f_p = float(model.f_pvalue) if model.f_pvalue is not None and np.isfinite(model.f_pvalue) else None
    return replace(
        checkpoint,
        ols_result=model,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=f_stat,
        f_p_value=f_p,
        source_feature_p_values=source_pvalues,
    )


def evaluate_frozen_candidate(
    checkpoint: FinalCandidate,
    holdout_df: pd.DataFrame,
) -> Tuple[dict, pd.DataFrame]:
    X = transform_with_preprocessor(holdout_df, checkpoint.preprocessor)
    X_const = sm.add_constant(
        X[checkpoint.selected_encoded_features], has_constant="add"
    ).reindex(columns=checkpoint.ols_result.params.index, fill_value=0.0)
    pred = np.asarray(checkpoint.ols_result.predict(X_const), dtype=float)
    actual = holdout_df[TARGET_COLUMN].to_numpy(dtype=float)
    errors = actual - pred
    if not np.isfinite(pred).all():
        raise RuntimeError("Holdout predictions contain non-finite values.")
    mapped_model = pd.Series(pd.NA, index=holdout_df.index, dtype="string")
    model_col = next(
        (c for c in checkpoint.preprocessor.categorical_features
         if c.lower() == "model"), None
    )
    if model_col is not None:
        values = normalize_category_series(holdout_df[model_col])
        levels = checkpoint.preprocessor.category_levels[model_col]
        fallback = (
            "__OTHER__" if "__OTHER__" in levels
            else checkpoint.preprocessor.reference_categories[model_col]
        )
        mapped_model = values.where(values.isin(levels), fallback).astype("string")
    predictions = pd.DataFrame({
        "SOURCE_ROW_NUMBER": holdout_df.index.to_numpy(dtype=int) + 1,
        "ACTUAL_PRICE": actual,
        "PREDICTED_PRICE": pred,
        "ERROR": errors,
        "ABS_ERROR": np.abs(errors),
        "NEGATIVE_PREDICTION": np.where(pred < 0, "Y", "N"),
        "MODEL_CATEGORY": mapped_model.to_numpy(),
    })
    other_denominator = int(predictions["MODEL_CATEGORY"].notna().sum())
    other_count = int(predictions["MODEL_CATEGORY"].eq("__OTHER__").sum())
    price_band_metrics = []
    banded = pd.cut(
        predictions["ACTUAL_PRICE"], PRICE_BAND_EDGES, include_lowest=True
    )
    for interval, part in predictions.groupby(banded, observed=True):
        band_errors = part["ERROR"].to_numpy(dtype=float)
        price_band_metrics.append({
            "price_band": str(interval),
            "count": len(part),
            "rmse": float(math.sqrt(np.mean(np.square(band_errors)))),
            "mae": float(np.mean(np.abs(band_errors))),
        })
    metrics = {
        "stage": "HOLDOUT_TEST_DEVELOPMENT_CHECKPOINT",
        "count": len(predictions),
        "rmse": float(math.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "negative_count": int((pred < 0).sum()),
        "negative_rate": float((pred < 0).mean()),
        "other_count": other_count,
        "other_denominator": other_denominator,
        "other_rate": (other_count / other_denominator
                       if other_denominator else None),
        "price_band_metrics": price_band_metrics,
    }
    return metrics, predictions


def rank_candidates(
    candidates: Sequence[FinalCandidate],
) -> List[FinalCandidate]:
    best_by_structure: Dict[Tuple[str, ...], FinalCandidate] = {}

    for candidate in candidates:
        # Different retained levels => genuinely different encoders/models,
        # even when the selected source column names happen to be identical.
        selected = candidate.selected_source_features
        structure = (
            tuple(sorted(selected)),
            tuple(sorted((col, tuple(candidate.preprocessor.category_levels.get(col, [])))
                         for col in selected)),
        )

        if structure not in best_by_structure:
            best_by_structure[structure] = candidate
            continue

        current = best_by_structure[structure]

        new_key = (
            candidate.cv_rmse,
            candidate.cv_mae,
            candidate.candidate_id,
        )
        current_key = (
            current.cv_rmse,
            current.cv_mae,
            current.candidate_id,
        )

        if new_key < current_key:
            best_by_structure[structure] = candidate

    unique_candidates = list(best_by_structure.values())

    unique_candidates.sort(
        key=lambda c: (
            c.cv_rmse,
            c.cv_mae,
            c.candidate_id,
        )
    )

    return unique_candidates


def model_id_for_rank(
    run_datetime: datetime,
    rank: int,
) -> str:
    dt = run_datetime + timedelta(seconds=rank - 1)
    return dt.strftime("%Y%m%d_%H%M%S")


def build_result_dataframe(
    top_candidates: Sequence[FinalCandidate],
    run_datetime: datetime,
    pcs_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    model_ids: Dict[int, str] = {}
    rows = []

    pcs_date_str = pcs_date.date().isoformat()
    train_date_str = run_datetime.strftime("%Y-%m-%d %H:%M:%S")

    for rank, candidate in enumerate(top_candidates, start=1):
        model_id = model_id_for_rank(run_datetime, rank)
        model_ids[candidate.candidate_id] = model_id

        n_observation = int(candidate.ols_result.nobs)
        if not math.isclose(float(candidate.ols_result.nobs), n_observation):
            raise RuntimeError("OLS nobs must be an integer row count.")
        rows.append(
            {
                "MODEL_ID": model_id,
                "MODEL_NAME": f"{MODEL_NAME_PREFIX}_RANK_{rank}",
                "TARGET_NAME": TARGET_COLUMN,
                "TRAIN_PCS_DATE": pcs_date_str,
                "TRAIN_DATE": train_date_str,
                "N_OBSERVATION": n_observation,
                "R_SQUARED": candidate.r_squared,
                "ADJ_R_SQUARED": candidate.adj_r_squared,
                "MAE": candidate.cv_mae,
                "RMSE": candidate.cv_rmse,
                "F_STATISTIC": candidate.f_statistic,
                "F_P_VALUE": candidate.f_p_value,
                "P_VALUE_THRESHOLD": P_VALUE_THRESHOLD,
                "CONFIDENCE_LEVEL": CONFIDENCE_LEVEL,
                "ACTIVE_FLAG": None,
                "APPROVED_BY": None,
                "APPROVED_DATE": None,
                "PCS_DATE": pcs_date_str,
            }
        )

    result_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return result_df, model_ids


def coefficient_value_or_none(model, feature_name: str):
    if feature_name not in model.params.index:
        return None, None

    coef = model.params.get(feature_name)
    p_value = model.pvalues.get(feature_name)

    coef_value = (
        float(coef)
        if coef is not None and np.isfinite(float(coef))
        else None
    )
    p_value_value = (
        float(p_value)
        if p_value is not None and np.isfinite(float(p_value))
        else None
    )

    return coef_value, p_value_value


def coefficient_ci_or_none(confidence_intervals: pd.DataFrame, feature_name: str):
    """Return the two-sided coefficient confidence interval or (None, None)."""
    if feature_name not in confidence_intervals.index:
        return None, None

    lower = float(confidence_intervals.loc[feature_name].iloc[0])
    upper = float(confidence_intervals.loc[feature_name].iloc[1])
    return (
        lower if np.isfinite(lower) else None,
        upper if np.isfinite(upper) else None,
    )


def build_coefficient_dataframe(
    top_candidates: Sequence[FinalCandidate],
    model_ids: Dict[int, str],
    pcs_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    pcs_date_str = pcs_date.date().isoformat()

    for candidate in top_candidates:
        model_id = model_ids[candidate.candidate_id]
        model = candidate.ols_result
        state = candidate.preprocessor
        # Same confidence level and OLS fit used for each coefficient row.
        ci = model.conf_int(alpha=1.0 - CONFIDENCE_LEVEL)

        selected_sources = set(candidate.selected_source_features)
        selected_encoded = set(candidate.selected_encoded_features)

        feature_seq = 1

        intercept_coef, intercept_p = coefficient_value_or_none(
            model,
            "const",
        )
        intercept_ci_lower, intercept_ci_upper = coefficient_ci_or_none(
            ci, "const"
        )

        rows.append(
            {
                "MODEL_ID": model_id,
                "FEATURE_SEQ": feature_seq,
                "SOURCE_COLUMN": None,
                "ORIGINAL_VALUE": None,
                "FEATURE_NAME": "(Intercept)",
                "FEATURE_TYPE": "INTERCEPT",
                "COEFFICIENT": intercept_coef,
                "P_VALUE": intercept_p,
                "SOURCE_FEATURE_P_VALUE": None,
                "CI_LOWER": intercept_ci_lower,
                "CI_UPPER": intercept_ci_upper,
                "IS_REFERENCE": "N",
                "PCS_DATE": pcs_date_str,
            }
        )
        feature_seq += 1

        for source_col in candidate.initial_source_features:
            if source_col not in selected_sources:
                continue

            if source_col in state.numeric_features:
                coef, p_value = coefficient_value_or_none(
                    model,
                    source_col,
                )
                ci_lower, ci_upper = coefficient_ci_or_none(ci, source_col)

                rows.append(
                    {
                        "MODEL_ID": model_id,
                        "FEATURE_SEQ": feature_seq,
                        "SOURCE_COLUMN": source_col,
                        "ORIGINAL_VALUE": None,
                        "FEATURE_NAME": source_col,
                        "FEATURE_TYPE": "NUMERIC",
                        "COEFFICIENT": coef,
                        "P_VALUE": p_value,
                        "SOURCE_FEATURE_P_VALUE": candidate.source_feature_p_values.get(source_col),
                        "CI_LOWER": ci_lower,
                        "CI_UPPER": ci_upper,
                        "IS_REFERENCE": "N",
                        "PCS_DATE": pcs_date_str,
                    }
                )
                feature_seq += 1
                continue

            if source_col in state.categorical_features:
                levels = state.category_levels[source_col]
                reference = state.reference_categories[source_col]

                for level in levels:
                    if level == reference:
                        rows.append(
                            {
                                "MODEL_ID": model_id,
                                "FEATURE_SEQ": feature_seq,
                                "SOURCE_COLUMN": source_col,
                                "ORIGINAL_VALUE": level,
                                "FEATURE_NAME": None,
                                "FEATURE_TYPE": "ONEHOT",
                                "COEFFICIENT": None,
                                "P_VALUE": None,
                                "SOURCE_FEATURE_P_VALUE": candidate.source_feature_p_values.get(source_col),
                                "CI_LOWER": None,
                                "CI_UPPER": None,
                                "IS_REFERENCE": "Y",
                                "PCS_DATE": pcs_date_str,
                            }
                        )
                        feature_seq += 1
                        continue

                    encoded_name = safe_feature_name(
                        source_col,
                        level,
                    )

                    if encoded_name not in selected_encoded:
                        continue

                    coef, p_value = coefficient_value_or_none(
                        model,
                        encoded_name,
                    )
                    ci_lower, ci_upper = coefficient_ci_or_none(ci, encoded_name)

                    rows.append(
                        {
                            "MODEL_ID": model_id,
                            "FEATURE_SEQ": feature_seq,
                            "SOURCE_COLUMN": source_col,
                            "ORIGINAL_VALUE": level,
                            "FEATURE_NAME": encoded_name,
                            "FEATURE_TYPE": "ONEHOT",
                            "COEFFICIENT": coef,
                            "P_VALUE": p_value,
                            "SOURCE_FEATURE_P_VALUE": candidate.source_feature_p_values.get(source_col),
                            "CI_LOWER": ci_lower,
                            "CI_UPPER": ci_upper,
                            "IS_REFERENCE": "N",
                            "PCS_DATE": pcs_date_str,
                        }
                    )
                    feature_seq += 1

    return pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS)


def build_top1_model_bundle(
    top_candidate: FinalCandidate,
    model_id: str,
    run_id: str,
    pcs_date: pd.Timestamp,
    schema_gate: Optional[SchemaGateResult] = None,
) -> dict:
    state = top_candidate.preprocessor

    bundle = {
        "artifact_version": "2.0",
        "training_evaluation_version": "V4",
        "target_cohort_rule": f"{TARGET_COLUMN} > {TARGET_PRICE_MIN_EXCLUSIVE_THB:g}",
        "holdout_metrics_apply_to": "DEVELOPMENT_CHECKPOINT_NOT_EXPORTED_REFIT",
        "exported_model_stage": "FULL_DATA_FROZEN_DESIGN_COEFFICIENT_REFIT",
        "encoding_profile": top_candidate.encoding_profile,
        "price_filter_mode": PRICE_FILTER_MODE,
        "price_filter_max_thb": PRICE_FILTER_MAX_THB,
        "confirmed_bad_listing_ids": list(CONFIRMED_BAD_LISTING_IDS),
        "cv_eligible_rmse": top_candidate.cv_eligible_rmse,
        "cv_eligible_mae": top_candidate.cv_eligible_mae,
        "cv_eligible_count": top_candidate.cv_eligible_count,
        "encoding_config": ENCODING_PROFILES[top_candidate.encoding_profile],
        "run_id": run_id,
        "model_id": model_id,
        "model_rank": 1,
        "model_name": f"{MODEL_NAME_PREFIX}_RANK_1",
        "target_column": TARGET_COLUMN,
        "pcs_date": pcs_date.date().isoformat(),
        "p_value_threshold": P_VALUE_THRESHOLD,
        "confidence_level": CONFIDENCE_LEVEL,
        "cv_rmse": top_candidate.cv_rmse,
        "cv_mae": top_candidate.cv_mae,
        "r_squared": top_candidate.r_squared,
        "adj_r_squared": top_candidate.adj_r_squared,
        "selected_source_features": top_candidate.selected_source_features,
        "selected_encoded_features": top_candidate.selected_encoded_features,
        # Persist the learned model-input schema. Prediction must use this
        # snapshot even if STG_USED_CAR gains, loses or changes other columns.
        "training_feature_schema": {
            feature: (
                "numeric"
                if feature in state.numeric_features
                else "categorical"
            )
            for feature in top_candidate.selected_source_features
        },
        "source_feature_p_values": top_candidate.source_feature_p_values,
        "preprocessor": {
            "numeric_features": state.numeric_features,
            "categorical_features": state.categorical_features,
            "numeric_medians": state.numeric_medians,
            "category_levels": state.category_levels,
            "reference_categories": state.reference_categories,
            "encoded_feature_names": state.encoded_feature_names,
            "feature_groups": state.feature_groups,
            "numeric_like_features": state.numeric_like_features,
        },
        "ols_result": top_candidate.ols_result,
    }

    if schema_gate is not None:
        bundle["feature_approval"] = {
            "registry_version": schema_gate.registry_version,
            "registry_checksum": schema_gate.registry_checksum,
            "schema_fingerprint": schema_gate.schema_fingerprint,
            "approved_features_for_run": list(schema_gate.approved_features),
        }

    return bundle


def resolve_run_datetime() -> Tuple[datetime, str]:
    requested = os.getenv("USED_CAR_RUN_ID", "").strip()
    if not requested:
        value = datetime.now()
        return value, value.strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", requested):
        raise ValueError("USED_CAR_RUN_ID must use YYYYMMDD_HHMMSS.")
    try:
        value = datetime.strptime(requested, "%Y%m%d_%H%M%S")
    except ValueError as exc:
        raise ValueError("USED_CAR_RUN_ID is not a valid date/time.") from exc
    if value.strftime("%Y%m%d_%H%M%S") != requested:
        raise ValueError("USED_CAR_RUN_ID did not round-trip exactly.")
    return value, requested


def planned_artifact_paths(pcs_date: pd.Timestamp, run_id: str) -> Dict[str, Path]:
    date_folder = pcs_date.strftime("%Y%m%d")
    train_dir = OUTPUT_DIR / date_folder
    analysis_dir = OUTPUT_DIR.parent / "analysis" / date_folder
    return {
        "result": train_dir / f"OLS_REGRESSION_RESULT_{run_id}.csv",
        "coefficient": train_dir / f"OLS_REGRESSION_COEFFICIENT_{run_id}.csv",
        "joblib": train_dir / f"used_car_models_{run_id}.joblib",
        "price_quality_all": analysis_dir / f"price_quality_all_{run_id}.csv",
        "price_outliers": analysis_dir / f"price_outliers_for_review_{run_id}.csv",
        "row_selection": analysis_dir / f"train_row_selection_{run_id}.csv",
        "evaluation": analysis_dir / f"training_evaluation_{run_id}.csv",
        "holdout_predictions": analysis_dir / f"holdout_predictions_{run_id}.csv",
        "evaluation_metadata": analysis_dir / f"training_evaluation_metadata_{run_id}.json",
        "schema_review": analysis_dir / f"schema_review_{run_id}.json",
    }


def assert_no_artifact_collisions(paths: Dict[str, Path]) -> None:
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "RUN_ID would overwrite existing artifacts: " + ", ".join(collisions)
        )


def export_evaluation_sidecars(
    paths: Dict[str, Path],
    run_id: str,
    pcs_date: pd.Timestamp,
    ranked_checkpoints: Sequence[FinalCandidate],
    holdout_metrics: dict,
    holdout_predictions: pd.DataFrame,
    split_metadata: dict,
    fold_metadata: dict,
    source_positive_count: int,
    eligible_count: int,
    candidate_outcomes: Sequence[dict],
    schema_gate: SchemaGateResult,
) -> None:
    rows = []
    for rank, candidate in enumerate(ranked_checkpoints, start=1):
        rows.append({
            "RECORD_TYPE": "CANDIDATE_SUMMARY",
            "STAGE": "DEVELOPMENT_OOF",
            "MODEL_RANK": rank,
            "CANDIDATE_ID": candidate.candidate_id,
            "ENCODING_PROFILE": candidate.encoding_profile,
            "PRICE_BAND": None,
            "N_OBSERVATION": candidate.cv_eligible_count,
            "RMSE": candidate.cv_rmse,
            "MAE": candidate.cv_mae,
            "FOLD_RMSE_MEAN": candidate.cv_fold_rmse_mean,
            "FOLD_RMSE_STD": candidate.cv_fold_rmse_std,
            "FOLD_MAE_MEAN": candidate.cv_fold_mae_mean,
            "FOLD_MAE_STD": candidate.cv_fold_mae_std,
            "NEGATIVE_RATE": candidate.cv_negative_rate,
            "MODEL_OTHER_COUNT": candidate.cv_other_count,
            "MODEL_OTHER_DENOMINATOR": candidate.cv_other_denominator,
            "MODEL_OTHER_RATE": candidate.cv_model_other_rate,
        })
        for band in candidate.cv_price_band_metrics:
            rows.append({
                "RECORD_TYPE": "PRICE_BAND",
                "STAGE": "DEVELOPMENT_OOF",
                "MODEL_RANK": rank,
                "CANDIDATE_ID": candidate.candidate_id,
                "ENCODING_PROFILE": candidate.encoding_profile,
                "PRICE_BAND": band["price_band"],
                "N_OBSERVATION": band["count"],
                "RMSE": band["rmse"],
                "MAE": band["mae"],
            })
    rows.append({
        "RECORD_TYPE": "CANDIDATE_SUMMARY",
        "STAGE": "HOLDOUT_TEST_DEVELOPMENT_CHECKPOINT",
        "MODEL_RANK": 1,
        "CANDIDATE_ID": ranked_checkpoints[0].candidate_id,
        "ENCODING_PROFILE": ranked_checkpoints[0].encoding_profile,
        "PRICE_BAND": None,
        "N_OBSERVATION": holdout_metrics["count"],
        "RMSE": holdout_metrics["rmse"],
        "MAE": holdout_metrics["mae"],
        "NEGATIVE_RATE": holdout_metrics["negative_rate"],
        "MODEL_OTHER_COUNT": holdout_metrics["other_count"],
        "MODEL_OTHER_DENOMINATOR": holdout_metrics["other_denominator"],
        "MODEL_OTHER_RATE": holdout_metrics["other_rate"],
    })
    for band in holdout_metrics["price_band_metrics"]:
        rows.append({
            "RECORD_TYPE": "PRICE_BAND",
            "STAGE": "HOLDOUT_TEST_DEVELOPMENT_CHECKPOINT",
            "MODEL_RANK": 1,
            "CANDIDATE_ID": ranked_checkpoints[0].candidate_id,
            "ENCODING_PROFILE": ranked_checkpoints[0].encoding_profile,
            "PRICE_BAND": band["price_band"],
            "N_OBSERVATION": band["count"],
            "RMSE": band["rmse"],
            "MAE": band["mae"],
        })
    paths["evaluation"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        paths["evaluation"], index=False, encoding="utf-8-sig"
    )
    holdout_predictions.to_csv(
        paths["holdout_predictions"], index=False, encoding="utf-8-sig",
        float_format="%.6f",
    )
    metadata = {
        "training_evaluation_version": "V4",
        "run_id": run_id,
        "pcs_date": pcs_date.date().isoformat(),
        "target": TARGET_COLUMN,
        "eligible_cohort_rule": f"{TARGET_COLUMN} > {TARGET_PRICE_MIN_EXCLUSIVE_THB:g}",
        "positive_target_rows": source_positive_count,
        "eligible_rows": eligible_count,
        "excluded_by_cohort_rule": source_positive_count - eligible_count,
        "outlier_policy": "GROUP_BASED_SUSPECTED_OUTLIER_IS_FLAG_ONLY",
        "feature_approval": {
            "registry_version": schema_gate.registry_version,
            "registry_checksum": schema_gate.registry_checksum,
            "schema_fingerprint": schema_gate.schema_fingerprint,
            "approved_features_for_run": list(schema_gate.approved_features),
            "missing_approved_features": list(schema_gate.missing_approved_features),
            "incompatible_approved_features": list(
                schema_gate.incompatible_approved_features
            ),
        },
        "development_holdout_split": split_metadata,
        "development_cv_folds": fold_metadata,
        "candidate_outcomes": list(candidate_outcomes),
        "ranking": ["POOLED_OOF_RMSE", "POOLED_OOF_MAE", "CANDIDATE_ID"],
        "holdout_evaluated_model_rank": 1,
        "holdout_metrics_apply_to": "DEVELOPMENT_CHECKPOINT",
        "exported_model_stage": "FULL_DATA_FROZEN_DESIGN_COEFFICIENT_REFIT",
        "holdout_is_independent_test_of_exported_refit": False,
        "principal_output_contract": {
            "result_columns": len(RESULT_COLUMNS),
            "coefficient_columns": len(COEFFICIENT_COLUMNS),
            "joblib_rank": 1,
        },
    }
    paths["evaluation_metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not 0.0 < CONFIDENCE_LEVEL < 1.0:
        raise ValueError("CONFIDENCE_LEVEL must be between 0 and 1 (exclusive).")
    if not DB_PASSWORD:
        raise RuntimeError("USED_CAR_DB_PASSWORD must be set in the environment.")

    warnings.filterwarnings(
        "ignore",
        message="covariance of constraints does not have full rank",
    )

    run_datetime, run_id = resolve_run_datetime()

    print("=" * 78)
    print("USED CAR OLS TRAINING EVALUATION V4")
    print("=" * 78)
    print(f"[RUN] RUN_ID={run_id}")
    print(f"[CONFIG] Source     : {DB_SCHEMA}.{SOURCE_TABLE}")
    print(f"[CONFIG] Target     : {TARGET_COLUMN}")
    print(f"[CONFIG] Cohort     : {TARGET_COLUMN} > {TARGET_PRICE_MIN_EXCLUSIVE_THB:,.0f} THB")
    print(f"[CONFIG] Dev/Holdout: {1-HOLDOUT_FRACTION:.0%}/{HOLDOUT_FRACTION:.0%}")
    print(f"[CONFIG] TOP N      : {TOP_N_MODELS}")
    print(f"[CONFIG] P-value    : <= {P_VALUE_THRESHOLD}")
    print(f"[CONFIG] Confidence : {CONFIDENCE_LEVEL:.0%}")
    print(f"[CONFIG] CV folds   : {CV_FOLDS}")
    print(f"[CONFIG] Output base: {OUTPUT_DIR.resolve()}")

    engine = build_engine()

    try:
        df = load_source_data(engine)
    finally:
        engine.dispose()

    print(
        f"[INFO] Loaded {len(df):,} rows x {len(df.columns):,} columns."
    )

    df = normalize_required_column_names(df)
    pcs_date = parse_single_pcs_date(df)

    print(
        f"[INFO] Snapshot PCS_DATE = {pcs_date.date().isoformat()}"
    )

    paths = planned_artifact_paths(pcs_date, run_id)
    assert_no_artifact_collisions(paths)

    registry, registry_checksum = load_feature_registry()
    schema_gate = evaluate_feature_approval_gate(df, registry, registry_checksum)
    export_schema_review_report(
        paths["schema_review"], schema_gate, pcs_date, run_id
    )
    print(
        f"[SCHEMA] Registry={schema_gate.registry_version} "
        f"status={schema_gate.initial_approval_status}"
    )
    print(
        f"[SCHEMA] Approved/present/type-compatible predictors: "
        f"{len(schema_gate.approved_features)}"
    )
    print(f"[SCHEMA] Review report: {paths['schema_review']}")
    assert_feature_registry_approved(schema_gate)

    # Price-quality screening is always informational: no suspected peer
    # outliers are removed just because they have a review flag.
    export_outlier_flag_only_reports(df, pcs_date, run_id)

    cleaned_df = clean_target(df).reset_index(drop=True)
    eligible_df, exclusion_reasons = select_eligible_cohort(cleaned_df)
    # Learn identity components on the full positive-price snapshot. An
    # ineligible row can still bridge two eligible duplicate listings.
    duplicate_groups = build_duplicate_groups(cleaned_df).loc[eligible_df.index]
    development_df, holdout_df, split_metadata = split_development_holdout(
        eligible_df, duplicate_groups
    )
    common_folds, fold_metadata = build_common_cv_folds(
        development_df, duplicate_groups.loc[development_df.index]
    )
    split_assignments = pd.Series("EXCLUDED", index=cleaned_df.index, dtype="string")
    split_assignments.loc[development_df.index] = "DEVELOPMENT"
    split_assignments.loc[holdout_df.index] = "HOLDOUT"
    export_train_row_selection_report(
        cleaned_df, exclusion_reasons, pcs_date, run_id, split_assignments
    )
    print(f"[COHORT] Positive-price rows : {len(cleaned_df):,}")
    print(f"[COHORT] Eligible rows       : {len(eligible_df):,}")
    print(f"[COHORT] Excluded <= 1,000   : {len(cleaned_df) - len(eligible_df):,}")
    print(f"[SPLIT] Development/Holdout : {len(development_df):,}/{len(holdout_df):,}")
    print(f"[SPLIT] Assignment checksum : {split_metadata['assignment_checksum']}")
    print(f"[CV] Fold checksum          : {fold_metadata['assignment_checksum']}")
    print("[PRICE FILTER] Peer-group SUSPECTED_OUTLIER flags are review-only.")

    candidate_universe = infer_candidate_universe(
        development_df, schema_gate.approved_features
    )

    print(
        f"[INFO] Candidate universe ({len(candidate_universe)}): "
        + ", ".join(candidate_universe)
    )

    initial_candidates = build_candidate_experiments(candidate_universe)
    print("[INFO] Core vehicle features detected:",
          [c for c in candidate_universe if c.lower() in CORE_VEHICLE_FEATURES])
    print("[INFO] Optional sub-model detected:",
          [c for c in candidate_universe if c.lower() in OPTIONAL_VEHICLE_FEATURES])
    print("[INFO] Baseline category limits:", ENCODING_PROFILES["BASELINE"])
    print("[INFO] Expanded category limits:", ENCODING_PROFILES["EXPANDED_BRAND_MODEL"])
    print("[INFO] High-price cutoff (CV diagnostics only):", f"{HIGH_PRICE_MIN_THB:,.0f} THB")

    if len(initial_candidates) < TOP_N_MODELS:
        raise RuntimeError(
            f"Only {len(initial_candidates)} candidate feature sets could be "
            f"generated, but TOP_N_MODELS={TOP_N_MODELS}."
        )

    print(
        f"[INFO] Generated {len(initial_candidates)} candidate feature sets."
    )

    development_checkpoints: List[FinalCandidate] = []
    candidate_outcomes: List[dict] = []

    paired_results = {}
    for candidate_id, experiment in enumerate(initial_candidates, start=1):
        feature_set = experiment.source_features
        profile = experiment.encoding_profile
        print(
            f"[TRAIN] Candidate {candidate_id}/{len(initial_candidates)} "
            f"profile={profile} ({len(feature_set)} source features)"
        )

        try:
            evaluation = evaluate_candidate_cv(
                development_df, feature_set, profile, folds=common_folds,
            )

            checkpoint = fit_final_candidate(
                df=development_df,
                candidate_id=candidate_id,
                initial_source_features=feature_set,
                cv_rmse=evaluation.rmse,
                cv_mae=evaluation.mae,
                encoding_profile=profile,
                cv_evaluation=evaluation,
            )

            development_checkpoints.append(checkpoint)
            paired_results[(tuple(feature_set), profile)] = checkpoint
            candidate_outcomes.append({
                "candidate_id": candidate_id,
                "encoding_profile": profile,
                "status": "SUCCESS",
                "failure_reason": None,
            })

            print(
                f"        POOLED_OOF_RMSE={evaluation.rmse:,.2f} | "
                f"POOLED_OOF_MAE={evaluation.mae:,.2f} | "
                f"SelectedFeatures="
                f"{len(checkpoint.selected_source_features)} | "
                f"VehicleFeatures="
                f"{[c for c in checkpoint.selected_source_features if c.lower() in CORE_VEHICLE_FEATURES + OPTIONAL_VEHICLE_FEATURES]}"
            )
            print(
                f"        Fold RMSE mean/std={evaluation.fold_rmse_mean:,.2f}/"
                f"{evaluation.fold_rmse_std:,.2f} | Fold MAE mean/std="
                f"{evaluation.fold_mae_mean:,.2f}/{evaluation.fold_mae_std:,.2f}"
            )
            print(
                f"        OOF MAE(>= {HIGH_PRICE_MIN_THB:,.0f}THB)="
                f"{evaluation.high_price_mae if evaluation.high_price_mae is not None else float('nan'):,.0f} | "
                f"OOF Negative={evaluation.negative_rate:.2%} | "
                f"OOF model=__OTHER__={evaluation.other_rate:.2%} "
                f"({evaluation.other_count}/{evaluation.other_denominator})"
                if evaluation.other_rate is not None else
                f"        OOF MAE(>= {HIGH_PRICE_MIN_THB:,.0f}THB)="
                f"{evaluation.high_price_mae if evaluation.high_price_mae is not None else float('nan'):,.0f} | "
                f"OOF Negative={evaluation.negative_rate:.2%} | OOF model category=N/A"
            )

        except Exception as exc:
            candidate_outcomes.append({
                "candidate_id": candidate_id,
                "encoding_profile": profile,
                "status": "FAILED",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
            print(
                f"[WARN] Candidate {candidate_id} skipped: {exc}"
            )

    print("\n[COMPARISON] Paired BASELINE vs EXPANDED (same feature sets and CV folds):")
    for base_features in generate_candidate_feature_sets(candidate_universe)[:MAX_EXPANDED_EXPERIMENTS]:
        b = paired_results.get((tuple(base_features), "BASELINE"))
        e = paired_results.get((tuple(base_features), "EXPANDED_BRAND_MODEL"))
        if b is not None and e is not None:
            print(f"  Features: {', '.join(base_features)}")
            print(f"  BASELINE RMSE={b.cv_rmse:,.2f}, MAE={b.cv_mae:,.2f}, "
                  f"tail_MAE={b.cv_high_price_mae}, model_OTHER={b.cv_model_other_rate}")
            print(f"  EXPANDED RMSE={e.cv_rmse:,.2f}, MAE={e.cv_mae:,.2f}, "
                  f"tail_MAE={e.cv_high_price_mae}, model_OTHER={e.cv_model_other_rate}")

    if not development_checkpoints:
        raise RuntimeError(
            "No candidate model completed successfully."
        )

    ranked_checkpoints = rank_candidates(development_checkpoints)

    if len(ranked_checkpoints) < TOP_N_MODELS:
        print(
            f"[WARN] Only {len(ranked_checkpoints)} unique valid model structures were "
            f"found; requested TOP_N_MODELS={TOP_N_MODELS}."
        )

    top_checkpoints = ranked_checkpoints[:TOP_N_MODELS]
    holdout_metrics, holdout_predictions = evaluate_frozen_candidate(
        top_checkpoints[0], holdout_df
    )
    print(
        f"[HOLDOUT] Rank 1 Development checkpoint | "
        f"RMSE={holdout_metrics['rmse']:,.2f} | "
        f"MAE={holdout_metrics['mae']:,.2f} | n={holdout_metrics['count']:,}"
    )

    top_candidates = [
        refit_candidate_coefficients(checkpoint, eligible_df)
        for checkpoint in top_checkpoints
    ]
    if any(int(candidate.ols_result.nobs) != len(eligible_df)
           for candidate in top_candidates):
        raise RuntimeError("Exported Full-data refit N_OBSERVATION mismatch.")

    paths["result"].parent.mkdir(parents=True, exist_ok=True)

    result_df, model_ids = build_result_dataframe(
        top_candidates=top_candidates,
        run_datetime=run_datetime,
        pcs_date=pcs_date,
    )

    coefficient_df = build_coefficient_dataframe(
        top_candidates=top_candidates,
        model_ids=model_ids,
        pcs_date=pcs_date,
    )

    result_df.to_csv(
        paths["result"],
        index=False,
        encoding="utf-8-sig",
    )
    coefficient_df.to_csv(
        paths["coefficient"],
        index=False,
        encoding="utf-8-sig",
    )

    top1 = top_candidates[0]
    top1_model_id = model_ids[top1.candidate_id]

    top1_bundle = build_top1_model_bundle(
        top_candidate=top1,
        model_id=top1_model_id,
        run_id=run_id,
        pcs_date=pcs_date,
        schema_gate=schema_gate,
    )

    joblib.dump(
        top1_bundle,
        paths["joblib"],
        compress=3,
    )

    export_evaluation_sidecars(
        paths=paths,
        run_id=run_id,
        pcs_date=pcs_date,
        ranked_checkpoints=ranked_checkpoints,
        holdout_metrics=holdout_metrics,
        holdout_predictions=holdout_predictions,
        split_metadata=split_metadata,
        fold_metadata=fold_metadata,
        source_positive_count=len(cleaned_df),
        eligible_count=len(eligible_df),
        candidate_outcomes=candidate_outcomes,
        schema_gate=schema_gate,
    )

    print()
    print("=" * 78)
    print("TOP MODELS")
    print("=" * 78)

    for rank, candidate in enumerate(top_candidates, start=1):
        model_id = model_ids[candidate.candidate_id]

        print(
            f"Rank {rank}: MODEL_ID={model_id} | "
            f"ENCODING={candidate.encoding_profile} | "
            f"RMSE={candidate.cv_rmse:,.2f} | "
            f"MAE={candidate.cv_mae:,.2f} | "
            f"Adj_R2={candidate.adj_r_squared:.6f}"
        )
        print(
            "         Features: "
            + ", ".join(candidate.selected_source_features)
        )
        print(
            f"         Development OOF eligible cohort: "
            f"RMSE={candidate.cv_eligible_rmse:,.2f} | "
            f"MAE={candidate.cv_eligible_mae:,.2f} | "
            f"n={candidate.cv_eligible_count:,}"
        )

    print()
    print("=" * 78)
    print("EXPORTED FILES")
    print("=" * 78)
    print(f"1) {paths['result']}")
    print(f"2) {paths['coefficient']}")
    print(f"3) {paths['joblib']}  <-- Rank 1 only")
    print(f"4) {paths['evaluation']}")
    print(f"5) {paths['holdout_predictions']}")
    print(f"6) {paths['evaluation_metadata']}")
    print()
    print("[DONE] Training and export completed successfully.")


if __name__ == "__main__":
    main()
