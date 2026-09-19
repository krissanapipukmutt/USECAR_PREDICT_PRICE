#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import math
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =============================================================================
# 1) ENVIRONMENT CONFIG
# =============================================================================

DB_SERVER = os.getenv("USED_CAR_DB_SERVER", "localhost")
DB_PORT = int(os.getenv("USED_CAR_DB_PORT", "1433"))
DB_DATABASE = os.getenv("USED_CAR_DB_DATABASE", "USED_CAR_DB")
DB_USER = os.getenv("USED_CAR_DB_USER", "sa")
DB_PASSWORD = os.getenv(
    "USED_CAR_DB_PASSWORD",
    "Local_Dev_Only_Pa55word!",
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
        "/Users/krissanap/Document/KMUTT/USECAR_PREDICT_PRICE/output",
    )
)

# =============================================================================
# 2) MODEL CONFIG
# =============================================================================

TARGET_COLUMN = "price"
PCS_DATE_COLUMN = "PCS_DATE"

TOP_N_MODELS = 3
P_VALUE_THRESHOLD = 0.05

# Two-sided confidence intervals for the fitted OLS coefficients.
# Independent of P_VALUE_THRESHOLD (which controls feature selection).
CONFIDENCE_LEVEL = 0.95

CV_FOLDS = 5
RANDOM_STATE = 42

MAX_CANDIDATES = 14
CANDIDATE_SUBSET_FRACTIONS = (0.90, 0.80, 0.70, 0.60, 0.50)
CANDIDATES_PER_FRACTION = 2

MAX_MISSING_RATIO = 0.95
MAX_CATEGORICAL_LEVELS = 60
MAX_CATEGORICAL_UNIQUE_RATIO = 0.50
MIN_CATEGORY_COUNT = 20
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

        unique_ratio = unique_count / max(len(non_null), 1)

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
) -> PreprocessorState:
    numeric_features = []
    categorical_features = []
    numeric_medians: Dict[str, float] = {}
    category_levels: Dict[str, List[str]] = {}
    reference_categories: Dict[str, str] = {}
    feature_groups: Dict[str, List[str]] = {}

    for col in source_features:
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            values = pd.to_numeric(s, errors="coerce").astype(float)
            median = values.median()
            if pd.isna(median):
                continue

            numeric_features.append(col)
            numeric_medians[col] = float(median)
            feature_groups[col] = [col]
            continue

        values = normalize_category_series(s)
        counts = values.value_counts(dropna=False)

        rare_levels = set(
            counts[counts < MIN_CATEGORY_COUNT].index.astype(str)
        )
        if rare_levels:
            values = values.where(
                ~values.isin(rare_levels),
                "__OTHER__",
            )
            counts = values.value_counts(dropna=False)

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
    )


def transform_with_preprocessor(
    df: pd.DataFrame,
    state: PreprocessorState,
) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)

    for col in state.numeric_features:
        values = pd.to_numeric(df[col], errors="coerce").astype(float)
        values = values.replace([np.inf, -np.inf], np.nan)
        output[col] = values.fillna(state.numeric_medians[col])

    for col in state.categorical_features:
        values = normalize_category_series(df[col])

        levels = state.category_levels[col]
        reference = state.reference_categories[col]
        fallback = "__OTHER__" if "__OTHER__" in levels else reference
        values = values.where(values.isin(levels), fallback)

        for level in levels:
            if level == reference:
                continue
            feature_name = safe_feature_name(col, level)
            output[feature_name] = (values == level).astype(float)

    for col in state.encoded_feature_names:
        if col not in output.columns:
            output[col] = 0.0

    return output[state.encoded_feature_names].astype(float)


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
    features = list(all_features)
    if not features:
        return []

    target_candidate_count = max(
        TOP_N_MODELS * 3,
        min(MAX_CANDIDATES, 6),
    )
    target_candidate_count = min(MAX_CANDIDATES, target_candidate_count)

    candidates: List[Tuple[str, ...]] = []
    seen = set()

    def add_candidate(cols: Sequence[str]):
        key = tuple(sorted(set(cols)))
        if len(key) < MIN_SOURCE_FEATURES:
            return
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    add_candidate(features)
    rng = np.random.default_rng(RANDOM_STATE)

    for fraction in CANDIDATE_SUBSET_FRACTIONS:
        subset_size = max(
            MIN_SOURCE_FEATURES,
            int(math.ceil(len(features) * fraction)),
        )
        subset_size = min(subset_size, len(features))

        for _ in range(CANDIDATES_PER_FRACTION):
            if subset_size == len(features):
                add_candidate(features)
                continue

            subset = rng.choice(
                features,
                size=subset_size,
                replace=False,
            ).tolist()
            add_candidate(subset)

            if len(candidates) >= target_candidate_count:
                break

        if len(candidates) >= target_candidate_count:
            break

    if len(candidates) < target_candidate_count and len(features) > 1:
        shuffled = list(features)
        rng.shuffle(shuffled)

        for drop_col in shuffled:
            add_candidate([c for c in features if c != drop_col])
            if len(candidates) >= target_candidate_count:
                break

    return [list(x) for x in candidates[:MAX_CANDIDATES]]


def evaluate_candidate_cv(
    df: pd.DataFrame,
    initial_source_features: Sequence[str],
) -> Tuple[float, float]:
    y_all = df[TARGET_COLUMN].astype(float)

    kfold = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rmses: List[float] = []
    maes: List[float] = []

    for train_idx, valid_idx in kfold.split(df):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        y_train = y_all.iloc[train_idx]
        y_valid = y_all.iloc[valid_idx]

        state = fit_preprocessor(train_df, initial_source_features)

        X_train = transform_with_preprocessor(train_df, state)
        X_valid = transform_with_preprocessor(valid_df, state)

        if X_train.shape[1] == 0:
            raise RuntimeError(
                "Candidate produced no encoded predictors."
            )

        model, _, selected_encoded, _ = (
            backward_eliminate_source_features(
                X_train,
                y_train,
                state.feature_groups,
            )
        )

        X_valid_selected = X_valid[selected_encoded]
        X_valid_const = sm.add_constant(
            X_valid_selected,
            has_constant="add",
        )

        pred = np.asarray(model.predict(X_valid_const), dtype=float)

        rmse = math.sqrt(mean_squared_error(y_valid, pred))
        mae = mean_absolute_error(y_valid, pred)

        rmses.append(float(rmse))
        maes.append(float(mae))

    return float(np.mean(rmses)), float(np.mean(maes))


def fit_final_candidate(
    df: pd.DataFrame,
    candidate_id: int,
    initial_source_features: Sequence[str],
    cv_rmse: float,
    cv_mae: float,
) -> FinalCandidate:
    state = fit_preprocessor(df, initial_source_features)

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
    )


def rank_candidates(
    candidates: Sequence[FinalCandidate],
) -> List[FinalCandidate]:
    best_by_structure: Dict[Tuple[str, ...], FinalCandidate] = {}

    for candidate in candidates:
        structure = tuple(sorted(candidate.selected_source_features))

        if structure not in best_by_structure:
            best_by_structure[structure] = candidate
            continue

        current = best_by_structure[structure]

        new_key = (
            candidate.cv_rmse,
            candidate.cv_mae,
            -candidate.adj_r_squared,
        )
        current_key = (
            current.cv_rmse,
            current.cv_mae,
            -current.adj_r_squared,
        )

        if new_key < current_key:
            best_by_structure[structure] = candidate

    unique_candidates = list(best_by_structure.values())

    unique_candidates.sort(
        key=lambda c: (
            c.cv_rmse,
            c.cv_mae,
            -c.adj_r_squared,
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
    n_observation: int,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    model_ids: Dict[int, str] = {}
    rows = []

    pcs_date_str = pcs_date.date().isoformat()
    train_date_str = run_datetime.strftime("%Y-%m-%d %H:%M:%S")

    for rank, candidate in enumerate(top_candidates, start=1):
        model_id = model_id_for_rank(run_datetime, rank)
        model_ids[candidate.candidate_id] = model_id

        rows.append(
            {
                "MODEL_ID": model_id,
                "MODEL_NAME": f"{MODEL_NAME_PREFIX}_RANK_{rank}",
                "TARGET_NAME": TARGET_COLUMN,
                "TRAIN_PCS_DATE": pcs_date_str,
                "TRAIN_DATE": train_date_str,
                "N_OBSERVATION": int(n_observation),
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
) -> dict:
    state = top_candidate.preprocessor

    bundle = {
        "artifact_version": "1.0",
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
        "source_feature_p_values": top_candidate.source_feature_p_values,
        "preprocessor": {
            "numeric_features": state.numeric_features,
            "categorical_features": state.categorical_features,
            "numeric_medians": state.numeric_medians,
            "category_levels": state.category_levels,
            "reference_categories": state.reference_categories,
            "encoded_feature_names": state.encoded_feature_names,
            "feature_groups": state.feature_groups,
        },
        "ols_result": top_candidate.ols_result,
    }

    return bundle


def main() -> None:
    if not 0.0 < CONFIDENCE_LEVEL < 1.0:
        raise ValueError("CONFIDENCE_LEVEL must be between 0 and 1 (exclusive).")

    warnings.filterwarnings(
        "ignore",
        message="covariance of constraints does not have full rank",
    )

    run_datetime = datetime.now()
    run_id = run_datetime.strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("USED CAR OLS TRAINING")
    print("=" * 78)
    print(f"[CONFIG] SQL Server : {DB_SERVER}:{DB_PORT} / {DB_DATABASE}")
    print(f"[CONFIG] Source     : {DB_SCHEMA}.{SOURCE_TABLE}")
    print(f"[CONFIG] Target     : {TARGET_COLUMN}")
    print(f"[CONFIG] TOP N      : {TOP_N_MODELS}")
    print(f"[CONFIG] P-value    : <= {P_VALUE_THRESHOLD}")
    print(f"[CONFIG] Confidence : {CONFIDENCE_LEVEL:.0%}")
    print(f"[CONFIG] CV folds   : {CV_FOLDS}")
    print(f"[CONFIG] Output     : {OUTPUT_DIR.resolve()}")

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

    df = clean_target(df)
    df = maybe_convert_numeric_like_columns(df)

    eligible_features, excluded_info = infer_eligible_features(df)

    print(
        f"[INFO] Eligible predictors ({len(eligible_features)}): "
        + ", ".join(eligible_features)
    )

    if excluded_info:
        print("[INFO] Excluded columns:")
        for col, reason in excluded_info:
            print(f"       - {col}: {reason}")

    initial_candidates = generate_candidate_feature_sets(
        eligible_features
    )

    if len(initial_candidates) < TOP_N_MODELS:
        raise RuntimeError(
            f"Only {len(initial_candidates)} candidate feature sets could be "
            f"generated, but TOP_N_MODELS={TOP_N_MODELS}."
        )

    print(
        f"[INFO] Generated {len(initial_candidates)} candidate feature sets."
    )

    fitted_candidates: List[FinalCandidate] = []

    for candidate_id, feature_set in enumerate(
        initial_candidates,
        start=1,
    ):
        print(
            f"[TRAIN] Candidate {candidate_id}/{len(initial_candidates)} "
            f"({len(feature_set)} source features)"
        )

        try:
            cv_rmse, cv_mae = evaluate_candidate_cv(
                df,
                feature_set,
            )

            final_candidate = fit_final_candidate(
                df=df,
                candidate_id=candidate_id,
                initial_source_features=feature_set,
                cv_rmse=cv_rmse,
                cv_mae=cv_mae,
            )

            fitted_candidates.append(final_candidate)

            print(
                f"        CV_RMSE={cv_rmse:,.2f} | "
                f"CV_MAE={cv_mae:,.2f} | "
                f"Adj_R2={final_candidate.adj_r_squared:.6f} | "
                f"SelectedFeatures="
                f"{len(final_candidate.selected_source_features)}"
            )

        except Exception as exc:
            print(
                f"[WARN] Candidate {candidate_id} skipped: {exc}"
            )

    if not fitted_candidates:
        raise RuntimeError(
            "No candidate model completed successfully."
        )

    ranked = rank_candidates(fitted_candidates)

    if len(ranked) < TOP_N_MODELS:
        print(
            f"[WARN] Only {len(ranked)} unique valid model structures were "
            f"found; requested TOP_N_MODELS={TOP_N_MODELS}."
        )

    top_candidates = ranked[:TOP_N_MODELS]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df, model_ids = build_result_dataframe(
        top_candidates=top_candidates,
        run_datetime=run_datetime,
        pcs_date=pcs_date,
        n_observation=len(df),
    )

    coefficient_df = build_coefficient_dataframe(
        top_candidates=top_candidates,
        model_ids=model_ids,
        pcs_date=pcs_date,
    )

    result_path = (
        OUTPUT_DIR
        / f"OLS_REGRESSION_RESULT_{run_id}.csv"
    )
    coefficient_path = (
        OUTPUT_DIR
        / f"OLS_REGRESSION_COEFFICIENT_{run_id}.csv"
    )
    joblib_path = (
        OUTPUT_DIR
        / f"used_car_models_{run_id}.joblib"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )
    coefficient_df.to_csv(
        coefficient_path,
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
    )

    joblib.dump(
        top1_bundle,
        joblib_path,
        compress=3,
    )

    print()
    print("=" * 78)
    print("TOP MODELS")
    print("=" * 78)

    for rank, candidate in enumerate(top_candidates, start=1):
        model_id = model_ids[candidate.candidate_id]

        print(
            f"Rank {rank}: MODEL_ID={model_id} | "
            f"RMSE={candidate.cv_rmse:,.2f} | "
            f"MAE={candidate.cv_mae:,.2f} | "
            f"Adj_R2={candidate.adj_r_squared:.6f}"
        )
        print(
            "         Features: "
            + ", ".join(candidate.selected_source_features)
        )

    print()
    print("=" * 78)
    print("EXPORTED FILES")
    print("=" * 78)
    print(f"1) {result_path}")
    print(f"2) {coefficient_path}")
    print(f"3) {joblib_path}  <-- Rank 1 only")
    print()
    print("[DONE] Training and export completed successfully.")


if __name__ == "__main__":
    main()
