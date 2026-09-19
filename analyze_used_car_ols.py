#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Used-car Rank-1 OLS error diagnostics (IN-SAMPLE, no retraining).

Reads current dbo.STG_USED_CAR, uses matching saved Top-1 joblib, and writes
CSV diagnostics to output/analysis/<STG_PCS_DATE YYYYMMDD>/.
IMPORTANT: These are IN-SAMPLE error descriptions, NOT out-of-fold/test metrics.
Run only on your own trusted .joblib files (joblib/pickle can execute code).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ===================== CHANGE SETTINGS HERE ================================
PROJECT_DIR = Path(os.getenv(
    "USED_CAR_PROJECT_DIR", "/Users/krissanap/Document/KMUTT/USECAR_PREDICT_PRICE"
)).expanduser()
TRAIN_DIR = Path(os.getenv("USED_CAR_ARTIFACT_DIR", str(PROJECT_DIR / "output" / "train"))).expanduser()
ANALYSIS_DIR = Path(os.getenv("USED_CAR_ANALYSIS_DIR", str(PROJECT_DIR / "output" / "analysis"))).expanduser()
RUN_ID: str | None = "20260920_011528"  # Set None for latest COMPLETE 3-file run.
TOP_ERRORS = 100
# Fixed bins, THB; 0..500k, 500k..1M, 1M..3M, >=3M.
PRICE_BINS = [0, 500_000, 1_000_000, 3_000_000, float("inf")]
PRICE_LABELS = ["<500k", "500k-1M", "1M-3M", ">=3M"]
# Keep False: STG snapshot must be the SAME training snapshot for in-sample audit.
ALLOW_DIFFERENT_PCS_DATE = False
# ============================================================================


def locate_run() -> tuple[str, Path, Path, Path]:
    hits: list[tuple[str, Path, Path, Path]] = []
    for result_path in TRAIN_DIR.glob("[0-9]" * 8 + "/OLS_REGRESSION_RESULT_*.csv"):
        run = result_path.stem.removeprefix("OLS_REGRESSION_RESULT_")
        if RUN_ID is not None and run != RUN_ID:
            continue
        folder = result_path.parent
        coef = folder / f"OLS_REGRESSION_COEFFICIENT_{run}.csv"
        model = folder / f"used_car_models_{run}.joblib"
        if coef.is_file() and model.is_file():
            hits.append((run, result_path, coef, model))
    if not hits:
        raise FileNotFoundError(
            f"No complete training run under {TRAIN_DIR}/YYYYMMDD. RUN_ID={RUN_ID!r}. "
            "Ensure the result CSV, coefficient CSV and joblib are in the same folder."
        )
    return max(hits, key=lambda row: row[0])


def load_model(run: str, result_file: Path, coef_file: Path, model_file: Path):
    result = pd.read_csv(result_file, dtype={"MODEL_ID": str}, encoding="utf-8-sig")
    coef = pd.read_csv(coef_file, dtype={"MODEL_ID": str}, encoding="utf-8-sig")
    if result.empty or coef.empty or result.MODEL_ID.duplicated().any():
        raise ValueError("Training CSV is empty or MODEL_ID duplicated.")
    bundle = joblib.load(model_file)  # ONLY load your own trusted .joblib.
    if not isinstance(bundle, dict) or bundle.get("run_id") != run or bundle.get("model_rank") != 1:
        raise ValueError("Joblib does not match the selected Rank-1 run.")
    model_id = str(bundle["model_id"])
    rank1 = result.loc[result.MODEL_ID == model_id]
    if len(rank1) != 1 or not (coef.MODEL_ID == model_id).any():
        raise ValueError("Rank-1 joblib MODEL_ID not found in both CSVs.")
    pcs = pd.to_datetime(bundle["pcs_date"]).date()
    if pd.to_datetime(rank1.iloc[0]["TRAIN_PCS_DATE"]).date() != pcs:
        raise ValueError("Joblib and result CSV TRAIN_PCS_DATE do not match.")
    if str(rank1.iloc[0]["TARGET_NAME"]).lower() != str(bundle["target_column"]).lower():
        raise ValueError("Model target is not the same as CSV TARGET_NAME.")
    return bundle, pcs


def get_preprocessor(training, bundle):
    p = bundle["preprocessor"]
    return training.PreprocessorState(
        numeric_features=list(p["numeric_features"]),
        categorical_features=list(p["categorical_features"]),
        numeric_medians=dict(p["numeric_medians"]),
        category_levels={k: list(v) for k, v in p["category_levels"].items()},
        reference_categories=dict(p["reference_categories"]),
        encoded_feature_names=list(p["encoded_feature_names"]),
        feature_groups={k: list(v) for k, v in p["feature_groups"].items()},
    )


def add_category_flags(df: pd.DataFrame, training, state) -> pd.DataFrame:
    """Flag exact category mapping used by TRAIN, including missing->OTHER."""
    for source in ("brand", "model", "sub_model"):
        flag = f"{source}_IS_OTHER"
        mapped = f"{source}_MODEL_CATEGORY"
        if source not in state.categorical_features:
            df[flag] = pd.NA
            df[mapped] = pd.NA
            continue
        # Training transform uses case-sensitive column names and these rules.
        values = training.normalize_category_series(df[source])
        levels = state.category_levels[source]
        reference = state.reference_categories[source]
        fallback = "__OTHER__" if "__OTHER__" in levels else reference
        values = values.where(values.isin(levels), fallback)
        df[mapped] = values
        df[flag] = values.eq("__OTHER__")
    return df


def metric_table(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Uses actual individual errors; NOT equivalent to mean of fold-level CV metrics."""
    output = []
    for key, part in df.groupby(groups, dropna=False, observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        output.append({
            **dict(zip(groups, key)),
            "N_CARS": len(part),
            "MAE_THB": part.ABS_ERROR.mean(),
            "RMSE_THB": np.sqrt(np.mean(np.square(part.ERROR_THB.to_numpy(dtype=float)))),
            "MEAN_SIGNED_ERROR_THB": part.ERROR_THB.mean(),
            "MEDIAN_ABS_ERROR_THB": part.ABS_ERROR.median(),
            "NEGATIVE_PREDICTION_COUNT": int((part.PREDICTED_PRICE < 0).sum()),
            "ACTUAL_AVG_THB": part.ACTUAL_PRICE.mean(),
            "PREDICTED_AVG_THB": part.PREDICTED_PRICE.mean(),
        })
    return pd.DataFrame(output)


def write_csv(df: pd.DataFrame, folder: Path, filename: str):
    path = folder / filename
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[SAVED] {path} ({len(df):,} rows)", flush=True)


def main():
    if not PROJECT_DIR.is_dir():
        raise FileNotFoundError(f"Project directory not found: {PROJECT_DIR}")
    sys.path.insert(0, str(PROJECT_DIR))
    import train_used_car_ols as training  # Does not run training.main().

    run, results_file, coef_file, model_file = locate_run()
    bundle, model_pcs_date = load_model(run, results_file, coef_file, model_file)
    engine = training.build_engine()
    try:
        df = training.load_source_data(engine)
    finally:
        engine.dispose()
    df = training.normalize_required_column_names(df)
    stg_pcs_date = training.parse_single_pcs_date(df).date()
    if model_pcs_date != stg_pcs_date:
        if not ALLOW_DIFFERENT_PCS_DATE:
            raise ValueError(
                f"STG PCS_DATE={stg_pcs_date} != TRAIN PCS_DATE={model_pcs_date}. "
                "The STG table holds only the latest snapshot. "
                "Restore the original STG snapshot or choose a matching run. "
                "Do not label new-snapshot diagnostics as in-sample."
            )
        raise ValueError("Cross-snapshot evaluation is not supported in this in-sample script.")

    # Match training's invalid-target removal AND numeric-like conversion.
    df = training.clean_target(df)
    df = training.maybe_convert_numeric_like_columns(df)
    missing = [c for c in bundle["selected_source_features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing selected model features in STG: {missing}")
    state = get_preprocessor(training, bundle)
    X = training.transform_with_preprocessor(df, state)
    X = sm.add_constant(X[list(bundle["selected_encoded_features"])], has_constant="add")
    fitted = bundle["ols_result"]
    fit_names = list(fitted.params.index)
    if set(X.columns) != set(fit_names):
        raise ValueError("Model input columns and trained parameters differ.")
    predicted = np.asarray(fitted.predict(X[fit_names]), dtype=float)
    actual = pd.to_numeric(df[str(bundle["target_column"])], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite predicted/actual price detected.")

    report = pd.DataFrame(index=df.index)
    for source in ("listing_id", "brand", "model", "sub_model", "model_year", "mileage"):
        report[source] = df[source] if source in df.columns else pd.NA
    report["ACTUAL_PRICE"] = actual
    report["PREDICTED_PRICE"] = predicted
    report["ERROR_THB"] = report.ACTUAL_PRICE - report.PREDICTED_PRICE
    report["ABS_ERROR"] = report.ERROR_THB.abs()
    report["APE_PCT"] = report.ABS_ERROR / report.ACTUAL_PRICE * 100
    report["PRICE_BAND"] = pd.cut(
        report.ACTUAL_PRICE, bins=PRICE_BINS, labels=PRICE_LABELS,
        right=False, include_lowest=True,
    )
    report = add_category_flags(report, training, state)
    flag_cols = [f"{source}_IS_OTHER" for source in ("brand", "model", "sub_model")]
    # Any selected predictor grouped into OTHER. Missing optional features stay NA.
    report["ANY_OTHER"] = report[flag_cols].fillna(False).astype(bool).any(axis=1)

    dest = ANALYSIS_DIR / stg_pcs_date.strftime("%Y%m%d")
    dest.mkdir(parents=True, exist_ok=True)
    suffix = f"{run}.csv"
    write_csv(metric_table(report, ["PRICE_BAND"]), dest, f"01_error_by_price_band_{suffix}")
    write_csv(report.nlargest(TOP_ERRORS, "ABS_ERROR"), dest, f"02_top_{TOP_ERRORS}_errors_{suffix}")
    negative = report.loc[report.PREDICTED_PRICE < 0].sort_values("PREDICTED_PRICE")
    write_csv(negative, dest, f"03_negative_predictions_{suffix}")
    write_csv(metric_table(report, ["ANY_OTHER"]), dest, f"04_any_other_vs_retained_{suffix}")
    grouped = []
    for source in ("brand", "model", "sub_model"):
        if source not in state.categorical_features:
            continue
        segment = metric_table(report, [f"{source}_IS_OTHER"])
        segment.insert(0, "SOURCE_FEATURE", source)
        segment = segment.rename(columns={f"{source}_IS_OTHER": "IS_OTHER"})
        grouped.append(segment)
    write_csv(pd.concat(grouped, ignore_index=True) if grouped else pd.DataFrame(),
              dest, f"05_other_by_feature_{suffix}")
    write_csv(report, dest, f"06_all_in_sample_predictions_{suffix}")

    overview = pd.DataFrame([{
        "RUN_ID": run,
        "MODEL_ID": bundle["model_id"],
        "PCS_DATE": stg_pcs_date.isoformat(),
        "EVALUATION_TYPE": "IN_SAMPLE_FULL_DATA_FIT",
        "N_CARS": len(report),
        "IN_SAMPLE_MAE_THB": report.ABS_ERROR.mean(),
        "IN_SAMPLE_RMSE_THB": np.sqrt(np.mean(np.square(report.ERROR_THB.to_numpy(dtype=float)))),
        "NEGATIVE_PREDICTION_COUNT": len(negative),
        "NEGATIVE_PREDICTION_PCT": 100 * len(negative) / len(report),
        "ANY_OTHER_COUNT": int(report.ANY_OTHER.sum()),
        "ANY_OTHER_PCT": 100 * report.ANY_OTHER.mean(),
        "NOTE": "IN-SAMPLE diagnostics only; do NOT compare directly with CV or test metrics.",
    }])
    write_csv(overview, dest, f"00_overview_{suffix}")
    print(f"[DONE] Run={run}; STG PCS_DATE={stg_pcs_date}; rows={len(report):,}")
    print(f"[WARNING] All errors are IN-SAMPLE; not out-of-fold/test performance. Folder: {dest}")


if __name__ == "__main__":
    main()
