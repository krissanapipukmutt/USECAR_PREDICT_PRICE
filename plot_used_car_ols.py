#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot five diagnostics for an existing Used Car OLS training run.

Does not retrain or change train_used_car_ols.py.
Requires trusted, matching result CSV, coefficient CSV, and top-1 joblib.
Reads latest STG_USED_CAR solely to produce IN-SAMPLE Rank-1 plots.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Save PNG without a GUI/display dependency.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# EDIT THESE SETTINGS ONLY (or override with environment variables).
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(os.getenv(
    "USED_CAR_PROJECT_DIR", "/Users/krissanap/Document/KMUTT/USECAR_PREDICT_PRICE"
)).expanduser()
ARTIFACT_DIR = Path(os.getenv("USED_CAR_ARTIFACT_DIR", str(PROJECT_DIR / "output" / "train"))).expanduser()
GRAPH_DIR = Path(os.getenv("USED_CAR_GRAPH_DIR", str(PROJECT_DIR / "output" / "graphs"))).expanduser()

# None = automatically choose the most recent COMPLETE matching 3-file run.
# Specify e.g. "20260919_145500" to freeze the run you want to visualize.
RUN_ID: str | None = None

# Plotting-only options; all rows are still used for numerical calculations.
PLOT_SAMPLE_N = 6000
HISTOGRAM_BINS = 75
MAX_COEFFICIENTS_IN_FIGURE = 18
FIGURE_DPI = 170
RANDOM_STATE = 42
ALLOW_DIFFERENT_PCS_DATE = False  # Keep False to prevent misleading comparisons.
CURRENT_GRAPH_DIR: Path | None = None  # Assigned after reading current STG PCS_DATE.

# Uses training script's centralized DB config. If localhost times out on Mac,
# run: USED_CAR_DB_SERVER=127.0.0.1 python3 plot_used_car_ols.py


def locate_run() -> tuple[str, Path, Path, Path]:
    """Search train/YYYYMMDD for complete matching runs, newest run first.

    RUN_ID=None selects the latest COMPLETE run across all date folders.
    A specified RUN_ID selects that run regardless of its PCS_DATE folder.
    """
    prefix = "OLS_REGRESSION_RESULT_"
    found = []
    for result in ARTIFACT_DIR.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/" + prefix + "*.csv"):
        run = result.stem[len(prefix):]
        if RUN_ID is not None and run != RUN_ID:
            continue
        folder = result.parent
        coefficient = folder / f"OLS_REGRESSION_COEFFICIENT_{run}.csv"
        model = folder / f"used_car_models_{run}.joblib"
        if all(path.is_file() for path in (result, coefficient, model)):
            found.append((run, result, coefficient, model))
    if not found:
        raise FileNotFoundError(
            f"No complete RESULT + COEFFICIENT + joblib run in {ARTIFACT_DIR}/YYYYMMDD. "
            f"RUN_ID={RUN_ID!r}. Check the train output folder and RUN_ID."
        )
    # RUN_ID timestamp controls recency, not the snapshot date directory.
    return max(found, key=lambda item: item[0])


def load_artifacts(run: str, result_path: Path, coefficient_path: Path, model_path: Path):
    result = pd.read_csv(result_path, encoding="utf-8-sig", dtype={"MODEL_ID": str})
    coefficient = pd.read_csv(coefficient_path, encoding="utf-8-sig", dtype={"MODEL_ID": str})
    if result.empty or coefficient.empty:
        raise ValueError("A result or coefficient CSV is empty.")
    if result["MODEL_ID"].duplicated().any():
        raise ValueError("Duplicate MODEL_IDs in result CSV.")
    if not set(coefficient["MODEL_ID"]).issubset(set(result["MODEL_ID"])):
        raise ValueError("Coefficient MODEL_ID not present in result CSV.")
    # joblib uses pickle internally. Load only your own trusted artifacts.
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("Expected a dictionary-style model bundle in .joblib.")
    if str(bundle.get("run_id")) != run:
        raise ValueError("The joblib RUN_ID does not match CSV RUN_ID.")
    top1_id = str(bundle.get("model_id"))
    result_top1 = result.loc[result["MODEL_ID"] == top1_id]
    if len(result_top1) != 1 or str(bundle.get("model_rank")) != "1":
        raise ValueError("The joblib TOP-1 model is missing/ambiguous in result CSV.")
    if str(result_top1.iloc[0]["TARGET_NAME"]).lower() != str(bundle.get("target_column")).lower():
        raise ValueError("Target in result CSV differs from joblib target.")
    csv_date = pd.to_datetime(result_top1.iloc[0]["TRAIN_PCS_DATE"]).date()
    model_date = pd.to_datetime(bundle.get("pcs_date")).date()
    if csv_date != model_date:
        raise ValueError("Joblib snapshot date differs from result CSV snapshot date.")
    coeff1 = coefficient.loc[coefficient["MODEL_ID"] == top1_id].copy()
    if coeff1.empty:
        raise ValueError("No coefficients for the top-1 model.")
    return result, coefficient, bundle, coeff1, csv_date


def load_latest_snapshot_and_predict(training, bundle: dict, expected_date) -> tuple[np.ndarray, np.ndarray, object]:
    engine = training.build_engine()
    try:
        data = training.load_source_data(engine)
    finally:
        engine.dispose()
    data = training.normalize_required_column_names(data)
    actual_date = training.parse_single_pcs_date(data).date()
    if actual_date != expected_date and not ALLOW_DIFFERENT_PCS_DATE:
        raise ValueError(
            f"STG_USED_CAR PCS_DATE={actual_date}, but model training "
            f"PCS_DATE={expected_date}. This is a different snapshot. "
            "Restore matching STG snapshot before plotting Actual vs Predicted "
            "or explicitly set ALLOW_DIFFERENT_PCS_DATE=True for a new-data evaluation."
        )
    needed = list(bundle["selected_source_features"])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"STG is missing model predictor columns: {missing}")
    target = str(bundle["target_column"])
    y = pd.to_numeric(data[target], errors="coerce")
    valid = y.notna() & np.isfinite(y.astype(float)) & (y > 0)
    data = data.loc[valid].copy()
    if data.empty:
        raise ValueError("No rows with valid positive actual price remain.")
    p = bundle["preprocessor"]
    state = training.PreprocessorState(
        numeric_features=list(p["numeric_features"]),
        categorical_features=list(p["categorical_features"]),
        numeric_medians=dict(p["numeric_medians"]),
        category_levels={k: list(v) for k, v in p["category_levels"].items()},
        reference_categories=dict(p["reference_categories"]),
        encoded_feature_names=list(p["encoded_feature_names"]),
        feature_groups={k: list(v) for k, v in p["feature_groups"].items()},
    )
    # Reuse the exact transformation implementation from the original script.
    encoded = training.transform_with_preprocessor(data, state)
    X = encoded[list(bundle["selected_encoded_features"])]
    X = sm.add_constant(X, has_constant="add")
    # Enforce exact fitted parameter order: protect against accidental name shifts.
    model = bundle["ols_result"]
    fit_names = list(model.params.index)
    if set(X.columns) != set(fit_names):
        raise ValueError(f"Prediction design columns do not match fitted model: {fit_names}")
    X = X[fit_names]
    predicted = np.asarray(model.predict(X), dtype=float)
    actual = y.loc[data.index].to_numpy(dtype=float)
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Prediction produced non-finite values or wrong row count.")
    return actual, predicted, actual_date


def save_plot(name: str) -> None:
    if CURRENT_GRAPH_DIR is None:
        raise RuntimeError("Graph folder is not initialized from STG PCS_DATE.")
    CURRENT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = CURRENT_GRAPH_DIR / name
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {path}", flush=True)


def money(v, _pos=None):
    return f"{v / 1_000_000:,.1f}M"


def plot_actual_vs_predicted(actual, predicted, run, model_id):
    rng = np.random.default_rng(RANDOM_STATE)
    ix = rng.choice(len(actual), size=min(PLOT_SAMPLE_N, len(actual)), replace=False)
    a, p = actual[ix], predicted[ix]
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.scatter(a, p, s=9, alpha=0.23, rasterized=True)
    lo = min(0.0, float(np.min(a)), float(np.min(p)))
    hi = float(np.quantile(np.r_[a, p], .995))
    if hi <= lo:
        hi = max(float(np.max(a)), float(np.max(p)), lo + 1)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", lw=1.3, label="Perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Actual price (THB)")
    ax.set_ylabel("Predicted price (THB)")
    ax.set_title(f"Actual vs Predicted | Top-1 model {model_id}")
    ax.text(.02, .02, f"IN-SAMPLE • run {run} • points shown: {len(ix):,}/{len(actual):,}\n"
            "Axes clipped at 99.5th percentile for readability", transform=ax.transAxes,
            va="bottom", fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(alpha=.15)
    save_plot(f"01_actual_vs_predicted_{run}.png")


def plot_residuals(actual, predicted, run):
    rng = np.random.default_rng(RANDOM_STATE)
    ix = rng.choice(len(actual), size=min(PLOT_SAMPLE_N, len(actual)), replace=False)
    residual = actual - predicted  # positive residual = model underestimates
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.scatter(predicted[ix], residual[ix], s=9, alpha=.22, rasterized=True)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Predicted price (THB)")
    ax.set_ylabel("Residual = Actual - Predicted (THB)")
    ax.set_title("Residual plot | Top-1 model (IN-SAMPLE)")
    ax.text(.02, .98, f"run {run} • {len(ix):,} sampled of {len(actual):,} rows",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(alpha=.15)
    save_plot(f"02_residual_plot_{run}.png")


def plot_model_comparison(result, run):
    result = result.copy()
    result["MODEL_RANK"] = result["MODEL_NAME"].astype(str).str.extract(r"RANK_(\d+)")[0]
    result["MODEL_RANK"] = pd.to_numeric(result["MODEL_RANK"], errors="coerce")
    result = result.sort_values(["MODEL_RANK", "RMSE"], na_position="last")
    labels = [f"Rank {int(x)}" if pd.notna(x) else mid
              for x, mid in zip(result["MODEL_RANK"], result["MODEL_ID"])]
    x = np.arange(len(result))
    width = .37
    fig, ax = plt.subplots(figsize=(max(8, 2.3 * len(result) + 3), 6))
    ax.bar(x - width / 2, result["RMSE"].astype(float), width, label="CV RMSE")
    ax.bar(x + width / 2, result["MAE"].astype(float), width, label="CV MAE")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cross-validation error (THB)")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_title("Top models | CV RMSE and CV MAE")
    ax.legend()
    ax.grid(axis="y", alpha=.17)
    # Adjusted R² is TRAIN-fit R²; do not interpret as cross-validation R².
    for i, (_, row) in enumerate(result.iterrows()):
        ax.text(i, max(float(row["RMSE"]), float(row["MAE"])) * 1.045,
                f"Train Adj R²={float(row['ADJ_R_SQUARED']):.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, float(result[["RMSE", "MAE"]].max().max()) * 1.23)
    fig.text(.5, .01, "CSV RMSE/MAE = mean of 5-fold CV; Adjusted R² = full-data fit (different evaluation basis).",
             ha="center", fontsize=9)
    save_plot(f"03_model_comparison_{run}.png")


def plot_price_distribution(actual, predicted, run):
    # Use the same edges for real and predicted prices; show clipped tail explicitly.
    upper = float(np.quantile(np.r_[actual, predicted], .99))
    upper = max(upper, 1.0)
    edges = np.linspace(0, upper, HISTOGRAM_BINS + 1)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.hist(actual[(actual >= 0) & (actual <= upper)], bins=edges,
            alpha=.6, label="Actual price")
    ax.hist(predicted[(predicted >= 0) & (predicted <= upper)], bins=edges,
            alpha=.5, label="Predicted price (in-sample)")
    ax.set_xlabel("Price (THB)")
    ax.set_ylabel("Listings")
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_title("Actual vs Predicted | Price distribution")
    ax.text(.98, .97, f"run {run}\nClipped display to 0–P99; negative predictions omitted",
            ha="right", va="top", transform=ax.transAxes, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=.17)
    save_plot(f"04_price_distribution_{run}.png")


def plot_coefficient_ci(coefficient_rank1, result_top1, run):
    data = coefficient_rank1.copy()
    data = data.loc[
        (data["IS_REFERENCE"].astype(str).str.upper() != "Y")
        & (data["FEATURE_TYPE"].astype(str).str.upper() != "INTERCEPT")
        & data["CI_LOWER"].notna() & data["CI_UPPER"].notna()
        & data["COEFFICIENT"].notna()
    ].copy()
    if data.empty:
        raise ValueError("No coefficient CI data to plot; check columns CI_LOWER and CI_UPPER.")
    for col in ("COEFFICIENT", "CI_LOWER", "CI_UPPER"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["COEFFICIENT", "CI_LOWER", "CI_UPPER"])
    # Sort by absolute raw coefficient for display only: coefficients of different
    # units/categories are NOT comparable measures of predictor importance.
    data = data.assign(abs_coef=data["COEFFICIENT"].abs())
    data = data.sort_values("abs_coef", ascending=False).head(MAX_COEFFICIENTS_IN_FIGURE)
    data = data.sort_values("COEFFICIENT")
    labels = data["FEATURE_NAME"].fillna("(unnamed feature)").astype(str).tolist()
    if len(data) != len(labels):
        raise ValueError("Coefficient plot labels mismatch.")
    y = np.arange(len(data))
    estimates = data["COEFFICIENT"].to_numpy(dtype=float)
    lower = data["CI_LOWER"].to_numpy(dtype=float)
    upper = data["CI_UPPER"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, max(5.5, .40 * len(data) + 2.0)))
    ax.errorbar(estimates, y, xerr=[np.maximum(estimates - lower, 0),
                                     np.maximum(upper - estimates, 0)],
                fmt="o", capsize=3, ms=4, elinewidth=1.3)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("OLS coefficient and confidence interval (THB per feature unit)")
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    level = float(result_top1["CONFIDENCE_LEVEL"]) * 100 if "CONFIDENCE_LEVEL" in result_top1 else 95
    ax.set_title(f"Top-1 coefficients | {level:g}% Confidence Intervals")
    ax.grid(axis="x", alpha=.15)
    fig.text(.5, .01, "Top absolute coefficients for display only. Units differ; intercept/reference omitted.",
             ha="center", fontsize=9)
    save_plot(f"05_coefficient_confidence_interval_{run}.png")


def main():
    global CURRENT_GRAPH_DIR
    if not PROJECT_DIR.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {PROJECT_DIR}")
    sys.path.insert(0, str(PROJECT_DIR))
    import train_used_car_ols as training  # Only definitions are imported; main() does not run.

    run, result_file, coef_file, model_file = locate_run()
    result, coefficient, bundle, rank1_coef, snapshot_date = load_artifacts(
        run, result_file, coef_file, model_file
    )
    top_id = str(bundle["model_id"])
    print(f"[INFO] Run={run}, Rank-1 MODEL_ID={top_id}, snapshot={snapshot_date}", flush=True)
    print("[INFO] Reading STG and predicting in-sample; no retraining.", flush=True)
    actual, predicted, stg_pcs_date = load_latest_snapshot_and_predict(training, bundle, snapshot_date)
    CURRENT_GRAPH_DIR = GRAPH_DIR / stg_pcs_date.strftime("%Y%m%d")
    print(f"[INFO] Predicted {len(actual):,} rows from STG_USED_CAR.", flush=True)

    plot_actual_vs_predicted(actual, predicted, run, top_id)
    plot_residuals(actual, predicted, run)
    plot_model_comparison(result, run)
    plot_price_distribution(actual, predicted, run)
    result_top1 = result.loc[result["MODEL_ID"] == top_id].iloc[0]
    plot_coefficient_ci(rank1_coef, result_top1, run)
    print(f"[DONE] Saved five PNG charts in: {CURRENT_GRAPH_DIR}", flush=True)
    print("[NOTE] Actual/Predicted, Residual, Distribution: IN-SAMPLE training snapshot."
          " Model comparison: CV RMSE/MAE from CSV; CI: fitted coefficients.")


if __name__ == "__main__":
    main()
