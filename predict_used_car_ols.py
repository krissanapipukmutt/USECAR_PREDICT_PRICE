#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict used-car prices with a previously trained Rank-1 OLS .joblib.

Does NOT connect to SQL Server, retrain, or change the database.
Load only .joblib files that you created/trust: joblib deserialization can run code.
Run this from the project containing the matching train_used_car_ols.py.
"""
from __future__ import annotations

import argparse
import re
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_DIR = Path(os.getenv(
    "USED_CAR_PROJECT_DIR", "/Users/krissanap/Document/KMUTT/USECAR_PREDICT_PRICE"
)).expanduser()
TRAIN_DIR = Path(os.getenv(
    "USED_CAR_ARTIFACT_DIR", str(PROJECT_DIR / "output" / "train")
)).expanduser()
PREDICT_DIR = Path(os.getenv(
    "USED_CAR_PREDICT_DIR", str(PROJECT_DIR / "output" / "predict")
)).expanduser()


def locate_bundle(run_id: str | None) -> Path:
    files = list(TRAIN_DIR.glob("[0-9]" * 8 + "/used_car_models_*.joblib"))
    if run_id:
        files = [p for p in files if p.stem == f"used_car_models_{run_id}"]
    if not files:
        raise FileNotFoundError(
            f"No saved model found under {TRAIN_DIR}/YYYYMMDD/ "
            f"for RUN_ID={run_id!r}."
        )
    return max(files, key=lambda p: p.stem)


def load_bundle(path: Path) -> dict:
    # The matching training module must be importable for unpickling.
    sys.path.insert(0, str(PROJECT_DIR))
    import train_used_car_ols as training

    bundle = joblib.load(path)  # Only your own trusted training artifacts.
    keys = {"run_id", "model_id", "model_rank", "pcs_date", "preprocessor",
            "selected_source_features", "selected_encoded_features", "ols_result"}
    if not isinstance(bundle, dict) or not keys.issubset(bundle):
        raise ValueError("Unexpected .joblib bundle structure; expected saved OLS Rank-1 model.")
    if bundle["model_rank"] != 1:
        raise ValueError("This script supports Rank-1 .joblib bundles only.")
    if path.stem != f"used_car_models_{bundle['run_id']}":
        raise ValueError("Run ID in the file name differs from run ID in the bundle.")
    expected_folder = pd.to_datetime(bundle["pcs_date"]).strftime("%Y%m%d")
    if path.parent.name != expected_folder:
        raise ValueError("Model PCS_DATE and training folder date do not match.")
    return bundle, training


def make_state(bundle: dict, training):
    # The bundle keeps preprocessing metadata for ALL candidate features,
    # including predictors eliminated before the final OLS fit.  Construct
    # a view of that metadata restricted to the saved model's inputs: a
    # caller should not have to supply eliminated fields such as transmission.
    data = bundle["preprocessor"]
    selected_sources = list(bundle["selected_source_features"])
    selected_encoded = list(bundle["selected_encoded_features"])
    groups = {name: list(data["feature_groups"][name]) for name in selected_sources}
    available_encoded = {name for names in groups.values() for name in names}
    unknown = [name for name in selected_encoded if name not in available_encoded]
    if unknown:
        raise ValueError(f"Saved model has encoded features absent from preprocessing metadata: {unknown}")
    numeric = [name for name in data["numeric_features"] if name in groups]
    categorical = [name for name in data["categorical_features"] if name in groups]
    if set(numeric + categorical) != set(selected_sources):
        raise ValueError("Saved model source features are inconsistent with preprocessing metadata.")
    return training.PreprocessorState(
        numeric_features=numeric,
        categorical_features=categorical,
        numeric_medians={name: data["numeric_medians"][name] for name in numeric},
        category_levels={name: list(data["category_levels"][name]) for name in categorical},
        reference_categories={name: data["reference_categories"][name] for name in categorical},
        encoded_feature_names=[name for name in data["encoded_feature_names"] if name in available_encoded],
        feature_groups=groups,
    )


def predict(df: pd.DataFrame, bundle: dict, training) -> pd.DataFrame:
    state = make_state(bundle, training)
    required = list(bundle["selected_source_features"])
    if df.empty:
        raise ValueError("No input cars were provided.")
    if len(set(str(c).casefold() for c in df.columns)) != len(df.columns):
        raise ValueError("Input contains columns with duplicate case-insensitive names.")
    # Case-insensitive input columns, preserve exact names used during training.
    lookup = {str(c).casefold(): c for c in df.columns}
    missing = [c for c in required if c.casefold() not in lookup]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}. Use --write-template to see them.")
    df = df.rename(columns={lookup[c.casefold()]: c for c in required}).copy()
    if df[required].isna().all(axis=1).any():
        raise ValueError("At least one input car has all model features missing.")
    encoded = training.transform_with_preprocessor(df, state)
    selected = list(bundle["selected_encoded_features"])
    absent = [name for name in selected if name not in encoded.columns]
    if absent:
        raise ValueError(f"Saved encoder did not produce these model columns: {absent}")
    design = sm.add_constant(encoded[selected].astype(float), has_constant="add")
    fit = bundle["ols_result"]
    fit_columns = list(fit.params.index)
    if set(fit_columns) != set(design.columns):
        raise ValueError("Saved regression coefficients do not match encoded features.")
    # Explicit ordering prevents accidental mismatches in a dot product.
    design = design.loc[:, fit_columns]
    raw_prediction = np.asarray(fit.predict(design), dtype=float)
    if raw_prediction.shape != (len(df),) or not np.isfinite(raw_prediction).all():
        raise ValueError("Model returned non-finite or unexpected predictions.")
    result = df.copy()
    result.insert(0, "MODEL_ID", str(bundle["model_id"]))
    result.insert(1, "TRAIN_PCS_DATE", str(bundle["pcs_date"]))
    result["PREDICTED_PRICE_THB"] = raw_prediction
    result["NEGATIVE_PREDICTION"] = raw_prediction < 0
    # Diagnostic mapping used by the model; do not overwrite the car's original values.
    for feature in state.categorical_features:
        value = training.normalize_category_series(df[feature])
        allowed = state.category_levels[feature]
        fallback = "__OTHER__" if "__OTHER__" in allowed else state.reference_categories[feature]
        result[f"{feature}_MODEL_CATEGORY"] = value.where(value.isin(allowed), fallback)
    return result


def parse_feature_pairs(items: list[str]) -> dict[str, object]:
    """Parse NAME=VALUE, preserving text such as sub-model names with spaces."""
    values: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid parameter {item!r}; use feature=value, "
                "e.g. brand=Toyota or --brand Toyota."
            )
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid feature name in {item!r}.")
        normalized = key.casefold()
        if normalized in values:
            raise ValueError(f"Feature {key!r} was supplied more than once.")
        if not value.strip():
            raise ValueError(f"Feature {key!r} has no value.")
        values[normalized] = (key, value)
    return {key: value for key, value in (pair for pair in values.values())}


def parse_dynamic_flags(unknown: list[str]) -> list[str]:
    """Support positional NAME=VALUE and --MODEL_FEATURE VALUE or --MODEL_FEATURE=VALUE."""
    pairs = []
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--") and token != "--":
            name = token[2:]
            if "=" in name:
                pairs.append(name)
                i += 1
            else:
                if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
                    raise ValueError(f"Missing value for {token}.")
                pairs.append(f"{name}={unknown[i + 1]}")
                i += 2
        elif "=" in token:
            pairs.append(token)
            i += 1
        else:
            raise ValueError(f"Unexpected argument {token!r}. Use NAME=VALUE or --NAME VALUE.")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict car prices from the saved Rank-1 OLS model (no SQL or retraining).")
    parser.add_argument("--run-id", default=None, help="Run ID, e.g. 20260919_230223; default: latest saved run.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--car-json", help='A single car as JSON, e.g. \'{"brand":"Toyota",...}\'.')
    group.add_argument("--input-csv", type=Path, help="CSV containing one or more cars (one car per row).")
    group.add_argument("--write-template", action="store_true", help="Create an empty CSV template with required columns.")
    group.add_argument("--show-features", action="store_true", help="Display required input columns and saved category mapping.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--feature", action="append", default=[], metavar="NAME=VALUE",
                        help="Supply a model feature; repeat, or use NAME=VALUE / --NAME VALUE directly.")
    args, unknown = parser.parse_known_args()
    dynamic_pairs = parse_dynamic_flags(unknown)
    direct = bool(args.feature or dynamic_pairs)
    modes = [bool(args.car_json), args.input_csv is not None,
             args.write_template, args.show_features, direct]
    if sum(modes) != 1:
        parser.error("Choose one input mode: direct features, --car-json, --input-csv, "
                     "--write-template, or --show-features.")

    path = locate_bundle(args.run_id)
    bundle, training = load_bundle(path)
    state = make_state(bundle, training)
    print(f"[MODEL] {bundle['model_id']} | PCS_DATE={bundle['pcs_date']}")
    print(f"[FILE]  {path}")
    print(f"[INPUT] {', '.join(bundle['selected_source_features'])}")
    folder = PREDICT_DIR / pd.to_datetime(bundle["pcs_date"]).strftime("%Y%m%d")

    if args.show_features:
        print("\n[NUMERIC] Median used for missing numeric values:")
        for name in state.numeric_features:
            print(f"  {name}: {state.numeric_medians[name]}")
        print("\n[CATEGORICAL] Retained levels and reference categories:")
        for name in state.categorical_features:
            print(f"  {name}: reference={state.reference_categories[name]!r}; "
                  f"levels={state.category_levels[name]!r}")
        return

    if args.write_template:
        folder.mkdir(parents=True, exist_ok=True)
        output = args.output_csv or (folder / f"cars_to_predict_template_{bundle['run_id']}.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=bundle["selected_source_features"]).to_csv(
            output, index=False, encoding="utf-8-sig")
        print(f"[SAVED TEMPLATE] {output}")
        print("Fill all required columns and save as a NEW CSV before --input-csv.")
        return

    if direct:
        data = parse_feature_pairs(args.feature + dynamic_pairs)
        expected = {name.casefold(): name for name in bundle["selected_source_features"]}
        unexpected = [name for name in data if name.casefold() not in expected]
        if unexpected:
            raise ValueError(
                f"Features not used by MODEL_ID={bundle['model_id']}: {unexpected}. "
                f"Required: {list(bundle['selected_source_features'])}. "
                "Run --show-features to see the model's input schema."
            )
        df = pd.DataFrame([data])
    elif args.car_json is not None:
        data = json.loads(args.car_json)
        if not isinstance(data, dict):
            raise ValueError("--car-json must contain one JSON object representing one car.")
        df = pd.DataFrame([data])
    else:
        if not args.input_csv.is_file():
            raise FileNotFoundError(f"Car input CSV not found: {args.input_csv}")
        df = pd.read_csv(args.input_csv, encoding="utf-8-sig", dtype=object)

    result = predict(df, bundle, training)
    folder.mkdir(parents=True, exist_ok=True)
    output = args.output_csv or (folder / f"used_car_predictions_{bundle['run_id']}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.2f")
    print(f"[SAVED] {len(result):,} car(s) -> {output}")
    print("\n" + result[["MODEL_ID", "PREDICTED_PRICE_THB", "NEGATIVE_PREDICTION"]].to_string(index=False))
    if result["NEGATIVE_PREDICTION"].any():
        print("[WARNING] Some OLS predictions are negative. Values are reported as-is, not clamped to 0.")
    print("[NOTE] This is a point prediction, not a guaranteed sale price or prediction interval.")


if __name__ == "__main__":
    main()
