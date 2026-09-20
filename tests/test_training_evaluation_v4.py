"""Focused synthetic-data tests for Training Evaluation V4.

These tests intentionally avoid SQL Server, production artifacts, and joblib
deserialization.  The module is compatible with both ``unittest`` and pytest.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_used_car_ols as training
from predict_used_car_ols import make_state, predict


_validator_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_schema_registry.py"
_validator_spec = importlib.util.spec_from_file_location("schema_registry_validator", _validator_path)
schema_registry_validator = importlib.util.module_from_spec(_validator_spec)
assert _validator_spec.loader is not None
_validator_spec.loader.exec_module(schema_registry_validator)


def synthetic_cars(n: int = 60) -> pd.DataFrame:
    index = pd.Index(np.arange(1000, 1000 + n), name="source_row")
    sequence = np.arange(n)
    return pd.DataFrame(
        {
            "listing_id": [f"L{i:04d}" for i in sequence],
            "source_url": [f"https://example.invalid/{i}" for i in sequence],
            "brand": np.where(sequence % 3 == 0, "A", "B"),
            "model": np.where(sequence % 4 == 0, "M1", "M2"),
            "year": 2014 + (sequence % 10),
            "mileage": 10_000 + sequence * 1_000,
            "price": 150_000 + sequence * 20_000,
        },
        index=index,
    )


def assert_partitions(testcase: unittest.TestCase, whole, left, right) -> None:
    testcase.assertFalse(set(left.index) & set(right.index))
    testcase.assertEqual(set(whole.index), set(left.index) | set(right.index))


class EligibleCohortTests(unittest.TestCase):
    def test_boundary_is_strictly_greater_than_1000(self) -> None:
        prices = [np.nan, -1, 0, 999.99, 1000, 1000.01, 250_000]
        prices.extend([200_000 + i for i in range(30)])
        source = pd.DataFrame({"price": prices, "listing_id": range(len(prices))})

        cleaned = training.clean_target(source)
        eligible, reasons = training.select_eligible_cohort(cleaned)

        self.assertEqual(eligible["price"].iloc[:2].tolist(), [1000.01, 250_000])
        self.assertTrue(eligible["price"].gt(1000).all())
        self.assertEqual(set(reasons.index), set(cleaned.index))
        self.assertTrue(reasons.loc[cleaned["price"].le(1000)].notna().all())
        self.assertTrue(reasons.loc[eligible.index].eq("").all())


class DynamicSchemaTests(unittest.TestCase):
    def source_with_snapshot(self, n: int = 40) -> pd.DataFrame:
        frame = synthetic_cars(n)
        frame["PCS_DATE"] = pd.Timestamp("2026-09-20")
        return frame

    def test_optional_columns_can_be_added_removed_or_renamed(self) -> None:
        base = training.normalize_required_column_names(self.source_with_snapshot())
        base_universe = training.infer_candidate_universe(
            base, ["brand", "model", "year", "mileage"]
        )

        added = base.assign(new_optional_score=np.arange(len(base), dtype=float))
        added_universe = training.infer_candidate_universe(
            added, ["brand", "model", "year", "mileage", "new_optional_score"]
        )
        self.assertNotIn("new_optional_score", base_universe)
        self.assertIn("new_optional_score", added_universe)

        removed = added.drop(columns=["new_optional_score", "year"])
        removed_universe = training.infer_candidate_universe(
            removed, ["brand", "model", "year", "mileage", "new_optional_score"]
        )
        self.assertNotIn("new_optional_score", removed_universe)
        self.assertNotIn("year", removed_universe)

        renamed = base.rename(columns={"year": "registration_year"})
        renamed_universe = training.infer_candidate_universe(
            renamed, ["brand", "model", "registration_year", "mileage"]
        )
        self.assertNotIn("year", renamed_universe)
        self.assertIn("registration_year", renamed_universe)

    def test_optional_numeric_dtype_change_is_learned_from_training_rows(self) -> None:
        numeric = self.source_with_snapshot()
        as_text = numeric.copy()
        as_text["mileage"] = as_text["mileage"].map(lambda value: f"{value:,}")

        numeric_state = training.fit_preprocessor(numeric, ["mileage"])
        text_state = training.fit_preprocessor(as_text, ["mileage"])
        self.assertEqual(numeric_state.numeric_features, ["mileage"])
        self.assertEqual(text_state.numeric_features, ["mileage"])
        pd.testing.assert_frame_equal(
            training.transform_with_preprocessor(numeric, numeric_state),
            training.transform_with_preprocessor(as_text, text_state),
        )

    def test_missing_or_ambiguous_mandatory_columns_fail_clearly(self) -> None:
        source = self.source_with_snapshot()
        for missing in training.MANDATORY_SOURCE_COLUMNS:
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(RuntimeError, "Required .* column"):
                    training.normalize_required_column_names(
                        source.drop(columns=[missing])
                    )

        ambiguous = source.copy()
        ambiguous["PRICE"] = ambiguous["price"]
        with self.assertRaisesRegex(RuntimeError, "duplicate case-insensitive"):
            training.normalize_required_column_names(ambiguous)

    def test_bundle_freezes_training_schema_and_ignores_new_input_columns(self) -> None:
        frame = synthetic_cars(45)
        checkpoint = training.fit_final_candidate(
            frame,
            candidate_id=1,
            initial_source_features=["brand", "year", "mileage"],
            cv_rmse=10.0,
            cv_mae=5.0,
        )
        bundle = training.build_top1_model_bundle(
            checkpoint,
            "20260920_130000",
            "20260920_130000",
            pd.Timestamp("2026-09-20"),
        )
        self.assertEqual(
            set(bundle["training_feature_schema"]),
            set(bundle["selected_source_features"]),
        )

        prediction_input = frame.iloc[-2:][checkpoint.selected_source_features].copy()
        prediction_input["column_added_after_training"] = [1, 2]
        predicted = predict(prediction_input, bundle, training)
        self.assertEqual(len(predicted), 2)
        with self.assertRaisesRegex(ValueError, "Missing model input columns"):
            predict(
                prediction_input.drop(columns=[checkpoint.selected_source_features[0]]),
                bundle,
                training,
            )


class FeatureApprovalGateTests(unittest.TestCase):
    def registry(self, approved_status: str = "APPROVED") -> dict:
        return {
            "registry_version": "test-1",
            "initial_approval_status": approved_status,
            "columns": [
                {"name": "price", "status": "TARGET",
                 "allowed_schema_types": ["numeric"]},
                {"name": "PCS_DATE", "status": "MANDATORY_METADATA",
                 "allowed_schema_types": ["datetime"]},
                {"name": "approved_feature", "status": "APPROVED",
                 "allowed_schema_types": ["numeric"],
                 "predict_availability": "AVAILABLE"},
                {"name": "pending_feature", "status": "PENDING_REVIEW",
                 "allowed_schema_types": ["numeric"],
                 "predict_availability": "NOT_REVIEWED"},
                {"name": "excluded_feature", "status": "EXCLUDED",
                 "allowed_schema_types": ["numeric"],
                 "predict_availability": "NOT_ALLOWED"},
            ],
        }

    def frame(self) -> pd.DataFrame:
        n = 40
        approved = np.arange(n, dtype=float)
        return pd.DataFrame({
            "price": 100_000 + approved * 50_000,
            "PCS_DATE": pd.to_datetime(["2026-09-20"] * n),
            "approved_feature": approved,
            # Deliberately stronger target relationship than the approved input.
            "pending_feature": 100_000 + approved * 50_000,
            "excluded_feature": approved * 2,
        })

    def gate(self, frame=None, registry=None):
        return training.evaluate_feature_approval_gate(
            self.frame() if frame is None else frame,
            self.registry() if registry is None else registry,
            "test-checksum",
        )

    def test_only_approved_feature_enters_candidate_universe(self) -> None:
        frame = self.frame()
        gate = self.gate(frame)
        universe = training.infer_candidate_universe(
            frame, gate.approved_features
        )
        self.assertEqual(universe, ["approved_feature"])
        eligible, _ = training.infer_eligible_features(frame[universe])
        self.assertEqual(eligible, ["approved_feature"])
        self.assertNotIn("pending_feature", universe)
        self.assertNotIn("excluded_feature", universe)

    def test_new_feature_is_pending_and_registry_is_not_mutated(self) -> None:
        registry = self.registry()
        before = copy.deepcopy(registry)
        frame = self.frame().assign(new_target_proxy=lambda x: x["price"])
        gate = self.gate(frame, registry)
        record = next(x for x in gate.records
                      if x["column_name"] == "new_target_proxy")
        self.assertEqual(record["registry_status"], "PENDING_REVIEW")
        self.assertEqual(record["schema_state"], "NEW")
        self.assertTrue(record["target_leakage_risk"])
        self.assertFalse(record["training_authorized"])
        self.assertEqual(registry, before)

    def test_missing_and_renamed_approved_features_are_reported(self) -> None:
        frame = self.frame().rename(
            columns={"approved_feature": "renamed_feature"}
        )
        gate = self.gate(frame)
        self.assertEqual(gate.missing_approved_features, ["approved_feature"])
        renamed = next(x for x in gate.records
                       if x["column_name"] == "renamed_feature")
        self.assertEqual(renamed["registry_status"], "PENDING_REVIEW")
        self.assertFalse(renamed["training_authorized"])

    def test_unsupported_approved_type_change_is_not_authorized(self) -> None:
        frame = self.frame()
        frame["approved_feature"] = "changed meaning"
        gate = self.gate(frame)
        self.assertEqual(
            gate.incompatible_approved_features, ["approved_feature"]
        )
        self.assertNotIn("approved_feature", gate.approved_features)
        record = next(x for x in gate.records
                      if x["column_name"] == "approved_feature")
        self.assertEqual(record["schema_state"], "TYPE_CHANGED")

    def test_initial_registry_requires_owner_approval(self) -> None:
        gate = self.gate(registry=self.registry("PENDING_OWNER_APPROVAL"))
        with self.assertRaisesRegex(RuntimeError, "not owner-approved"):
            training.assert_feature_registry_approved(gate)

    def test_project_registry_matches_initial_owner_approval(self) -> None:
        registry, checksum = training.load_feature_registry()
        approved_expected = {
            "brand", "model", "sub_model", "model_year", "mileage",
            "fuel_type", "transmission", "engine_size", "body_type",
            "color", "number_of_seats",
        }
        pending_expected = {"province", "location", "seller_name", "seller_type"}
        values = {}
        for entry in registry["columns"]:
            name = entry["name"]
            allowed = entry["allowed_schema_types"]
            if name == "PCS_DATE":
                values[name] = pd.to_datetime(["2026-09-20"] * 5)
            elif "numeric" in allowed:
                values[name] = np.arange(5, dtype=float) + 1
            elif "datetime" in allowed:
                values[name] = pd.to_datetime(["2026-09-20"] * 5)
            else:
                values[name] = [f"value-{i}" for i in range(5)]
        frame = pd.DataFrame(values)
        gate = training.evaluate_feature_approval_gate(
            frame, registry, checksum
        )
        self.assertEqual(gate.initial_approval_status, "APPROVED")
        self.assertEqual(set(gate.approved_features), approved_expected)
        training.assert_feature_registry_approved(gate)
        statuses = {x["name"]: x["status"] for x in registry["columns"]}
        self.assertTrue(all(statuses[x] == "PENDING_REVIEW" for x in pending_expected))
        self.assertEqual(
            {x for x, status in statuses.items() if status == "APPROVED"},
            approved_expected,
        )

    def test_sql_date_and_pandas_date_object_are_compatible_metadata_types(self) -> None:
        registry, _ = training.load_feature_registry()
        pcs_entry = next(x for x in registry["columns"] if x["name"] == "PCS_DATE")
        self.assertEqual(schema_registry_validator.sql_schema_type("date"), "datetime")
        pandas_type = training.schema_type_of(pd.Series([date(2026, 9, 20)]))
        self.assertEqual(pandas_type, "text")
        self.assertIn("datetime", pcs_entry["allowed_schema_types"])
        self.assertIn("text", pcs_entry["allowed_schema_types"])

    def test_pending_feature_never_reaches_preprocessing(self) -> None:
        frame = self.frame()
        gate = self.gate(frame)
        universe = training.infer_candidate_universe(
            frame, gate.approved_features
        )
        state = training.fit_preprocessor(frame, universe)
        self.assertNotIn("pending_feature", state.feature_groups)
        self.assertNotIn("pending_feature", state.encoded_feature_names)

    def test_schema_review_report_contains_metadata_without_row_values(self) -> None:
        gate = self.gate()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "schema_review.json"
            training.export_schema_review_report(
                path, gate, pd.Timestamp("2026-09-20"), "20260920_140000"
            )
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        self.assertEqual(payload["registry_version"], "test-1")
        self.assertEqual(payload["registry_checksum"], "test-checksum")
        self.assertIn("schema_fingerprint", payload)
        self.assertEqual(len(payload["columns"]), len(gate.records))
        self.assertNotIn(str(self.frame()["price"].iloc[-1]), text)


class MetadataSchemaValidationTests(unittest.TestCase):
    def test_offline_validator_accepts_approved_pending_and_excluded_contract(self) -> None:
        columns = [
            {"ordinal_position": 1, "column_name": "PCS_DATE",
             "sql_data_type": "date", "nullable": "NO"},
            {"ordinal_position": 2, "column_name": "price",
             "sql_data_type": "bigint", "nullable": "YES"},
            {"ordinal_position": 3, "column_name": "brand",
             "sql_data_type": "nvarchar", "nullable": "YES"},
            {"ordinal_position": 4, "column_name": "province",
             "sql_data_type": "nvarchar", "nullable": "YES"},
            {"ordinal_position": 5, "column_name": "raw_price",
             "sql_data_type": "nvarchar", "nullable": "YES"},
        ]
        report = {
            "report_type": "SQL_SERVER_COLUMN_METADATA_ONLY",
            "schema_name": "dbo", "table_name": "STG_USED_CAR",
            "column_count": len(columns), "columns": columns,
            "schema_fingerprint_sha256":
                schema_registry_validator.canonical_fingerprint(columns),
        }
        registry = {
            "registry_version": "approved-test", "initial_approval_status": "APPROVED",
            "columns": [
                {"name": "PCS_DATE", "status": "MANDATORY_METADATA",
                 "allowed_schema_types": ["datetime", "text"]},
                {"name": "price", "status": "TARGET",
                 "allowed_schema_types": ["numeric", "text"]},
                {"name": "brand", "status": "APPROVED",
                 "allowed_schema_types": ["text"]},
                {"name": "province", "status": "PENDING_REVIEW",
                 "allowed_schema_types": ["text"]},
                {"name": "raw_price", "status": "EXCLUDED",
                 "allowed_schema_types": ["numeric", "text"]},
            ],
        }
        result = schema_registry_validator.validate(report, registry, "checksum")
        self.assertTrue(result["valid"])
        self.assertEqual(result["approved_compatible_features"], ["brand"])
        authorized = {x["column_name"]: x["predictor_authorized"]
                      for x in result["comparisons"]}
        self.assertTrue(authorized["brand"])
        self.assertFalse(authorized["province"])
        self.assertFalse(authorized["price"])
        self.assertFalse(authorized["PCS_DATE"])
        self.assertFalse(authorized["raw_price"])

    def test_offline_validator_blocks_missing_or_incompatible_approved_feature(self) -> None:
        columns = [
            {"ordinal_position": 1, "column_name": "PCS_DATE",
             "sql_data_type": "date", "nullable": "NO"},
            {"ordinal_position": 2, "column_name": "price",
             "sql_data_type": "bigint", "nullable": "YES"},
            {"ordinal_position": 3, "column_name": "brand",
             "sql_data_type": "bigint", "nullable": "YES"},
        ]
        report = {
            "report_type": "SQL_SERVER_COLUMN_METADATA_ONLY",
            "schema_name": "dbo", "table_name": "STG_USED_CAR",
            "column_count": 3, "columns": columns,
            "schema_fingerprint_sha256":
                schema_registry_validator.canonical_fingerprint(columns),
        }
        registry = {
            "registry_version": "approved-test", "initial_approval_status": "APPROVED",
            "columns": [
                {"name": "PCS_DATE", "status": "MANDATORY_METADATA",
                 "allowed_schema_types": ["datetime"]},
                {"name": "price", "status": "TARGET",
                 "allowed_schema_types": ["numeric"]},
                {"name": "brand", "status": "APPROVED",
                 "allowed_schema_types": ["text"]},
                {"name": "model", "status": "APPROVED",
                 "allowed_schema_types": ["text"]},
            ],
        }
        result = schema_registry_validator.validate(report, registry, "checksum")
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_columns"], ["model"])
        self.assertEqual(result["approved_compatible_features"], [])


class SplitAndFoldTests(unittest.TestCase):
    def test_grouping_before_cohort_filter_keeps_transitive_identity_links(self) -> None:
        frame = synthetic_cars(35)
        # The ineligible middle row is the only bridge between two eligible rows.
        frame.loc[1000, "listing_id"] = "shared-listing"
        frame.loc[1001, "listing_id"] = "shared-listing"
        frame.loc[1001, "source_url"] = "https://example.invalid/bridge"
        frame.loc[1002, "source_url"] = "https://example.invalid/bridge"
        frame.loc[1001, "price"] = 1000

        cleaned = training.clean_target(frame)
        full_groups = training.build_duplicate_groups(cleaned)
        eligible, _ = training.select_eligible_cohort(cleaned)
        eligible_groups = full_groups.loc[eligible.index]

        self.assertNotIn(1001, eligible.index)
        self.assertEqual(eligible_groups.loc[1000], eligible_groups.loc[1002])

    def test_duplicate_identity_is_transitive_and_stays_in_one_partition(self) -> None:
        frame = synthetic_cars()
        # Row 1000 links to 1001 by listing_id, while 1001 links to 1002 by URL.
        frame.loc[1001, "listing_id"] = frame.loc[1000, "listing_id"]
        frame.loc[1002, "source_url"] = frame.loc[1001, "source_url"]
        groups = training.build_duplicate_groups(frame)

        self.assertEqual(groups.loc[1000], groups.loc[1001])
        self.assertEqual(groups.loc[1001], groups.loc[1002])

        dev, holdout, metadata = training.split_development_holdout(frame, groups)
        assert_partitions(self, frame, dev, holdout)
        locations = [1000 in dev.index, 1001 in dev.index, 1002 in dev.index]
        self.assertTrue(all(locations) or not any(locations))
        self.assertEqual(len(dev) + len(holdout), len(frame))
        self.assertIsInstance(metadata, dict)

    def test_split_is_reproducible_and_close_to_80_20(self) -> None:
        frame = synthetic_cars(100)
        groups = training.build_duplicate_groups(frame)
        first_dev, first_holdout, _ = training.split_development_holdout(frame, groups)
        second_dev, second_holdout, _ = training.split_development_holdout(frame, groups)

        self.assertEqual(first_dev.index.tolist(), second_dev.index.tolist())
        self.assertEqual(first_holdout.index.tolist(), second_holdout.index.tolist())
        self.assertLessEqual(abs(len(first_holdout) - 20), 2)

    def test_grouped_holdout_selects_closest_split_within_tolerance(self) -> None:
        frame = synthetic_cars(100)
        groups = pd.Series(
            [f"G{i // 2:03d}" for i in range(100)], index=frame.index
        )

        class FakeGroupedSplitter:
            def __init__(self, **_kwargs):
                pass

            def split(self, _df, _bands, _groups):
                positions = np.arange(100)
                for size in (30, 22, 18, 24, 10):
                    valid = positions[:size]
                    yield positions[size:], valid

        with patch.object(training, "StratifiedGroupKFold", FakeGroupedSplitter):
            dev, holdout, _ = training.split_development_holdout(frame, groups)
        self.assertEqual(len(holdout), 22)
        self.assertAlmostEqual(len(holdout) / len(frame), 0.22)
        assert_partitions(self, frame, dev, holdout)

    def test_grouped_holdout_rejects_when_no_split_is_within_tolerance(self) -> None:
        frame = synthetic_cars(100)
        groups = pd.Series(
            [f"G{i // 2:03d}" for i in range(100)], index=frame.index
        )

        class FakeGroupedSplitter:
            def __init__(self, **_kwargs):
                pass

            def split(self, _df, _bands, _groups):
                positions = np.arange(100)
                for size in (5, 10, 30, 35, 40):
                    valid = positions[:size]
                    yield positions[size:], valid

        with patch.object(training, "StratifiedGroupKFold", FakeGroupedSplitter):
            with self.assertRaises(ValueError):
                training.split_development_holdout(frame, groups)

    def test_common_folds_cover_development_once_without_group_leakage(self) -> None:
        frame = synthetic_cars(75)
        frame.loc[1001, "listing_id"] = frame.loc[1000, "listing_id"]
        groups = training.build_duplicate_groups(frame)
        dev, _, _ = training.split_development_holdout(frame, groups)
        dev_groups = groups.loc[dev.index]

        folds_a, metadata_a = training.build_common_cv_folds(dev, dev_groups)
        folds_b, metadata_b = training.build_common_cv_folds(dev, dev_groups)

        seen_valid = []
        for (train_pos, valid_pos), (train_pos_b, valid_pos_b) in zip(folds_a, folds_b):
            np.testing.assert_array_equal(train_pos, train_pos_b)
            np.testing.assert_array_equal(valid_pos, valid_pos_b)
            self.assertFalse(set(train_pos) & set(valid_pos))
            train_groups = set(dev_groups.iloc[train_pos])
            valid_groups = set(dev_groups.iloc[valid_pos])
            self.assertFalse(train_groups & valid_groups)
            seen_valid.extend(valid_pos.tolist())

        self.assertEqual(sorted(seen_valid), list(range(len(dev))))
        self.assertEqual(
            metadata_a["assignment_checksum"], metadata_b["assignment_checksum"]
        )


class RankingAndExportTests(unittest.TestCase):
    def test_unseen_category_maps_to_other_when_all_train_categories_retained(self) -> None:
        train = pd.DataFrame({
            "brand": ["A"] * 25 + ["B"] * 25,
            "price": np.arange(50, dtype=float) + 100_000,
        })
        state = training.fit_preprocessor(train, ["brand"])
        transformed = training.transform_with_preprocessor(
            pd.DataFrame({"brand": ["NEW"]}), state
        )

        self.assertIn("__OTHER__", state.category_levels["brand"])
        other_column = training.safe_feature_name("brand", "__OTHER__")
        self.assertEqual(float(transformed.loc[0, other_column]), 1.0)

    def test_numeric_like_category_with_commas_matches_numeric_transform(self) -> None:
        values = ["1,000", "2,500", "3,750", "4,000"] * 10
        train = pd.DataFrame({
            "mileage_text": pd.Series(values, dtype="category"),
            "price": np.arange(40, dtype=float) + 100_000,
        })
        state = training.fit_preprocessor(train, ["mileage_text"])
        categorical_input = pd.DataFrame({
            "mileage_text": pd.Series(["1,000", "3,750"], dtype="category")
        })
        numeric_equivalent = pd.DataFrame({"mileage_text": [1000.0, 3750.0]})

        actual = training.transform_with_preprocessor(categorical_input, state)
        expected = training.transform_with_preprocessor(numeric_equivalent, state)

        self.assertIn("mileage_text", state.numeric_like_features)
        pd.testing.assert_frame_equal(actual, expected)

    def test_cv_metrics_are_pooled_oof_and_preprocessing_is_fold_local(self) -> None:
        frame = synthetic_cars(50)
        # This category exists only in row position 0, which is validation in fold 1.
        frame.iloc[0, frame.columns.get_loc("model")] = "VALIDATION_ONLY"
        positions = np.arange(len(frame))
        folds = []
        for fold_id in range(5):
            valid = positions[fold_id * 10:(fold_id + 1) * 10]
            train = np.setdiff1d(positions, valid)
            folds.append((train, valid))

        evaluation = training.evaluate_candidate_cv(
            frame,
            ["model", "year", "mileage"],
            folds=folds,
        )
        oof = evaluation.oof_predictions
        errors = oof["ACTUAL_PRICE"] - oof["PREDICTED_PRICE"]

        self.assertAlmostEqual(
            evaluation.rmse, float(np.sqrt(np.mean(np.square(errors)))), places=8
        )
        self.assertAlmostEqual(
            evaluation.mae, float(np.mean(np.abs(errors))), places=8
        )
        self.assertEqual(evaluation.count, len(frame))
        rare = oof.loc[oof["ROW_POSITION"].eq(0), "MODEL_CATEGORY"].item()
        self.assertEqual(rare, "__OTHER__")

    def test_result_and_coefficient_contract_column_counts(self) -> None:
        self.assertEqual(training.RESULT_COLUMNS, [
            "MODEL_ID", "MODEL_NAME", "TARGET_NAME", "TRAIN_PCS_DATE",
            "TRAIN_DATE", "N_OBSERVATION", "R_SQUARED", "ADJ_R_SQUARED",
            "MAE", "RMSE", "F_STATISTIC", "F_P_VALUE",
            "P_VALUE_THRESHOLD", "CONFIDENCE_LEVEL", "ACTIVE_FLAG",
            "APPROVED_BY", "APPROVED_DATE", "PCS_DATE",
        ])
        self.assertEqual(training.COEFFICIENT_COLUMNS, [
            "MODEL_ID", "FEATURE_SEQ", "SOURCE_COLUMN", "ORIGINAL_VALUE",
            "FEATURE_NAME", "FEATURE_TYPE", "COEFFICIENT", "P_VALUE",
            "SOURCE_FEATURE_P_VALUE", "CI_LOWER", "CI_UPPER",
            "IS_REFERENCE", "PCS_DATE",
        ])

    def test_ranking_uses_rmse_then_mae_then_candidate_id(self) -> None:
        class Candidate:
            def __init__(self, candidate_id, rmse, mae):
                self.candidate_id = candidate_id
                self.cv_rmse = rmse
                self.cv_mae = mae
                self.selected_source_features = [f"unique_{candidate_id}"]
                self.preprocessor = type("State", (), {"category_levels": {}})()

        candidates = [Candidate(9, 10, 4), Candidate(3, 10, 4), Candidate(8, 9, 99)]
        ranked = training.rank_candidates(candidates)
        self.assertEqual([candidate.candidate_id for candidate in ranked], [8, 3, 9])

    def test_result_n_observation_comes_from_each_fitted_model(self) -> None:
        frame = synthetic_cars(40)
        checkpoint = training.fit_final_candidate(
            frame,
            candidate_id=1,
            initial_source_features=["year", "mileage"],
            cv_rmse=1.0,
            cv_mae=1.0,
        )
        result, _ = training.build_result_dataframe(
            [checkpoint],
            datetime(2026, 9, 20, 12, 0, 0),
            pd.Timestamp("2026-09-20"),
        )
        self.assertEqual(result.loc[0, "N_OBSERVATION"], int(checkpoint.ols_result.nobs))


class FrozenRefitAndBundleTests(unittest.TestCase):
    def test_holdout_evaluation_does_not_mutate_development_checkpoint(self) -> None:
        frame = synthetic_cars(50)
        dev, holdout = frame.iloc[:40], frame.iloc[40:].copy()
        checkpoint = training.fit_final_candidate(
            dev,
            candidate_id=7,
            initial_source_features=["brand", "year", "mileage"],
            cv_rmse=123.0,
            cv_mae=45.0,
        )
        coefficients_before = checkpoint.ols_result.params.copy()
        state_before = copy.deepcopy(checkpoint.preprocessor)

        metrics, predictions = training.evaluate_frozen_candidate(
            checkpoint, holdout
        )

        pd.testing.assert_series_equal(
            checkpoint.ols_result.params, coefficients_before
        )
        self.assertEqual(checkpoint.preprocessor, state_before)
        self.assertEqual(len(predictions), len(holdout))
        self.assertEqual(metrics["count"], len(holdout))
        self.assertEqual(checkpoint.cv_rmse, 123.0)
        self.assertEqual(checkpoint.cv_mae, 45.0)

    def test_full_refit_preserves_checkpoint_preprocessor_and_features(self) -> None:
        frame = synthetic_cars(50)
        dev = frame.iloc[:40]
        checkpoint = training.fit_final_candidate(
            dev,
            candidate_id=1,
            initial_source_features=["brand", "year", "mileage"],
            cv_rmse=10.0,
            cv_mae=5.0,
        )
        before_state = copy.deepcopy(checkpoint.preprocessor)
        before_sources = list(checkpoint.selected_source_features)
        before_encoded = list(checkpoint.selected_encoded_features)

        refit = training.refit_candidate_coefficients(checkpoint, frame)

        self.assertEqual(refit.preprocessor, before_state)
        self.assertEqual(refit.selected_source_features, before_sources)
        self.assertEqual(refit.selected_encoded_features, before_encoded)
        self.assertEqual(int(refit.ols_result.nobs), len(frame))

    def test_v4_bundle_remains_compatible_with_prediction_functions(self) -> None:
        frame = synthetic_cars(45)
        checkpoint = training.fit_final_candidate(
            frame,
            candidate_id=1,
            initial_source_features=["brand", "year", "mileage"],
            cv_rmse=10.0,
            cv_mae=5.0,
        )
        schema_gate = training.SchemaGateResult(
            registry_version="approved-test-1",
            registry_checksum="checksum",
            schema_fingerprint="fingerprint",
            initial_approval_status="APPROVED",
            approved_features=["brand", "year", "mileage"],
            missing_approved_features=[],
            incompatible_approved_features=[],
            records=[],
        )
        bundle = training.build_top1_model_bundle(
            checkpoint,
            "20260920_120000",
            "20260920_120000",
            pd.Timestamp("2026-09-20"),
            schema_gate=schema_gate,
        )

        required_legacy_keys = {
            "run_id", "model_id", "model_rank", "model_name", "target_column",
            "pcs_date", "selected_source_features", "selected_encoded_features",
            "preprocessor", "ols_result", "encoding_profile", "encoding_config",
            "cv_rmse", "cv_mae", "r_squared", "adj_r_squared",
        }
        required_preprocessor_keys = {
            "numeric_features", "categorical_features", "numeric_medians",
            "category_levels", "reference_categories", "encoded_feature_names",
            "feature_groups",
        }
        self.assertTrue(required_legacy_keys.issubset(bundle))
        self.assertTrue(required_preprocessor_keys.issubset(bundle["preprocessor"]))
        self.assertEqual(
            bundle["feature_approval"]["registry_version"], "approved-test-1"
        )
        self.assertEqual(
            bundle["feature_approval"]["approved_features_for_run"],
            ["brand", "year", "mileage"],
        )

        restored = make_state(bundle, training)
        expected_medians = {
            feature: checkpoint.preprocessor.numeric_medians[feature]
            for feature in checkpoint.selected_source_features
            if feature in checkpoint.preprocessor.numeric_medians
        }
        self.assertEqual(restored.numeric_medians, expected_medians)
        predicted = predict(
            frame.iloc[-2:][checkpoint.selected_source_features], bundle, training
        )
        self.assertEqual(len(predicted), 2)
        self.assertTrue(np.isfinite(predicted["PREDICTED_PRICE_THB"]).all())
        self.assertNotIn("price", bundle["selected_source_features"])


class RunIdAndCollisionTests(unittest.TestCase):
    def test_explicit_valid_run_id_round_trips(self) -> None:
        with patch.dict(os.environ, {"USED_CAR_RUN_ID": "20260920_123456"}):
            run_datetime, run_id = training.resolve_run_datetime()
        self.assertEqual(run_id, "20260920_123456")
        self.assertEqual(run_datetime, datetime(2026, 9, 20, 12, 34, 56))

    def test_missing_run_id_uses_timestamp_format(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USED_CAR_RUN_ID", None)
            run_datetime, run_id = training.resolve_run_datetime()
        self.assertEqual(run_datetime.strftime("%Y%m%d_%H%M%S"), run_id)

    def test_invalid_run_ids_are_rejected(self) -> None:
        for value in ("2026-09-20", "20260920_12345", "20260230_120000", "x"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"USED_CAR_RUN_ID": value}):
                    with self.assertRaises(ValueError):
                        training.resolve_run_datetime()

    def test_collision_check_refuses_any_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "OLS_REGRESSION_RESULT_20260920_123456.csv"
            existing.touch()
            paths = {
                "result": existing,
                "coefficient": root / "OLS_REGRESSION_COEFFICIENT_20260920_123456.csv",
                "joblib": root / "used_car_models_20260920_123456.joblib",
            }
            with self.assertRaises(FileExistsError) as caught:
                training.assert_no_artifact_collisions(paths)
            self.assertIn(existing.name, str(caught.exception))

    def test_collision_check_accepts_all_new_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {"result": root / "new.csv", "joblib": root / "new.joblib"}
            self.assertIsNone(training.assert_no_artifact_collisions(paths))


if __name__ == "__main__":
    unittest.main()
