import csv
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime
from enum import Enum

from src.experiment_tracker import ExperimentTracker


class TestExperimentTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = ":memory:"
        self.tracker = ExperimentTracker(self.db_path)

    def tearDown(self) -> None:
        if hasattr(self, "tracker"):
            self.tracker.conn.close()

    def test_table_creation(self) -> None:
        required_tables = {
            "experiments",
            "runs",
            "models",
            "predictions",
            "metrics",
            "tags",
            "artifacts",
        }
        cursor = self.tracker.conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('experiments', 'runs', 'models', 'predictions', 'metrics', 'tags', 'artifacts')
        """
        )
        created_tables = {row[0] for row in cursor.fetchall()}
        self.assertEqual(
            created_tables, required_tables, "All required tables should be created"
        )

    def test_create_experiment(self) -> None:
        exp_name = "test_experiment_1"
        exp_description = "Test experiment description"
        exp_id = self.tracker.create_experiment(exp_name, exp_description)
        self.assertIsInstance(exp_id, int, "Should return integer ID")

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT experiment_name, experiment_description, created_time FROM experiments WHERE experiment_id=?",
            (exp_id,),
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Experiment record should exist")
        self.assertEqual(result[0], exp_name, "Stored name should match input")
        self.assertEqual(
            result[1], exp_description, "Stored description should match input"
        )
        self.assertIsNotNone(result[2], "Should have creation timestamp")

    def test_start_run_valid(self) -> None:
        exp_name = "experiment_for_run"
        exp_id = self.tracker.create_experiment(exp_name)
        run_id = self.tracker.start_run(exp_id)
        self.assertIsInstance(run_id, int, "Should return integer run ID")

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT experiment_id, run_status, run_start_time FROM runs WHERE run_id=?",
            (run_id,),
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Run record should exist")
        self.assertEqual(result[0], exp_id, "Experiment ID should match")
        self.assertEqual(result[1], "RUNNING", "Status should be RUNNING")
        self.assertIsNotNone(result[2], "Start time should be set")

    def test_start_run_invalid_experiment(self) -> None:
        invalid_exp_id = 9999
        with self.assertRaises(sqlite3.IntegrityError):
            self.tracker.start_run(invalid_exp_id)

    def test_log_model(self) -> None:
        exp_id = self.tracker.create_experiment("model_logging_test")
        run_id = self.tracker.start_run(exp_id)

        test_params = {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.001}
        self.tracker.log_model(run_id, model_name="test_model", parameters=test_params)

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT model_name, parameters FROM models WHERE run_id=?", (run_id,)
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Model record should exist")
        self.assertEqual(result[0], "test_model", "Model name should match")
        self.assertEqual(
            json.loads(result[1]),
            test_params,
            "Parameters should match after deserialization",
        )

    def test_log_predictions(self) -> None:
        exp_id = self.tracker.create_experiment("prediction_test")
        run_id = self.tracker.start_run(exp_id)

        test_preds = [0.1, 0.2, 0.3]
        test_actuals = [0.15, 0.25, 0.35]
        self.tracker.log_predictions(run_id, test_preds, test_actuals)

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT prediction, actual FROM predictions WHERE run_id=? ORDER BY idx",
            (run_id,),
        )
        results = cursor.fetchall()

        self.assertEqual(len(results), 3, "Should have 3 individual prediction records")

        for i, (pred, actual) in enumerate(results):
            self.assertEqual(pred, test_preds[i], f"Prediction {i} should match")
            self.assertEqual(actual, test_actuals[i], f"Actual {i} should match")

    def test_metrics_table_exists(self):
        """Ensure that the metrics table exists in the database."""
        cursor = self.tracker.conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='metrics'
        """
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Metrics table should exist in the database")

    def test_get_predictions(self):
        exp_id = self.tracker.create_experiment("get_predictions_test")
        run_id = self.tracker.start_run(exp_id)

        test_preds = [0.1, 0.2, 0.3]
        test_actuals = [0.15, 0.25, 0.35]
        self.tracker.log_predictions(run_id, test_preds, test_actuals)

        result = self.tracker.get_predictions(run_id)

        self.assertIn("predictions", result)
        self.assertIn("actuals", result)
        self.assertIn("index", result)

        self.assertEqual(result["predictions"], test_preds)
        self.assertEqual(result["actuals"], test_actuals)
        self.assertEqual(len(result["index"]), 3)

    def test_log_predictions_validation(self) -> None:
        exp_id = self.tracker.create_experiment("validation_test")
        run_id = self.tracker.start_run(exp_id)

        with self.assertRaises(ValueError):
            self.tracker.log_predictions(run_id, [1, 2], [1])  # mismatched lengths

        with self.assertRaises(TypeError):
            self.tracker.log_predictions(run_id, "not a list", [1])  # invalid type

    def test_end_run_success(self) -> None:
        exp_id = self.tracker.create_experiment("success_test")
        run_id = self.tracker.start_run(exp_id)
        self.tracker.end_run(run_id)

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT run_status, run_end_time, error FROM runs WHERE run_id=?", (run_id,)
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Run record should exist")
        self.assertEqual(result[0], "COMPLETED", "Status should be COMPLETED")
        self.assertIsNotNone(result[1], "End time should be set")
        self.assertIsNone(result[2], "Error should be null for successful run")

    def test_end_run_failure(self) -> None:
        exp_id = self.tracker.create_experiment("failure_test")
        run_id = self.tracker.start_run(exp_id)
        error_msg = "Division by zero error"
        self.tracker.end_run(run_id, success=False, error=error_msg)

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT run_status, run_end_time, error FROM runs WHERE run_id=?", (run_id,)
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Run record should exist")
        self.assertEqual(result[0], "FAILED", "Status should be FAILED")
        self.assertIsNotNone(result[1], "End time should be set")
        self.assertEqual(result[2], error_msg, "Error message should match input")

    def test_full_workflow(self):
        exp_id = self.tracker.create_experiment("Integration Test")
        run_id = self.tracker.start_run(exp_id)
        test_params = {"kernel": "linear", "C": 1.0}
        test_preds = [0.1, 0.2, 0.3]
        test_actuals = [0.15, 0.25, 0.35]

        self.tracker.log_model(run_id, "SVC", test_params)
        self.tracker.log_predictions(run_id, test_preds, test_actuals)
        self.tracker.end_run(run_id)

        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT experiment_name FROM experiments WHERE experiment_id=?", (exp_id,)
        )
        self.assertEqual(cursor.fetchone()[0], "Integration Test")

        cursor.execute("SELECT run_status FROM runs WHERE run_id=?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], "COMPLETED")

        cursor.execute("SELECT parameters FROM models WHERE run_id=?", (run_id,))
        self.assertEqual(json.loads(cursor.fetchone()[0]), test_params)

        cursor.execute(
            "SELECT prediction, actual FROM predictions WHERE run_id=? ORDER BY idx",
            (run_id,),
        )
        results = cursor.fetchall()

        stored_preds = [row[0] for row in results]
        stored_actuals = [row[1] for row in results]

        self.assertEqual(stored_preds, test_preds)
        self.assertEqual(stored_actuals, test_actuals)

    def test_calculate_metrics(self):
        exp_id = self.tracker.create_experiment("metrics_test")
        run_id = self.tracker.start_run(exp_id)
        preds = [1.0, 2.0, 3.0]
        actuals = [1.1, 1.9, 3.05]
        import math

        n = len(preds)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)
        mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / n
        self.tracker.log_predictions(run_id, preds, actuals)
        cursor = self.tracker.conn.cursor()
        cursor.execute(
            "SELECT metric, metric_value FROM metrics WHERE run_id=?", (run_id,)
        )
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 2, "Should insert two metrics entries")
        metrics = {name: value for name, value in rows}
        self.assertAlmostEqual(metrics["rmse"], rmse, places=5)
        self.assertAlmostEqual(metrics["mae"], mae, places=5)

    def test_custom_metrics(self):
        exp_id = self.tracker.create_experiment("custom_metrics_test")
        run_id = self.tracker.start_run(exp_id)
        preds = [1.0, 2.0, 3.0, 4.0]
        actuals = [1.1, 1.9, 3.05, 3.9]

        def max_error(preds, actuals):
            return max(abs(p - a) for p, a in zip(preds, actuals))

        def bias(preds, actuals):
            return sum(p - a for p, a in zip(preds, actuals)) / len(preds)

        custom_metrics = {"max_error": max_error, "bias": bias}

        expected_max_error = max(abs(p - a) for p, a in zip(preds, actuals))
        expected_bias = sum(p - a for p, a in zip(preds, actuals)) / len(preds)

        self.tracker.log_predictions(
            run_id, preds, actuals, custom_metrics=custom_metrics
        )

        metrics = self.tracker.get_metrics(run_id)

        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)

        self.assertIn("max_error", metrics)
        self.assertIn("bias", metrics)
        self.assertAlmostEqual(metrics["max_error"], expected_max_error, places=5)
        self.assertAlmostEqual(metrics["bias"], expected_bias, places=5)

    def test_log_metric_directly(self):
        exp_id = self.tracker.create_experiment("direct_metric_test")
        run_id = self.tracker.start_run(exp_id)

        self.tracker.log_metric(run_id, "accuracy", 0.95)
        self.tracker.log_metric(run_id, "f1_score", 0.92)

        self.tracker.log_metric(run_id, "accuracy", 0.96)

        metrics = self.tracker.get_metrics(run_id)

        self.assertEqual(metrics["accuracy"], 0.96)
        self.assertEqual(metrics["f1_score"], 0.92)

    def test_export_experiment(self):
        export_dir = tempfile.mkdtemp()
        try:
            exp_id = self.tracker.create_experiment(
                "Export Test", "Testing export functionality"
            )

            run_id1 = self.tracker.start_run(exp_id)
            self.tracker.log_model(
                run_id1, "TestModel", {"param1": 1, "param2": "test"}
            )
            preds1 = [0.1, 0.2, 0.3]
            actuals1 = [0.15, 0.25, 0.35]
            self.tracker.log_predictions(run_id1, preds1, actuals1)
            self.tracker.log_metric(run_id1, "custom_metric", 0.95)
            self.tracker.end_run(run_id1)

            run_id2 = self.tracker.start_run(exp_id)
            self.tracker.log_model(
                run_id2, "TestModel2", {"param1": 2, "param2": "test2"}
            )
            self.tracker.end_run(run_id2, success=False, error="Test error")

            export_path = self.tracker.export_experiment(exp_id, export_dir)

            self.assertTrue(os.path.exists(export_path))
            self.assertTrue(
                os.path.exists(os.path.join(export_path, "experiments.csv"))
            )
            self.assertTrue(os.path.exists(os.path.join(export_path, "runs.csv")))
            self.assertTrue(os.path.exists(os.path.join(export_path, "models.csv")))
            self.assertTrue(
                os.path.exists(os.path.join(export_path, "predictions.csv"))
            )
            self.assertTrue(os.path.exists(os.path.join(export_path, "metrics.csv")))

            with open(os.path.join(export_path, "experiments.csv"), "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                self.assertEqual(
                    headers,
                    [
                        "experiment_id",
                        "experiment_name",
                        "experiment_description",
                        "created_time",
                    ],
                )

            with open(os.path.join(export_path, "runs.csv"), "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                self.assertEqual(
                    headers,
                    [
                        "run_id",
                        "experiment_id",
                        "run_status",
                        "run_start_time",
                        "run_end_time",
                        "error",
                    ],
                )
                rows = list(reader)
                self.assertEqual(len(rows), 2)

            with open(os.path.join(export_path, "models.csv"), "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                self.assertEqual(
                    headers, ["model_id", "run_id", "model_name", "parameters"]
                )
                rows = list(reader)
                self.assertEqual(len(rows), 2)

        finally:
            shutil.rmtree(export_dir)

    def test_import_experiment(self):
        export_dir = tempfile.mkdtemp()
        try:
            source_tracker = ExperimentTracker(":memory:")
            exp_id = source_tracker.create_experiment(
                "Import Test", "Testing import functionality"
            )

            run_id = source_tracker.start_run(exp_id)
            source_tracker.log_model(run_id, "ImportModel", {"learning_rate": 0.01})
            preds = [0.1, 0.2, 0.3]
            actuals = [0.15, 0.25, 0.35]
            source_tracker.log_predictions(run_id, preds, actuals)
            source_tracker.log_metric(run_id, "custom_score", 0.88)
            source_tracker.end_run(run_id)

            export_path = source_tracker.export_experiment(exp_id, export_dir)

            self.assertTrue(
                os.path.exists(os.path.join(export_path, "experiments.csv"))
            )
            self.assertTrue(os.path.exists(os.path.join(export_path, "runs.csv")))
            self.assertTrue(os.path.exists(os.path.join(export_path, "models.csv")))
            self.assertTrue(
                os.path.exists(os.path.join(export_path, "predictions.csv"))
            )
            self.assertTrue(os.path.exists(os.path.join(export_path, "metrics.csv")))

            target_tracker = ExperimentTracker(":memory:")
            new_exp_id = target_tracker.import_experiment(export_path)

            imported_exp = target_tracker.get_experiment(new_exp_id)
            self.assertEqual(imported_exp["experiment_name"], "Import Test")
            self.assertEqual(
                imported_exp["experiment_description"], "Testing import functionality"
            )

            runs = target_tracker.get_run_history(new_exp_id)
            self.assertEqual(len(runs), 1)

            new_run_id = runs[0]["run_id"]
            model = target_tracker.get_model(new_run_id)
            self.assertIsNotNone(model)
            self.assertEqual(model["model_name"], "ImportModel")
            self.assertEqual(model["parameters"]["learning_rate"], 0.01)

            predictions = target_tracker.get_predictions(new_run_id)
            self.assertEqual(predictions["predictions"], preds)
            self.assertEqual(predictions["actuals"], actuals)

            metrics = target_tracker.get_metrics(new_run_id)
            self.assertTrue("rmse" in metrics)
            self.assertTrue("mae" in metrics)
            self.assertTrue("custom_score" in metrics)
            self.assertEqual(metrics["custom_score"], 0.88)

        finally:
            source_tracker.conn.close()
            target_tracker.conn.close()
            shutil.rmtree(export_dir)

    def test_tags(self):
        exp_id = self.tracker.create_experiment("Tag Test", "Testing tag functionality")
        run_id = self.tracker.start_run(exp_id)

        self.tracker.log_tag("experiment", exp_id, "version", "v1.0")
        self.tracker.log_tag("experiment", exp_id, "owner", "test_user")
        self.tracker.log_tag("experiment", exp_id, "priority", "high")

        self.tracker.log_tag("run", run_id, "model_type", "regression")
        self.tracker.log_tag("run", run_id, "dataset", "test_data")

        exp_tags = self.tracker.get_tags("experiment", exp_id)
        self.assertEqual(len(exp_tags), 3)
        self.assertEqual(exp_tags["version"], "v1.0")
        self.assertEqual(exp_tags["owner"], "test_user")
        self.assertEqual(exp_tags["priority"], "high")

        run_tags = self.tracker.get_tags("run", run_id)
        self.assertEqual(len(run_tags), 2)
        self.assertEqual(run_tags["model_type"], "regression")
        self.assertEqual(run_tags["dataset"], "test_data")

        tagged_experiments = self.tracker.get_tagged_entities(
            "experiment", "priority", "high"
        )
        self.assertIn(exp_id, tagged_experiments)

        tagged_runs = self.tracker.get_tagged_entities(
            "run", "model_type", "regression"
        )
        self.assertIn(run_id, tagged_runs)

        deleted = self.tracker.delete_tag("experiment", exp_id, "priority")
        self.assertEqual(deleted, 1)

        exp_tags_after_delete = self.tracker.get_tags("experiment", exp_id)
        self.assertEqual(len(exp_tags_after_delete), 2)
        self.assertNotIn("priority", exp_tags_after_delete)

    def test_export_import_tags(self):
        export_dir = tempfile.mkdtemp()
        try:
            source_tracker = ExperimentTracker(":memory:")
            exp_id = source_tracker.create_experiment("Tag Export Test")
            run_id = source_tracker.start_run(exp_id)

            source_tracker.log_tag("experiment", exp_id, "category", "test")
            source_tracker.log_tag("run", run_id, "algorithm", "random_forest")

            source_tracker.end_run(run_id)
            export_path = source_tracker.export_experiment(exp_id, export_dir)

            self.assertTrue(os.path.exists(os.path.join(export_path, "tags.csv")))

            target_tracker = ExperimentTracker(":memory:")
            new_exp_id = target_tracker.import_experiment(export_path)

            runs = target_tracker.get_run_history(new_exp_id)
            new_run_id = runs[0]["run_id"]

            exp_tags = target_tracker.get_tags("experiment", new_exp_id)
            self.assertEqual(exp_tags["category"], "test")

            run_tags = target_tracker.get_tags("run", new_run_id)
            self.assertEqual(run_tags["algorithm"], "random_forest")

        finally:
            if "source_tracker" in locals():
                source_tracker.conn.close()
            if "target_tracker" in locals():
                target_tracker.conn.close()
            shutil.rmtree(export_dir)

    def test_get_experiment(self):
        exp_name = "get_experiment_test"
        exp_description = "Test experiment retrieval"
        exp_id = self.tracker.create_experiment(exp_name, exp_description)
        experiment = self.tracker.get_experiment(exp_id)
        self.assertIsNotNone(experiment, "Experiment should be retrieved successfully")
        self.assertEqual(experiment.get("experiment_name"), exp_name)
        self.assertEqual(experiment.get("experiment_description"), exp_description)

    def test_get_run_history(self):
        exp_name = "run_history_test"
        exp_id = self.tracker.create_experiment(exp_name)
        run_ids = [self.tracker.start_run(exp_id) for _ in range(3)]
        self.tracker.end_run(run_ids[1])
        history = self.tracker.get_run_history(exp_id)
        self.assertTrue(
            len(history) >= 3, "Run history should contain at least three runs"
        )
        for run in history:
            self.assertEqual(run["experiment_id"], exp_id)
        start_times = [run["run_start_time"] for run in history]
        self.assertEqual(
            start_times,
            sorted(start_times, reverse=True),
            "Runs should be ordered by start_time descending",
        )

    def test_get_model(self):
        exp_id = self.tracker.create_experiment("get_model_test")
        run_id = self.tracker.start_run(exp_id)
        params = {"alpha": 0.5, "l1_ratio": 0.7}
        self.tracker.log_model(run_id, "TestModel", params)
        model = self.tracker.get_model(run_id)
        self.assertIsNotNone(model, "Should retrieve one model")
        self.assertEqual(model["model_name"], "TestModel")
        self.assertEqual(model["parameters"], params, "Model parameters should match")

    def test_list_experiments(self) -> None:
        exp1 = self.tracker.create_experiment("Test 1")
        exp2 = self.tracker.create_experiment("Test 2")

        experiments = self.tracker.list_experiments()
        self.assertEqual(len(experiments), 2)
        experiment_ids = {exp["experiment_id"] for exp in experiments}
        self.assertIn(exp1, experiment_ids)
        self.assertIn(exp2, experiment_ids)

    def test_find_experiments(self) -> None:
        exp1 = self.tracker.create_experiment("ML Model Test")
        exp2 = self.tracker.create_experiment("Data Analysis")
        exp3 = self.tracker.create_experiment("ML Feature Engineering")

        ml_experiments = self.tracker.find_experiments("ML")
        self.assertEqual(len(ml_experiments), 2)
        experiment_ids = {exp["experiment_id"] for exp in ml_experiments}
        self.assertIn(exp1, experiment_ids)
        self.assertIn(exp3, experiment_ids)
        self.assertNotIn(exp2, experiment_ids)

    def test_delete_experiment(self) -> None:
        exp_id = self.tracker.create_experiment("To Delete")
        self.assertIsNotNone(self.tracker.get_experiment(exp_id))

        self.tracker.delete_experiment(exp_id)
        self.assertIsNone(self.tracker.get_experiment(exp_id))

    def test_delete_nonexistent_experiment(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.delete_experiment(9999)

    def test_log_predictions_with_index(self) -> None:
        exp_id = self.tracker.create_experiment("index_test")
        run_id = self.tracker.start_run(exp_id)

        test_preds = [0.1, 0.2, 0.3]
        test_actuals = [0.15, 0.25, 0.35]
        test_index = [10, 20, 30]

        self.tracker.log_predictions(run_id, test_preds, test_actuals, index=test_index)

        result = self.tracker.get_predictions(run_id)

        self.assertEqual(result["predictions"], test_preds)
        self.assertEqual(result["actuals"], test_actuals)

    def test_log_predictions_without_actuals(self) -> None:
        exp_id = self.tracker.create_experiment("predictions_only_test")
        run_id = self.tracker.start_run(exp_id)

        test_preds = [0.8, 0.6, 0.9, 0.7]
        self.tracker.log_predictions(run_id, test_preds)

        result = self.tracker.get_predictions(run_id)
        self.assertEqual(result["predictions"], test_preds)
        self.assertEqual(result["actuals"], [None] * len(test_preds))

        with self.assertRaises(ValueError):
            self.tracker.get_metrics(run_id)

    def test_log_predictions_with_index_no_actuals(self) -> None:
        exp_id = self.tracker.create_experiment("indexed_predictions_only_test")
        run_id = self.tracker.start_run(exp_id)

        test_preds = [0.5, 0.4, 0.3]
        test_index = [100, 200, 300]
        self.tracker.log_predictions(run_id, test_preds, index=test_index)

        result = self.tracker.get_predictions(run_id)
        self.assertEqual(result["predictions"], test_preds)
        self.assertEqual(result["actuals"], [None] * len(test_preds))
        self.assertEqual(result["index"], test_index)

        with self.assertRaises(ValueError):
            self.tracker.get_metrics(run_id)
        self.assertEqual(result["index"], test_index)

    def test_log_predictions_index_length_validation(self) -> None:
        exp_id = self.tracker.create_experiment("validation_test")
        run_id = self.tracker.start_run(exp_id)

        with self.assertRaises(ValueError):
            self.tracker.log_predictions(
                run_id, [1, 2], [1, 2], index=[1]
            )  # mismatched index length

    def test_enhanced_serialization(self) -> None:
        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        exp_id = self.tracker.create_experiment("serialization_test")
        run_id = self.tracker.start_run(exp_id)

        params = {
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
            "status": Status.ACTIVE,
            "normal_value": 42,
        }

        self.tracker.log_model(run_id, "test_model", params)
        model = self.tracker.get_model(run_id)

        self.assertEqual(model["parameters"]["timestamp"], "2023-01-01T12:00:00")
        self.assertEqual(model["parameters"]["status"], "ACTIVE")
        self.assertEqual(model["parameters"]["normal_value"], 42)

    def test_artifacts(self) -> None:
        exp_id = self.tracker.create_experiment("artifact_test")
        run_id = self.tracker.start_run(exp_id)

        test_data = b"test binary data"
        artifact_id = self.tracker.log_artifact(run_id, test_data, "model", "test.pkl")

        self.assertIsInstance(artifact_id, int)

        artifacts = self.tracker.get_artifacts(run_id)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["artifact_type"], "model")
        self.assertEqual(artifacts[0]["filename"], "test.pkl")

        retrieved_data = self.tracker.get_artifact_data(artifact_id)
        self.assertEqual(retrieved_data, test_data)

    def test_enhanced_log_predictions(self) -> None:
        exp_id = self.tracker.create_experiment("enhanced_predictions_test")
        run_id = self.tracker.start_run(exp_id)

        preds = [1.0, 2.0, 3.0]
        actuals = [1.1, 1.9, 3.05]

        self.tracker.log_predictions(
            run_id, preds, actuals, metrics=["rmse", "mae", "mape"]
        )

        metrics = self.tracker.get_metrics(run_id)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)

    def test_find_runs(self) -> None:
        exp_id = self.tracker.create_experiment("find_runs_test")

        run_id1 = self.tracker.start_run(exp_id)
        self.tracker.log_tag("run", run_id1, "model", "forest")
        self.tracker.log_tag("run", run_id1, "dataset", "iris")

        run_id2 = self.tracker.start_run(exp_id)
        self.tracker.log_tag("run", run_id2, "model", "svm")
        self.tracker.log_tag("run", run_id2, "dataset", "iris")

        forest_runs = self.tracker.find_runs({"model": "forest"}, exp_id)
        self.assertEqual(len(forest_runs), 1)
        self.assertIn(run_id1, forest_runs)

        iris_runs = self.tracker.find_runs({"dataset": "iris"}, exp_id)
        self.assertEqual(len(iris_runs), 2)

    def test_context_manager(self) -> None:
        exp_id = self.tracker.create_experiment("context_test")

        with self.tracker.run(exp_id, tags={"model": "test", "version": "1.0"}) as run:
            run.log_model("test_model", {"param": 1})
            run.log_predictions([1.0, 2.0], [1.1, 1.9])

        runs = self.tracker.get_run_history(exp_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_status"], "COMPLETED")

        tags = self.tracker.get_tags("run", runs[0]["run_id"])
        self.assertEqual(tags["model"], "test")
        self.assertEqual(tags["version"], "1.0")

    def test_context_manager_failure(self) -> None:
        exp_id = self.tracker.create_experiment("context_failure_test")

        try:
            with self.tracker.run(exp_id) as run:
                run.log_model("test_model", {"param": 1})
                raise ValueError("Test error")
        except ValueError:
            pass

        runs = self.tracker.get_run_history(exp_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_status"], "FAILED")
        self.assertIn("Test error", runs[0]["error"])

    def test_aggregate(self) -> None:
        exp_id = self.tracker.create_experiment("aggregate_test")

        run_id1 = self.tracker.start_run(exp_id)
        self.tracker.log_model(run_id1, "model_a", {"param": 1})
        self.tracker.log_metric(run_id1, "accuracy", 0.85)
        self.tracker.log_tag("run", run_id1, "model", "model_a")

        run_id2 = self.tracker.start_run(exp_id)
        self.tracker.log_model(run_id2, "model_a", {"param": 2})
        self.tracker.log_metric(run_id2, "accuracy", 0.90)
        self.tracker.log_tag("run", run_id2, "model", "model_a")

        results = self.tracker.aggregate(exp_id, "accuracy", group_by=["model"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model"], "model_a")
        self.assertAlmostEqual(results[0]["accuracy_mean"], 0.875, places=3)


    def test_log_metrics_batch(self) -> None:
        exp_id = self.tracker.create_experiment("batch_metrics_test")
        run_id = self.tracker.start_run(exp_id)

        metrics_dict = {
            "mape_h1": 0.05,
            "mape_h2": 0.08,
            "mape_h3": 0.12,
            "mape_h4": 0.15,
        }
        self.tracker.log_metrics(run_id, metrics_dict)

        stored_metrics = self.tracker.get_metrics(run_id)
        for name, value in metrics_dict.items():
            self.assertIn(name, stored_metrics)
            self.assertAlmostEqual(stored_metrics[name], value, places=5)

    def test_log_metrics_batch_via_run_handle(self) -> None:
        exp_id = self.tracker.create_experiment("batch_metrics_handle_test")

        with self.tracker.run(exp_id) as run:
            run.log_metrics({"accuracy": 0.95, "f1": 0.92, "precision": 0.94})

        runs = self.tracker.get_run_history(exp_id)
        metrics = self.tracker.get_metrics(runs[0]["run_id"])
        self.assertEqual(metrics["accuracy"], 0.95)
        self.assertEqual(metrics["f1"], 0.92)
        self.assertEqual(metrics["precision"], 0.94)

    def test_log_metrics_batch_updates_existing(self) -> None:
        exp_id = self.tracker.create_experiment("batch_update_test")
        run_id = self.tracker.start_run(exp_id)

        self.tracker.log_metric(run_id, "accuracy", 0.80)
        self.tracker.log_metrics(run_id, {"accuracy": 0.90, "f1": 0.85})

        metrics = self.tracker.get_metrics(run_id)
        self.assertEqual(metrics["accuracy"], 0.90)
        self.assertEqual(metrics["f1"], 0.85)

    def test_get_all_runs(self) -> None:
        exp_id = self.tracker.create_experiment("get_all_runs_test")

        run_id1 = self.tracker.start_run(exp_id)
        self.tracker.log_model(run_id1, "model_a", {"lr": 0.01})
        self.tracker.log_tag("run", run_id1, "fold", "0")
        self.tracker.log_metric(run_id1, "accuracy", 0.85)
        self.tracker.log_predictions(run_id1, [1.0, 2.0], [1.1, 1.9])
        self.tracker.end_run(run_id1)

        run_id2 = self.tracker.start_run(exp_id)
        self.tracker.log_model(run_id2, "model_b", {"lr": 0.001})
        self.tracker.log_tag("run", run_id2, "fold", "1")
        self.tracker.log_metric(run_id2, "accuracy", 0.90)
        self.tracker.end_run(run_id2)

        all_runs = self.tracker.get_all_runs(exp_id)

        self.assertEqual(len(all_runs), 2)

        for run in all_runs:
            self.assertIn("run_id", run)
            self.assertIn("run_status", run)
            self.assertIn("model", run)
            self.assertIn("tags", run)
            self.assertIn("metrics", run)

        run1_data = next(r for r in all_runs if r["run_id"] == run_id1)
        self.assertEqual(run1_data["model"]["model_name"], "model_a")
        self.assertEqual(run1_data["tags"]["fold"], "0")
        self.assertEqual(run1_data["metrics"]["accuracy"], 0.85)

        run2_data = next(r for r in all_runs if r["run_id"] == run_id2)
        self.assertEqual(run2_data["model"]["model_name"], "model_b")
        self.assertEqual(run2_data["tags"]["fold"], "1")
        self.assertEqual(run2_data["metrics"]["accuracy"], 0.90)

    def test_get_all_runs_empty_experiment(self) -> None:
        exp_id = self.tracker.create_experiment("empty_experiment")
        all_runs = self.tracker.get_all_runs(exp_id)
        self.assertEqual(all_runs, [])

    def test_get_all_runs_partial_data(self) -> None:
        exp_id = self.tracker.create_experiment("partial_data_test")

        run_id = self.tracker.start_run(exp_id)
        self.tracker.end_run(run_id)

        all_runs = self.tracker.get_all_runs(exp_id)

        self.assertEqual(len(all_runs), 1)
        self.assertIsNone(all_runs[0]["model"])
        self.assertEqual(all_runs[0]["tags"], {})
        self.assertEqual(all_runs[0]["metrics"], {})


if __name__ == "__main__":
    unittest.main()
