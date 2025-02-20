import json
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experiment_tracker import ExperimentTracker


class TestExperimentTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = ":memory:"
        self.tracker = ExperimentTracker(self.db_path)

    def tearDown(self) -> None:
        if hasattr(self, "tracker"):
            self.tracker.conn.close()

    def test_table_creation(self) -> None:
        required_tables = {"experiments", "runs", "models", "predictions"}
        cursor = self.tracker.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('experiments', 'runs', 'models', 'predictions')
        """)
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
            "SELECT name, description, created_at FROM experiments WHERE id=?",
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
            "SELECT experiment_id, status, start_time FROM runs WHERE id=?", (run_id,)
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
        self.tracker.log_model(run_id, "test_model", test_params)

        cursor = self.tracker.conn.cursor()
        cursor.execute("SELECT name, parameters FROM models WHERE run_id=?", (run_id,))
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
            "SELECT predictions, actuals FROM predictions WHERE run_id=?", (run_id,)
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Prediction record should exist")
        self.assertEqual(
            json.loads(result[0].decode("utf-8")),
            test_preds,
            "Predictions should match after deserialization",
        )
        self.assertEqual(
            json.loads(result[1].decode("utf-8")),
            test_actuals,
            "Actuals should match after deserialization",
        )

    def test_metrics_table_exists(self):
        """Ensure that the metrics table exists in the database."""
        cursor = self.tracker.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='metrics'
        """)
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Metrics table should exist in the database")

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
        cursor.execute("SELECT status, end_time, error FROM runs WHERE id=?", (run_id,))
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
        cursor.execute("SELECT status, end_time, error FROM runs WHERE id=?", (run_id,))
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
        cursor.execute("SELECT name FROM experiments WHERE id=?", (exp_id,))
        self.assertEqual(cursor.fetchone()[0], "Integration Test")

        cursor.execute("SELECT status FROM runs WHERE id=?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], "COMPLETED")

        cursor.execute("SELECT parameters FROM models WHERE run_id=?", (run_id,))
        self.assertEqual(json.loads(cursor.fetchone()[0]), test_params)

        cursor.execute(
            "SELECT predictions, actuals FROM predictions WHERE run_id=?", (run_id,)
        )
        preds_blob, actuals_blob = cursor.fetchone()
        self.assertEqual(json.loads(preds_blob.decode("utf-8")), test_preds)
        self.assertEqual(json.loads(actuals_blob.decode("utf-8")), test_actuals)

    def test_calculate_metrics(self):
        exp_id = self.tracker.create_experiment("metrics_test")
        run_id = self.tracker.start_run(exp_id)
        preds = [1.0, 2.0, 3.0]
        actuals = [1.1, 1.9, 3.05]
        import math

        n = len(preds)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)
        mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / n
        mean_actual = sum(actuals) / n
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        r2 = (
            1 - (sum((a - p) ** 2 for p, a in zip(preds, actuals)) / ss_tot)
            if ss_tot != 0
            else 0
        )
        self.tracker.log_predictions(run_id, preds, actuals)
        cursor = self.tracker.conn.cursor()
        cursor.execute("SELECT name, value FROM metrics WHERE run_id=?", (run_id,))
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 3, "Should insert three metrics entries")
        metrics = {name: value for name, value in rows}
        self.assertAlmostEqual(metrics["rmse"], rmse, places=5)
        self.assertAlmostEqual(metrics["mae"], mae, places=5)
        self.assertAlmostEqual(metrics["r2"], r2, places=5)

    def test_get_experiment(self):
        exp_name = "get_experiment_test"
        exp_description = "Test experiment retrieval"
        exp_id = self.tracker.create_experiment(exp_name, exp_description)
        experiment = self.tracker.get_experiment(exp_id)
        self.assertIsNotNone(experiment, "Experiment should be retrieved successfully")
        self.assertEqual(experiment.get("name"), exp_name)
        self.assertEqual(experiment.get("description"), exp_description)

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
        start_times = [run["start_time"] for run in history]
        self.assertEqual(
            start_times,
            sorted(start_times, reverse=True),
            "Runs should be ordered by start_time descending",
        )

    def test_get_models(self):
        exp_id = self.tracker.create_experiment("get_models_test")
        run_id = self.tracker.start_run(exp_id)
        params = {"alpha": 0.5, "l1_ratio": 0.7}
        self.tracker.log_model(run_id, "TestModel", params)
        models = self.tracker.get_models(run_id)
        self.assertEqual(len(models), 1, "Should retrieve one model")
        model = models[0]
        self.assertEqual(model["name"], "TestModel")
        self.assertEqual(model["parameters"], params, "Model parameters should match")


if __name__ == "__main__":
    unittest.main()
