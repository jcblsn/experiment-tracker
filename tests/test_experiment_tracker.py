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

    def test_log_predictions_validation(self) -> None:
        exp_id = self.tracker.create_experiment("validation_test")
        run_id = self.tracker.start_run(exp_id)

        with self.assertRaises(ValueError):
            self.tracker.log_predictions(run_id, [1, 2], [1])  # mismatched lengths

        with self.assertRaises(TypeError):
            self.tracker.log_predictions(run_id, "not a list", [1])  # invalid type


if __name__ == "__main__":
    unittest.main()
