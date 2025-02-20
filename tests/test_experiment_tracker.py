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
        exp_id = self.tracker.create_experiment(exp_name)
        self.assertIsInstance(exp_id, int, "Should return integer ID")

        cursor = self.tracker.conn.cursor()
        cursor.execute("SELECT name, created_at FROM experiments WHERE id=?", (exp_id,))
        result = cursor.fetchone()

        self.assertIsNotNone(result, "Experiment record should exist")
        self.assertEqual(result[0], exp_name, "Stored name should match input")
        self.assertIsNotNone(result[1], "Should have creation timestamp")

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


if __name__ == "__main__":
    unittest.main()
