import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.experiment_tracker import ExperimentTracker


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test.db")
        tracker = ExperimentTracker(cls.db_path)

        exp_id = tracker.create_experiment(
            "Test Experiment", "A test experiment for CLI"
        )
        cls.exp_id = exp_id

        run_id1 = tracker.start_run(exp_id)
        tracker.log_model(run_id1, "model_a", {"lr": 0.01, "epochs": 10})
        tracker.log_tag("run", run_id1, "model", "model_a")
        tracker.log_tag("run", run_id1, "fold", "0")
        tracker.log_metric(run_id1, "accuracy", 0.85)
        tracker.log_metric(run_id1, "rmse", 0.15)
        tracker.end_run(run_id1)
        cls.run_id1 = run_id1

        run_id2 = tracker.start_run(exp_id)
        tracker.log_model(run_id2, "model_b", {"lr": 0.001, "epochs": 20})
        tracker.log_tag("run", run_id2, "model", "model_b")
        tracker.log_tag("run", run_id2, "fold", "1")
        tracker.log_metric(run_id2, "accuracy", 0.90)
        tracker.log_metric(run_id2, "rmse", 0.10)
        tracker.end_run(run_id2)
        cls.run_id2 = run_id2

        tracker.conn.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def run_cli(self, *args):
        cmd = [sys.executable, "-m", "experiment_tracker.cli", "--db", self.db_path]
        cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        return result

    def test_list_experiments(self):
        result = self.run_cli("list")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Test Experiment", result.stdout)
        self.assertIn(str(self.exp_id), result.stdout)

    def test_list_experiments_json(self):
        result = self.run_cli("list", "--format", "json")
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertEqual(data[0]["experiment_name"], "Test Experiment")

    def test_list_experiments_csv(self):
        result = self.run_cli("list", "--format", "csv")
        self.assertEqual(result.returncode, 0)
        self.assertIn("experiment_id", result.stdout)
        self.assertIn("experiment_name", result.stdout)

    def test_list_experiments_search(self):
        result = self.run_cli("list", "--search", "Test")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Test Experiment", result.stdout)

    def test_show_experiment(self):
        result = self.run_cli("show", str(self.exp_id))
        self.assertEqual(result.returncode, 0)
        self.assertIn("Test Experiment", result.stdout)
        self.assertIn("COMPLETED", result.stdout)

    def test_show_experiment_json(self):
        result = self.run_cli("show", str(self.exp_id), "--format", "json")
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertEqual(data["experiment_name"], "Test Experiment")
        self.assertIn("run_counts", data)

    def test_show_nonexistent_experiment(self):
        result = self.run_cli("show", "9999")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error", result.stderr)

    def test_runs(self):
        result = self.run_cli("runs", str(self.exp_id))
        self.assertEqual(result.returncode, 0)
        self.assertIn("model_a", result.stdout)
        self.assertIn("model_b", result.stdout)

    def test_runs_json(self):
        result = self.run_cli("runs", str(self.exp_id), "--format", "json")
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

    def test_runs_filter_by_tag(self):
        result = self.run_cli("runs", str(self.exp_id), "--tag", "model=model_a")
        self.assertEqual(result.returncode, 0)
        self.assertIn("model_a", result.stdout)
        self.assertNotIn("model_b", result.stdout)

    def test_runs_filter_by_status(self):
        result = self.run_cli("runs", str(self.exp_id), "--status", "COMPLETED")
        self.assertEqual(result.returncode, 0)
        self.assertIn("COMPLETED", result.stdout)

    def test_metrics(self):
        result = self.run_cli("metrics", str(self.run_id1))
        self.assertEqual(result.returncode, 0)
        self.assertIn("accuracy", result.stdout)
        self.assertIn("0.85", result.stdout)

    def test_metrics_json(self):
        result = self.run_cli("metrics", str(self.run_id1), "--format", "json")
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertEqual(data["run_id"], self.run_id1)
        self.assertEqual(data["metrics"]["accuracy"], 0.85)

    def test_best_maximize(self):
        result = self.run_cli("best", str(self.exp_id), "--metric", "accuracy")
        self.assertEqual(result.returncode, 0)
        self.assertIn(str(self.run_id2), result.stdout)
        self.assertIn("0.9", result.stdout)

    def test_best_minimize(self):
        result = self.run_cli(
            "best", str(self.exp_id), "--metric", "rmse", "--minimize"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(str(self.run_id2), result.stdout)
        self.assertIn("0.1", result.stdout)

    def test_best_json(self):
        result = self.run_cli(
            "best", str(self.exp_id), "--metric", "accuracy", "--format", "json"
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertEqual(data["run_id"], self.run_id2)
        self.assertEqual(data["accuracy"], 0.90)

    def test_best_with_tag_filter(self):
        result = self.run_cli(
            "best", str(self.exp_id), "--metric", "accuracy", "--tag", "model=model_a"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(str(self.run_id1), result.stdout)
        self.assertIn("0.85", result.stdout)

    def test_aggregate_basic(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "accuracy"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("accuracy_mean", result.stdout)
        self.assertIn("accuracy_std", result.stdout)
        self.assertIn("accuracy_count", result.stdout)

    def test_aggregate_group_by_tag(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "accuracy", "--group-by", "model"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("model_a", result.stdout)
        self.assertIn("model_b", result.stdout)

    def test_aggregate_group_by_param(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "accuracy", "--group-by-param", "lr"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("0.01", result.stdout)
        self.assertIn("0.001", result.stdout)

    def test_aggregate_custom_agg(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "accuracy",
            "--agg", "min", "--agg", "max"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("accuracy_min", result.stdout)
        self.assertIn("accuracy_max", result.stdout)
        self.assertNotIn("accuracy_mean", result.stdout)

    def test_aggregate_json(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "accuracy",
            "--group-by", "model", "--format", "json"
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertIn("model", data[0])
        self.assertIn("accuracy_mean", data[0])

    def test_aggregate_with_tag_filter(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "accuracy",
            "--tag", "fold=0"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("1", result.stdout)

    def test_aggregate_nonexistent_metric(self):
        result = self.run_cli(
            "aggregate", str(self.exp_id), "--metric", "nonexistent"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error", result.stderr)

    def test_compare(self):
        result = self.run_cli("compare", str(self.run_id1), str(self.run_id2))
        self.assertEqual(result.returncode, 0)
        self.assertIn("model_a", result.stdout)
        self.assertIn("model_b", result.stdout)
        self.assertIn("accuracy", result.stdout)

    def test_compare_json(self):
        result = self.run_cli(
            "compare", str(self.run_id1), str(self.run_id2), "--format", "json"
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertEqual(len(data), 2)
        self.assertIn("metrics", data[0])

    def test_compare_single_run_error(self):
        result = self.run_cli("compare", str(self.run_id1))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error", result.stderr)

    def test_sql(self):
        result = self.run_cli("sql", "SELECT COUNT(*) as count FROM runs")
        self.assertEqual(result.returncode, 0)
        self.assertIn("2", result.stdout)

    def test_sql_json(self):
        result = self.run_cli(
            "sql", "SELECT experiment_name FROM experiments", "--format", "json"
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertEqual(data[0]["experiment_name"], "Test Experiment")

    def test_sql_csv(self):
        result = self.run_cli(
            "sql", "SELECT experiment_name FROM experiments", "--format", "csv"
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("experiment_name", result.stdout)
        self.assertIn("Test Experiment", result.stdout)

    def test_sql_error(self):
        result = self.run_cli("sql", "SELECT * FROM nonexistent_table")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SQL Error", result.stderr)

    def test_export(self):
        export_dir = tempfile.mkdtemp()
        try:
            result = self.run_cli("export", str(self.exp_id), export_dir)
            self.assertEqual(result.returncode, 0)
            self.assertIn("Exported", result.stdout)
            export_subdir = os.listdir(export_dir)[0]
            full_path = os.path.join(export_dir, export_subdir)
            self.assertTrue(os.path.exists(os.path.join(full_path, "experiments.csv")))
            self.assertTrue(os.path.exists(os.path.join(full_path, "runs.csv")))
        finally:
            shutil.rmtree(export_dir)

    def test_export_nonexistent_experiment(self):
        export_dir = tempfile.mkdtemp()
        try:
            result = self.run_cli("export", "9999", export_dir)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Error", result.stderr)
        finally:
            shutil.rmtree(export_dir)


class TestCLIDatabaseDiscovery(unittest.TestCase):
    def test_no_database_error(self):
        temp_dir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "experiment_tracker.cli", "list"],
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No database found", result.stderr)
        finally:
            shutil.rmtree(temp_dir)

    def test_env_var_database(self):
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "env_test.db")
        tracker = ExperimentTracker(db_path)
        tracker.create_experiment("Env Test")
        tracker.conn.close()

        try:
            env = os.environ.copy()
            env["EXPT_DB"] = db_path
            result = subprocess.run(
                [sys.executable, "-m", "experiment_tracker.cli", "list"],
                capture_output=True,
                text=True,
                env=env,
                cwd=temp_dir,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("Env Test", result.stdout)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
