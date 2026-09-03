import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiment_tracker import ExperimentTracker, scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(path: str) -> int:
    with ExperimentTracker(path) as tracker:
        experiment_id = tracker.experiment(
            "GSL_CV", "walk-forward", tags={"train_start": "1960-01-01"}
        )
        for name, base, family in (
            ("blend", 0.521, "blend"),
            ("blend3_swe", 0.510, "blend"),
            ("naive_last", 1.331, "baseline"),
        ):
            with tracker.run(
                experiment_id, name=name, params={"alpha": "gcv"}, tags={"family": family}
            ) as run:
                predictions = [base, base * 2]
                actuals = [0.0, 0.0]
                run.log_predictions(predictions, actuals, dims={"h": 6})
                run.log_metric("mae", scoring.mae(predictions, actuals), dims={"h": 6})
                run.log_metric("mae", base * 4, dims={"h": 24})
                run.set_note(f"{name} note")
    return experiment_id


class CLITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.db_path = os.path.join(self.directory, "forecast_experiments.db")
        self.experiment_id = build(self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory)

    def run_cli(self, *args, cwd=None, env=None, expect=0):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.path.join(ROOT, "src")
        environment.pop("EXPT_DB", None)
        environment.update(env or {})
        result = subprocess.run(
            [sys.executable, "-m", "experiment_tracker.cli", *args],
            capture_output=True,
            text=True,
            cwd=cwd or self.directory,
            env=environment,
        )
        if expect is not None:
            self.assertEqual(
                expect,
                result.returncode,
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
        return result


class TestDatabaseDiscovery(CLITestCase):
    def test_the_only_db_in_the_directory_is_found_without_a_flag(self) -> None:
        result = self.run_cli("list")
        self.assertIn("GSL_CV", result.stdout)

    def test_an_explicit_flag_still_works(self) -> None:
        result = self.run_cli("--db", self.db_path, "list", cwd=ROOT)
        self.assertIn("GSL_CV", result.stdout)

    def test_the_environment_variable_still_works(self) -> None:
        result = self.run_cli("list", cwd=ROOT, env={"EXPT_DB": self.db_path})
        self.assertIn("GSL_CV", result.stdout)

    def test_no_database_explains_the_options(self) -> None:
        empty = tempfile.mkdtemp()
        try:
            result = self.run_cli("list", cwd=empty, expect=1)
            self.assertIn("No database found", result.stderr)
        finally:
            shutil.rmtree(empty)

    def test_several_databases_ask_which(self) -> None:
        shutil.copy(self.db_path, os.path.join(self.directory, "other.db"))
        result = self.run_cli("list", expect=1)
        self.assertIn("Several databases", result.stderr)
        self.assertIn("other.db", result.stderr)

    def test_a_legacy_file_does_not_count_as_a_candidate(self) -> None:
        shutil.copy(self.db_path, os.path.join(self.directory, "old.legacy.db"))
        result = self.run_cli("list")
        self.assertIn("GSL_CV", result.stdout)


class TestSchema(CLITestCase):
    def test_schema_prints_tables_and_columns(self) -> None:
        result = self.run_cli("schema")
        for table in ("experiments", "runs", "metrics", "predictions", "artifacts"):
            self.assertIn(table, result.stdout)
        self.assertIn("dims", result.stdout)

    def test_schema_names_the_metric_column(self) -> None:
        self.assertIn("metric TEXT", self.run_cli("schema").stdout)


class TestListAndShow(CLITestCase):
    def test_list_counts_runs(self) -> None:
        self.assertIn("3", self.run_cli("list").stdout)

    def test_show_defaults_to_the_newest_experiment(self) -> None:
        self.assertIn("GSL_CV", self.run_cli("show").stdout)

    def test_show_accepts_a_name(self) -> None:
        self.assertIn("GSL_CV", self.run_cli("show", "GSL_CV").stdout)

    def test_show_reports_tags_and_provenance(self) -> None:
        out = self.run_cli("show").stdout
        self.assertIn("train_start=1960-01-01", out)
        self.assertIn("command", out)

    def test_show_json(self) -> None:
        payload = json.loads(self.run_cli("show", "--format", "json").stdout)
        self.assertEqual("GSL_CV", payload[0]["name"])

    def test_an_unknown_experiment_exits_with_a_message(self) -> None:
        result = self.run_cli("show", "9999", expect=1)
        self.assertIn("No experiment", result.stderr)


class TestRuns(CLITestCase):
    def test_runs_lists_every_run(self) -> None:
        out = self.run_cli("runs").stdout
        for name in ("blend", "blend3_swe", "naive_last"):
            self.assertIn(name, out)

    def test_filtering_on_two_tags_returns_the_matching_run(self) -> None:
        out = self.run_cli("runs", "-t", "family=blend", "-t", "alpha=missing").stdout
        self.assertIn("No results.", out)

    def test_filtering_on_one_tag_narrows_the_list(self) -> None:
        out = self.run_cli("runs", "-t", "family=baseline").stdout
        self.assertIn("naive_last", out)
        self.assertNotIn("blend3_swe", out)

    def test_notes_appear(self) -> None:
        self.assertIn("blend note", self.run_cli("runs").stdout)


class TestBest(CLITestCase):
    def test_best_minimizes_without_a_flag(self) -> None:
        """The old default maximized, so it answered with the baseline."""
        out = self.run_cli("best", "-m", "mae", "-d", "h=6").stdout
        self.assertIn("blend3_swe", out)
        self.assertNotIn("naive_last", out)

    def test_maximize_is_opt_in(self) -> None:
        self.assertIn(
            "naive_last",
            self.run_cli("best", "-m", "mae", "-d", "h=6", "--maximize").stdout,
        )

    def test_best_reports_the_dims_it_chose_on(self) -> None:
        self.assertIn('{"h":6}', self.run_cli("best", "-m", "mae", "-d", "h=6").stdout)

    def test_a_dim_filter_is_typed(self) -> None:
        at_24 = self.run_cli("best", "-m", "mae", "-d", "h=24", "--format", "json").stdout
        self.assertEqual({"h": 24}, json.loads(at_24)[0]["dims"])

    def test_an_absent_metric_exits_with_a_message(self) -> None:
        result = self.run_cli("best", "-m", "crps", expect=1)
        self.assertIn("No runs carry", result.stderr)


class TestMetrics(CLITestCase):
    def test_metrics_list_dims(self) -> None:
        self.assertIn('{"h":6}', self.run_cli("metrics", "-m", "mae").stdout)

    def test_pivot_spreads_a_dim_across_columns(self) -> None:
        out = self.run_cli("metrics", "-m", "mae", "--pivot", "h").stdout
        header = out.splitlines()[0]
        self.assertIn("6", header)
        self.assertIn("24", header)
        self.assertIn("blend", out)

    def test_filtering_by_run(self) -> None:
        out = self.run_cli("metrics", "1", "--format", "json").stdout
        self.assertTrue(all(row["run_id"] == 1 for row in json.loads(out)))


class TestDiff(CLITestCase):
    def test_diff_reports_a_delta(self) -> None:
        out = self.run_cli("diff", "1", "2", "-m", "mae").stdout
        self.assertIn("delta", out)
        self.assertIn("blend3_swe", out)

    def test_diff_of_three_runs_has_no_delta_column(self) -> None:
        out = self.run_cli("diff", "1", "2", "3", "-m", "mae").stdout
        self.assertNotIn("delta", out)

    def test_diff_needs_two_runs(self) -> None:
        self.run_cli("diff", "1", expect=1)


class TestLog(CLITestCase):
    def test_log_is_one_line_per_run(self) -> None:
        out = self.run_cli("log", "-m", "mae", "-d", "h=6").stdout
        body = [line for line in out.splitlines()[2:] if line.strip()]
        self.assertEqual(3, len(body))

    def test_log_carries_the_note_and_the_metric(self) -> None:
        out = self.run_cli("log", "-m", "mae", "-d", "h=6").stdout
        self.assertIn("blend note", out)
        self.assertIn("mae", out.splitlines()[0])

    def test_log_csv_is_a_ledger(self) -> None:
        out = self.run_cli("log", "-m", "mae", "-d", "h=6", "--format", "csv").stdout
        self.assertTrue(out.startswith("date,run_id,model,status,mae,note"))


class TestAudit(CLITestCase):
    def test_a_sound_metric_agrees(self) -> None:
        out = self.run_cli("audit", "1", "-m", "mae", "-d", "h=6").stdout
        self.assertIn("agrees     True", out)

    def test_a_wrong_metric_disagrees_and_exits_non_zero(self) -> None:
        with ExperimentTracker(self.db_path) as tracker:
            tracker.log_metric(1, "mae", 99.0, dims={"h": 6})
        result = self.run_cli("audit", "1", "-m", "mae", "-d", "h=6", expect=1)
        self.assertIn("agrees     False", result.stdout)

    def test_a_metric_that_cannot_be_recomputed_is_rejected(self) -> None:
        self.run_cli("audit", "1", "-m", "cov90", expect=2)


class TestSnapshot(CLITestCase):
    def test_snapshot_writes_committable_files(self) -> None:
        target = os.path.join(self.directory, "results")
        self.run_cli("snapshot", str(self.experiment_id), target)
        self.assertEqual({"experiment.json", "runs.csv", "metrics.csv"}, set(os.listdir(target)))

    def test_predictions_are_opt_in(self) -> None:
        target = os.path.join(self.directory, "results")
        self.run_cli("snapshot", str(self.experiment_id), target, "--predictions")
        self.assertIn("predictions.csv", os.listdir(target))


class TestArtifacts(CLITestCase):
    def test_listing_and_fetching(self) -> None:
        with ExperimentTracker(self.db_path) as tracker:
            tracker.log_artifact(b"rows", "cv", "cv.parquet", run=1)
        self.assertIn("cv.parquet", self.run_cli("artifacts").stdout)
        target = os.path.join(self.directory, "out.parquet")
        self.run_cli("artifacts", "--get", "1", "-o", target)
        with open(target, "rb") as stream:
            self.assertEqual(b"rows", stream.read())


class TestRemove(CLITestCase):
    def test_rm_refuses_without_confirmation(self) -> None:
        result = self.run_cli("rm", str(self.experiment_id), expect=1)
        self.assertIn("Would delete", result.stderr)
        self.assertIn("GSL_CV", self.run_cli("list").stdout)

    def test_rm_with_yes_deletes_the_experiment_and_its_runs(self) -> None:
        self.run_cli("rm", str(self.experiment_id), "--yes")
        self.assertIn("No results.", self.run_cli("list").stdout)
        self.assertIn("No results.", self.run_cli("runs", "--all").stdout)


class TestSql(CLITestCase):
    def test_sql_still_works_as_the_escape_hatch(self) -> None:
        out = self.run_cli("sql", "SELECT COUNT(*) AS n FROM runs").stdout
        self.assertIn("3", out)

    def test_a_bad_query_reports_the_error(self) -> None:
        result = self.run_cli("sql", "SELECT * FROM nope", expect=1)
        self.assertIn("SQL error", result.stderr)


if __name__ == "__main__":
    unittest.main()
