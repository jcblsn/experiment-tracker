import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime
from enum import Enum

from experiment_tracker import ExperimentTracker, dims_key, scoring


class Colour(Enum):
    RED = 1


class TrackerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ExperimentTracker(":memory:")
        self.experiment_id = self.tracker.experiment("bench", "a benchmark")

    def tearDown(self) -> None:
        self.tracker.close()

    def a_run(self, name="blend", **kwargs):
        return self.tracker.start_run(self.experiment_id, name=name, **kwargs)


class TestSchema(TrackerTestCase):
    def test_tables(self) -> None:
        rows = self.tracker.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        names = {row["name"] for row in rows}
        self.assertEqual(
            {"experiments", "runs", "metrics", "predictions", "tags", "artifacts"},
            names - {"sqlite_sequence"},
        )

    def test_models_table_is_gone(self) -> None:
        rows = self.tracker.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'models'"
        ).fetchall()
        self.assertEqual([], rows)

    def test_every_foreign_key_is_indexed(self) -> None:
        indexed = {
            row["name"]
            for row in self.tracker.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        for expected in (
            "runs_experiment",
            "metrics_run",
            "predictions_run",
            "tags_entity",
            "artifacts_run",
            "artifacts_experiment",
        ):
            self.assertIn(expected, indexed)

    def test_reopening_a_file_is_not_destructive(self) -> None:
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "e.db")
            with ExperimentTracker(path) as first:
                experiment_id = first.experiment("kept")
            with ExperimentTracker(path) as second:
                self.assertEqual("kept", second.get_experiment(experiment_id)["name"])
        finally:
            shutil.rmtree(directory)


class TestExperiments(TrackerTestCase):
    def test_columns_are_unprefixed(self) -> None:
        record = self.tracker.get_experiment(self.experiment_id)
        self.assertEqual("bench", record["name"])
        self.assertEqual("a benchmark", record["description"])
        self.assertIn("created_at", record)

    def test_provenance_is_captured(self) -> None:
        record = self.tracker.get_experiment(self.experiment_id)
        self.assertTrue(record["python"])
        self.assertIn("argv", record)
        self.assertIn("git_commit", record)

    def test_explicit_provenance_wins_for_an_imported_run(self) -> None:
        """Importing a historical record must not stamp the importer's commit on it."""
        experiment_id = self.tracker.experiment(
            "imported",
            provenance={"git_commit": "4b86efd0", "git_dirty": 0, "argv": "gsl-cv"},
        )
        record = self.tracker.get_experiment(experiment_id)
        self.assertEqual("4b86efd0", record["git_commit"])
        self.assertEqual("gsl-cv", record["argv"])

    def test_partial_provenance_keeps_the_captured_rest(self) -> None:
        experiment_id = self.tracker.experiment(
            "imported", provenance={"git_commit": "4b86efd0"}
        )
        record = self.tracker.get_experiment(experiment_id)
        self.assertEqual("4b86efd0", record["git_commit"])
        self.assertTrue(record["python"])

    def test_get_or_create_reuses_the_newest_of_a_name(self) -> None:
        first = self.tracker.experiment("loop", get_or_create=True)
        second = self.tracker.experiment("loop", get_or_create=True)
        self.assertEqual(first, second)

    def test_without_get_or_create_a_second_call_inserts(self) -> None:
        first = self.tracker.experiment("loop")
        second = self.tracker.experiment("loop")
        self.assertNotEqual(first, second)

    def test_unknown_experiment_reads_as_none(self) -> None:
        self.assertIsNone(self.tracker.get_experiment(9999))

    def test_latest_experiment_replaces_a_max_id_query(self) -> None:
        newest = self.tracker.experiment("newest")
        self.assertEqual(newest, self.tracker.latest_experiment()["experiment_id"])
        self.assertEqual(
            self.experiment_id, self.tracker.latest_experiment("bench")["experiment_id"]
        )

    def test_latest_experiment_of_an_unknown_name_is_none(self) -> None:
        self.assertIsNone(self.tracker.latest_experiment("absent"))

    def test_experiments_counts_runs(self) -> None:
        self.a_run()
        self.a_run(name="other")
        listed = {row["experiment_id"]: row for row in self.tracker.experiments()}
        self.assertEqual(2, listed[self.experiment_id]["runs"])

    def test_note_round_trips(self) -> None:
        self.tracker.set_note("experiment", self.experiment_id, "estimator changed here")
        record = self.tracker.get_experiment(self.experiment_id)
        self.assertEqual("estimator changed here", record["note"])

    def test_note_on_an_unknown_entity_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.set_note("run", 9999, "x")


class TestRuns(TrackerTestCase):
    def test_name_and_params_live_on_the_run(self) -> None:
        run_id = self.a_run(params={"alpha": "gcv", "features": ["swe", "prec"]})
        run = self.tracker.get_run(run_id)
        self.assertEqual("blend", run["name"])
        self.assertEqual({"alpha": "gcv", "features": ["swe", "prec"]}, run["params"])

    def test_params_default_to_an_empty_mapping(self) -> None:
        self.assertEqual({}, self.tracker.get_run(self.a_run())["params"])

    def test_params_serialize_awkward_values(self) -> None:
        run_id = self.a_run(params={"when": datetime(2026, 9, 3), "colour": Colour.RED, "n": 3})
        params = self.tracker.get_run(run_id)["params"]
        self.assertEqual("2026-09-03T00:00:00", params["when"])
        self.assertEqual("RED", params["colour"])
        self.assertEqual(3, params["n"])

    def test_context_manager_completes_a_run(self) -> None:
        with self.tracker.run(self.experiment_id, name="blend") as run:
            run_id = run.run_id
        run = self.tracker.get_run(run_id)
        self.assertEqual("completed", run["status"])
        self.assertIsNotNone(run["ended_at"])

    def test_context_manager_records_a_failure_and_re_raises(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            with self.tracker.run(self.experiment_id, name="broken") as run:
                run_id = run.run_id
                raise ZeroDivisionError("no cutoffs succeeded")
        run = self.tracker.get_run(run_id)
        self.assertEqual("failed", run["status"])
        self.assertIn("no cutoffs", run["error"])

    def test_a_run_needs_a_real_experiment(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.start_run(9999)

    def test_runs_span_experiments_when_none_is_given(self) -> None:
        other = self.tracker.experiment("second")
        self.a_run()
        self.tracker.start_run(other, name="blend")
        self.assertEqual(2, len(self.tracker.runs()))
        self.assertEqual(1, len(self.tracker.runs(experiment=other)))

    def test_runs_filter_by_name_and_status(self) -> None:
        self.a_run(name="blend")
        failed = self.a_run(name="broken")
        self.tracker.end_run(failed, success=False, error="boom")
        self.assertEqual(["blend"], [r["name"] for r in self.tracker.runs(name="blend")])
        self.assertEqual(["broken"], [r["name"] for r in self.tracker.runs(status="failed")])

    def test_runs_accepts_an_experiment_row_as_well_as_an_id(self) -> None:
        self.a_run()
        record = self.tracker.get_experiment(self.experiment_id)
        self.assertEqual(1, len(self.tracker.runs(experiment=record)))

    def test_order_by_rejects_anything_not_a_run_column(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.runs(order_by="value; DROP TABLE runs")

    def test_order_by_descending(self) -> None:
        first = self.a_run()
        second = self.a_run(name="other")
        ordered = [r["run_id"] for r in self.tracker.runs(order_by="run_id desc")]
        self.assertEqual([second, first], ordered)

    def test_unknown_run_reads_as_none(self) -> None:
        self.assertIsNone(self.tracker.get_run(9999))

    def test_run_note_round_trips(self) -> None:
        with self.tracker.run(self.experiment_id, name="blend") as run:
            run.set_note("keep: peak and lead 6 both improve")
            run_id = run.run_id
        self.assertEqual("keep: peak and lead 6 both improve", self.tracker.get_run(run_id)["note"])


class TestMetrics(TrackerTestCase):
    def test_metrics_carry_dims(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metrics(run_id, {"mae": 0.52, "rmse": 0.68}, dims={"h": 6})
        rows = self.tracker.metrics(runs=run_id)
        self.assertEqual({("mae", 0.52), ("rmse", 0.68)}, {(r["metric"], r["value"]) for r in rows})
        self.assertEqual({"h": 6}, rows[0]["dims"])

    def test_dims_expand_into_columns(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "peak_mae", 0.573, dims={"issue": "feb"})
        row = self.tracker.metrics(runs=run_id)[0]
        self.assertEqual("feb", row["issue"])
        self.assertEqual("blend", row["run_name"])

    def test_a_dim_named_like_a_base_column_does_not_clobber_it(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 1.0, dims={"run_id": "not-an-id"})
        row = self.tracker.metrics(runs=run_id)[0]
        self.assertEqual(run_id, row["run_id"])
        self.assertEqual("not-an-id", row["dims"]["run_id"])

    def test_the_same_metric_at_different_dims_coexists(self) -> None:
        run_id = self.a_run()
        for h in (1, 6, 24):
            self.tracker.log_metric(run_id, "mae", 0.1 * h, dims={"h": h})
        self.assertEqual(3, len(self.tracker.metrics(runs=run_id, metric="mae")))

    def test_relogging_the_same_metric_and_dims_updates(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 1.0, dims={"h": 6})
        self.tracker.log_metric(run_id, "mae", 2.0, dims={"h": 6})
        rows = self.tracker.metrics(runs=run_id)
        self.assertEqual(1, len(rows))
        self.assertEqual(2.0, rows[0]["value"])

    def test_dims_uniqueness_survives_key_reordering(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 1.0, dims={"h": 6, "issue": "feb"})
        self.tracker.log_metric(run_id, "mae", 2.0, dims={"issue": "feb", "h": 6})
        rows = self.tracker.metrics(runs=run_id)
        self.assertEqual(1, len(rows), "reordered keys must not create a second row")
        self.assertEqual(2.0, rows[0]["value"])

    def test_dims_filter_uses_subset_matching(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 1.0, dims={"h": 6, "season": "acc"})
        self.tracker.log_metric(run_id, "mae", 2.0, dims={"h": 24, "season": "acc"})
        rows = self.tracker.metrics(runs=run_id, dims={"h": 6})
        self.assertEqual([1.0], [r["value"] for r in rows])

    def test_metrics_are_not_rounded_on_the_way_in(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 0.5098987654321012, dims={"h": 6})
        self.assertEqual(0.5098987654321012, self.tracker.metrics(runs=run_id)[0]["value"])

    def test_metrics_of_an_unknown_run_are_empty_not_an_error(self) -> None:
        self.assertEqual([], self.tracker.metrics(runs=9999))

    def test_logging_a_metric_on_an_unknown_run_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.log_metric(9999, "mae", 1.0)

    def test_a_failed_run_with_no_metrics_still_appears(self) -> None:
        good = self.a_run(name="blend")
        self.tracker.log_metric(good, "mae", 0.5, dims={"h": 6})
        failed = self.a_run(name="broken")
        self.tracker.end_run(failed, success=False, error="failed at every cutoff")
        names = [r["name"] for r in self.tracker.runs()]
        self.assertIn("broken", names, "a run without metrics must not vanish")


class TestPredictions(TrackerTestCase):
    def test_per_row_dims_make_rows_addressable(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(
            run_id,
            [1.0, 2.0],
            [1.1, 2.1],
            dims=[{"cutoff": "2020-01-01", "h": 1}, {"cutoff": "2020-02-01", "h": 6}],
        )
        rows = self.tracker.predictions(runs=run_id, dims={"h": 6})
        self.assertEqual(1, len(rows))
        self.assertEqual("2020-02-01", rows[0]["cutoff"])

    def test_one_shared_dims_mapping_applies_to_every_row(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0, 2.0], [1.0, 2.0], dims={"h": 3})
        self.assertEqual(2, len(self.tracker.predictions(runs=run_id, dims={"h": 3})))

    def test_no_prediction_row_has_an_empty_dims_when_dims_are_given(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0], [1.0], dims={"h": 1})
        self.assertEqual({"h": 1}, self.tracker.predictions(runs=run_id)[0]["dims"])

    def test_actuals_stay_optional(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0, 2.0], dims={"h": 1})
        self.assertIsNone(self.tracker.predictions(runs=run_id)[0]["actual"])

    def test_logging_predictions_writes_no_metrics(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0, 2.0], [1.5, 2.5], dims={"h": 1})
        self.assertEqual(
            [], self.tracker.metrics(runs=run_id), "predictions must not score themselves"
        )

    def test_replace_clears_the_previous_rows(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0, 2.0], dims={"h": 1})
        self.tracker.log_predictions(run_id, [3.0], dims={"h": 1})
        self.assertEqual([3.0], [r["prediction"] for r in self.tracker.predictions(runs=run_id)])

    def test_replace_false_appends(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0], dims={"h": 1})
        self.tracker.log_predictions(run_id, [2.0], dims={"h": 2}, replace=False)
        self.assertEqual(2, len(self.tracker.predictions(runs=run_id)))

    def test_mismatched_lengths_raise(self) -> None:
        run_id = self.a_run()
        with self.assertRaises(ValueError):
            self.tracker.log_predictions(run_id, [1.0, 2.0], [1.0])

    def test_mismatched_dims_length_raises(self) -> None:
        run_id = self.a_run()
        with self.assertRaises(ValueError):
            self.tracker.log_predictions(run_id, [1.0, 2.0], dims=[{"h": 1}])


class TestTags(TrackerTestCase):
    def test_filtering_on_two_tags_finds_the_run_holding_both(self) -> None:
        first = self.a_run()
        second = self.a_run(name="other")
        self.tracker.log_tags("run", first, {"model": "ols", "fold": "3"})
        self.tracker.log_tags("run", second, {"model": "ols", "fold": "4"})
        found = self.tracker.runs(tags={"model": "ols", "fold": "3"})
        self.assertEqual([first], [r["run_id"] for r in found])

    def test_filtering_on_three_tags(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tags("run", run_id, {"a": "1", "b": "2", "c": "3"})
        self.assertEqual(
            [run_id],
            [r["run_id"] for r in self.tracker.runs(tags={"a": "1", "b": "2", "c": "3"})],
        )

    def test_a_tag_that_does_not_match_excludes_the_run(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tags("run", run_id, {"model": "ols"})
        self.assertEqual([], self.tracker.runs(tags={"model": "ridge"}))

    def test_re_tagging_updates_rather_than_duplicating(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tag("run", run_id, "model", "ols")
        self.tracker.log_tag("run", run_id, "model", "ridge")
        self.assertEqual({"model": "ridge"}, self.tracker.tags("run", run_id))
        self.assertEqual(
            [run_id], [r["run_id"] for r in self.tracker.runs(tags={"model": "ridge"})]
        )

    def test_a_repeated_identical_tag_does_not_hide_the_run(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tag("run", run_id, "model", "ols")
        self.tracker.log_tag("run", run_id, "model", "ols")
        self.assertEqual([run_id], [r["run_id"] for r in self.tracker.runs(tags={"model": "ols"})])

    def test_a_quote_in_a_tag_value_is_data_not_syntax(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tag("run", run_id, "note", "it's fine")
        self.assertEqual(
            [run_id], [r["run_id"] for r in self.tracker.runs(tags={"note": "it's fine"})]
        )

    def test_a_quote_in_a_tag_name_is_data_not_syntax(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tag("run", run_id, "it's", "yes")
        self.assertEqual([run_id], [r["run_id"] for r in self.tracker.runs(tags={"it's": "yes"})])

    def test_a_tag_name_cannot_inject_sql(self) -> None:
        run_id = self.a_run()
        self.tracker.log_tag("run", run_id, "x' OR '1'='1", "v")
        self.assertEqual([], self.tracker.runs(tags={"x": "v"}))
        self.assertIsNotNone(self.tracker.get_run(run_id))

    def test_run_tags_come_back_on_the_run(self) -> None:
        run_id = self.a_run(tags={"model": "ols"})
        self.assertEqual({"model": "ols"}, self.tracker.get_run(run_id)["tags"])

    def test_experiment_tags(self) -> None:
        self.tracker.log_tags("experiment", self.experiment_id, {"train_start": "1960-01-01"})
        self.assertEqual(
            {"train_start": "1960-01-01"},
            self.tracker.tags("experiment", self.experiment_id),
        )

    def test_delete_tag(self) -> None:
        run_id = self.a_run(tags={"model": "ols"})
        self.assertEqual(1, self.tracker.delete_tag("run", run_id, "model"))
        self.assertEqual({}, self.tracker.tags("run", run_id))

    def test_tagging_an_unknown_entity_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.log_tag("run", 9999, "model", "ols")

    def test_an_unknown_entity_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.log_tag("dataset", 1, "model", "ols")


class TestBest(TrackerTestCase):
    def setUp(self) -> None:
        super().setUp()
        for name, value in (("blend", 0.52), ("naive_last", 1.33), ("swe_head", 0.51)):
            run_id = self.a_run(name=name)
            self.tracker.log_metric(run_id, "mae", value, dims={"h": 6})

    def test_best_minimizes_by_default(self) -> None:
        self.assertEqual(
            "swe_head",
            self.tracker.best(self.experiment_id, "mae", dims={"h": 6})["name"],
        )

    def test_maximize_is_opt_in(self) -> None:
        found = self.tracker.best(self.experiment_id, "mae", dims={"h": 6}, maximize=True)
        self.assertEqual("naive_last", found["name"])

    def test_best_reports_the_value_it_chose_on(self) -> None:
        found = self.tracker.best(self.experiment_id, "mae", dims={"h": 6})
        self.assertEqual(0.51, found["value"])
        self.assertEqual("mae", found["metric"])

    def test_best_on_an_absent_metric_is_none(self) -> None:
        self.assertIsNone(self.tracker.best(self.experiment_id, "crps"))

    def test_best_can_filter_by_tag(self) -> None:
        tagged = self.a_run(name="tagged", tags={"family": "blend"})
        self.tracker.log_metric(tagged, "mae", 0.9, dims={"h": 6})
        found = self.tracker.best(
            self.experiment_id, "mae", dims={"h": 6}, tags={"family": "blend"}
        )
        self.assertEqual("tagged", found["name"])


class TestCompare(TrackerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.first = self.a_run(name="blend")
        self.second = self.a_run(name="blend3_swe")
        self.tracker.log_metric(self.first, "mae", 0.521, dims={"h": 6})
        self.tracker.log_metric(self.second, "mae", 0.510, dims={"h": 6})

    def test_compare_puts_a_column_per_run(self) -> None:
        row = self.tracker.compare([self.first, self.second])[0]
        self.assertEqual(0.521, row[str(self.first)])
        self.assertEqual(0.510, row[str(self.second)])

    def test_compare_reports_the_delta_for_two_runs(self) -> None:
        row = self.tracker.compare([self.first, self.second])[0]
        self.assertAlmostEqual(-0.011, row["delta"])

    def test_a_metric_missing_from_one_run_gives_no_delta(self) -> None:
        self.tracker.log_metric(self.first, "crps", 0.19, dims={"h": 6})
        rows = {r["metric"]: r for r in self.tracker.compare([self.first, self.second])}
        self.assertIsNone(rows["crps"]["delta"])

    def test_compare_can_restrict_to_named_metrics(self) -> None:
        self.tracker.log_metric(self.first, "crps", 0.19, dims={"h": 6})
        rows = self.tracker.compare([self.first, self.second], metrics=["mae"])
        self.assertEqual({"mae"}, {r["metric"] for r in rows})

    def test_compare_needs_two_runs(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.compare([self.first])


class TestAudit(TrackerTestCase):
    def test_a_stored_metric_is_recomputed_from_its_rows(self) -> None:
        run_id = self.a_run()
        predictions, actuals = [1.0, 2.0, 3.0], [1.2, 2.1, 2.7]
        self.tracker.log_predictions(run_id, predictions, actuals, dims={"h": 6})
        self.tracker.log_metric(run_id, "mae", scoring.mae(predictions, actuals), dims={"h": 6})
        result = self.tracker.audit(run_id, "mae", dims={"h": 6})
        self.assertTrue(result["agrees"])
        self.assertEqual(3, result["rows"])

    def test_a_wrong_stored_metric_is_caught(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(run_id, [1.0, 2.0], [1.0, 2.0], dims={"h": 6})
        self.tracker.log_metric(run_id, "mae", 9.9, dims={"h": 6})
        result = self.tracker.audit(run_id, "mae", dims={"h": 6})
        self.assertFalse(result["agrees"])
        self.assertEqual(9.9, result["stored"])
        self.assertEqual(0.0, result["recomputed"])

    def test_audit_pools_the_rows_the_dims_select(self) -> None:
        run_id = self.a_run()
        self.tracker.log_predictions(
            run_id,
            [1.0, 5.0],
            [1.0, 6.0],
            dims=[{"cutoff": "a", "h": 6}, {"cutoff": "b", "h": 6}],
        )
        result = self.tracker.audit(run_id, "mae", dims={"h": 6})
        self.assertEqual(2, result["rows"])
        self.assertAlmostEqual(0.5, result["recomputed"])

    def test_audit_without_rows_is_none(self) -> None:
        run_id = self.a_run()
        self.assertIsNone(self.tracker.audit(run_id, "mae"))

    def test_audit_of_an_unscoreable_metric_raises(self) -> None:
        run_id = self.a_run()
        with self.assertRaises(ValueError):
            self.tracker.audit(run_id, "cov90")


class TestArtifacts(TrackerTestCase):
    def test_a_run_artifact_belongs_to_its_experiment_too(self) -> None:
        run_id = self.a_run()
        self.tracker.log_artifact(b"rows", "cv", "cv.parquet", run=run_id)
        by_experiment = self.tracker.artifacts(experiment=self.experiment_id)
        self.assertEqual(1, len(by_experiment))
        self.assertEqual(run_id, by_experiment[0]["run_id"])

    def test_an_experiment_scoped_artifact_needs_no_run(self) -> None:
        artifact_id = self.tracker.log_artifact(
            b"rows", "cv", "cv.parquet", experiment=self.experiment_id
        )
        self.assertIsNone(self.tracker.artifacts(experiment=self.experiment_id)[0]["run_id"])
        self.assertEqual(b"rows", self.tracker.artifact_data(artifact_id))

    def test_an_artifact_needs_an_owner(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.log_artifact(b"x", "cv", "cv.parquet")

    def test_listing_reports_size_without_the_blob(self) -> None:
        self.tracker.log_artifact(b"12345", "cv", "cv.parquet", experiment=self.experiment_id)
        row = self.tracker.artifacts(experiment=self.experiment_id)[0]
        self.assertEqual(5, row["bytes"])
        self.assertNotIn("data", row)

    def test_log_file_reads_from_disk_and_names_the_kind(self) -> None:
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "cv_results.parquet")
            with open(path, "wb") as stream:
                stream.write(b"rows")
            run_id = self.a_run()
            artifact_id = self.tracker.log_file(path, run=run_id)
            row = self.tracker.artifacts(run=run_id)[0]
            self.assertEqual("cv_results.parquet", row["filename"])
            self.assertEqual("parquet", row["kind"])
            self.assertEqual(b"rows", self.tracker.artifact_data(artifact_id))
        finally:
            shutil.rmtree(directory)

    def test_unknown_artifact_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.artifact_data(9999)


class TestDeletion(TrackerTestCase):
    def test_deleting_an_experiment_with_runs_succeeds(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 1.0, dims={"h": 6})
        self.tracker.log_predictions(run_id, [1.0], [1.0], dims={"h": 6})
        self.tracker.log_artifact(b"x", "cv", "cv.parquet", run=run_id)
        self.tracker.log_tags("run", run_id, {"model": "ols"})
        self.tracker.delete_experiment(self.experiment_id)
        self.assertIsNone(self.tracker.get_experiment(self.experiment_id))

    def test_deletion_cascades_to_every_child(self) -> None:
        run_id = self.a_run()
        self.tracker.log_metric(run_id, "mae", 1.0)
        self.tracker.log_predictions(run_id, [1.0])
        self.tracker.log_artifact(b"x", "cv", "cv.parquet", run=run_id)
        self.tracker.delete_experiment(self.experiment_id)
        self.assertEqual([], self.tracker.runs())
        self.assertEqual([], self.tracker.metrics())
        self.assertEqual([], self.tracker.predictions())
        self.assertEqual([], self.tracker.artifacts())

    def test_deletion_removes_experiment_tags(self) -> None:
        self.tracker.log_tags("experiment", self.experiment_id, {"train_start": "1960"})
        self.tracker.delete_experiment(self.experiment_id)
        self.assertEqual({}, self.tracker.tags("experiment", self.experiment_id))

    def test_deleting_an_unknown_experiment_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.delete_experiment(9999)


class TestSnapshot(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(":memory:")
        self.experiment_id = self.tracker.experiment("GSL_CV", "walk-forward")
        self.tracker.log_tags("experiment", self.experiment_id, {"train_start": "1960-01-01"})
        with self.tracker.run(self.experiment_id, name="blend", params={"alpha": "gcv"}) as run:
            run.log_metrics({"mae": 0.521}, dims={"h": 6})
            run.log_metrics({"mae": 1.999}, dims={"h": 24})
            run.log_predictions([1.0, 2.0], [1.1, 2.1], dims={"h": 6})
            run.set_note("keep")

    def tearDown(self) -> None:
        self.tracker.close()
        shutil.rmtree(self.directory)

    def out(self, name):
        with open(os.path.join(self.directory, name)) as stream:
            return stream.read()

    def test_summary_files_are_written(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory)
        self.assertEqual(
            {"experiment.json", "runs.csv", "metrics.csv"},
            set(os.listdir(self.directory)),
        )

    def test_predictions_are_omitted_by_default(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory)
        self.assertNotIn("predictions.csv", os.listdir(self.directory))

    def test_predictions_can_be_asked_for(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory, predictions=True)
        self.assertIn("predictions.csv", os.listdir(self.directory))
        self.assertIn("1.1", self.out("predictions.csv"))

    def test_the_commit_travels_with_the_numbers(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory)
        record = json.loads(self.out("experiment.json"))
        self.assertIn("git_commit", record)
        self.assertEqual({"train_start": "1960-01-01"}, record["tags"])

    def test_metrics_carry_their_dims(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory)
        import csv

        with open(os.path.join(self.directory, "metrics.csv")) as stream:
            rows = list(csv.DictReader(stream))
        at_six = [r for r in rows if r["dims"] == '{"h":6}']
        self.assertEqual(1, len(at_six))
        self.assertEqual("0.521", at_six[0]["value"])

    def test_the_note_reaches_the_committed_record(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory)
        self.assertIn("keep", self.out("runs.csv"))

    def test_two_snapshots_of_one_experiment_are_byte_identical(self) -> None:
        self.tracker.snapshot(self.experiment_id, self.directory)
        first = [self.out(n) for n in ("experiment.json", "runs.csv", "metrics.csv")]
        self.tracker.snapshot(self.experiment_id, self.directory)
        second = [self.out(n) for n in ("experiment.json", "runs.csv", "metrics.csv")]
        self.assertEqual(first, second, "a snapshot must be stable to commit")

    def test_snapshot_of_an_unknown_experiment_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.snapshot(9999, self.directory)


class TestDimsKey(unittest.TestCase):
    def test_key_order_does_not_change_the_key(self) -> None:
        self.assertEqual(dims_key({"a": 1, "b": 2}), dims_key({"b": 2, "a": 1}))

    def test_empty_dims_are_an_empty_object(self) -> None:
        self.assertEqual("{}", dims_key(None))
        self.assertEqual("{}", dims_key({}))


class TestConnection(unittest.TestCase):
    def test_the_tracker_closes_its_connection(self) -> None:
        tracker = ExperimentTracker(":memory:")
        tracker.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            tracker.conn.execute("SELECT 1")

    def test_used_as_a_context_manager(self) -> None:
        with ExperimentTracker(":memory:") as tracker:
            tracker.experiment("x")
        with self.assertRaises(sqlite3.ProgrammingError):
            tracker.conn.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
