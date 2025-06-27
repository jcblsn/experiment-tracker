#!/usr/bin/env python3
import os
import shutil
import tempfile

from src.experiment_tracker.experiment_tracker import ExperimentTracker


def main() -> None:
    tracker = ExperimentTracker("experiments.db")
    print("Initialized experiment tracker with database: experiments.db")

    exp_id = tracker.create_experiment(
        "Main Experiment", "Demonstration of experiment tracker integration"
    )
    tracker.add_tag("experiment", exp_id, "demo", "true")
    tracker.add_tag("experiment", exp_id, "purpose", "demonstration")
    print(f"Created experiment with ID: {exp_id}")

    run_id = tracker.start_run(exp_id)
    tracker.add_tag("run", run_id, "demo", "true")
    print(
        f"\nStarting a new run with run ID {run_id} for experiment {exp_id} with tags: {tracker.get_tags('run', run_id)}"
    )

    model_params = {"model_type": "ExampleModel", "learning_rate": 0.01, "epochs": 10}
    tracker.log_model(run_id, "ExampleModel", model_params)
    print(f"\nLogged model 'ExampleModel' with parameters: {model_params}")

    def max_error(preds, actuals):
        return max(abs(p - a) for p, a in zip(preds, actuals))

    def median_error(preds, actuals):
        errors = sorted(abs(p - a) for p, a in zip(preds, actuals))
        mid = len(errors) // 2
        return (
            errors[mid] if len(errors) % 2 == 1 else (errors[mid - 1] + errors[mid]) / 2
        )

    custom_metrics = {"max_error": max_error, "median_error": median_error}
    print(f"\nDefined custom metrics: {list(custom_metrics.keys())}")

    preds = [0.125, 0.372, 0.933]
    actuals = [0.15, 0.25, 0.35]
    indices = [
        1577836800,
        1577923200,
        1578009600,
    ]  # timestamps as integers, for example
    tracker.log_predictions(
        run_id, preds, actuals, index=indices, custom_metrics=custom_metrics
    )
    print(f"\nLogged prediction values: {tracker.get_predictions(run_id)}")
    print(f"Metrics for run {run_id}: {tracker.get_metrics(run_id)}")

    tracker.end_run(run_id)
    print(
        f"\nEnded run {run_id} with status: {tracker.get_run_history(exp_id)[0]['run_status']}"
    )

    tracker.add_tag("experiment", exp_id, "project", "demo")
    tracker.add_tag("experiment", exp_id, "version", "1.0")
    tracker.add_tag("run", run_id, "model_type", "simple")
    tracker.add_tag("run", run_id, "dataset", "synthetic")

    print("\nChecking experiment and run tags:")
    print(f" Experiment tags: {tracker.get_tags('experiment', exp_id)}")
    print(f" Run tags: {tracker.get_tags('run', run_id)}")

    print("\nListing recent experiments:")
    experiments = tracker.list_experiments(limit=5)
    for exp in experiments:
        print(
            f" {exp['experiment_id']}: {exp['experiment_name']} ({exp['created_time']})"
        )

    print("\nSearching for experiments containing 'Main':")
    found_experiments = tracker.find_experiments("Main")
    for exp in found_experiments[:5]:
        print(
            f" {exp['experiment_id']}: {exp['experiment_name']} ({exp['created_time']})"
        )
    if len(found_experiments) > 5:
        print(f" ...\n and {len(found_experiments) - 5} other(s)")

    run_id2 = tracker.start_run(exp_id)
    print(f"\nStarted second run with ID: {run_id2}")

    tracker.log_model(run_id2, "ExampleModel", model_params)

    preds2 = [0.264, 0.394, 0.876]
    tracker.log_predictions(
        run_id2, preds2, actuals, index=indices, custom_metrics=custom_metrics
    )
    print(f"Logged predictions for run {run_id2}")
    print(f"Metrics for run {run_id2}: {tracker.get_metrics(run_id2)}")

    tracker.end_run(run_id2)
    print(f"Ended run {run_id2}")

    export_dir = tempfile.mkdtemp()
    print(f"\nCreated temporary directory for export: {export_dir}")

    try:
        exp_export_dir = tracker.export_experiment(exp_id, export_dir)
        print(f"\nExported experiment to: {exp_export_dir}")

        new_db_path = "imported_experiments.db"
        if os.path.exists(new_db_path):
            os.remove(new_db_path)
            print(f"Removed existing database: {new_db_path}")

        new_tracker = ExperimentTracker(new_db_path)
        print(f"Created new tracker with database: {new_db_path}")

        new_exp_id = new_tracker.import_experiment(exp_export_dir)
        print(f"\nImported experiment with ID: {new_exp_id}")

        metrics = new_tracker.get_metrics(
            new_tracker.get_run_history(new_exp_id)[0]["run_id"]
        )
        print(f"\nMetrics from imported run: {metrics}")

        print("\nFinding best model across all runs:")
        preferred_metric = "rmse"
        best_model = tracker.get_best_model(exp_id, metric=preferred_metric)
        if best_model:
            print(f" Best model by {preferred_metric}: {best_model['model_name']}")
            print(f" Parameters: {best_model['parameters']}")
            print(
                f" Average {best_model['metric_name']}: {best_model['average_metric']}"
            )
            print(f" Based on {best_model['num_runs']} runs")
            print(f" Run IDs: {best_model['run_ids']}")

        new_tracker.conn.close()
        tracker.conn.close()
    finally:
        shutil.rmtree(export_dir)
        print(f"\nCleaned up temporary directory: {export_dir}")


if __name__ == "__main__":
    main()
