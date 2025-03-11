#!/usr/bin/env python3
import os
import shutil
import tempfile

from src.experiment_tracker import ExperimentTracker


def main() -> None:
    tracker = ExperimentTracker("experiments.db")

    exp_id = tracker.create_experiment(
        "Main Experiment", "Demonstration of experiment tracker integration"
    )
    print(f"Created experiment with ID: {exp_id}")

    run_id = tracker.start_run(exp_id)

    model_params = {"model_type": "ExampleModel", "learning_rate": 0.01, "epochs": 10}
    tracker.log_model(run_id, "ExampleModel", model_params)

    def max_error(preds, actuals):
        return max(abs(p - a) for p, a in zip(preds, actuals))

    def median_error(preds, actuals):
        errors = sorted(abs(p - a) for p, a in zip(preds, actuals))
        mid = len(errors) // 2
        return (
            errors[mid] if len(errors) % 2 == 1 else (errors[mid - 1] + errors[mid]) / 2
        )

    custom_metrics = {"max_error": max_error, "median_error": median_error}

    preds = [0.1, 0.2, 0.3]
    actuals = [0.15, 0.25, 0.35]
    tracker.log_predictions(run_id, preds, actuals, custom_metrics=custom_metrics)

    tracker.add_tag("experiment", exp_id, "project", "demo")
    tracker.add_tag("experiment", exp_id, "version", "1.0")
    tracker.add_tag("run", run_id, "model_type", "simple")
    tracker.add_tag("run", run_id, "dataset", "synthetic")
    print("Added tags to experiment and run")

    exp_tags = tracker.get_tags("experiment", exp_id)
    run_tags = tracker.get_tags("run", run_id)
    print(f"Experiment tags: {exp_tags}")
    print(f"Run tags: {run_tags}")

    tracker.end_run(run_id)

    run_id2 = tracker.start_run(exp_id)
    model_params2 = {"model_type": "ExampleModel", "learning_rate": 0.05, "epochs": 5}
    tracker.log_model(run_id2, "ExampleModel", model_params2)

    preds2 = [0.12, 0.22, 0.32]
    actuals2 = [0.15, 0.25, 0.35]
    tracker.log_predictions(run_id2, preds2, actuals2)
    tracker.end_run(run_id2)
    print("Completed two experiment runs")

    export_dir = tempfile.mkdtemp()
    try:
        exp_export_dir = tracker.export_experiment(exp_id, export_dir)
        print(f"Exported experiment to: {exp_export_dir}")

        new_db_path = "imported_experiments.db"
        if os.path.exists(new_db_path):
            os.remove(new_db_path)

        new_tracker = ExperimentTracker(new_db_path)
        new_exp_id = new_tracker.import_experiment(exp_export_dir)
        print(f"Imported experiment with ID: {new_exp_id}")

        metrics = new_tracker.get_metrics(
            new_tracker.get_run_history(new_exp_id)[0]["id"]
        )
        print(f"Metrics from imported run: {metrics}")

        new_tracker.conn.close()
        tracker.conn.close()
    finally:
        shutil.rmtree(export_dir)


if __name__ == "__main__":
    main()
