import os
import shutil
import tempfile
import pickle


from src.experiment_tracker.experiment_tracker import ExperimentTracker


def main() -> None:
    tracker = ExperimentTracker("experiments.db")
    print("Initialized experiment tracker with database: experiments.db")

    exp_id = tracker.create_experiment(
        experiment_name="Main Experiment",
        experiment_description="Demonstration of experiment tracker integration",
    )
    tracker.log_tag("experiment", exp_id, "demo", "true")
    tracker.log_tag("experiment", exp_id, "purpose", "demonstration")
    print(f"Created experiment with ID: {exp_id}")

    with tracker.run(exp_id, tags={"model": "ols", "dataset": "demo"}) as run:
        model_params = {"intercept": True, "normalize": False}
        run.log_model(model_name="OLS", parameters=model_params)
        print(f"\nLogged model 'OLS' with parameters: {model_params}")

        def max_error(preds, actuals):
            return max(abs(p - a) for p, a in zip(preds, actuals))

        preds = [0.125, 0.372, 0.933]
        actuals = [0.15, 0.25, 0.35]
        run.log_predictions(
            predictions=preds,
            actual_values=actuals,
            custom_metrics={"max_error": max_error},
        )

        model_data = {"coefficients": [1.2, -0.5, 2.1], "intercept": 0.1}
        artifact_id = run.log_artifact(
            data=pickle.dumps(model_data),
            artifact_type="model",
            filename="ols_model.pkl",
        )
        print(f"Logged artifact with ID: {artifact_id}")

    print("Run completed")

    found_runs = tracker.find_runs({"model": "ols"}, exp_id)
    print(f"Found runs: {found_runs}")

    run_id = found_runs[0]
    print(f"Metrics for run {run_id}: {tracker.get_metrics(run_id)}")

    tracker.log_tag("experiment", exp_id, "project", "demo")
    tracker.log_tag("experiment", exp_id, "version", "1.0")
    tracker.log_tag("run", run_id, "model_type", "simple")
    tracker.log_tag("run", run_id, "dataset", "synthetic")

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

        new_tracker.conn.close()
        tracker.conn.close()
    finally:
        shutil.rmtree(export_dir)
        print(f"\nCleaned up temporary directory: {export_dir}")


if __name__ == "__main__":
    main()
