#!/usr/bin/env python3
from src.experiment_tracker import ExperimentTracker


def main() -> None:
    tracker = ExperimentTracker("experiments.db")

    exp_id = tracker.create_experiment(
        "Main Experiment", "Demonstration of experiment tracker integration"
    )
    print(f"Created experiment: {exp_id}")

    run_id = tracker.start_run(exp_id)
    print(f"Started run: {run_id}")

    model_params = {"model_type": "ExampleModel", "learning_rate": 0.01, "epochs": 10}
    tracker.log_model(run_id, "ExampleModel", model_params)
    print("Logged model")

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
    print("Logged predictions with custom metrics")

    tracker.end_run(run_id)
    print("Ended run")

    experiment = tracker.get_experiment(exp_id)
    run_history = tracker.get_run_history(exp_id)
    models = tracker.get_models(run_id)
    predictions = tracker.get_predictions(run_id)
    try:
        metrics = tracker.get_metrics(run_id)
    except ValueError:
        metrics = None

    print("Experiment details:", experiment)
    print("Run history:", run_history)
    print("Logged models:", models)
    print("Logged predictions:", predictions)
    print("Metrics:", metrics)

    tracker.conn.close()
    print("Closed database connection.")


if __name__ == "__main__":
    main()
