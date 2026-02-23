from src.experiment_tracker.experiment_tracker import ExperimentTracker


def main() -> None:
    tracker = ExperimentTracker(":memory:")

    exp_id = tracker.create_experiment(
        "Model Comparison",
        "Comparing linear vs polynomial regression on synthetic data",
    )
    print(f"Created experiment: {exp_id}")

    models = {
        "linear": {
            "params": {"degree": 1, "regularization": 0.01},
            "predictions": [1.2, 2.1, 3.0, 4.2],
            "actuals": [1.0, 2.0, 3.0, 4.0],
        },
        "quadratic": {
            "params": {"degree": 2, "regularization": 0.001},
            "predictions": [1.05, 1.98, 3.02, 3.95],
            "actuals": [1.0, 2.0, 3.0, 4.0],
        },
        "cubic": {
            "params": {"degree": 3, "regularization": 0.0001},
            "predictions": [0.99, 2.01, 2.98, 4.02],
            "actuals": [1.0, 2.0, 3.0, 4.0],
        },
    }

    print("\nRunning models...")
    for name, data in models.items():
        with tracker.run(exp_id, tags={"model_type": name}) as run:
            run.log_model(name, data["params"])
            run.log_predictions(data["predictions"], data["actuals"])
        print(f"  {name}: logged predictions and metrics")

    print("\n--- Results ---")

    linear_runs = tracker.find_runs({"model_type": "linear"}, exp_id)
    print(f"\nLinear model runs: {linear_runs}")

    metrics = tracker.get_metrics(linear_runs[0])
    print(f"Linear model metrics: {metrics}")

    print("\nModel comparison (RMSE):")
    results = tracker.aggregate(exp_id, "rmse", group_by=["model_type"])
    for row in results:
        print(f"  {row['model_type']}: {row['rmse_mean']:.4f}")

    best = tracker.best_run(exp_id, "rmse", minimize=True)
    print(f"\nBest model: {best['tags']['model_type']} (RMSE={best['metrics']['rmse']:.4f})")


if __name__ == "__main__":
    main()
