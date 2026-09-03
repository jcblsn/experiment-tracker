#!/usr/bin/env python3
"""A worked example: log a small sweep, then read it back."""

from experiment_tracker import ExperimentTracker, scoring


def main() -> None:
    with ExperimentTracker(":memory:") as tracker:
        experiment = tracker.experiment(
            "Model comparison",
            "Linear against polynomial regression on synthetic data",
            tags={"dataset": "synthetic"},
        )

        actuals = [1.0, 2.0, 3.0, 4.0]
        models = {
            "linear": ([1.2, 2.1, 3.0, 4.2], {"degree": 1}),
            "quadratic": ([1.05, 1.98, 3.02, 3.95], {"degree": 2}),
            "cubic": ([0.99, 2.01, 2.98, 4.02], {"degree": 3}),
        }

        for name, (predictions, params) in models.items():
            with tracker.run(experiment, name=name, params=params) as run:
                # Scoring is a separate step, so nothing is written that you did not ask for.
                run.log_metrics(scoring.score(predictions, actuals))
                run.log_predictions(predictions, actuals, dims={"split": "test"})
                run.set_note(f"degree {params['degree']}")

        print("Runs, worst to best by RMSE:")
        for row in sorted(
            tracker.metrics(experiment=experiment, metric="rmse"),
            key=lambda r: -r["value"],
        ):
            print(f"  {row['run_name']:<10} rmse={row['value']:.4f}")

        best = tracker.best(experiment, "rmse")
        print(f"\nBest: {best['name']} at rmse={best['value']:.4f}")

        linear = tracker.runs(name="linear")[0]["run_id"]
        cubic = tracker.runs(name="cubic")[0]["run_id"]
        print("\nlinear against cubic:")
        for row in tracker.compare([linear, cubic], metrics=["rmse", "mae"]):
            print(
                f"  {row['metric']:<5} {row[str(linear)]:.4f} -> {row[str(cubic)]:.4f}"
                f"  delta {row['delta']:+.4f}"
            )

        print("\nThe stored metric recomputed from its own prediction rows:")
        print(" ", tracker.audit(linear, "rmse"))


if __name__ == "__main__":
    main()
