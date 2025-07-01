# experiment-tracker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/jcblsn/experiment-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/jcblsn/experiment-tracker/actions/workflows/ci.yml)

## Overview

A very lightweight alternative to MLFlow's experiment tracking capabilities.

Last year, I was using MLFlow for a complex modeling project at work and was surprised to find it slow to a halt after only a few hundred runs. This repo was inspired by my subsequent research about alternatives, especially Eduardo Blancas' [Who needs MLflow when you have SQLite?](https://ploomber.io/blog/experiment-tracking/).

The system uses a SQLite backend for direct SQL queries, operates locally without a server, and has no dependencies outside the standard library. This makes it desirable for solo projects where simplicity and speed are desired.

## Installation

Requires Python 3.12 or later.

```bash
uv pip install "git+https://github.com/jcblsn/experiment-tracker"
```

For more about `uv` see [here](https://docs.astral.sh/uv/).

## Usage

Example usage (for illustration only):

```python
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("experiments.db")

preds = [...] # your model predictions
actuals = [...] # actual values
model_bytes = b"" # serialized model bytes

# Create experiment
exp_id = tracker.create_experiment(experiment_name="Comparing models", experiment_description="For demonstration purposes")

# Run with context manager
with tracker.run(exp_id, tags={"model": "ols", "dataset": "train"}) as run:
    run.log_model(model_name="OLS", parameters={"intercept": True, "normalize": False})
    run.log_predictions(predictions=preds, actual_values=actuals, metrics=["rmse", "mae"])
    run.log_artifact(data=model_bytes, artifact_type="model", filename="ols_model.pkl")

# Query runs by tags
run_ids = tracker.find_runs({"model": "ols"}, exp_id)

# Aggregate metrics across runs
results = tracker.aggregate(exp_id, "rmse", group_by=["model"])
```

## Schema

- experiments: experiment_id, experiment_name, experiment_description, created_time
- runs: run_id, experiment_id, run_status, run_start_time, run_end_time, error
- models: model_id, run_id, model_name, parameters
- predictions: prediction_id, run_id, idx, prediction, actual
- metrics: run_id, metric, metric_value
- tags: tag_id, entity_type, entity_id, tag_name, tag_value
- artifacts: artifact_id, run_id, artifact_type, filename, data, created_time

## Testing

 ```bash
  # Run all tests
  python -m unittest discover tests
```