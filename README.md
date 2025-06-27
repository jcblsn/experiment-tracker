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

```python
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("experiments.db")

# Create experiment, run
exp_id = tracker.create_experiment("Model Comparison", "Testing different algorithms")

run_id = tracker.start_run(exp_id)

# Log results
tracker.log_model(run_id, "OLS", params={"intercept": True, "transform_response": "log"})
tracker.log_predictions(run_id, preds=[0.8, 0.9, 0.7], actuals=[0.85, 0.88, 0.72])
tracker.log_metric(run_id, "SSE", 0.0033)

# Add tags
tracker.add_tag("run", run_id, tag_name="dataset", tag_value="train")

tracker.end_run(run_id)

# Find best performing model
best_model = tracker.get_best_model(exp_id, "rmse")
```

## Schema

- **experiments**: experiment_id, experiment_name, experiment_description, created_time
- **runs**: run_id, experiment_id, run_status, run_start_time, run_end_time, error
- **models**: model_id, run_id, model_name, parameters
- **predictions**: prediction_id, run_id, idx, prediction, actual
- **metrics**: run_id, metric, value
- **tags**: tag_id, entity_type, entity_id, name, value

## Testing

 ```bash
  # Run all tests
  python -m unittest discover tests
```