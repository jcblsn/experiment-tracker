# experiment-tracker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/jcblsn/experiment-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/jcblsn/experiment-tracker/actions/workflows/ci.yml)

## Overview

A very lightweight alternative to MLflow's experiment tracking.

Last year, I was using MLflow for a complex modelling project at work and was surprised to find it slow to a halt after only a few hundred runs. This repo was inspired by my subsequent research about alternatives, especially Eduardo Blancas' [Who needs MLflow when you have SQLite?](https://ploomber.io/blog/experiment-tracking/).

A SQLite file, direct SQL when you want it, no server, and nothing outside the standard library.

## Installation

Requires Python 3.11 or later.

```bash
uv pip install "git+https://github.com/jcblsn/experiment-tracker@v2.0.0"
```

## Usage

```python
from experiment_tracker import ExperimentTracker, scoring

with ExperimentTracker("experiments.db") as tracker:
    experiment = tracker.experiment("Comparing models", "for demonstration")

    with tracker.run(experiment, name="ols", params={"intercept": True}) as run:
        run.log_metrics(scoring.score(predictions, actuals))
        run.log_predictions(predictions, actuals, dims={"split": "test"})
        run.set_note("keep: beats the baseline at every lead")

    tracker.best(experiment, "rmse")
    tracker.compare([1, 2], metrics=["rmse"])
    tracker.snapshot(experiment, "results/")
```

### Metrics carry their dimensions

A metric usually has an axis: a forecast lead, a fold, a class. Encoding that axis in the
name gives you `mae_h1` through `mae_h24` and a namespace nothing can group over. Pass
`dims` instead:

```python
for h in range(1, 25):
    run.log_metrics({"mae": mae_at(h), "rmse": rmse_at(h)}, dims={"h": h})

tracker.metrics(experiment=experiment, metric="mae", dims={"h": 6})
```

`dims` is a JSON object stored beside the value, canonically serialized, so
`{"h": 6, "fold": 2}` and `{"fold": 2, "h": 6}` are the same key. Filters use subset
matching: `dims={"h": 6}` selects every row at lead 6 whatever else keys it.

### Scoring is explicit

`log_predictions` stores rows. It does not compute anything. Use `scoring.score`,
`scoring.rmse`, `scoring.mae`, or `scoring.mape` and pass the result to `log_metrics`.
Earlier versions computed metrics as a side effect of logging predictions, which wrote a
value pooled over every dimension under a name indistinguishable from the per-dimension
ones.

### Predictions are for auditing

Give prediction rows the same `dims` as the metrics they support, and a stored number can
be checked rather than merely trusted:

```bash
expt audit 42 --metric mae --dim h=6
```

That recomputes `mae` from the rows at lead 6 and reports whether it matches what was
stored. It exits non-zero when it does not.

## CLI

`expt` finds the database automatically when the working directory holds exactly one
`.db` file. Otherwise pass `--db PATH` or set `EXPT_DB`.

```bash
expt schema                       # tables and columns, so you need not guess
expt list                         # experiments, newest first
expt show [<id|name>]             # one experiment, with provenance and tags
expt runs [<id|name>] -t k=v      # runs, filtered by any number of tags
expt metrics [<run>] --pivot h    # metrics, one column per value of a dim
expt best -m mae -d h=6           # lowest wins; --maximize to invert
expt diff 12 13 -m mae            # metric deltas between runs
expt log [<id|name>] -m mae       # one line per run, for a ledger
expt audit 42 -m mae -d h=6       # recompute a metric from its rows
expt snapshot <id> results/       # committable files
expt artifacts [--get <id>]       # list or fetch stored files
expt rm <id> --yes                # delete an experiment and its runs
expt sql "SELECT ..."             # the escape hatch
```

## Snapshots

The database is a working file you probably gitignore, so it cannot be the citation for a
published number. `snapshot` writes files that can be:

- `experiment.json`: description, note, tags, and the captured git commit, dirty flag,
  command line, and Python version.
- `runs.csv`: one row per run with its params, tags, note, and status.
- `metrics.csv`: long format, one row per metric and dims.
- `predictions.csv`: only with `predictions=True`, since row-level output is usually large.

Rows are sorted and columns are fixed, so re-running a snapshot of an unchanged experiment
produces identical bytes and an unchanged diff.

Because predictions are omitted by default, `expt audit` works against the database rather
than against a snapshot. If you need a committed record to be auditable on its own, pass
`--predictions`.

## Schema

- `experiments`: experiment_id, name, description, note, created_at, git_commit, git_dirty, argv, python
- `runs`: run_id, experiment_id, name, params, status, note, started_at, ended_at, error
- `metrics`: run_id, metric, dims, value
- `predictions`: prediction_id, run_id, dims, prediction, actual
- `tags`: entity_type, entity_id, name, value
- `artifacts`: artifact_id, experiment_id, run_id, kind, filename, data, created_at

`status` is one of `running`, `completed`, `failed`. A run holds its own name and params;
there is no separate models table.

## Testing

```bash
python -m unittest discover tests
```

No test dependencies. `unittest` ships with Python, which is the point.
