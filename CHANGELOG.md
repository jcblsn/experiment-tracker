# Changelog

## 2.0.0

A breaking rework, aimed at reading. The previous version was written to far more often
than it was read, and the reasons were all on the read side: comparisons scoped to a
single experiment, a CLI that needed a flag on every invocation, column names that could
not be guessed, and filters that returned wrong answers without saying so.

### Fixed

- `find_runs` returned an empty list for any filter of two or more tags. Conditions were
  ANDed against one joined alias, so no row could satisfy two of them. Tag filtering now
  uses one EXISTS per tag and is exercised for two and three tags.
- A duplicate tag silently dropped its entity from tag filters. Tags are now unique per
  entity and name, and re-tagging updates the value.
- `aggregate` and `best_run` interpolated values, metric names, and tag names into SQL
  with f-strings, so an apostrophe in a tag value raised `OperationalError`. Every query
  is parameterized.
- `delete_experiment` raised `IntegrityError` for any experiment that had a run, which is
  every real one. Foreign keys now cascade.
- `get_metrics` and `get_predictions` raised `ValueError` on the empty case, which is a
  normal case, and a private `_get_metrics_safe` existed to work around it. Reads return
  empty results.
- The command line `best` defaulted to maximizing while the library `best_run` defaulted
  to minimizing, so the default answer for an error metric was the worst run. Both now
  minimize, and `--maximize` inverts.
- Only one index existed in the whole database. Every foreign key is indexed.
- Values were rounded on the way in by `smart_round`. Measurements are stored as given and
  rounded only for display.

### Changed

- Metrics and predictions carry a `dims` JSON object, so a lead, fold, or class is data
  rather than part of the metric name. Canonically serialized, so key order does not
  create duplicates. Filters use subset matching.
- Logging predictions no longer computes metrics. Use the new `scoring` module and pass
  the result to `log_metrics`.
- The `models` table is gone. A run carries its own `name` and `params`.
- Columns lost their table-name prefixes: `experiment_name` is `name`, `created_time` is
  `created_at`, `run_status` is `status`, `run_start_time` is `started_at`. `status`
  values are lowercase.
- `experiment()` records the git commit, dirty flag, command line, and Python version.
  `get_or_create=True` supports one long-lived named benchmark.
- Reads span experiments by default: `runs`, `metrics`, and `predictions` take
  `experiment=None`.
- Artifacts can belong to an experiment as well as to a run.
- `ExperimentTracker` closes its connection and works as a context manager.
- Runs and experiments carry a `note`, for the reasoning behind a decision.

### Added

- `scoring`: `rmse`, `mae`, `mape`, and `score`.
- `latest_experiment()`, replacing a hand-written `max(experiment_id)` query.
- `compare()` and `expt diff`, reporting metric deltas between runs.
- `audit()` and `expt audit`, recomputing a stored metric from the prediction rows that
  support it.
- `snapshot()` and `expt snapshot`, writing deterministic files meant to be committed.
- `expt schema`, `expt log`, `expt rm`, `expt artifacts`.
- `expt` finds a single `.db` in the working directory without a flag.
- `py.typed`.

### Removed

`create_experiment`, `log_model`, `get_model`, `get_run_history`, `get_all_runs`,
`find_runs`, `get_tagged_entities`, `latest_runs`, `aggregate`, `best_run`,
`export_experiment`, `import_experiment`, and `smart_round` on the write path.

### Migration

The schema is not compatible. Start a new database and keep the old file. `snapshot` on
the old version has no equivalent, so export anything you need first with `expt sql`.

## 1.0.0

Initial release.
