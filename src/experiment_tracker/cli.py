#!/usr/bin/env python3
import argparse
import csv
import glob
import io
import json
import os
import sqlite3
import sys
from typing import Any

from .experiment_tracker import ExperimentTracker, dims_key
from .scoring import SCORERS


def find_database(explicit_path: str | None = None) -> str:
    """Locate the database without making every command carry a flag.

    Looking only for ./experiments.db meant a project naming its file anything else had to
    pass --db every time, which is enough friction to stop the command being used at all.
    """
    if explicit_path:
        return explicit_path
    env_path = os.environ.get("EXPT_DB")
    if env_path:
        return env_path
    candidates = sorted(path for path in glob.glob("*.db") if not path.endswith(".legacy.db"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        sys.exit(
            "No database found. Pass --db PATH, set EXPT_DB, or run from a directory "
            "holding one .db file."
        )
    sys.exit("Several databases here: " + ", ".join(candidates) + ". Pass --db PATH to choose one.")


def show(value: float | None, places: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{places}g}"
    return str(value)


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "No results."
    if columns is None:
        columns = list(rows[0].keys())
    cells = [[show(row.get(c)) for c in columns] for row in rows]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in cells)) if cells else len(str(col))
        for i, col in enumerate(columns)
    ]
    lines = [
        "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in cells]
    return "\n".join(lines)


def emit(data: Any, fmt: str, columns: list[str] | None = None) -> None:
    if isinstance(data, dict):
        data = [data]
    if fmt == "json":
        print(json.dumps(data, indent=2, default=str))
        return
    if fmt == "csv":
        if not data:
            return
        columns = columns or list(data[0].keys())
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
        print(out.getvalue().strip())
        return
    print(format_table(data, columns))


def parse_pairs(values: list[str] | None) -> dict[str, str]:
    out = {}
    for item in values or []:
        key, _, value = item.partition("=")
        out[key] = value
    return out


def parse_dims(values: list[str] | None) -> dict[str, Any]:
    """Dim values are typed, so h=6 must filter as the integer 6, not the string."""
    out: dict[str, Any] = {}
    for key, value in parse_pairs(values).items():
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def opened(args) -> ExperimentTracker:
    return ExperimentTracker(find_database(args.db))


def resolve_experiment(tracker: ExperimentTracker, value: str | None) -> dict:
    """Accept an id, a name, or nothing meaning the newest."""
    if value is None:
        found = tracker.latest_experiment()
        if found is None:
            sys.exit("No experiments yet.")
        return found
    if str(value).isdigit():
        found = tracker.get_experiment(int(value))
        if found is None:
            sys.exit(f"No experiment {value}.")
        return found
    found = tracker.latest_experiment(str(value))
    if found is None:
        sys.exit(f"No experiment matching {value!r}.")
    return found


def cmd_schema(args) -> None:
    """Print the tables and columns, so nobody has to guess a column name again."""
    tracker = opened(args)
    for table in ("experiments", "runs", "metrics", "predictions", "tags", "artifacts"):
        columns = tracker.conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = ", ".join(f"{c['name']} {c['type'] or 'ANY'}" for c in columns)
        count = tracker.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        print(f"{table} ({count} rows)\n  {names}\n")


def cmd_list(args) -> None:
    tracker = opened(args)
    rows = tracker.experiments(name=args.search, limit=args.limit)
    emit(
        rows,
        args.format,
        ["experiment_id", "name", "runs", "created_at", "git_commit"],
    )


def cmd_show(args) -> None:
    tracker = opened(args)
    record = resolve_experiment(tracker, args.experiment)
    experiment_id = record["experiment_id"]
    record["tags"] = tracker.tags("experiment", experiment_id)
    runs = tracker.runs(experiment=experiment_id)
    record["runs"] = len(runs)
    if args.format == "json":
        emit(record, "json")
        return
    print(f"Experiment {experiment_id}: {record['name']}")
    if record["description"]:
        print(f"  {record['description']}")
    if record["note"]:
        print(f"  note: {record['note']}")
    commit = record["git_commit"] or "unknown"
    dirty = " (dirty tree)" if record["git_dirty"] else ""
    print(f"  created {record['created_at']}  commit {commit[:12]}{dirty}")
    if record["argv"]:
        print(f"  command {record['argv']}")
    if record["tags"]:
        print("  tags " + ", ".join(f"{k}={v}" for k, v in record["tags"].items()))
    statuses: dict[str, int] = {}
    for run in runs:
        statuses[run["status"]] = statuses.get(run["status"], 0) + 1
    print("  runs " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))


def cmd_runs(args) -> None:
    tracker = opened(args)
    experiment = None if args.all else resolve_experiment(tracker, args.experiment)
    rows = tracker.runs(
        experiment=None if experiment is None else experiment["experiment_id"],
        tags=parse_pairs(args.tag) or None,
        status=args.status,
        name=args.name,
    )
    for row in rows:
        row["tags"] = ", ".join(f"{k}={v}" for k, v in row["tags"].items())
    emit(
        rows,
        args.format,
        ["run_id", "experiment_id", "name", "status", "started_at", "tags", "note"],
    )


def cmd_metrics(args) -> None:
    tracker = opened(args)
    rows = tracker.metrics(runs=args.run, metric=args.metric, dims=parse_dims(args.dim) or None)
    if not rows:
        print("No results.")
        return
    if not args.pivot:
        for row in rows:
            row["dims"] = dims_key(row["dims"])
        emit(rows, args.format, ["run_id", "run_name", "metric", "dims", "value"])
        return
    axis = args.pivot
    keys = sorted({row["dims"].get(axis) for row in rows if axis in row["dims"]})
    table: dict[tuple, dict] = {}
    for row in rows:
        if axis not in row["dims"]:
            continue
        key = (row["run_name"] or row["run_id"], row["metric"])
        entry = table.setdefault(key, {"run": key[0], "metric": key[1]})
        entry[str(row["dims"][axis])] = row["value"]
    emit(
        [table[k] for k in sorted(table, key=lambda k: (str(k[1]), str(k[0])))],
        args.format,
        ["run", "metric"] + [str(k) for k in keys],
    )


def cmd_best(args) -> None:
    tracker = opened(args)
    experiment = None if args.all else resolve_experiment(tracker, args.experiment)
    found = tracker.best(
        None if experiment is None else experiment["experiment_id"],
        args.metric,
        dims=parse_dims(args.dim) or None,
        maximize=args.maximize,
        tags=parse_pairs(args.tag) or None,
    )
    if found is None:
        sys.exit(f"No runs carry metric {args.metric!r}.")
    if args.format == "json":
        emit(found, "json")
        return
    direction = "highest" if args.maximize else "lowest"
    print(f"{direction} {args.metric}: run {found['run_id']} {found['name']}")
    print(f"  {args.metric} = {found['value']}  dims {dims_key(found['dims'])}")
    if found["params"]:
        print(f"  params {json.dumps(found['params'], sort_keys=True)}")


def cmd_diff(args) -> None:
    tracker = opened(args)
    rows = tracker.compare(
        args.run_ids, metrics=args.metric or None, dims=parse_dims(args.dim) or None
    )
    if not rows:
        print("No shared metrics.")
        return
    names = {}
    for run_id in args.run_ids:
        run = tracker.get_run(run_id)
        names[str(run_id)] = f"{run['name'] or run_id}" if run else str(run_id)
    out = []
    for row in rows:
        entry = {"metric": row["metric"], "dims": dims_key(row["dims"])}
        for run_id in args.run_ids:
            entry[names[str(run_id)]] = row.get(str(run_id))
        if "delta" in row:
            entry["delta"] = row["delta"]
        out.append(entry)
    columns = ["metric", "dims"] + [names[str(r)] for r in args.run_ids]
    if len(args.run_ids) == 2:
        columns.append("delta")
    emit(out, args.format, columns)


def cmd_log(args) -> None:
    """One line per run: the ledger, generated instead of typed by hand."""
    tracker = opened(args)
    experiment = None if args.all else resolve_experiment(tracker, args.experiment)
    runs = tracker.runs(experiment=None if experiment is None else experiment["experiment_id"])
    wanted = args.metric or []
    dims = parse_dims(args.dim) or None
    rows = []
    for run in runs:
        entry = {
            "date": (run["started_at"] or "")[:10],
            "run_id": run["run_id"],
            "model": run["name"] or "",
            "status": run["status"],
        }
        for metric in wanted:
            found = tracker.metrics(runs=run["run_id"], metric=metric, dims=dims)
            entry[metric] = found[0]["value"] if found else None
        entry["note"] = run["note"] or (run["error"] or "")
        rows.append(entry)
    emit(rows, args.format, ["date", "run_id", "model", "status", *wanted, "note"])


def cmd_audit(args) -> None:
    tracker = opened(args)
    result = tracker.audit(args.run_id, args.metric, dims=parse_dims(args.dim) or None)
    if result is None:
        sys.exit("No prediction rows with actuals for those dims.")
    if args.format == "json":
        emit(result, "json")
        return
    print(f"run {result['run_id']}  {result['metric']}  dims {dims_key(result['dims'])}")
    print(f"  rows       {result['rows']}")
    print(f"  stored     {result['stored']}")
    print(f"  recomputed {result['recomputed']}")
    print(f"  agrees     {result['agrees']}")
    if not result["agrees"]:
        sys.exit(1)


def cmd_snapshot(args) -> None:
    tracker = opened(args)
    experiment = resolve_experiment(tracker, args.experiment)
    directory = tracker.snapshot(
        experiment["experiment_id"], args.directory, predictions=args.predictions
    )
    print(f"Wrote experiment {experiment['experiment_id']} to {directory}")


def cmd_artifacts(args) -> None:
    tracker = opened(args)
    if args.get is not None:
        data = tracker.artifact_data(args.get)
        target = args.output or next(
            (row["filename"] for row in tracker.artifacts() if row["artifact_id"] == args.get),
            f"artifact_{args.get}",
        )
        with open(target, "wb") as stream:
            stream.write(data)
        print(f"Wrote {len(data)} bytes to {target}")
        return
    experiment = None if args.all else resolve_experiment(tracker, args.experiment)
    rows = tracker.artifacts(experiment=None if experiment is None else experiment["experiment_id"])
    emit(
        rows,
        args.format,
        ["artifact_id", "run_id", "kind", "filename", "bytes", "created_at"],
    )


def cmd_rm(args) -> None:
    tracker = opened(args)
    experiment = resolve_experiment(tracker, args.experiment)
    if not args.yes:
        runs = len(tracker.runs(experiment=experiment["experiment_id"]))
        sys.exit(
            f"Would delete experiment {experiment['experiment_id']} "
            f"({experiment['name']}) and its {runs} runs. Pass --yes to confirm."
        )
    tracker.delete_experiment(experiment["experiment_id"])
    print(f"Deleted experiment {experiment['experiment_id']}")


def cmd_sql(args) -> None:
    conn = sqlite3.connect(find_database(args.db))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(args.query)
    except sqlite3.Error as error:
        sys.exit(f"SQL error: {error}")
    rows = [dict(row) for row in cursor.fetchall()]
    if not rows:
        print("No results.")
        return
    emit(rows, args.format, [d[0] for d in cursor.description])


def add_format(parser, choices=("table", "json", "csv")) -> None:
    parser.add_argument("--format", "-f", choices=list(choices), default="table")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expt", description="Query experiment tracking data")
    parser.add_argument(
        "--db",
        metavar="PATH",
        help="Database path (default: EXPT_DB, or the only .db file here)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("schema", help="Print tables and columns")
    p.set_defaults(func=cmd_schema)

    p = subparsers.add_parser("list", help="List experiments")
    p.add_argument("--limit", "-n", type=int, default=10)
    p.add_argument("--search", "-s", help="Filter by name")
    add_format(p)
    p.set_defaults(func=cmd_list)

    p = subparsers.add_parser("show", help="Show one experiment")
    p.add_argument("experiment", nargs="?", help="Id or name (default: the newest)")
    add_format(p, ("table", "json"))
    p.set_defaults(func=cmd_show)

    p = subparsers.add_parser("runs", help="List runs")
    p.add_argument("experiment", nargs="?", help="Id or name (default: the newest)")
    p.add_argument("--all", action="store_true", help="Every experiment")
    p.add_argument("--tag", "-t", action="append", help="Filter by tag (KEY=VALUE)")
    p.add_argument("--name", help="Filter by run name")
    p.add_argument("--status", choices=["running", "completed", "failed"])
    add_format(p)
    p.set_defaults(func=cmd_runs)

    p = subparsers.add_parser("metrics", help="Metrics, optionally pivoted on a dim")
    p.add_argument("run", type=int, nargs="?", help="Run id (default: every run)")
    p.add_argument("--metric", "-m", help="Filter by metric name")
    p.add_argument("--dim", "-d", action="append", help="Filter by dim (KEY=VALUE)")
    p.add_argument("--pivot", "-p", metavar="DIM", help="Spread this dim across columns")
    add_format(p)
    p.set_defaults(func=cmd_metrics)

    p = subparsers.add_parser("best", help="Best run by a metric")
    p.add_argument("experiment", nargs="?", help="Id or name (default: the newest)")
    p.add_argument("--metric", "-m", required=True)
    p.add_argument("--dim", "-d", action="append", help="Restrict to dims (KEY=VALUE)")
    p.add_argument(
        "--maximize",
        action="store_true",
        help="Highest wins (default: lowest, since metrics are usually errors)",
    )
    p.add_argument("--tag", "-t", action="append")
    p.add_argument("--all", action="store_true", help="Every experiment")
    add_format(p, ("table", "json"))
    p.set_defaults(func=cmd_best)

    p = subparsers.add_parser("diff", help="Compare runs, with deltas")
    p.add_argument("run_ids", type=int, nargs="+")
    p.add_argument("--metric", "-m", action="append")
    p.add_argument("--dim", "-d", action="append")
    add_format(p)
    p.set_defaults(func=cmd_diff)

    p = subparsers.add_parser("log", help="One line per run")
    p.add_argument("experiment", nargs="?", help="Id or name (default: the newest)")
    p.add_argument("--all", action="store_true", help="Every experiment")
    p.add_argument("--metric", "-m", action="append", help="Column per metric")
    p.add_argument("--dim", "-d", action="append", help="Dims for those metrics")
    add_format(p)
    p.set_defaults(func=cmd_log)

    p = subparsers.add_parser("audit", help="Recompute a metric from its prediction rows")
    p.add_argument("run_id", type=int)
    p.add_argument("--metric", "-m", required=True, choices=sorted(SCORERS))
    p.add_argument("--dim", "-d", action="append")
    add_format(p, ("table", "json"))
    p.set_defaults(func=cmd_audit)

    p = subparsers.add_parser("snapshot", help="Write committable files")
    p.add_argument("experiment", nargs="?", help="Id or name (default: the newest)")
    p.add_argument("directory")
    p.add_argument("--predictions", action="store_true", help="Include prediction rows")
    p.set_defaults(func=cmd_snapshot)

    p = subparsers.add_parser("artifacts", help="List or fetch artifacts")
    p.add_argument("experiment", nargs="?", help="Id or name (default: the newest)")
    p.add_argument("--all", action="store_true", help="Every experiment")
    p.add_argument("--get", type=int, metavar="ID", help="Write this artifact to a file")
    p.add_argument("--output", "-o", help="Where --get writes")
    add_format(p)
    p.set_defaults(func=cmd_artifacts)

    p = subparsers.add_parser("rm", help="Delete an experiment and its runs")
    p.add_argument("experiment")
    p.add_argument("--yes", action="store_true", help="Confirm the deletion")
    p.set_defaults(func=cmd_rm)

    p = subparsers.add_parser("sql", help="Run SQL")
    p.add_argument("query")
    add_format(p)
    p.set_defaults(func=cmd_sql)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
