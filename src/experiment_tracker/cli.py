#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import sqlite3
import sys
from typing import Any

from .experiment_tracker import ExperimentTracker


def find_database(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    env_path = os.environ.get("EXPT_DB")
    if env_path:
        return env_path
    default_path = "./experiments.db"
    if os.path.exists(default_path):
        return default_path
    sys.exit(
        "Error: No database found. Specify --db PATH, set EXPT_DB, or run from a directory with experiments.db"
    )


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "No results."
    if columns is None:
        columns = list(rows[0].keys())
    col_widths = {col: len(str(col)) for col in columns}
    for row in rows:
        for col in columns:
            val = row.get(col, "")
            col_widths[col] = max(col_widths[col], len(str(val) if val is not None else ""))
    header = "  ".join(str(col).ljust(col_widths[col]) for col in columns)
    separator = "  ".join("-" * col_widths[col] for col in columns)
    lines = [header, separator]
    for row in rows:
        line = "  ".join(
            str(row.get(col, "") if row.get(col) is not None else "").ljust(col_widths[col])
            for col in columns
        )
        lines.append(line)
    return "\n".join(lines)


def format_output(data: Any, fmt: str, columns: list[str] | None = None) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)
    elif fmt == "csv":
        if not data:
            return ""
        if isinstance(data, dict):
            data = [data]
        output = io.StringIO()
        if columns is None:
            columns = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue().strip()
    else:
        if isinstance(data, dict):
            data = [data]
        return format_table(data, columns)


def cmd_list(args):
    tracker = ExperimentTracker(find_database(args.db))
    if args.search:
        experiments = tracker.find_experiments(args.search)
    else:
        experiments = tracker.list_experiments(limit=args.limit)
    for exp in experiments:
        cursor = tracker.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM runs WHERE experiment_id = ?",
            (exp["experiment_id"],),
        )
        exp["runs"] = cursor.fetchone()[0]
    columns = ["experiment_id", "experiment_name", "runs", "created_time"]
    print(format_output(experiments, args.format, columns))


def cmd_show(args):
    tracker = ExperimentTracker(find_database(args.db))
    exp = tracker.get_experiment(args.experiment_id)
    if exp is None:
        sys.exit(f"Error: Experiment {args.experiment_id} not found.")
    cursor = tracker.conn.cursor()
    cursor.execute(
        "SELECT run_status, COUNT(*) as count FROM runs WHERE experiment_id = ? GROUP BY run_status",
        (args.experiment_id,),
    )
    status_counts = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.execute(
        "SELECT run_id, run_status, run_start_time FROM runs WHERE experiment_id = ? ORDER BY run_start_time DESC LIMIT 5",
        (args.experiment_id,),
    )
    recent_runs = [
        {"run_id": row[0], "run_status": row[1], "run_start_time": row[2]}
        for row in cursor.fetchall()
    ]
    tags = tracker.get_tags("experiment", args.experiment_id)
    output = {
        "experiment_id": exp["experiment_id"],
        "experiment_name": exp["experiment_name"],
        "experiment_description": exp["experiment_description"],
        "created_time": exp["created_time"],
        "run_counts": status_counts,
        "tags": tags,
        "recent_runs": recent_runs,
    }
    if args.format == "json":
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"Experiment: {exp['experiment_name']} (ID: {exp['experiment_id']})")
        if exp["experiment_description"]:
            print(f"Description: {exp['experiment_description']}")
        print(f"Created: {exp['created_time']}")
        print()
        print("Run counts:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        if tags:
            print()
            print("Tags:")
            for name, value in tags.items():
                print(f"  {name}: {value}")
        if recent_runs:
            print()
            print("Recent runs:")
            print(format_table(recent_runs))


def cmd_runs(args):
    tracker = ExperimentTracker(find_database(args.db))
    exp = tracker.get_experiment(args.experiment_id)
    if exp is None:
        sys.exit(f"Error: Experiment {args.experiment_id} not found.")
    tag_filters = {}
    if args.tag:
        for t in args.tag:
            if "=" in t:
                key, val = t.split("=", 1)
                tag_filters[key] = val
            else:
                tag_filters[t] = ""
    if tag_filters:
        run_ids = tracker.find_runs(tag_filters, experiment_id=args.experiment_id)
        runs = []
        for rid in run_ids:
            cursor = tracker.conn.cursor()
            cursor.execute(
                "SELECT run_id, run_status, run_start_time, run_end_time, error FROM runs WHERE run_id = ?",
                (rid,),
            )
            row = cursor.fetchone()
            if row:
                runs.append(
                    {
                        "run_id": row[0],
                        "run_status": row[1],
                        "run_start_time": row[2],
                        "run_end_time": row[3],
                        "error": row[4],
                    }
                )
    else:
        runs = tracker.get_run_history(args.experiment_id)
    if args.status:
        runs = [r for r in runs if r["run_status"] == args.status]
    for run in runs:
        run["tags"] = tracker.get_tags("run", run["run_id"])
        model = tracker.get_model(run["run_id"])
        run["model"] = model["model_name"] if model else None
        try:
            run["metrics"] = tracker._get_metrics_safe(run["run_id"])
        except ValueError:
            run["metrics"] = {}
    if args.format == "json":
        print(json.dumps(runs, indent=2, default=str))
    else:
        display_rows = []
        for run in runs:
            row = {
                "run_id": run["run_id"],
                "status": run["run_status"],
                "model": run["model"] or "",
                "started": run["run_start_time"],
            }
            if run["tags"]:
                row["tags"] = ", ".join(f"{k}={v}" for k, v in run["tags"].items())
            else:
                row["tags"] = ""
            if run["metrics"]:
                row["metrics"] = ", ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in run["metrics"].items()
                )
            else:
                row["metrics"] = ""
            display_rows.append(row)
        print(format_output(display_rows, args.format))


def cmd_metrics(args):
    tracker = ExperimentTracker(find_database(args.db))
    try:
        metrics = tracker.get_metrics(args.run_id)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    if args.format == "json":
        print(json.dumps({"run_id": args.run_id, "metrics": metrics}, indent=2))
    else:
        print(f"Metrics for run {args.run_id}:")
        print()
        rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        print(format_output(rows, args.format, ["metric", "value"]))


def cmd_best(args):
    tracker = ExperimentTracker(find_database(args.db))
    exp = tracker.get_experiment(args.experiment_id)
    if exp is None:
        sys.exit(f"Error: Experiment {args.experiment_id} not found.")
    tag_filters = {}
    if args.tag:
        for t in args.tag:
            if "=" in t:
                key, val = t.split("=", 1)
                tag_filters[key] = val
            else:
                tag_filters[t] = ""
    best = tracker.best_run(
        args.experiment_id,
        args.metric,
        minimize=args.minimize,
        where_tags=tag_filters if tag_filters else None,
    )
    if best is None:
        sys.exit(f"Error: No runs with metric '{args.metric}' found.")
    model = best.get("model")
    tags = best.get("tags", {})
    metrics = best.get("metrics", {})
    output = {
        "run_id": best["run_id"],
        "model": model["model_name"] if model else None,
        "parameters": model["parameters"] if model else None,
        args.metric: metrics.get(args.metric),
        "all_metrics": metrics,
        "tags": tags,
    }
    if args.format == "json":
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"Best run by {args.metric} ({'min' if args.minimize else 'max'}):")
        print()
        print(f"Run ID: {best['run_id']}")
        if model:
            print(f"Model: {model['model_name']}")
        print(f"{args.metric}: {metrics.get(args.metric)}")
        if tags:
            print(f"Tags: {', '.join(f'{k}={v}' for k, v in tags.items())}")


def cmd_aggregate(args):
    tracker = ExperimentTracker(find_database(args.db))
    exp = tracker.get_experiment(args.experiment_id)
    if exp is None:
        sys.exit(f"Error: Experiment {args.experiment_id} not found.")
    tag_filters = {}
    if args.tag:
        for t in args.tag:
            if "=" in t:
                key, val = t.split("=", 1)
                tag_filters[key] = val
            else:
                tag_filters[t] = ""
    group_by = args.group_by if args.group_by else None
    group_by_params = args.group_by_param if args.group_by_param else None
    aggregations = args.agg if args.agg else None
    results = tracker.aggregate(
        args.experiment_id,
        args.metric,
        group_by=group_by,
        group_by_params=group_by_params,
        where_tags=tag_filters if tag_filters else None,
        aggregations=aggregations,
    )
    count_key = f"{args.metric}_count"
    if not results or (len(results) == 1 and results[0].get(count_key) == 0):
        sys.exit(f"Error: No runs with metric '{args.metric}' found.")
    print(format_output(results, args.format))


def cmd_compare(args):
    tracker = ExperimentTracker(find_database(args.db))
    if len(args.run_ids) < 2:
        sys.exit("Error: At least 2 run IDs required for comparison.")
    comparisons = []
    for run_id in args.run_ids:
        cursor = tracker.conn.cursor()
        cursor.execute(
            "SELECT run_id, run_status, run_start_time FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            sys.exit(f"Error: Run {run_id} not found.")
        model = tracker.get_model(run_id)
        tags = tracker.get_tags("run", run_id)
        metrics = tracker._get_metrics_safe(run_id)
        comparisons.append(
            {
                "run_id": run_id,
                "status": row[1],
                "model": model["model_name"] if model else None,
                "parameters": model["parameters"] if model else None,
                "tags": tags,
                "metrics": metrics,
            }
        )
    if args.format == "json":
        print(json.dumps(comparisons, indent=2, default=str))
    else:
        all_metrics = set()
        for c in comparisons:
            all_metrics.update(c["metrics"].keys())
        rows = []
        for c in comparisons:
            row = {
                "run_id": c["run_id"],
                "model": c["model"] or "",
                "status": c["status"],
            }
            for m in sorted(all_metrics):
                row[m] = c["metrics"].get(m, "")
            rows.append(row)
        columns = ["run_id", "model", "status"] + sorted(all_metrics)
        print(format_output(rows, args.format, columns))


def cmd_sql(args):
    db_path = find_database(args.db)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(args.query)
    except sqlite3.Error as e:
        sys.exit(f"SQL Error: {e}")
    rows = cursor.fetchall()
    if not rows:
        print("No results.")
        return
    columns = [desc[0] for desc in cursor.description]
    data = [dict(row) for row in rows]
    print(format_output(data, args.format, columns))


def cmd_export(args):
    tracker = ExperimentTracker(find_database(args.db))
    try:
        output_dir = tracker.export_experiment(args.experiment_id, args.output_dir)
        print(f"Exported experiment {args.experiment_id} to {output_dir}")
    except ValueError as e:
        sys.exit(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        prog="expt",
        description="Query and inspect experiment tracking data",
    )
    parser.add_argument(
        "--db", metavar="PATH", help="Path to database (default: EXPT_DB env or ./experiments.db)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--limit", "-n", type=int, default=10, help="Max experiments to show")
    list_parser.add_argument("--search", "-s", help="Search by name pattern")
    list_parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table"
    )
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment_id", type=int, help="Experiment ID")
    show_parser.add_argument(
        "--format", "-f", choices=["table", "json"], default="table"
    )
    show_parser.set_defaults(func=cmd_show)

    runs_parser = subparsers.add_parser("runs", help="List runs for an experiment")
    runs_parser.add_argument("experiment_id", type=int, help="Experiment ID")
    runs_parser.add_argument(
        "--tag", "-t", action="append", help="Filter by tag (KEY=VALUE)"
    )
    runs_parser.add_argument(
        "--status", choices=["RUNNING", "COMPLETED", "FAILED"], help="Filter by status"
    )
    runs_parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table"
    )
    runs_parser.set_defaults(func=cmd_runs)

    metrics_parser = subparsers.add_parser("metrics", help="Show metrics for a run")
    metrics_parser.add_argument("run_id", type=int, help="Run ID")
    metrics_parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table"
    )
    metrics_parser.set_defaults(func=cmd_metrics)

    best_parser = subparsers.add_parser("best", help="Find best run by metric")
    best_parser.add_argument("experiment_id", type=int, help="Experiment ID")
    best_parser.add_argument("--metric", "-m", required=True, help="Metric to optimize")
    best_parser.add_argument(
        "--minimize", action="store_true", help="Minimize instead of maximize"
    )
    best_parser.add_argument(
        "--tag", "-t", action="append", help="Filter by tag (KEY=VALUE)"
    )
    best_parser.add_argument(
        "--format", "-f", choices=["table", "json"], default="table"
    )
    best_parser.set_defaults(func=cmd_best)

    agg_parser = subparsers.add_parser("aggregate", help="Aggregate metrics across runs")
    agg_parser.add_argument("experiment_id", type=int, help="Experiment ID")
    agg_parser.add_argument("--metric", "-m", required=True, help="Metric to aggregate")
    agg_parser.add_argument(
        "--group-by", "-g", action="append", help="Group by tag name"
    )
    agg_parser.add_argument(
        "--group-by-param", "-p", action="append", help="Group by model parameter (e.g., 'degree')"
    )
    agg_parser.add_argument(
        "--tag", "-t", action="append", help="Filter by tag (KEY=VALUE)"
    )
    agg_parser.add_argument(
        "--agg", "-a", action="append",
        choices=["mean", "std", "count", "min", "max", "sum"],
        help="Aggregation functions (default: mean, std, count)"
    )
    agg_parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table"
    )
    agg_parser.set_defaults(func=cmd_aggregate)

    compare_parser = subparsers.add_parser("compare", help="Compare runs side-by-side")
    compare_parser.add_argument("run_ids", type=int, nargs="+", help="Run IDs to compare")
    compare_parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table"
    )
    compare_parser.set_defaults(func=cmd_compare)

    sql_parser = subparsers.add_parser("sql", help="Run SQL query")
    sql_parser.add_argument("query", help="SQL query to execute")
    sql_parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table"
    )
    sql_parser.set_defaults(func=cmd_sql)

    export_parser = subparsers.add_parser("export", help="Export experiment to CSV")
    export_parser.add_argument("experiment_id", type=int, help="Experiment ID")
    export_parser.add_argument("output_dir", help="Output directory")
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
