import csv
import json
import math
import os
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Union


def smart_round(
    value: Union[int, float],
    precision: int = 6,
    *,
    keep_exact_integers: bool = True,
) -> Union[int, float]:
    if value == 0:
        return 0

    if keep_exact_integers and float(value).is_integer():
        return int(value)

    abs_val = abs(value)

    if abs_val >= 1:
        return round(value, precision)

    digits = precision - 1 - int(math.floor(math.log10(abs_val)))
    return round(value, digits)


def default_serializer(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.name
    elif hasattr(obj, "__dict__"):
        return str(obj)
    else:
        return obj


class RunHandle:
    def __init__(self, tracker: "ExperimentTracker", run_id: int):
        self.tracker = tracker
        self.run_id = run_id
        self._error = None

    def log_model(
        self,
        model_name: str,
        parameters: dict,
        serializer: Callable[[Any], Any] | None = None,
    ) -> None:
        self.tracker.log_model(self.run_id, model_name, parameters, serializer)

    def log_predictions(
        self,
        predictions: list[float],
        actual_values: list[float] | None = None,
        index: list[int] | None = None,
        metrics: list[str] | None = None,
        update: bool = True,
        custom_metrics: dict[str, callable] | None = None,
    ) -> None:
        self.tracker.log_predictions(
            self.run_id,
            predictions,
            actual_values,
            index,
            metrics,
            update,
            custom_metrics,
        )

    def log_artifact(self, data: bytes, artifact_type: str, filename: str) -> int:
        return self.tracker.log_artifact(self.run_id, data, artifact_type, filename)

    def log_metric(self, metric_name: str, value: float) -> None:
        self.tracker.log_metric(self.run_id, metric_name, value)

    def log_tag(self, tag_name: str, tag_value: str = "") -> int:
        return self.tracker.log_tag("run", self.run_id, tag_name, tag_value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.tracker.end_run(self.run_id, success=True)
        else:
            error_msg = str(exc_val) if exc_val else f"{exc_type.__name__} occurred"
            self.tracker.end_run(self.run_id, success=False, error=error_msg)
        return False


class ExperimentTracker:
    def __init__(self, db_path: str = "experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_db()

    def _initialize_db(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                experiment_description TEXT,
                created_time TEXT DEFAULT (datetime('now'))
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                run_status TEXT CHECK(run_status IN ('RUNNING', 'COMPLETED', 'FAILED')),
                run_start_time TEXT,
                run_end_time TEXT,
                error TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_name TEXT NOT NULL,
                parameters TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                idx INTEGER,
                prediction REAL,
                actual REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                run_id INTEGER,
                metric TEXT NOT NULL,
                metric_value REAL NOT NULL,
                UNIQUE(run_id, metric),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER PRIMARY KEY,
                entity_type TEXT CHECK(entity_type IN ('experiment', 'run')),
                entity_id INTEGER,
                tag_name TEXT NOT NULL,
                tag_value TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                artifact_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                data BLOB NOT NULL,
                created_time TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """
        )

        self.conn.commit()

    def create_experiment(
        self, experiment_name: str, experiment_description: str | None = None
    ) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (experiment_name, experiment_description) VALUES (?, ?)",
            (experiment_name, experiment_description),
        )
        self.conn.commit()
        return cursor.lastrowid

    def start_run(self, experiment_id: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO runs (experiment_id, run_status, run_start_time) VALUES (?, 'RUNNING', datetime('now'))",
            (experiment_id,),
        )
        self.conn.commit()
        return cursor.lastrowid

    def run(self, experiment_id: int, tags: dict[str, str] | None = None) -> RunHandle:
        run_id = self.start_run(experiment_id)
        if tags:
            for tag_name, tag_value in tags.items():
                self.log_tag("run", run_id, tag_name, tag_value)
        return RunHandle(self, run_id)

    def log_model(
        self,
        run_id: int,
        model_name: str,
        parameters: dict,
        serializer: Callable[[Any], Any] | None = None,
    ) -> None:
        cursor = self.conn.cursor()
        if serializer is None:
            serializer = default_serializer

        def serialize_recursive(obj):
            if isinstance(obj, dict):
                return {k: serialize_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_recursive(item) for item in obj]
            else:
                return serializer(obj)

        serialized_params = json.dumps(serialize_recursive(parameters))
        cursor.execute(
            "INSERT INTO models (run_id, model_name, parameters) VALUES (?, ?, ?)",
            (run_id, model_name, serialized_params),
        )
        self.conn.commit()

    def _log_metric(self, run_id: int, metric_name: str, value: float) -> None:
        cursor = self.conn.cursor()
        rounded_value = smart_round(value)
        cursor.execute(
            """INSERT INTO metrics (run_id, metric, metric_value)
               VALUES (?, ?, ?)
               ON CONFLICT(run_id, metric) DO UPDATE SET metric_value = ?""",
            (run_id, metric_name, rounded_value, rounded_value),
        )

    def _calculate_default_metrics(
        self,
        run_id: int,
        predictions: list[float],
        actual_values: list[float],
        metrics: list[str] | None = None,
    ) -> None:
        if metrics is None:
            metrics = ["rmse", "mae"]

        n = len(predictions)
        available_metrics = {
            "rmse": lambda: math.sqrt(
                sum((p - a) ** 2 for p, a in zip(predictions, actual_values)) / n
            ),
            "mae": lambda: sum(abs(p - a) for p, a in zip(predictions, actual_values))
            / n,
            "mape": lambda: (
                (
                    sum(
                        abs((a - p) / a)
                        for p, a in zip(predictions, actual_values)
                        if a != 0
                    )
                    / sum(1 for a in actual_values if a != 0)
                )
                if any(a != 0 for a in actual_values)
                else 0
            ),
        }

        for metric_name in metrics:
            if metric_name in available_metrics:
                value = available_metrics[metric_name]()
                self._log_metric(run_id, metric_name, value)

    def log_predictions(
        self,
        run_id: int,
        predictions: list[float],
        actual_values: list[float] | None = None,
        index: list[int] | None = None,
        metrics: list[str] | None = None,
        update: bool = True,
        custom_metrics: dict[str, callable] | None = None,
    ) -> None:
        if not isinstance(predictions, list):
            raise TypeError("predictions must be a list")
        if actual_values is not None:
            if not isinstance(actual_values, list):
                raise TypeError("actual_values must be a list")
            if len(predictions) != len(actual_values):
                raise ValueError(
                    "predictions and actual_values must have the same length"
                )
        if index is not None and len(index) != len(predictions):
            raise ValueError("index must have the same length as predictions")

        cursor = self.conn.cursor()
        cursor.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Run ID {run_id} does not exist")

        if update:
            cursor.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))

        if actual_values is None:
            if index is None:
                for pred in predictions:
                    cursor.execute(
                        "INSERT INTO predictions (run_id, prediction) VALUES (?, ?)",
                        (run_id, smart_round(pred)),
                    )
            else:
                for pred, idx in zip(predictions, index):
                    cursor.execute(
                        "INSERT INTO predictions (run_id, idx, prediction) VALUES (?, ?, ?)",
                        (run_id, idx, smart_round(pred)),
                    )
        else:
            if index is None:
                for pred, actual in zip(predictions, actual_values):
                    cursor.execute(
                        "INSERT INTO predictions (run_id, prediction, actual) VALUES (?, ?, ?)",
                        (run_id, smart_round(pred), smart_round(actual)),
                    )
            else:
                for pred, actual, idx in zip(predictions, actual_values, index):
                    cursor.execute(
                        "INSERT INTO predictions (run_id, idx, prediction, actual) VALUES (?, ?, ?, ?)",
                        (run_id, idx, smart_round(pred), smart_round(actual)),
                    )

        if actual_values is not None:
            self._calculate_default_metrics(run_id, predictions, actual_values, metrics)

            if custom_metrics:
                for name, metric_fn in custom_metrics.items():
                    value = metric_fn(predictions, actual_values)
                    self._log_metric(run_id, name, value)

        self.conn.commit()

    def log_artifact(
        self, run_id: int, data: bytes, artifact_type: str, filename: str
    ) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Run ID {run_id} does not exist")

        cursor.execute(
            "INSERT INTO artifacts (run_id, artifact_type, filename, data) VALUES (?, ?, ?, ?)",
            (run_id, artifact_type, filename, data),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_artifacts(self, run_id: int) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT artifact_id, artifact_type, filename, created_time FROM artifacts WHERE run_id = ? ORDER BY created_time",
            (run_id,),
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_artifact_data(self, artifact_id: int) -> bytes:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT data FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Artifact ID {artifact_id} does not exist")
        return row[0]

    def end_run(
        self, run_id: int, success: bool = True, error: str | None = None
    ) -> None:
        cursor = self.conn.cursor()
        status = "COMPLETED" if success else "FAILED"
        cursor.execute(
            """UPDATE runs
            SET run_status = ?,
                run_end_time = datetime('now'),
                error = ?
            WHERE run_id = ?""",
            (status, error, run_id),
        )
        self.conn.commit()

    def get_experiment(self, experiment_id: int) -> dict | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT experiment_id, experiment_name, experiment_description, created_time FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))

    def get_run_history(self, experiment_id: int) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT run_id, experiment_id, run_status, run_start_time, run_end_time, error FROM runs WHERE experiment_id = ? ORDER BY run_start_time DESC",
            (experiment_id,),
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_model(self, run_id: int) -> dict | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT model_id, run_id, model_name, parameters FROM models WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [col[0] for col in cursor.description]
        model = dict(zip(columns, row))
        model["parameters"] = json.loads(model["parameters"])
        return model

    def get_predictions(self, run_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT prediction, actual, idx FROM predictions WHERE run_id = ? ORDER BY idx",
            (run_id,),
        )
        results = cursor.fetchall()
        if not results:
            raise ValueError(f"No predictions found for run id {run_id}")

        preds = [row[0] for row in results]
        actuals = [row[1] for row in results]
        index = [row[2] for row in results]

        return {"predictions": preds, "actuals": actuals, "index": index}

    def log_metric(self, run_id: int, metric_name: str, value: float) -> None:
        self._log_metric(run_id, metric_name, value)
        self.conn.commit()

    def get_metrics(self, run_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT metric, metric_value FROM metrics WHERE run_id = ?", (run_id,)
        )
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No metrics found for run id {run_id}")
        return {name: value for name, value in rows}

    def log_tag(
        self, entity_type: str, entity_id: int, tag_name: str, tag_value: str = ""
    ) -> int:
        if entity_type not in ["experiment", "run"]:
            raise ValueError("entity_type must be either 'experiment' or 'run'")

        cursor = self.conn.cursor()
        if entity_type == "experiment":
            cursor.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
                (entity_id,),
            )
        else:
            cursor.execute("SELECT run_id FROM runs WHERE run_id = ?", (entity_id,))

        if cursor.fetchone() is None:
            raise ValueError(
                f"{entity_type.capitalize()} with ID {entity_id} does not exist"
            )

        cursor.execute(
            "INSERT INTO tags (entity_type, entity_id, tag_name, tag_value) VALUES (?, ?, ?, ?)",
            (entity_type, entity_id, tag_name, tag_value),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_tags(self, entity_type: str, entity_id: int) -> dict:
        if entity_type not in ["experiment", "run"]:
            raise ValueError("entity_type must be either 'experiment' or 'run'")

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT tag_name, tag_value FROM tags WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )
        return {name: value for name, value in cursor.fetchall()}

    def get_tagged_entities(
        self, entity_type: str, tag_name: str, tag_value: str = None
    ) -> list:
        if entity_type not in ["experiment", "run"]:
            raise ValueError("entity_type must be either 'experiment' or 'run'")

        cursor = self.conn.cursor()
        if tag_value is None:
            cursor.execute(
                "SELECT DISTINCT entity_id FROM tags WHERE entity_type = ? AND tag_name = ?",
                (entity_type, tag_name),
            )
        else:
            cursor.execute(
                "SELECT DISTINCT entity_id FROM tags WHERE entity_type = ? AND tag_name = ? AND tag_value = ?",
                (entity_type, tag_name, tag_value),
            )

        return [row[0] for row in cursor.fetchall()]

    def find_runs(
        self, tags: dict[str, str], experiment_id: int | None = None
    ) -> list[int]:
        cursor = self.conn.cursor()

        base_query = """
            SELECT DISTINCT r.run_id 
            FROM runs r
            JOIN tags t ON t.entity_type = 'run' AND t.entity_id = r.run_id
        """

        conditions = []
        params = []

        if experiment_id is not None:
            conditions.append("r.experiment_id = ?")
            params.append(experiment_id)

        for tag_name, tag_value in tags.items():
            conditions.append("(t.tag_name = ? AND t.tag_value = ?)")
            params.extend([tag_name, tag_value])

        if conditions:
            query = base_query + " WHERE " + " AND ".join(conditions)
        else:
            query = "SELECT run_id FROM runs" + (
                f" WHERE experiment_id = {experiment_id}" if experiment_id else ""
            )
            params = []

        query += f" GROUP BY r.run_id HAVING COUNT(*) = {len(tags)}" if tags else ""

        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def aggregate(
        self,
        experiment_id: int,
        metric: str,
        group_by: list[str] | None = None,
        where_tags: dict[str, str] | None = None,
        aggregations: list[str] | None = None,
    ) -> list[dict]:
        if aggregations is None:
            aggregations = ["mean", "std", "count"]

        cursor = self.conn.cursor()

        agg_funcs = {
            "mean": "AVG(m.metric_value)",
            "std": "SQRT(AVG(m.metric_value * m.metric_value) - AVG(m.metric_value) * AVG(m.metric_value))",
            "count": "COUNT(m.metric_value)",
            "min": "MIN(m.metric_value)",
            "max": "MAX(m.metric_value)",
            "sum": "SUM(m.metric_value)",
        }

        select_parts = []
        for agg_name in aggregations:
            if agg_name in agg_funcs:
                select_parts.append(f"{agg_funcs[agg_name]} as {metric}_{agg_name}")

        base_query = f"""
            SELECT {', '.join(select_parts)}
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
        """

        if group_by:
            group_select = []
            for col in group_by:
                if col == "model":
                    base_query += " JOIN models mo ON r.run_id = mo.run_id"
                    group_select.append("mo.model_name as model")
                else:
                    group_select.append(f"t_{col}.tag_value as {col}")
                    base_query += f" JOIN tags t_{col} ON t_{col}.entity_type = 'run' AND t_{col}.entity_id = r.run_id AND t_{col}.tag_name = '{col}'"

            select_parts = group_select + select_parts
            base_query = f"SELECT {', '.join(select_parts)} FROM metrics m JOIN runs r ON m.run_id = r.run_id"

            for col in group_by:
                if col == "model":
                    base_query += " JOIN models mo ON r.run_id = mo.run_id"
                else:
                    base_query += f" JOIN tags t_{col} ON t_{col}.entity_type = 'run' AND t_{col}.entity_id = r.run_id AND t_{col}.tag_name = '{col}'"

        conditions = [f"r.experiment_id = {experiment_id}", f"m.metric = '{metric}'"]

        if where_tags:
            for tag_name, tag_value in where_tags.items():
                base_query += f" JOIN tags wt_{tag_name} ON wt_{tag_name}.entity_type = 'run' AND wt_{tag_name}.entity_id = r.run_id"
                conditions.append(
                    f"wt_{tag_name}.tag_name = '{tag_name}' AND wt_{tag_name}.tag_value = '{tag_value}'"
                )

        query = base_query + " WHERE " + " AND ".join(conditions)

        if group_by:
            query += f" GROUP BY {', '.join(group_by)}"

        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def delete_tag(
        self, entity_type: str, entity_id: int, tag_name: str, tag_value: str = None
    ) -> int:
        if entity_type not in ["experiment", "run"]:
            raise ValueError("entity_type must be either 'experiment' or 'run'")

        cursor = self.conn.cursor()
        if tag_value is None:
            cursor.execute(
                "DELETE FROM tags WHERE entity_type = ? AND entity_id = ? AND tag_name = ?",
                (entity_type, entity_id, tag_name),
            )
        else:
            cursor.execute(
                "DELETE FROM tags WHERE entity_type = ? AND entity_id = ? AND tag_name = ? AND tag_value = ?",
                (entity_type, entity_id, tag_name, tag_value),
            )

        deleted_count = cursor.rowcount
        self.conn.commit()
        return deleted_count

    def export_experiment(self, experiment_id: int, export_dir: str) -> str:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment ID {experiment_id} does not exist")

        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        exp_name = experiment["experiment_name"].replace(" ", "_")
        exp_export_dir = os.path.join(
            export_dir, f"experiment_{experiment_id}_{exp_name}"
        )
        os.makedirs(exp_export_dir, exist_ok=True)

        self._export_experiment_data(experiment, exp_export_dir)
        self._export_runs_data(experiment_id, exp_export_dir)
        self._export_tags_data(experiment_id, exp_export_dir)

        return exp_export_dir

    def _export_experiment_data(self, experiment: dict, export_dir: str) -> None:
        with open(os.path.join(export_dir, "experiments.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "experiment_id",
                    "experiment_name",
                    "experiment_description",
                    "created_time",
                ]
            )
            writer.writerow(
                [
                    experiment["experiment_id"],
                    experiment["experiment_name"],
                    experiment["experiment_description"] or "",
                    experiment["created_time"],
                ]
            )

    def _export_runs_data(self, experiment_id: int, export_dir: str) -> None:
        runs = self.get_run_history(experiment_id)

        with open(os.path.join(export_dir, "runs.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "experiment_id",
                    "run_status",
                    "run_start_time",
                    "run_end_time",
                    "error",
                ]
            )
            for run in runs:
                writer.writerow(
                    [
                        run["run_id"],
                        experiment_id,
                        run["run_status"],
                        run["run_start_time"],
                        run["run_end_time"] or "",
                        run["error"] or "",
                    ]
                )

        self._export_models_data(runs, export_dir)
        self._export_predictions_data(runs, export_dir)
        self._export_metrics_data(runs, export_dir)

    def _export_models_data(self, runs: list, export_dir: str) -> None:
        with open(os.path.join(export_dir, "models.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_id", "run_id", "model_name", "parameters"])
            for run in runs:
                try:
                    model = self.get_model(run["run_id"])
                    if model:
                        writer.writerow(
                            [
                                model["model_id"],
                                run["run_id"],
                                model["model_name"],
                                json.dumps(model["parameters"]),
                            ]
                        )
                except ValueError:
                    continue

    def _export_predictions_data(self, runs: list, export_dir: str) -> None:
        with open(os.path.join(export_dir, "predictions.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "prediction", "actual", "idx"])
            for run in runs:
                try:
                    predictions = self.get_predictions(run["run_id"])
                    for i, pred in enumerate(predictions["predictions"]):
                        writer.writerow(
                            [
                                run["run_id"],
                                pred,
                                predictions["actuals"][i],
                                (
                                    predictions["index"][i]
                                    if i < len(predictions["index"])
                                    else ""
                                ),
                            ]
                        )
                except ValueError:
                    continue

    def _export_metrics_data(self, runs: list, export_dir: str) -> None:
        with open(os.path.join(export_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "metric", "metric_value"])
            for run in runs:
                try:
                    metrics = self.get_metrics(run["run_id"])
                    for name, value in metrics.items():
                        writer.writerow([run["run_id"], name, value])
                except ValueError:
                    continue

    def _export_tags_data(self, experiment_id: int, export_dir: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT tag_id, entity_type, entity_id, tag_name, tag_value FROM tags WHERE (entity_type = 'experiment' AND entity_id = ?) OR (entity_type = 'run' AND entity_id IN (SELECT run_id FROM runs WHERE experiment_id = ?))",
            (experiment_id, experiment_id),
        )
        tags = cursor.fetchall()

        with open(os.path.join(export_dir, "tags.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["tag_id", "entity_type", "entity_id", "tag_name", "tag_value"]
            )
            for tag in tags:
                writer.writerow(tag)

    def import_experiment(self, import_dir: str) -> int:
        if not os.path.exists(import_dir):
            raise ValueError(f"Import directory {import_dir} does not exist")

        exp_file = os.path.join(import_dir, "experiments.csv")
        if not os.path.exists(exp_file):
            raise ValueError(f"experiments.csv not found in {import_dir}")

        with open(exp_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
            exp_name = row[1]
            exp_description = row[2] if row[2] else None

        experiment_id = self.create_experiment(exp_name, exp_description)

        runs_file = os.path.join(import_dir, "runs.csv")
        if not os.path.exists(runs_file):
            return experiment_id

        run_id_map = {}

        with open(runs_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                original_run_id = int(row[0])
                status = row[2]
                error = row[5] if len(row) > 5 and row[5] else None

                run_id = self.start_run(experiment_id)
                run_id_map[original_run_id] = run_id

                if status != "RUNNING":
                    self.end_run(run_id, success=(status == "COMPLETED"), error=error)

        models_file = os.path.join(import_dir, "models.csv")
        if os.path.exists(models_file):
            with open(models_file, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    original_run_id = int(row[1])
                    if original_run_id in run_id_map:
                        new_run_id = run_id_map[original_run_id]
                        model_name = row[2]
                        parameters = json.loads(row[3])
                        self.log_model(new_run_id, model_name, parameters)

        predictions_file = os.path.join(import_dir, "predictions.csv")
        if os.path.exists(predictions_file):
            prediction_data = {}

            with open(predictions_file, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    original_run_id = int(row[0])
                    if original_run_id in run_id_map:
                        if original_run_id not in prediction_data:
                            prediction_data[original_run_id] = {
                                "preds": [],
                                "actuals": [],
                            }

                        prediction_data[original_run_id]["preds"].append(float(row[1]))
                        prediction_data[original_run_id]["actuals"].append(
                            float(row[2])
                        )

            for original_run_id, data in prediction_data.items():
                new_run_id = run_id_map[original_run_id]
                if data["preds"] and data["actuals"]:
                    self.log_predictions(new_run_id, data["preds"], data["actuals"])

        metrics_file = os.path.join(import_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    original_run_id = int(row[0])
                    if original_run_id in run_id_map:
                        new_run_id = run_id_map[original_run_id]
                        metric_name = row[1]
                        metric_value = float(row[2])

                        if metric_name not in ["rmse", "mae"]:
                            self.log_metric(new_run_id, metric_name, metric_value)

        tags_file = os.path.join(import_dir, "tags.csv")
        if os.path.exists(tags_file):
            with open(tags_file, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    entity_type = row[1]
                    original_entity_id = int(row[2])
                    tag_name = row[3]
                    tag_value = row[4] if len(row) > 4 and row[4] else ""

                    if entity_type == "experiment":
                        new_entity_id = experiment_id
                    elif entity_type == "run" and original_entity_id in run_id_map:
                        new_entity_id = run_id_map[original_entity_id]
                    else:
                        continue

                    self.log_tag(entity_type, new_entity_id, tag_name, tag_value)

        return experiment_id

    def list_experiments(self, limit: int = 10) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT experiment_id, experiment_name, experiment_description, created_time FROM experiments ORDER BY created_time DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def find_experiments(self, name_pattern: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT experiment_id, experiment_name, experiment_description, created_time FROM experiments WHERE experiment_name LIKE ? ORDER BY created_time DESC",
            (f"%{name_pattern}%",),
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def delete_experiment(self, experiment_id: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        if cursor.fetchone() is None:
            raise ValueError(f"Experiment ID {experiment_id} does not exist")

        cursor.execute(
            "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        self.conn.commit()
