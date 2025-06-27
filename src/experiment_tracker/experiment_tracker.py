import csv
import json
import math
import os
import sqlite3
from typing import Union


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


class ExperimentTracker:
    def __init__(self, db_path: str = "experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_db()

    def _initialize_db(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                experiment_description TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                run_status TEXT CHECK(run_status IN ('RUNNING', 'COMPLETED', 'FAILED')),
                run_start_time TIMESTAMP,
                run_end_time TIMESTAMP,
                error TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_name TEXT NOT NULL,
                parameters TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                idx INTEGER,
                prediction REAL,
                actual REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id INTEGER,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                UNIQUE(run_id, metric),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER PRIMARY KEY,
                entity_type TEXT CHECK(entity_type IN ('experiment', 'run')),
                entity_id INTEGER,
                name TEXT NOT NULL,
                value TEXT
            )
        """)

        self.conn.commit()

    def create_experiment(self, name: str, description: str | None = None) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (experiment_name, experiment_description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()
        return cursor.lastrowid

    def start_run(self, experiment_id: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO runs (experiment_id, run_status, run_start_time) VALUES (?, 'RUNNING', CURRENT_TIMESTAMP)",
            (experiment_id,),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_model(self, run_id: int, name: str, params: dict) -> None:
        cursor = self.conn.cursor()
        serialized_params = json.dumps(params)
        cursor.execute(
            "INSERT INTO models (run_id, model_name, parameters) VALUES (?, ?, ?)",
            (run_id, name, serialized_params),
        )
        self.conn.commit()

    def _log_metric(self, run_id: int, name: str, value: float) -> None:
        cursor = self.conn.cursor()
        rounded_value = smart_round(value)
        cursor.execute(
            """INSERT INTO metrics (run_id, metric, value)
               VALUES (?, ?, ?)
               ON CONFLICT(run_id, metric) DO UPDATE SET value = ?""",
            (run_id, name, rounded_value, rounded_value),
        )

    def _calculate_default_metrics(
        self, run_id: int, preds: list[float], actuals: list[float]
    ) -> None:
        n = len(preds)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)
        mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / n

        metrics = [("rmse", smart_round(rmse)), ("mae", smart_round(mae))]
        for name, value in metrics:
            self._log_metric(run_id, name, value)

    def log_predictions(
        self,
        run_id: int,
        preds: list[float],
        actuals: list[float],
        index: list[int] | None = None,
        custom_metrics: dict[str, callable] | None = None,
    ) -> None:
        if not isinstance(preds, list) or not isinstance(actuals, list):
            raise TypeError("preds and actuals must be lists")
        if len(preds) != len(actuals):
            raise ValueError("preds and actuals must have the same length")
        if index is not None and len(index) != len(preds):
            raise ValueError("index must have the same length as preds and actuals")

        cursor = self.conn.cursor()
        cursor.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Run ID {run_id} does not exist")

        if index is None:
            for pred, actual in zip(preds, actuals):
                cursor.execute(
                    "INSERT INTO predictions (run_id, prediction, actual) VALUES (?, ?, ?)",
                    (run_id, smart_round(pred), smart_round(actual)),
                )
        else:
            for pred, actual, idx in zip(preds, actuals, index):
                cursor.execute(
                    "INSERT INTO predictions (run_id, idx, prediction, actual) VALUES (?, ?, ?, ?)",
                    (run_id, idx, smart_round(pred), smart_round(actual)),
                )

        self._calculate_default_metrics(run_id, preds, actuals)

        if custom_metrics:
            for name, metric_fn in custom_metrics.items():
                value = metric_fn(preds, actuals)
                self._log_metric(run_id, name, value)

        self.conn.commit()

    def end_run(
        self, run_id: int, success: bool = True, error: str | None = None
    ) -> None:
        cursor = self.conn.cursor()
        status = "COMPLETED" if success else "FAILED"
        cursor.execute(
            """UPDATE runs
            SET run_status = ?,
                run_end_time = CURRENT_TIMESTAMP,
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

    def log_metric(self, run_id: int, name: str, value: float) -> None:
        self._log_metric(run_id, name, value)
        self.conn.commit()

    def get_metrics(self, run_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT metric, value FROM metrics WHERE run_id = ?", (run_id,))
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No metrics found for run id {run_id}")
        return {name: value for name, value in rows}

    def add_tag(
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
            "INSERT INTO tags (entity_type, entity_id, name, value) VALUES (?, ?, ?, ?)",
            (entity_type, entity_id, tag_name, tag_value),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_tags(self, entity_type: str, entity_id: int) -> dict:
        if entity_type not in ["experiment", "run"]:
            raise ValueError("entity_type must be either 'experiment' or 'run'")

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name, value FROM tags WHERE entity_type = ? AND entity_id = ?",
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
                "SELECT DISTINCT entity_id FROM tags WHERE entity_type = ? AND name = ?",
                (entity_type, tag_name),
            )
        else:
            cursor.execute(
                "SELECT DISTINCT entity_id FROM tags WHERE entity_type = ? AND name = ? AND value = ?",
                (entity_type, tag_name, tag_value),
            )

        return [row[0] for row in cursor.fetchall()]

    def delete_tag(
        self, entity_type: str, entity_id: int, tag_name: str, tag_value: str = None
    ) -> int:
        if entity_type not in ["experiment", "run"]:
            raise ValueError("entity_type must be either 'experiment' or 'run'")

        cursor = self.conn.cursor()
        if tag_value is None:
            cursor.execute(
                "DELETE FROM tags WHERE entity_type = ? AND entity_id = ? AND name = ?",
                (entity_type, entity_id, tag_name),
            )
        else:
            cursor.execute(
                "DELETE FROM tags WHERE entity_type = ? AND entity_id = ? AND name = ? AND value = ?",
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
                                predictions["index"][i]
                                if i < len(predictions["index"])
                                else "",
                            ]
                        )
                except ValueError:
                    continue

    def _export_metrics_data(self, runs: list, export_dir: str) -> None:
        with open(os.path.join(export_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "metric", "value"])
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
            "SELECT tag_id, entity_type, entity_id, name, value FROM tags WHERE (entity_type = 'experiment' AND entity_id = ?) OR (entity_type = 'run' AND entity_id IN (SELECT run_id FROM runs WHERE experiment_id = ?))",
            (experiment_id, experiment_id),
        )
        tags = cursor.fetchall()

        with open(os.path.join(export_dir, "tags.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag_id", "entity_type", "entity_id", "name", "value"])
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

                    self.add_tag(entity_type, new_entity_id, tag_name, tag_value)

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

    def get_best_model(self, experiment_id: int, metric: str = "rmse") -> dict | None:
        runs = self.get_run_history(experiment_id)
        if not runs:
            return None

        model_metrics = {}

        for run in runs:
            run_id = run["run_id"]

            try:
                model = self.get_model(run_id)
                metrics = self.get_metrics(run_id)

                if not model or metric not in metrics:
                    continue
                model_key = (
                    model["model_name"],
                    str(sorted(model["parameters"].items())),
                )

                if model_key not in model_metrics:
                    model_metrics[model_key] = {
                        "model_name": model["model_name"],
                        "parameters": model["parameters"],
                        "metric_values": [],
                        "run_ids": [],
                    }

                model_metrics[model_key]["metric_values"].append(metrics[metric])
                model_metrics[model_key]["run_ids"].append(run_id)

            except ValueError:
                continue

        if not model_metrics:
            return None

        best_model = None
        best_avg_metric = float("inf") if metric in ["rmse", "mae"] else float("-inf")

        for model_key, data in model_metrics.items():
            avg_metric = sum(data["metric_values"]) / len(data["metric_values"])

            is_better = (
                metric in ["rmse", "mae"] and avg_metric < best_avg_metric
            ) or (metric not in ["rmse", "mae"] and avg_metric > best_avg_metric)

            if is_better:
                best_avg_metric = avg_metric
                best_model = {
                    "model_name": data["model_name"],
                    "parameters": data["parameters"],
                    "average_metric": smart_round(avg_metric),
                    "metric_name": metric,
                    "num_runs": len(data["metric_values"]),
                    "run_ids": data["run_ids"],
                }

        return best_model
