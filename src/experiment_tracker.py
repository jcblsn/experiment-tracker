import csv
import json
import math
import os
import sqlite3


class ExperimentTracker:
    def __init__(self, db_path: str = "experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_db()

    def _initialize_db(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                status TEXT CHECK(status IN ('RUNNING', 'COMPLETED', 'FAILED')),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                error TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                name TEXT NOT NULL,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction REAL,
                actual REAL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id INTEGER,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                UNIQUE(run_id, name),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
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
            "INSERT INTO experiments (name, description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()
        return cursor.lastrowid

    def start_run(self, experiment_id: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO runs (experiment_id, status, start_time) VALUES (?, 'RUNNING', CURRENT_TIMESTAMP)",
            (experiment_id,),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_model(self, run_id: int, name: str, params: dict) -> None:
        cursor = self.conn.cursor()
        serialized_params = json.dumps(params)
        cursor.execute(
            "INSERT INTO models (run_id, name, parameters) VALUES (?, ?, ?)",
            (run_id, name, serialized_params),
        )
        self.conn.commit()

    def _log_metric(self, run_id: int, name: str, value: float) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO metrics (run_id, name, value)
               VALUES (?, ?, ?)
               ON CONFLICT(run_id, name) DO UPDATE SET value = ?""",
            (run_id, name, value, value),
        )

    def _calculate_default_metrics(
        self, run_id: int, preds: list[float], actuals: list[float]
    ) -> None:
        n = len(preds)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)
        mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / n

        metrics = [("rmse", rmse), ("mae", mae)]
        for name, value in metrics:
            self._log_metric(run_id, name, value)

    def log_predictions(
        self,
        run_id: int,
        preds: list[float],
        actuals: list[float],
        custom_metrics: dict[str, callable] | None = None,
    ) -> None:
        if not isinstance(preds, list) or not isinstance(actuals, list):
            raise TypeError("preds and actuals must be lists")
        if len(preds) != len(actuals):
            raise ValueError("preds and actuals must have the same length")

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM runs WHERE id = ?", (run_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Run ID {run_id} does not exist")

        for pred, actual in zip(preds, actuals):
            cursor.execute(
                "INSERT INTO predictions (run_id, prediction, actual) VALUES (?, ?, ?)",
                (run_id, pred, actual),
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
            SET status = ?,
                end_time = CURRENT_TIMESTAMP,
                error = ?
            WHERE id = ?""",
            (status, error, run_id),
        )
        self.conn.commit()

    def get_experiment(self, experiment_id: int) -> dict | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, name, description, created_at FROM experiments WHERE id = ?",
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
            "SELECT id, experiment_id, status, start_time, end_time, error FROM runs WHERE experiment_id = ? ORDER BY start_time DESC",
            (experiment_id,),
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_models(self, run_id: int) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, run_id, name, parameters, created_at FROM models WHERE run_id = ?",
            (run_id,),
        )
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        models = []
        for row in rows:
            entry = dict(zip(columns, row))
            entry["parameters"] = json.loads(entry["parameters"])
            models.append(entry)
        return models

    def get_predictions(self, run_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT prediction, actual, timestamp FROM predictions WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        )
        results = cursor.fetchall()
        if not results:
            raise ValueError(f"No predictions found for run id {run_id}")

        preds = [row[0] for row in results]
        actuals = [row[1] for row in results]
        timestamps = [row[2] for row in results]

        return {"predictions": preds, "actuals": actuals, "timestamps": timestamps}

    def log_metric(self, run_id: int, name: str, value: float) -> None:
        self._log_metric(run_id, name, value)
        self.conn.commit()

    def get_metrics(self, run_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, value FROM metrics WHERE run_id = ?", (run_id,))
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
            cursor.execute("SELECT id FROM experiments WHERE id = ?", (entity_id,))
        else:
            cursor.execute("SELECT id FROM runs WHERE id = ?", (entity_id,))

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
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment ID {experiment_id} does not exist")

        exp_name = experiment["name"].replace(" ", "_")
        exp_export_dir = os.path.join(
            export_dir, f"experiment_{experiment_id}_{exp_name}"
        )
        os.makedirs(exp_export_dir, exist_ok=True)

        with open(
            os.path.join(exp_export_dir, "experiments.csv"), "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "description", "created_at"])
            writer.writerow(
                [
                    experiment["id"],
                    experiment["name"],
                    experiment["description"] or "",
                    experiment["created_at"],
                ]
            )

        runs = self.get_run_history(experiment_id)

        with open(os.path.join(exp_export_dir, "runs.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["id", "experiment_id", "status", "start_time", "end_time", "error"]
            )
            for run in runs:
                writer.writerow(
                    [
                        run["id"],
                        experiment_id,
                        run["status"],
                        run["start_time"],
                        run["end_time"] or "",
                        run["error"] or "",
                    ]
                )

        with open(os.path.join(exp_export_dir, "models.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "run_id", "name", "parameters", "created_at"])

        with open(
            os.path.join(exp_export_dir, "predictions.csv"), "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "prediction", "actual", "timestamp"])

        with open(os.path.join(exp_export_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "name", "value"])

        with open(os.path.join(exp_export_dir, "tags.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "entity_type", "entity_id", "name", "value"])

        for run in runs:
            run_id = run["id"]

            try:
                models = self.get_models(run_id)
                with open(
                    os.path.join(exp_export_dir, "models.csv"), "a", newline=""
                ) as f:
                    writer = csv.writer(f)
                    for model in models:
                        writer.writerow(
                            [
                                model["id"],
                                run_id,
                                model["name"],
                                json.dumps(model["parameters"]),
                                model["created_at"],
                            ]
                        )
            except Exception:
                pass

            try:
                predictions = self.get_predictions(run_id)
                with open(
                    os.path.join(exp_export_dir, "predictions.csv"), "a", newline=""
                ) as f:
                    writer = csv.writer(f)
                    for i in range(len(predictions["predictions"])):
                        writer.writerow(
                            [
                                run_id,
                                predictions["predictions"][i],
                                predictions["actuals"][i],
                                predictions["timestamps"][i]
                                if i < len(predictions["timestamps"])
                                else "",
                            ]
                        )
            except Exception:
                pass

            try:
                metrics = self.get_metrics(run_id)
                with open(
                    os.path.join(exp_export_dir, "metrics.csv"), "a", newline=""
                ) as f:
                    writer = csv.writer(f)
                    for name, value in metrics.items():
                        writer.writerow([run_id, name, value])
            except Exception:
                pass

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, entity_type, entity_id, name, value FROM tags WHERE entity_type = 'experiment' AND entity_id = ?",
                (experiment_id,),
            )
            experiment_tags = cursor.fetchall()

            if experiment_tags:
                with open(
                    os.path.join(exp_export_dir, "tags.csv"), "a", newline=""
                ) as f:
                    writer = csv.writer(f)
                    for tag in experiment_tags:
                        writer.writerow(tag)
        except Exception:
            pass

        try:
            for run in runs:
                run_id = run["id"]
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT id, entity_type, entity_id, name, value FROM tags WHERE entity_type = 'run' AND entity_id = ?",
                    (run_id,),
                )
                run_tags = cursor.fetchall()

                if run_tags:
                    with open(
                        os.path.join(exp_export_dir, "tags.csv"), "a", newline=""
                    ) as f:
                        writer = csv.writer(f)
                        for tag in run_tags:
                            writer.writerow(tag)
        except Exception:
            pass

        return exp_export_dir

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
