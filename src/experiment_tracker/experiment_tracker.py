import json
import math
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
                run_id INTEGER,
                predictions BLOB,
                actuals BLOB,
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

    def _calculate_default_metrics(
        self, run_id: int, preds: list[float], actuals: list[float]
    ) -> None:
        n = len(preds)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / n)

        mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / n

        mean_actual = sum(actuals) / n
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum((a - p) ** 2 for a, p in zip(actuals, preds))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        cursor = self.conn.cursor()
        metrics = [("rmse", rmse), ("mae", mae), ("r2", r2)]
        cursor.executemany(
            "INSERT INTO metrics (run_id, name, value) VALUES (?, ?, ?)",
            [(run_id, name, value) for name, value in metrics],
        )

    def log_predictions(
        self, run_id: int, preds: list[float], actuals: list[float]
    ) -> None:
        if not isinstance(preds, list) or not isinstance(actuals, list):
            raise TypeError("preds and actuals must be lists")
        if len(preds) != len(actuals):
            raise ValueError("preds and actuals must have the same length")

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM runs WHERE id = ?", (run_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Run ID {run_id} does not exist")

        serialized_preds = json.dumps(preds).encode("utf-8")
        serialized_actuals = json.dumps(actuals).encode("utf-8")

        cursor.execute(
            "INSERT INTO predictions (run_id, predictions, actuals) VALUES (?, ?, ?)",
            (run_id, serialized_preds, serialized_actuals),
        )

        self._calculate_default_metrics(run_id, preds, actuals)

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

    def get_predictions(self, run_id: int) -> dict | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT predictions, actuals FROM predictions WHERE run_id = ?", (run_id,)
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No predictions found for run id {run_id}")
        preds = json.loads(result[0].decode("utf-8"))
        actuals = json.loads(result[1].decode("utf-8"))
        return {"predictions": preds, "actuals": actuals}

    def get_metrics(self, run_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, value FROM metrics WHERE run_id = ?", (run_id,))
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No metrics found for run id {run_id}")
        return {name: value for name, value in rows}
