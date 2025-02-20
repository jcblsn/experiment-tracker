import json
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
