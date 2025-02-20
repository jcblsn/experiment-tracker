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

    def create_experiment(self, name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO experiments (name) VALUES (?)", (name,))
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
        pass

    def log_predictions(
        self, run_id: int, preds: list[float], actuals: list[float]
    ) -> None:
        pass

    def end_run(self, run_id: int, success: bool = True, error: str = None) -> None:
        pass
