import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any, Self

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    git_commit TEXT,
    git_dirty INTEGER,
    argv TEXT,
    python TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    name TEXT,
    params TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
    note TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    dims TEXT NOT NULL DEFAULT '{}',
    value REAL NOT NULL,
    UNIQUE(run_id, metric, dims)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    dims TEXT NOT NULL DEFAULT '{}',
    prediction REAL,
    actual REAL
);

CREATE TABLE IF NOT EXISTS tags (
    entity_type TEXT NOT NULL CHECK(entity_type IN ('experiment', 'run')),
    entity_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    UNIQUE(entity_type, entity_id, name)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    data BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS tags_entity ON tags(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS artifacts_experiment ON artifacts(experiment_id);
"""

ENTITY_TYPES = ("experiment", "run")


def dims_key(dims: Mapping[str, Any] | None) -> str:
    """Canonical JSON for a dims mapping.

    Sorted and separator-free, so UNIQUE(run_id, metric, dims) still holds when a caller
    passes the same dims with its keys in another order.
    """
    return json.dumps(dict(dims or {}), sort_keys=True, separators=(",", ":"))


def default_serializer(obj: Any) -> Any:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, int | float | str | bool) or obj is None:
        return obj
    return str(obj)


def serialize_params(params: Mapping[str, Any] | None, serializer=None) -> str | None:
    if params is None:
        return None
    fn = serializer or default_serializer

    def walk(obj):
        if isinstance(obj, Mapping):
            return {str(k): walk(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [walk(v) for v in obj]
        return fn(obj)

    return json.dumps(walk(dict(params)), sort_keys=True)


def _git(args: Sequence[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _provenance() -> dict[str, Any]:
    """What is needed to rebuild a run: the commit, whether the tree was dirty, the command.

    A date alone is not a vintage. Outside a work tree the git fields are None rather than
    an empty string, so "not a repository" is distinguishable from "clean".
    """
    commit = _git(["rev-parse", "HEAD"])
    dirty = None
    if commit is not None:
        status = _git(["status", "--porcelain"])
        dirty = None if status is None else int(bool(status))
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "argv": " ".join(sys.argv),
        "python": sys.version.split()[0],
    }


def _entity_id(value: Any) -> int:
    """Accept an id or a row dict wherever an experiment or run is named."""
    if isinstance(value, Mapping):
        for key in ("experiment_id", "run_id"):
            if key in value:
                return int(value[key])
        raise ValueError("mapping has no experiment_id or run_id")
    return int(value)


class RunHandle:
    def __init__(self, tracker: "ExperimentTracker", run_id: int):
        self.tracker = tracker
        self.run_id = run_id

    def log_metric(self, metric: str, value: float, dims: Mapping[str, Any] | None = None) -> None:
        self.tracker.log_metric(self.run_id, metric, value, dims)

    def log_metrics(
        self, metrics: Mapping[str, float], dims: Mapping[str, Any] | None = None
    ) -> None:
        self.tracker.log_metrics(self.run_id, metrics, dims)

    def log_predictions(
        self,
        predictions: Sequence[float],
        actuals: Sequence[float] | None = None,
        dims: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        replace: bool = True,
    ) -> None:
        self.tracker.log_predictions(self.run_id, predictions, actuals, dims, replace)

    def log_artifact(self, data: bytes, kind: str, filename: str) -> int:
        return self.tracker.log_artifact(data, kind, filename, run=self.run_id)

    def log_file(self, path: str, kind: str | None = None) -> int:
        return self.tracker.log_file(path, kind, run=self.run_id)

    def log_tag(self, name: str, value: str = "") -> None:
        self.tracker.log_tag("run", self.run_id, name, value)

    def set_note(self, note: str) -> None:
        self.tracker.set_note("run", self.run_id, note)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.tracker.end_run(self.run_id)
        else:
            message = str(exc_val) or exc_type.__name__
            self.tracker.end_run(self.run_id, success=False, error=message)
        return False


class ExperimentTracker:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # ----- writes -----

    def experiment(
        self,
        name: str,
        description: str | None = None,
        note: str | None = None,
        tags: Mapping[str, Any] | None = None,
        get_or_create: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> int:
        """Create an experiment and record how it was produced.

        get_or_create returns the newest experiment of this name instead of adding another,
        which is what one long-lived named benchmark needs.

        provenance overrides the captured values. Give it only when importing a run that
        happened elsewhere: the captured commit would then describe the import rather than
        the work, which is worse than no commit at all.
        """
        if get_or_create:
            row = self.conn.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?"
                " ORDER BY experiment_id DESC LIMIT 1",
                (name,),
            ).fetchone()
            if row is not None:
                if tags:
                    self.log_tags("experiment", row["experiment_id"], tags)
                return int(row["experiment_id"])

        prov = {**_provenance(), **(provenance or {})}
        cursor = self.conn.execute(
            "INSERT INTO experiments (name, description, note, git_commit, git_dirty,"
            " argv, python) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                description,
                note,
                prov.get("git_commit"),
                prov.get("git_dirty"),
                prov.get("argv"),
                prov.get("python"),
            ),
        )
        experiment_id = int(cursor.lastrowid)
        if tags:
            self.log_tags("experiment", experiment_id, tags)
        self.conn.commit()
        return experiment_id

    def start_run(
        self,
        experiment: int | Mapping[str, Any],
        name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        note: str | None = None,
    ) -> int:
        experiment_id = _entity_id(experiment)
        if self.get_experiment(experiment_id) is None:
            raise ValueError(f"experiment {experiment_id} does not exist")
        cursor = self.conn.execute(
            "INSERT INTO runs (experiment_id, name, params, status, note)"
            " VALUES (?, ?, ?, 'running', ?)",
            (experiment_id, name, serialize_params(params), note),
        )
        run_id = int(cursor.lastrowid)
        if tags:
            self.log_tags("run", run_id, tags)
        self.conn.commit()
        return run_id

    def run(
        self,
        experiment: int | Mapping[str, Any],
        name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        note: str | None = None,
    ) -> RunHandle:
        return RunHandle(self, self.start_run(experiment, name, params, tags, note))

    def end_run(self, run: int, success: bool = True, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, ended_at = datetime('now'), error = ? WHERE run_id = ?",
            ("completed" if success else "failed", error, _entity_id(run)),
        )
        self.conn.commit()

    def set_params(self, run: int, params: Mapping[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET params = ? WHERE run_id = ?",
            (serialize_params(params), _entity_id(run)),
        )
        self.conn.commit()

    def set_note(self, entity_type: str, entity_id: int, note: str) -> None:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}")
        table = "experiments" if entity_type == "experiment" else "runs"
        column = f"{entity_type}_id"
        cursor = self.conn.execute(
            f"UPDATE {table} SET note = ? WHERE {column} = ?",
            (note, _entity_id(entity_id)),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"{entity_type} {entity_id} does not exist")
        self.conn.commit()

    def log_metric(
        self,
        run: int,
        metric: str,
        value: float,
        dims: Mapping[str, Any] | None = None,
    ) -> None:
        self.log_metrics(run, {metric: value}, dims)

    def log_metrics(
        self,
        run: int,
        metrics: Mapping[str, float],
        dims: Mapping[str, Any] | None = None,
    ) -> None:
        run_id = self._require_run(run)
        key = dims_key(dims)
        self.conn.executemany(
            "INSERT INTO metrics (run_id, metric, dims, value) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(run_id, metric, dims) DO UPDATE SET value = excluded.value",
            [(run_id, name, key, float(value)) for name, value in metrics.items()],
        )
        self.conn.commit()

    def log_predictions(
        self,
        run: int,
        predictions: Sequence[float],
        actuals: Sequence[float] | None = None,
        dims: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        replace: bool = True,
    ) -> None:
        """Store rows keyed by dims. Computes no metrics; use the scoring module.

        dims is either one mapping shared by every row, or one mapping per row. Passing
        neither stores rows that cannot be addressed later, so it warrants a reason.
        """
        run_id = self._require_run(run)
        predictions = list(predictions)
        if actuals is not None:
            actuals = list(actuals)
            if len(actuals) != len(predictions):
                raise ValueError("predictions and actuals must have the same length")
        if isinstance(dims, Mapping) or dims is None:
            keys = [dims_key(dims)] * len(predictions)
        else:
            dims = list(dims)
            if len(dims) != len(predictions):
                raise ValueError("dims must have the same length as predictions")
            keys = [dims_key(d) for d in dims]

        if replace:
            self.conn.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
        rows = [
            (
                run_id,
                keys[i],
                float(predictions[i]),
                None if actuals is None else float(actuals[i]),
            )
            for i in range(len(predictions))
        ]
        self.conn.executemany(
            "INSERT INTO predictions (run_id, dims, prediction, actual) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def log_artifact(
        self,
        data: bytes,
        kind: str,
        filename: str,
        run: int | None = None,
        experiment: int | None = None,
    ) -> int:
        """Attach a file. A run-scoped artifact belongs to its experiment too."""
        run_id = None if run is None else self._require_run(run)
        experiment_id = None if experiment is None else _entity_id(experiment)
        if run_id is not None and experiment_id is None:
            row = self.conn.execute(
                "SELECT experiment_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            experiment_id = int(row["experiment_id"])
        if run_id is None and experiment_id is None:
            raise ValueError("give a run or an experiment")
        if experiment_id is not None and self.get_experiment(experiment_id) is None:
            raise ValueError(f"experiment {experiment_id} does not exist")
        cursor = self.conn.execute(
            "INSERT INTO artifacts (experiment_id, run_id, kind, filename, data)"
            " VALUES (?, ?, ?, ?, ?)",
            (experiment_id, run_id, kind, filename, data),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def log_file(
        self,
        path: str,
        kind: str | None = None,
        run: int | None = None,
        experiment: int | None = None,
    ) -> int:
        with open(path, "rb") as stream:
            data = stream.read()
        name = os.path.basename(path)
        return self.log_artifact(
            data,
            kind or (os.path.splitext(name)[1].lstrip(".") or "file"),
            name,
            run=run,
            experiment=experiment,
        )

    def log_tag(self, entity_type: str, entity_id: int, name: str, value: str = "") -> None:
        self.log_tags(entity_type, entity_id, {name: value})

    def log_tags(self, entity_type: str, entity_id: int, tags: Mapping[str, Any]) -> None:
        """Tags are unique per entity and name, so re-tagging updates instead of duplicating.

        Duplicates used to accumulate silently and then drop the entity from tag filters.
        """
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}")
        entity_id = _entity_id(entity_id)
        table = "experiments" if entity_type == "experiment" else "runs"
        column = f"{entity_type}_id"
        exists = self.conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ?", (entity_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"{entity_type} {entity_id} does not exist")
        self.conn.executemany(
            "INSERT INTO tags (entity_type, entity_id, name, value) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(entity_type, entity_id, name) DO UPDATE SET value = excluded.value",
            [(entity_type, entity_id, k, "" if v is None else str(v)) for k, v in tags.items()],
        )
        self.conn.commit()

    def delete_tag(self, entity_type: str, entity_id: int, name: str) -> int:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}")
        cursor = self.conn.execute(
            "DELETE FROM tags WHERE entity_type = ? AND entity_id = ? AND name = ?",
            (entity_type, _entity_id(entity_id), name),
        )
        self.conn.commit()
        return cursor.rowcount

    def delete_experiment(self, experiment: int) -> None:
        """Cascades to runs, metrics, predictions, and artifacts.

        Without ON DELETE CASCADE this raised IntegrityError for every experiment that had
        a run, which is every real one.
        """
        experiment_id = _entity_id(experiment)
        cursor = self.conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"experiment {experiment_id} does not exist")
        self.conn.execute(
            "DELETE FROM tags WHERE entity_type = 'experiment' AND entity_id = ?",
            (experiment_id,),
        )
        self.conn.commit()

    def _require_run(self, run: int) -> int:
        run_id = _entity_id(run)
        row = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"run {run_id} does not exist")
        return run_id

    # ----- reads -----

    RUN_ORDER_COLUMNS = ("run_id", "name", "status", "started_at", "ended_at")

    @staticmethod
    def _dims_filter(column: str, dims: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
        """Match rows whose dims include every given key and value.

        Subset semantics, so dims={"h": 6} selects everything at lead 6 regardless of what
        else keys the row. The json path is bound as a value, not interpolated.
        """
        if not dims:
            return "", []
        clauses, params = [], []
        for key, value in sorted(dims.items()):
            clauses.append(f"json_extract({column}, ?) = ?")
            params.extend([f"$.{key}", value])
        return " AND " + " AND ".join(clauses), params

    @staticmethod
    def _tags_filter(
        entity_type: str, id_column: str, tags: Mapping[str, Any] | None
    ) -> tuple[str, list[Any]]:
        """One EXISTS per tag.

        The previous version ANDed conditions against a single joined alias, so no row
        could satisfy two tags and any filter of two or more returned nothing.
        """
        if not tags:
            return "", []
        clauses, params = [], []
        for name, value in sorted(tags.items()):
            clauses.append(
                "EXISTS (SELECT 1 FROM tags t WHERE t.entity_type = ?"
                f" AND t.entity_id = {id_column} AND t.name = ? AND t.value = ?)"
            )
            params.extend([entity_type, name, "" if value is None else str(value)])
        return " AND " + " AND ".join(clauses), params

    def _expand_dims(self, row: sqlite3.Row, base: Sequence[str]) -> dict[str, Any]:
        out = {key: row[key] for key in base}
        parsed = json.loads(row["dims"])
        out["dims"] = parsed
        for key, value in parsed.items():
            if key not in out:
                out[key] = value
        return out

    def get_experiment(self, experiment: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (_entity_id(experiment),),
        ).fetchone()
        return None if row is None else dict(row)

    def experiments(
        self, name: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT e.*, COUNT(r.run_id) AS runs FROM experiments e"
            " LEFT JOIN runs r ON r.experiment_id = e.experiment_id"
        )
        params: list[Any] = []
        if name:
            query += " WHERE e.name LIKE ?"
            params.append(f"%{name}%")
        query += " GROUP BY e.experiment_id ORDER BY e.experiment_id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        return [dict(row) for row in self.conn.execute(query, params)]

    def latest_experiment(self, name: str | None = None) -> dict[str, Any] | None:
        """The newest experiment, optionally of one name.

        Replaces the max(experiment_id) query that every consumer had to write itself.
        """
        found = self.experiments(name=name, limit=1)
        return found[0] if found else None

    def get_run(self, run: int) -> dict[str, Any] | None:
        found = self.runs(run_id=_entity_id(run))
        return found[0] if found else None

    def runs(
        self,
        experiment: int | Mapping[str, Any] | None = None,
        name: str | None = None,
        tags: Mapping[str, Any] | None = None,
        status: str | None = None,
        since: str | None = None,
        run_id: int | None = None,
        order_by: str = "run_id",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Runs, across experiments when experiment is None.

        Every comparison used to be scoped to one experiment, but a model appears once per
        experiment, so the interesting axis is between them.
        """
        column, _, direction = order_by.partition(" ")
        if column not in self.RUN_ORDER_COLUMNS:
            raise ValueError(f"order_by must name one of {self.RUN_ORDER_COLUMNS}")
        direction = "DESC" if direction.lower() == "desc" else "ASC"

        query = "SELECT r.* FROM runs r WHERE 1 = 1"
        params: list[Any] = []
        if experiment is not None:
            query += " AND r.experiment_id = ?"
            params.append(_entity_id(experiment))
        if run_id is not None:
            query += " AND r.run_id = ?"
            params.append(int(run_id))
        if name:
            query += " AND r.name = ?"
            params.append(name)
        if status:
            query += " AND r.status = ?"
            params.append(status)
        if since:
            query += " AND r.started_at >= ?"
            params.append(since)
        clause, tag_params = self._tags_filter("run", "r.run_id", tags)
        query += clause
        params.extend(tag_params)
        query += f" ORDER BY r.{column} {direction}"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))

        rows = [dict(row) for row in self.conn.execute(query, params)]
        if not rows:
            return []
        for row in rows:
            row["params"] = json.loads(row["params"]) if row["params"] else {}
            row["tags"] = {}
        by_id = {row["run_id"]: row for row in rows}
        placeholders = ",".join("?" * len(by_id))
        for tag in self.conn.execute(
            "SELECT entity_id, name, value FROM tags WHERE entity_type = 'run'"
            f" AND entity_id IN ({placeholders})",
            list(by_id),
        ):
            by_id[tag["entity_id"]]["tags"][tag["name"]] = tag["value"]
        return rows

    def metrics(
        self,
        experiment: int | Mapping[str, Any] | None = None,
        runs: int | Sequence[int] | None = None,
        metric: str | None = None,
        dims: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Tidy metric rows with dims expanded into columns.

        A dim whose name collides with a base column stays available under the dims key.
        """
        query = (
            "SELECT r.experiment_id, m.run_id, r.name AS run_name, m.metric, m.dims,"
            " m.value FROM metrics m JOIN runs r ON r.run_id = m.run_id WHERE 1 = 1"
        )
        params: list[Any] = []
        if experiment is not None:
            query += " AND r.experiment_id = ?"
            params.append(_entity_id(experiment))
        if runs is not None:
            ids = [runs] if isinstance(runs, int) else list(runs)
            query += f" AND m.run_id IN ({','.join('?' * len(ids))})"
            params.extend(int(i) for i in ids)
        if metric:
            query += " AND m.metric = ?"
            params.append(metric)
        clause, dim_params = self._dims_filter("m.dims", dims)
        query += clause
        params.extend(dim_params)
        query += " ORDER BY m.run_id, m.metric, m.dims"
        base = ("experiment_id", "run_id", "run_name", "metric", "value")
        return [self._expand_dims(row, base) for row in self.conn.execute(query, params)]

    def predictions(
        self,
        runs: int | Sequence[int] | None = None,
        experiment: int | Mapping[str, Any] | None = None,
        dims: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT r.experiment_id, p.run_id, r.name AS run_name, p.dims,"
            " p.prediction, p.actual FROM predictions p"
            " JOIN runs r ON r.run_id = p.run_id WHERE 1 = 1"
        )
        params: list[Any] = []
        if experiment is not None:
            query += " AND r.experiment_id = ?"
            params.append(_entity_id(experiment))
        if runs is not None:
            ids = [runs] if isinstance(runs, int) else list(runs)
            query += f" AND p.run_id IN ({','.join('?' * len(ids))})"
            params.extend(int(i) for i in ids)
        clause, dim_params = self._dims_filter("p.dims", dims)
        query += clause
        params.extend(dim_params)
        query += " ORDER BY p.run_id, p.prediction_id"
        base = ("experiment_id", "run_id", "run_name", "prediction", "actual")
        return [self._expand_dims(row, base) for row in self.conn.execute(query, params)]

    def tags(self, entity_type: str, entity_id: int) -> dict[str, str]:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}")
        return {
            row["name"]: row["value"]
            for row in self.conn.execute(
                "SELECT name, value FROM tags WHERE entity_type = ? AND entity_id = ?"
                " ORDER BY name",
                (entity_type, _entity_id(entity_id)),
            )
        }

    def best(
        self,
        experiment: int | Mapping[str, Any] | None,
        metric: str,
        dims: Mapping[str, Any] | None = None,
        maximize: bool = False,
        tags: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """The run with the lowest value of metric, or the highest when maximize is set.

        Minimizing is the default because the usual metric is an error. The old default
        disagreed with its own command line flag, which answered with the worst run.
        """
        candidates = self.metrics(experiment=experiment, metric=metric, dims=dims)
        if tags is not None:
            allowed = {row["run_id"] for row in self.runs(experiment=experiment, tags=tags)}
            candidates = [row for row in candidates if row["run_id"] in allowed]
        if not candidates:
            return None
        winner = (max if maximize else min)(candidates, key=lambda row: row["value"])
        run = self.get_run(winner["run_id"])
        if run is None:
            return None
        run["metric"] = metric
        run["value"] = winner["value"]
        run["dims"] = winner["dims"]
        return run

    def compare(
        self,
        run_ids: Sequence[int],
        metrics: Sequence[str] | None = None,
        dims: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """One row per metric and dims, one column per run, with a delta for two runs.

        This is the pivot every consumer retyped by hand to decide keep or revert.
        """
        ids = [_entity_id(r) for r in run_ids]
        if len(ids) < 2:
            raise ValueError("compare needs at least 2 runs")
        rows = self.metrics(runs=ids, dims=dims)
        if metrics:
            wanted = set(metrics)
            rows = [row for row in rows if row["metric"] in wanted]

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["metric"], dims_key(row["dims"]))
            entry = grouped.setdefault(key, {"metric": row["metric"], "dims": row["dims"]})
            entry[str(row["run_id"])] = row["value"]

        out = []
        for key in sorted(grouped):
            entry = grouped[key]
            if len(ids) == 2:
                first, second = entry.get(str(ids[0])), entry.get(str(ids[1]))
                entry["delta"] = None if first is None or second is None else second - first
            out.append(entry)
        return out

    def audit(
        self, run: int, metric: str, dims: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Recompute a stored metric from the prediction rows that support it.

        This is the reason prediction rows are kept in the same file as the metrics: a
        published number should be checkable, not merely recorded.
        """
        from . import scoring

        if metric not in scoring.SCORERS:
            raise ValueError(f"cannot recompute {metric}; known: {sorted(scoring.SCORERS)}")
        stored = self.metrics(runs=_entity_id(run), metric=metric, dims=dims)
        exact = [row for row in stored if not dims or row["dims"] == dict(dims)]
        rows = [
            row
            for row in self.predictions(runs=_entity_id(run), dims=dims)
            if row["actual"] is not None
        ]
        if not rows:
            return None
        recomputed = scoring.SCORERS[metric](
            [row["prediction"] for row in rows], [row["actual"] for row in rows]
        )
        stored_value = exact[0]["value"] if exact else None
        return {
            "run_id": _entity_id(run),
            "metric": metric,
            "dims": dict(dims or {}),
            "rows": len(rows),
            "stored": stored_value,
            "recomputed": recomputed,
            "agrees": stored_value is not None
            and abs(stored_value - recomputed) <= 1e-9 * max(1.0, abs(stored_value)),
        }

    def artifacts(
        self, experiment: int | None = None, run: int | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT artifact_id, experiment_id, run_id, kind, filename,"
            " LENGTH(data) AS bytes, created_at FROM artifacts WHERE 1 = 1"
        )
        params: list[Any] = []
        if experiment is not None:
            query += " AND experiment_id = ?"
            params.append(_entity_id(experiment))
        if run is not None:
            query += " AND run_id = ?"
            params.append(_entity_id(run))
        query += " ORDER BY artifact_id"
        return [dict(row) for row in self.conn.execute(query, params)]

    def artifact_data(self, artifact_id: int) -> bytes:
        row = self.conn.execute(
            "SELECT data FROM artifacts WHERE artifact_id = ?", (int(artifact_id),)
        ).fetchone()
        if row is None:
            raise ValueError(f"artifact {artifact_id} does not exist")
        return row["data"]

    def snapshot(
        self,
        experiment: int | Mapping[str, Any],
        directory: str,
        predictions: bool = False,
    ) -> str:
        """Write the experiment as deterministic files meant to be committed.

        The database is a working file that .gitignore excludes, so it cannot be the
        citation for a published number. These files can: sorted rows, stable columns, and
        the commit that produced them in experiment.json.

        Prediction rows are omitted by default. That keeps the committed record small and
        leaves metrics auditable from the database rather than from the snapshot.
        """
        experiment_id = _entity_id(experiment)
        record = self.get_experiment(experiment_id)
        if record is None:
            raise ValueError(f"experiment {experiment_id} does not exist")
        os.makedirs(directory, exist_ok=True)

        record["tags"] = self.tags("experiment", experiment_id)
        record["artifacts"] = [
            {k: v for k, v in row.items() if k != "data"}
            for row in self.artifacts(experiment=experiment_id)
        ]
        with open(os.path.join(directory, "experiment.json"), "w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")

        run_rows = self.runs(experiment=experiment_id, order_by="run_id")
        _write_csv(
            os.path.join(directory, "runs.csv"),
            [
                "run_id",
                "name",
                "status",
                "started_at",
                "ended_at",
                "note",
                "error",
                "params",
                "tags",
            ],
            [
                {
                    **{
                        k: row[k]
                        for k in (
                            "run_id",
                            "name",
                            "status",
                            "started_at",
                            "ended_at",
                            "note",
                            "error",
                        )
                    },
                    "params": json.dumps(row["params"], sort_keys=True),
                    "tags": json.dumps(row["tags"], sort_keys=True),
                }
                for row in run_rows
            ],
        )

        _write_csv(
            os.path.join(directory, "metrics.csv"),
            ["run_id", "run_name", "metric", "dims", "value"],
            [
                {
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "metric": row["metric"],
                    "dims": dims_key(row["dims"]),
                    "value": row["value"],
                }
                for row in sorted(
                    self.metrics(experiment=experiment_id),
                    key=lambda r: (r["run_id"], r["metric"], dims_key(r["dims"])),
                )
            ],
        )

        if predictions:
            _write_csv(
                os.path.join(directory, "predictions.csv"),
                ["run_id", "run_name", "dims", "prediction", "actual"],
                [
                    {
                        "run_id": row["run_id"],
                        "run_name": row["run_name"],
                        "dims": dims_key(row["dims"]),
                        "prediction": row["prediction"],
                        "actual": row["actual"],
                    }
                    for row in self.predictions(experiment=experiment_id)
                ],
            )
        return directory


def _write_csv(path: str, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    import csv

    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})
