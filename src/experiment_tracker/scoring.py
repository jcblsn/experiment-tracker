"""Scores as plain functions.

These are deliberately not called by the write path. Logging predictions used to compute
metrics as a side effect, which put a value pooled over every dimension beside the
per-dimension values under an indistinguishable name. Compute a score, then log it.
"""

import math
from collections.abc import Sequence


def _pairs(predictions: Sequence[float], actuals: Sequence[float]) -> list[tuple[float, float]]:
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length")
    if not predictions:
        raise ValueError("cannot score an empty sequence")
    return list(zip(predictions, actuals, strict=True))


def rmse(predictions: Sequence[float], actuals: Sequence[float]) -> float:
    pairs = _pairs(predictions, actuals)
    return math.sqrt(sum((p - a) ** 2 for p, a in pairs) / len(pairs))


def mae(predictions: Sequence[float], actuals: Sequence[float]) -> float:
    pairs = _pairs(predictions, actuals)
    return sum(abs(p - a) for p, a in pairs) / len(pairs)


def mape(predictions: Sequence[float], actuals: Sequence[float]) -> float:
    """Mean absolute percentage error over the rows whose actual is non-zero.

    Raises when every actual is zero, rather than returning 0, which would read as a
    perfect score.
    """
    pairs = [(p, a) for p, a in _pairs(predictions, actuals) if a != 0]
    if not pairs:
        raise ValueError("mape needs at least one non-zero actual")
    return sum(abs((a - p) / a) for p, a in pairs) / len(pairs)


SCORERS = {"rmse": rmse, "mae": mae, "mape": mape}


def score(
    predictions: Sequence[float],
    actuals: Sequence[float],
    metrics: Sequence[str] = ("rmse", "mae"),
) -> dict[str, float]:
    """Named scores, shaped to pass straight to log_metrics."""
    unknown = set(metrics) - set(SCORERS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    return {name: SCORERS[name](predictions, actuals) for name in metrics}
