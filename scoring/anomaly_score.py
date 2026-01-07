# scoring/anomaly_score.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, Optional

import math


@dataclass(frozen=True)
class Thresholds:
    p95: float
    p99: float
    p999: float


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on already sorted list."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = int(p * n)
    if idx >= n:
        idx = n - 1
    if idx < 0:
        idx = 0
    return float(sorted_vals[idx])


def compute_thresholds(scores: Iterable[float]) -> Thresholds:
    vals = [float(x) for x in scores if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))]
    vals.sort()
    return Thresholds(
        p95=_percentile(vals, 0.95),
        p99=_percentile(vals, 0.99),
        p999=_percentile(vals, 0.999),
    )


def severity_from_score(score: float, th: Thresholds) -> str:
    """
    NONE  : score < p95
    LOW   : p95 <= score < p99
    MED   : p99 <= score < p999
    HIGH  : score >= p999
    """
    s = float(score)
    if s >= th.p999:
        return "HIGH"
    if s >= th.p99:
        return "MED"
    if s >= th.p95:
        return "LOW"
    return "NONE"


def alert_from_severity(sev: str, min_alert: str = "MED") -> bool:
    order = {"NONE": 0, "LOW": 1, "MED": 2, "HIGH": 3}
    return order.get(sev, 0) >= order.get(min_alert, 2)


def score_to_soc_decision(score: float, thresholds: Thresholds, min_alert: str = "MED") -> Tuple[bool, str]:
    sev = severity_from_score(score, thresholds)
    alert = alert_from_severity(sev, min_alert=min_alert)
    return alert, sev
