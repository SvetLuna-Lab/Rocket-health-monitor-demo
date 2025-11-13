from typing import Iterable, List, Sequence, Tuple

import math
import statistics

import numpy as np
from sklearn.ensemble import IsolationForest


def zscore_anomaly(series: Sequence[float], threshold: float = 3.0) -> List[int]:
    """
    Point anomaly detection by Z-score on a single series.
    Returns 0/1 flags for each point.
    """
    if not series:
        return []
    mu = statistics.fmean(series)
    sigma = statistics.pstdev(series) or 1e-9
    flags = []
    for x in series:
        z = abs((x - mu) / sigma)
        flags.append(1 if z >= threshold else 0)
    return flags


def ewma_anomaly(series: Sequence[float], alpha: float = 0.3, k_sigma: float = 3.0) -> List[int]:
    """
    Simple EWMA control-style signal: flag when deviation from EWMA exceeds k*sigma of residuals.
    """
    if not series:
        return []
    ewma_vals: List[float] = []
    prev = series[0]
    for x in series:
        prev = alpha * x + (1 - alpha) * prev
        ewma_vals.append(prev)
    residuals = [abs(x - m) for x, m in zip(series, ewma_vals)]
    sigma = statistics.pstdev(residuals) or 1e-9
    return [1 if abs(x - m) >= k_sigma * sigma else 0 for x, m in zip(series, ewma_vals)]


def isolation_forest_flags(X: Iterable[Iterable[float]], contamination: float = 0.1, random_state: int = 42) -> List[int]:
    """
    IsolationForest over multivariate features.
    Returns 1 for anomaly, 0 otherwise.
    """
    arr = np.array(list(X), dtype=float)
    if arr.size == 0:
        return []
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = clf.fit_predict(arr)  # -1 = anomaly, 1 = normal
    return [1 if v == -1 else 0 for v in y_pred]
