"""Beam-prediction metrics, decomposed by regime.

Any method exposing ``predict_scores(windows) -> (M, 64)`` can be evaluated.  Scores only
need to induce a ranking over beams (higher = better); the top-1 beam is the chosen beam.

Metrics
-------
* top-k accuracy (k = 1, 3, 5): is the optimal target beam among the k highest-scoring beams?
* power loss (dB): ``10 log10(P[optimal] / P[chosen])`` on the *target* power vector, i.e. the
  gap between the chosen beam's received power and the best achievable.  0 dB is perfect.

Every metric is reported for the **overall** set and separately for **stable** and
**transition** windows, because the thesis hypothesis lives in that decomposition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..data.sequences import WindowedDataset
from .regimes import REGIME_NAMES, STABLE, TRANSITION


@runtime_checkable
class ScorePredictor(Protocol):
    def predict_scores(self, windows: WindowedDataset) -> np.ndarray:  # (M, B)
        ...


def top_k_accuracy(scores: np.ndarray, target_beam: np.ndarray, k: int) -> float:
    if scores.shape[0] == 0:
        return float("nan")
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean(np.any(topk == target_beam[:, None], axis=1)))


def power_loss_db(scores: np.ndarray, target_power: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Per-window dB gap between optimal-beam power and chosen-beam (argmax score) power."""
    if scores.shape[0] == 0:
        return np.empty(0)
    chosen = np.argmax(scores, axis=1)
    p_chosen = np.maximum(target_power[np.arange(len(chosen)), chosen], floor)
    p_opt = np.maximum(target_power.max(axis=1), floor)
    return 10.0 * np.log10(p_opt / p_chosen)


@dataclass
class MetricResult:
    """Metrics for one method on one regime slice."""

    n: int
    top1: float
    top3: float
    top5: float
    power_loss_db_mean: float
    power_loss_db_median: float
    power_loss_db_p90: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _slice_metrics(scores: np.ndarray, windows: WindowedDataset, mask: np.ndarray) -> MetricResult:
    s = scores[mask]
    tb = windows.target_beam[mask]
    tp = windows.target_power[mask]
    loss = power_loss_db(s, tp)
    nan = float("nan")
    return MetricResult(
        n=int(mask.sum()),
        top1=top_k_accuracy(s, tb, 1),
        top3=top_k_accuracy(s, tb, 3),
        top5=top_k_accuracy(s, tb, 5),
        power_loss_db_mean=float(loss.mean()) if loss.size else nan,
        power_loss_db_median=float(np.median(loss)) if loss.size else nan,
        power_loss_db_p90=float(np.percentile(loss, 90)) if loss.size else nan,
    )


def compute_metrics(
    scores: np.ndarray, windows: WindowedDataset, regimes: np.ndarray
) -> dict[str, MetricResult]:
    """Return ``{"overall": ..., "stable": ..., "transition": ...}`` for given scores."""
    scores = np.asarray(scores)
    regimes = np.asarray(regimes)
    if scores.shape != (len(windows), windows.n_beams):
        raise ValueError(f"scores must be {(len(windows), windows.n_beams)}, got {scores.shape}")
    if regimes.shape != (len(windows),):
        raise ValueError("regimes must have one label per window")
    return {
        "overall": _slice_metrics(scores, windows, np.ones(len(windows), dtype=bool)),
        REGIME_NAMES[STABLE]: _slice_metrics(scores, windows, regimes == STABLE),
        REGIME_NAMES[TRANSITION]: _slice_metrics(scores, windows, regimes == TRANSITION),
    }


def evaluate_methods(
    methods: Mapping[str, ScorePredictor], windows: WindowedDataset, regimes: np.ndarray
) -> dict[str, dict[str, MetricResult]]:
    """Evaluate several ``predict_scores`` methods on the same windows/regime labels."""
    return {name: compute_metrics(m.predict_scores(windows), windows, regimes) for name, m in methods.items()}


def format_regime_table(
    results: Mapping[str, Mapping[str, MetricResult]],
    regimes: Iterable[str] = ("overall", "stable", "transition"),
) -> str:
    """Pretty-print a regime-decomposed table for one or more methods."""
    header = f"{'method':<18}{'regime':<12}{'n':>7}{'top-1':>8}{'top-3':>8}{'top-5':>8}{'loss dB':>10}{'med dB':>9}{'p90 dB':>9}"
    lines = [header, "-" * len(header)]
    for name, per_regime in results.items():
        for reg in regimes:
            r = per_regime[reg]
            lines.append(
                f"{name:<18}{reg:<12}{r.n:>7}{r.top1:>8.3f}{r.top3:>8.3f}{r.top5:>8.3f}"
                f"{r.power_loss_db_mean:>10.2f}{r.power_loss_db_median:>9.2f}{r.power_loss_db_p90:>9.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
