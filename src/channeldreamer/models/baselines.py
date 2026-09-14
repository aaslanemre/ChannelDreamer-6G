"""Reactive and Markov baselines.

Both expose ``predict_scores(windows) -> (M, 64)`` so they plug into
:mod:`channeldreamer.eval.metrics` and :mod:`channeldreamer.envs` exactly like any learned model.
"""

from __future__ import annotations

import numpy as np

from ..data.deepsense import optimal_beam
from ..data.sequences import WindowedDataset


class ReactiveBaseline:
    """Hold the last observed optimal beam.

    The top-1 prediction is ``argmax(history[-1])``.  For top-k metrics the beams are ranked
    by the last observed power vector (``rank_by_power=True``, default) - i.e. "keep the
    current beam, fall back to its best-measured neighbours" - or as a pure one-hot
    (``rank_by_power=False``) where top-k collapses to top-1.
    """

    def __init__(self, rank_by_power: bool = True):
        self.rank_by_power = rank_by_power

    def fit(self, windows: WindowedDataset) -> ReactiveBaseline:  # nothing to learn
        return self

    def predict_scores(self, windows: WindowedDataset) -> np.ndarray:
        last = windows.histories[:, -1, :]
        if self.rank_by_power:
            return np.asarray(last, dtype=np.float64)
        scores = np.zeros_like(last, dtype=np.float64)
        scores[np.arange(len(windows)), optimal_beam(last)] = 1.0
        return scores


class MarkovBaseline:
    """Empirical first-order transition model ``P(beam_{t+k} | beam_t)``.

    Fitted on training windows using (last-history optimal beam -> target optimal beam)
    counts with additive (Laplace) smoothing.  Scores for a window are the row of the
    transition matrix for its current beam.
    """

    def __init__(self, n_beams: int = 64, smoothing: float = 0.1):
        self.n_beams = n_beams
        self.smoothing = smoothing
        self.transition: np.ndarray | None = None
        self.counts: np.ndarray | None = None

    def fit(self, windows: WindowedDataset) -> MarkovBaseline:
        cur = windows.last_beam()
        nxt = windows.target_beam
        counts = np.zeros((self.n_beams, self.n_beams), dtype=np.float64)
        np.add.at(counts, (cur, nxt), 1.0)
        self.counts = counts
        smoothed = counts + self.smoothing
        # rows never observed: bias towards staying (identity) so it degrades to reactive
        unseen = counts.sum(axis=1) == 0
        smoothed[unseen] += np.eye(self.n_beams)[unseen]
        self.transition = smoothed / smoothed.sum(axis=1, keepdims=True)
        return self

    def predict_scores(self, windows: WindowedDataset) -> np.ndarray:
        if self.transition is None:
            raise RuntimeError("MarkovBaseline.fit() must be called before predict_scores()")
        return self.transition[windows.last_beam()]


class OracleBaseline:
    """Upper bound: scores equal to the target power vector (perfect foresight)."""

    def fit(self, windows: WindowedDataset) -> OracleBaseline:
        return self

    def predict_scores(self, windows: WindowedDataset) -> np.ndarray:
        return np.asarray(windows.target_power, dtype=np.float64)
