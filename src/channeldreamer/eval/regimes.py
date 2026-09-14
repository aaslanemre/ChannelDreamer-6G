"""STABLE / TRANSITION regime labelling.

This is the central evaluation mechanism of the thesis.  The hypothesis is that a world-model
policy only *meaningfully* beats reactive baselines **at transitions** (blockage, corner
turns, new dominant path), not during stable segments where "hold the current beam" is
already near-optimal.  Every metric is therefore reported decomposed by regime.

A step ``t`` is labelled TRANSITION when, on a *smoothed* power series,

  (a) ``|argmax(t) - argmax(t-1)|``  >  ``beam_jump_threshold``      (beam-index jump), or
  (b) ``max_k peak_dB(t-k) - peak_dB(t)``  >  ``power_drop_db_threshold``
      for ``k`` in ``1..drop_lookback``                                (local peak-power drop),

and STABLE otherwise.  Labels can be dilated by ``dilation`` steps on each side so that the
few steps surrounding an event (where prediction is hardest) count as transition too.
Smoothing / difference operations never cross segment boundaries.
"""

from __future__ import annotations

import numpy as np

from ..data.deepsense import optimal_beam

STABLE = 0
TRANSITION = 1
REGIME_NAMES = {STABLE: "stable", TRANSITION: "transition"}


def to_db(power: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), floor))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Causal-centred moving average along axis 0 with edge replication (window >= 1)."""
    if window <= 1:
        return x
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    xp = np.concatenate([np.repeat(x[:1], pad_l, axis=0), x, np.repeat(x[-1:], pad_r, axis=0)])
    kernel = np.ones(window) / window
    if x.ndim == 1:
        return np.convolve(xp, kernel, mode="valid")
    return np.stack([np.convolve(xp[:, j], kernel, mode="valid") for j in range(x.shape[1])], axis=1)


def label_regimes(
    power: np.ndarray,
    segment_ids: np.ndarray | None = None,
    *,
    beam_jump_threshold: int = 3,
    power_drop_db_threshold: float = 3.0,
    smoothing_window: int = 3,
    drop_lookback: int = 3,
    dilation: int = 1,
    power_is_db: bool = False,
) -> np.ndarray:
    """Label each time step STABLE (0) or TRANSITION (1).

    Parameters
    ----------
    power
        ``(N, B)`` beam-power series (linear unless ``power_is_db``).
    segment_ids
        ``(N,)`` segment ids; processing is done per contiguous segment.
    beam_jump_threshold
        Criterion (a): TRANSITION if ``|Δ optimal beam| > threshold``.
    power_drop_db_threshold
        Criterion (b): TRANSITION if the smoothed peak power drops by more than this many dB
        relative to any of the previous ``drop_lookback`` steps.
    smoothing_window
        Moving-average window (in steps) applied to the linear power series before both
        criteria are evaluated.  1 disables smoothing.
    dilation
        Number of steps on each side of a detected transition that are also labelled
        TRANSITION.
    """
    power = np.asarray(power, dtype=np.float64)
    n = power.shape[0]
    seg = np.zeros(n, dtype=np.int64) if segment_ids is None else np.asarray(segment_ids)
    labels = np.zeros(n, dtype=np.int64)
    lin = 10 ** (power / 10) if power_is_db else power

    boundaries = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for s, e in zip(starts, ends):
        p = _moving_average(lin[s:e], smoothing_window)
        beams = optimal_beam(p)
        peak_db = to_db(p.max(axis=1))
        flag = np.zeros(e - s, dtype=bool)
        # (a) beam-index jump
        flag[1:] |= np.abs(np.diff(beams)) > beam_jump_threshold
        # (b) local peak-power drop over a lookback
        for k in range(1, drop_lookback + 1):
            if k < e - s:
                flag[k:] |= (peak_db[:-k] - peak_db[k:]) > power_drop_db_threshold
        if dilation > 0 and flag.any():
            idx = np.flatnonzero(flag)
            for d in range(-dilation, dilation + 1):
                j = idx + d
                j = j[(j >= 0) & (j < e - s)]
                flag[j] = True
        labels[s:e] = flag.astype(np.int64)
    return labels


def regime_for_windows(step_labels: np.ndarray, target_index: np.ndarray) -> np.ndarray:
    """Map per-step regime labels onto windows via the index of each window's target step."""
    return np.asarray(step_labels)[np.asarray(target_index)]


def regime_summary(labels: np.ndarray) -> str:
    n = len(labels)
    n_tr = int(np.sum(labels == TRANSITION))
    return f"{n} steps: {n - n_tr} stable ({100 * (n - n_tr) / max(n, 1):.1f}%), {n_tr} transition ({100 * n_tr / max(n, 1):.1f}%)"
