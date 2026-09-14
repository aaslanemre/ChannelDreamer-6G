"""Shared sequence windowing.

Every model and baseline in this project consumes windows produced by :func:`make_windows`,
so that all methods see *exactly* the same (history, target) pairs and comparisons are fair.

A window is::

    history  = power[t-H+1 : t+1]        shape (H, 64)
    target   = power[t + k]              shape (64,)     k = horizon >= 1

and is only emitted when every index from ``t-H+1`` to ``t+k`` lies inside a single segment
(an independent drive / pass-by event).  Windows never cross segment boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deepsense import optimal_beam


@dataclass
class WindowedDataset:
    """(M, H, 64) histories with aligned targets and bookkeeping."""

    histories: np.ndarray  # (M, H, B) power
    target_power: np.ndarray  # (M, B) power at t+k
    target_beam: np.ndarray  # (M,) argmax of target_power
    target_index: np.ndarray  # (M,) flat index of the target step in the source array
    last_index: np.ndarray  # (M,) flat index of the last history step (t)
    segment_ids: np.ndarray  # (M,) segment id of the window
    history: int
    horizon: int

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    @property
    def n_beams(self) -> int:
        return int(self.histories.shape[-1])

    def last_beam(self) -> np.ndarray:
        """Optimal beam at the last history step (the 'currently used' beam)."""
        return optimal_beam(self.histories[:, -1, :])

    def subset(self, mask: np.ndarray) -> WindowedDataset:
        mask = np.asarray(mask)
        return WindowedDataset(
            histories=self.histories[mask],
            target_power=self.target_power[mask],
            target_beam=self.target_beam[mask],
            target_index=self.target_index[mask],
            last_index=self.last_index[mask],
            segment_ids=self.segment_ids[mask],
            history=self.history,
            horizon=self.horizon,
        )


def make_windows(
    power: np.ndarray,
    segment_ids: np.ndarray | None,
    history: int,
    horizon: int = 1,
    stride: int = 1,
) -> WindowedDataset:
    """Turn a flat ``(N, B)`` power array into ``(M, H, B)`` history windows + targets.

    Parameters
    ----------
    power
        ``(N, B)`` beam-power array in temporal order.
    segment_ids
        ``(N,)`` integer id per step; windows never straddle a change of id.  ``None`` means
        one single segment.
    history
        H, number of past steps in each window (including the current step t).
    horizon
        k >= 1, prediction offset: the target is the step ``t + k``.
    stride
        Step between consecutive window end points within a segment.
    """
    power = np.asarray(power)
    if power.ndim != 2:
        raise ValueError(f"power must be (N, B), got {power.shape}")
    n = power.shape[0]
    if history < 1 or horizon < 1 or stride < 1:
        raise ValueError("history, horizon and stride must all be >= 1")
    seg = np.zeros(n, dtype=np.int64) if segment_ids is None else np.asarray(segment_ids)
    if seg.shape != (n,):
        raise ValueError(f"segment_ids must be ({n},), got {seg.shape}")

    hist_list, tgt_idx, last_idx, seg_list = [], [], [], []
    # iterate over contiguous runs of equal segment id
    boundaries = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for s, e in zip(starts, ends):
        # t ranges over last-history positions such that t-H+1 >= s and t+k <= e-1
        for t in range(s + history - 1, e - horizon, stride):
            hist_list.append(power[t - history + 1 : t + 1])
            tgt_idx.append(t + horizon)
            last_idx.append(t)
            seg_list.append(seg[s])

    if not hist_list:
        histories = np.empty((0, history, power.shape[1]), dtype=power.dtype)
        tgt = np.empty((0,), dtype=np.int64)
        return WindowedDataset(
            histories, np.empty((0, power.shape[1]), dtype=power.dtype), tgt, tgt,
            tgt.copy(), tgt.copy(), history, horizon,
        )
    histories = np.stack(hist_list)
    tgt_idx_arr = np.asarray(tgt_idx, dtype=np.int64)
    target_power = power[tgt_idx_arr]
    return WindowedDataset(
        histories=histories,
        target_power=target_power,
        target_beam=optimal_beam(target_power),
        target_index=tgt_idx_arr,
        last_index=np.asarray(last_idx, dtype=np.int64),
        segment_ids=np.asarray(seg_list, dtype=np.int64),
        history=history,
        horizon=horizon,
    )


def split_segments(
    segment_ids: np.ndarray, test_fraction: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Segment-level train/test split. Returns (train_segment_ids, test_segment_ids).

    Splitting by whole segments (rather than by sample) prevents leakage of near-duplicate
    consecutive frames between train and test.
    """
    uniq = np.unique(segment_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test = max(1, round(test_fraction * len(uniq))) if len(uniq) > 1 else 0
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])
