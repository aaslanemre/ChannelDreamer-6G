"""Offline beam-selection MDP built on DeepSense 6G beam-power vectors.

Why this offline problem is *fully observed* and why plain sequence prediction yields a
valid dynamics model
================================================================================

Most offline RL suffers from two problems: (1) the reward of actions **not** taken by the
logging policy is unknown (we only see the logged action's reward), and (2) the agent's
actions influence future states, so a dynamics model must be learned from (s, a, s')
triples and evaluated off-policy under distribution shift.

Beam management on DeepSense 6G is special on both counts:

1. **Every action's reward is observed at every step.**  Each sample records the received
   power for *all* 64 codebook beams (a full beam sweep).  Choosing beam ``a`` at step ``t``
   would have yielded ``power[t, a]``, for *any* ``a``.  There is no counterfactual to
   estimate: the reward table ``R[t, a]`` is a measured ``(M, 64)`` matrix.

2. **Transitions are exogenous.**  The beam the base station selects does not change where
   the vehicle drives, hence does not change the next power vector.  The state process
   ``power[t] -> power[t+1]`` is an autonomous stochastic process, independent of the
   action.  Therefore a world model trained by *plain sequence prediction* of the power
   vectors (no action conditioning needed for the dynamics part) is a valid dynamics
   model for the MDP, and imagined rollouts do not suffer from action-induced covariate
   shift.

The only action-dependent part of the return is an optional **switching penalty** that
charges a cost for changing the serving beam (modelling beam-switch overhead / signalling
cost).  This is where a proactive policy can trade a small immediate power loss for
avoiding a costly late switch at a transition - the setting the thesis studies.

Observations, actions, rewards
------------------------------
* observation: history window ``(H, 64)`` (plus optional side modalities in later phases)
* action: integer beam index in ``[0, 64)``
* reward: ``r_t = R[t, a_t] - c * 1[a_t != a_{t-1}]`` with ``R`` in dB (default) or linear
  power and ``c = switching_penalty``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.sequences import WindowedDataset


@dataclass
class PolicyReward:
    per_step: np.ndarray  # (M,) net reward
    beam_reward: np.ndarray  # (M,) reward of the chosen beam before penalty
    switch_cost: np.ndarray  # (M,) penalty paid
    n_switches: int
    mean: float
    total: float
    regret: float  # mean (optimal reward - chosen reward), before penalty


class OfflineBeamEnv:
    """Fully-observed offline beam MDP over a :class:`WindowedDataset`.

    Parameters
    ----------
    windows
        Shared windows; the reward for a window is measured on its *target* power vector.
    reward_unit
        ``"db"`` (10 log10 of power) or ``"linear"``.
    switching_penalty
        Cost ``c`` charged whenever the chosen beam differs from the previous one
        (in the same units as the reward).
    db_floor
        Power floor used before the log to keep dB rewards finite.
    """

    def __init__(
        self,
        windows: WindowedDataset,
        *,
        reward_unit: str = "db",
        switching_penalty: float = 0.0,
        db_floor: float = 1e-12,
    ):
        if reward_unit not in ("db", "linear"):
            raise ValueError("reward_unit must be 'db' or 'linear'")
        self.windows = windows
        self.reward_unit = reward_unit
        self.switching_penalty = float(switching_penalty)
        self.db_floor = db_floor

    # ------------------------------------------------------------------ rewards
    @property
    def n_actions(self) -> int:
        return self.windows.n_beams

    def reward_table(self, unit: str | None = None) -> np.ndarray:
        """``(M, 64)`` reward of *every* candidate beam at every step (measured, not estimated)."""
        unit = unit or self.reward_unit
        p = np.asarray(self.windows.target_power, dtype=np.float64)
        if unit == "linear":
            return p.copy()
        return 10.0 * np.log10(np.maximum(p, self.db_floor))

    def optimal_actions(self) -> np.ndarray:
        return np.argmax(self.reward_table(), axis=1)

    def switching_penalty_cost(self, prev_beam: np.ndarray | int, action: np.ndarray | int) -> np.ndarray:
        """Cost for changing beams: ``c * 1[action != prev_beam]`` (broadcasts)."""
        prev = np.asarray(prev_beam)
        act = np.asarray(action)
        return self.switching_penalty * (prev != act).astype(np.float64)

    def evaluate_policy_reward(
        self, actions: np.ndarray, *, initial_beam: np.ndarray | None = None
    ) -> PolicyReward:
        """Net reward of a policy's chosen actions on every window.

        ``initial_beam`` (``(M,)``) is the beam in use *before* the action at each window;
        by default it is the optimal beam of the last history step (i.e. the beam a
        reactive system would currently be serving), so the penalty is paid exactly when
        the policy *changes* beam relative to the current one.
        """
        actions = np.asarray(actions, dtype=np.int64)
        m = len(self.windows)
        if actions.shape != (m,):
            raise ValueError(f"actions must be ({m},), got {actions.shape}")
        if (actions < 0).any() or (actions >= self.n_actions).any():
            raise ValueError("actions out of range")
        prev = self.windows.last_beam() if initial_beam is None else np.asarray(initial_beam)
        table = self.reward_table()
        beam_reward = table[np.arange(m), actions]
        cost = self.switching_penalty_cost(prev, actions)
        net = beam_reward - cost
        opt = table.max(axis=1)
        return PolicyReward(
            per_step=net,
            beam_reward=beam_reward,
            switch_cost=cost,
            n_switches=int((prev != actions).sum()),
            mean=float(net.mean()) if m else float("nan"),
            total=float(net.sum()),
            regret=float((opt - beam_reward).mean()) if m else float("nan"),
        )

    # ------------------------------------------------------------- gym-ish API
    def observation(self, i: int) -> np.ndarray:
        """History window ``(H, 64)`` for step ``i``; the exogenous next state is window i+1."""
        return self.windows.histories[i]
