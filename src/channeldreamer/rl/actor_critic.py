"""DreamerV3-style actor-critic trained on imagined rollouts.  **STUB (Phase 4).**

Intended loop
-------------
1. Sample history windows from the offline dataset; encode with the world model to obtain
   a start latent ``s_0 = (h_0, z_0)`` for each window.
2. Imagine ``horizon`` steps with the RSSM prior.  Because dynamics are exogenous, the
   imagined latent trajectory does not depend on the policy; the actor only chooses a beam
   ``a_t ~ pi(a | s_t)`` at each imagined step.
3. Reward at each imagined step: ``r̂_t(a_t) - c * 1[a_t != a_{t-1}]`` where ``r̂_t(.)`` is
   the predicted 64-way reward table and ``c`` the switching penalty (exact, not learned).
4. Critic ``v(s_t)`` regresses lambda-returns (symlog, twohot) with a slow-EMA target.
5. Actor maximises return via REINFORCE with the critic as baseline (discrete actions),
   plus entropy regularisation; return normalisation by percentile (DreamerV3).
6. Evaluation uses the real reward table (no model) through
   :class:`channeldreamer.envs.OfflineBeamEnv`, decomposed by regime.

The actor also exposes ``predict_scores(windows)`` (the policy logits) so it is evaluated
with the same metric code as the baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActorCriticConfig:
    imagination_horizon: int = 15
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy_scale: float = 3e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    critic_ema_decay: float = 0.98
    batch_size: int = 16
    mixed_precision: bool = True


class Actor:
    """Categorical policy over 64 beams from the latent state.  Not implemented in Phase 1."""

    def __init__(self, latent_dim: int, n_actions: int = 64, hidden_dim: int = 512):
        raise NotImplementedError

    def forward(self, state: Any) -> Any:
        """Return logits ``(B, n_actions)``."""
        raise NotImplementedError


class Critic:
    """State-value head with twohot symlog output.  Not implemented in Phase 1."""

    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        raise NotImplementedError

    def forward(self, state: Any) -> Any:
        raise NotImplementedError


class ImaginedActorCritic:
    """Trainer tying world model, actor and critic together.  Not implemented in Phase 1."""

    def __init__(self, world_model: Any, config: ActorCriticConfig, switching_penalty: float = 0.0):
        raise NotImplementedError

    def imagine_rollout(self, start_state: Any) -> Any:
        raise NotImplementedError

    def lambda_returns(self, rewards: Any, values: Any, continues: Any) -> Any:
        raise NotImplementedError

    def train_step(self, batch: Any) -> dict[str, float]:
        raise NotImplementedError

    def predict_scores(self, windows: Any) -> Any:
        """Policy logits ``(M, 64)`` on real history windows - shared evaluation interface."""
        raise NotImplementedError
