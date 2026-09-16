"""DreamerV3-style actor-critic trained on imagined RSSM rollouts (Phase 4).

Loop
----
1. Encode real history windows with the (frozen) world model; every posterior state of the
   batch is a rollout start ``s_0 = (h_0, z_0)`` with a known "current" beam ``a_{-1}`` (the
   measured optimal beam of that real step, exactly what a reactive system would be serving
   and what :meth:`OfflineBeamEnv.evaluate_policy_reward` uses as ``initial_beam``).
2. Imagine ``H = imagination_horizon`` steps with the RSSM prior.  At each imagined state
   ``s_j`` the actor draws ``a_j ~ pi(. | s_j)`` as a **straight-through one-hot** over the 64
   beams (forward: a sampled one-hot; backward: the softmax gradient), then
   ``s_{j+1} = img_step(s_j)``.  Dynamics are exogenous, so the latent trajectory must not
   depend on ``a_j``: the world model is trained with ``RSSMConfig.action_dim = 0``.  **Do not
   condition the RSSM on the logged behaviour action** (``actions_from_obs``, the one-hot of
   the previous optimal beam): that action is a near-perfect predictor of the next power
   vector, the GRU learns to read it as an observation, and in imagination the actor then
   games the model - whichever beam it picks, the model predicts that beam is good, and the
   policy collapses to one state-independent beam with ~20 dB real regret while the model's
   own one-step prediction is near-perfect (verified on synthetic data).  With
   ``action_dim = reward_dim`` the sampled one-hot is still fed to the GRU (kept only for the
   ablation).
3. Reward of the imagined step, mirroring :mod:`channeldreamer.envs.offline_beam_env`
   (``r = R[a] - c * 1[a != prev]``), taken in expectation under the policy::

       r_{j+1} = sum_k pi(k | s_j) R̂(s_{j+1})[k]  -  c * (1 - pi(a_{j-1} | s_j))

   with ``R̂`` the world model's *differentiable* 64-way reward table
   (:meth:`WorldModel.reward_table`) and ``c`` the switching penalty, an external
   hyper-parameter that the world model never learns.  Because the *entire* table is
   predicted, the expectation over the categorical action is exact: the actor receives the
   exact softmax policy gradient of the imagined reward (offset-invariant, zero variance)
   through the differentiable action probabilities, and no REINFORCE score function is
   needed.  (Indexing the table with the straight-through sample instead gives a gradient
   equal to the raw dB reward at the sampled slot; with all-negative dB rewards that pushes
   every sampled beam down and the policy collapses to one state-independent beam -
   verified on synthetic data, see ``tests/test_actor_critic.py``.)  ``a_{j-1}`` is the real
   current beam for ``j = 0`` and the sampled imagined beam afterwards, so the switching
   term is exactly the environment's expected penalty given the previous beam.
4. Lambda-returns ``R_j`` over ``s_0 .. s_{H-1}`` bootstrapped with the (slow EMA) critic's
   value at every imagined state and at the horizon (:func:`lambda_returns`).
5. Critic: MSE (in symlog space) to the detached lambda-return.
   Actor: maximise the lambda-return (gradients flow through the straight-through actions,
   the reward table and the dynamics) plus an entropy bonus; returns are scaled by an EMA of
   their 5-95 percentile range (DreamerV3) so the loss is insensitive to the dB units.

Alignment with the data: window ``(history up to t, target t+1)``.  The actor acts on the
*real* posterior state ``s_t``; its beam serves step ``t+1`` and is scored against the measured
power at ``t+1`` by the shared metric / MDP code, identical to every baseline.

The trainer exposes ``predict_scores(windows) -> (M, 64)`` (the policy logits at the
posterior state of each window) so it is evaluated with the very same code as the baselines.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.sequences import WindowedDataset
from ..models.encoders import symexp, symlog
from ..models.world_model import State, WorldModel

MAX_GRAD_NORM = 100.0  # DreamerV3 default for actor and critic
RETURN_NORM_DECAY = 0.99  # EMA of the return percentile range (DreamerV3: 0.99)
RETURN_NORM_PERCENTILES = (0.05, 0.95)


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


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Categorical policy over the beam codebook from the latent state ``feat = [h, z]``."""

    def __init__(self, latent_dim: int, n_actions: int = 64, hidden_dim: int = 256, unimix: float = 0.01):
        super().__init__()
        self.n_actions = n_actions
        self.unimix = unimix
        self.net = _mlp(latent_dim, hidden_dim, n_actions)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Logits ``(..., n_actions)``."""
        return self.net(feat).float()

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax with ``unimix`` uniform mixing (keeps every beam's probability > 0)."""
        p = logits.softmax(-1)
        return (1.0 - self.unimix) * p + self.unimix / self.n_actions

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training action: straight-through one-hot ``(..., n_actions)`` and the logits.

        Forward value is a one-hot drawn from the policy (exploration); backward pass uses the
        gradient of the mixed probabilities, so ``d(action)/d(theta) = d(probs)/d(theta)``.
        """
        logits = self(feat)
        probs = self.probs(logits)
        idx = torch.multinomial(probs.reshape(-1, self.n_actions), 1).reshape(*probs.shape[:-1])
        one_hot = F.one_hot(idx, self.n_actions).to(probs.dtype)
        return one_hot + (probs - probs.detach()), logits  # exactly 0/1 forward, softmax gradient backward

    @torch.no_grad()
    def act(self, feat: torch.Tensor) -> torch.Tensor:
        """Evaluation action: greedy beam index ``(...,)``."""
        return self(feat).argmax(-1)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = self.probs(logits)
        return -(p * torch.log(p)).sum(-1)


class Critic(nn.Module):
    """State-value head; the network predicts ``symlog(v)`` and is zero-initialised (DreamerV3)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp(latent_dim, hidden_dim, 1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward_symlog(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).float().squeeze(-1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Value ``(...,)`` in reward units."""
        return symexp(self.forward_symlog(feat))


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, continues: torch.Tensor,
                   gamma: float, lambda_: float) -> torch.Tensor:
    """TD(lambda) returns for an imagined trajectory (Hafner et al. 2020, eq. 4).

    ``rewards[:, j] = r_{j+1}``, ``values[:, j] = v(s_{j+1})`` and ``continues[:, j] = c_{j+1}``
    are the reward, bootstrap value and continuation flag of the state reached after the
    ``j``-th imagined action, ``j = 0..H-1``.  Returns ``R[:, j]`` for the states ``s_0..s_{H-1}``::

        R_{H-1} = r_H + gamma c_H v_H
        R_j     = r_{j+1} + gamma c_{j+1} [ (1 - lambda) v_{j+1} + lambda R_{j+1} ]

    ``lambda = 0`` gives one-step TD targets, ``lambda = 1`` the discounted Monte-Carlo return
    bootstrapped at the horizon.
    """
    if rewards.shape != values.shape or rewards.shape != continues.shape:
        raise ValueError("rewards, values and continues must have the same (B, H) shape")
    horizon = rewards.shape[1]
    nxt = values[:, -1]
    out = [None] * horizon
    for j in reversed(range(horizon)):
        if j == horizon - 1:
            ret = rewards[:, j] + gamma * continues[:, j] * nxt
        else:
            ret = rewards[:, j] + gamma * continues[:, j] * ((1.0 - lambda_) * values[:, j] + lambda_ * nxt)
        out[j] = ret
        nxt = ret
    return torch.stack(out, dim=1)


@dataclass
class ImaginedRollout:
    feats: torch.Tensor  # (B, H+1, F) latent features of s_0 .. s_H
    actions: torch.Tensor  # (B, H, A) straight-through one-hot samples a_0 .. a_{H-1} (fed to the GRU)
    probs: torch.Tensor  # (B, H, A) policy probabilities at s_0 .. s_{H-1}
    logits: torch.Tensor  # (B, H, A) policy logits at s_0 .. s_{H-1}
    reward_table: torch.Tensor  # (B, H, A) R̂(s_1 .. s_H) in dB
    beam_reward: torch.Tensor  # (B, H) E_pi[ R̂(s_{j+1})[a_j] ]
    switch: torch.Tensor  # (B, H) E_pi[ 1[a_j != a_{j-1}] ] = 1 - pi(a_{j-1} | s_j)
    reward: torch.Tensor  # (B, H) beam_reward - c * switch
    continues: torch.Tensor  # (B, H) predicted continuation of s_1 .. s_H


class ImaginedActorCritic:
    """Trainer tying a frozen world model, an actor and a critic together.

    The world model is put in eval mode and its parameters are frozen: Phase 4 trains only the
    actor and the critic (gradients still flow *through* the RSSM and the reward head into the
    actor's straight-through actions).
    """

    def __init__(self, world_model: WorldModel, config: ActorCriticConfig | None = None,
                 switching_penalty: float = 0.0, hidden_dim: int = 256):
        self.wm = world_model.eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)
        self.cfg = config or ActorCriticConfig()
        self.switching_penalty = float(switching_penalty)
        c = self.wm.cfg
        # The beam codebook is the reward table's width.  With ``action_dim == 0`` the RSSM is a pure
        # exogenous sequence model and the imagined latent trajectory is independent of the actor's
        # choice (the intended setting, see the module docstring); with ``action_dim > 0`` the sampled
        # one-hot is also fed to the GRU.
        self.n_actions = c.reward_dim
        if c.action_dim not in (0, c.reward_dim):
            raise ValueError("RSSMConfig.action_dim must be 0 (exogenous) or equal to reward_dim")
        self.action_conditioned = bool(c.action_dim)
        dev = self.wm.device
        self.actor = Actor(c.feat_dim, self.n_actions, hidden_dim).to(dev)
        self.critic = Critic(c.feat_dim, hidden_dim).to(dev)
        self.critic_ema = copy.deepcopy(self.critic)
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr, eps=1e-5)
        self.return_scale: float | None = None
        self.n_updates = 0

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return self.wm.device

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16,
                              enabled=self.cfg.mixed_precision and self.device.type == "cuda")

    @torch.no_grad()
    def start_states(self, batch: dict[str, torch.Tensor]) -> tuple[State, torch.Tensor]:
        """Every posterior state of a real ``(B, T)`` batch as a rollout start ``(B*T, ...)``,
        with the measured optimal beam of that step as the current beam ``a_{-1}`` (one-hot)."""
        post, _, _ = self.wm.observe(batch)
        flat = State(post.deter.reshape(-1, post.deter.shape[-1]),
                     post.stoch.reshape(-1, post.stoch.shape[-1]),
                     post.logits.reshape(-1, *post.logits.shape[-2:]))
        prev = F.one_hot(batch["power_db"].reshape(-1, self.n_actions).argmax(-1), self.n_actions).float()
        return flat, prev

    # ------------------------------------------------------------------ rollout
    def imagine_rollout(self, start: State, prev_action: torch.Tensor,
                        horizon: int | None = None) -> ImaginedRollout:
        """Imagine ``horizon`` steps from ``start`` (``(B, ...)``) under the current actor."""
        h = self.cfg.imagination_horizon if horizon is None else horizon
        state = start.detach()
        prev = prev_action.float()
        feats, actions, probs, logits, switches = [state.feat()], [], [], [], []
        for _ in range(h):
            with self._autocast():
                a, lg = self.actor.sample(feats[-1])
            p = self.actor.probs(lg)
            state = self.wm.rssm.img_step(state, a if self.action_conditioned else None)
            feats.append(state.feat())
            actions.append(a)
            probs.append(p)
            logits.append(lg)
            switches.append(1.0 - (p * prev).sum(-1))  # E_pi[ 1[a_j != a_{j-1}] ] given the previous beam
            prev = a
        feats_t = torch.stack(feats, 1)
        actions_t = torch.stack(actions, 1)
        probs_t = torch.stack(probs, 1)
        next_states = State(feats_t[:, 1:, : start.deter.shape[-1]], feats_t[:, 1:, start.deter.shape[-1]:], None)
        table = self.wm.reward_table(next_states)  # differentiable (B, H, A)
        with self.wm._autocast():
            cont = torch.sigmoid(self.wm.continue_head(feats_t[:, 1:]).float().squeeze(-1))
        # Expected reward under the policy.  The whole table is predicted, so E_pi[R̂[a]] is exact and
        # its gradient is the exact softmax policy gradient (offset-invariant, zero variance); a plain
        # straight-through R̂[a_sampled] has gradient R̂[k] at the sampled slot, which for all-negative
        # dB rewards pushes every sampled beam down and collapses the policy (verified on synthetic data).
        beam_reward = (table * probs_t).sum(-1)
        switch = torch.stack(switches, 1)
        reward = beam_reward - self.switching_penalty * switch
        return ImaginedRollout(feats_t, actions_t, probs_t, torch.stack(logits, 1), table, beam_reward, switch, reward, cont)

    # ------------------------------------------------------------------- update
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One actor + critic update from a real batch of sequences.  Returns diagnostics."""
        cfg = self.cfg
        self.actor.train()
        self.critic.train()
        start, prev = self.start_states(batch)
        roll = self.imagine_rollout(start, prev)

        # lambda-returns for s_0..s_{H-1}; bootstrap values from the slow critic, differentiable
        # w.r.t. the imagined states so the actor also receives value gradients (dynamics backprop)
        with self._autocast():
            values = self.critic_ema(roll.feats[:, 1:])
        returns = lambda_returns(roll.reward, values, roll.continues, cfg.gamma, cfg.lambda_)

        # DreamerV3 return normalisation: divide by max(1, EMA of the 5-95 percentile range)
        lo, hi = torch.quantile(returns.detach().float().flatten(), torch.tensor(RETURN_NORM_PERCENTILES, device=returns.device))
        rng = float(hi - lo)
        self.return_scale = rng if self.return_scale is None else RETURN_NORM_DECAY * self.return_scale + (1 - RETURN_NORM_DECAY) * rng
        scale = max(1.0, self.return_scale)

        entropy = self.actor.entropy(roll.logits)
        actor_loss = -(returns / scale).mean() - cfg.entropy_scale * entropy.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.actor_opt.step()

        # critic regresses the detached returns on detached states
        target = returns.detach()
        with self._autocast():
            v_symlog = self.critic.forward_symlog(roll.feats[:, :-1].detach())
        critic_loss = F.mse_loss(v_symlog, symlog(target))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_opt.step()
        with torch.no_grad():
            for p_ema, p in zip(self.critic_ema.parameters(), self.critic.parameters()):
                p_ema.lerp_(p, 1.0 - cfg.critic_ema_decay)
        self.n_updates += 1

        with torch.no_grad():
            table = roll.reward_table
            regret = (table.max(-1).values - roll.beam_reward).mean()  # vs the model's own best beam
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "return_mean": float(target.mean()),
            "return_scale": float(scale),
            "value_mean": float(values.detach().mean()),
            "reward_mean": float(roll.reward.detach().mean()),
            "beam_reward_mean": float(roll.beam_reward.detach().mean()),
            "imagined_regret_db": float(regret),
            "switch_rate": float(roll.switch.detach().mean()),
            "entropy": float(entropy.detach().mean()),
            "actor_grad_norm": float(actor_gn),
            "critic_grad_norm": float(critic_gn),
        }

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def predict_scores(self, windows: WindowedDataset, batch_size: int = 256,
                       extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """``(M, 64)`` policy logits at the posterior state of each history window - the shared
        evaluation interface (top-1 = the greedy beam for step ``t + 1``).

        ``extra`` maps side-modality batch keys to window-aligned ``(M, H, ...)`` arrays
        (camera_feat, lidar_tokens, lidar_centroids, trajectory), as for
        :meth:`WorldModel.predict_scores`.
        """
        self.actor.eval()
        out = []
        for s in range(0, len(windows), batch_size):
            hist = torch.as_tensor(10 * np.log10(np.maximum(windows.histories[s : s + batch_size], 1e-12)),
                                   dtype=torch.float32, device=self.device)
            batch = {"power_db": hist}
            if extra:
                for key, arr in extra.items():
                    batch[key] = torch.as_tensor(arr[s : s + batch_size], device=self.device)
            state = self.wm.encode_history(batch)
            with self._autocast():
                out.append(self.actor(state.feat()).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, self.n_actions))

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_ema": self.critic_ema.state_dict(), "config": self.cfg.__dict__,
                "switching_penalty": self.switching_penalty, "return_scale": self.return_scale,
                "n_updates": self.n_updates}


__all__ = ["Actor", "ActorCriticConfig", "Critic", "ImaginedActorCritic", "ImaginedRollout", "lambda_returns"]
