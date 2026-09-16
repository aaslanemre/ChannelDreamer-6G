"""DreamerV3-style recurrent state-space model (RSSM) world model for beam-power dynamics.

Equations (Hafner et al. 2023, "Mastering Diverse Domains through World Models")::

    sequence model    h_t   = f_phi(h_{t-1}, z_{t-1}, a_{t-1})          (GRU)
    encoder           z_t   ~ q_phi(z_t | h_t, o_t)                     (posterior, categorical)
    dynamics predictor ẑ_t  ~ p_phi(ẑ_t | h_t)                          (prior, categorical)
    reward predictor  r̂_t   ~ p_phi(r̂_t | h_t, z_t)                     (here: the full 64-way reward table)
    continue predictor ĉ_t  ~ p_phi(ĉ_t | h_t, z_t)
    decoder           ô_t   ~ p_phi(ô_t | h_t, z_t)

Loss (per step, averaged over batch and time)::

    L = beta_pred * (L_recon + L_reward + L_cont)
      + beta_dyn  * max(1, KL[ sg(q(z_t|h_t,o_t)) || p(ẑ_t|h_t) ])     # trains the prior
      + beta_rep  * max(1, KL[ q(z_t|h_t,o_t) || sg(p(ẑ_t|h_t)) ])     # trains the posterior

with ``sg`` the stop-gradient, free bits = 1 nat, beta_dyn = 0.5, beta_rep = 0.1 (DreamerV3
defaults).  Latents are ``stoch_discrete`` categorical variables with ``stoch_classes`` classes,
sampled with straight-through gradients and 1 % uniform mixing (``unimix``).  Reconstruction
and reward targets are in symlog space with MSE.

Why plain sequence prediction is a valid dynamics model here
------------------------------------------------------------
Beam choice does not influence the vehicle's trajectory (see
:mod:`channeldreamer.envs.offline_beam_env`), so ``o_{t+1}`` is independent of ``a_t``.  The
action input to the GRU is kept for fidelity to DreamerV3 and for Phase 4 (the actor's
action must flow through the imagination so the reward head can score it); with
``action_dim=0`` it is disabled and the model is a pure exogenous sequence model.  The
reward head predicts the **entire** 64-way reward table ``R[t, :]`` (received power in dB of
every candidate beam), which DeepSense observes at every step; the policy's reward is
``R[t, a_t] - switching penalty`` and is computed analytically from the table.

Memory
------
``deter_dim=256, 16x16 categorical latents, hidden 256`` is ~1.5 M parameters; a batch of
16 sequences x 16 steps trains in well under 1 GB on the RTX 4070 including the encoders'
projections (the frozen ResNet features are cached offline).  bf16 autocast is on by default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.sequences import WindowedDataset
from .encoders import EncoderConfig, MultimodalEncoder, symlog


@dataclass
class RSSMConfig:
    obs_dim: int = 64  # beam power vector (dB)
    action_dim: int = 64  # one-hot beam; 0 disables action conditioning
    embed_dim: int = 256  # output of the multimodal encoder
    deter_dim: int = 256
    stoch_discrete: int = 16
    stoch_classes: int = 16
    hidden_dim: int = 256
    reward_dim: int = 64  # full reward table
    unimix: float = 0.01
    kl_free_bits: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    mixed_precision: bool = True
    grad_checkpointing: bool = False  # reserved; sequences are short enough so far

    @property
    def stoch_dim(self) -> int:
        return self.stoch_discrete * self.stoch_classes

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + self.stoch_dim


@dataclass
class State:
    """RSSM state.  ``logits`` are the categorical logits that produced ``stoch``."""

    deter: torch.Tensor  # (..., deter_dim)
    stoch: torch.Tensor  # (..., stoch_discrete * stoch_classes) flattened one-hot (straight-through)
    logits: torch.Tensor  # (..., stoch_discrete, stoch_classes)

    def feat(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)

    def detach(self) -> State:
        return State(self.deter.detach(), self.stoch.detach(), self.logits.detach())

    def __getitem__(self, idx) -> State:
        return State(self.deter[idx], self.stoch[idx], self.logits[idx])


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 1) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def _stack_states(states: list[State], dim: int = 1) -> State:
    return State(torch.stack([s.deter for s in states], dim), torch.stack([s.stoch for s in states], dim),
                 torch.stack([s.logits for s in states], dim))


class RSSM(nn.Module):
    """Recurrent state-space model: GRU + categorical posterior / prior."""

    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.cfg = c = config
        in_dim = c.stoch_dim + c.action_dim
        self.img_in = nn.Sequential(nn.Linear(in_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(c.hidden_dim, c.deter_dim)
        self.prior_net = _mlp(c.deter_dim, c.hidden_dim, c.stoch_dim)
        self.post_net = _mlp(c.deter_dim + c.embed_dim, c.hidden_dim, c.stoch_dim)

    # ---------------------------------------------------------------- latents
    def initial_state(self, batch_size: int, device: torch.device | None = None) -> State:
        device = device or next(self.parameters()).device
        c = self.cfg
        return State(
            torch.zeros(batch_size, c.deter_dim, device=device),
            torch.zeros(batch_size, c.stoch_dim, device=device),
            torch.zeros(batch_size, c.stoch_discrete, c.stoch_classes, device=device),
        )

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits.float(), dim=-1)
        return (1 - self.cfg.unimix) * p + self.cfg.unimix / self.cfg.stoch_classes

    def _sample(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Straight-through categorical sample -> flattened ``(..., stoch_dim)``."""
        probs = self._probs(logits)
        if deterministic:
            idx = probs.argmax(-1)
        else:
            idx = torch.distributions.Categorical(probs=probs).sample()
        one_hot = F.one_hot(idx, self.cfg.stoch_classes).to(probs.dtype)
        sample = one_hot + probs - probs.detach()
        return sample.flatten(-2)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.cfg.stoch_discrete, self.cfg.stoch_classes)

    # ------------------------------------------------------------------ steps
    def img_step(self, prev: State, action: torch.Tensor | None, deterministic: bool = False) -> State:
        """Prior step: ``h_t = f(h_{t-1}, z_{t-1}, a_{t-1})``, ``ẑ_t ~ p(ẑ_t | h_t)``."""
        x = prev.stoch
        if self.cfg.action_dim:
            if action is None:
                action = torch.zeros(*x.shape[:-1], self.cfg.action_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, action.to(x.dtype)], dim=-1)
        h = self.gru(self.img_in(x), prev.deter)
        logits = self._logits(self.prior_net(h))
        return State(h, self._sample(logits, deterministic), logits)

    def obs_step(self, prev: State, action: torch.Tensor | None, embed: torch.Tensor,
                 deterministic: bool = False) -> tuple[State, State]:
        """Posterior step: returns ``(posterior, prior)`` sharing the same ``h_t``."""
        prior = self.img_step(prev, action, deterministic)
        logits = self._logits(self.post_net(torch.cat([prior.deter, embed], dim=-1)))
        post = State(prior.deter, self._sample(logits, deterministic), logits)
        return post, prior

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor | None = None,
                initial: State | None = None, deterministic: bool = False) -> tuple[State, State]:
        """Filter a sequence.  ``embeds (B, T, E)``, ``actions (B, T, A)`` = a_{t-1} aligned with
        step t (row 0 is the action before the first observation; zeros if unknown).
        Returns ``(posteriors, priors)`` with time-stacked tensors ``(B, T, ...)``."""
        b, t, _ = embeds.shape
        state = initial if initial is not None else self.initial_state(b, embeds.device)
        posts, priors = [], []
        for i in range(t):
            a = actions[:, i] if actions is not None else None
            state, prior = self.obs_step(state, a, embeds[:, i], deterministic)
            posts.append(state)
            priors.append(prior)
        return _stack_states(posts), _stack_states(priors)

    def imagine(self, start: State, horizon: int,
                policy: Callable[[torch.Tensor], torch.Tensor] | None = None,
                deterministic: bool = False) -> tuple[State, torch.Tensor | None]:
        """Roll the prior forward ``horizon`` steps from ``start`` (``(B, ...)``).

        ``policy(feat) -> action (B, action_dim)`` chooses the action fed into the next step
        (Phase 4 actor); ``None`` feeds zeros (exogenous rollout).  Returns imagined states
        ``(B, horizon, ...)`` and the actions taken ``(B, horizon, A)`` (or None).
        """
        state, states, acts = start, [], []
        for _ in range(horizon):
            a = policy(state.feat()) if policy is not None else None
            state = self.img_step(state, a, deterministic)
            states.append(state)
            if a is not None:
                acts.append(a)
        return _stack_states(states), (torch.stack(acts, 1) if acts else None)

    # ------------------------------------------------------------------- KL
    def kl_divergence(self, post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        """KL[q || p] between categorical latents, summed over latent variables -> ``(...,)``."""
        q = self._probs(post_logits)
        p = self._probs(prior_logits)
        kl = (q * (torch.log(q) - torch.log(p))).sum(-1)  # per discrete variable
        return kl.sum(-1)

    def kl_loss(self, post: State, prior: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DreamerV3 KL balancing with free bits: returns ``(dyn_loss, rep_loss, raw_kl)`` (scalars)."""
        c = self.cfg
        dyn = self.kl_divergence(post.logits.detach(), prior.logits)
        rep = self.kl_divergence(post.logits, prior.logits.detach())
        free = torch.full_like(dyn, c.kl_free_bits)
        return torch.maximum(dyn, free).mean(), torch.maximum(rep, free).mean(), dyn.detach().mean()


class WorldModel(nn.Module):
    """Multimodal encoder + RSSM + decoder / reward-table / continue heads."""

    def __init__(self, config: RSSMConfig, encoder_config: EncoderConfig | None = None):
        super().__init__()
        self.cfg = c = config
        self.enc_cfg = encoder_config or EncoderConfig(embed_dim=c.embed_dim, modalities=("power",))
        if self.enc_cfg.embed_dim != c.embed_dim:
            raise ValueError("EncoderConfig.embed_dim must equal RSSMConfig.embed_dim")
        self.encoder = MultimodalEncoder(self.enc_cfg)
        self.rssm = RSSM(c)
        self.decoder = _mlp(c.feat_dim, c.hidden_dim, c.obs_dim, layers=2)
        self.reward_head = _mlp(c.feat_dim, c.hidden_dim, c.reward_dim, layers=2)
        self.continue_head = _mlp(c.feat_dim, c.hidden_dim, 1, layers=1)
        # zero-init output layers as in DreamerV3 for stable starts
        for head in (self.reward_head, self.continue_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    # ----------------------------------------------------------------- utils
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.mixed_precision and self.device.type == "cuda")

    @staticmethod
    def actions_from_obs(power_db: torch.Tensor, n_actions: int) -> torch.Tensor:
        """Behaviour actions ``a_{t-1}`` = one-hot of the optimal beam at ``t-1`` (reactive
        behaviour policy; zeros for the first step)."""
        best = power_db.argmax(-1)
        one_hot = F.one_hot(best, n_actions).to(power_db.dtype)
        return torch.cat([torch.zeros_like(one_hot[:, :1]), one_hot[:, :-1]], dim=1)

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(batch)

    def observe(self, batch: dict[str, torch.Tensor], initial: State | None = None,
                deterministic: bool = False) -> tuple[State, State, torch.Tensor]:
        """Encode a batch and filter it.  Returns ``(posteriors, priors, embeds)``."""
        with self._autocast():
            embeds = self.embed(batch)
        actions = batch.get("actions")
        if actions is None and self.cfg.action_dim:
            actions = self.actions_from_obs(batch["power_db"], self.cfg.action_dim)
        post, prior = self.rssm.observe(embeds.float(), actions, initial, deterministic)
        return post, prior, embeds

    # ------------------------------------------------------------------ loss
    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Total DreamerV3 world-model loss on ``batch`` (see module docstring).

        Required: ``power_db (B, T, 64)``.  Optional: ``reward (B, T, reward_dim)`` (defaults
        to ``power_db``, i.e. the measured reward table), ``cont (B, T)`` continuation flags
        (default all ones), ``actions``, and the side-modality keys of
        :class:`~channeldreamer.models.encoders.MultimodalEncoder`.
        """
        c = self.cfg
        obs = batch["power_db"].float()
        post, prior, _ = self.observe(batch)
        feat = post.feat()
        with self._autocast():
            recon = self.decoder(feat).float()
            reward = self.reward_head(feat).float()
            cont_logit = self.continue_head(feat).float().squeeze(-1)
        target_obs = symlog(obs)
        target_rew = symlog(batch.get("reward", obs).float())
        cont = batch.get("cont", torch.ones_like(cont_logit)).float()
        l_recon = F.mse_loss(recon, target_obs)
        l_reward = F.mse_loss(reward, target_rew)
        l_cont = F.binary_cross_entropy_with_logits(cont_logit, cont)
        l_dyn, l_rep, raw_kl = self.rssm.kl_loss(post, prior)
        total = (c.recon_scale * l_recon + c.reward_scale * l_reward + c.continue_scale * l_cont
                 + c.kl_dyn_scale * l_dyn + c.kl_rep_scale * l_rep)
        _f = lambda t: float(t.detach())  # noqa: E731  (avoid grad->scalar warning)
        metrics = {"loss": _f(total), "recon": _f(l_recon), "reward": _f(l_reward),
                   "cont": _f(l_cont), "kl_dyn": _f(l_dyn), "kl_rep": _f(l_rep), "kl": _f(raw_kl)}
        return total, metrics

    # ------------------------------------------------------------ interfaces
    @torch.no_grad()
    def encode_history(self, batch: dict[str, torch.Tensor]) -> State:
        """Final posterior state after observing a ``(B, H, ...)`` history (policy input)."""
        post, _, _ = self.observe(batch, deterministic=True)
        return post[:, -1]

    @torch.no_grad()
    def predict_reward_table(self, state: State) -> torch.Tensor:
        """Reward table in dB (symexp of the head) for the given state(s)."""
        from .encoders import symexp

        with self._autocast():
            return symexp(self.reward_head(state.feat()).float())

    @torch.no_grad()
    def predict_scores(self, windows: WindowedDataset, batch_size: int = 256,
                       extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """``(M, 64)`` predicted reward table at ``t + horizon`` - the shared evaluation interface.

        Observes the ``H`` history steps with the posterior, then imagines ``horizon`` steps
        with the prior under the reactive behaviour policy (act = argmax of the predicted
        table) and returns the reward head at the final imagined step.
        """
        self.eval()
        k = windows.horizon
        out = []
        for s in range(0, len(windows), batch_size):
            hist = torch.as_tensor(10 * np.log10(np.maximum(windows.histories[s : s + batch_size], 1e-12)),
                                   dtype=torch.float32, device=self.device)
            batch = {"power_db": hist}
            if extra:
                for key, arr in extra.items():
                    batch[key] = torch.as_tensor(arr[s : s + batch_size], device=self.device)
            state = self.encode_history(batch)
            table = self.predict_reward_table(state)
            for _ in range(k):
                a = F.one_hot(table.argmax(-1), self.cfg.action_dim).float() if self.cfg.action_dim else None
                state = self.rssm.img_step(state, a, deterministic=True)
                table = self.predict_reward_table(state)
            out.append(table.cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, self.cfg.reward_dim))


def make_sequence_batch(
    windows: WindowedDataset,
    idx: np.ndarray,
    *,
    camera_feat: np.ndarray | None = None,
    lidar_tokens: np.ndarray | None = None,
    lidar_centroids: np.ndarray | None = None,
    trajectory: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a world-model batch from windows ``idx``: ``power_db (B, H, 64)`` plus per-step side
    modalities indexed by the *flat sample index* of each history step.

    Side arrays are indexed ``(N, ...)`` over the source scenario (aligned with ``windows``'
    flat indices); each history step ``t`` of a window maps to flat index
    ``last_index - H + 1 + t``.
    """
    idx = np.asarray(idx)
    hist = windows.histories[idx]
    power_db = 10 * np.log10(np.maximum(hist, 1e-12))
    batch = {"power_db": torch.as_tensor(power_db, dtype=torch.float32, device=device)}
    h = windows.history
    flat = windows.last_index[idx][:, None] - h + 1 + np.arange(h)[None, :]  # (B, H)
    if camera_feat is not None:
        batch["camera_feat"] = torch.as_tensor(camera_feat[flat], dtype=torch.float32, device=device)
    if lidar_tokens is not None:
        batch["lidar_tokens"] = torch.as_tensor(lidar_tokens[flat], dtype=torch.float32, device=device)
        batch["lidar_centroids"] = torch.as_tensor(lidar_centroids[flat], dtype=torch.float32, device=device)
    if trajectory is not None:
        batch["trajectory"] = torch.as_tensor(trajectory[flat], dtype=torch.float32, device=device)
    return batch


__all__ = ["RSSM", "RSSMConfig", "State", "WorldModel", "make_sequence_batch"]
