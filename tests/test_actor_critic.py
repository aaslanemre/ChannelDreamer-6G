"""Phase-4 tests: actor / critic shapes, lambda-returns, the imagined-rollout training step and
the effect of the switching penalty.  All synthetic, CPU, small."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from channeldreamer.data import generate_synthetic, make_windows
from channeldreamer.models.encoders import EncoderConfig
from channeldreamer.models.world_model import RSSMConfig, WorldModel, make_sequence_batch
from channeldreamer.rl import Actor, ActorCriticConfig, Critic, ImaginedActorCritic, lambda_returns

SMALL_ENC = {"embed_dim": 64, "power_hidden": 64, "modalities": ("power",)}


def _small_rssm(**kw) -> RSSMConfig:
    # action_dim=0: exogenous dynamics, the imagined trajectory does not depend on the actor (see the
    # ImaginedActorCritic docstring for why conditioning on the logged behaviour action is a trap)
    base = {"embed_dim": 64, "deter_dim": 64, "stoch_discrete": 8, "stoch_classes": 8, "hidden_dim": 64,
            "mixed_precision": False, "action_dim": 0}
    base.update(kw)
    return RSSMConfig(**base)


def _synthetic_world_model(seed: int = 0, train_steps: int = 0, **rssm_kw):
    torch.manual_seed(seed)
    syn = generate_synthetic(n_segments=3, segment_length=60, seed=seed)
    w = make_windows(syn.power, syn.segment_ids, history=6, horizon=1)
    wm = WorldModel(_small_rssm(**rssm_kw), EncoderConfig(**SMALL_ENC))
    rng = np.random.default_rng(seed)
    if train_steps:
        opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
        for _ in range(train_steps):
            loss, _ = wm.loss(make_sequence_batch(w, rng.choice(len(w), 8, replace=False)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return wm, w, rng


# ------------------------------------------------------------------- shapes
def test_actor_and_critic_shapes_and_straight_through():
    torch.manual_seed(0)
    feat = torch.randn(5, 128)
    actor = Actor(128, n_actions=64, hidden_dim=32)
    critic = Critic(128, hidden_dim=32)
    assert actor(feat).shape == (5, 64)
    act, logits = actor.sample(feat)
    assert act.shape == (5, 64) and logits.shape == (5, 64)
    assert torch.allclose(act.sum(-1), torch.ones(5), atol=1e-5)  # forward value is one-hot
    assert torch.all((act.detach() == 0) | (act.detach() == 1)) and act.requires_grad
    act.sum().backward()  # straight-through: the sampled one-hot is differentiable w.r.t. theta
    assert any(p.grad is not None for p in actor.parameters())
    greedy = actor.act(feat)
    assert greedy.shape == (5,) and greedy.dtype == torch.long and 0 <= int(greedy.min()) and int(greedy.max()) < 64
    assert critic(feat).shape == (5,)
    assert torch.allclose(critic(feat), torch.zeros(5))  # zero-initialised value head
    ent = actor.entropy(logits)
    assert ent.shape == (5,) and torch.all(ent > 0) and torch.all(ent <= np.log(64) + 1e-5)


# ----------------------------------------------------------- lambda-returns
def test_lambda_returns_analytic_cases():
    b, h, r, v, gamma = 2, 6, -3.0, -10.0, 0.9
    rewards = torch.full((b, h), r)
    values = torch.full((b, h), v)
    cont = torch.ones(b, h)
    # lambda = 0: one-step TD target  r + gamma v  at every step
    td0 = lambda_returns(rewards, values, cont, gamma, 0.0)
    assert torch.allclose(td0, torch.full((b, h), r + gamma * v))
    # lambda = 1: discounted Monte-Carlo return bootstrapped at the horizon
    mc = lambda_returns(rewards, values, cont, gamma, 1.0)
    expected = torch.tensor([[sum(gamma**i * r for i in range(h - j)) + gamma ** (h - j) * v for j in range(h)]] * b)
    assert torch.allclose(mc, expected, atol=1e-5)
    # general lambda against a plain python recursion with random inputs
    torch.manual_seed(1)
    rw, va, co = torch.randn(b, h), torch.randn(b, h), (torch.rand(b, h) > 0.2).float()
    lam = 0.95
    ref = torch.zeros(b, h)
    for i in range(b):
        nxt = va[i, -1]
        for j in reversed(range(h)):
            nxt = rw[i, j] + gamma * co[i, j] * (nxt if j == h - 1 else (1 - lam) * va[i, j] + lam * nxt)
            ref[i, j] = nxt
    assert torch.allclose(lambda_returns(rw, va, co, gamma, lam), ref, atol=1e-5)
    # continuation 0 everywhere: the return is just the immediate reward
    assert torch.allclose(lambda_returns(rw, va, torch.zeros(b, h), gamma, lam), rw)
    with pytest.raises(ValueError):
        lambda_returns(rw, va[:, :-1], co, gamma, lam)


# ---------------------------------------------------------------- training
def test_train_step_tiny_batch_finite_and_predict_scores():
    wm, w, rng = _synthetic_world_model(seed=0, train_steps=5)
    cfg = ActorCriticConfig(imagination_horizon=4, mixed_precision=False)
    trainer = ImaginedActorCritic(wm, cfg, switching_penalty=0.5, hidden_dim=32)
    assert all(not p.requires_grad for p in wm.parameters())  # world model frozen
    batch = make_sequence_batch(w, rng.choice(len(w), 3, replace=False))
    start, prev = trainer.start_states(batch)
    assert start.deter.shape == (3 * 6, 64) and prev.shape == (18, 64)
    assert torch.allclose(prev.sum(-1), torch.ones(18))
    roll = trainer.imagine_rollout(start, prev)
    assert roll.feats.shape == (18, 5, 128) and roll.actions.shape == (18, 4, 64)
    assert roll.reward_table.shape == (18, 4, 64) and roll.reward.shape == (18, 4)
    assert roll.beam_reward.requires_grad and roll.reward.requires_grad  # differentiable path to the actor (via pi)
    assert not roll.reward_table.requires_grad  # exogenous model, frozen: the table itself carries no gradient
    assert torch.all((roll.switch.detach() >= 0) & (roll.switch.detach() <= 1))  # expected switch probability
    assert torch.allclose(roll.reward.detach(), roll.beam_reward.detach() - 0.5 * roll.switch.detach())
    for _ in range(3):
        m = trainer.train_step(make_sequence_batch(w, rng.choice(len(w), 3, replace=False)))
        assert set(m) >= {"actor_loss", "critic_loss", "switch_rate", "entropy", "return_mean"}
        assert all(np.isfinite(v) for v in m.values()), m
        assert 0.0 <= m["switch_rate"] <= 1.0 and 0.0 < m["entropy"] <= np.log(64) + 1e-4
    assert trainer.n_updates == 3
    assert all(torch.isfinite(p).all() for p in trainer.actor.parameters())
    scores = trainer.predict_scores(w)
    assert scores.shape == (len(w), 64) and np.isfinite(scores).all()


def test_switching_penalty_reduces_switching():
    """Controlled synthetic check of the switching-cost mechanism, isolated from world-model
    quality.  The world model's dynamics and reward table are replaced by a toy system in which
    tracking and holding are *distinguishable* optima:

    * start state: ``deter = 3 * one_hot(b)`` and ``stoch = one_hot(b)`` with ``b`` the currently
      served beam; imagined dynamics cyclically shift ``deter`` by one beam per step and keep
      ``stoch`` fixed, so ``s_j`` encodes both the held beam ``b`` (stoch) and the model's best
      beam for the next step, ``b + j + 1`` (deter, predictable one step ahead);
    * reward table: ``-5 dB`` everywhere except ``+3 dB`` (i.e. ``-2 dB``) at ``b + j + 1``.

    With ``c = 0`` the optimal policy tracks the best beam (a switch at every step); with
    ``c = 10 > 3`` it must hold ``b`` (no switches).  The real trainer (actor, critic,
    lambda-returns, expected switching penalty, straight-through samples) is used unchanged."""
    import torch.nn.functional as F

    from channeldreamer.models.world_model import State

    gain = 3.0
    rates, regrets = {}, {}
    for c in (0.0, 10.0):
        wm, w, rng = _synthetic_world_model(seed=0, train_steps=0)
        assert wm.cfg.deter_dim == 64 and wm.cfg.stoch_dim == 64
        wm.reward_table = lambda state: -5.0 + gain * torch.roll(state.deter, 1, dims=-1) / 3.0  # best = held+... shifted
        wm.rssm.img_step = lambda state, action=None, deterministic=False: State(
            torch.roll(state.deter, 1, dims=-1), state.stoch, state.logits)

        @torch.no_grad()
        def start_states(batch):
            b = batch["power_db"].reshape(-1, 64).argmax(-1)
            prev = F.one_hot(b, 64).float()
            return State(3.0 * prev, prev.clone(), torch.zeros(len(b), 8, 8)), prev

        cfg = ActorCriticConfig(imagination_horizon=5, actor_lr=3e-3, critic_lr=3e-3, entropy_scale=1e-2,
                                mixed_precision=False)
        torch.manual_seed(0)
        trainer = ImaginedActorCritic(wm, cfg, switching_penalty=c, hidden_dim=64)
        trainer.start_states = start_states
        last, reg = [], []
        for _ in range(200):
            m = trainer.train_step(make_sequence_batch(w, rng.choice(len(w), 8, replace=False)))
            assert all(np.isfinite(v) for v in m.values()), m
            last.append(m["switch_rate"])
            reg.append(m["imagined_regret_db"])
        rates[c] = float(np.mean(last[-10:]))
        regrets[c] = float(np.mean(reg[-10:]))
    # the reward table at s_{j+1} is best at beam (b + j + 1): check the toy system is wired as described
    st, prev = start_states({"power_db": F.one_hot(torch.tensor([5, 9]), 64).float()[:, None, :]})
    nxt = wm.rssm.img_step(st)
    assert wm.reward_table(nxt).argmax(-1).tolist() == [7, 11] and torch.allclose(nxt.stoch, prev)
    # c = 0: tracks the best beam -> switches (almost) every step with low model regret
    assert regrets[0.0] < 1.0, regrets
    assert rates[0.0] > 0.7, rates
    # c = 10 > gain: holds the current beam -> measurably fewer switches, paying the model regret
    assert rates[10.0] < 0.5 * rates[0.0], rates
    assert rates[10.0] < 0.35, rates  # entropy bonus keeps ~1 - pi(b|s) above zero
    assert regrets[10.0] > regrets[0.0], regrets
