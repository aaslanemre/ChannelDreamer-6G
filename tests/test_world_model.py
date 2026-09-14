"""Phase-3 tests: multimodal encoders, LiDAR tokenizer, GPS trajectories and the DreamerV3 RSSM.

Synthetic tests always run; real-data tests are skipped when ``data/scenario33`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from channeldreamer.data import generate_synthetic, make_windows
from channeldreamer.data.modalities import (
    farthest_point_sampling,
    group_point_cloud,
    latlon_to_local_m,
    read_ply_points,
    trajectory_windows,
)
from channeldreamer.models.encoders import (
    EncoderConfig,
    LidarEncoder,
    LidarTokenizer,
    MultimodalEncoder,
    PowerEncoder,
    TrajectoryEncoder,
)
from channeldreamer.models.world_model import RSSM, RSSMConfig, WorldModel, make_sequence_batch

DATA = Path("data")
SCENARIO_DIR = DATA / "scenario33"
needs_real_data = pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="real DeepSense scenario 33 not present")
SMALL_ENC = {"embed_dim": 64, "power_hidden": 64, "lidar_token_dim": 16, "trajectory_hidden": 32}


def _small_rssm(**kw) -> RSSMConfig:
    base = {"embed_dim": 64, "deter_dim": 64, "stoch_discrete": 8, "stoch_classes": 8, "hidden_dim": 64, "mixed_precision": False}
    base.update(kw)
    return RSSMConfig(**base)


# ----------------------------------------------------------------- modalities
def test_latlon_projection_and_trajectory_windows():
    origin = np.array([33.4, -111.9])
    ll = np.array([[33.4, -111.9], [33.4 + 1e-4, -111.9], [33.4, -111.9 + 1e-4]])
    xy = latlon_to_local_m(ll, origin)
    np.testing.assert_allclose(xy[0], 0.0, atol=1e-9)
    assert abs(xy[1, 1] - 11.1) < 0.2 and abs(xy[1, 0]) < 1e-6  # 1e-4 deg lat ~ 11.1 m north
    assert abs(xy[2, 0] - 9.3) < 0.3  # 1e-4 deg lon at 33.4N ~ 9.3 m east
    pos = np.cumsum(np.ones((10, 2)), axis=0)
    seg = np.array([0] * 5 + [1] * 5)
    tw = trajectory_windows(pos, seg, n_tokens=4, scale_m=1.0)
    assert tw.shape == (10, 4, 4)
    np.testing.assert_allclose(tw[5, :, :2], np.broadcast_to(pos[5], (4, 2)))  # clamped to segment start
    np.testing.assert_allclose(tw[5, :, 2:], 0.0)  # no displacement across the boundary
    np.testing.assert_allclose(tw[9, :, 2:], 1.0)  # steady 1 m/step inside the segment


def test_point_cloud_grouping_and_tokenizer():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(size=(500, 3)) * 5, rng.uniform(0, 255, (500, 1))], axis=1).astype(np.float32)
    idx = farthest_point_sampling(pts[:, :3], 8)
    assert len(np.unique(idx)) == 8
    groups, cents = group_point_cloud(pts, n_points=256, n_groups=8, group_size=16, max_range_m=None)
    assert groups.shape == (8, 16, 4) and cents.shape == (8, 3)
    assert np.abs(groups[:, :, :3]).max() < 30
    cfg = EncoderConfig(**SMALL_ENC, lidar_n_points=256, lidar_n_groups=8, lidar_group_size=16)
    out = LidarTokenizer(cfg).tokenize_points(pts)
    assert out["tokens"].shape == (8, 16) and np.isfinite(out["tokens"]).all()
    out2 = LidarTokenizer(cfg).tokenize_points(pts)  # deterministic (seeded)
    np.testing.assert_allclose(out["tokens"], out2["tokens"])
    emb = LidarEncoder(cfg)(torch.from_numpy(out["tokens"])[None, None], torch.from_numpy(out["centroids"])[None, None])
    assert emb.shape == (1, 1, 64)


def test_power_trajectory_and_fusion_encoders():
    cfg = EncoderConfig(**SMALL_ENC, modalities=("power", "trajectory", "lidar"))
    enc = MultimodalEncoder(cfg)
    b = {"power_db": torch.randn(3, 5, 64), "trajectory": torch.randn(3, 5, cfg.trajectory_tokens, 4)}
    e = enc(b)  # lidar absent -> learned absent vector
    assert e.shape == (3, 5, 64) and torch.isfinite(e).all()
    assert PowerEncoder(cfg)(torch.randn(2, 64)).shape == (2, 64)
    assert TrajectoryEncoder(cfg)(torch.randn(2, cfg.trajectory_tokens, 4)).shape == (2, 64)


# ----------------------------------------------------------------------- RSSM
def test_rssm_observe_imagine_shapes():
    cfg = _small_rssm()
    rssm = RSSM(cfg)
    e = torch.randn(3, 7, cfg.embed_dim)
    post, prior = rssm.observe(e)
    assert post.deter.shape == (3, 7, 64) and post.stoch.shape == (3, 7, 64) and post.logits.shape == (3, 7, 8, 8)
    assert torch.allclose(post.deter, prior.deter)  # shared h_t
    # straight-through samples are one-hot per discrete variable in the forward pass
    oh = post.stoch.view(3, 7, 8, 8)
    assert torch.allclose(oh.sum(-1), torch.ones(3, 7, 8), atol=1e-5)
    img, acts = rssm.imagine(post[:, -1], 4, policy=lambda f: torch.eye(64)[torch.randint(0, 64, (f.shape[0],))])
    assert img.deter.shape == (3, 4, 64) and acts.shape == (3, 4, 64)
    img0, acts0 = rssm.imagine(post[:, -1], 2)  # exogenous rollout without a policy
    assert img0.deter.shape == (3, 2, 64) and acts0 is None
    d1 = rssm.imagine(post[:, -1], 3, deterministic=True)[0].deter
    d2 = rssm.imagine(post[:, -1], 3, deterministic=True)[0].deter
    assert torch.allclose(d1, d2)


def test_rssm_kl_balancing_and_free_bits():
    cfg = _small_rssm(kl_free_bits=1.0)
    rssm = RSSM(cfg)
    post, prior = rssm.observe(torch.randn(2, 4, cfg.embed_dim))
    dyn, rep, raw = rssm.kl_loss(post, prior)
    assert dyn.item() >= 1.0 and rep.item() >= 1.0  # free bits clamp
    assert raw.item() >= 0.0
    # stop-gradients: dyn trains the prior only, rep trains the posterior only
    # (checked without free bits, which would otherwise zero the gradient below 1 nat, and on a
    # single step from a fixed initial state: over longer sequences the dyn loss legitimately
    # reaches the posterior of *earlier* steps through the recurrence)
    rssm = RSSM(_small_rssm(kl_free_bits=0.0))
    post, prior = rssm.observe(torch.randn(2, 1, cfg.embed_dim))
    dyn, rep, raw = rssm.kl_loss(post, prior)
    dyn.backward(retain_graph=True)
    assert all(p.grad is None or torch.all(p.grad == 0) for p in rssm.post_net.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in rssm.prior_net.parameters())
    rssm.zero_grad()
    rep.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in rssm.post_net.parameters())
    assert all(p.grad is None or torch.all(p.grad == 0) for p in rssm.prior_net.parameters())
    # identical distributions -> zero raw KL
    same = rssm.kl_divergence(post.logits, post.logits)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-5)


def test_world_model_loss_backward_synthetic():
    syn = generate_synthetic(n_segments=2, segment_length=50, seed=2)
    w = make_windows(syn.power, syn.segment_ids, history=6, horizon=2)
    cfg = _small_rssm()
    wm = WorldModel(cfg, EncoderConfig(**SMALL_ENC, modalities=("power",)))
    batch = make_sequence_batch(w, np.arange(4))
    assert batch["power_db"].shape == (4, 6, 64)
    loss, m = wm.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)
    for k in ("recon", "reward", "cont", "kl_dyn", "kl_rep"):
        assert np.isfinite(m[k])
    # a few optimiser steps reduce the loss
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    first = None
    for _ in range(15):
        loss, m = wm.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or m["loss"]
    assert m["loss"] < first
    scores = wm.predict_scores(w)  # shared evaluation interface (imagines `horizon` steps)
    assert scores.shape == (len(w), 64) and np.isfinite(scores).all()
    assert wm.encode_history(batch).deter.shape == (4, 64)


def test_world_model_without_actions():
    cfg = _small_rssm(action_dim=0)
    wm = WorldModel(cfg, EncoderConfig(**SMALL_ENC, modalities=("power",)))
    loss, _ = wm.loss({"power_db": torch.randn(2, 5, 64)})
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ real data
@pytest.fixture(scope="module")
def scenario():
    from channeldreamer.data import load_scenario

    return load_scenario(DATA, 33, keep_frame=True)


@needs_real_data
def test_camera_encoder_on_real_images(scenario):
    from channeldreamer.models.encoders import CameraEncoder

    cfg = EncoderConfig(embed_dim=64)
    enc = CameraEncoder(cfg)
    assert enc.frozen and all(not p.requires_grad for p in enc.backbone.parameters())
    x = enc.preprocess(scenario.modality_paths["image"][:4])
    assert x.shape == (4, 3, 224, 224)
    emb = enc(x.view(2, 2, 3, 224, 224))
    assert emb.shape == (2, 2, 64) and torch.isfinite(emb).all()
    feats = enc.encode_paths(scenario.modality_paths["image"][:3], batch_size=2)
    assert feats.shape == (3, 512) and np.isfinite(feats).all()
    assert enc.forward_features(torch.from_numpy(feats)).shape == (3, 64)


@needs_real_data
def test_lidar_tokenizer_on_real_ply(scenario, tmp_path):
    pts = read_ply_points(scenario.modality_paths["lidar"][0])
    assert pts.ndim == 2 and pts.shape[1] == 4 and len(pts) > 1000
    tok = LidarTokenizer(EncoderConfig())
    for i in range(3):
        out = tok.tokenize_to_cache(scenario.modality_paths["lidar"][i], tmp_path / f"lidar_tokens_{i}.npz")
        assert out["tokens"].shape == (16, 64) and out["centroids"].shape == (16, 3) and out["groups"].shape == (16, 32, 4)
        assert np.isfinite(out["tokens"]).all()
        assert np.linalg.norm(out["centroids"], axis=1).max() <= 60.0 + 1e-3
    with np.load(tmp_path / "lidar_tokens_0.npz") as z:
        assert set(z.keys()) == {"tokens", "centroids", "groups"}


@needs_real_data
def test_gps_trajectory_on_real_csv(scenario):
    from channeldreamer.data.modalities import load_gps

    gps = load_gps(scenario)
    assert gps.user_local_m.shape == (scenario.n_samples, 2) and gps.n_missing == 0
    assert np.abs(gps.user_local_m).max() < 500  # user stays within a few hundred metres of the BS
    tw = trajectory_windows(gps.user_local_m, scenario.segment_ids, 16)
    assert tw.shape == (scenario.n_samples, 16, 4) and np.isfinite(tw).all()
    assert TrajectoryEncoder(EncoderConfig(embed_dim=64))(torch.from_numpy(tw[:5])).shape == (5, 64)


@needs_real_data
def test_world_model_integration_real_batch(scenario, tmp_path):
    """Encode a real windowed sequence (power + camera + trajectory) through observe(),
    compute the full loss and backpropagate without NaN.  Reports peak GPU memory."""
    from channeldreamer.data.modalities import load_gps
    from channeldreamer.models.encoders import CameraEncoder

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 64  # first 64 samples are enough for a (B=2, T=6) batch
    w = make_windows(scenario.power[:n], scenario.segment_ids[:n], history=6, horizon=1)
    gps = load_gps(scenario)
    traj = trajectory_windows(gps.user_local_m[:n], scenario.segment_ids[:n], 16)
    cam = CameraEncoder(EncoderConfig(embed_dim=128)).to(dev)
    cam_feat = cam.encode_paths(scenario.modality_paths["image"][:n], batch_size=16, device=dev)
    assert cam_feat.shape == (n, 512)
    idx = np.array([0, len(w) // 2])
    batch = make_sequence_batch(w, idx, camera_feat=cam_feat, trajectory=traj, device=dev)
    assert batch["power_db"].shape == (2, 6, 64) and batch["camera_feat"].shape == (2, 6, 512)
    assert batch["trajectory"].shape == (2, 6, 16, 4)

    rcfg = RSSMConfig(embed_dim=128, deter_dim=128, stoch_discrete=8, stoch_classes=8, hidden_dim=128)
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=128, modalities=("power", "camera", "trajectory"))).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    post, _prior, emb = wm.observe(batch)
    assert emb.shape == (2, 6, 128) and post.deter.shape == (2, 6, 128) and post.logits.shape == (2, 6, 8, 8)
    loss, m = wm.loss(batch)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in m.values())
    loss.backward()
    grads = [p.grad for p in wm.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        print(f"\n[integration] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")
