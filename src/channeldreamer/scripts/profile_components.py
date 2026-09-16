"""Per-component peak GPU memory for the Phase-3 stack (RTX 4070, 12 GB).

    python -m channeldreamer.scripts.profile_components

Measures, each in isolation with ``torch.cuda.reset_peak_memory_stats()`` in between:
camera encoder (real scenario 33 images), LiDAR offline tokenizer + online encoder (real .ply),
GPS/trajectory encoder, RSSM alone (synthetic), and the combined WorldModel forward+backward
on a real B=8, T=8 batch - once with cached camera features (the training path) and once with
raw images through the backbone (reference; ~4x more expensive, do not train this way).
Requires ``data/scenario33`` and its ``cache/`` (see ``docs/data_setup.md``).
"""
import time, numpy as np, torch
from ..data import load_scenario, make_windows
from ..data.modalities import (load_camera_features, load_gps, load_lidar_tokens,
                                            trajectory_windows, read_ply_points)
from ..models.encoders import (CameraEncoder, EncoderConfig, LidarEncoder, LidarTokenizer,
                                            TrajectoryEncoder, count_parameters)
from ..models.world_model import RSSM, RSSMConfig, WorldModel, make_sequence_batch


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    B, T = 8, 8
    rows = []  # noqa

    def stage(name, fn):
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 2**20
        t0 = time.time(); extra = fn() or ""; torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        rows.append((name, base, peak, reserved, time.time() - t0, extra))
        print(f"[{name}] baseline {base:.1f} MB | peak alloc {peak:.1f} MB | peak reserved {reserved:.1f} MB | {time.time()-t0:.2f}s {extra}")
        torch.cuda.synchronize(); torch.cuda.empty_cache()

    ds = load_scenario("data", "33", keep_frame=True)
    cache = ds.scenario_dir / "cache"
    ecfg = EncoderConfig()
    rcfg = RSSMConfig()
    imgs = ds.modality_paths["image"]; plys = ds.modality_paths["lidar"]

    # 1. camera encoder alone: 8 real images, forward + backward through trainable projection
    def s_camera():
        cam = CameraEncoder(ecfg).to(dev)
        x = cam.preprocess(imgs[:8]).to(dev)
        torch.cuda.reset_peak_memory_stats()  # exclude host->device staging of the weights
        e = cam(x); e.sum().backward()
        return f"(8x3x224x224 -> {tuple(e.shape)}, trainable {count_parameters(cam):,})"
    stage("1a camera encoder, 8 real imgs, frozen backbone", s_camera)

    def s_camera_unfrozen():
        cam = CameraEncoder(EncoderConfig(freeze_camera_backbone=False)).to(dev)
        x = cam.preprocess(imgs[:8]).to(dev)
        torch.cuda.reset_peak_memory_stats()
        e = cam(x); e.sum().backward()
        return f"(trainable {count_parameters(cam):,})"
    stage("1b camera encoder, 8 real imgs, backbone UNFROZEN (reference)", s_camera_unfrozen)

    # 2. lidar: offline tokenizer on 8 real .ply on GPU, then online LidarEncoder fwd+bwd on those tokens
    toks = []
    def s_lidar_tok():
        tok = LidarTokenizer(ecfg, device="cuda")
        npts = []
        for p in plys[:8]:
            pts = read_ply_points(p); npts.append(len(pts))
            toks.append(tok.tokenize_points(pts))
        return f"(points/file {min(npts)}..{max(npts)}, tokens {toks[0]['tokens'].shape})"
    stage("2a lidar offline pre-tokenizer, 8 real .ply (GPU)", s_lidar_tok)

    def s_lidar_enc():
        enc = LidarEncoder(ecfg).to(dev)
        t = torch.as_tensor(np.stack([o["tokens"] for o in toks]), device=dev)
        c = torch.as_tensor(np.stack([o["centroids"] for o in toks]), device=dev)
        torch.cuda.reset_peak_memory_stats()
        e = enc(t, c); e.sum().backward()
        return f"({tuple(t.shape)} -> {tuple(e.shape)}, trainable {count_parameters(enc):,})"
    stage("2b lidar online encoder, 8 real token sets, fwd+bwd", s_lidar_enc)

    # 3. GPS / trajectory encoder alone on 8 real trajectory windows
    gps = load_gps(ds)
    traj = trajectory_windows(gps.user_local_m, ds.segment_ids, ecfg.trajectory_tokens)
    def s_traj():
        enc = TrajectoryEncoder(ecfg).to(dev)
        x = torch.as_tensor(traj[:8], device=dev)
        torch.cuda.reset_peak_memory_stats()
        e = enc(x); e.sum().backward()
        return f"({tuple(x.shape)} -> {tuple(e.shape)}, trainable {count_parameters(enc):,})"
    stage("3 GPS/trajectory encoder, 8 real windows, fwd+bwd", s_traj)

    # 4. RSSM alone, small synthetic batch
    def s_rssm():
        rssm = RSSM(rcfg).to(dev)
        emb = torch.randn(B, T, rcfg.embed_dim, device=dev)
        torch.cuda.reset_peak_memory_stats()
        post, prior = rssm.observe(emb)
        dyn, rep, _ = rssm.kl_loss(post, prior); (dyn + rep).backward()
        return f"(B={B},T={T}, deter {rcfg.deter_dim}, {rcfg.stoch_discrete}x{rcfg.stoch_classes}, params {count_parameters(rssm):,})"
    stage("4 RSSM alone, synthetic B=8 T=8", s_rssm)

    # 5. combined: full WorldModel (power+camera+lidar+trajectory) loss + backward + AdamW step, real batch
    w = make_windows(ds.power, ds.segment_ids, history=T, horizon=1)
    idx = np.random.default_rng(0).choice(len(w), B, replace=False)
    cam_feat = load_camera_features(cache, ds.sample_index)
    lt, lc = load_lidar_tokens(cache / "lidar", ds.sample_index)
    def combined(raw_images: bool, opt_step: bool):
        def fn():
            wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim)).to(dev)
            batch = make_sequence_batch(w, idx, camera_feat=None if raw_images else cam_feat,
                                        lidar_tokens=lt, lidar_centroids=lc, trajectory=traj, device=dev)
            if raw_images:
                flat = (w.last_index[idx][:, None] - T + 1 + np.arange(T)[None, :]).reshape(-1)
                x = wm.encoder.encoders["camera"].preprocess([imgs[i] for i in flat])
                batch["camera_images"] = x.view(B, T, 3, 224, 224).to(dev)
            opt = torch.optim.AdamW([q for q in wm.parameters() if q.requires_grad], lr=1e-4) if opt_step else None
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2 if opt_step else 1):
                loss, m = wm.loss(batch)
                if opt: opt.zero_grad(set_to_none=True)
                loss.backward()
                if opt: opt.step()
            return f"(keys {sorted(batch)}, loss {m['loss']:.3f}, trainable {count_parameters(wm):,})"
        return fn
    stage("5a combined fwd+bwd, cached camera feats (Phase-4 path)", combined(False, False))
    stage("5b combined fwd+bwd+AdamW step x2, cached camera feats", combined(False, True))
    stage("5c combined fwd+bwd, RAW images through backbone (B*T=64 imgs)", combined(True, False))

    print("\nGPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MB")
    print(f"{'stage':<66}{'peak alloc MB':>14}{'peak resv MB':>14}{'s':>7}")
    for n, b, p, r, t, _ in rows:
        print(f"{n:<66}{p:>14.1f}{r:>14.1f}{t:>7.2f}")


if __name__ == "__main__":
    main()
