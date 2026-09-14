"""Peak-GPU-memory report for the Phase-3 components on real DeepSense data.

    python -m channeldreamer.scripts.profile_phase3 --data-root data --scenario 33 --batch 16 --seq 16

Stages: (1) camera backbone on raw images, (2) LiDAR + trajectory + power encoders on cached
tokens, (3) RSSM alone on random embeddings, (4) full WorldModel loss + backward on a real
multimodal batch.  Prints ``torch.cuda.max_memory_allocated()`` per stage so Phase-4 headroom
on the 12 GB RTX 4070 is known.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..data import load_scenario, make_windows
from ..data.modalities import load_camera_features, load_gps, load_lidar_tokens, trajectory_windows
from ..models.encoders import CameraEncoder, EncoderConfig, MultimodalEncoder, count_parameters
from ..models.world_model import RSSM, RSSMConfig, WorldModel, make_sequence_batch


def _peak(reset: bool = True) -> float:
    torch.cuda.synchronize()
    mb = torch.cuda.max_memory_allocated() / 2**20
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return mb


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--deter-dim", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda")
    ds = load_scenario(args.data_root, args.scenario, max_samples=args.max_samples, keep_frame=True)
    cache = ds.scenario_dir / "cache"
    enc_cfg = EncoderConfig()
    print(f"[data] {ds.n_samples} samples; batch {args.batch} x seq {args.seq}")

    # (1) camera backbone on raw images (what we avoid at train time by caching)
    torch.cuda.reset_peak_memory_stats()
    cam = CameraEncoder(enc_cfg).to(dev)
    imgs = cam.preprocess(ds.modality_paths["image"][: args.batch * args.seq]).to(dev)
    t0 = time.time()
    feats = cam(imgs.view(args.batch, args.seq, 3, enc_cfg.image_size, enc_cfg.image_size))
    print(f"[1 camera backbone] {tuple(feats.shape)} raw images -> embed: peak {_peak():.1f} MB, {time.time() - t0:.2f}s")
    del cam, imgs, feats
    torch.cuda.empty_cache()

    # (2) cached-modality encoders (power + lidar + trajectory + camera projection)
    w = make_windows(ds.power, ds.segment_ids, history=args.seq, horizon=1)
    idx = np.random.default_rng(0).choice(len(w), args.batch, replace=False)
    gps = load_gps(ds)
    traj = trajectory_windows(gps.user_local_m, ds.segment_ids, enc_cfg.trajectory_tokens)
    cam_feat = load_camera_features(cache, ds.sample_index)
    lidar_tok, lidar_cent = load_lidar_tokens(cache / "lidar", ds.sample_index)
    batch = make_sequence_batch(w, idx, camera_feat=cam_feat, lidar_tokens=lidar_tok, lidar_centroids=lidar_cent,
                                trajectory=traj, device=dev)
    torch.cuda.reset_peak_memory_stats()
    enc = MultimodalEncoder(enc_cfg).to(dev)
    e = enc(batch)
    e.sum().backward()
    print(f"[2 encoders (cached feats)] {tuple(e.shape)}: peak {_peak():.1f} MB, trainable params {count_parameters(enc):,}")
    del enc, e
    torch.cuda.empty_cache()

    # (3) RSSM alone
    rcfg = RSSMConfig(deter_dim=args.deter_dim)
    torch.cuda.reset_peak_memory_stats()
    rssm = RSSM(rcfg).to(dev)
    post, prior = rssm.observe(torch.randn(args.batch, args.seq, rcfg.embed_dim, device=dev))
    dyn, rep, _ = rssm.kl_loss(post, prior)
    (dyn + rep).backward()
    print(f"[3 RSSM alone] deter {rcfg.deter_dim}, {rcfg.stoch_discrete}x{rcfg.stoch_classes} latents: "
          f"peak {_peak():.1f} MB, params {count_parameters(rssm):,}")
    del rssm, post, prior
    torch.cuda.empty_cache()

    # (4) full world model, real multimodal batch, one optimiser step
    torch.cuda.reset_peak_memory_stats()
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim)).to(dev)
    opt = torch.optim.AdamW([q for q in wm.parameters() if q.requires_grad], lr=1e-4)
    t0 = time.time()
    for step in range(3):
        loss, m = wm.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"[4 world model step] loss {m['loss']:.3f} (recon {m['recon']:.3f} reward {m['reward']:.3f} kl {m['kl']:.2f}): "
          f"peak {_peak():.1f} MB, trainable params {count_parameters(wm):,}, {(time.time() - t0) / 3 * 1e3:.0f} ms/step")
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"[gpu] {torch.cuda.get_device_name(0)}: {total:.0f} MB total")


if __name__ == "__main__":
    main()
