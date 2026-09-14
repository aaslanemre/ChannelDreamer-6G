"""Validate that a real DeepSense 6G scenario loads correctly.

    python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33

Prints the resolved scenario folder / CSV, the regex-resolved columns, the power-vector
shape, label cross-validation results and segment statistics.  On failure the actual
exception and the CSV header are shown rather than guessed at.

Offline caches for the multimodal world model (Phase 3), written under
``<scenario_dir>/cache/`` so training never touches raw point clouds or JPEGs::

    python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33 --pretokenize-lidar
    python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33 --precompute-camera
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.deepsense import find_csv, load_scenario, resolve_columns, resolve_scenario_dir
from ..eval.regimes import label_regimes, regime_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--time-gap-s", type=float, default=None,
                   help="also break segments at timestamp gaps larger than this (seconds)")
    p.add_argument("--check-files", action="store_true",
                   help="verify that referenced camera/lidar/radar files exist on disk")
    p.add_argument("--pretokenize-lidar", action="store_true",
                   help="offline LiDAR grouping + mini-PointNet tokens -> cache/lidar/lidar_tokens_<index>.npz")
    p.add_argument("--precompute-camera", action="store_true",
                   help="frozen ResNet backbone features for every image -> cache/camera_features.npz")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="override the cache directory (default <scenario_dir>/cache)")
    p.add_argument("--overwrite", action="store_true", help="recompute cached files that already exist")
    p.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto (camera features)")
    return p


def pretokenize_lidar(ds, cache_dir: Path, overwrite: bool = False) -> int:
    from tqdm import tqdm

    from ..data.modalities import lidar_cache_path
    from ..models.encoders import EncoderConfig, LidarTokenizer

    cfg = EncoderConfig()
    tok = LidarTokenizer(cfg)
    out_dir = cache_dir / "lidar"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for path, idx in tqdm(list(zip(ds.modality_paths["lidar"], ds.sample_index)), desc="lidar tokens"):
        out = lidar_cache_path(out_dir, idx)
        if out.exists() and not overwrite:
            continue
        tok.tokenize_to_cache(path, out)
        done += 1
    print(f"lidar tokens    : {done} written, cfg groups={cfg.lidar_n_groups} size={cfg.lidar_group_size} "
          f"dim={cfg.lidar_token_dim} -> {out_dir}")
    return done


def precompute_camera(ds, cache_dir: Path, device: str = "auto", overwrite: bool = False) -> Path:
    import torch

    from ..data.modalities import camera_cache_path
    from ..models.encoders import CameraEncoder, EncoderConfig

    out = camera_cache_path(cache_dir)
    if out.exists() and not overwrite:
        print(f"camera features : {out} exists (use --overwrite to recompute)")
        return out
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    cfg = EncoderConfig()
    enc = CameraEncoder(cfg).to(dev)
    paths = ds.modality_paths["image"]
    feats = enc.encode_paths(paths, batch_size=32, device=dev)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats.astype(np.float32), sample_index=np.asarray(ds.sample_index))
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0
    print(f"camera features : {feats.shape} ({cfg.camera_backbone}, frozen) -> {out}  peak GPU {peak:.1f} MB")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sdir = resolve_scenario_dir(args.data_root, args.scenario)
        csv_path = find_csv(sdir)
    except Exception:  # noqa: BLE001 - show the real error
        traceback.print_exc()
        return 1
    print(f"scenario folder : {sdir}")
    print(f"csv             : {csv_path}")
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    print(f"csv header      : {header}")
    resolved = resolve_columns(header)
    print("resolved columns:")
    for k, v in resolved.items():
        print(f"    {k:<12} -> {v}")
    if resolved["power"] is None:
        print("ERROR: could not resolve the beam-power column from the header above", file=sys.stderr)
        return 2

    try:
        ds = load_scenario(args.data_root, args.scenario, max_samples=args.max_samples,
                           time_gap_s=args.time_gap_s)
    except Exception:  # noqa: BLE001
        print("ERROR while loading power vectors:", file=sys.stderr)
        traceback.print_exc()
        return 3

    print("\n" + ds.summary())
    beams = ds.optimal_beam()
    print(f"optimal beam    : range [{beams.min()}, {beams.max()}], "
          f"{len(np.unique(beams))} distinct beams used")
    labels = label_regimes(ds.power, ds.segment_ids)
    print(f"regimes (default thresholds): {regime_summary(labels)}")
    print("modalities      : " + ", ".join(f"{k} ({len(v)} refs)" for k, v in ds.modality_paths.items()))

    if args.check_files:
        for k, paths in ds.modality_paths.items():
            missing = sum(1 for p in paths if not Path(p).exists())
            print(f"    {k:<20} missing files: {missing}/{len(paths)}")
    cache_dir = Path(args.cache_dir) if args.cache_dir else ds.scenario_dir / "cache"
    if args.pretokenize_lidar:
        pretokenize_lidar(ds, cache_dir, overwrite=args.overwrite)
    if args.precompute_camera:
        precompute_camera(ds, cache_dir, device=args.device, overwrite=args.overwrite)
    print("\nOK: scenario loads correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
