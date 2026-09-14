"""Multimodal encoders for side information (camera, LiDAR, GPS trajectory) and beam power.

VRAM budget: everything must train on a single **RTX 4070 with 12 GB**.  Consequently:

* :class:`CameraEncoder` - torchvision ResNet-18 / ResNet-34 (ImageNet weights), **frozen by
  default** (``EncoderConfig.freeze_camera_backbone``).  Because it is frozen, its 512-d pooled
  features are computed once offline (``prepare_data.py --precompute-camera``) and cached; the
  trainable part is only the projection to ``embed_dim``.  Raw JPEGs never enter the training
  loop.  bf16 autocast + channels-last when the backbone does run.
* :class:`LidarTokenizer` (offline, numpy + a tiny frozen mini-PointNet) turns each ``.ply``
  cloud into ``lidar_n_groups`` tokens (Point-BERT-style FPS + kNN grouping, then per-group
  shared MLP + max-pool).  Tokens and centroids are cached as ``.npz`` per sample by
  ``prepare_data.py --pretokenize-lidar``.  :class:`LidarEncoder` (online, trainable) is a
  one-layer transformer over the cached tokens with a centroid positional MLP.  The frozen
  tokenizer is a *random-feature* placeholder for a learned dVAE codebook; it is seeded and
  deterministic, and the grouped points are cached alongside so a learned tokenizer can be
  trained later without re-reading point clouds.
* :class:`TrajectoryEncoder` - MLP over the last ``trajectory_tokens`` GPS samples (east,
  north, and per-step displacement relative to the base station), see
  :func:`channeldreamer.data.modalities.trajectory_windows`.
* :class:`PowerEncoder` - MLP over the 64-beam power vector in dB (symlog-normalised).

:class:`MultimodalEncoder` fuses whatever modalities are present in a batch dict into one
``(B, T, embed_dim)`` embedding for the RSSM.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.modalities import group_point_cloud, read_ply_points

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EncoderConfig:
    embed_dim: int = 256
    # camera
    camera_backbone: str = "resnet18"  # resnet18 | resnet34
    camera_pretrained: bool = True
    freeze_camera_backbone: bool = True
    image_size: int = 224
    # lidar (offline tokenizer + online encoder)
    lidar_n_points: int = 2048
    lidar_n_groups: int = 16
    lidar_group_size: int = 32
    lidar_token_dim: int = 64
    lidar_max_range_m: float = 60.0
    lidar_seed: int = 0
    lidar_layers: int = 1
    # trajectory
    trajectory_tokens: int = 16
    trajectory_features: int = 4
    trajectory_hidden: int = 64
    # power
    power_dim: int = 64
    power_hidden: int = 256
    mixed_precision: bool = True
    modalities: Sequence[str] = field(default_factory=lambda: ("power", "camera", "lidar", "trajectory"))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class BaseEncoder(nn.Module):
    """Interface: ``forward(x) -> (..., embed_dim)``; leading batch/time dims are preserved."""

    embed_dim: int


# --------------------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------------------


class PowerEncoder(BaseEncoder):
    """MLP over the 64-beam power vector in dB (symlog-normalised)."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.power_dim, cfg.power_hidden), nn.LayerNorm(cfg.power_hidden), nn.SiLU(),
            nn.Linear(cfg.power_hidden, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim), nn.SiLU(),
        )

    def forward(self, power_db: torch.Tensor) -> torch.Tensor:
        return self.net(symlog(power_db))


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------


class CameraEncoder(BaseEncoder):
    """ResNet backbone (frozen by default) + trainable projection to ``embed_dim``.

    ``forward(images)`` takes ``(B, 3, H, W)`` or ``(B, T, 3, H, W)`` normalised tensors.
    ``forward_features(feats)`` takes cached backbone features ``(..., feature_dim)``.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        import torchvision

        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        ctor = {"resnet18": torchvision.models.resnet18, "resnet34": torchvision.models.resnet34}[cfg.camera_backbone]
        weights = "IMAGENET1K_V1" if cfg.camera_pretrained else None
        net = ctor(weights=weights)
        self.feature_dim = net.fc.in_features  # 512 for resnet18/34
        net.fc = nn.Identity()
        self.backbone = net.to(memory_format=torch.channels_last)
        self.frozen = cfg.freeze_camera_backbone
        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        self.proj = nn.Sequential(nn.Linear(self.feature_dim, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim), nn.SiLU())

    def train(self, mode: bool = True) -> CameraEncoder:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # keep BN statistics fixed
        return self

    # ---- preprocessing
    def preprocess(self, paths: Iterable[str | Path]) -> torch.Tensor:
        """Load JPEGs -> ``(N, 3, image_size, image_size)`` ImageNet-normalised float tensor."""
        from PIL import Image

        s = self.cfg.image_size
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        out = []
        for p in paths:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((s, s), Image.BILINEAR)
                x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            out.append((x - mean) / std)
        return torch.stack(out)

    # ---- forward
    def backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` -> ``(N, feature_dim)`` pooled backbone features."""
        images = images.contiguous(memory_format=torch.channels_last)
        enabled = self.cfg.mixed_precision and images.is_cuda
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            return self.backbone(images).float()

    def forward_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        lead = images.shape[:-3]
        feats = self.backbone_features(images.reshape(-1, *images.shape[-3:]))
        return self.proj(feats).reshape(*lead, self.embed_dim)

    @torch.no_grad()
    def encode_paths(self, paths: Sequence[str | Path], batch_size: int = 32,
                     device: torch.device | str | None = None) -> np.ndarray:
        """Compute backbone features for image files -> ``(N, feature_dim)`` numpy (for caching)."""
        device = torch.device(device) if device is not None else next(self.backbone.parameters()).device
        out = []
        for s in range(0, len(paths), batch_size):
            x = self.preprocess(paths[s : s + batch_size]).to(device)
            out.append(self.backbone_features(x).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, self.feature_dim), dtype=np.float32)


# --------------------------------------------------------------------------------------
# LiDAR
# --------------------------------------------------------------------------------------


class MiniPointNet(nn.Module):
    """Shared MLP + max-pool over a group of points -> one token."""

    def __init__(self, in_dim: int, token_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, hidden), nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, token_dim))

    def forward(self, groups: torch.Tensor) -> torch.Tensor:  # (G, S, P) -> (G, token_dim)
        f = self.mlp1(groups)
        g = f.max(dim=-2, keepdim=True).values.expand_as(f)
        return self.mlp2(torch.cat([f, g], dim=-1)).max(dim=-2).values


class LidarTokenizer:
    """Offline: ``.ply`` -> (tokens, centroids, groups).  Deterministic (seeded, frozen weights)."""

    def __init__(self, cfg: EncoderConfig, weights: str | Path | None = None, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        torch.manual_seed(cfg.lidar_seed)
        self.net = MiniPointNet(in_dim=4, token_dim=cfg.lidar_token_dim).to(self.device).eval()
        if weights is not None:
            self.net.load_state_dict(torch.load(weights, map_location=self.device))
        for p in self.net.parameters():
            p.requires_grad_(False)

    def group(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        if points.shape[1] < 4:  # no intensity column: append zeros
            points = np.concatenate([points, np.zeros((len(points), 4 - points.shape[1]), np.float32)], 1)
        groups, centroids = group_point_cloud(points[:, :4], cfg.lidar_n_points, cfg.lidar_n_groups,
                                              cfg.lidar_group_size, cfg.lidar_max_range_m, cfg.lidar_seed)
        groups[:, :, 3] = groups[:, :, 3] / 255.0  # intensity to [0, 1]
        return groups, centroids

    @torch.no_grad()
    def tokenize_points(self, points: np.ndarray) -> dict[str, np.ndarray]:
        groups, centroids = self.group(points)
        tokens = self.net(torch.from_numpy(groups).to(self.device)).cpu().numpy().astype(np.float32)
        return {"tokens": tokens, "centroids": centroids, "groups": groups}

    def tokenize_file(self, path: str | Path) -> dict[str, np.ndarray]:
        return self.tokenize_points(read_ply_points(path))

    def tokenize_to_cache(self, path: str | Path, out_path: str | Path) -> dict[str, np.ndarray]:
        out = self.tokenize_file(path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **out)
        return out


class LidarEncoder(BaseEncoder):
    """Online: cached tokens ``(..., G, D)`` + centroids ``(..., G, 3)`` -> ``(..., embed_dim)``."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        d = cfg.embed_dim
        self.token_in = nn.Linear(cfg.lidar_token_dim, d)
        self.pos = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, d))
        self.scale = cfg.lidar_max_range_m
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.lidar_layers, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        lead = tokens.shape[:-2]
        t = tokens.reshape(-1, *tokens.shape[-2:])
        c = centroids.reshape(-1, *centroids.shape[-2:]) / self.scale
        x = self.token_in(t) + self.pos(c)
        x = torch.cat([self.query.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0]).reshape(*lead, self.embed_dim)


# --------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------


class TrajectoryEncoder(BaseEncoder):
    """MLP over ``(..., trajectory_tokens, trajectory_features)`` GPS context."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        n_in = cfg.trajectory_tokens * cfg.trajectory_features
        self.net = nn.Sequential(
            nn.Linear(n_in, cfg.trajectory_hidden), nn.LayerNorm(cfg.trajectory_hidden), nn.SiLU(),
            nn.Linear(cfg.trajectory_hidden, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim), nn.SiLU(),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        lead = traj.shape[:-2]
        return self.net(traj.reshape(*lead, -1))


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


class MultimodalEncoder(nn.Module):
    """Fuse available modalities into ``(B, T, embed_dim)``.

    Batch keys (all optional except ``power_db``): ``power_db (B,T,64)``, ``camera_feat (B,T,F)``
    cached backbone features **or** ``camera_images (B,T,3,H,W)``, ``lidar_tokens (B,T,G,D)`` +
    ``lidar_centroids (B,T,G,3)``, ``trajectory (B,T,K,4)``.  Missing modalities contribute a
    learned "absent" vector so the same network trains with any subset.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.mods = tuple(cfg.modalities)
        self.encoders = nn.ModuleDict()
        if "power" in self.mods:
            self.encoders["power"] = PowerEncoder(cfg)
        if "camera" in self.mods:
            self.encoders["camera"] = CameraEncoder(cfg)
        if "lidar" in self.mods:
            self.encoders["lidar"] = LidarEncoder(cfg)
        if "trajectory" in self.mods:
            self.encoders["trajectory"] = TrajectoryEncoder(cfg)
        self.absent = nn.ParameterDict({m: nn.Parameter(torch.zeros(cfg.embed_dim)) for m in self.mods})
        self.fuse = nn.Sequential(nn.Linear(len(self.mods) * cfg.embed_dim, cfg.embed_dim),
                                  nn.LayerNorm(cfg.embed_dim), nn.SiLU())

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        lead = batch["power_db"].shape[:-1]
        for m in self.mods:
            enc = self.encoders[m]
            if m == "power":
                parts.append(enc(batch["power_db"]))
            elif m == "camera" and "camera_feat" in batch:
                parts.append(enc.forward_features(batch["camera_feat"]))
            elif m == "camera" and "camera_images" in batch:
                parts.append(enc(batch["camera_images"]))
            elif m == "lidar" and "lidar_tokens" in batch:
                parts.append(enc(batch["lidar_tokens"], batch["lidar_centroids"]))
            elif m == "trajectory" and "trajectory" in batch:
                parts.append(enc(batch["trajectory"]))
            else:
                parts.append(self.absent[m].expand(*lead, -1))
        return self.fuse(torch.cat(parts, dim=-1))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


__all__ = [
    "BaseEncoder",
    "CameraEncoder",
    "EncoderConfig",
    "LidarEncoder",
    "LidarTokenizer",
    "MiniPointNet",
    "MultimodalEncoder",
    "PowerEncoder",
    "TrajectoryEncoder",
    "count_parameters",
    "symexp",
    "symlog",
]
_ = F  # keep torch.nn.functional import available for subclasses
