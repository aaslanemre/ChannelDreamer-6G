"""Multimodal encoders for side information (camera, LiDAR, GPS trajectory).  **STUB.**

VRAM budget: everything must train on a single **RTX 4070 with 12 GB**.  Consequently:

* ``CameraEncoder``   - ResNet-18 (ImageNet init) with frozen early stages, 224x224 inputs,
                        bf16 autocast, channels-last.  Feature dim 512.
* ``LidarEncoder``    - Point-BERT-style tokenizer.  Point clouds are **pre-tokenised
                        offline** (FPS + kNN grouping -> mini-PointNet -> discrete tokens via
                        dVAE codebook) by ``prepare_data.py`` and cached as ``.npz``; only the
                        small transformer over tokens is trained online.  This keeps the
                        per-batch point-cloud memory off the GPU.
* ``TrajectoryEncoder`` - MLP over the last ``H`` GPS (lat, lon, speed, heading) samples,
                        normalised per scenario.

All encoders map to a common ``embed_dim`` so they can be concatenated / summed before the
RSSM posterior.  Batch sizes and image resolution are configurable in YAML; defaults are
chosen for 12 GB (batch 16 x seq 32 for camera-augmented training).
"""

from __future__ import annotations

from typing import Any


class BaseEncoder:
    """Interface: ``forward(x) -> (B, T, embed_dim)``."""

    embed_dim: int

    def forward(self, x: Any) -> Any:
        raise NotImplementedError


class PowerEncoder(BaseEncoder):
    """MLP over the 64-dim beam power (dB, symlog-normalised).  Phase 3."""

    def __init__(self, obs_dim: int = 64, embed_dim: int = 256):
        self.obs_dim, self.embed_dim = obs_dim, embed_dim
        raise NotImplementedError


class CameraEncoder(BaseEncoder):
    """ResNet-18 backbone; see module docstring for the 12 GB memory plan.  Phase 4."""

    def __init__(self, embed_dim: int = 256, freeze_stages: int = 2, image_size: int = 224):
        raise NotImplementedError


class LidarEncoder(BaseEncoder):
    """Point-BERT-style transformer over *offline pre-tokenised* LiDAR groups.  Phase 4."""

    def __init__(self, embed_dim: int = 256, n_groups: int = 64, codebook_size: int = 8192):
        raise NotImplementedError

    @staticmethod
    def pretokenize(ply_paths: list[str], out_path: str, n_groups: int = 64, group_size: int = 32) -> None:
        """Offline: FPS + kNN grouping of every point cloud, saved as ``.npz`` token ids."""
        raise NotImplementedError


class TrajectoryEncoder(BaseEncoder):
    """MLP over the last H GPS/speed/heading samples.  Phase 4."""

    def __init__(self, history: int = 8, embed_dim: int = 64):
        raise NotImplementedError
