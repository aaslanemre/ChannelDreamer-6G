"""Side-modality loading for DeepSense scenarios: GPS trajectories, LiDAR point clouds, camera paths.

All heavy per-sample work (LiDAR grouping / tokenisation, frozen camera backbone features) is
meant to run **offline once** through ``prepare_data.py`` and be cached under
``<scenario_dir>/cache/``; training then reads small arrays and never touches raw point
clouds or JPEGs.  This is how the multimodal world model fits in 12 GB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .deepsense import DeepSenseScenario

EARTH_RADIUS_M = 6_371_000.0


# --------------------------------------------------------------------------------------
# GPS / trajectory
# --------------------------------------------------------------------------------------


def read_gps_file(path: str | Path) -> np.ndarray:
    """Read a DeepSense GPS text file (``lat`` and ``lon`` on separate lines) -> ``(2,)``."""
    vals = np.loadtxt(path, dtype=np.float64)
    vals = np.atleast_1d(vals)
    if vals.size < 2:
        raise ValueError(f"{path}: expected at least lat, lon; got {vals}")
    return vals[:2]


def latlon_to_local_m(latlon: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Equirectangular projection of ``(N, 2)`` lat/lon (deg) to ``(N, 2)`` east/north metres."""
    lat0, lon0 = np.deg2rad(origin[0]), np.deg2rad(origin[1])
    lat, lon = np.deg2rad(latlon[:, 0]), np.deg2rad(latlon[:, 1])
    east = (lon - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    north = (lat - lat0) * EARTH_RADIUS_M
    return np.stack([east, north], axis=1)


@dataclass
class GPSData:
    user_latlon: np.ndarray  # (N, 2)
    bs_latlon: np.ndarray  # (2,)
    user_local_m: np.ndarray  # (N, 2) east/north relative to the base station
    speed_kmph: np.ndarray | None  # (N,) if present in the CSV
    n_missing: int  # rows whose GPS file was missing / unparsable (filled by neighbour)


def load_gps(ds: DeepSenseScenario, user_unit: str = "unit2", bs_unit: str = "unit1") -> GPSData:
    """Load per-sample user GPS (``unit2_loc`` files) and the base-station position (``unit1_loc``)."""
    user_key = next((k for k in ds.modality_paths if k.startswith("gps:") and user_unit in k), None)
    bs_key = next((k for k in ds.modality_paths if k.startswith("gps:") and bs_unit in k), None)
    if user_key is None:
        raise KeyError(f"no GPS path column for {user_unit}; modality_paths = {list(ds.modality_paths)}")
    paths = ds.modality_paths[user_key]
    user = np.full((len(paths), 2), np.nan)
    missing = 0
    for i, p in enumerate(paths):
        try:
            user[i] = read_gps_file(p)
        except (OSError, ValueError):
            missing += 1
    # forward/backward fill missing rows within the array (rare)
    if missing:
        good = ~np.isnan(user).any(axis=1)
        idx = np.where(good, np.arange(len(user)), 0)
        np.maximum.accumulate(idx, out=idx)
        user = user[idx]
        if not good[0]:
            first = np.argmax(good)
            user[:first] = user[first]
    bs = read_gps_file(ds.modality_paths[bs_key][0]) if bs_key else np.nanmean(user, axis=0)
    speed = None
    if ds.frame is not None:
        col = next((c for c in ds.frame.columns if re.search(r"spd|speed", c, re.IGNORECASE)), None)
        if col is not None:
            speed = ds.frame[col].to_numpy(dtype=np.float64)
    return GPSData(user, bs, latlon_to_local_m(user, bs), speed, missing)


def trajectory_windows(
    local_m: np.ndarray,
    segment_ids: np.ndarray,
    n_tokens: int = 16,
    scale_m: float = 50.0,
) -> np.ndarray:
    """``(N, n_tokens, 4)`` trajectory context per step: the last ``n_tokens`` positions
    (east, north, normalised by ``scale_m``) and their per-step displacement.

    Steps before the start of a segment are clamped to the segment's first sample, so the
    context never crosses a segment boundary.
    """
    n = len(local_m)
    seg = np.asarray(segment_ids)
    out = np.zeros((n, n_tokens, 4), dtype=np.float32)
    starts = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        starts[i] = starts[i - 1] if seg[i] == seg[i - 1] else i
    for t in range(n):
        idx = np.clip(np.arange(t - n_tokens + 1, t + 1), starts[t], t)
        pos = local_m[idx] / scale_m
        vel = (local_m[idx] - local_m[np.clip(idx - 1, starts[t], t)]) / scale_m
        out[t, :, :2] = pos
        out[t, :, 2:] = vel
    return out


# --------------------------------------------------------------------------------------
# LiDAR
# --------------------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_points(path: str | Path) -> np.ndarray:
    """Read a PLY vertex list (ASCII or binary little/big endian) -> ``(N, P)`` float32 array of
    the scalar vertex properties in file order (DeepSense: x, y, z, intensity)."""
    path = Path(path)
    with open(path, "rb") as fh:
        header: list[str] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        fmt = next(h.split()[1] for h in header if h.startswith("format"))
        n_vertex, props, in_vertex = 0, [], False
        for h in header:
            parts = h.split()
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices are not supported")
                props.append((parts[2], _PLY_TYPES[parts[1]]))
        if fmt == "ascii":
            data = np.loadtxt(fh, dtype=np.float64, max_rows=n_vertex)
            return np.atleast_2d(data).astype(np.float32)
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + t) for name, t in props])
        raw = np.frombuffer(fh.read(dtype.itemsize * n_vertex), dtype=dtype, count=n_vertex)
        return np.stack([raw[name].astype(np.float32) for name, _ in props], axis=1)


def farthest_point_sampling(xyz: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Indices of ``n`` farthest points (greedy), starting from a seeded random point."""
    m = len(xyz)
    if n >= m:
        return np.arange(m)
    rng = np.random.default_rng(seed)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(m)
    dist = np.full(m, np.inf)
    for i in range(1, n):
        dist = np.minimum(dist, ((xyz - xyz[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(np.argmax(dist))
    return idx


def group_point_cloud(
    points: np.ndarray,
    n_points: int = 2048,
    n_groups: int = 16,
    group_size: int = 32,
    max_range_m: float | None = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-BERT-style grouping: subsample, FPS centroids, kNN groups.

    Returns ``groups (n_groups, group_size, P)`` with xyz **relative to the centroid** (extra
    columns such as intensity kept as-is) and ``centroids (n_groups, 3)`` in the sensor frame.
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float32)
    if max_range_m is not None:
        pts = pts[np.linalg.norm(pts[:, :3], axis=1) <= max_range_m]
    if len(pts) == 0:
        pts = np.zeros((1, points.shape[1]), dtype=np.float32)
    if len(pts) > n_points:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    cidx = farthest_point_sampling(pts[:, :3], n_groups, seed=seed)
    centroids = pts[cidx, :3]
    if len(centroids) < n_groups:  # tiny cloud: repeat centroids
        centroids = centroids[np.resize(np.arange(len(centroids)), n_groups)]
    d2 = ((pts[None, :, :3] - centroids[:, None, :]) ** 2).sum(-1)  # (G, N)
    k = min(group_size, len(pts))
    nn = np.argsort(d2, axis=1)[:, :k]
    if k < group_size:
        nn = nn[:, np.resize(np.arange(k), group_size)]
    groups = pts[nn].copy()  # (G, S, P)
    groups[:, :, :3] -= centroids[:, None, :]
    return groups.astype(np.float32), centroids.astype(np.float32)


def lidar_cache_path(cache_dir: str | Path, sample_index: int) -> Path:
    return Path(cache_dir) / f"lidar_tokens_{int(sample_index)}.npz"


def load_lidar_tokens(cache_dir: str | Path, sample_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load cached ``tokens (N, G, D)`` and ``centroids (N, G, 3)`` for the given sample indices."""
    toks, cents = [], []
    for i in sample_index:
        with np.load(lidar_cache_path(cache_dir, i)) as z:
            toks.append(z["tokens"])
            cents.append(z["centroids"])
    return np.stack(toks), np.stack(cents)


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------


def camera_cache_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "camera_features.npz"


def load_camera_features(cache_dir: str | Path, sample_index: np.ndarray) -> np.ndarray:
    """Load cached frozen-backbone features ``(N, F)`` aligned to ``sample_index``."""
    with np.load(camera_cache_path(cache_dir)) as z:
        feats, idx = z["features"], z["sample_index"]
    lookup = {int(i): k for k, i in enumerate(idx)}
    try:
        rows = [lookup[int(i)] for i in sample_index]
    except KeyError as exc:
        raise KeyError(f"sample index {exc} not in camera cache {camera_cache_path(cache_dir)}") from exc
    return feats[rows]
