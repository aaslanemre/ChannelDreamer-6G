"""WiWorld-RealData complex-CIR loader (Phase 2 scaffold).

**Real-data validation pending WiWorld-RealData download — see docs/data_setup.md.**

What is documented so far (docs/positioning.md, C3): WiWorld-RealData provides *complex-valued
channel impulse responses* (CIR) measured on a **dual-band** link (3.7 GHz and 6.775 GHz) along
a **single public route**, with a per-sample **quality flag**.  DeepSense provides beam-domain
power only, so this dataset is what the physics-grounded latent constraints (IDM, VICReg,
SGCS, phase consistency) will be trained on.

Because the manifest has not been inspected yet, **nothing about its column names or file
layout is hard-coded**.  The loader follows the DeepSense loader pattern
(:mod:`channeldreamer.data.deepsense`):

1. resolve the dataset folder (``root/wiworld*`` or an explicit path);
2. find the manifest (shortest ``*.csv`` / ``*.json`` / ``*.parquet``, configurable);
3. resolve columns through :class:`WiWorldColumns`: explicit names win, otherwise regex
   patterns are tried, otherwise the column is reported as unresolved (never guessed);
4. load one complex CIR per sample and per band from the referenced files, validating that
   every sample has the same shape and that the values are complex (or a documented
   real/imag layout, see :class:`WiWorldLayout`);
5. parse the quality flag into a boolean ``good`` mask (which values count as good is
   configurable) and optionally drop bad samples.

Everything is exercised against :func:`channeldreamer.data.synthetic.generate_synthetic_cir`,
whose :meth:`~channeldreamer.data.synthetic.SyntheticCIRData.write_wiworld_layout` writes a
manifest + per-sample files in the *assumed* layout, so the loader can be tested end to end
with zero real data.  When the real data arrive, adjust :class:`WiWorldColumns` /
:class:`WiWorldLayout` to the actual manifest and re-run ``prepare_data``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BANDS_HZ: tuple[float, ...] = (3.7e9, 6.775e9)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class WiWorldColumns:
    """Manifest column resolution.  Explicit names take precedence over regex patterns.

    ``cir`` may be a single column referencing a file holding *all* bands, or one column per
    band (``cir_per_band``), in the order of :attr:`WiWorldLayout.bands_hz`.
    """

    cir: str | None = None
    cir_per_band: Sequence[str] | None = None
    quality: str | None = None
    time: str | None = None
    position: Sequence[str] | None = None  # e.g. ("lat", "lon") or ("x", "y")
    segment: str | None = None  # route / run / segment identifier
    index: str | None = None

    patterns: dict[str, str] = field(
        default_factory=lambda: {
            "cir": r"cir|impulse|channel|csi|h_",
            "quality": r"quality|qual|flag|valid|status",
            "time": r"time|stamp|utc",
            "position": r"(^|_)(lat|lon|x|y|east|north|pos)(_|$)",
            "segment": r"seg|route|run|track|trajectory",
            "index": r"^index$|^id$|sample",
        }
    )


@dataclass
class WiWorldLayout:
    """How CIR samples are stored on disk (all assumptions, all configurable)."""

    bands_hz: tuple[float, ...] = DEFAULT_BANDS_HZ
    file_format: str = "npy"  # "npy" | "npz" | "txt_complex"
    npz_key: str | None = None  # key inside .npz holding the CIR; None -> first array
    complex_layout: str = "native"  # "native" complex dtype | "last_axis_ri" (..., 2) | "first_axis_ri" (2, ...)
    band_axis: int | None = 0  # axis of the per-sample array indexing bands (single-file mode)
    good_quality_values: tuple[Any, ...] = (1, True, "1", "true", "good", "ok", "valid")
    manifest_glob: tuple[str, ...] = ("*.csv", "*.json", "*.parquet")


# --------------------------------------------------------------------------------------
# Resolution helpers
# --------------------------------------------------------------------------------------


def resolve_wiworld_dir(root: str | Path, name: str = "wiworld") -> Path:
    """``root/<name>*`` (case-insensitive) or an explicit existing directory."""
    p = Path(name)
    if p.is_dir():
        return p
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root {root} does not exist")
    cands = [c for c in sorted(root.iterdir()) if c.is_dir() and c.name.lower().startswith(name.lower())]
    if not cands:
        raise FileNotFoundError(f"no folder starting with {name!r} under {root}: {[c.name for c in root.iterdir()]}")
    return cands[0]


def find_manifest(dataset_dir: str | Path, globs: Iterable[str] = WiWorldLayout().manifest_glob) -> Path:
    dataset_dir = Path(dataset_dir)
    found: list[Path] = []
    for g in globs:
        found += list(dataset_dir.glob(g)) or list(dataset_dir.glob(f"**/{g}"))
    if not found:
        raise FileNotFoundError(f"no manifest ({', '.join(globs)}) under {dataset_dir}")
    return min(found, key=lambda p: (len(p.name), p.name))


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.DataFrame(data["samples"] if isinstance(data, dict) and "samples" in data else data)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported manifest format {path.suffix}")


def resolve_wiworld_columns(header: Iterable[str], columns: WiWorldColumns | None = None) -> dict[str, Any]:
    """Return ``{key: column(s) or None}``.  Explicit names are validated against the header."""
    columns = columns or WiWorldColumns()
    header = list(header)
    out: dict[str, Any] = {}

    def _check(name: str) -> str:
        if name not in header:
            raise KeyError(f"manifest column {name!r} not found; header = {header}")
        return name

    def _match(key: str) -> list[str]:
        rx = re.compile(columns.patterns[key], re.IGNORECASE)
        return [c for c in header if rx.search(c)]

    if columns.cir_per_band is not None:
        out["cir_per_band"] = [_check(c) for c in columns.cir_per_band]
        out["cir"] = None
    else:
        out["cir_per_band"] = None
        if columns.cir is not None:
            out["cir"] = _check(columns.cir)
        else:
            m = _match("cir")
            out["cir"] = m[0] if len(m) == 1 else None
            if len(m) > 1:  # several candidates -> treat as per-band columns, in header order
                out["cir_per_band"] = m
    for key in ("quality", "time", "segment", "index"):
        explicit = getattr(columns, key)
        if explicit is not None:
            out[key] = _check(explicit)
        else:
            m = _match(key)
            out[key] = m[0] if m else None
    if columns.position is not None:
        out["position"] = [_check(c) for c in columns.position]
    else:
        out["position"] = _match("position") or None
    return out


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_cir_file(path: Path, layout: WiWorldLayout) -> np.ndarray:
    if layout.file_format == "npy":
        arr = np.load(path)
    elif layout.file_format == "npz":
        with np.load(path) as z:
            key = layout.npz_key or next(iter(z.keys()))
            arr = z[key]
    elif layout.file_format == "txt_complex":
        arr = np.loadtxt(path, dtype=np.complex128)
    else:
        raise ValueError(f"unknown file_format {layout.file_format!r}")
    return _to_complex(np.asarray(arr), layout.complex_layout, path)


def _to_complex(arr: np.ndarray, complex_layout: str, path: Path | None = None) -> np.ndarray:
    if complex_layout == "native":
        if not np.iscomplexobj(arr):
            raise ValueError(f"{path}: expected complex dtype, got {arr.dtype} (set complex_layout)")
        return arr.astype(np.complex64)
    if complex_layout == "last_axis_ri":
        if arr.shape[-1] != 2:
            raise ValueError(f"{path}: last axis must be (re, im), got shape {arr.shape}")
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)
    if complex_layout == "first_axis_ri":
        if arr.shape[0] != 2:
            raise ValueError(f"{path}: first axis must be (re, im), got shape {arr.shape}")
        return (arr[0] + 1j * arr[1]).astype(np.complex64)
    raise ValueError(f"unknown complex_layout {complex_layout!r}")


def parse_quality(values: Iterable[Any], good_values: Iterable[Any]) -> np.ndarray:
    """Boolean mask: True where the quality flag is one of ``good_values`` (case-insensitive strings)."""
    good = {str(v).strip().lower() for v in good_values}
    return np.array([str(v).strip().lower() in good for v in values], dtype=bool)


@dataclass
class WiWorldDataset:
    dataset_dir: Path
    manifest_path: Path
    columns: dict[str, Any]
    layout: WiWorldLayout
    cir: np.ndarray  # (N, n_bands, ...) complex64
    good: np.ndarray  # (N,) bool quality mask
    segment_ids: np.ndarray  # (N,)
    sample_index: np.ndarray  # (N,)
    timestamps: np.ndarray | None = None
    positions: np.ndarray | None = None  # (N, D)
    quality_raw: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return int(self.cir.shape[0])

    @property
    def n_bands(self) -> int:
        return int(self.cir.shape[1])

    def power_delay_profile_db(self, floor: float = 1e-12) -> np.ndarray:
        """``(N, n_bands, ...)`` |h|^2 in dB, a real-valued view useful for regime labelling."""
        return 10.0 * np.log10(np.maximum(np.abs(self.cir) ** 2, floor))

    def summary(self) -> str:
        segs = np.unique(self.segment_ids)
        return "\n".join([
            f"dataset dir   : {self.dataset_dir}",
            f"manifest      : {self.manifest_path.name}",
            f"samples       : {self.n_samples} ({int(self.good.sum())} good, {int((~self.good).sum())} flagged)",
            f"cir shape     : {self.cir.shape} {self.cir.dtype}  bands={self.layout.bands_hz}",
            f"segments      : {len(segs)}",
            f"columns       : {self.columns}",
        ])


def load_wiworld(
    root: str | Path,
    name: str = "wiworld",
    *,
    columns: WiWorldColumns | None = None,
    layout: WiWorldLayout | None = None,
    drop_bad: bool = False,
    max_samples: int | None = None,
) -> WiWorldDataset:
    """Load a WiWorld-style dataset into memory (see module docstring for the assumptions)."""
    layout = layout or WiWorldLayout()
    dataset_dir = resolve_wiworld_dir(root, name)
    manifest_path = find_manifest(dataset_dir, layout.manifest_glob)
    frame = read_manifest(manifest_path)
    if max_samples is not None:
        frame = frame.iloc[:max_samples].reset_index(drop=True)
    cols = resolve_wiworld_columns(frame.columns, columns)
    if cols["cir"] is None and cols["cir_per_band"] is None:
        raise KeyError(f"could not resolve a CIR column in {manifest_path.name}; header = {list(frame.columns)}")

    n_bands = len(layout.bands_hz)
    samples: list[np.ndarray] = []
    for i in range(len(frame)):
        if cols["cir_per_band"] is not None:
            if len(cols["cir_per_band"]) != n_bands:
                raise ValueError(f"{len(cols['cir_per_band'])} per-band columns but layout lists {n_bands} bands")
            bands = [_read_cir_file(dataset_dir / str(frame[c].iloc[i]).lstrip("./"), layout) for c in cols["cir_per_band"]]
            arr = np.stack(bands, axis=0)
        else:
            arr = _read_cir_file(dataset_dir / str(frame[cols["cir"]].iloc[i]).lstrip("./"), layout)
            if layout.band_axis is None:
                arr = arr[None]
            elif layout.band_axis != 0:
                arr = np.moveaxis(arr, layout.band_axis, 0)
            if arr.shape[0] != n_bands:
                raise ValueError(f"sample {i}: expected {n_bands} bands on axis 0, got shape {arr.shape}")
        if samples and arr.shape != samples[0].shape:
            raise ValueError(f"sample {i}: shape {arr.shape} differs from first sample {samples[0].shape}")
        samples.append(arr)
    cir = np.stack(samples) if samples else np.empty((0, n_bands, 0), dtype=np.complex64)

    if cols["quality"] is not None:
        quality_raw = frame[cols["quality"]].to_numpy()
        good = parse_quality(quality_raw, layout.good_quality_values)
    else:
        quality_raw, good = None, np.ones(len(frame), dtype=bool)

    if cols["segment"] is not None:
        _, seg = np.unique(frame[cols["segment"]].astype(str).to_numpy(), return_inverse=True)
    else:
        seg = np.zeros(len(frame), dtype=np.int64)
    idx = (pd.to_numeric(frame[cols["index"]], errors="coerce").fillna(-1).to_numpy(np.int64)
           if cols["index"] is not None else np.arange(len(frame)))
    ts = pd.to_numeric(frame[cols["time"]], errors="coerce").to_numpy(float) if cols["time"] is not None else None
    pos = frame[cols["position"]].apply(pd.to_numeric, errors="coerce").to_numpy(float) if cols["position"] else None

    ds = WiWorldDataset(dataset_dir, manifest_path, cols, layout, cir, good, seg.astype(np.int64), idx,
                        ts, pos, quality_raw)
    if drop_bad:
        keep = ds.good
        ds = WiWorldDataset(ds.dataset_dir, ds.manifest_path, ds.columns, ds.layout, ds.cir[keep], ds.good[keep],
                            ds.segment_ids[keep], ds.sample_index[keep],
                            None if ts is None else ts[keep], None if pos is None else pos[keep],
                            None if quality_raw is None else quality_raw[keep])
    return ds
