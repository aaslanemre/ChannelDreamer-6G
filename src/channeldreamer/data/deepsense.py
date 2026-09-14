"""DeepSense 6G scenario loader (beam-domain power vectors).

The observation used throughout this project is the 64-element *received power per beam*
vector (one value per codebook beam of the 64-beam phased array at unit1), **not** complex
CSI.  The optimal beam is simply ``argmax`` over that vector.

DeepSense stores each scenario as a CSV index whose rows reference per-sample files::

    index,unit1_rgb,unit1_pwr_60ghz,unit1_lidar,unit1_radar,unit1_loc,unit2_loc,
    unit1_beam,unit1_max_pwr,time_stamp,seq_index,<gps fields...>

``unit1_pwr_60ghz`` points at a ``.txt`` file with exactly 64 lines (one float per line).
``unit1_beam`` / ``unit1_max_pwr`` are precomputed labels which we cross-validate against
``argmax(power)``.  Note that released scenarios use **1-based** beam labels; the loader
detects the offset (0 or 1) empirically rather than assuming it.

Column names vary slightly between scenario releases, so columns are resolved by regex
rather than by exact name.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import NUM_BEAMS

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Path / column resolution
# --------------------------------------------------------------------------------------

COLUMN_PATTERNS: dict[str, str] = {
    # value: regex applied case-insensitively to column names
    "power": r"(pwr|power)(?!.*max)",  # unit1_pwr_60ghz, but not unit1_max_pwr
    "beam_label": r"(^|_)beam(_|$)(?!.*pwr)",
    "max_power": r"max_?pwr|max_?power",
    "image": r"rgb|image|img|camera",
    "lidar": r"lidar",
    "radar": r"radar",
    "gps": r"(^|_)(loc|gps|position)(_|$)|lat|lon",
    "time": r"time|stamp",
    "sequence": r"seq",
}

# Columns that are single references (one column) vs. sets (many columns)
_MULTI_COLUMNS = {"gps"}


def resolve_scenario_dir(root: str | Path, scenario: int | str) -> Path:
    """Find ``root/scenarioN`` (case-insensitive, any separator) or ``root/N``.

    Accepts an integer (``33``), a name (``"scenario33"``), or an existing directory path.
    """
    root = Path(root)
    p = Path(str(scenario))
    if p.is_dir():
        return p
    if (root / str(scenario)).is_dir():
        return root / str(scenario)

    num = re.sub(r"\D", "", str(scenario))
    if not root.is_dir():
        raise FileNotFoundError(f"data root {root} does not exist")
    candidates = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = re.fullmatch(r"(?i)scen[a-z]*[_\- ]?(\d+)", child.name) or re.fullmatch(r"(\d+)", child.name)
        if m and num and m.group(1) == num:
            candidates.append(child)
    if not candidates:
        listing = ", ".join(c.name for c in root.iterdir() if c.is_dir())
        raise FileNotFoundError(
            f"no scenario folder matching {scenario!r} under {root} (found: {listing})"
        )
    return candidates[0]


def find_csv(scenario_dir: str | Path) -> Path:
    """Shortest ``.csv`` filename containing 'dev', else the shortest ``.csv`` overall."""
    scenario_dir = Path(scenario_dir)
    csvs = sorted(scenario_dir.glob("*.csv"), key=lambda p: (len(p.name), p.name))
    if not csvs:
        csvs = sorted(scenario_dir.glob("**/*.csv"), key=lambda p: (len(p.name), p.name))
    if not csvs:
        raise FileNotFoundError(f"no .csv index found under {scenario_dir}")
    dev = [c for c in csvs if "dev" in c.name.lower()]
    return dev[0] if dev else csvs[0]


def resolve_columns(columns: Iterable[str], preferred_unit: str = "unit1") -> dict[str, object]:
    """Map semantic keys (power, image, ...) to actual CSV column names via regex.

    Returns a dict ``{key: column_name_or_None}`` (``gps`` maps to a list of columns).
    When several columns match, columns whose name contains ``preferred_unit`` win.
    """
    cols = list(columns)
    resolved: dict[str, object] = {}
    for key, pattern in COLUMN_PATTERNS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        matches = [c for c in cols if rx.search(c)]
        if key == "power":
            # the raw power *reference* column must not be the scalar max-power label
            matches = [c for c in matches if not re.search(r"max", c, re.IGNORECASE)]
        if key == "beam_label":
            matches = [c for c in matches if not re.search(r"pwr|power", c, re.IGNORECASE)]
        if key in _MULTI_COLUMNS:
            resolved[key] = matches
            continue
        if not matches:
            resolved[key] = None
            continue
        pref = [c for c in matches if preferred_unit in c.lower()]
        resolved[key] = (pref or matches)[0]
    return resolved


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def optimal_beam(power: np.ndarray) -> np.ndarray:
    """Optimal beam index (0-based) = argmax of the beam-power vector along the last axis.

    NaN entries (dropped sweep measurements, present in a few DeepSense rows) are ignored,
    matching how the released ``unit1_beam`` labels were computed.
    """
    power = np.asarray(power)
    if power.dtype.kind == "f" and np.isnan(power).any():
        return np.nanargmax(power, axis=-1)
    return np.argmax(power, axis=-1)


def fill_nan_power(power: np.ndarray, policy: str = "interpolate") -> tuple[np.ndarray, int]:
    """Handle NaN entries inside beam-power vectors.  Returns (power, n_rows_affected).

    ``"interpolate"``  linear interpolation along the *beam axis* (the beam pattern is smooth
                       across adjacent codebook beams), edges extended; a row that is all-NaN
                       falls back to the previous finite row.
    ``"keep"``         leave NaNs in place (argmax stays NaN-safe, metrics will not be).
    ``"raise"``        raise ``ValueError`` if any NaN is present.
    """
    power = np.asarray(power, dtype=np.float64)
    bad = ~np.isfinite(power)
    n_rows = int(bad.any(axis=1).sum())
    if n_rows == 0 or policy == "keep":
        return power, n_rows
    if policy == "raise":
        raise ValueError(f"{n_rows} power rows contain non-finite values")
    if policy != "interpolate":
        raise ValueError(f"unknown nan policy {policy!r}")
    out = power.copy()
    idx = np.arange(power.shape[1])
    last_good: np.ndarray | None = None
    for i in range(out.shape[0]):
        row_bad = bad[i]
        if not row_bad.any():
            last_good = out[i]
            continue
        if row_bad.all():
            out[i] = last_good if last_good is not None else 0.0
            continue
        out[i, row_bad] = np.interp(idx[row_bad], idx[~row_bad], out[i, ~row_bad])
        last_good = out[i]
    return out, n_rows


def _read_power_file(path: Path, n_beams: int = NUM_BEAMS) -> np.ndarray:
    vals = np.loadtxt(path, dtype=np.float64)
    vals = np.atleast_1d(vals)
    if vals.shape != (n_beams,):
        raise ValueError(f"{path}: expected {n_beams} values, got shape {vals.shape}")
    return vals


def _parse_timestamps(series: pd.Series) -> np.ndarray | None:
    """Parse DeepSense ``HH:MM:SS-microseconds`` stamps to seconds; None if unparseable."""
    out = np.full(len(series), np.nan)
    rx = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:[-.:](\d+))?$")
    for i, s in enumerate(series.astype(str)):
        m = rx.match(s.strip())
        if not m:
            continue
        h, mi, se = (int(m.group(k)) for k in (1, 2, 3))
        frac = m.group(4)
        micro = float(f"0.{frac}") if frac else 0.0
        out[i] = h * 3600 + mi * 60 + se + micro
    if np.isnan(out).all():
        return None
    return out


@dataclass
class DeepSenseScenario:
    """In-memory DeepSense scenario: (N, 64) beam power + labels + modality file references."""

    scenario_dir: Path
    csv_path: Path
    columns: dict
    power: np.ndarray  # (N, 64), linear power as stored
    segment_ids: np.ndarray  # (N,) int, contiguous drive/pass-by identifier
    sample_index: np.ndarray  # (N,) original DeepSense row index
    beam_label: np.ndarray | None = None  # (N,) 0-based after offset correction
    beam_label_offset: int = 0  # 1 if the CSV labels were 1-based
    label_agreement: float | None = None  # fraction of rows where label == argmax(power)
    label_agreement_clean: float | None = None  # same, restricted to rows without NaN beams
    max_power_label: np.ndarray | None = None
    max_power_agreement: float | None = None
    timestamps_s: np.ndarray | None = None
    modality_paths: dict[str, list[Path]] = field(default_factory=dict)
    frame: pd.DataFrame | None = None
    n_nan_rows: int = 0  # rows whose raw power vector contained NaN entries
    nan_policy: str = "interpolate"

    @property
    def n_samples(self) -> int:
        return int(self.power.shape[0])

    @property
    def n_beams(self) -> int:
        return int(self.power.shape[1])

    def optimal_beam(self) -> np.ndarray:
        return optimal_beam(self.power)

    def summary(self) -> str:
        segs, counts = np.unique(self.segment_ids, return_counts=True)
        lines = [
            f"scenario dir      : {self.scenario_dir}",
            f"csv               : {self.csv_path.name}",
            f"samples           : {self.n_samples}",
            f"power shape       : {self.power.shape}",
            (f"segments          : {len(segs)} (len min/median/max = "
            f"{counts.min()}/{int(np.median(counts))}/{counts.max()})"),
            f"power range       : [{self.power.min():.4g}, {self.power.max():.4g}] (linear)",
        ]
        if self.beam_label is not None:
            lines.append(
                f"beam label        : offset={self.beam_label_offset} (CSV is "
                f"{'1' if self.beam_label_offset else '0'}-based), "
                f"agreement with argmax(power) = {100 * self.label_agreement:.2f}% "
                f"({100 * self.label_agreement_clean:.2f}% on rows without NaN beams)"
            )
        if self.max_power_label is not None:
            lines.append(
                f"max_pwr label     : agreement with max(power) = "
                f"{100 * self.max_power_agreement:.2f}%"
            )
        if self.timestamps_s is not None:
            dt = np.diff(self.timestamps_s)
            dt = dt[(dt > 0) & (dt < 5)]
            if dt.size:
                lines.append(f"median sample dt  : {np.median(dt) * 1e3:.1f} ms")
        return "\n".join(lines)


def _segment_ids_from_frame(
    frame: pd.DataFrame, columns: dict, timestamps: np.ndarray | None, time_gap_s: float | None
) -> np.ndarray:
    n = len(frame)
    seq_col = columns.get("sequence")
    if seq_col is not None:
        raw = pd.to_numeric(frame[seq_col], errors="coerce").to_numpy()
        # re-number contiguous runs so that non-contiguous reuse of an id still splits
        seg = np.zeros(n, dtype=np.int64)
        cur = 0
        for i in range(1, n):
            if raw[i] != raw[i - 1]:
                cur += 1
            seg[i] = cur
    else:
        seg = np.zeros(n, dtype=np.int64)
    if time_gap_s is not None and timestamps is not None:
        gaps = np.diff(timestamps)
        breaks = np.where(np.isnan(gaps) | (gaps > time_gap_s) | (gaps < 0))[0] + 1
        extra = np.zeros(n, dtype=np.int64)
        extra[breaks] = 1
        seg = seg + np.cumsum(extra)
        _, seg = np.unique(seg, return_inverse=True)
    return seg


def load_scenario(
    root: str | Path,
    scenario: int | str,
    *,
    max_samples: int | None = None,
    time_gap_s: float | None = None,
    keep_frame: bool = False,
    n_beams: int = NUM_BEAMS,
    nan_policy: str = "interpolate",
) -> DeepSenseScenario:
    """Load a DeepSense scenario into memory.

    Parameters
    ----------
    root, scenario
        ``root/scenarioN`` (or ``root/N``) is resolved by :func:`resolve_scenario_dir`.
    max_samples
        Optional cap on the number of rows read (useful for smoke tests).
    time_gap_s
        If given, additionally break segments where consecutive timestamps differ by more
        than this many seconds (on top of the CSV's own ``seq_index``).
    nan_policy
        How to treat NaN entries inside power vectors (see :func:`fill_nan_power`).  A few
        DeepSense rows have 1-9 dropped beams; the default interpolates across beams.
    """
    scenario_dir = resolve_scenario_dir(root, scenario)
    csv_path = find_csv(scenario_dir)
    frame = pd.read_csv(csv_path)
    if max_samples is not None:
        frame = frame.iloc[:max_samples].reset_index(drop=True)
    columns = resolve_columns(frame.columns)
    if columns["power"] is None:
        raise KeyError(
            f"could not resolve a beam-power column in {csv_path.name}; header = "
            f"{list(frame.columns)}"
        )

    power_col = columns["power"]
    power = np.empty((len(frame), n_beams), dtype=np.float64)
    for i, rel in enumerate(frame[power_col].astype(str)):
        power[i] = _read_power_file(scenario_dir / rel.lstrip("./"), n_beams)
    raw_argmax = optimal_beam(power)  # NaN-safe, compared with the released labels below
    clean_rows = np.isfinite(power).all(axis=1)
    power, n_nan_rows = fill_nan_power(power, nan_policy)
    if n_nan_rows:
        log.info("%d/%d rows had NaN power entries (policy=%s)", n_nan_rows, len(frame), nan_policy)

    timestamps = _parse_timestamps(frame[columns["time"]]) if columns["time"] else None
    segment_ids = _segment_ids_from_frame(frame, columns, timestamps, time_gap_s)
    index_col = "index" if "index" in frame.columns else frame.columns[0]
    sample_index = pd.to_numeric(frame[index_col], errors="coerce").fillna(-1).to_numpy(int)

    ds = DeepSenseScenario(
        scenario_dir=scenario_dir,
        csv_path=csv_path,
        columns=columns,
        power=power,
        segment_ids=segment_ids,
        sample_index=sample_index,
        timestamps_s=timestamps,
        frame=frame if keep_frame else None,
        n_nan_rows=n_nan_rows,
        nan_policy=nan_policy,
    )

    # ---- cross-validate precomputed labels ------------------------------------------
    am = raw_argmax
    if columns["beam_label"] is not None:
        raw = pd.to_numeric(frame[columns["beam_label"]], errors="coerce").to_numpy()
        agree = {off: float(np.mean(raw - off == am)) for off in (0, 1)}
        offset = max(agree, key=agree.get)
        ds.beam_label = (raw - offset).astype(np.int64)
        ds.beam_label_offset = offset
        ds.label_agreement = agree[offset]
        ds.label_agreement_clean = float(np.mean((raw - offset == am)[clean_rows])) if clean_rows.any() else None
        if ds.label_agreement_clean is not None and ds.label_agreement_clean < 0.99:
            log.warning(
                "unit beam label agrees with argmax(power) on only %.1f%% of clean rows",
                100 * ds.label_agreement_clean,
            )
    if columns["max_power"] is not None:
        mp = pd.to_numeric(frame[columns["max_power"]], errors="coerce").to_numpy()
        ds.max_power_label = mp
        ds.max_power_agreement = float(np.mean(np.isclose(mp, np.nanmax(power, axis=1), rtol=1e-4)))

    for key in ("image", "lidar", "radar"):
        col = columns.get(key)
        if col is not None:
            ds.modality_paths[key] = [scenario_dir / str(p).lstrip("./") for p in frame[col]]
    for col in columns.get("gps") or []:
        if not pd.api.types.is_numeric_dtype(frame[col]):  # file references, not numeric fields
            ds.modality_paths[f"gps:{col}"] = [scenario_dir / str(p).lstrip("./") for p in frame[col]]
    return ds
