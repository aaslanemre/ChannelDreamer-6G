"""Synthetic DeepSense-like beam-power generator.

Simulates a moving user whose optimal beam (a continuous 'angle' in beam-index units) drifts
slowly under a random walk, with occasional *sharp transitions*: an abrupt beam jump
accompanied by a dip in peak power (think: blockage, turning a corner, a new dominant
reflector).  Each step yields a 64-element received-power vector shaped like a beam pattern
centred on the true angle plus noise, so ``argmax`` recovers the optimal beam, exactly as
in DeepSense 6G.  Data are organised in independent *segments* (drive / pass-by events).

This lets the complete pipeline - windowing, regime labelling, metrics, offline MDP and
baselines - be exercised with zero real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import NUM_BEAMS


@dataclass
class SyntheticBeamData:
    power: np.ndarray  # (N, B) linear power
    segment_ids: np.ndarray  # (N,)
    true_angle: np.ndarray  # (N,) continuous optimal-beam position
    event_mask: np.ndarray  # (N,) bool, True at injected sharp transitions (ground truth)

    @property
    def n_samples(self) -> int:
        return int(self.power.shape[0])


def _beam_pattern(angle: float, n_beams: int, beamwidth: float) -> np.ndarray:
    idx = np.arange(n_beams)
    main = np.exp(-0.5 * ((idx - angle) / beamwidth) ** 2)
    # weak sidelobe floor so that non-adjacent beams still receive some power
    side = 0.05 * np.exp(-0.5 * ((idx - angle) / (4 * beamwidth)) ** 2)
    return main + side


def generate_synthetic(
    n_segments: int = 12,
    segment_length: int = 400,
    *,
    n_beams: int = NUM_BEAMS,
    drift_std: float = 0.15,
    transition_prob: float = 0.01,
    jump_min: int = 6,
    jump_max: int = 20,
    dip_db: float = 8.0,
    dip_recovery_steps: int = 5,
    beamwidth: float = 2.0,
    noise_std: float = 0.02,
    peak_power: float = 1.0,
    seed: int = 0,
) -> SyntheticBeamData:
    """Generate ``n_segments`` independent segments of ``segment_length`` steps each.

    Parameters
    ----------
    drift_std
        Std of the per-step random walk of the true beam angle (in beam units).  Small
        values give long STABLE stretches.
    transition_prob
        Per-step probability of a sharp transition event.
    jump_min, jump_max
        Magnitude (in beam indices) of a transition jump; sign is random.
    dip_db
        Peak-power drop (dB) at a transition, recovering linearly over
        ``dip_recovery_steps`` steps.
    noise_std
        Std of additive Gaussian noise on the linear power vector (clipped at a small
        positive floor so dB conversion is safe).
    """
    rng = np.random.default_rng(seed)
    n = n_segments * segment_length
    power = np.empty((n, n_beams), dtype=np.float64)
    seg_ids = np.repeat(np.arange(n_segments), segment_length)
    angles = np.empty(n)
    events = np.zeros(n, dtype=bool)

    row = 0
    for _ in range(n_segments):
        angle = rng.uniform(4, n_beams - 5)
        velocity = rng.normal(0.0, drift_std)  # slow directional drift per segment
        dip_left = 0
        dip_level = 0.0
        for t in range(segment_length):
            if t > 0 and rng.random() < transition_prob:
                jump = rng.integers(jump_min, jump_max + 1) * rng.choice([-1, 1])
                angle = float(np.clip(angle + jump, 0, n_beams - 1))
                dip_left = dip_recovery_steps
                dip_level = dip_db
                events[row] = True
                velocity = rng.normal(0.0, drift_std)
            else:
                angle = float(np.clip(angle + velocity + rng.normal(0.0, drift_std / 3), 0, n_beams - 1))
                if angle <= 0 or angle >= n_beams - 1:  # bounce at the edge of the codebook
                    velocity = -velocity
            gain_db = -dip_level * (dip_left / dip_recovery_steps) if dip_left > 0 else 0.0
            dip_left = max(0, dip_left - 1)
            vec = peak_power * 10 ** (gain_db / 10) * _beam_pattern(angle, n_beams, beamwidth)
            vec = vec + rng.normal(0.0, noise_std, size=n_beams)
            power[row] = np.clip(vec, 1e-4, None)
            angles[row] = angle
            row += 1

    return SyntheticBeamData(power=power, segment_ids=seg_ids, true_angle=angles, event_mask=events)


# ======================================================================================
# Synthetic complex CIR (WiWorld-RealData stand-in, Phase 2)
# ======================================================================================


@dataclass
class SyntheticCIRData:
    """Complex dual-band CIR sequence from a simulated multipath channel along one route."""

    cir: np.ndarray  # (N, n_bands, n_taps) complex64
    segment_ids: np.ndarray  # (N,)
    good: np.ndarray  # (N,) bool quality flag
    positions: np.ndarray  # (N, 2) metres along a planar route
    bands_hz: tuple[float, ...]
    tap_spacing_s: float

    @property
    def n_samples(self) -> int:
        return int(self.cir.shape[0])

    def write_wiworld_layout(self, out_dir: str | Path, *, per_band_columns: bool = False,
                             manifest_name: str = "manifest.csv") -> Path:
        """Write ``manifest.csv`` + ``cir/sample_<i>.npy`` files in the layout assumed by
        :mod:`channeldreamer.data.wiworld` (a *stand-in* for the real, not yet inspected, layout)."""
        out_dir = Path(out_dir)
        (out_dir / "cir").mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(self.n_samples):
            row = {"sample_id": i, "route_id": int(self.segment_ids[i]), "timestamp_s": 0.1 * i,
                   "pos_x_m": float(self.positions[i, 0]), "pos_y_m": float(self.positions[i, 1]),
                   "quality_flag": "good" if self.good[i] else "bad"}
            if per_band_columns:
                for b, f in enumerate(self.bands_hz):
                    rel = f"cir/sample_{i}_band{b}.npy"
                    np.save(out_dir / rel, self.cir[i, b])
                    row[f"cir_{f / 1e9:.3f}ghz"] = "./" + rel
            else:
                rel = f"cir/sample_{i}.npy"
                np.save(out_dir / rel, self.cir[i])
                row["cir_path"] = "./" + rel
            rows.append(row)

        pd.DataFrame(rows).to_csv(out_dir / manifest_name, index=False)
        return out_dir / manifest_name


def generate_synthetic_cir(
    n_segments: int = 2,
    segment_length: int = 200,
    *,
    n_taps: int = 64,
    bands_hz: tuple[float, ...] = (3.7e9, 6.775e9),
    tap_spacing_s: float = 10e-9,
    n_paths: int = 4,
    speed_mps: float = 1.5,
    sample_period_s: float = 0.1,
    noise_std: float = 0.02,
    bad_fraction: float = 0.05,
    seed: int = 0,
) -> SyntheticCIRData:
    """Simulate complex CIRs of a walker on a route with ``n_paths`` specular multipath components.

    Each path has a slowly drifting delay and a band-dependent phase ``exp(-j 2 pi f tau)``; the
    same geometry is observed at every band so cross-band structure exists for the physics
    losses of Phase 2.  Taps are placed with a sinc pulse on a uniform delay grid.  A random
    ``bad_fraction`` of samples are flagged bad and replaced by noise (quality-flag handling).
    """
    rng = np.random.default_rng(seed)
    n = n_segments * segment_length
    n_bands = len(bands_hz)
    cir = np.zeros((n, n_bands, n_taps), dtype=np.complex64)
    seg = np.repeat(np.arange(n_segments), segment_length)
    pos = np.zeros((n, 2))
    good = rng.random(n) > bad_fraction
    grid = np.arange(n_taps) * tap_spacing_s
    row = 0
    for _ in range(n_segments):
        heading = rng.uniform(0, 2 * np.pi)
        p = rng.uniform(-20, 20, size=2)
        delays = np.sort(rng.uniform(0, 0.5 * n_taps * tap_spacing_s, size=n_paths))
        gains = 10 ** (-rng.uniform(0, 15, size=n_paths) / 20)
        drift = rng.normal(0, 0.02 * tap_spacing_s, size=n_paths)
        for t in range(segment_length):
            heading += rng.normal(0, 0.05)
            p = p + speed_mps * sample_period_s * np.array([np.cos(heading), np.sin(heading)])
            delays = np.clip(delays + drift + rng.normal(0, 0.005 * tap_spacing_s, size=n_paths),
                             0, (n_taps - 2) * tap_spacing_s)
            for b, f in enumerate(bands_hz):
                h = np.zeros(n_taps, dtype=np.complex128)
                for k in range(n_paths):
                    phase = np.exp(-2j * np.pi * f * delays[k])
                    h += gains[k] * phase * np.sinc((grid - delays[k]) / tap_spacing_s)
                h += noise_std * (rng.normal(size=n_taps) + 1j * rng.normal(size=n_taps)) / np.sqrt(2)
                if not good[row]:
                    h = 3 * noise_std * (rng.normal(size=n_taps) + 1j * rng.normal(size=n_taps))
                cir[row, b] = h
            pos[row] = p
            row += 1
    return SyntheticCIRData(cir=cir, segment_ids=seg, good=good, positions=pos, bands_hz=bands_hz,
                            tap_spacing_s=tap_spacing_s)
