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

import numpy as np

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
