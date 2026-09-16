# ChannelDreamer-6G — Methodology

## 0. Findings From Real Data (Phase 1)

Three facts surfaced while building and validating the Phase-1 pipeline on real DeepSense 6G
Scenario 33 (3981 samples, 18 drive segments, ~92 ms sampling). They are recorded here because
each one changes how later phases must be run.

**(a) Beam labels are 1-based, and the offset must be detected, not assumed.** The released
`unit1_beam` column runs from 1 to 64 while `argmax` over the 64-element power vector runs
from 0 to 63. The loader therefore compares the released label against `argmax(power)` under
both a 0 and a 1 offset and keeps whichever agrees on more rows (Scenario 33 resolves to
offset 1 with 100 % agreement on clean rows). Hard-coding either convention would silently
shift every label by one beam in scenarios released with the other convention; a one-beam
shift is invisible in top-3/top-5 accuracy but destroys top-1 and the dB power-loss metric.

**(b) A small number of rows have NaN beams, and they poison the regime labeller.** 38 of
3981 rows (about 1 %) contain between 1 and 9 NaN entries in the 64-vector (62 NaN values in
total), a sweep-dropout artefact of the measurement pipeline. The released `unit1_beam` label
on those rows points at the NaN beam (it was evidently produced with a plain `argmax`, which
treats NaN as the maximum), whereas `unit1_max_pwr` was computed NaN-aware. Left untreated,
the NaNs propagate through the moving-average smoothing and the dB conversion used by the
regime labeller, which then reported 451 transition steps instead of the 266 obtained after
handling. The loader now uses a NaN-safe `argmax` (`nanargmax`) everywhere and applies a
configurable `nan_policy`: `interpolate` (default; linear interpolation across adjacent beams
within the vector, since the beam pattern is smooth across the codebook), `keep` (leave NaNs
in place), or `raise` (fail loudly). With the default policy the remaining 0.95 % disagreement
between our optimal beam and the released label is exactly the set of NaN rows, where our
label is the correct one.

**(c) A 0.5 dB switching penalty already flips the ranking between oracle and reactive.** On
the held-out Scenario 33 segments, the oracle (perfect foresight, always choose the best
beam) pays the switching cost on 554 of 937 steps because the optimal beam index flickers
between adjacent beams at 92 ms sampling. Its net reward (−3.79 dB mean) ends up *below* the
reactive baseline that never switches (−3.70 dB), even though the oracle has zero regret in
received power. Typical per-step regret of the reactive policy in stable segments is only
~0.18 dB, i.e. well below the 0.5 dB penalty. This is an early calibration signal for the
Phase-4 reward design: the switching-cost weight must be swept, and a penalty larger than the
typical adjacent-beam power gap turns "never switch" into the trivially optimal policy in
stable segments. It also argues for evaluating switching cost against the *transition* regime
separately, where regret is 0.43 dB and the trade-off is genuinely non-trivial.

## 0.1 GPU memory budget for training (Phase 3 profiling)

Measured on the lab RTX 4070 (12 GB) with `python -m channeldreamer.scripts.profile_components`
(each component in isolation, peak stats reset between stages, then everything combined in one
forward + backward pass on a real Scenario 33 batch of B = 8, T = 8). The full Phase-3 stack —
camera projection, LiDAR encoder, GPS/trajectory encoder, power encoder and the RSSM — uses
well under 1 GB combined when the camera and LiDAR features come from the offline cache
(`data/scenario33/cache`, built by `prepare_data --pretokenize-lidar --precompute-camera`):
about 0.1 GB for the combined forward + backward pass, 0.12 GB with AdamW state. See the
script for the per-component breakdown.

**Rule for Phase 4: training must always use the cached feature path, never raw images through
the ResNet backbone at train time.** Feeding raw images costs roughly 4x more (0.36 GB vs
0.10 GB at B = 8, T = 8, and ~0.5 GB reserved), and since activation memory scales with B x T
the raw path would not fit a realistic Phase-4 batch such as B = 64, T = 32 (32x the profiled
size, i.e. on the order of 10 GB before the actor-critic) in 12 GB of VRAM, whereas the cached
path at that size stays around 3 GB and leaves ample headroom for the actor and critic. The
backbone is frozen anyway, so caching loses nothing.

## 1. The offline RL formulation (risk R4, the big one)

DeepSense is a recorded dataset, not an interactive simulator. Two properties
make it tractable. Fully-observed reward: the 64-beam power vector gives the
received power of every beam at each recorded instant, not only the beam that
was used. Exogenous transitions: beam selection does not move the user, so
next-observation depends only on user motion, not on the agent's action — a
world model learned by plain sequence prediction is therefore a valid dynamics
model, and the policy only optimizes the reward stream.

Open question for the advisor: is the exogeneity assumption acceptable, or does
handover (which could change which BS serves the user) break it?

## 2. Regime labelling (transition boundaries)

Current approach: flag a transition if |delta optimal-beam-index| exceeds a
threshold, or peak power drops more than a dB threshold, both on a smoothed
series. This is a transparent proxy, not ground truth. Required before
publication: report thresholds, run sensitivity analysis, cross-check against
DeepSense blockage-task labels where available.

## 3. Reward design

Reward = received power (dB) of chosen beam, minus a switching-cost penalty
when the beam changes. The switching-cost weight must be swept and reported
with sensitivity, not a single tuned value (see Section 0(c) above for an
early real-data signal on this).

## 4. Baselines (must all appear in the paper)

1. Reactive (hold last beam).
2. Markov transition model.
3. Predict-then-act: transformer forecaster + greedy controller — isolates the
   value of the decision mechanism from the value of better prediction.
4. Ablation: world-model policy with imagination horizon = 1.

## 5. Metrics

Regime-decomposed, not aggregate. Top-1/3/5 accuracy and power loss (dB),
split stable vs. transition. Power loss is the more honest operational metric
because an adjacent wrong beam is cheap and a far wrong beam is catastrophic;
top-k accuracy cannot see that difference (confirmed on real scenario 33,
where top-1 is noisy from beam flicker at 92ms sampling but dB loss is
stable).
