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

**Finding (Phase 4 smoke run): exogeneity must be enforced architecturally, not just
assumed.** The Phase-3 RSSM was, for DreamerV3 fidelity, conditioned on the logged behaviour
action - the one-hot of the *previous optimal beam* (`WorldModel.actions_from_obs`). That
action is a deterministic function of the previous observation and a near-perfect predictor
of the next power vector, so the GRU learned to read it as an observation. The model looked
excellent (on synthetic data: 0.08 dB one-step prediction regret, current-beam linear-probe
accuracy 0.88 from the latent) but the accuracy came from the leaked action, not from the
dynamics: with the action input removed the same small model's probe accuracy fell to 0.34.
In imagination the leak is fatal - the actor's own action enters the GRU, so whichever beam
the actor picks the model predicts that beam is good, and the learned policy collapsed to one
state-independent beam with ~21 dB *real* regret on the same model. The RSSM is therefore
trained with `RSSMConfig.action_dim = 0` (a pure exogenous sequence model; the imagined
latent trajectory cannot depend on the policy), which is exactly the assumption above turned
into a constraint; the action-conditioned model survives only as an ablation flag
(`train_actor_critic.py --action-conditioned`). This is the concrete evidence that R4 is not
a formality: an offline world model that is *allowed* to see the logged action will use it,
and the policy trained inside it will exploit that.

**Finding (same run): the actor's gradient must use the exact expectation over the predicted
reward table.** The original instruction was a straight-through categorical action with the
imagined reward read from the table at the sampled index. That estimator's gradient at the
sampled slot equals the raw reward value; because rewards are received power in dB (all
negative, -1 to -9 dB), every sampled beam is pushed down regardless of its relative merit and
the policy collapses to a single beam within a few dozen updates. Since the world model
predicts the *entire* 64-way table, the expected reward under the policy,
`sum_k pi(k|s) R̂[k]`, is available in closed form; its gradient is the exact softmax policy
gradient, offset-invariant and zero-variance. This is a scoped deviation, not an abandonment:
straight-through samples still drive the dynamics input of the RSSM and define the previous
beam in the switching-cost indicator, and evaluation still acts greedily (`argmax pi`).

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

## 6. Phase 4: actor-critic decision layer (design + smoke run)

Implementation: `channeldreamer.rl.actor_critic` (trainer, actor, critic, lambda-returns) and
`scripts/train_actor_critic.py` (world model -> actor-critic -> regime-decomposed evaluation
against the Phase-1 baselines on the same split). Tests: `tests/test_actor_critic.py`.

### 6.1 Design decisions

**Action representation: categorical over the 64-beam codebook, straight-through samples,
exact expected reward.** At each imagined state the actor samples a beam as a straight-through
one-hot (forward: sampled one-hot, backward: softmax gradient); the sample is what the dynamics
and the "previous beam" of the switching indicator see. The *reward* term, however, is taken
in exact expectation under the policy, `sum_k pi(k|s_j) R̂(s_{j+1})[k]`, which is possible
only because the world model predicts the entire 64-way reward table. This is the differentiable
action path whose gradient flow was verified at the end of Phase 3 (reward table -> action ->
actor), with no REINFORCE score function. Why not index the table with the straight-through
sample directly: that estimator's gradient at the sampled slot is the raw reward value; with
dB rewards (all negative) it pushes every sampled beam down and the policy collapses to a
single state-independent beam within a few dozen updates (observed on synthetic data). The
expectation is offset-invariant and has zero variance.

**The world model must be exogenous (`RSSMConfig.action_dim = 0`).** Phase 3 trained the RSSM
with the logged behaviour action, the one-hot of the *previous optimal beam*. That action is a
near-perfect predictor of the next power vector, so the GRU learns to read it as an
observation. In imagination the actor then games the model: whichever beam it picks, the model
predicts that beam is good. On synthetic data this gave a policy with ~21 dB real regret while
the same model's one-step prediction had 0.08 dB regret; the leak also inflated the apparent
quality of the action-conditioned model (removing the action, the small synthetic model's
linear-probe accuracy for the current beam fell from 0.88 to 0.34). Under the exogeneity
argument of Section 1 the imagined trajectory must not depend on the action at all, which is
now the default; the action-conditioned model is kept only as an ablation flag.

**Switching cost: external, mirrored from the environment, swept not learned.** The imagined
reward is `E_pi[R̂] - c * (1 - pi(a_{j-1} | s_j))`, i.e. the expectation of the environment's
`R[a] - c * 1[a != prev]` given the previous beam (the real currently served beam at the first
imagined step, the sampled imagined beam afterwards). `c` is a hyper-parameter of the trainer,
never a quantity the world model learns; the smoke run uses `c in {0, 0.5}` dB to match the
Phase-1 oracle-vs-reactive experiment (Section 0(c)). A controlled synthetic test (toy dynamics
in which the model's best beam provably advances one step per imagined step and the held beam
is readable from the state) confirms that `c = 10` cuts the switching rate of the learned
policy about four-fold relative to `c = 0`.

**Returns and critic.** Lambda-returns (`gamma = 0.99, lambda = 0.95`) over a 15-step imagined
rollout, bootstrapped with a slow-EMA critic; critic trained by MSE (symlog space) to the
detached return; actor maximises the return scaled by an EMA of its 5-95 percentile range
plus an entropy bonus. Note that with exogenous dynamics the critic's value carries no action
dependence, so the actor's gradient comes from the per-step expected rewards and from the
switching term coupling consecutive imagined steps; the critic is kept for fidelity and for
the action-conditioned ablation.

### 6.2 Smoke run on real Scenario 33 (not the experiment)

Setup: Phase-1 windowing and split (H = 8, k = 1, 14 train / 4 test segments, 937 test
windows of which 90 transition), cached camera + LiDAR features and GPS trajectories, world
model with `deter_dim = 256`, 16x16 latents, 1500 steps of B = 16 x T = 16 (35 s, peak
165 MB); one actor-critic per penalty, 300 updates each from 256 rollout starts, horizon 15
(8 s each, peak 180 MB including the world model). Power loss in dB on the held-out segments:

| method | stable | transition | net reward @ c = 0.5 (overall) | switch rate |
|---|---|---|---|---|
| reactive | 0.18 | 0.43 | -3.70 | 0 % |
| Markov | 0.20 | 0.45 | -3.89 | 35 % |
| predict-then-act | 0.12 | 0.46 | -3.82 | 36 % |
| world model, greedy one-step | 1.35 | 0.92 | -5.22 | 83 % |
| actor-critic, c = 0 | 1.18 | 0.82 | -5.00 | 72 % |
| actor-critic, c = 0.5 | 1.38 | 0.82 | -5.18 | 73 % |
| oracle | 0.00 | 0.00 | -3.79 | 59 % |

**Honest read: no early signal for the central hypothesis, and the reason is upstream of the
decision layer.** The actor-critic does what it is asked: its regret against the world model's
*own* reward table is 0.15-0.3 dB after 300 updates, and the switching penalty lowers the
imagined switch rate (0.23 -> 0.17). But the world model's table is wrong by ~1.3 dB per beam,
so every model-based policy loses ~1 dB to the reactive baseline on stable windows and
~0.4 dB at transitions. Two diagnostics pin the bottleneck on the latent, not the imagination:
the *posterior* state cannot reproduce the reward table it has just observed (regret 1.26 dB,
per-beam RMSE 1.35 dB at step t), and the total KL stayed at 0.4-0.6 nats, below the 1-nat
free-bits floor, i.e. the latent carries almost no information about the observation. With a
nearly state-independent table the model-optimal policy is a constant beam, and that is what
the actors learned: 2 distinct beams (c = 0) and 1 beam (c = 0.5) over the 937 test windows
versus 54 distinct truly-optimal beams. The positive "transition minus stable" advantage of the
model-based rows (+0.6 to +0.8 dB) is therefore an artefact of a constant policy being bad
everywhere while the reactive baseline is worst at transitions; it is not evidence for the
hypothesis. Predict-then-act, by contrast, already beats reactive on stable windows (0.12 vs
0.18 dB) and matches it at transitions.

Consequence for the plan: the Phase-3 item "world-model training validated against the
baselines" was never closed, and Phase 4 cannot show anything until the world model's greedy
one-step regret is at least at the reactive level (0.20 dB).

### 6.3 Diagnostic: is it under-training or the recipe?

One longer run with the same recipe (`--wm-steps 6000`, 127 s, everything else unchanged; a
diagnostic, not a sweep) answers this: the world model is simply under-trained at 1500 steps.

| world-model steps | posterior table regret at t | greedy one-step regret (stable / transition) | distinct beams (wm-greedy) |
|---|---|---|---|
| 1500 | 1.26 dB | 1.35 / 0.92 dB | 10 |
| 6000 | 0.35 dB | 0.39 / 0.49 dB | 34 |
| reactive | - | 0.18 / 0.43 dB | - |

At 6000 steps the reconstruction loss was still falling (0.054 -> 0.023) and the KL had risen
to 0.8 nats, i.e. the latent had started to carry the observation; the greedy one-step
prediction is within 0.2 dB of reactive on stable windows and 0.06 dB behind it at
transitions. The recipe therefore works and needs more steps (or a higher learning rate), not
a redesign; the symlog / free-bits concerns above remain worth checking but are not the
blocker. The actor-critic in that run was still far from converged after 300 updates at
`actor_lr = 3e-5` (imagined regret against the model's own table ~1 dB, entropy 1.0-1.6 nats,
3 distinct beams chosen), so it lost to its own world model's greedy policy (0.92 vs 0.39 dB
stable). With a nearly constant table (the 1500-step model) the actor's job was trivial; with a
state-dependent table it needs many more updates than the smoke budget.

**Where this leaves the hypothesis:** untested, not refuted. Neither smoke run produced a
policy that beats predict-then-act anywhere, and the only positive "transition minus stable"
deltas came from degenerate policies. The decisions to take before the real experiment, in
order: (1) train the world model until greedy one-step regret <= reactive on both regimes and
report that table as a Phase-3 result; (2) give the actor-critic enough updates (and likely a
larger `actor_lr`) to reach < 0.2 dB imagined regret against the model with a state-dependent
table; (3) only then sweep the switching cost and the imagination horizon (including the
horizon-1 ablation). GPU is not a constraint: 165-180 MB peak for the whole stack.

## 7. Phase 3 result: world-model one-step prediction vs. the baselines

Closed with `python -m channeldreamer.scripts.train_world_model --max-steps 60000 --eval-every 1000`
(exogenous RSSM, `deter_dim = 256`, 16x16 latents, cached camera + LiDAR + GPS features, B = 16
x T = 16, `lr = 1e-4`). The stopping rule never looks at the test segments: two of the 14
training segments (420 windows, 21 transition) are held out as validation, and training stops
when the greedy one-step regret on validation is <= reactive's on *both* regimes for two
consecutive evaluations, or on a 10-evaluation plateau, or at the step cap. The run met the
target at 18k and 19k steps (340 s, peak 290 MB) and the 19k checkpoint is kept.

Trend: the loss fell fast to ~0.70 by 3k steps and then barely moved (0.69-0.70), while the
prediction regret kept improving for another 15k steps as the reconstruction term went
0.040 -> 0.018 and the KL rose from 0.71 to 0.91 nats (the latent progressively took up the
observation). Validation regret, stable / transition, in dB: 1.46 / 1.59 (1k), 0.73 / 1.11 (3k),
0.25 / 0.85 (5k), 0.14 / 0.59 (7k), 0.13 / 0.48 (9k), 0.10 / 0.46 (15k), 0.09 / 0.36 (19k);
reactive on the same windows 0.105 / 0.442.

Held-out test segments (937 windows, 90 transition), the same table as the Phase-1 baselines:

| method | top-1 | top-3 | loss dB stable | loss dB transition | loss dB overall |
|---|---|---|---|---|---|
| reactive | 0.409 | 0.733 | 0.18 | 0.43 | 0.20 |
| Markov | 0.422 | 0.711 | 0.20 | 0.45 | 0.22 |
| predict-then-act | 0.417 | 0.769 | 0.12 | 0.46 | 0.15 |
| world model, greedy one-step | 0.410 | 0.772 | 0.15 | 0.42 | 0.18 |
| oracle | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |

The world model beats reactive on stable windows (0.15 vs 0.18 dB) and matches it at
transitions (0.42 vs 0.43 dB; with 90 transition windows this difference is within noise), so
the Phase-3 target "at least reactive on both regimes" is met on the held-out segments as
well as on validation. It is not yet as good as predict-then-act on stable windows
(0.15 vs 0.12 dB) and both are no better than reactive at transitions, which is the honest
state of one-step *prediction* on this scenario: the transition regime is where nothing
predicts yet. Diagnostics that were broken in the smoke runs are now healthy: the posterior
state reproduces the observed table with 0.15 dB regret (per-beam RMSE 0.74 dB, down from
1.35), and the greedy policy uses 49 distinct beams over the test set (truly optimal: 54).
This checkpoint (`runs/world_model/checkpoint.pt`, gitignored) is the frozen world model for
Phase 4.

## 8. Stage 2 result: actor-critic (c = 0) on the frozen world model

Full record in `docs/experiments.md` (Stage 2) and `results/stage2_c0_run2_*.json`. Setup and
stopping discipline as in Section 7: the actor-critic of Section 6 is trained on the frozen
19k-step world model with the imagined regret against the model's own reward table monitored on
validation rollouts, and the held-out test segments are evaluated once, with the same four
methods and split as the Stage-1 table.

Two runs were needed. With the DreamerV3-style defaults (`actor_lr = 3e-4`, `entropy_scale =
3e-4`) the policy plateaued at 0.15 dB imagined regret and used only 11-13 of the 64 beams: the
exact softmax policy gradient of Section 6.1 is proportional to the probability of a beam, so
once a beam is no longer selected it cannot be rediscovered, and a weak entropy bonus lets the
policy quantise to a coarse set of "hub" beams (adjacent to the model's best beam, hence a low
*imagined* regret, but a real regret worse than the model's own argmax). A ten-fold larger
entropy bonus and learning rate (`3e-3`, `1e-3`) reached 0.056 dB in 750 updates with 21
distinct beams and passed both pre-test checks (collapse: 21 beams on validation; state
dependence: 0.1 % of rollout starts above 0.5 dB mean regret).

Held-out power loss (dB): reactive 0.18 / 0.43, predict-then-act 0.12 / 0.46, world-model
greedy one-step 0.15 / 0.42, actor-critic (c = 0) 0.15 / 0.53 (stable / transition).

**Reading.** At zero switching cost the decision layer is, by construction, trying to reproduce
the world model's greedy one-step beam, and it does so on stable windows (tie at 0.15 dB) while
losing 0.11 dB at transitions and using far fewer distinct beams (22 vs 49 on test). So the
first genuine test of the central hypothesis is negative at this stage: the actor-critic shows
no advantage over prediction alone, and its transition-regime loss is the *worst* of the four
methods. This is not yet the switching-cost setting in which a proactive policy could pay off
(Section 0(c)); it is the ablation that shows the policy machinery is at best neutral. Two
caveats limit the conclusion: (1) the stopping rule fired on the first crossings of the 0.10 dB
target while regret, beam count and entropy were still improving steeply, so the policy is
target-converged rather than plateau-converged; (2) 90 transition windows give roughly
+/-0.1 dB resolution. Stage 3 (switching-cost and horizon sweep) should therefore use a
plateau-based stop and keep reporting the world-model greedy policy alongside, since the
hypothesis now has a concrete bar to clear: beat 0.42 dB at transitions.
