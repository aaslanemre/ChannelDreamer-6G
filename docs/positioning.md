# ChannelDreamer-6G
## Related Work, Research Gap, and Positioning

**Emre Can Aslan**, Doctoral Candidate, Programa de Engenharia Elétrica (PEE/COPPE), UFRJ
**Advisors:** Prof. Thomas, Dr. Rodrigo Couto

## Executive Summary

Wireless world models have improved quickly over the past year. They now forecast
the wireless channel several steps ahead with good accuracy, on both simulated
and measured data. Separately, another group of papers has started putting world
models inside reinforcement-learning agents to optimize networks. These two
lines have grown apart, and that gap matters. The models trained on realistic
channel data stop at prediction. The models that actually take actions are
tested on simulated environments, or they work above the physical layer.

ChannelDreamer-6G proposes to close that loop. It uses DreamerV3's latent
imagination not as a forecast, but as the basis for an actor-critic policy that
makes proactive beam-management decisions. The policy is trained and tested on
real measurements.

The main scientific claim is narrower than "a world model for wireless," and
that narrowness is what makes it defensible. The claim is that imagined
multi-step rollouts give a measurable advantage over reactive methods at
LoS/NLoS transition boundaries, and that this advantage can be measured,
reproduced on a public benchmark, and traced to the decision mechanism rather
than to better prediction alone.

## 1. Scope and Method

This is a targeted review, not a full survey. Seven works were reviewed closely:
WWM, WiFo, Wireless Dreamer, Homomorphic World Model, MobiWorld, Ghassemi et al.
(closest beam-management competitor), and DMWM (a methodological reference, not
a wireless paper). Six of these seven were read in full primary form. WWM was
not obtained in primary form; its figures come from consistent secondary
summaries and are flagged accordingly below.

## 2. Problem Framing

Beam management in fast-moving mmWave systems is usually treated as a
prediction problem: estimate the best beam index or channel state at time
t+k, then act on that estimate. But the real operational problem is a decision
problem: the system needs the action that gives the best cumulative link
quality over time, accounting for switching cost and the fact that a choice
made now limits the choices available next. A model that forecasts CSI with
lower error does not automatically give a policy with higher throughput. The
gap is widest near blockage events, where forecast uncertainty is highest and
reacting late is most expensive. This difference, prediction accuracy versus
decision quality, is the line this thesis is positioned on.

## 3. Related Work

| Work | Data regime | Output | Decision mechanism | Validation |
|---|---|---|---|---|
| WWM (arXiv:2603.25216) [not read in primary form] | Hybrid: ~700K Sionna RT ray-traced + real 6G prototype uplink CSI (~800K total) | Prediction (CSI, compression, beam prediction, localization) | None reported — no RL, actor-critic, or reward | Real prototype fine-tuning + zero-shot cross-city |
| WiFo (arXiv:2412.08908) | Simulated (QuaDRiGa, 160K samples) | Prediction (space-time-frequency CSI) | None (MAE + ViT foundation model) | Simulated only, zero-shot tested |
| Wireless Dreamer (in arXiv:2506.00417, IEEE Comm. Mag. survey) | Simulated (UAV trajectory, Gaussian weather model) | Decision (UAV trajectory, resource reconfiguration) | World model + Q-learning (not actor-critic) | Simulated only |
| Homomorphic World Model (arXiv:2603.20048, ICC 2026) | Real (DICHASUS, 32 Rx antennas, indoor, static environment) | Prediction (channel charting, latent CSI dynamics) | None (JEPA + Lie-algebra transitions; MDP defined but no reward) | Real, static environment |
| MobiWorld (arXiv:2507.09462) | Real (heterogeneous field data) | Decision (BS sleep control, user offloading) | Diffusion-based generative simulator feeding PPO/MAPPO | Real, network-management layer |
| Ghassemi et al. (arXiv:2410.19859, closest competitor) | Real DeepSense features, simulated LOS path-loss reward | Decision (beam selection) | Model-free tabular Q-learning (not a world model) | Simulated RL environment, DeepSense Scenario 32, aggregate metrics only |
| ChannelDreamer-6G (proposed) | Real (DeepSense 6G) | Decision (proactive beam switching) | DreamerV3 RSSM + actor-critic over imagined rollouts | Real measurements, transition-focused evaluation |

Two observations shape the positioning. First, multi-step prediction is not a
differentiator — several reviewed models already do it. Second, feeding
actions into a model as conditioning input is not the same as choosing them —
WWM and the Homomorphic World Model both do the former, neither does the
latter.

## 4. The Gap

No reviewed work combines all three of: (1) learning from real hardware
measurements, (2) producing chosen actions through an actor-critic mechanism
over imagined rollouts, (3) operating at the physical layer (beam
selection/handover) rather than network management. Wireless Dreamer has the
right architecture and layer but is simulation-only. MobiWorld has real data
and a real actor-critic but operates at network-management granularity.
Ghassemi et al. is the closest on task (beam selection, real DeepSense
features) but uses model-free Q-learning with a simulated path-loss reward,
not a world model, and evaluates on aggregate metrics only, not decomposed by
channel regime.

**Positioning statement:**

> Existing wireless world models forecast the channel accurately but do not
> choose actions. Existing world-model-based RL approaches have the right
> decision architecture but have not been tested on real measurements. And
> real-data RL approaches for wireless work at the network-management level,
> not the physical layer. ChannelDreamer-6G sits at the point these lines do
> not cover: it learns a policy over DreamerV3's imagined latent rollouts to
> make proactive beam-management decisions from real measurements, learning
> from measured received power rather than a simulated reward, and it tests
> that policy exactly where reactive methods break down — at LoS/NLoS
> transition boundaries.

No "first" claim is made. This is a bounded intersection, which is both more
honest and harder to argue against.

## 5. Contributions

**C1. Model-based (world-model) offline RL on real wireless measurements.**
Extends the Wireless Dreamer framework from simulated UAV scenarios to real
DeepSense measurements. Differs from the closest competitor (Ghassemi et al.)
on four axes: model-based world-model imagination vs. model-free reaction;
multi-step anticipation vs. one-shot; reward grounded in measured power vs.
simulated path-loss; evaluation decomposed by channel regime vs. aggregate
only.

**C2. Transition-focused evaluation of proactive beam management.**
Frames beam switching as an MDP over DeepSense's native 64-beam codebook,
evaluated against published baselines, broken down by channel regime: stable
LoS, stable NLoS, and transition boundaries.

**C3. Physics-grounded latent constraints for decision-oriented world models.**
Adapts IDM loss and VICReg variance/covariance terms from the Homomorphic
World Model (verified reference weights: teacher-forcing 1.0, rollout 2.0,
variance 2.0, covariance 10.0, IDM 1.0), together with an SGCS-style
structural-similarity term and a phase-consistency term, into DreamerV3's
RSSM, checking whether these constraints improve policy quality, not just
prediction accuracy. Requires WiWorld-RealData (complex CIR); DeepSense alone
is insufficient here because it provides beam-domain power, not complex CSI.

**C4. Cross-regime transfer of decision performance.** (depends on C1)
Whether policy performance transfers between real and synthetic regimes, and
across DeepSense's 40+ scenarios.

## 6. What Makes This Publishable

The intended insight, stated as a hypothesis before running experiments: *the
advantage of latent imagination over reactive baselines shows up at LoS/NLoS
transitions and is small in stable channel segments.* This is testable, and a
negative result is still publishable because it characterizes where world
models fail to help, not just where they succeed. Required baselines: reactive,
Markov, predict-then-act (transformer forecaster + greedy controller — this
isolates the value of the decision mechanism from the value of better
prediction), published DeepSense results on the matching scenario, and WWM's
94.0% top-1 beam accuracy as external context (different benchmark, not a
like-for-like target).

## 7. Likely Objections and Prepared Answers

**"WWM already gets 94% top-1 beam accuracy. What does this add?"**
WWM performs beam prediction: supervised classification from current
observation to currently-optimal beam, each sample independent. This thesis
performs beam management: sequential decision-making over imagined futures,
accounting for switching cost, where each choice constrains the next.
Different problem formulations even when both emit a beam index. WWM reports
no RL/actor-critic/reward anywhere. Caveat: WWM's number is on a different,
largely simulated benchmark, so it is context, not a target; it is not known
whether WWM uses a 64-beam codebook, so no shared-action-space claim is made.

**"Ghassemi et al. already did RL on DeepSense beam selection."**
Correct, and it is the closest work, cited prominently. It differs on the four
axes in C1: no world model, one-shot not multi-step, reward is simulated
path-loss not measured power, evaluation is aggregate not regime-decomposed.

**"Why DreamerV3 instead of a transformer foundation model?"**
For forecasting, a transformer is stronger and simpler — which is why one is
the required predict-then-act baseline. The claim is that the actor-critic
over imagined rollouts helps specifically on the decision task. If the
baseline matches the proposed method, that is a real negative result and will
be reported.

## 8. Risks

**R1.** The gap may close during the thesis. Re-run this comparison quarterly;
prioritize a conference submission early for a timestamp.

**R2.** Checked as of this review: no work combining real DeepSense data,
world-model imagination, and actor-critic beam decisions was found. Re-check
before submission.

**R3.** Compute and latent rollout error. DreamerV3 with image/point-cloud
encoders is training-intensive; 12 GB local VRAM requires modest batch sizes,
mixed precision, and offline LiDAR pre-tokenization. Long imagined rollouts
accumulate prediction error (Wireless Dreamer reports this independently) —
keep imagination horizon as a tuned hyperparameter, not assumed-longer-better.

**R4. Offline RL on a fixed dataset — the most consequential open question.**
DeepSense is recorded, not interactive; the agent cannot explore beyond what
was measured. Mitigating property: the 64-beam power vector makes the reward
for EVERY candidate beam fully observed at each step (not just the logged
one), and beam choice does not influence user trajectory, so transitions are
exogenous — a world model trained by plain sequence prediction is a valid
dynamics model. Open question for advisors: does exogeneity hold under
handover, where beam/BS choice could plausibly affect future observations?

**R5.** The central hypothesis might not hold. Stated in advance; a
well-analyzed negative result is still publishable, likely at a conference
rather than a top journal.

## References (author names/identifiers to verify before external circulation)

| # | Title | Identifier |
|---|---|---|
| 1 | A Wireless World Model for AI-Native 6G Networks [not read in primary form] | arXiv:2603.25216 |
| 2 | WiFo: Wireless Foundation Model for Channel Prediction | arXiv:2412.08908 |
| 3 | World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks | arXiv:2506.00417 |
| 4 | Structured Latent Dynamics in Wireless CSI via Homomorphic World Models | arXiv:2603.20048 |
| 5 | MobiWorld: World Models for Mobile Wireless Network | arXiv:2507.09462 |
| 6 | Multi-Modal Transformer and RL-Based Beam Management (Ghassemi et al.) | arXiv:2410.19859 |
| 7 | Mastering Diverse Domains through World Models (DreamerV3) | arXiv:2301.04104 |
| 8 | DMWM: Dual-Mind World Model with Long-Term Imagination (methodology reference, not wireless) | NeurIPS 2025 |
