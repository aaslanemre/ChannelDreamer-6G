# ChannelDreamer-6G

**Contribution in one line:** a DreamerV3-style world model trained purely by sequence prediction on real
DeepSense 6G beam-power measurements, used for *offline* model-based RL of proactive mmWave beam
management, evaluated where it matters: **at transitions**.

## How this differs from Ghassemi et al. 2024 (arXiv:2410.19859)

Ghassemi et al. study multimodal foundation-model-based beam prediction on DeepSense 6G. This thesis
differs on four axes:

| Axis | Ghassemi et al. 2024 | This work |
|---|---|---|
| **Problem framing** | Supervised beam *prediction* (classification of the next optimal beam) | Offline *decision making*: an MDP with switching costs, solved by model-based RL |
| **Model class** | Feed-forward multimodal encoder to beam logits | Latent recurrent world model (RSSM) + actor-critic trained on imagined rollouts |
| **Evaluation** | Aggregate top-k accuracy | Metrics **decomposed by regime** (stable vs. transition) + net MDP reward with switching penalty |
| **Use of the data** | Each sample is an i.i.d. labelled example | Exploits that DeepSense measures the reward of *every* beam at every step, giving a fully-observed offline MDP with exogenous dynamics |

## Central falsifiable hypothesis

> A world-model policy only *meaningfully* outperforms reactive baselines (hold the current best beam;
> first-order Markov) at **TRANSITION** steps (beam jumps, blockage-induced power dips). In **STABLE**
> segments the reactive baseline is already near-optimal, and any gain there is within noise.

Regimes are labelled from the smoothed beam-power series (`channeldreamer.eval.regimes`): a step is a
transition when the optimal beam index jumps by more than a threshold or the peak power drops by more
than a threshold in dB. Every metric is reported as *overall / stable / transition*. The hypothesis is
refuted if the world-model policy's advantage at transitions is not significantly larger than at stable
steps, or if it fails to beat the Markov baseline at transitions at all.

## Phases

1. **Data pipeline + baselines (this release).** DeepSense loader, shared windowing, synthetic
   generator, regime labelling, regime-decomposed metrics, fully-observed offline MDP, reactive and
   Markov baselines, tests.
2. **Supervised sequence models.** GRU / transformer beam predictors under the same windowing, to
   separate "better sequence model" from "better decision making".
3. **World model.** DreamerV3-style RSSM trained by sequence prediction on power vectors
   (`models/world_model.py`).
4. **Actor-critic on imagined rollouts** with switching penalty (`rl/actor_critic.py`), then
   multimodal side information (camera / LiDAR / GPS, `models/encoders.py`).

## Offline-RL methodological note

DeepSense 6G records the received power for all 64 codebook beams at every step, so the reward of *every*
action, not only the logged one, is observed: the reward table is measured, not estimated, and no
off-policy correction or counterfactual reward model is needed. Moreover, the base station's beam choice
does not affect the vehicle's trajectory, so state transitions are **exogenous**: a dynamics model
learned by plain next-step prediction of the power vector is a valid model of the MDP and imagined
rollouts suffer no action-induced distribution shift. The only action-dependent term is the switching
penalty, which is known analytically. See `channeldreamer/envs/offline_beam_env.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m channeldreamer.scripts.train_baseline --synthetic
python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33
python -m channeldreamer.scripts.train_baseline --data-root data --scenario 33
```

Data are expected under `data/scenarioN/` (DeepSense 6G layout: a `*_dev.csv` index referencing
`unit1/mmWave_data/*.txt` power files with 64 lines each). The data folder is git-ignored.

## Hardware

Developed on a single RTX 4070 (12 GB). Mixed precision is on by default for every learned model and
batch sizes / latent sizes live in YAML (`configs/`), not in code.
