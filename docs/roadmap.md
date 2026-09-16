# ChannelDreamer-6G — Roadmap

Hardware constraint for every phase: a single local RTX 4070 with **12 GB VRAM** (driver CUDA
12.2, torch installed from the cu126 index). Batch sizes, sequence lengths and latent sizes
live in YAML, mixed precision (bf16 autocast) is the default for every learned model, and
heavy modalities (LiDAR) are pre-tokenised offline so training never touches raw point clouds.

| Phase | Scope | Status |
|---|---|---|
| 0 | Positioning: related work, gap, contributions, risks (`docs/positioning.md`) | Done |
| 1 | MVP pipeline on DeepSense beam power: loader, shared windowing, synthetic generator, regime labelling, regime-decomposed metrics, fully-observed offline MDP, reactive + Markov baselines | Done: 14 tests passing, verified on synthetic data and real Scenario 33. Predict-then-act (transformer forecaster + greedy controller) being added now |
| 2 | WiWorld-RealData complex-CIR pipeline (dual-band 3.7 / 6.775 GHz, single public route) for the physics-grounded latent constraints of C3 | Scaffolded against a synthetic CIR generator; real-data validation pending the dataset download |
| 3 | Multi-modal world model: camera (ResNet), LiDAR (offline PointNet-style tokens), GPS trajectory (MLP) encoders feeding a DreamerV3-style RSSM trained by sequence prediction | Done: camera (frozen ResNet-18, cached features), LiDAR (offline FPS+kNN grouping + mini-PointNet tokens, cached), GPS trajectory MLP and the DreamerV3 RSSM are implemented and tested on real Scenario 33 batches; training loop and evaluation vs. baselines are next |
| 4 | Actor-critic decision layer over imagined RSSM rollouts with switching-cost reward; horizon-1 ablation; switching-cost sweep | In progress: straight-through categorical actor + critic on imagined rollouts implemented and tested (`rl/actor_critic.py`); smoke run on real Scenario 33 done (`scripts/train_actor_critic.py`, see `docs/methodology.md` §6); full experiment and sweeps not started |
| 5 | Transfer and generalisation: real vs. synthetic regimes, across DeepSense scenarios, cross-band on WiWorld | Not started |

## Phase details

**Phase 1 (done).** `python -m channeldreamer.scripts.train_baseline --synthetic` and
`--data-root data --scenario 33` both run end to end and print regime-decomposed tables. See
`docs/methodology.md` §0 for the real-data findings (1-based labels, NaN beams, switching-cost
calibration).

**Phase 2 (scaffold).** `channeldreamer.data.wiworld` mirrors the DeepSense loader pattern:
manifest resolution, quality-flag parsing, complex dual-band CIR loading, all with configurable
column names because the manifest format has not been inspected yet. Tested only against
`generate_synthetic_cir()`.

**Phase 3 (in progress).** Encoders in `channeldreamer.models.encoders`, RSSM in
`channeldreamer.models.world_model`. Offline caches (`prepare_data.py --pretokenize-lidar
--precompute-camera`) keep raw point clouds and JPEGs out of the training loop. Start small
(`deter_dim=256`) and grow only if memory allows; `scripts/profile_phase3.py` reports
`torch.cuda.max_memory_allocated()` per stage so Phase-4 headroom is known. Remaining: the
world-model training script, regime-decomposed evaluation of `WorldModel.predict_scores`
against the baselines, and the physics losses of C3 once WiWorld data are available.

**Phase 4.** Actor + critic on imagined rollouts (`channeldreamer.rl.actor_critic`), with the
switching penalty computed analytically from the known action sequence. Required ablation:
imagination horizon = 1.

**Phase 5.** Reuse the same windowing, regime labeller and metrics on other scenarios and on
synthetic regimes to test whether the transition-time advantage transfers.

## Model-usage guidance for development

- **Sonnet** for routine implementation: loaders, scripts, tests, plotting, refactors.
- **Fable** for RSSM and physics-loss design (KL balancing, VICReg / IDM terms, phase
  consistency), for the actor-critic derivation, and for hard debugging (NaN losses, memory
  blow-ups, silent shape bugs).
