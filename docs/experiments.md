# Experiment log

One entry per stage / run, in plain language. Numbers here are copied from the linked
`results/*.json` files, which carry the timestamp and git commit of the run that produced them.
Checkpoints live under `runs/` (gitignored); everything needed to re-plot is in `results/`.

Shared setup for every entry unless stated otherwise: DeepSense 6G Scenario 33 (3981 samples,
18 drive segments), windows `H = 8`, `k = 1`, segment-level split with seed 0: 14 training
segments (of which segments 11 and 15 are the validation segments for stopping rules) and
4 held-out test segments `[2, 3, 10, 12]` = 937 windows, 90 of them in the transition regime.
Regime labels: `configs/phase1_baseline.yaml`. Metric: power loss in dB of the chosen beam vs
the best beam at `t + 1` (0 = perfect), reported per regime.

## Stage 0 - Phase-1 baselines (backfilled)

- **What:** reactive (hold the last observed best beam), first-order Markov, predict-then-act
  (beam-equivariant transformer forecaster + greedy argmax, 68k params, 40 epochs with early
  stopping), oracle. No world model involved.
- **Headline:** predict-then-act is the strongest baseline on stable windows (0.12 dB vs
  reactive 0.18); at transitions nothing beats reactive (0.43 dB): Markov 0.45, predict-then-act
  0.46. Top-1 accuracy is ~0.41 for all three because the optimal beam flickers between
  adjacent beams at 92 ms sampling; the dB loss is the informative metric.
- **Files:** `results/stage0_baselines.json`.

## Stage 1 - world-model training (Phase 3 closure)

- **What:** `python -m channeldreamer.scripts.train_world_model --max-steps 60000 --eval-every 1000`.
  Exogenous RSSM (`action_dim = 0`, `deter_dim = 256`, 16x16 categorical latents), cached
  camera / LiDAR / GPS features, `B = 16 x T = 16`, AdamW `lr = 1e-4`. Stopping rule on the
  validation segments only: greedy one-step regret <= reactive on both regimes for two
  consecutive evaluations.
- **Headline:** target met at 18k and 19k steps (340 s, peak 290 MB); the 19k checkpoint is the
  frozen world model for Phase 4. Held-out test: 0.15 dB stable / 0.42 dB transition vs
  reactive 0.18 / 0.43 - beats reactive on stable windows, matches it at transitions (90
  windows, within noise). Still behind predict-then-act on stable windows (0.12). The regret
  kept improving for ~15k steps after the loss curve had flattened, driven by the KL rising from
  0.7 to 0.9 nats as the latent took up the observation. Posterior reproduces the observed table
  with 0.15 dB regret; greedy policy uses 49 distinct beams (optimal: 54) - no collapse.
- **Files:** `results/stage1_world_model_regret.json`, `figures/stage1_world_model_regret.png`
  (test curve), `figures/stage1_world_model_regret_val.png` (validation curve used for the
  stopping rule), `results/stage1_final_comparison.json`, `figures/stage1_final_comparison.png`.
  Checkpoint: `runs/world_model/checkpoint.pt`. Write-up: `docs/methodology.md` §7.

## Stage 2 - actor-critic at c = 0 on the frozen world model

*(pending: entry is added when `results/stage2_*.json` exist)*
