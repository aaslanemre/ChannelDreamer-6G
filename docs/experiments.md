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

- **What:** `python -m channeldreamer.scripts.train_policy --wm-checkpoint runs/world_model/checkpoint.pt
  --switching-penalty 0 --imagination-horizon 15 --target 0.10`. Actor + critic on the frozen
  19k-step world model, 256 rollout starts per update, horizon 15. Stopping metric: greedy
  imagined regret against the model's own reward table on 64 fixed validation rollouts;
  target 0.10 dB (below the world model's 0.15 dB held-out error, so the actor's suboptimality
  cannot be the limiting factor); stop after two consecutive evaluations at or below target,
  or on a 20-evaluation plateau.
- **Run 1** (`actor_lr = 3e-4`, `entropy_scale = 3e-4`): plateaued at 0.148 dB (best at update
  14750 of 19750, 8 min) without reaching the target. Checks passed formally (13 distinct beams
  on validation, 2.7 % of rollout starts over 0.5 dB) but the policy was coarse: only 11 of 64
  beams ever received more than 1 % probability, and it agreed with the world model's greedy
  beam on just 16 % of validation windows, typically choosing an adjacent beam. Held-out:
  0.19 dB stable / 0.55 dB transition - worse than the world model's greedy one-step policy
  (0.15 / 0.42) on both regimes. Diagnosis: the exact softmax policy gradient is proportional
  to the probability of a beam, so beams the policy stops selecting can no longer be
  rediscovered (vanishing gradient), and the entropy bonus was too weak to prevent it.
  Files: `results/stage2_c0_imagined_regret.json`, `figures/stage2_c0_imagined_regret.png`,
  `results/stage2_c0_final_comparison.json`, `figures/stage2_c0_final_comparison.png`.
- **Run 2** (`actor_lr = 1e-3`, `entropy_scale = 3e-3`, everything else identical): target met
  at updates 500 and 750 (21 s), best greedy imagined regret 0.056 dB; 21 distinct beams on
  validation, 0.1 % of starts over 0.5 dB. **This is the Stage-2 result.** Held-out test, power
  loss in dB (same split and baselines as Stage 1):

  | method | stable | transition | overall | distinct beams |
  |---|---|---|---|---|
  | reactive | 0.18 | 0.43 | 0.20 | 54 |
  | predict-then-act | 0.12 | 0.46 | 0.15 | 55 |
  | world model, greedy one-step | 0.15 | 0.42 | 0.18 | 49 |
  | actor-critic, c = 0 | 0.15 | 0.53 | 0.19 | 22 |

- **Headline (honest):** at c = 0 the decision layer adds nothing beyond the world model's own
  one-step prediction. It ties the world-model greedy policy on stable windows (0.15 dB) and
  is *worse* at transitions (0.53 vs 0.42 dB; 90 windows), i.e. the opposite of the central
  hypothesis at this stage. Net MDP reward at zero penalty: reactive -3.695, predict-then-act
  -3.645, world-model greedy -3.668, actor-critic -3.680 dB.
- **Caveat on the stopping rule:** the two-consecutive-crossings rule stopped run 2 while the
  imagined regret (0.103 -> 0.079 -> 0.056), the distinct-beam count (14 -> 17 -> 21) and the
  entropy were all still changing steeply (see the figure). The policy is therefore converged
  *to the target*, not to a plateau; with 22 distinct beams on test against 49 for the greedy
  world-model policy it still discards resolution the model has. Stage 3 should require a
  plateau (or a tighter target, e.g. 0.03 dB) before evaluation.
- **Files:** `results/stage2_c0_run2_imagined_regret.json`, `figures/stage2_c0_run2_imagined_regret.png`,
  `results/stage2_c0_run2_final_comparison.json`, `figures/stage2_c0_run2_final_comparison.png`.
  Checkpoint: `runs/policy_c0_run2/policy.pt`. Write-up: `docs/methodology.md` §8.
