"""Stage 2/3: train the actor-critic on a frozen world model until convergence, then evaluate.

    python -m channeldreamer.scripts.train_policy --wm-checkpoint runs/world_model/checkpoint.pt \
        --switching-penalty 0 --imagination-horizon 15 --out runs/policy_c0

Discipline (same as ``train_world_model``): the training segments are split into fit and
validation segments with the same seed, so the validation segments are identical to Stage 1's;
the test segments are not touched until the final table.  Every ``--eval-every`` updates the
policy is evaluated on a fixed set of validation rollouts:

* **greedy imagined regret** (dB): ``max_k R̂(s_{j+1})[k] - R̂(s_{j+1})[argmax pi(.|s_j)]``
  averaged over imagined steps - the actor's suboptimality against the model's own table under
  the greedy action used at evaluation time; the stopping metric;
* expected imagined regret under the stochastic policy, entropy, switch rate;
* distinct beams chosen on the real validation windows (collapse check);
* per-rollout spread of the greedy regret (state-dependence check): p50 / p90 / max and the
  fraction of rollout starts whose mean regret exceeds 0.5 dB.

Stop when the greedy imagined regret is <= ``--target`` for ``--confirm`` consecutive
evaluations, on a ``--patience`` plateau, or at ``--max-updates``.  The best-validation actor
is kept and used for the final held-out comparison against reactive, predict-then-act and the
world model's own greedy one-step policy on the test segments.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data import load_scenario, make_windows, split_segments
from ..data.modalities import load_camera_features, load_gps, load_lidar_tokens, trajectory_windows
from ..envs import OfflineBeamEnv
from ..eval import compute_metrics, format_regime_table, label_regimes, regime_for_windows
from ..models import PredictThenActBaseline, ReactiveBaseline
from ..models.encoders import EncoderConfig
from ..models.world_model import RSSMConfig, WorldModel, make_sequence_batch
from ..rl import ActorCriticConfig, ImaginedActorCritic
from ..utils import load_config, seed_everything
from .train_actor_critic import DEFAULT_CONFIG, _Scorer, window_side_arrays


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--val-segments", type=int, default=2)
    p.add_argument("--switching-penalty", type=float, default=0.0)
    p.add_argument("--imagination-horizon", type=int, default=15)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--entropy-scale", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=16, help="real sequences per update (x T rollout starts)")
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--max-updates", type=int, default=30000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--target", type=float, default=0.10, help="greedy imagined regret (dB) to reach on validation")
    p.add_argument("--confirm", type=int, default=2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-rollouts", type=int, default=64, help="validation sequences used for imagined evaluation")
    p.add_argument("--no-predict-then-act", action="store_true")
    p.add_argument("--out", default="runs/policy")
    return p


@torch.no_grad()
def imagined_eval(trainer: ImaginedActorCritic, batches: list[dict[str, torch.Tensor]]) -> dict:
    """Greedy / expected imagined regret against the model's table on fixed validation rollouts."""
    trainer.actor.eval()
    per_start, greedy, expected, switch, ent = [], [], [], [], []
    for batch in batches:
        start, prev = trainer.start_states(batch)
        roll = trainer.imagine_rollout(start, prev)
        best = roll.reward_table.max(-1).values  # (B, H)
        a_greedy = roll.logits.argmax(-1)  # (B, H)
        g = best - roll.reward_table.gather(-1, a_greedy[..., None]).squeeze(-1)
        greedy.append(g.flatten())
        per_start.append(g.mean(1))
        expected.append((best - roll.beam_reward).flatten())
        switch.append(roll.switch.flatten())
        ent.append(trainer.actor.entropy(roll.logits).flatten())
    g_all, per = torch.cat(greedy), torch.cat(per_start)
    trainer.actor.train()
    return {
        "greedy_regret_db": float(g_all.mean()), "expected_regret_db": float(torch.cat(expected).mean()),
        "switch_rate": float(torch.cat(switch).mean()), "entropy": float(torch.cat(ent).mean()),
        "per_start_p50": float(per.quantile(0.5)), "per_start_p90": float(per.quantile(0.9)),
        "per_start_max": float(per.max()), "frac_starts_over_0.5db": float((per > 0.5).float().mean()),
    }


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, [])
    seed_everything(cfg.seed)
    dev = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- data (Stage-1 split)
    ds = load_scenario(args.data_root, args.scenario, keep_frame=True)
    step_labels = label_regimes(ds.power, ds.segment_ids, **cfg.regimes.to_dict())
    d = cfg.data
    windows = make_windows(ds.power, ds.segment_ids, history=d.history, horizon=d.horizon, stride=d.stride)
    train_segs, test_segs = split_segments(ds.segment_ids, d.test_fraction, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    val_segs = np.sort(rng.choice(train_segs, args.val_segments, replace=False))
    fit_segs = np.setdiff1d(train_segs, val_segs)
    W = {n: windows.subset(np.isin(windows.segment_ids, s)) for n, s in (("fit", fit_segs), ("val", val_segs), ("test", test_segs), ("train", train_segs))}
    R = {n: regime_for_windows(step_labels, w.target_index) for n, w in W.items()}
    print(f"[split] fit segs {fit_segs.tolist()}, val segs {val_segs.tolist()}, test segs {test_segs.tolist()}; "
          f"val {len(W['val'])} windows ({int((R['val'] == 1).sum())} transition), test {len(W['test'])} ({int((R['test'] == 1).sum())})")

    ck = torch.load(args.wm_checkpoint, map_location=dev, weights_only=False)
    mods = tuple(ck["modalities"])
    cache = ds.scenario_dir / "cache"
    side: dict[str, np.ndarray] = {}
    if "camera" in mods:
        side["camera_feat"] = load_camera_features(cache, ds.sample_index)
    if "lidar" in mods:
        side["lidar_tokens"], side["lidar_centroids"] = load_lidar_tokens(cache / "lidar", ds.sample_index)
    if "trajectory" in mods:
        side["trajectory"] = trajectory_windows(load_gps(ds).user_local_m, ds.segment_ids, EncoderConfig().trajectory_tokens)
    extras = {n: window_side_arrays(w, **side) for n, w in W.items()}

    seq_all = make_windows(ds.power, ds.segment_ids, history=args.seq, horizon=1)
    seq = {n: seq_all.subset(np.isin(seq_all.segment_ids, s)) for n, s in (("fit", fit_segs), ("val", val_segs))}

    def batch_from(sw, idx):
        return make_sequence_batch(sw, idx, camera_feat=side.get("camera_feat"), lidar_tokens=side.get("lidar_tokens"),
                                   lidar_centroids=side.get("lidar_centroids"), trajectory=side.get("trajectory"), device=dev)

    val_idx = np.random.default_rng(1).choice(len(seq["val"]), min(args.val_rollouts, len(seq["val"])), replace=False)
    val_batches = [batch_from(seq["val"], val_idx[s : s + 16]) for s in range(0, len(val_idx), 16)]

    # ------------------------------------------------------------- frozen world model
    rcfg = RSSMConfig(**ck["rssm_config"])
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim, modalities=mods)).to(dev)
    wm.load_state_dict(ck["world_model"])
    print(f"[world model] {args.wm_checkpoint}: step {ck.get('step')}, exogenous={not rcfg.action_dim}, held-out regret at save {ck.get('test')}")

    ac_cfg = ActorCriticConfig(imagination_horizon=args.imagination_horizon, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                               entropy_scale=args.entropy_scale, batch_size=args.batch)
    torch.manual_seed(cfg.seed)
    trainer = ImaginedActorCritic(wm, ac_cfg, switching_penalty=args.switching_penalty)
    print(f"[actor-critic] c={args.switching_penalty:g} dB, horizon {ac_cfg.imagination_horizon}, actor/critic lr {args.actor_lr:g}/{args.critic_lr:g}, "
          f"{args.batch}x{args.seq} rollout starts per update; stop at greedy imagined regret <= {args.target} dB on "
          f"{len(val_idx)} validation rollouts for {args.confirm} consecutive evals")

    # ------------------------------------------------------------- training loop
    log, best, best_state, best_step, since_best, met, stop_reason = [], float("inf"), None, 0, 0, 0, "max_updates"
    run = {}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.max_updates + 1):
        m = trainer.train_step(batch_from(seq["fit"], rng.choice(len(seq["fit"]), args.batch, replace=False)))
        if not all(np.isfinite(v) for v in m.values()):
            raise RuntimeError(f"non-finite diagnostics at update {step}: {m}")
        for k, v in m.items():
            run[k] = run.get(k, 0.0) + v / args.eval_every
        if step % args.eval_every == 0:
            ev = imagined_eval(trainer, val_batches)
            beams = int(len(np.unique(trainer.predict_scores(W["val"], extra=extras["val"]).argmax(1))))
            entry = {"step": step, "seconds": time.time() - t0, "train": run, "val": ev, "val_distinct_beams": beams}
            log.append(entry)
            run = {}
            target = ev["greedy_regret_db"] <= args.target
            met = met + 1 if target else 0
            improved = ev["greedy_regret_db"] < best - 1e-4
            if improved:
                best, best_step, since_best = ev["greedy_regret_db"], step, 0
                best_state = copy.deepcopy(trainer.actor.state_dict())
            else:
                since_best += 1
            print(f"upd {step:6d} ({entry['seconds']:4.0f}s) actor {entry['train']['actor_loss']:+.3f} critic {entry['train']['critic_loss']:.3f} | "
                  f"val imagined regret greedy {ev['greedy_regret_db']:.3f} expected {ev['expected_regret_db']:.3f} dB, "
                  f"p90/max per start {ev['per_start_p90']:.2f}/{ev['per_start_max']:.2f}, >0.5dB {100 * ev['frac_starts_over_0.5db']:.0f}%, "
                  f"entropy {ev['entropy']:.2f}, switch {ev['switch_rate']:.2f}, distinct beams {beams}"
                  f"{' *' if improved else ''}{' TARGET' if target else ''}", flush=True)
            if met >= args.confirm:
                stop_reason = f"target met for {args.confirm} consecutive evaluations"
                break
            if since_best >= args.patience:
                stop_reason = f"validation plateau: no improvement for {args.patience} evaluations"
                break
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[done] {stop_reason}; best validation greedy imagined regret {best:.3f} dB at update {best_step}; peak GPU {peak:.0f} MB")
    trainer.actor.load_state_dict(best_state)

    # ------------------------------------------------------------- checks before test
    ev = imagined_eval(trainer, val_batches)
    val_actions = trainer.predict_scores(W["val"], extra=extras["val"]).argmax(1)
    checks = {"val_distinct_beams": int(len(np.unique(val_actions))), "val_distinct_optimal": int(len(np.unique(W["val"].target_beam))),
              "val_imagined": ev}
    print("\n=== Checks on validation (best actor) ===")
    print(f"distinct beams chosen on {len(W['val'])} validation windows: {checks['val_distinct_beams']} (truly optimal: {checks['val_distinct_optimal']})")
    print(f"greedy imagined regret {ev['greedy_regret_db']:.3f} dB; per-rollout-start mean regret p50 {ev['per_start_p50']:.3f} "
          f"p90 {ev['per_start_p90']:.3f} max {ev['per_start_max']:.3f} dB; starts over 0.5 dB: {100 * ev['frac_starts_over_0.5db']:.1f}%")
    # spot check: three individual rollouts
    with torch.no_grad():
        start, prev = trainer.start_states(val_batches[0])
        roll = trainer.imagine_rollout(start, prev)
        best_beam = roll.reward_table.argmax(-1); chosen = roll.logits.argmax(-1)
        reg = (roll.reward_table.max(-1).values - roll.reward_table.gather(-1, chosen[..., None]).squeeze(-1))
    spot = []
    for i in (0, len(chosen) // 2, len(chosen) - 1):
        spot.append({"start": i, "current_beam": int(prev[i].argmax()), "chosen": chosen[i].tolist(), "model_best": best_beam[i].tolist(),
                     "regret_db": [round(x, 2) for x in reg[i].tolist()]})
        print(f"rollout start {i}: current beam {int(prev[i].argmax())}\n   chosen     {chosen[i].tolist()}\n   model best {best_beam[i].tolist()}"
              f"\n   regret dB  {[round(x, 2) for x in reg[i].tolist()]}")
    checks["spot_rollouts"] = spot
    passed = checks["val_distinct_beams"] >= 10 and ev["frac_starts_over_0.5db"] < 0.10
    print(f"checks {'PASSED' if passed else 'FAILED'} (need >= 10 distinct beams and < 10% of starts over 0.5 dB)")

    # ------------------------------------------------------------- final held-out comparison
    results, rewards = {}, {}
    if passed:
        methods = {"reactive": ReactiveBaseline().fit(W["train"])}
        if not args.no_predict_then_act:
            methods["predict-then-act"] = PredictThenActBaseline(**dict(cfg.predict_then_act.to_dict()), seed=int(cfg.seed)).fit(W["train"])
        methods["wm-greedy"] = _Scorer(wm, extras["test"])
        methods[f"actor-critic c={args.switching_penalty:g}"] = _Scorer(trainer, extras["test"])
        scores = {n: np.asarray(m.predict_scores(W["test"])) for n, m in methods.items()}
        results = {n: compute_metrics(s, W["test"], R["test"]) for n, s in scores.items()}
        print("\n=== Held-out test segments: regime-decomposed metrics ===")
        print(format_regime_table(results))
        print("distinct beams on test: " + ", ".join(f"{n} {len(np.unique(s.argmax(1)))}" for n, s in scores.items()))
        for c in sorted({0.0, args.switching_penalty}):
            env = OfflineBeamEnv(W["test"], reward_unit=cfg.env.reward_unit, switching_penalty=c)
            rewards[f"c={c:g}"] = {}
            print(f"\nMDP net reward (dB) at switching penalty {c:g}: " + "; ".join(
                f"{n} {env.evaluate_policy_reward(s.argmax(1)).mean:.3f} ({100 * np.mean(s.argmax(1) != W['test'].last_beam()):.0f}% switches)"
                for n, s in scores.items()))
            for n, s in scores.items():
                pr = env.evaluate_policy_reward(s.argmax(1))
                rewards[f"c={c:g}"][n] = {"mean": pr.mean, "regret": pr.regret, "n_switches": pr.n_switches}
        results = {m: {r: v.as_dict() for r, v in per.items()} for m, per in results.items()}
    else:
        print("\nchecks failed: the held-out comparison was NOT run")

    summary = {"args": vars(args), "wm_step": ck.get("step"), "log": log, "best_step": best_step, "best_val_greedy_regret": best,
               "stop_reason": stop_reason, "peak_gpu_mb": peak, "checks": checks, "checks_passed": passed,
               "test_metrics": results, "test_rewards": rewards, "val_segments": val_segs.tolist(), "test_segments": test_segs.tolist()}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    torch.save({"actor_critic": trainer.state_dict(), "wm_checkpoint": args.wm_checkpoint}, out / "policy.pt")
    print(f"\n[saved] {out / 'summary.json'} and {out / 'policy.pt'}")
    return summary


if __name__ == "__main__":
    main()
