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
from ..utils import load_config, plot_regime_bars, plot_regret_curve, save_figure, save_results, seed_everything
from .train_actor_critic import DEFAULT_CONFIG, _Scorer, window_side_arrays


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--wm-checkpoint", default=None)
    p.add_argument("--record-only", default=None, metavar="RUN_DIR",
                   help="do not train: write results/ and figures/ from RUN_DIR/summary.json (runs that predate the recording code)")
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
    p.add_argument("--stop-on", choices=("target", "plateau"), default="target",
                   help="'target': stop after --confirm consecutive evals at/below --target (Stage 2); "
                        "'plateau': ignore the target and stop only when validation has not improved for --patience evals (Stage 3)")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-rollouts", type=int, default=64, help="validation sequences used for imagined evaluation")
    p.add_argument("--no-predict-then-act", action="store_true")
    p.add_argument("--init-actor", default=None, metavar="POLICY_PT",
                   help="warm-start the actor (and critic) from a saved policy.pt, e.g. the converged c=0 policy of the same horizon "
                        "(curriculum on the switching penalty; a cold start with c > 0 collapses to a few beams)")
    p.add_argument("--out", default="runs/policy")
    p.add_argument("--experiment", default=None,
                   help="name for results/<experiment>_*.json and figures/<experiment>_*.png (default: derived from --out)")
    return p


@torch.no_grad()
def imagined_eval(trainer: ImaginedActorCritic, batches: list[dict[str, torch.Tensor]]) -> dict:
    """Validation rollouts: the actor's greedy behaviour against the model's table, with and without
    the switching penalty, plus two reference policies computed on the same imagined trajectories:

    * ``track``: choose the model's best beam at every step, paying every switch;
    * ``hold``: keep the currently served beam for the whole rollout (no switches).

    ``greedy_net_reward_db`` (actor, penalty included) is the objective; for ``c = 0`` it equals
    ``track`` minus the table regret.  The state-dependence check asks how many rollout starts
    the actor loses to the better of the two references by more than 0.5 dB.
    """
    trainer.actor.eval()
    c = trainer.switching_penalty
    acc = {k: [] for k in ("greedy_regret", "expected_regret", "switch", "entropy", "actor_net", "track_net", "hold_net", "actor_switch")}
    per_gap, per_regret = [], []
    for batch in batches:
        start, prev = trainer.start_states(batch)
        roll = trainer.imagine_rollout(start, prev)
        table = roll.reward_table  # (B, H, A)
        best_val, best_beam = table.max(-1)
        prev0 = prev.argmax(-1)  # (B,)
        a_g = roll.logits.argmax(-1)  # (B, H)
        a_prev = torch.cat([prev0[:, None], a_g[:, :-1]], 1)
        actor_val = table.gather(-1, a_g[..., None]).squeeze(-1)
        actor_sw = (a_g != a_prev).float()
        actor_net = actor_val - c * actor_sw
        track_prev = torch.cat([prev0[:, None], best_beam[:, :-1]], 1)
        track_net = best_val - c * (best_beam != track_prev).float()
        hold_net = table.gather(-1, prev0[:, None, None].expand(-1, table.shape[1], 1)).squeeze(-1)
        g = best_val - actor_val
        acc["greedy_regret"].append(g.flatten())
        per_regret.append(g.mean(1))
        acc["expected_regret"].append((best_val - roll.beam_reward).flatten())
        acc["switch"].append(roll.switch.flatten())
        acc["actor_switch"].append(actor_sw.flatten())
        acc["entropy"].append(trainer.actor.entropy(roll.logits).flatten())
        acc["actor_net"].append(actor_net.flatten())
        acc["track_net"].append(track_net.flatten())
        acc["hold_net"].append(hold_net.flatten())
        per_gap.append(torch.maximum(track_net.mean(1), hold_net.mean(1)) - actor_net.mean(1))
    m = {k: torch.cat(v) for k, v in acc.items()}
    per_gap, per_regret = torch.cat(per_gap), torch.cat(per_regret)
    trainer.actor.train()
    return {
        "greedy_regret_db": float(m["greedy_regret"].mean()), "expected_regret_db": float(m["expected_regret"].mean()),
        "greedy_net_reward_db": float(m["actor_net"].mean()), "track_net_reward_db": float(m["track_net"].mean()),
        "hold_net_reward_db": float(m["hold_net"].mean()), "greedy_switch_rate": float(m["actor_switch"].mean()),
        "switch_rate": float(m["switch"].mean()), "entropy": float(m["entropy"].mean()),
        "per_start_p50": float(per_regret.quantile(0.5)), "per_start_p90": float(per_regret.quantile(0.9)),
        "per_start_max": float(per_regret.max()), "frac_starts_over_0.5db": float((per_regret > 0.5).float().mean()),
        "gap_to_reference_p90": float(per_gap.quantile(0.9)), "frac_starts_gap_over_0.5db": float((per_gap > 0.5).float().mean()),
    }


def record_policy_run(exp: str, summary: dict) -> None:
    """Persist a run's regret curve, checks and (if the checks passed) the held-out table under
    ``results/<exp>_*.json`` and ``figures/<exp>_*.png``.  Works from ``summary.json`` alone."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = summary["args"]
    hp = summary.get("hyperparameters") or {k: a[k] for k in ("switching_penalty", "imagination_horizon", "actor_lr", "critic_lr",
                                                                "entropy_scale", "batch", "seq", "target", "confirm", "patience")}
    split = summary.get("split") or {"val_segments": summary.get("val_segments"), "test_segments": summary.get("test_segments")}
    summary.setdefault("policy_checkpoint", str(Path(a["out"]) / "policy.pt"))
    summary.setdefault("distinct_beams_test", {})
    summary.setdefault("distinct_optimal_beams_test", None)
    curve = [{"step": e["step"], "seconds": e["seconds"], "greedy_regret_db": e["val"]["greedy_regret_db"],
              "expected_regret_db": e["val"]["expected_regret_db"], "entropy": e["val"]["entropy"], "switch_rate": e["val"]["switch_rate"],
              "per_start_p90": e["val"]["per_start_p90"], "frac_starts_over_0.5db": e["val"]["frac_starts_over_0.5db"],
              "greedy_net_reward_db": e["val"].get("greedy_net_reward_db"), "track_net_reward_db": e["val"].get("track_net_reward_db"),
              "hold_net_reward_db": e["val"].get("hold_net_reward_db"), "greedy_switch_rate": e["val"].get("greedy_switch_rate"),
              "frac_starts_gap_over_0.5db": e["val"].get("frac_starts_gap_over_0.5db"),
              "val_distinct_beams": e["val_distinct_beams"], "actor_loss": e["train"]["actor_loss"], "critic_loss": e["train"]["critic_loss"]}
             for e in summary["log"]]
    save_results(f"{exp}_imagined_regret", {
        "description": "Actor-critic on the frozen world model: greedy/expected imagined regret against the model's reward table on "
                       "validation rollouts vs update, with collapse (distinct beams) and state-dependence (per-start spread) checks",
        "world_model_checkpoint": summary["args"]["wm_checkpoint"], "world_model_step": summary["wm_step"],
        "policy_checkpoint": summary["policy_checkpoint"], "split": split, "hyperparameters": hp, "stop_reason": summary["stop_reason"],
        "best_step": summary["best_step"], "peak_gpu_mb": summary["peak_gpu_mb"], "checks": summary["checks"],
        "checks_passed": summary["checks_passed"], "curve": curve})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    st = [c["step"] for c in curve]
    if curve and curve[0].get("greedy_net_reward_db") is not None:
        axes[0].plot(st, [c["greedy_net_reward_db"] for c in curve], marker="o", ms=3, label="actor (greedy), net reward")
        axes[0].plot(st, [c["track_net_reward_db"] for c in curve], ls="--", color="k", alpha=0.7, label="track model's best beam")
        axes[0].plot(st, [c["hold_net_reward_db"] for c in curve], ls=":", color="k", alpha=0.7, label="hold current beam")
        axes[0].set_ylabel(f"imagined net reward, c={hp['switching_penalty']:g} dB")
    else:
        axes[0].plot(st, [c["greedy_regret_db"] for c in curve], marker="o", ms=3, label="greedy")
        axes[0].plot(st, [c["expected_regret_db"] for c in curve], marker="o", ms=3, label="expected (stochastic policy)")
        axes[0].axhline(hp["target"], color="k", ls="--", lw=1, label=f"target {hp['target']} dB")
        axes[0].set_ylabel("imagined regret vs model table (dB)")
    axes[0].set_xlabel("actor-critic update")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{exp}: imagined regret (validation)")
    axes[1].plot(st, [c["val_distinct_beams"] for c in curve], marker="o", ms=3, color="#2ca02c", label="distinct beams (val windows)")
    axes[1].set_xlabel("actor-critic update")
    axes[1].set_ylabel("distinct beams")
    axes[1].grid(True, alpha=0.3)
    ax2 = axes[1].twinx()
    ax2.plot(st, [c["entropy"] for c in curve], color="#9467bd", label="policy entropy (nats)")
    ax2.set_ylabel("entropy (nats)")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8)
    axes[1].set_title("collapse check")
    fig.tight_layout()
    save_figure(f"{exp}_imagined_regret", fig)
    if summary["checks_passed"] and summary["test_metrics"]:
        results = summary["test_metrics"]
        table = {n: {r: results[n][r]["power_loss_db_mean"] for r in ("stable", "transition", "overall")} for n in results}
        save_results(f"{exp}_final_comparison", {
            "description": "Held-out test segments: power loss (dB) by regime for reactive, predict-then-act, world-model greedy and "
                           "the actor-critic; same split as stage1_final_comparison",
            "world_model_checkpoint": summary["args"]["wm_checkpoint"], "policy_checkpoint": summary["policy_checkpoint"],
            "split": split, "hyperparameters": hp, "metrics": results, "power_loss_db": table, "mdp_rewards": summary["test_rewards"],
            "distinct_beams_test": summary["distinct_beams_test"], "distinct_optimal_beams_test": summary["distinct_optimal_beams_test"]})
        save_figure(f"{exp}_final_comparison", plot_regime_bars(table, f"{exp}: held-out power loss by regime"))
    print(f"[results] results/{exp}_*.json, figures/{exp}_*.png")


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if not args.record_only and not args.wm_checkpoint:
        raise SystemExit("--wm-checkpoint is required unless --record-only is given")
    if args.record_only:
        summary = json.load(open(Path(args.record_only) / "summary.json"))
        record_policy_run(args.experiment or Path(args.record_only).name, summary)
        return summary
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
    if args.init_actor:
        init = torch.load(args.init_actor, map_location=dev, weights_only=False)["actor_critic"]
        trainer.actor.load_state_dict(init["actor"])
        trainer.critic.load_state_dict(init["critic"])
        trainer.critic_ema.load_state_dict(init["critic_ema"])
        print(f"[actor-critic] warm-started actor/critic from {args.init_actor} (trained at c={init['switching_penalty']:g})")
    print(f"[actor-critic] c={args.switching_penalty:g} dB, horizon {ac_cfg.imagination_horizon}, actor/critic lr {args.actor_lr:g}/{args.critic_lr:g}, "
          f"{args.batch}x{args.seq} rollout starts per update; stop at greedy imagined regret <= {args.target} dB on "
          f"{len(val_idx)} validation rollouts for {args.confirm} consecutive evals")

    # ------------------------------------------------------------- training loop
    log, best, best_state, best_step, since_best, met, stop_reason = [], -float("inf"), None, 0, 0, 0, "max_updates"
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
            # score to maximise: the imagined objective (net reward incl. penalty) in plateau mode, -regret in target mode
            score = ev["greedy_net_reward_db"] if args.stop_on == "plateau" else -ev["greedy_regret_db"]
            improved = score > best + 1e-4
            if improved:
                best, best_step, since_best = score, step, 0
                best_state = copy.deepcopy(trainer.actor.state_dict())
            else:
                since_best += 1
            print(f"upd {step:6d} ({entry['seconds']:4.0f}s) actor {entry['train']['actor_loss']:+.3f} critic {entry['train']['critic_loss']:.3f} | "
                  f"val net reward greedy {ev['greedy_net_reward_db']:.3f} (track {ev['track_net_reward_db']:.3f}, hold {ev['hold_net_reward_db']:.3f}) | "
                  f"table regret {ev['greedy_regret_db']:.3f} dB, gap>0.5dB {100 * ev['frac_starts_gap_over_0.5db']:.0f}% of starts, "
                  f"entropy {ev['entropy']:.2f}, greedy switch {ev['greedy_switch_rate']:.2f}, distinct beams {beams}"
                  f"{' *' if improved else ''}{' TARGET' if target else ''}", flush=True)
            if args.stop_on == "target" and met >= args.confirm:
                stop_reason = f"target met for {args.confirm} consecutive evaluations"
                break
            if since_best >= args.patience:
                stop_reason = f"validation plateau: no improvement for {args.patience} evaluations"
                break
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[done] {stop_reason}; best validation score ({'net reward' if args.stop_on == 'plateau' else '-regret'}) {best:.3f} dB "
          f"at update {best_step}; peak GPU {peak:.0f} MB")
    trainer.actor.load_state_dict(best_state)

    # ------------------------------------------------------------- checks before test
    ev = imagined_eval(trainer, val_batches)
    val_actions = trainer.predict_scores(W["val"], extra=extras["val"]).argmax(1)
    checks = {"val_distinct_beams": int(len(np.unique(val_actions))), "val_distinct_optimal": int(len(np.unique(W["val"].target_beam))),
              "val_imagined": ev}
    print("\n=== Checks on validation (best actor) ===")
    print(f"distinct beams chosen on {len(W['val'])} validation windows: {checks['val_distinct_beams']} (truly optimal: {checks['val_distinct_optimal']})")
    print(f"greedy imagined net reward {ev['greedy_net_reward_db']:.3f} dB vs references: track-best-beam {ev['track_net_reward_db']:.3f}, "
          f"hold-current-beam {ev['hold_net_reward_db']:.3f} dB (c={args.switching_penalty:g}); table regret {ev['greedy_regret_db']:.3f} dB")
    print(f"per-rollout-start: table regret p50 {ev['per_start_p50']:.3f} p90 {ev['per_start_p90']:.3f} max {ev['per_start_max']:.3f} dB; "
          f"gap to the better reference p90 {ev['gap_to_reference_p90']:.3f} dB, starts losing > 0.5 dB to it: {100 * ev['frac_starts_gap_over_0.5db']:.1f}%")
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
    passed = checks["val_distinct_beams"] >= 10 and ev["frac_starts_gap_over_0.5db"] < 0.10
    print(f"checks {'PASSED' if passed else 'FAILED'} (need >= 10 distinct beams and < 10% of starts losing > 0.5 dB to the better reference policy)")

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
               "test_metrics": results, "test_rewards": rewards, "policy_checkpoint": str(out / "policy.pt"),
               "hyperparameters": {"switching_penalty": args.switching_penalty, "imagination_horizon": args.imagination_horizon,
                                   "actor_lr": args.actor_lr, "critic_lr": args.critic_lr, "entropy_scale": args.entropy_scale,
                                   "batch": args.batch, "seq": args.seq, "target": args.target, "confirm": args.confirm,
                                   "patience": args.patience, "stop_on": args.stop_on, "init_actor": args.init_actor,
                                   "gamma": ac_cfg.gamma, "lambda": ac_cfg.lambda_},
               "split": {"fit_segments": fit_segs.tolist(), "val_segments": val_segs.tolist(), "test_segments": test_segs.tolist(),
                         "n_val_windows": len(W["val"]), "n_test_windows": len(W["test"]), "n_test_transition": int((R["test"] == 1).sum())},
               "distinct_beams_test": {n: int(len(np.unique(s.argmax(1)))) for n, s in scores.items()} if passed else {},
               "distinct_optimal_beams_test": int(len(np.unique(W["test"].target_beam)))}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    torch.save({"actor_critic": trainer.state_dict(), "wm_checkpoint": args.wm_checkpoint}, out / "policy.pt")
    print(f"\n[saved] {out / 'summary.json'} and {out / 'policy.pt'}")

    record_policy_run(args.experiment or out.name, summary)
    return summary


if __name__ == "__main__":
    main()
