"""Phase-4 smoke run: world model -> actor-critic on imagined rollouts -> regime-decomposed eval.

    python -m channeldreamer.scripts.train_actor_critic --data-root data --scenario 33 \
        --wm-steps 600 --ac-steps 300 --switching-penalties 0.0,0.5 --out runs/phase4_smoke

Steps
-----
1. Load real Scenario 33 with the Phase-1 config (same windowing, regime labels and
   segment-level split as ``train_baseline``), plus the cached camera / LiDAR features and
   GPS trajectories (never raw images at train time, see ``docs/methodology.md`` §0.1).
2. Train the multimodal world model by sequence prediction on the training segments.
3. For each switching penalty ``c``: train a fresh actor + critic on imagined rollouts from
   the frozen world model (:class:`channeldreamer.rl.ImaginedActorCritic`).
4. Evaluate every policy and the existing baselines (reactive, Markov, predict-then-act,
   oracle, plus the world model's own greedy one-step prediction) on the held-out segments
   with the shared regime-decomposed metrics and the offline MDP net reward at each penalty.

This is a smoke run (a few hundred updates), not the full experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data import load_scenario, make_windows, split_segments
from ..data.modalities import load_camera_features, load_gps, load_lidar_tokens, trajectory_windows
from ..envs import OfflineBeamEnv
from ..eval import compute_metrics, format_regime_table, label_regimes, regime_for_windows
from ..eval.regimes import regime_summary
from ..models import MarkovBaseline, OracleBaseline, PredictThenActBaseline, ReactiveBaseline
from ..models.encoders import EncoderConfig, count_parameters
from ..models.world_model import RSSMConfig, WorldModel, make_sequence_batch
from ..rl import ActorCriticConfig, ImaginedActorCritic
from ..utils import load_config, seed_everything

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "phase1_baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--modalities", default="power,camera,lidar,trajectory")
    # world model
    p.add_argument("--wm-steps", type=int, default=600)
    p.add_argument("--wm-lr", type=float, default=1e-4)
    p.add_argument("--wm-batch", type=int, default=16)
    p.add_argument("--wm-seq", type=int, default=16)
    p.add_argument("--deter-dim", type=int, default=256)
    p.add_argument("--action-conditioned", action="store_true",
                   help="ablation only: feed actions to the RSSM (default: exogenous, action_dim=0; see rl.actor_critic)")
    # actor-critic
    p.add_argument("--ac-steps", type=int, default=300)
    p.add_argument("--ac-batch", type=int, default=16, help="real sequences per update (every posterior step is a rollout start)")
    p.add_argument("--imagination-horizon", type=int, default=15)
    p.add_argument("--actor-lr", type=float, default=3e-5)
    p.add_argument("--critic-lr", type=float, default=3e-5)
    p.add_argument("--entropy-scale", type=float, default=3e-4)
    p.add_argument("--switching-penalties", default="0.0,0.5", help="comma-separated dB penalties; one policy each")
    p.add_argument("--no-predict-then-act", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default=None, help="directory for metrics.json and checkpoint.pt")
    return p


def window_side_arrays(windows, **side: np.ndarray | None) -> dict[str, np.ndarray]:
    """Window-aligned ``(M, H, ...)`` side-modality arrays from per-step ``(N, ...)`` arrays."""
    h = windows.history
    flat = windows.last_index[:, None] - h + 1 + np.arange(h)[None, :]
    return {k: np.asarray(v[flat], dtype=np.float32) for k, v in side.items() if v is not None}


class _Scorer:
    """Bind ``extra`` side arrays so an object fits the ``predict_scores(windows)`` protocol."""

    def __init__(self, model, extra: dict[str, np.ndarray]):
        self.model, self.extra = model, extra

    def predict_scores(self, windows):
        return self.model.predict_scores(windows, extra=self.extra)


def _peak_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; the Phase-4 smoke run targets the RTX 4070")
    dev = torch.device("cuda")
    mods = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    penalties = [float(x) for x in args.switching_penalties.split(",")]

    # ---------------------------------------------------------------- data
    ds = load_scenario(args.data_root, args.scenario, max_samples=args.max_samples, keep_frame=True)
    print("[data] " + ds.summary().replace("\n", "\n       "))
    step_labels = label_regimes(ds.power, ds.segment_ids, **cfg.regimes.to_dict())
    print(f"[regimes] {regime_summary(step_labels)}")
    d = cfg.data
    windows = make_windows(ds.power, ds.segment_ids, history=d.history, horizon=d.horizon, stride=d.stride)
    train_segs, test_segs = split_segments(ds.segment_ids, d.test_fraction, seed=cfg.seed)
    train = windows.subset(np.isin(windows.segment_ids, train_segs))
    test = windows.subset(np.isin(windows.segment_ids, test_segs))
    test_regimes = regime_for_windows(step_labels, test.target_index)
    print(f"[windows] H={d.history} k={d.horizon}: {len(train)} train ({len(train_segs)} segs), "
          f"{len(test)} test ({len(test_segs)} segs); test regimes: {regime_summary(test_regimes)}")

    cache = ds.scenario_dir / "cache"
    side: dict[str, np.ndarray | None] = {}
    if "camera" in mods:
        side["camera_feat"] = load_camera_features(cache, ds.sample_index)
    if "lidar" in mods:
        side["lidar_tokens"], side["lidar_centroids"] = load_lidar_tokens(cache / "lidar", ds.sample_index)
    if "trajectory" in mods:
        side["trajectory"] = trajectory_windows(load_gps(ds).user_local_m, ds.segment_ids, EncoderConfig().trajectory_tokens)
    print(f"[modalities] {mods}; cached side arrays: " + ", ".join(f"{k} {v.shape}" for k, v in side.items()))
    test_extra = window_side_arrays(test, **side)

    # sequences for world-model / actor-critic training: train segments only
    seq_w = make_windows(ds.power, ds.segment_ids, history=args.wm_seq, horizon=1)
    seq_w = seq_w.subset(np.isin(seq_w.segment_ids, train_segs))
    rng = np.random.default_rng(cfg.seed)

    def sample_batch(n: int) -> dict[str, torch.Tensor]:
        idx = rng.choice(len(seq_w), n, replace=False)
        return make_sequence_batch(seq_w, idx, camera_feat=side.get("camera_feat"), lidar_tokens=side.get("lidar_tokens"),
                                   lidar_centroids=side.get("lidar_centroids"), trajectory=side.get("trajectory"), device=dev)

    # ---------------------------------------------------------- world model
    rcfg = RSSMConfig(deter_dim=args.deter_dim, action_dim=64 if args.action_conditioned else 0)
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim, modalities=mods)).to(dev)
    opt = torch.optim.AdamW([q for q in wm.parameters() if q.requires_grad], lr=args.wm_lr)
    print(f"[world model] {count_parameters(wm):,} trainable params; {args.wm_steps} steps of "
          f"B={args.wm_batch} x T={args.wm_seq} on {len(seq_w)} train sequences")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    wm_log = []
    for step in range(1, args.wm_steps + 1):
        wm.train()
        loss, m = wm.loss(sample_batch(args.wm_batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        if step % args.log_every == 0 or step == args.wm_steps:
            wm_log.append({"step": step, **m})
            print(f"  wm step {step:5d}  loss {m['loss']:.3f}  recon {m['recon']:.3f}  reward {m['reward']:.3f}  "
                  f"kl {m['kl']:.2f}  ({time.time() - t0:.0f}s)")
    wm_peak = _peak_mb()
    print(f"[world model] done in {time.time() - t0:.0f}s, peak GPU {wm_peak:.0f} MB")

    methods = {
        "reactive": ReactiveBaseline().fit(train),
        "markov": MarkovBaseline(n_beams=windows.n_beams).fit(train),
    }
    if not args.no_predict_then_act:
        pta_cfg = dict(cfg.get("predict_then_act", {}).to_dict())
        pta_cfg.setdefault("seed", int(cfg.seed))
        t1 = time.time()
        methods["predict-then-act"] = PredictThenActBaseline(**pta_cfg).fit(train)
        print(f"[predict-then-act] trained in {time.time() - t1:.0f}s: {methods['predict-then-act'].describe()[:80]}...")
    methods["wm-greedy"] = _Scorer(wm, test_extra)  # world model alone: argmax of the imagined reward table

    # --------------------------------------------------------- actor-critic
    ac_cfg = ActorCriticConfig(imagination_horizon=args.imagination_horizon, actor_lr=args.actor_lr,
                               critic_lr=args.critic_lr, entropy_scale=args.entropy_scale, batch_size=args.ac_batch)
    ac_logs, trainers = {}, {}
    for c in penalties:
        name = f"actor-critic c={c:g}"
        torch.manual_seed(cfg.seed)
        trainer = ImaginedActorCritic(wm, ac_cfg, switching_penalty=c)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        log = []
        print(f"[{name}] {args.ac_steps} updates, {args.ac_batch}x{args.wm_seq} rollout starts, horizon {ac_cfg.imagination_horizon}")
        for step in range(1, args.ac_steps + 1):
            m = trainer.train_step(sample_batch(args.ac_batch))
            if not all(np.isfinite(v) for v in m.values()):
                raise RuntimeError(f"non-finite actor-critic diagnostics at step {step}: {m}")
            if step % args.log_every == 0 or step == args.ac_steps:
                log.append({"step": step, **m})
                print(f"  ac step {step:5d}  actor {m['actor_loss']:+.4f}  critic {m['critic_loss']:.4f}  "
                      f"reward {m['reward_mean']:.2f}  regret {m['imagined_regret_db']:.2f} dB  "
                      f"switch {m['switch_rate']:.2f}  entropy {m['entropy']:.2f}  ({time.time() - t1:.0f}s)")
        peak = _peak_mb()
        print(f"[{name}] done in {time.time() - t1:.0f}s, peak GPU {peak:.0f} MB (world model + actor + critic)")
        ac_logs[name] = {"log": log, "peak_gpu_mb": peak, "seconds": time.time() - t1}
        trainers[name] = trainer
        methods[name] = _Scorer(trainer, test_extra)
    methods["oracle"] = OracleBaseline().fit(train)

    # ------------------------------------------------------------- evaluate
    scores = {name: np.asarray(m.predict_scores(test)) for name, m in methods.items()}
    results = {name: compute_metrics(s, test, test_regimes) for name, s in scores.items()}
    print("\n=== Regime-decomposed beam prediction metrics (held-out segments) ===")
    print(format_regime_table(results))

    rewards: dict[str, dict] = {}
    for c in penalties:
        env = OfflineBeamEnv(test, reward_unit=cfg.env.reward_unit, switching_penalty=c)
        opt_reward = env.reward_table().max(axis=1)
        print(f"\n=== Offline MDP net reward ({cfg.env.reward_unit}), switching penalty {c:g} dB ===")
        header = f"{'method':<20}{'regime':<12}{'n':>6}{'mean net':>10}{'regret':>9}{'switches':>10}{'switch %':>10}"
        print(header); print("-" * len(header))
        for name, s in scores.items():
            pr = env.evaluate_policy_reward(np.argmax(s, axis=1))
            row = {}
            for reg_name, mask in (("overall", np.ones(len(test), bool)), ("stable", test_regimes == 0), ("transition", test_regimes == 1)):
                if not mask.any():
                    continue
                n_sw = int(pr.switch_cost[mask].astype(bool).sum()) if c else int((np.argmax(s, 1)[mask] != test.last_beam()[mask]).sum())
                row[reg_name] = {"n": int(mask.sum()), "mean_net": float(pr.per_step[mask].mean()),
                                 "regret": float((opt_reward - pr.beam_reward)[mask].mean()), "n_switches": n_sw,
                                 "switch_rate": n_sw / int(mask.sum())}
                r = row[reg_name]
                print(f"{name if reg_name == 'overall' else '':<20}{reg_name:<12}{r['n']:>6}{r['mean_net']:>10.3f}"
                      f"{r['regret']:>9.3f}{r['n_switches']:>10d}{100 * r['switch_rate']:>9.1f}%")
            rewards[f"c={c:g}"] = rewards.get(f"c={c:g}", {})
            rewards[f"c={c:g}"][name] = row

    # world-model diagnostics: can the *posterior* state even reproduce the table it just observed?
    with torch.no_grad():
        cur = test.histories[:, -1]
        hist = torch.as_tensor(10 * np.log10(np.maximum(test.histories, 1e-12)), dtype=torch.float32, device=dev)
        batch = {"power_db": hist, **{k: torch.as_tensor(v, device=dev) for k, v in test_extra.items()}}
        post, _, _ = wm.observe(batch, deterministic=True)
        tab_post = wm.predict_reward_table(post[:, -1]).cpu().numpy()
    cur_db = 10 * np.log10(np.maximum(cur, 1e-12))
    post_regret = float(np.mean(cur_db.max(1) - cur_db[np.arange(len(cur)), tab_post.argmax(1)]))
    post_rmse = float(np.sqrt(np.mean((tab_post - cur_db) ** 2)))
    diag = {"posterior_table_regret_db": post_regret, "posterior_table_rmse_db": post_rmse,
            "distinct_beams": {name: int(len(np.unique(np.argmax(s, 1)))) for name, s in scores.items()},
            "distinct_optimal_beams": int(len(np.unique(test.target_beam)))}
    print("\n=== World-model / policy diagnostics (held-out windows) ===")
    print(f"posterior reward table at t vs the observed R[t]: regret {post_regret:.3f} dB, per-beam RMSE {post_rmse:.2f} dB "
          f"(0 = the latent encodes the current table exactly)")
    print("distinct beams chosen over the test set: " + ", ".join(f"{k} {v}" for k, v in diag["distinct_beams"].items())
          + f"; truly optimal beams {diag['distinct_optimal_beams']}")

    # early read on the central hypothesis: advantage over reactive, stable vs transition
    print("\n=== Power-loss advantage over the reactive baseline (dB, positive = better than reactive) ===")
    base = results["reactive"]
    print(f"{'method':<20}{'stable':>10}{'transition':>12}{'delta (tr - st)':>17}")
    for name, per in results.items():
        st = base["stable"].power_loss_db_mean - per["stable"].power_loss_db_mean
        tr = base["transition"].power_loss_db_mean - per["transition"].power_loss_db_mean
        print(f"{name:<20}{st:>10.3f}{tr:>12.3f}{tr - st:>17.3f}")

    out = {
        "args": vars(args), "config": cfg.to_dict(), "n_train_windows": len(train), "n_test_windows": len(test),
        "test_regimes": {"stable": int((test_regimes == 0).sum()), "transition": int((test_regimes == 1).sum())},
        "world_model": {"log": wm_log, "peak_gpu_mb": wm_peak, "n_parameters": count_parameters(wm)},
        "actor_critic": ac_logs,
        "metrics": {m: {r: v.as_dict() for r, v in per.items()} for m, per in results.items()},
        "rewards": rewards,
        "diagnostics": diag,
    }
    if args.out:
        od = Path(args.out)
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        torch.save({"world_model": wm.state_dict(), "rssm_config": rcfg.__dict__, "modalities": mods,
                    "actor_critic": {k: t.state_dict() for k, t in trainers.items()}}, od / "checkpoint.pt")
        print(f"\n[saved] {od / 'metrics.json'} and {od / 'checkpoint.pt'}")
    return out


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
