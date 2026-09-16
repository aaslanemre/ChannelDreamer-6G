"""Phase-3 closure: train the exogenous world model until its greedy one-step prediction is as
good as the reactive baseline on held-out segments, on both regimes.

    python -m channeldreamer.scripts.train_world_model --data-root data --scenario 33 \
        --max-steps 60000 --eval-every 1000 --out runs/world_model

Stopping rule (decided on *validation* segments split off the training segments, never on
the test segments): stop when the greedy one-step regret is <= the reactive regret on both
the stable and the transition windows for ``--confirm`` consecutive evaluations, or when the
overall validation regret has not improved for ``--patience`` evaluations, or at
``--max-steps``.  Test-segment numbers are logged at every evaluation for reporting only; the
checkpoint kept is the one with the best validation regret.
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
from ..eval import compute_metrics, label_regimes, regime_for_windows
from ..models import ReactiveBaseline
from ..models.encoders import EncoderConfig, count_parameters
from ..models.world_model import RSSMConfig, WorldModel, make_sequence_batch
from ..utils import load_config, seed_everything
from .train_actor_critic import DEFAULT_CONFIG, window_side_arrays


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    p.add_argument("--modalities", default="power,camera,lidar,trajectory")
    p.add_argument("--val-segments", type=int, default=2, help="training segments held out for the stopping rule")
    p.add_argument("--max-steps", type=int, default=60000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--confirm", type=int, default=2, help="consecutive evals meeting the target before stopping")
    p.add_argument("--patience", type=int, default=10, help="evals without validation improvement before stopping")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--deter-dim", type=int, default=256)
    p.add_argument("--action-conditioned", action="store_true", help="ablation only (see rl.actor_critic)")
    p.add_argument("--out", default="runs/world_model")
    return p


def regime_regret(scores: np.ndarray, windows, regimes: np.ndarray) -> dict[str, float]:
    m = compute_metrics(scores, windows, regimes)
    return {k: float(v.power_loss_db_mean) for k, v in m.items()}


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda")
    mods = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_scenario(args.data_root, args.scenario, keep_frame=True)
    step_labels = label_regimes(ds.power, ds.segment_ids, **cfg.regimes.to_dict())
    d = cfg.data
    windows = make_windows(ds.power, ds.segment_ids, history=d.history, horizon=d.horizon, stride=d.stride)
    train_segs, test_segs = split_segments(ds.segment_ids, d.test_fraction, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    val_segs = np.sort(rng.choice(train_segs, args.val_segments, replace=False))
    fit_segs = np.setdiff1d(train_segs, val_segs)
    splits = {name: windows.subset(np.isin(windows.segment_ids, segs)) for name, segs in
              (("val", val_segs), ("test", test_segs))}
    regimes = {name: regime_for_windows(step_labels, w.target_index) for name, w in splits.items()}
    print(f"[windows] H={d.history} k={d.horizon}: fit segs {fit_segs.tolist()}, val segs {val_segs.tolist()}, "
          f"test segs {test_segs.tolist()}; val {len(splits['val'])} windows "
          f"({int((regimes['val'] == 1).sum())} transition), test {len(splits['test'])} windows "
          f"({int((regimes['test'] == 1).sum())} transition)")

    cache = ds.scenario_dir / "cache"
    side: dict[str, np.ndarray] = {}
    if "camera" in mods:
        side["camera_feat"] = load_camera_features(cache, ds.sample_index)
    if "lidar" in mods:
        side["lidar_tokens"], side["lidar_centroids"] = load_lidar_tokens(cache / "lidar", ds.sample_index)
    if "trajectory" in mods:
        side["trajectory"] = trajectory_windows(load_gps(ds).user_local_m, ds.segment_ids, EncoderConfig().trajectory_tokens)
    extras = {name: window_side_arrays(w, **side) for name, w in splits.items()}
    reactive = {name: regime_regret(ReactiveBaseline().predict_scores(w), w, regimes[name]) for name, w in splits.items()}
    for name in splits:
        print(f"[reactive {name}] regret stable {reactive[name]['stable']:.3f} / transition {reactive[name]['transition']:.3f} "
              f"/ overall {reactive[name]['overall']:.3f} dB")

    seq_w = make_windows(ds.power, ds.segment_ids, history=args.seq, horizon=1)
    seq_w = seq_w.subset(np.isin(seq_w.segment_ids, fit_segs))
    rcfg = RSSMConfig(deter_dim=args.deter_dim, action_dim=64 if args.action_conditioned else 0)
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim, modalities=mods)).to(dev)
    opt = torch.optim.AdamW([q for q in wm.parameters() if q.requires_grad], lr=args.lr)
    print(f"[world model] {count_parameters(wm):,} trainable params, exogenous={not args.action_conditioned}; "
          f"{len(seq_w)} fit sequences of T={args.seq}, B={args.batch}, lr {args.lr}")

    def evaluate(name: str) -> dict[str, float]:
        wm.eval()
        r = regime_regret(wm.predict_scores(splits[name], extra=extras[name]), splits[name], regimes[name])
        wm.train()
        return r

    log, best_val, best_step, since_best, met, t0 = [], float("inf"), 0, 0, 0, time.time()
    run_loss = {}
    stop_reason = "max_steps"
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.max_steps + 1):
        wm.train()
        idx = rng.choice(len(seq_w), args.batch, replace=False)
        batch = make_sequence_batch(seq_w, idx, camera_feat=side.get("camera_feat"), lidar_tokens=side.get("lidar_tokens"),
                                    lidar_centroids=side.get("lidar_centroids"), trajectory=side.get("trajectory"), device=dev)
        loss, m = wm.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        for k, v in m.items():
            run_loss[k] = run_loss.get(k, 0.0) + v / args.eval_every
        if step % args.eval_every == 0:
            val, test = evaluate("val"), evaluate("test")
            entry = {"step": step, "seconds": time.time() - t0, "loss": run_loss, "val": val, "test": test}
            log.append(entry)
            run_loss = {}
            target = val["stable"] <= reactive["val"]["stable"] and val["transition"] <= reactive["val"]["transition"]
            met = met + 1 if target else 0
            improved = val["overall"] < best_val - 1e-4
            if improved:
                best_val, best_step, since_best = val["overall"], step, 0
                torch.save({"world_model": wm.state_dict(), "rssm_config": rcfg.__dict__, "modalities": mods,
                            "step": step, "val": val, "test": test}, out / "checkpoint.pt")
            else:
                since_best += 1
            print(f"step {step:6d} ({entry['seconds']:5.0f}s) loss {entry['loss']['loss']:.3f} recon {entry['loss']['recon']:.3f} "
                  f"kl {entry['loss']['kl']:.2f} | val regret st {val['stable']:.3f} tr {val['transition']:.3f} "
                  f"(reactive {reactive['val']['stable']:.3f}/{reactive['val']['transition']:.3f}) | "
                  f"test st {test['stable']:.3f} tr {test['transition']:.3f} (reactive {reactive['test']['stable']:.3f}/"
                  f"{reactive['test']['transition']:.3f}){' *' if improved else ''}{' TARGET' if target else ''}", flush=True)
            if met >= args.confirm:
                stop_reason = f"target met on validation for {args.confirm} consecutive evaluations"
                break
            if since_best >= args.patience:
                stop_reason = f"validation plateau: no improvement for {args.patience} evaluations"
                break
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[done] {stop_reason}; best validation overall regret {best_val:.3f} dB at step {best_step}; peak GPU {peak:.0f} MB")
    summary = {"args": vars(args), "reactive": reactive, "log": log, "best_step": best_step, "best_val_overall": best_val,
               "stop_reason": stop_reason, "peak_gpu_mb": peak, "fit_segments": fit_segs.tolist(),
               "val_segments": val_segs.tolist(), "test_segments": test_segs.tolist()}
    with open(out / "train_log.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[saved] {out / 'checkpoint.pt'} (best validation) and {out / 'train_log.json'}")
    return summary


if __name__ == "__main__":
    main()
