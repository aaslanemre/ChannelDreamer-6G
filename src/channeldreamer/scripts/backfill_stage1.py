"""Backfill persistent records for Stage 0 (Phase-1 baselines) and Stage 1 (world-model training).

    python -m channeldreamer.scripts.backfill_stage1 --wm-run runs/world_model

Reads the Stage-1 training log (``runs/world_model/train_log.json``) for the regret curve and
re-evaluates the baselines and the saved world-model checkpoint on the held-out test split, then
writes ``results/stage0_baselines.json``, ``results/stage1_world_model_regret.json`` (+ figure),
``results/stage1_final_comparison.json`` (+ figure).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import load_scenario, make_windows, split_segments
from ..data.modalities import load_camera_features, load_gps, load_lidar_tokens, trajectory_windows
from ..eval import compute_metrics, label_regimes, regime_for_windows
from ..models import MarkovBaseline, OracleBaseline, PredictThenActBaseline, ReactiveBaseline
from ..models.encoders import EncoderConfig
from ..models.world_model import RSSMConfig, WorldModel
from ..utils import load_config, plot_regime_bars, plot_regret_curve, save_figure, save_results, seed_everything
from .train_actor_critic import DEFAULT_CONFIG, window_side_arrays


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--wm-run", default="runs/world_model")
    args = p.parse_args(argv)
    cfg = load_config(str(DEFAULT_CONFIG), [])
    seed_everything(cfg.seed)
    run = Path(args.wm_run)
    train_log = json.load(open(run / "train_log.json"))

    ds = load_scenario(args.data_root, args.scenario, keep_frame=True)
    labels = label_regimes(ds.power, ds.segment_ids, **cfg.regimes.to_dict())
    d = cfg.data
    w = make_windows(ds.power, ds.segment_ids, history=d.history, horizon=d.horizon, stride=d.stride)
    tr_segs, te_segs = split_segments(ds.segment_ids, d.test_fraction, seed=cfg.seed)
    train, test = w.subset(np.isin(w.segment_ids, tr_segs)), w.subset(np.isin(w.segment_ids, te_segs))
    reg = regime_for_windows(labels, test.target_index)
    split = {"history": d.history, "horizon": d.horizon, "test_fraction": d.test_fraction, "seed": cfg.seed,
             "train_segments": tr_segs.tolist(), "test_segments": te_segs.tolist(), "n_test_windows": len(test),
             "n_test_transition": int((reg == 1).sum()), "val_segments": train_log["val_segments"]}

    # ---- Stage 0: Phase-1 baselines on the test split
    pta = PredictThenActBaseline(**dict(cfg.predict_then_act.to_dict()), seed=int(cfg.seed)).fit(train)
    methods = {"reactive": ReactiveBaseline().fit(train), "markov": MarkovBaseline().fit(train), "predict-then-act": pta,
               "oracle": OracleBaseline().fit(train)}
    scores = {n: np.asarray(m.predict_scores(test)) for n, m in methods.items()}
    metrics = {n: {r: v.as_dict() for r, v in compute_metrics(s, test, reg).items()} for n, s in scores.items()}
    save_results("stage0_baselines", {"description": "Phase-1 baselines, regime-decomposed, held-out test segments of Scenario 33",
                                      "split": split, "predict_then_act_config": cfg.predict_then_act.to_dict(), "metrics": metrics,
                                      "distinct_beams": {n: int(len(np.unique(s.argmax(1)))) for n, s in scores.items()}})

    # ---- Stage 1: regret curve from the training log
    curve = [{"step": e["step"], "seconds": e["seconds"], "loss": e["loss"]["loss"], "recon": e["loss"]["recon"], "kl": e["loss"]["kl"],
              "val_stable": e["val"]["stable"], "val_transition": e["val"]["transition"], "val_overall": e["val"]["overall"],
              "stable": e["test"]["stable"], "transition": e["test"]["transition"], "overall": e["test"]["overall"]} for e in train_log["log"]]
    save_results("stage1_world_model_regret", {
        "description": "Exogenous world model, greedy one-step regret vs training step; 'stable'/'transition' are the held-out TEST "
                       "segments, 'val_*' the validation segments used for the stopping rule",
        "checkpoint": str(run / "checkpoint.pt"), "split": split, "hyperparameters": train_log["args"],
        "reactive": train_log["reactive"], "stop_reason": train_log["stop_reason"], "best_step": train_log["best_step"],
        "peak_gpu_mb": train_log["peak_gpu_mb"], "curve": curve})
    fig = plot_regret_curve(curve, train_log["reactive"]["test"], "Stage 1: world-model greedy one-step regret (held-out test)",
                            xlabel="world-model training step")
    save_figure("stage1_world_model_regret", fig)
    val_curve = [{"step": c["step"], "stable": c["val_stable"], "transition": c["val_transition"]} for c in curve]
    save_figure("stage1_world_model_regret_val", plot_regret_curve(val_curve, train_log["reactive"]["val"],
                "Stage 1: world-model greedy one-step regret (validation, stopping rule)", xlabel="world-model training step"))

    # ---- Stage 1: final comparison with the saved checkpoint
    ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=False)
    rcfg = RSSMConfig(**ck["rssm_config"])
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim, modalities=ck["modalities"])).cuda()
    wm.load_state_dict(ck["world_model"])
    wm.eval()
    cache = ds.scenario_dir / "cache"
    side = {"camera_feat": load_camera_features(cache, ds.sample_index)}
    side["lidar_tokens"], side["lidar_centroids"] = load_lidar_tokens(cache / "lidar", ds.sample_index)
    side["trajectory"] = trajectory_windows(load_gps(ds).user_local_m, ds.segment_ids, 16)
    scores["wm-greedy"] = wm.predict_scores(test, extra=window_side_arrays(test, **side))
    final = {n: {r: v.as_dict() for r, v in compute_metrics(scores[n], test, reg).items()} for n in ("reactive", "predict-then-act", "wm-greedy")}
    table = {n: {r: final[n][r]["power_loss_db_mean"] for r in ("stable", "transition", "overall")} for n in final}
    save_results("stage1_final_comparison", {
        "description": "Held-out test segments: reactive vs predict-then-act vs world-model greedy one-step (power loss dB by regime)",
        "checkpoint": str(run / "checkpoint.pt"), "world_model_step": ck["step"], "split": split, "metrics": final, "power_loss_db": table,
        "distinct_beams": {n: int(len(np.unique(scores[n].argmax(1)))) for n in final}, "distinct_optimal_beams": int(len(np.unique(test.target_beam)))})
    save_figure("stage1_final_comparison", plot_regime_bars(table, "Stage 1: held-out power loss by regime"))
    print("saved:", *sorted(str(p) for p in list(Path("results").glob("stage[01]*")) + list(Path("figures").glob("stage1*"))), sep="\n  ")


if __name__ == "__main__":
    main()
