"""Does the world model overstate the gain from switching beams (optimizer's curse on a noisy table)?

    python -m channeldreamer.scripts.diag_model_optimism --wm-checkpoint runs/world_model/checkpoint.pt

On held-out windows, compares the model's *predicted* gain of its best beam over the currently
served beam at ``t+1`` with the *realized* gain of that same switch on the measured table, and with
the gain an oracle could obtain.  Writes ``results/stage3_model_optimism.json`` and
``figures/stage3_model_optimism.png``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..data import load_scenario, make_windows, split_segments
from ..data.modalities import load_camera_features, load_gps, load_lidar_tokens, trajectory_windows
from ..eval import label_regimes, regime_for_windows
from ..models.encoders import EncoderConfig
from ..models.world_model import RSSMConfig, WorldModel
from ..utils import load_config, save_figure, save_results
from .train_actor_critic import DEFAULT_CONFIG, window_side_arrays


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--scenario", default="33")
    p.add_argument("--wm-checkpoint", default="runs/world_model/checkpoint.pt")
    p.add_argument("--penalties", default="0.25,0.5,1.0")
    p.add_argument("--experiment", default="stage3_model_optimism")
    args = p.parse_args(argv)
    cfg = load_config(str(DEFAULT_CONFIG), [])
    penalties = [float(x) for x in args.penalties.split(",")]
    ds = load_scenario(args.data_root, args.scenario, keep_frame=True)
    labels = label_regimes(ds.power, ds.segment_ids, **cfg.regimes.to_dict())
    d = cfg.data
    w = make_windows(ds.power, ds.segment_ids, history=d.history, horizon=d.horizon, stride=d.stride)
    train_segs, test_segs = split_segments(ds.segment_ids, d.test_fraction, seed=cfg.seed)
    val_segs = np.sort(np.random.default_rng(cfg.seed).choice(train_segs, 2, replace=False))
    cache = ds.scenario_dir / "cache"
    ck = torch.load(args.wm_checkpoint, map_location="cuda", weights_only=False)
    mods = tuple(ck["modalities"])
    side = {}
    if "camera" in mods:
        side["camera_feat"] = load_camera_features(cache, ds.sample_index)
    if "lidar" in mods:
        side["lidar_tokens"], side["lidar_centroids"] = load_lidar_tokens(cache / "lidar", ds.sample_index)
    if "trajectory" in mods:
        side["trajectory"] = trajectory_windows(load_gps(ds).user_local_m, ds.segment_ids, EncoderConfig().trajectory_tokens)
    rcfg = RSSMConfig(**ck["rssm_config"])
    wm = WorldModel(rcfg, EncoderConfig(embed_dim=rcfg.embed_dim, modalities=mods)).cuda()
    wm.load_state_dict(ck["world_model"])
    wm.eval()

    out = {"world_model_checkpoint": args.wm_checkpoint, "world_model_step": ck.get("step"), "penalties": penalties, "splits": {}}
    for name, segs in (("validation", val_segs), ("test", test_segs)):
        win = w.subset(np.isin(w.segment_ids, segs))
        reg = regime_for_windows(labels, win.target_index)
        table = wm.predict_scores(win, extra=window_side_arrays(win, **side))
        cur, a, n = win.last_beam(), table.argmax(1), np.arange(len(win))
        real = 10 * np.log10(np.maximum(win.target_power, 1e-12))
        pred_gain, real_gain, oracle_gain = table[n, a] - table[n, cur], real[n, a] - real[n, cur], real.max(1) - real[n, cur]
        rec = {"segments": segs.tolist(), "n_windows": len(win), "model_beam_differs_from_current_pct": float(100 * np.mean(a != cur)), "by_regime": {}}
        for r, rname in ((None, "overall"), (0, "stable"), (1, "transition")):
            m = np.ones(len(win), bool) if r is None else reg == r
            rec["by_regime"][rname] = {
                "n": int(m.sum()), "predicted_gain_db": float(pred_gain[m].mean()), "realized_gain_db": float(real_gain[m].mean()),
                "oracle_gain_db": float(oracle_gain[m].mean()),
                **{f"pct_predicted_gain_over_{c:g}": float(100 * np.mean(pred_gain[m] > c)) for c in penalties},
                **{f"pct_realized_gain_over_{c:g}": float(100 * np.mean(real_gain[m] > c)) for c in penalties},
                **{f"pct_oracle_gain_over_{c:g}": float(100 * np.mean(oracle_gain[m] > c)) for c in penalties}}
            b = rec["by_regime"][rname]
            print(f"[{name}] {rname:<10} n={b['n']:4d}  gain of the model's beam over the current beam: predicted {b['predicted_gain_db']:.3f} dB, "
                  f"realized {b['realized_gain_db']:.3f} dB (oracle {b['oracle_gain_db']:.3f}); predicted > 0.5 dB on "
                  f"{b.get('pct_predicted_gain_over_0.5', float('nan')):.0f}% of windows, realized > 0.5 dB on {b.get('pct_realized_gain_over_0.5', float('nan')):.0f}%")
        out["splits"][name] = rec
    save_results(args.experiment, {"description": "Predicted vs realized gain of switching to the world model's best beam (optimizer's curse diagnostic)", **out})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    regs = ("stable", "transition")
    x = np.arange(len(regs))
    t = out["splits"]["test"]["by_regime"]
    for i, (key, lab, col) in enumerate((("predicted_gain_db", "predicted by the model", "#1f77b4"), ("realized_gain_db", "realized (same switch)", "#d62728"),
                                         ("oracle_gain_db", "oracle (best possible)", "#7f7f7f"))):
        vals = [t[r][key] for r in regs]
        bars = ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=lab, color=col)
        for b_, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b_.get_x() + b_.get_width() / 2, b_.get_height()), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(regs)
    ax.set_ylabel("gain of switching from the current beam (dB)")
    ax.set_title("Held-out test: gain of the world model's best beam over the current beam")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(args.experiment, fig)
    print(f"[results] results/{args.experiment}.json, figures/{args.experiment}.png")
    return out


if __name__ == "__main__":
    main()
