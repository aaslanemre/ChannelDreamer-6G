"""Stage 3: switching-cost x imagination-horizon sweep of the actor-critic on the frozen world model.

    python -m channeldreamer.scripts.sweep_stage3 --wm-checkpoint runs/world_model/checkpoint.pt \
        --penalties 0,0.25,0.5,1.0 --horizons 3,5,10 --out runs/stage3

Each cell runs ``train_policy`` (plateau-based stopping, Stage-2 run-2 actor hyper-parameters
by default) as a subprocess and records ``results/stage3_c{c}_h{H}_*.json``; the driver then
aggregates the held-out tables into ``results/stage3_sweep.json`` / ``.csv`` and
``figures/stage3_sweep.png``.  Cells whose records already exist are skipped (resumable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..utils import append_results_row, save_figure, save_results


def cell_name(c: float, h: int) -> str:
    return f"stage3_c{c:g}_h{h}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm-checkpoint", default="runs/world_model/checkpoint.pt")
    p.add_argument("--penalties", default="0,0.25,0.5,1.0")
    p.add_argument("--horizons", default="3,5,10")
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--entropy-scale", type=float, default=3e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--max-updates", type=int, default=30000)
    p.add_argument("--out", default="runs/stage3")
    p.add_argument("--aggregate-only", action="store_true")
    return p


def aggregate(penalties: list[float], horizons: list[int], out_name: str = "stage3_sweep") -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows, cells = [], {}
    for c in penalties:
        for h in horizons:
            name = cell_name(c, h)
            fc, ir = Path("results") / f"{name}_final_comparison.json", Path("results") / f"{name}_imagined_regret.json"
            if not ir.exists():
                continue
            reg = json.load(open(ir))
            row = {"penalty_db": c, "horizon": h, "stop_reason": reg["stop_reason"], "best_step": reg["best_step"],
                   "val_greedy_imagined_regret_db": reg["checks"]["val_imagined"]["greedy_regret_db"],
                   "val_distinct_beams": reg["checks"]["val_distinct_beams"], "checks_passed": reg["checks_passed"]}
            if fc.exists():
                fin = json.load(open(fc))
                ac = [k for k in fin["power_loss_db"] if k.startswith("actor-critic")][0]
                for m, key in (("actor-critic", ac), ("wm-greedy", "wm-greedy"), ("reactive", "reactive"), ("predict-then-act", "predict-then-act")):
                    for r in ("stable", "transition", "overall"):
                        row[f"{m}_loss_{r}"] = fin["power_loss_db"][key][r]
                    rew = fin["mdp_rewards"].get(f"c={c:g}", {}).get(key, {})
                    row[f"{m}_net_reward_at_c"] = rew.get("mean")
                    row[f"{m}_switches"] = rew.get("n_switches")
                row["actor-critic_distinct_beams_test"] = fin["distinct_beams_test"].get(ac)
                row["n_test_windows"] = fin["split"]["n_test_windows"]
            rows.append(row)
            cells[name] = row
    if not rows:
        raise SystemExit("no stage3 records found")
    save_results(out_name, {"description": "Stage 3 sweep: switching penalty x imagination horizon; held-out power loss (dB) by regime, "
                            "MDP net reward at the trained penalty, switches, collapse checks", "penalties": penalties,
                            "horizons": horizons, "cells": cells})
    csv_path = Path("results") / f"{out_name}.csv"
    if csv_path.exists():
        csv_path.unlink()
    for row in rows:
        append_results_row(out_name, row)

    # figure: transition / stable loss vs penalty, one line per horizon, with baselines as dashed lines
    done = [r for r in rows if "actor-critic_loss_transition" in r]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in ((axes[0], "loss_transition", "held-out power loss, TRANSITION (dB)"),
                           (axes[1], "loss_stable", "held-out power loss, STABLE (dB)"),
                           (axes[2], "net_reward_at_c", "held-out MDP net reward at the trained c (dB)")):
        for h in horizons:
            pts = sorted((r["penalty_db"], r[f"actor-critic_{key}"]) for r in done if r["horizon"] == h)
            if pts:
                ax.plot([p_ for p_, _ in pts], [v for _, v in pts], marker="o", label=f"actor-critic H={h}")
        for m, ls in (("wm-greedy", "--"), ("reactive", ":"), ("predict-then-act", "-.")):
            pts = sorted((r["penalty_db"], r[f"{m}_{key}"]) for r in done if r["horizon"] == horizons[0])
            if pts:
                ax.plot([p_ for p_, _ in pts], [v for _, v in pts], ls=ls, color="k", alpha=0.6, label=m)
        ax.set_xlabel("switching penalty c (dB)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(out_name, fig)
    print(f"[aggregate] {len(rows)} cells -> results/{out_name}.json, results/{out_name}.csv, figures/{out_name}.png")
    return cells


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    penalties = [float(x) for x in args.penalties.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    if not args.aggregate_only:
        # curriculum on the penalty: the c=0 cell of each horizon is trained cold and every c>0 cell of that
        # horizon is warm-started from it (a cold start with c>0 collapses to a handful of beams)
        order = sorted(penalties)
        if order[0] != 0.0:
            raise SystemExit("the sweep needs c=0 in --penalties (warm start for the other cells)")
        for h in horizons:
            for c in order:
                name = cell_name(c, h)
                if (Path("results") / f"{name}_imagined_regret.json").exists():
                    print(f"[skip] {name} already recorded", flush=True)
                    continue
                cmd = [sys.executable, "-m", "channeldreamer.scripts.train_policy", "--wm-checkpoint", args.wm_checkpoint,
                       "--switching-penalty", str(c), "--imagination-horizon", str(h), "--stop-on", "plateau",
                       "--patience", str(args.patience), "--eval-every", str(args.eval_every), "--max-updates", str(args.max_updates),
                       "--actor-lr", str(args.actor_lr), "--critic-lr", str(args.critic_lr), "--entropy-scale", str(args.entropy_scale),
                       "--out", str(Path(args.out) / name), "--experiment", name]
                if c > 0:
                    cmd += ["--init-actor", str(Path(args.out) / cell_name(0.0, h) / "policy.pt")]
                print(f"[run] {name}: {' '.join(cmd[2:])}", flush=True)
                log = Path(args.out) / f"{name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, "w") as fh:
                    rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
                tail = [ln for ln in open(log) if ln.startswith(("[done]", "distinct beams chosen", "checks ", "actor-critic c="))]
                print(f"[{name}] exit {rc}\n" + "".join("    " + ln for ln in tail), flush=True)
    aggregate(penalties, horizons)


if __name__ == "__main__":
    main()
