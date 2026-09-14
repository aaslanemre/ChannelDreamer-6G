"""Run the Phase-1 pipeline end to end and print regime-decomposed baseline tables.

Examples::

    python -m channeldreamer.scripts.train_baseline --synthetic
    python -m channeldreamer.scripts.train_baseline --data-root data --scenario 33
    python -m channeldreamer.scripts.train_baseline --synthetic --set data.history=16 --set data.horizon=3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import generate_synthetic, load_scenario, make_windows, split_segments
from ..envs import OfflineBeamEnv
from ..eval import evaluate_methods, format_regime_table, label_regimes, regime_for_windows
from ..eval.regimes import regime_summary
from ..models import MarkovBaseline, OracleBaseline, PredictThenActBaseline, ReactiveBaseline
from ..utils import load_config, seed_everything

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "phase1_baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    src.add_argument("--data-root", type=str, help="DeepSense root containing scenarioN/ folders")
    p.add_argument("--scenario", type=str, default="33", help="scenario number/name (with --data-root)")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                   help="override a config value, e.g. --set data.history=16")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--predict-then-act", action="store_true",
                   help="also train the transformer predict-then-act baseline (config key: predict_then_act)")
    p.add_argument("--verbose", action="store_true", help="print per-epoch losses")
    p.add_argument("--json-out", type=str, default=None, help="optional path to dump metrics as JSON")
    return p


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    if args.synthetic:
        syn = generate_synthetic(seed=cfg.seed, **cfg.synthetic.to_dict())
        power, seg = syn.power, syn.segment_ids
        print(f"[data] synthetic: {power.shape}, {len(np.unique(seg))} segments, "
              f"{int(syn.event_mask.sum())} injected transition events")
    else:
        ds = load_scenario(args.data_root, args.scenario, max_samples=args.max_samples)
        power, seg = ds.power, ds.segment_ids
        print("[data] " + ds.summary().replace("\n", "\n       "))

    reg_cfg = cfg.regimes.to_dict()
    step_labels = label_regimes(power, seg, **reg_cfg)
    print(f"[regimes] {regime_summary(step_labels)}  (cfg: {reg_cfg})")

    d = cfg.data
    windows = make_windows(power, seg, history=d.history, horizon=d.horizon, stride=d.stride)
    train_segs, test_segs = split_segments(seg, d.test_fraction, seed=cfg.seed)
    train = windows.subset(np.isin(windows.segment_ids, train_segs))
    test = windows.subset(np.isin(windows.segment_ids, test_segs))
    print(f"[windows] H={d.history} k={d.horizon}: {len(windows)} total, "
          f"{len(train)} train ({len(train_segs)} segs), {len(test)} test ({len(test_segs)} segs)")
    test_regimes = regime_for_windows(step_labels, test.target_index)

    methods = {
        "reactive": ReactiveBaseline().fit(train),
        "markov": MarkovBaseline(n_beams=windows.n_beams).fit(train),
    }
    if args.predict_then_act:
        pta_cfg = cfg.get("predict_then_act", {})
        pta_cfg = dict(pta_cfg.to_dict() if hasattr(pta_cfg, "to_dict") else pta_cfg)
        pta_cfg.setdefault("mixed_precision", bool(cfg.hardware.mixed_precision))
        pta_cfg.setdefault("device", str(cfg.hardware.device))
        pta_cfg.setdefault("seed", int(cfg.seed))
        print(f"[predict-then-act] training on {len(train)} windows ...")
        pta = PredictThenActBaseline(**pta_cfg).fit(train, verbose=args.verbose)
        print(f"[predict-then-act] {pta.describe()}")
        methods["predict-then-act"] = pta
    methods["oracle"] = OracleBaseline().fit(train)
    results = evaluate_methods(methods, test, test_regimes)
    print("\n=== Regime-decomposed beam prediction metrics (test segments) ===")
    print(format_regime_table(results))

    env = OfflineBeamEnv(test, reward_unit=cfg.env.reward_unit, switching_penalty=cfg.env.switching_penalty)
    print(f"\n=== Offline MDP net reward ({cfg.env.reward_unit}, switching penalty {cfg.env.switching_penalty}) ===")
    header = f"{'method':<18}{'regime':<12}{'mean net':>10}{'regret':>9}{'switches':>10}"
    print(header); print("-" * len(header))
    rewards = {}
    for name, m in methods.items():
        actions = np.argmax(m.predict_scores(test), axis=1)
        pr = env.evaluate_policy_reward(actions)
        rewards[name] = {"mean": pr.mean, "regret": pr.regret, "n_switches": pr.n_switches}
        print(f"{name:<18}{'overall':<12}{pr.mean:>10.3f}{pr.regret:>9.3f}{pr.n_switches:>10d}")
        for reg_id, reg_name in ((0, "stable"), (1, "transition")):
            mask = test_regimes == reg_id
            if mask.any():
                print(f"{'':<18}{reg_name:<12}{pr.per_step[mask].mean():>10.3f}"
                      f"{(env.reward_table().max(axis=1) - pr.beam_reward)[mask].mean():>9.3f}"
                      f"{int(pr.switch_cost[mask].astype(bool).sum()):>10d}")

    out = {
        "config": cfg.to_dict(),
        "n_test_windows": len(test),
        "predict_then_act": (
            {"peak_gpu_memory_bytes": methods["predict-then-act"].peak_memory_bytes,
             "n_parameters": methods["predict-then-act"].n_parameters(),
             "epochs": len(methods["predict-then-act"].history_log)}
            if "predict-then-act" in methods else None
        ),
        "metrics": {m: {r: v.as_dict() for r, v in per.items()} for m, per in results.items()},
        "rewards": rewards,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[saved] {args.json_out}")
    return out


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
