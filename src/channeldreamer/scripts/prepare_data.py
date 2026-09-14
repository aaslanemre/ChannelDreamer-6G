"""Validate that a real DeepSense 6G scenario loads correctly.

    python -m channeldreamer.scripts.prepare_data --data-root data --scenario 33

Prints the resolved scenario folder / CSV, the regex-resolved columns, the power-vector
shape, label cross-validation results and segment statistics.  On failure the actual
exception and the CSV header are shown rather than guessed at.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.deepsense import find_csv, load_scenario, resolve_columns, resolve_scenario_dir
from ..eval.regimes import label_regimes, regime_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--time-gap-s", type=float, default=None,
                   help="also break segments at timestamp gaps larger than this (seconds)")
    p.add_argument("--check-files", action="store_true",
                   help="verify that referenced camera/lidar/radar files exist on disk")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sdir = resolve_scenario_dir(args.data_root, args.scenario)
        csv_path = find_csv(sdir)
    except Exception:  # noqa: BLE001 - show the real error
        traceback.print_exc()
        return 1
    print(f"scenario folder : {sdir}")
    print(f"csv             : {csv_path}")
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    print(f"csv header      : {header}")
    resolved = resolve_columns(header)
    print("resolved columns:")
    for k, v in resolved.items():
        print(f"    {k:<12} -> {v}")
    if resolved["power"] is None:
        print("ERROR: could not resolve the beam-power column from the header above", file=sys.stderr)
        return 2

    try:
        ds = load_scenario(args.data_root, args.scenario, max_samples=args.max_samples,
                           time_gap_s=args.time_gap_s)
    except Exception:  # noqa: BLE001
        print("ERROR while loading power vectors:", file=sys.stderr)
        traceback.print_exc()
        return 3

    print("\n" + ds.summary())
    beams = ds.optimal_beam()
    print(f"optimal beam    : range [{beams.min()}, {beams.max()}], "
          f"{len(np.unique(beams))} distinct beams used")
    labels = label_regimes(ds.power, ds.segment_ids)
    print(f"regimes (default thresholds): {regime_summary(labels)}")
    print("modalities      : " + ", ".join(f"{k} ({len(v)} refs)" for k, v in ds.modality_paths.items()))

    if args.check_files:
        for k, paths in ds.modality_paths.items():
            missing = sum(1 for p in paths if not Path(p).exists())
            print(f"    {k:<20} missing files: {missing}/{len(paths)}")
    print("\nOK: scenario loads correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
