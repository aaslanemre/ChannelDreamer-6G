"""Evaluate a trained checkpoint (world model / actor) with regime-decomposed metrics.  **Thin stub.**

    python -m channeldreamer.scripts.evaluate --checkpoint runs/xyz/model.pt --data-root data --scenario 33

Once Phase 3/4 exist, this will load the checkpoint, build windows with the *same* shared
windowing as the baselines, and call :func:`channeldreamer.eval.evaluate_methods` on
``{"reactive", "markov", "<checkpoint>"}``.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--scenario", type=str, default="33")
    p.add_argument("--config", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"checkpoint evaluation is not implemented yet (Phase 3/4). Requested: {args.checkpoint}")
    print("Use `python -m channeldreamer.scripts.train_baseline` for Phase-1 baselines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
