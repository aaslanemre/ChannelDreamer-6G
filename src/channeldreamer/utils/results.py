"""Persistent experiment records: ``results/<experiment>.json|csv`` and ``figures/<experiment>.png``.

Deliberately small.  Every record carries a timestamp and the git commit so a number in the
paper can be traced back to the code that produced it.

    from channeldreamer.utils.results import save_results, save_figure, plot_regret_curve
    save_results("stage1_world_model_regret", {"curve": [...], "reactive": {...}})
    save_figure("stage1_world_model_regret", plot_regret_curve(curve, reactive, "..."))
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def git_commit(short: bool = True) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short" if short else "HEAD", "HEAD"] if short else ["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip() != ""
        return out + ("-dirty" if dirty else "")
    except Exception:  # not a git checkout
        return "unknown"


def _stamp(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(), **(extra or {})}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "__dict__") and not isinstance(x, type):
        return _jsonable(vars(x))
    return x


def save_results(experiment: str, data: dict[str, Any], results_dir: Path | str = RESULTS_DIR) -> Path:
    """Write ``results/<experiment>.json`` (timestamp + commit + ``data``)."""
    path = Path(results_dir) / f"{experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_stamp(_jsonable(data)), fh, indent=2)
    return path


def append_results_row(experiment: str, row: dict[str, Any], results_dir: Path | str = RESULTS_DIR) -> Path:
    """Append one flat row (timestamp + commit + ``row``) to ``results/<experiment>.csv``."""
    path = Path(results_dir) / f"{experiment}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _stamp(row)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    return path


def save_figure(experiment: str, fig, figures_dir: Path | str = FIGURES_DIR, dpi: int = 150) -> Path:
    path = Path(figures_dir) / f"{experiment}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_regret_curve(curve: list[dict[str, float]], reference: dict[str, float] | None = None, title: str = "",
                      ylabel: str = "regret (dB)", xlabel: str = "step", reference_label: str = "reactive", logy: bool = False):
    """Line plot of ``stable`` / ``transition`` vs ``step`` with dashed horizontal reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    steps = [c["step"] for c in curve]
    colors = {"stable": "#1f77b4", "transition": "#d62728"}
    for key in ("stable", "transition"):
        ax.plot(steps, [c[key] for c in curve], marker="o", ms=3, color=colors[key], label=key)
        if reference and key in reference:
            ax.axhline(reference[key], color=colors[key], ls="--", lw=1, alpha=0.8, label=f"{reference_label} ({key})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_regime_bars(table: dict[str, dict[str, float]], title: str = "", ylabel: str = "power loss (dB)",
                     regimes: tuple[str, ...] = ("stable", "transition")):
    """Grouped bars: one group per method, one bar per regime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = list(table.keys())
    x = np.arange(len(methods))
    width = 0.8 / len(regimes)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"stable": "#1f77b4", "transition": "#d62728", "overall": "#7f7f7f"}
    for i, reg in enumerate(regimes):
        vals = [table[m][reg] for m in methods]
        bars = ax.bar(x + (i - (len(regimes) - 1) / 2) * width, vals, width, label=reg, color=colors.get(reg))
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


__all__ = ["append_results_row", "git_commit", "plot_regime_bars", "plot_regret_curve", "save_figure", "save_results"]
