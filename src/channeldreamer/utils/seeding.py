"""Deterministic seeding for numpy / python / torch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed python, numpy and (if installed) torch. Returns a numpy Generator for local use.

    torch is imported lazily so that numpy-only pipelines (baselines, tests) do not pay the
    import cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency but keep numpy path alive
        pass
    return np.random.default_rng(seed)
