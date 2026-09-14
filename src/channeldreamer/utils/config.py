"""Minimal YAML config with dotted access and CLI overrides.

Usage::

    cfg = load_config("configs/phase1_baseline.yaml", overrides=["data.history=16"])
    cfg.data.history          # -> 16
    cfg["regimes"]["beam_jump_threshold"]
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict whose string keys are also attributes; nested dicts are wrapped recursively."""

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any):
        super().__init__()
        merged = dict(data or {})
        merged.update(kwargs)
        for k, v in merged.items():
            self[k] = v

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}

    def merged(self, other: Mapping[str, Any]) -> Config:
        out = copy.deepcopy(self)
        _deep_update(out, other)
        return out


def _deep_update(base: Config, other: Mapping[str, Any]) -> None:
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Config):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules ('16' -> 16, 'true' -> True)."""
    return yaml.safe_load(text)


def apply_overrides(cfg: Config, overrides: Iterable[str] | None) -> Config:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), _parse_scalar(value.strip()))
    return cfg


def load_config(path: str | Path | None, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML file (or defaults when path is None) and apply ``key=value`` overrides."""
    cfg = Config(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        cfg = cfg.merged(loaded)
    return apply_overrides(cfg, overrides)


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "data": {
        "history": 8,
        "horizon": 1,
        "stride": 1,
        "test_fraction": 0.25,
        "power_unit": "linear",
    },
    "synthetic": {
        "n_segments": 12,
        "segment_length": 400,
        "drift_std": 0.15,
        "transition_prob": 0.01,
        "jump_min": 6,
        "jump_max": 20,
        "dip_db": 8.0,
        "beamwidth": 2.0,
        "noise_std": 0.02,
    },
    "regimes": {
        "beam_jump_threshold": 3,
        "power_drop_db_threshold": 3.0,
        "smoothing_window": 3,
        "drop_lookback": 3,
        "dilation": 1,
    },
    "env": {
        "reward_unit": "db",
        "switching_penalty": 0.5,
    },
    "hardware": {
        "device": "auto",
        "mixed_precision": True,
        "batch_size": 32,
        "max_vram_gb": 12,
    },
}
