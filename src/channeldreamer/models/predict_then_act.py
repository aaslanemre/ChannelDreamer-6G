"""Predict-then-act baseline: a small transformer forecaster + greedy controller.

This is the most important baseline in the paper.  It isolates *better prediction* from
*better decision making*: a sequence model forecasts the beam distribution at ``t + k`` from the
``(H, 64)`` history, and the controller greedily picks the argmax.  If a world-model policy
(Phase 4) only matches this baseline at transitions, the decision mechanism adds nothing
beyond forecasting, and that is the negative result to report.

Architecture
------------
Beam prediction must be **shift-equivariant along the codebook**: held-out drive segments use
beam indices that may never appear in training, so any dense head over absolute beam index
overfits (verified: a temporal transformer with a dense 64-way head lost to the reactive
baseline on unseen segments).  Both architectures below share weights across beams.

* ``arch="transformer"`` (default): the 64 *beams are the tokens*; each token's feature is that
  beam's ``H``-step dB history (normalised) projected to ``d_model``.  Position along the
  codebook enters only through a depthwise convolutional positional encoding (relative, so
  the model is equivariant up to the codebook edges).  ``n_layers`` pre-norm encoder layers
  attend across beams; a per-token linear head gives one logit per beam.
* ``arch="tcn"``: a stack of 1-D convolutions over the beam axis with the ``H`` history steps
  as input channels (temporal convolutional network transposed to the beam axis).

In both cases the logits are **added to a residual of the last observed (normalised dB)
power vector** scaled by a learned scalar, and the head is zero-initialised, so at
initialisation the model *is* the reactive baseline and only learns corrections to it
(``residual_last_step=False`` disables this).

Loss
----
``"ce"``: cross-entropy against the target optimal beam.
``"power"``: power-weighted soft-label cross-entropy, where the soft target is the softmax of
the target power vector in dB (``target_temperature`` controls sharpness).  This rewards
near-optimal beams instead of treating an adjacent beam as fully wrong, which matches the dB
power-loss metric better than plain cross-entropy.

Memory
------
Default size (``d_model=64, n_layers=2, n_heads=4``) has ~70k parameters and trains on
``batch_size=256`` windows of ``H=8`` steps in well under 100 MB on the GPU; bf16 autocast is
enabled by default.  Everything is configurable through :class:`PredictThenActConfig`.

The class exposes ``predict_scores(windows) -> (M, 64)`` exactly like the other baselines.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.sequences import WindowedDataset


@dataclass
class PredictThenActConfig:
    n_beams: int = 64
    arch: str = "transformer"  # "transformer" | "tcn"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    kernel_size: int = 5  # conv kernel along the beam axis (positional encoding / tcn)
    loss: str = "power"  # "ce" | "power"
    residual_last_step: bool = True
    residual_init: float = 3.0  # initial scale of the last-step residual (normalised dB units)
    target_temperature: float = 1.0  # dB scale for the soft target in "power" loss
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 256
    mixed_precision: bool = True
    device: str = "auto"
    seed: int = 0
    val_fraction: float = 0.1  # fraction of training *windows* used for early stopping
    patience: int = 6


class _Forecaster(nn.Module):
    """Beam-equivariant forecaster: (B, H, 64) normalised dB -> (B, 64) logits."""

    def __init__(self, cfg: PredictThenActConfig, history: int):
        super().__init__()
        self.cfg = cfg
        pad = cfg.kernel_size // 2
        if cfg.arch == "transformer":
            self.input = nn.Linear(history, cfg.d_model)  # per-beam token from its time history
            self.conv_pe = nn.Conv1d(cfg.d_model, cfg.d_model, cfg.kernel_size, padding=pad, groups=cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, 1)
        elif cfg.arch == "tcn":
            layers: list[nn.Module] = []
            c_in = history
            for _ in range(cfg.n_layers):
                layers += [nn.Conv1d(c_in, cfg.d_model, cfg.kernel_size, padding=pad), nn.GELU(), nn.Dropout(cfg.dropout)]
                c_in = cfg.d_model
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Conv1d(c_in, 1, 1)
        else:
            raise ValueError(f"unknown arch {cfg.arch!r}")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.res_scale = nn.Parameter(torch.tensor(float(cfg.residual_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, H, 64) normalised dB
        if self.cfg.arch == "transformer":
            tok = self.input(x.transpose(1, 2))  # (B, 64, d)
            tok = tok + self.conv_pe(tok.transpose(1, 2)).transpose(1, 2)
            h = self.encoder(tok)
            logits = self.head(self.norm(h)).squeeze(-1)  # (B, 64)
        else:
            logits = self.head(self.tcn(x)).squeeze(1)  # conv over the beam axis, H channels
        if self.cfg.residual_last_step:
            logits = logits + self.res_scale * x[:, -1]
        return logits


def _to_db(power: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, floor))


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class PredictThenActBaseline:
    """Transformer forecaster of the beam distribution at ``t+k`` + greedy argmax controller."""

    def __init__(self, config: PredictThenActConfig | None = None, **overrides):
        self.cfg = config or PredictThenActConfig()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise TypeError(f"unknown config field {k!r}")
            setattr(self.cfg, k, v)
        self.device = resolve_device(self.cfg.device)
        self.model: _Forecaster | None = None
        self.mean_db: float = 0.0
        self.std_db: float = 1.0
        self.history_: int | None = None
        self.history_log: list[dict] = []
        self.peak_memory_bytes: int | None = None
        self.train_seconds: float = 0.0

    # ------------------------------------------------------------------ helpers
    def _normalise(self, histories: np.ndarray) -> torch.Tensor:
        x = (_to_db(histories) - self.mean_db) / self.std_db
        return torch.as_tensor(x, dtype=torch.float32)

    def _targets(self, windows: WindowedDataset) -> torch.Tensor:
        if self.cfg.loss == "ce":
            return torch.as_tensor(windows.target_beam, dtype=torch.long)
        soft = torch.as_tensor(_to_db(windows.target_power) / self.cfg.target_temperature, dtype=torch.float32)
        return torch.softmax(soft, dim=-1)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss == "ce":
            return F.cross_entropy(logits.float(), target)
        return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()

    def _autocast(self):
        enabled = self.cfg.mixed_precision and self.device.type == "cuda"
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)

    # -------------------------------------------------------------------- train
    def fit(self, windows: WindowedDataset, verbose: bool = False) -> PredictThenActBaseline:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.history_ = windows.history
        db = _to_db(windows.histories)
        self.mean_db, self.std_db = float(db.mean()), float(db.std() + 1e-6)

        x_all = self._normalise(windows.histories)
        y_all = self._targets(windows)
        m = len(windows)
        perm = rng.permutation(m)
        n_val = int(cfg.val_fraction * m) if m >= 20 else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        self.model = _Forecaster(cfg, windows.history).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        best_val, best_state, bad_epochs = math.inf, None, 0
        t0 = time.time()
        for epoch in range(cfg.epochs):
            self.model.train()
            order = rng.permutation(tr_idx)
            tot, cnt = 0.0, 0
            for s in range(0, len(order), cfg.batch_size):
                idx = torch.as_tensor(order[s : s + cfg.batch_size])
                xb, yb = x_all[idx].to(self.device), y_all[idx].to(self.device)
                with self._autocast():
                    loss = self._loss(self.model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
                cnt += len(idx)
            sched.step()
            train_loss = tot / max(cnt, 1)
            val_loss = self._eval_loss(x_all[val_idx], y_all[val_idx]) if n_val else train_loss
            self.history_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if verbose:
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
            if val_loss < best_val - 1e-5:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.train_seconds = time.time() - t0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
        return self

    @torch.no_grad()
    def _eval_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        assert self.model is not None
        self.model.eval()
        tot = 0.0
        for s in range(0, len(x), self.cfg.batch_size):
            xb, yb = x[s : s + self.cfg.batch_size].to(self.device), y[s : s + self.cfg.batch_size].to(self.device)
            with self._autocast():
                tot += float(self._loss(self.model(xb), yb)) * len(xb)
        return tot / max(len(x), 1)

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict_logits(self, windows: WindowedDataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PredictThenActBaseline.fit() must be called before predict_scores()")
        if windows.history != self.history_:
            raise ValueError(f"model was trained with history={self.history_}, got {windows.history}")
        self.model.eval()
        x = self._normalise(windows.histories)
        out = []
        for s in range(0, len(x), self.cfg.batch_size):
            with self._autocast():
                out.append(self.model(x[s : s + self.cfg.batch_size].to(self.device)).float().cpu())
        return torch.cat(out).numpy() if out else np.empty((0, self.cfg.n_beams))

    def predict_scores(self, windows: WindowedDataset) -> np.ndarray:
        """``(M, 64)`` forecast log-probabilities of the optimal beam at ``t+k``."""
        return self.predict_logits(windows)

    # --------------------------------------------------------------------- misc
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0

    def describe(self) -> str:
        mem = f"{self.peak_memory_bytes / 2**20:.1f} MB" if self.peak_memory_bytes is not None else "n/a (cpu)"
        return (f"predict-then-act: {self.n_parameters()} params, {len(self.history_log)} epochs, "
                f"{self.train_seconds:.1f}s on {self.device}, peak GPU memory {mem}, cfg={asdict(self.cfg)}")
