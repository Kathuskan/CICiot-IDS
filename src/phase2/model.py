"""
model.py — Phase 2
==================
Shared neural network architecture (`CICIoTNet`) and deterministic feature
preprocessor (`FlowPreprocessor`) used across training, evaluation, and inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

# Skewed continuous / count columns that benefit from log1p compression
DEFAULT_LOG1P_COLUMNS: tuple[str, ...] = (
    "Header_Length",
    "Time_To_Live",
    "Rate",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
)


class FlowPreprocessor:
    """
    Fits and applies log1p compression + z-score standardization on flow features.

    All fitted parameters are serialized to JSON so evaluation and inference
    reproduce the exact training transformation without scikit-learn or pickle.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        log1p_columns: Sequence[str] = DEFAULT_LOG1P_COLUMNS,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.feature_columns: list[str] = list(feature_columns) if feature_columns else []
        self.log1p_columns: list[str] = list(log1p_columns)
        self.mean: np.ndarray | None = mean
        self.std: np.ndarray | None = std
        self.threshold: float = float(threshold)

    def fit(self, df: pd.DataFrame) -> "FlowPreprocessor":
        """Compute feature column order, log1p indices, mean, and std from training data."""
        self.feature_columns = list(df.columns)
        self.log1p_columns = [c for c in self.log1p_columns if c in self.feature_columns]

        arr = self._apply_log1p(df[self.feature_columns].to_numpy(dtype=np.float64, copy=True))
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std == 0.0] = 1.0

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame of raw flow features into float32 standardized array."""
        if self.mean is None or self.std is None:
            raise RuntimeError("FlowPreprocessor must be fit or loaded before calling transform().")

        arr = df[self.feature_columns].to_numpy(dtype=np.float64, copy=True)
        arr = self._apply_log1p(arr)
        arr = (arr - self.mean) / self.std
        return arr.astype(np.float32)

    def _apply_log1p(self, arr: np.ndarray) -> np.ndarray:
        """Apply log1p to configured skewed non-negative columns in-place."""
        col_to_idx = {col: idx for idx, col in enumerate(self.feature_columns)}
        for col in self.log1p_columns:
            idx = col_to_idx.get(col)
            if idx is not None:
                arr[:, idx] = np.log1p(np.clip(arr[:, idx], a_min=0.0, a_max=None))
        return arr

    def save(self, path: str | Path) -> None:
        """Persist preprocessing parameters to a JSON file."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save an unfitted FlowPreprocessor.")
        payload = {
            "feature_columns": self.feature_columns,
            "log1p_columns": self.log1p_columns,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "threshold": self.threshold,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FlowPreprocessor":
        """Load preprocessing parameters from a JSON file."""
        payload = json.loads(Path(path).read_text())
        return cls(
            feature_columns=payload["feature_columns"],
            log1p_columns=payload["log1p_columns"],
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
            threshold=float(payload.get("threshold", 0.5)),
        )


class CICIoTNet(nn.Module):
    """
    Tabular deep neural network for binary IoT intrusion detection.

    Architecture:
        Input(in_features)
        -> [Linear -> BatchNorm1d -> SiLU -> Dropout] x len(hidden_dims)
        -> Linear(last_hidden, 1)  [raw logits]
    """

    def __init__(
        self,
        in_features: int = 39,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev_dim = in_features
        for idx, dim in enumerate(hidden_dims):
            drop_rate = dropout if idx < len(hidden_dims) - 1 else dropout * 0.5
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.SiLU(),
                nn.Dropout(drop_rate),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_device(requested: str = "auto") -> torch.device:
    """Pick the best available PyTorch compute device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
