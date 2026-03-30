# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp
"""
Server-side data store for vibration signals.

Signals are kept in memory on the MCP server so that large numpy arrays
never need to transit through the LLM conversation context.  The agent
only sees compact summaries (statistics, peak lists, diagnosis reports).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def _kurtosis(x: NDArray) -> float:
    """Excess kurtosis (Fisher definition, normal = 0)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)  # sample std, consistent with main.py & fault_detection.py
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


@dataclass
class DataEntry:
    """A stored signal with metadata."""

    signal: NDArray[np.floating]
    sample_rate: float
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.signal.shape[0]

    @property
    def n_channels(self) -> int:
        return self.signal.shape[1] if self.signal.ndim > 1 else 1

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate > 0 else 0

    def summary(self) -> dict:
        """Return a compact summary (no raw data)."""
        sig = self.signal
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)

        channel_stats = {}
        labels = self.metadata.get(
            "axis_labels",
            ["X", "Y", "Z"][: sig.shape[1]]
            if sig.shape[1] <= 3
            else [f"CH{i}" for i in range(sig.shape[1])],
        )
        for i, label in enumerate(labels):
            if i >= sig.shape[1]:
                break
            col = sig[:, i]
            rms = float(np.sqrt(np.mean(col**2)))
            channel_stats[label] = {
                "mean": round(float(np.mean(col)), 6),
                "std": round(float(np.std(col)), 6),
                "rms": round(rms, 6),
                "peak": round(float(np.max(np.abs(col))), 6),
                "crest_factor": round(float(np.max(np.abs(col)) / rms), 2)
                if rms > 0
                else 0,
                "kurtosis": round(float(_kurtosis(col)), 2),
            }

        return {
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "sample_rate_hz": self.sample_rate,
            "duration_s": round(self.duration_s, 4),
            "channel_stats": channel_stats,
            "metadata": {
                k: v for k, v in self.metadata.items() if k != "axis_labels"
            },
        }


class DataStore:
    """Simple in-memory store for vibration signals."""

    def __init__(self) -> None:
        self._entries: dict[str, DataEntry] = {}

    def put(
        self,
        data_id: str,
        signal: NDArray[np.floating],
        sample_rate: float,
        metadata: dict | None = None,
    ) -> str:
        """Store a signal.  Returns the data_id."""
        self._entries[data_id] = DataEntry(
            signal=np.asarray(signal, dtype=np.float64),
            sample_rate=sample_rate,
            metadata=metadata or {},
        )
        return data_id

    def put_auto(
        self,
        signal: NDArray[np.floating],
        sample_rate: float,
        metadata: dict | None = None,
    ) -> str:
        """Store a signal with an auto-generated ID."""
        h = hashlib.md5(signal.tobytes()[:1024]).hexdigest()[:8]
        data_id = f"sig_{h}_{int(time.time()) % 100000}"
        return self.put(data_id, signal, sample_rate, metadata)

    def get(self, data_id: str) -> DataEntry | None:
        return self._entries.get(data_id)

    def remove(self, data_id: str) -> bool:
        return self._entries.pop(data_id, None) is not None

    def list_ids(self) -> list[str]:
        return list(self._entries.keys())

    def list_entries(self) -> list[dict]:
        """Return summaries of all stored entries."""
        return [{"data_id": k, **v.summary()} for k, v in self._entries.items()]


# Global singleton — shared across all tools in this server
store = DataStore()
