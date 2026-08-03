from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DETECTION_COLUMNS = ("x1", "y1", "x2", "y2", "score", "class_id")


@dataclass(frozen=True)
class DetectionBatch:
    """Detector-independent Nx6 array: x1, y1, x2, y2, score, class_id."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.size == 0:
            values = np.empty((0, 6), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(
                "DetectionBatch must have shape (N, 6) with columns "
                f"{DETECTION_COLUMNS}; got {values.shape}"
            )
        if values.size and not np.isfinite(values).all():
            raise ValueError("DetectionBatch contains non-finite values")
        object.__setattr__(self, "values", values)

    @classmethod
    def empty(cls) -> "DetectionBatch":
        return cls(np.empty((0, 6), dtype=np.float32))

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def xyxy(self) -> np.ndarray:
        return self.values[:, :4]

    @property
    def confidence(self) -> np.ndarray:
        return self.values[:, 4]

    @property
    def class_id(self) -> np.ndarray:
        return self.values[:, 5].astype(np.int32, copy=False)


@dataclass(frozen=True)
class TrackedObject:
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    track_id: int

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def track_ids(tracks: Iterable[TrackedObject]) -> tuple[int, ...]:
    return tuple(sorted({track.track_id for track in tracks}))


@dataclass(frozen=True)
class OccupancySnapshot:
    raw_count: int
    stable_count: int
    raw_track_ids: tuple[int, ...]
    stable_track_ids: tuple[int, ...]

