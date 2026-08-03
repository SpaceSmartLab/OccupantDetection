from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..schema import DetectionBatch


class Detector(ABC):
    name: str
    checkpoint: str

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> DetectionBatch:
        """Return person detections in the shared Nx6 format."""

    def close(self) -> None:
        """Release model resources when an implementation needs it."""

    def metadata(self) -> dict[str, Any]:
        return {"detector": self.name, "checkpoint": self.checkpoint}

