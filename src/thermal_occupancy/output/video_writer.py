from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class AnnotatedVideoWriter:
    def __init__(
        self,
        *,
        path: Path,
        codec: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for annotated video output."
            ) from exc
        if len(codec) != 4:
            raise ValueError("Video codec must contain exactly four characters")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer: Any = cv2.VideoWriter(
            str(path), fourcc, float(fps), (int(width), int(height))
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"Could not open output video: {path}. "
                "Try another output.video_codec such as mp4v."
            )

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class NullVideoWriter:
    def write(self, frame_bgr: np.ndarray) -> None:
        del frame_bgr

    def close(self) -> None:
        return None

