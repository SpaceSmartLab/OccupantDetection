from __future__ import annotations

import numpy as np

from ..schema import DetectionBatch, TrackedObject


class SharedByteTrack:
    """One ByteTrack implementation and parameter set shared by all detectors."""

    name = "bytetrack"

    def __init__(
        self,
        *,
        frame_rate: float,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_matching_threshold: float,
    ) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "Supervision 0.26.1 is required for the shared ByteTrack adapter."
            ) from exc

        self._sv = sv
        self.settings = {
            "track_activation_threshold": float(track_activation_threshold),
            "lost_track_buffer": int(lost_track_buffer),
            "minimum_matching_threshold": float(minimum_matching_threshold),
            "frame_rate": int(round(frame_rate)),
        }
        self._tracker = sv.ByteTrack(**self.settings)

    def update(self, detections: DetectionBatch) -> list[TrackedObject]:
        sv_detections = self._sv.Detections(
            xyxy=detections.xyxy.astype(np.float32, copy=False),
            confidence=detections.confidence.astype(np.float32, copy=False),
            class_id=detections.class_id,
        )
        tracked = self._tracker.update_with_detections(sv_detections)
        if len(tracked) == 0:
            return []
        if tracked.tracker_id is None:
            raise RuntimeError("ByteTrack returned detections without tracker IDs")

        confidence = (
            tracked.confidence
            if tracked.confidence is not None
            else np.ones(len(tracked), dtype=np.float32)
        )
        class_id = (
            tracked.class_id
            if tracked.class_id is not None
            else np.zeros(len(tracked), dtype=np.int32)
        )
        return [
            TrackedObject(
                xyxy=tuple(float(value) for value in tracked.xyxy[index]),
                score=float(confidence[index]),
                class_id=int(class_id[index]),
                track_id=int(tracked.tracker_id[index]),
            )
            for index in range(len(tracked))
        ]

