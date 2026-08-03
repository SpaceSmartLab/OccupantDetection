from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import OccupancySnapshot, TrackedObject


@dataclass
class RunMetrics:
    inference_ms: list[float] = field(default_factory=list)
    processing_ms: list[float] = field(default_factory=list)
    raw_counts: list[int] = field(default_factory=list)
    stable_counts: list[int] = field(default_factory=list)
    unique_track_ids: set[int] = field(default_factory=set)
    detection_total: int = 0
    track_total: int = 0

    def add(
        self,
        *,
        inference_ms: float,
        processing_ms: float,
        detection_count: int,
        tracks: list[TrackedObject],
        occupancy: OccupancySnapshot,
    ) -> None:
        self.inference_ms.append(float(inference_ms))
        self.processing_ms.append(float(processing_ms))
        self.raw_counts.append(occupancy.raw_count)
        self.stable_counts.append(occupancy.stable_count)
        self.detection_total += int(detection_count)
        self.track_total += len(tracks)
        self.unique_track_ids.update(track.track_id for track in tracks)

    def summary(
        self,
        *,
        model_name: str,
        checkpoint: str,
        input_video: Path,
        input_fps: float,
        input_width: int,
        input_height: int,
        total_input_frames: int,
        frames_processed: int,
        warmup_iterations: int,
        model_load_sec: float,
        wall_time_sec: float,
        tracker_settings: dict[str, Any],
    ) -> dict[str, Any]:
        inference = np.asarray(self.inference_ms, dtype=np.float64)
        processing = np.asarray(self.processing_ms, dtype=np.float64)
        stable = np.asarray(self.stable_counts, dtype=np.float64)
        raw = np.asarray(self.raw_counts, dtype=np.float64)
        processed_duration = frames_processed / input_fps
        tracker_text = ";".join(f"{k}={v}" for k, v in tracker_settings.items())

        return {
            "model": model_name,
            "checkpoint": checkpoint,
            "tracker": "bytetrack",
            "tracker_settings": tracker_text,
            "input_video": str(input_video),
            "input_width": input_width,
            "input_height": input_height,
            "input_fps": round(input_fps, 6),
            "total_input_frames": total_input_frames,
            "frames_processed": frames_processed,
            "processed_duration_sec": round(processed_duration, 6),
            "warmup_iterations": warmup_iterations,
            "model_load_sec": round(model_load_sec, 6),
            "wall_time_sec": round(wall_time_sec, 6),
            "end_to_end_fps": _safe_round(frames_processed / wall_time_sec),
            "mean_inference_ms": _mean(inference),
            "p50_inference_ms": _percentile(inference, 50),
            "p95_inference_ms": _percentile(inference, 95),
            "mean_processing_ms": _mean(processing),
            "total_person_detections": self.detection_total,
            "mean_person_detections_per_frame": _safe_round(
                self.detection_total / max(1, frames_processed)
            ),
            "unique_track_ids": len(self.unique_track_ids),
            "mean_tracked_people": _safe_round(
                self.track_total / max(1, frames_processed)
            ),
            "mean_raw_occupancy": _mean(raw),
            "mean_stable_occupancy": _mean(stable),
            "max_stable_occupancy": int(stable.max()) if stable.size else 0,
            "occupied_frame_ratio": _safe_round(
                float(np.mean(stable > 0)) if stable.size else 0.0
            ),
        }


def _mean(values: np.ndarray) -> float:
    return _safe_round(float(values.mean()) if values.size else 0.0)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return _safe_round(
        float(np.percentile(values, percentile)) if values.size else 0.0
    )


def _safe_round(value: float) -> float:
    return round(float(value), 6)
