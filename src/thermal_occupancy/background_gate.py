from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .schema import DetectionBatch


@dataclass(frozen=True)
class BackgroundGateResult:
    detections: DetectionBatch
    raw_count: int
    rejected_count: int
    retained_by_track_count: int
    changed_pixel_ratios: tuple[float, ...]


class EmptyBackgroundGate:
    """Reject static detections that still look like the known-empty scene.

    A detection that overlaps a previously approved track is retained even when
    its foreground score drops. This lets an admitted person remain tracked
    after sitting still while preventing empty chairs from creating new tracks.
    """

    def __init__(
        self,
        reference_gray: np.ndarray,
        *,
        pixel_difference_threshold: int,
        minimum_changed_pixel_ratio: float,
        approved_track_iou_threshold: float,
        blur_kernel_size: int = 5,
    ) -> None:
        reference = np.asarray(reference_gray, dtype=np.uint8)
        if reference.ndim != 2 or reference.size == 0:
            raise ValueError("Background reference must be a non-empty grayscale image")
        if not 0 <= int(pixel_difference_threshold) <= 255:
            raise ValueError("pixel_difference_threshold must be in [0, 255]")
        if not 0.0 <= float(minimum_changed_pixel_ratio) <= 1.0:
            raise ValueError("minimum_changed_pixel_ratio must be in [0, 1]")
        if not 0.0 <= float(approved_track_iou_threshold) <= 1.0:
            raise ValueError("approved_track_iou_threshold must be in [0, 1]")
        if int(blur_kernel_size) <= 0 or int(blur_kernel_size) % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer")

        self.reference_gray = reference
        self.pixel_difference_threshold = int(pixel_difference_threshold)
        self.minimum_changed_pixel_ratio = float(minimum_changed_pixel_ratio)
        self.approved_track_iou_threshold = float(approved_track_iou_threshold)
        self.blur_kernel_size = int(blur_kernel_size)

    @classmethod
    def from_frames(
        cls,
        frames_bgr: Sequence[np.ndarray],
        *,
        pixel_difference_threshold: int,
        minimum_changed_pixel_ratio: float,
        approved_track_iou_threshold: float,
        blur_kernel_size: int = 5,
    ) -> "EmptyBackgroundGate":
        if not frames_bgr:
            raise ValueError("At least one empty-scene frame is required")
        grayscale = [
            _prepare_gray(frame, blur_kernel_size=blur_kernel_size)
            for frame in frames_bgr
        ]
        expected_shape = grayscale[0].shape
        if any(frame.shape != expected_shape for frame in grayscale):
            raise ValueError("All background frames must have the same dimensions")
        reference = np.median(np.stack(grayscale, axis=0), axis=0).astype(np.uint8)
        return cls(
            reference,
            pixel_difference_threshold=pixel_difference_threshold,
            minimum_changed_pixel_ratio=minimum_changed_pixel_ratio,
            approved_track_iou_threshold=approved_track_iou_threshold,
            blur_kernel_size=blur_kernel_size,
        )

    def filter(
        self,
        frame_bgr: np.ndarray,
        detections: DetectionBatch,
        *,
        approved_boxes: Iterable[tuple[float, float, float, float]] = (),
    ) -> BackgroundGateResult:
        raw_count = len(detections)
        if raw_count == 0:
            return BackgroundGateResult(
                detections=DetectionBatch.empty(),
                raw_count=0,
                rejected_count=0,
                retained_by_track_count=0,
                changed_pixel_ratios=(),
            )

        gray = _prepare_gray(frame_bgr, blur_kernel_size=self.blur_kernel_size)
        if gray.shape != self.reference_gray.shape:
            raise ValueError(
                "Frame dimensions do not match the empty-scene background reference"
            )

        changed = np.abs(
            gray.astype(np.int16) - self.reference_gray.astype(np.int16)
        ) >= self.pixel_difference_threshold
        height, width = gray.shape
        prior_boxes = tuple(approved_boxes)
        accepted_indices: list[int] = []
        changed_ratios: list[float] = []
        retained_by_track_count = 0

        for index, raw_box in enumerate(detections.xyxy):
            box = tuple(float(value) for value in raw_box)
            x1, y1, x2, y2 = _clip_box(box, width=width, height=height)
            patch = changed[y1:y2, x1:x2]
            changed_ratio = float(patch.mean()) if patch.size else 0.0
            changed_ratios.append(changed_ratio)

            retained_by_track = any(
                _box_iou(box, approved_box) >= self.approved_track_iou_threshold
                for approved_box in prior_boxes
            )
            if retained_by_track:
                retained_by_track_count += 1
            if retained_by_track or changed_ratio >= self.minimum_changed_pixel_ratio:
                accepted_indices.append(index)

        accepted = (
            DetectionBatch(detections.values[accepted_indices])
            if accepted_indices
            else DetectionBatch.empty()
        )
        return BackgroundGateResult(
            detections=accepted,
            raw_count=raw_count,
            rejected_count=raw_count - len(accepted),
            retained_by_track_count=retained_by_track_count,
            changed_pixel_ratios=tuple(changed_ratios),
        )

    @property
    def settings(self) -> dict[str, int | float]:
        return {
            "pixel_difference_threshold": self.pixel_difference_threshold,
            "minimum_changed_pixel_ratio": self.minimum_changed_pixel_ratio,
            "approved_track_iou_threshold": self.approved_track_iou_threshold,
            "blur_kernel_size": self.blur_kernel_size,
        }


def _prepare_gray(frame_bgr: np.ndarray, *, blur_kernel_size: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for background comparison") from exc
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)


def _clip_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = min(width, max(0, int(np.floor(x1))))
    top = min(height, max(0, int(np.floor(y1))))
    right = min(width, max(left, int(np.ceil(x2))))
    bottom = min(height, max(top, int(np.ceil(y2))))
    return left, top, right, bottom


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
