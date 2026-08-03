from __future__ import annotations

from typing import Any

from .base import Detector
from .rtdetr_v2 import RTDETRv2SDetector
from .yolo11 import YOLO11sDetector


def create_detector(
    name: str,
    *,
    common: dict[str, Any],
    model: dict[str, Any],
) -> Detector:
    shared = {
        "checkpoint": str(model["checkpoint"]),
        "confidence_threshold": float(common["confidence_threshold"]),
        "person_class_id": int(common.get("person_class_id", 0)),
        "image_size": int(common.get("image_size", 640)),
        "device": common.get("device", "auto"),
    }
    if name == "yolo11s":
        return YOLO11sDetector(
            **shared,
            nms_iou_threshold=float(model.get("nms_iou_threshold", 0.7)),
        )
    if name == "rtdetrv2_s":
        return RTDETRv2SDetector(**shared)
    raise ValueError(f"Unsupported detector: {name}")

