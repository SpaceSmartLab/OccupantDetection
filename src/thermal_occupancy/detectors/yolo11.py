from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..cache import configure_local_model_caches
from ..schema import DetectionBatch
from .base import Detector


class YOLO11sDetector(Detector):
    name = "yolo11s"

    def __init__(
        self,
        *,
        checkpoint: str,
        confidence_threshold: float,
        person_class_id: int,
        image_size: int,
        device: str | int | None,
        nms_iou_threshold: float,
    ) -> None:
        cache_root = configure_local_model_caches()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO11s. Install the project dependencies."
            ) from exc

        requested_checkpoint = Path(checkpoint)
        if (
            requested_checkpoint.parent == Path(".")
            and requested_checkpoint.suffix == ".pt"
            and not requested_checkpoint.exists()
        ):
            model_dir = cache_root / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            requested_checkpoint = model_dir / requested_checkpoint.name

        self.checkpoint = str(requested_checkpoint)
        self.confidence_threshold = float(confidence_threshold)
        self.person_class_id = int(person_class_id)
        self.image_size = int(image_size)
        self.device = None if device == "auto" else device
        self.nms_iou_threshold = float(nms_iou_threshold)
        self._model = YOLO(self.checkpoint)

    def detect(self, frame_bgr: np.ndarray) -> DetectionBatch:
        result = self._model.predict(
            source=frame_bgr,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionBatch.empty()

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        confidence = (
            boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        class_id = boxes.cls.detach().cpu().numpy().astype(np.float32, copy=False)
        values = np.column_stack((xyxy, confidence, class_id))
        return DetectionBatch(values)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "nms_iou_threshold": self.nms_iou_threshold,
        }
