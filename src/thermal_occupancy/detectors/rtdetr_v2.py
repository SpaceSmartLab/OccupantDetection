from __future__ import annotations

from typing import Any

import numpy as np

from ..cache import configure_local_model_caches
from ..schema import DetectionBatch
from .base import Detector


class RTDETRv2SDetector(Detector):
    """RT-DETRv2 small adapter backed by the official R18VD checkpoint."""

    name = "rtdetrv2_s"

    def __init__(
        self,
        *,
        checkpoint: str,
        confidence_threshold: float,
        person_class_id: int,
        image_size: int,
        device: str | int | None,
    ) -> None:
        configure_local_model_caches()
        try:
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and Transformers are required for RT-DETRv2-S. "
                "Install the project dependencies."
            ) from exc

        self._torch = torch
        self.checkpoint = checkpoint
        self.confidence_threshold = float(confidence_threshold)
        self.person_class_id = int(person_class_id)
        self.image_size = int(image_size)
        self.device = _resolve_torch_device(torch, device)
        processor_kwargs = {
            "size": {"height": self.image_size, "width": self.image_size}
        }
        try:
            self._processor = RTDetrImageProcessor.from_pretrained(
                checkpoint,
                local_files_only=True,
                **processor_kwargs,
            )
            self._model = RTDetrV2ForObjectDetection.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
        except OSError:
            self._processor = RTDetrImageProcessor.from_pretrained(
                checkpoint,
                **processor_kwargs,
            )
            self._model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
        self._model.to(self.device)
        self._model.eval()

    def detect(self, frame_bgr: np.ndarray) -> DetectionBatch:
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        inputs = self._processor(images=frame_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        height, width = frame_bgr.shape[:2]
        target_sizes = self._torch.tensor(
            [(height, width)], dtype=self._torch.int64, device=self.device
        )
        result = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]
        labels = result["labels"]
        person_mask = labels == self.person_class_id
        if int(person_mask.sum().item()) == 0:
            return DetectionBatch.empty()

        xyxy = (
            result["boxes"][person_mask]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        confidence = (
            result["scores"][person_mask]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        class_id = (
            labels[person_mask]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        values = np.column_stack((xyxy, confidence, class_id))
        return DetectionBatch(values)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "device": str(self.device),
        }


def _resolve_torch_device(torch: Any, value: str | int | None) -> Any:
    if value in (None, "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))
