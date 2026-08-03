from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    values: dict[str, Any]
    source_path: Path

    @property
    def input_video(self) -> Path:
        return _resolve_path(
            self.values["input_video"], self.source_path.parent, must_exist=True
        )

    @property
    def output_root(self) -> Path:
        return _resolve_path(
            self.values.get("output_root", "outputs"),
            self.source_path.parent.parent,
            must_exist=False,
        )

    @property
    def detector_names(self) -> list[str]:
        names = list(self.values["detectors"]["enabled"])
        supported = {"yolo11s", "rtdetrv2_s"}
        unknown = set(names) - supported
        if unknown:
            raise ValueError(f"Unsupported detectors: {sorted(unknown)}")
        if not names:
            raise ValueError("At least one detector must be enabled")
        return names

    def section(self, *keys: str) -> dict[str, Any]:
        node: Any = self.values
        for key in keys:
            node = node[key]
        if not isinstance(node, Mapping):
            raise TypeError(f"Configuration section {'.'.join(keys)} is not a mapping")
        return dict(node)

    def with_overrides(
        self,
        *,
        input_video: str | Path | None = None,
        output_root: str | Path | None = None,
        max_frames: int | None = None,
        detector_names: list[str] | None = None,
    ) -> "PipelineConfig":
        values = deepcopy(self.values)
        if input_video is not None:
            values["input_video"] = str(Path(input_video).expanduser().resolve())
        if output_root is not None:
            values["output_root"] = str(Path(output_root).expanduser().resolve())
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be greater than zero")
            values.setdefault("runtime", {})["max_frames"] = max_frames
        if detector_names is not None:
            values["detectors"]["enabled"] = detector_names
        return PipelineConfig(values=values, source_path=self.source_path)


def load_config(path: str | Path) -> PipelineConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("The YAML root must be a mapping")
    _validate(values)
    return PipelineConfig(values=values, source_path=source_path)


def _resolve_path(value: str | Path, base: Path, *, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")
    return path


def _validate(values: dict[str, Any]) -> None:
    required = ("input_video", "detectors", "tracker", "occupancy", "output")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")

    polygon = values["occupancy"]["roi"]["polygon"]
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("occupancy.roi.polygon must contain at least 3 points")
    for point in polygon:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(value, (int, float)) for value in point)
            or not all(0.0 <= float(value) <= 1.0 for value in point)
        ):
            raise ValueError("ROI points must be normalized [x, y] pairs in [0, 1]")

