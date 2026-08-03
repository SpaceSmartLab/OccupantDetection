from pathlib import Path

import pytest
import yaml

from thermal_occupancy.config import load_config


def test_invalid_roi_is_rejected(tmp_path: Path) -> None:
    config = {
        "input_video": "missing.mp4",
        "detectors": {"enabled": ["yolo11s"]},
        "tracker": {},
        "occupancy": {"roi": {"polygon": [[0, 0], [1, 1]]}},
        "output": {},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)

