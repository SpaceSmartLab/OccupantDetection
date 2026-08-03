from __future__ import annotations

import os
from pathlib import Path


def configure_local_model_caches() -> Path:
    """Keep framework settings and downloaded weights in a writable local cache."""
    configured = os.environ.get("THERMAL_OCCUPANCY_CACHE_DIR")
    cache_root = (
        Path(configured).expanduser()
        if configured
        else Path.cwd() / ".cache"
    ).resolve()
    ultralytics_dir = cache_root / "ultralytics"
    huggingface_dir = cache_root / "huggingface"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    huggingface_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("HF_HOME", str(huggingface_dir))
    return cache_root
