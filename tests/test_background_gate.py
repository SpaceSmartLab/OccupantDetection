import numpy as np

from thermal_occupancy.background_gate import EmptyBackgroundGate
from thermal_occupancy.schema import DetectionBatch


def _gate(background: np.ndarray) -> EmptyBackgroundGate:
    return EmptyBackgroundGate.from_frames(
        [background],
        pixel_difference_threshold=25,
        minimum_changed_pixel_ratio=0.2,
        approved_track_iou_threshold=0.3,
        blur_kernel_size=1,
    )


def test_static_background_detection_is_rejected() -> None:
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _gate(background).filter(
        background.copy(), DetectionBatch([[20, 20, 60, 80, 0.9, 0]])
    )
    assert len(result.detections) == 0
    assert result.rejected_count == 1


def test_changed_detection_is_accepted() -> None:
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    current = background.copy()
    current[20:80, 20:60] = 255
    result = _gate(background).filter(
        current, DetectionBatch([[20, 20, 60, 80, 0.9, 0]])
    )
    assert len(result.detections) == 1
    assert result.changed_pixel_ratios[0] == 1.0


def test_approved_track_overlap_retains_stationary_detection() -> None:
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _gate(background).filter(
        background.copy(),
        DetectionBatch([[20, 20, 60, 80, 0.9, 0]]),
        approved_boxes=[(21, 21, 61, 81)],
    )
    assert len(result.detections) == 1
    assert result.retained_by_track_count == 1
