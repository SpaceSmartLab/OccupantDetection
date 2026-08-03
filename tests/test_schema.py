import numpy as np
import pytest

from thermal_occupancy.schema import DetectionBatch


def test_detection_batch_has_common_nx6_shape() -> None:
    batch = DetectionBatch([[1, 2, 3, 4, 0.8, 0]])
    assert batch.values.shape == (1, 6)
    assert batch.class_id.tolist() == [0]


def test_empty_detection_batch_is_normalized() -> None:
    assert DetectionBatch([]).values.shape == (0, 6)


def test_invalid_detection_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        DetectionBatch(np.zeros((2, 5), dtype=np.float32))

