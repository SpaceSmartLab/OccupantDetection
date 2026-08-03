from thermal_occupancy.occupancy.roi import NormalizedPolygonROI
from thermal_occupancy.schema import TrackedObject


def _track(track_id: int, xyxy: tuple[float, float, float, float]) -> TrackedObject:
    return TrackedObject(xyxy=xyxy, score=0.9, class_id=0, track_id=track_id)


def test_bottom_center_anchor_controls_roi_membership() -> None:
    roi = NormalizedPolygonROI(
        [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    )
    inside = _track(1, (40, 20, 60, 60))
    outside = _track(2, (0, 0, 20, 20))
    assert roi.contains_track(inside, width=101, height=101)
    assert not roi.contains_track(outside, width=101, height=101)


def test_full_frame_roi_includes_boundary() -> None:
    roi = NormalizedPolygonROI([(0, 0), (1, 0), (1, 1), (0, 1)])
    boundary = _track(1, (0, 0, 10, 100))
    assert roi.contains_track(boundary, width=101, height=101)

