from thermal_occupancy.schema import DetectionBatch
from thermal_occupancy.trackers import SharedByteTrack


def test_shared_bytetrack_accepts_common_detection_format() -> None:
    tracker = SharedByteTrack(
        frame_rate=25,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )
    tracks = tracker.update(DetectionBatch([[10, 10, 30, 50, 0.9, 0]]))
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_id == 0

