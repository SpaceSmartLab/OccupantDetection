from __future__ import annotations

from ..schema import OccupancySnapshot, TrackedObject
from .roi import NormalizedPolygonROI
from .temporal import TemporalOccupancyStabilizer


class OccupancyCounter:
    def __init__(
        self,
        *,
        roi: NormalizedPolygonROI,
        stabilizer: TemporalOccupancyStabilizer,
        width: int,
        height: int,
    ) -> None:
        self.roi = roi
        self.stabilizer = stabilizer
        self.width = width
        self.height = height

    def update(
        self, frame_index: int, tracks: list[TrackedObject]
    ) -> OccupancySnapshot:
        raw_ids = tuple(
            sorted(self.roi.inside_track_ids(tracks, self.width, self.height))
        )
        stable_ids = self.stabilizer.update(frame_index, raw_ids)
        return OccupancySnapshot(
            raw_count=len(raw_ids),
            stable_count=len(stable_ids),
            raw_track_ids=raw_ids,
            stable_track_ids=stable_ids,
        )
