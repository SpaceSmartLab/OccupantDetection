from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..schema import TrackedObject


@dataclass(frozen=True)
class NormalizedPolygonROI:
    points: tuple[tuple[float, float], ...]
    anchor: str = "bottom_center"

    def __init__(
        self, points: Sequence[Sequence[float]], anchor: str = "bottom_center"
    ) -> None:
        normalized = tuple((float(point[0]), float(point[1])) for point in points)
        if len(normalized) < 3:
            raise ValueError("An ROI polygon needs at least three points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in normalized):
            raise ValueError("ROI coordinates must be normalized to [0, 1]")
        if anchor not in {"bottom_center", "center"}:
            raise ValueError("ROI anchor must be 'bottom_center' or 'center'")
        object.__setattr__(self, "points", normalized)
        object.__setattr__(self, "anchor", anchor)

    def pixel_points(self, width: int, height: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                min(width - 1, max(0, round(x * (width - 1)))),
                min(height - 1, max(0, round(y * (height - 1)))),
            )
            for x, y in self.points
        )

    def contains_track(
        self, track: TrackedObject, width: int, height: int
    ) -> bool:
        point = track.bottom_center if self.anchor == "bottom_center" else track.center
        return _point_in_polygon(point, self.pixel_points(width, height))

    def inside_track_ids(
        self, tracks: Iterable[TrackedObject], width: int, height: int
    ) -> set[int]:
        return {
            track.track_id
            for track in tracks
            if self.contains_track(track, width=width, height=height)
        }


def _point_in_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[int, int]]
) -> bool:
    """Ray casting with boundary points treated as inside."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _point_on_segment(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-6:
        return False
    return (
        min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6
        and min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6
    )

