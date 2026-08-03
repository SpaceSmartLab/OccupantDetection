from __future__ import annotations

import numpy as np

from ..occupancy.roi import NormalizedPolygonROI
from ..schema import OccupancySnapshot, TrackedObject


def annotate_frame(
    frame_bgr: np.ndarray,
    *,
    model_name: str,
    timestamp_hms: str,
    tracks: list[TrackedObject],
    occupancy: OccupancySnapshot,
    roi: NormalizedPolygonROI,
    draw_confidence: bool,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame annotation.") from exc

    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    polygon = np.asarray(roi.pixel_points(width, height), dtype=np.int32)
    cv2.polylines(
        canvas, [polygon], isClosed=True, color=(0, 215, 255), thickness=2
    )

    occupied_ids = set(occupancy.stable_track_ids)
    raw_ids = set(occupancy.raw_track_ids)
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.xyxy)
        if track.track_id in occupied_ids:
            color = (0, 220, 0)
        elif track.track_id in raw_ids:
            color = (0, 165, 255)
        else:
            color = (220, 120, 20)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"person ID:{track.track_id}"
        if draw_confidence:
            label += f" {track.score:.2f}"
        _draw_label(cv2, canvas, label, x1, y1, color)

    header_height = 72
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, header_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.putText(
        canvas,
        f"{model_name} + ByteTrack | {timestamp_hms}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"Occupancy: {occupancy.stable_count} "
            f"(raw ROI tracks: {occupancy.raw_count})"
        ),
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_label(
    cv2: object,
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    label_y = max(text_height + baseline + 2, y)
    cv2.rectangle(
        frame,
        (x, label_y - text_height - baseline - 4),
        (x + text_width + 4, label_y + 2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 2, label_y - baseline - 1),
        font,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )
