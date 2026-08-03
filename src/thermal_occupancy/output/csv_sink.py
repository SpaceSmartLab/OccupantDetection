from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


FRAME_FIELDS = (
    "frame_index",
    "timestamp_sec",
    "timestamp_hms",
    "raw_detection_count",
    "background_rejected_count",
    "background_retained_count",
    "detection_count",
    "tracked_count",
    "track_ids",
    "raw_occupancy_count",
    "raw_occupancy_track_ids",
    "occupancy_count",
    "occupancy_track_ids",
    "occupied",
    "inference_ms",
    "processing_ms",
)


class FrameCsvSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=FRAME_FIELDS)
        self._writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({field: row.get(field, "") for field in FRAME_FIELDS})

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> "FrameCsvSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    write_csv_rows(path, [row])


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("At least one row is required")
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
