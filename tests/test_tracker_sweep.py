from __future__ import annotations

import csv
from pathlib import Path

import pytest

from thermal_occupancy.evaluation.tracker_sweep import compare_tracker_sweep
from thermal_occupancy.output.csv_sink import FrameCsvSink
from thermal_occupancy.pipeline import _normalize_activation_thresholds


def _write_frames(path: Path, counts: list[int]) -> None:
    with FrameCsvSink(path) as sink:
        for frame_index, count in enumerate(counts):
            sink.write(
                {
                    "frame_index": frame_index,
                    "occupancy_count": count,
                }
            )


def _summary(threshold: float) -> dict[str, float | int | str]:
    return {
        "variant": f"activation_{threshold:.2f}",
        "track_activation_threshold": threshold,
        "effective_new_track_min_confidence": threshold + 0.1,
        "mean_stable_occupancy": 1.0,
        "max_stable_occupancy": 2,
        "occupied_frame_ratio": 0.5,
        "unique_track_ids": 3,
        "count_transition_count": 2,
        "occupied_episode_count": 1,
    }


def test_compare_tracker_sweep_writes_all_pairwise_rows(tmp_path: Path) -> None:
    counts = {
        "activation_0_20": [0, 1, 2],
        "activation_0_30": [0, 1, 1],
        "activation_0_40": [0, 0, 1],
    }
    frame_paths: dict[str, Path] = {}
    for name, values in counts.items():
        path = tmp_path / f"{name}.csv"
        _write_frames(path, values)
        frame_paths[name] = path

    outputs = compare_tracker_sweep(
        run_summaries={
            "activation_0_20": _summary(0.2),
            "activation_0_30": _summary(0.3),
            "activation_0_40": _summary(0.4),
        },
        frame_csv_paths=frame_paths,
        output_dir=tmp_path / "comparison",
    )

    with outputs["pairwise_agreement_csv"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["common_frames"] == "3"
    assert outputs["sweep_summary_csv"].exists()
    assert outputs["sweep_report"].exists()


def test_activation_threshold_validation() -> None:
    assert _normalize_activation_thresholds([0.2, 0.3]) == (0.2, 0.3)
    with pytest.raises(ValueError):
        _normalize_activation_thresholds([0.2])
    with pytest.raises(ValueError):
        _normalize_activation_thresholds([0.2, 0.2])
    with pytest.raises(ValueError):
        _normalize_activation_thresholds([0.1, 0.2])
