import csv
from pathlib import Path

from thermal_occupancy.evaluation.comparison import compare_model_runs
from thermal_occupancy.output.csv_sink import FrameCsvSink


def _write_frames(path: Path, counts: list[int]) -> None:
    with FrameCsvSink(path) as sink:
        for frame_index, count in enumerate(counts):
            sink.write(
                {
                    "frame_index": frame_index,
                    "occupancy_count": count,
                }
            )


def test_comparison_writes_model_and_agreement_outputs(tmp_path: Path) -> None:
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    _write_frames(left_path, [0, 1, 2])
    _write_frames(right_path, [0, 1, 1])

    outputs = compare_model_runs(
        run_summaries={
            "yolo11s": {
                "model": "yolo11s",
                "mean_inference_ms": 10,
                "end_to_end_fps": 20,
                "mean_stable_occupancy": 1,
                "max_stable_occupancy": 2,
                "unique_track_ids": 2,
            },
            "rtdetrv2_s": {
                "model": "rtdetrv2_s",
                "mean_inference_ms": 20,
                "end_to_end_fps": 10,
                "mean_stable_occupancy": 0.67,
                "max_stable_occupancy": 1,
                "unique_track_ids": 1,
            },
        },
        frame_csv_paths={
            "yolo11s": left_path,
            "rtdetrv2_s": right_path,
        },
        output_dir=tmp_path / "comparison",
    )

    with outputs["agreement_csv"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        agreement = next(csv.DictReader(handle))
    assert agreement["common_frames"] == "3"
    assert agreement["count_mae"] == "0.333333"
    assert outputs["model_summary_csv"].exists()
    assert outputs["comparison_report"].exists()

