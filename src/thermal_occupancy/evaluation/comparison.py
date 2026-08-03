from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..output.csv_sink import write_csv_rows


def compare_model_runs(
    *,
    run_summaries: Mapping[str, Mapping[str, Any]],
    frame_csv_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Path]:
    if len(run_summaries) < 2:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(run_summaries)
    left_name, right_name = model_names[:2]
    left = _read_frame_counts(frame_csv_paths[left_name])
    right = _read_frame_counts(frame_csv_paths[right_name])
    common_frames = sorted(set(left) & set(right))
    if not common_frames:
        raise RuntimeError("No common frame rows are available for model comparison")

    left_counts = np.asarray([left[index] for index in common_frames], dtype=float)
    right_counts = np.asarray([right[index] for index in common_frames], dtype=float)
    difference = left_counts - right_counts
    correlation = _correlation(left_counts, right_counts)

    agreement = {
        "model_a": left_name,
        "model_b": right_name,
        "common_frames": len(common_frames),
        "count_mae": round(float(np.mean(np.abs(difference))), 6),
        "count_rmse": round(float(np.sqrt(np.mean(difference**2))), 6),
        "mean_count_difference_a_minus_b": round(float(difference.mean()), 6),
        "exact_count_agreement_ratio": round(
            float(np.mean(left_counts == right_counts)), 6
        ),
        "occupied_state_agreement_ratio": round(
            float(np.mean((left_counts > 0) == (right_counts > 0))), 6
        ),
        "occupancy_count_correlation": correlation,
        "note": (
            "Agreement is not ground-truth accuracy. Add labeled thermal data "
            "for precision/recall/mAP/MOT metrics."
        ),
    }

    model_summary_path = output_dir / "model_summary.csv"
    agreement_path = output_dir / "agreement.csv"
    report_path = output_dir / "comparison_report.md"
    write_csv_rows(model_summary_path, run_summaries.values())
    write_csv_rows(agreement_path, [agreement])
    report_path.write_text(
        _render_report(run_summaries, agreement), encoding="utf-8"
    )
    return {
        "model_summary_csv": model_summary_path,
        "agreement_csv": agreement_path,
        "comparison_report": report_path,
    }


def _read_frame_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[int(row["frame_index"])] = int(row["occupancy_count"])
    return counts


def _correlation(left: np.ndarray, right: np.ndarray) -> float | str:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return "undefined"
    value = float(np.corrcoef(left, right)[0, 1])
    return round(value, 6) if math.isfinite(value) else "undefined"


def _render_report(
    summaries: Mapping[str, Mapping[str, Any]], agreement: Mapping[str, Any]
) -> str:
    lines = [
        "# Detector comparison",
        "",
        "| Model | Mean inference (ms) | End-to-end FPS | Mean occupancy | Max occupancy | Unique IDs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, summary in summaries.items():
        lines.append(
            f"| {model_name} | {summary['mean_inference_ms']} | "
            f"{summary['end_to_end_fps']} | "
            f"{summary['mean_stable_occupancy']} | "
            f"{summary['max_stable_occupancy']} | "
            f"{summary['unique_track_ids']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-model agreement",
            "",
            f"- Common frames: {agreement['common_frames']}",
            f"- Count MAE: {agreement['count_mae']}",
            f"- Exact count agreement: {agreement['exact_count_agreement_ratio']}",
            (
                "- Occupied/unoccupied agreement: "
                f"{agreement['occupied_state_agreement_ratio']}"
            ),
            f"- Count correlation: {agreement['occupancy_count_correlation']}",
            "",
            (
                "> These values compare model outputs with each other; they are not "
                "accuracy metrics. Ground-truth thermal annotations are required for "
                "precision, recall, mAP, HOTA, IDF1, or MOTA."
            ),
            "",
        ]
    )
    return "\n".join(lines)

