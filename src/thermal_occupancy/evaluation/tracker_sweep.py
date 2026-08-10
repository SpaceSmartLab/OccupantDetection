from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..output.csv_sink import write_csv_rows


def compare_tracker_sweep(
    *,
    run_summaries: Mapping[str, Mapping[str, Any]],
    frame_csv_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Write one sweep summary and all pairwise count-agreement results."""
    if len(run_summaries) < 2:
        raise ValueError("A tracker sweep requires at least two variants")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_counts = {
        name: _read_frame_counts(path) for name, path in frame_csv_paths.items()
    }
    pairwise_rows: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(run_summaries, 2):
        left = frame_counts[left_name]
        right = frame_counts[right_name]
        common_frames = sorted(set(left) & set(right))
        if not common_frames:
            raise RuntimeError(
                f"No common frames for {left_name} and {right_name}"
            )
        left_counts = np.asarray(
            [left[index] for index in common_frames], dtype=np.float64
        )
        right_counts = np.asarray(
            [right[index] for index in common_frames], dtype=np.float64
        )
        difference = left_counts - right_counts
        pairwise_rows.append(
            {
                "variant_a": left_name,
                "variant_b": right_name,
                "common_frames": len(common_frames),
                "count_mae": _round(np.mean(np.abs(difference))),
                "count_rmse": _round(np.sqrt(np.mean(difference**2))),
                "mean_count_difference_a_minus_b": _round(
                    np.mean(difference)
                ),
                "exact_count_agreement_ratio": _round(
                    np.mean(left_counts == right_counts)
                ),
                "occupied_state_agreement_ratio": _round(
                    np.mean((left_counts > 0) == (right_counts > 0))
                ),
                "occupancy_count_correlation": _correlation(
                    left_counts, right_counts
                ),
                "note": "Agreement is not ground-truth accuracy.",
            }
        )

    summary_path = output_dir / "sweep_summary.csv"
    pairwise_path = output_dir / "pairwise_agreement.csv"
    report_path = output_dir / "sweep_report.md"
    write_csv_rows(summary_path, run_summaries.values())
    write_csv_rows(pairwise_path, pairwise_rows)
    report_path.write_text(
        _render_report(run_summaries, pairwise_rows), encoding="utf-8"
    )
    return {
        "sweep_summary_csv": summary_path,
        "pairwise_agreement_csv": pairwise_path,
        "sweep_report": report_path,
    }


def _read_frame_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[int(row["frame_index"])] = int(row["occupancy_count"])
    return counts


def _round(value: float | np.floating[Any]) -> float:
    return round(float(value), 6)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | str:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return "undefined"
    value = float(np.corrcoef(left, right)[0, 1])
    return round(value, 6) if math.isfinite(value) else "undefined"


def _render_report(
    summaries: Mapping[str, Mapping[str, Any]],
    pairwise_rows: list[Mapping[str, Any]],
) -> str:
    lines = [
        "# Experiment 3 tracker activation sweep",
        "",
        "YOLO detections are shared across all variants at a detector minimum "
        "confidence of 0.10. The background gate is disabled.",
        "",
        "| Variant | ByteTrack activation | Effective new-track minimum | "
        "Mean occupancy | Max | Occupied ratio | Unique IDs | Count transitions | "
        "Occupied episodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['track_activation_threshold']} | "
            f"{summary['effective_new_track_min_confidence']} | "
            f"{summary['mean_stable_occupancy']} | "
            f"{summary['max_stable_occupancy']} | "
            f"{summary['occupied_frame_ratio']} | "
            f"{summary['unique_track_ids']} | "
            f"{summary['count_transition_count']} | "
            f"{summary['occupied_episode_count']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise output agreement",
            "",
            "| Variant A | Variant B | Count MAE | Exact agreement | "
            "Occupied agreement |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['variant_a']} | {row['variant_b']} | "
            f"{row['count_mae']} | {row['exact_count_agreement_ratio']} | "
            f"{row['occupied_state_agreement_ratio']} |"
        )
    lines.extend(
        [
            "",
            "> Pairwise agreement compares system outputs only. It does not measure "
            "accuracy without ground-truth occupant labels.",
            "",
        ]
    )
    return "\n".join(lines)
