from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import (
    inspect_video,
    run_comparison,
    run_model,
    run_tracker_activation_sweep,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-occupancy",
        description=(
            "Compare YOLO11s and RT-DETRv2-S with one shared ByteTrack, "
            "ROI occupancy, and temporal stabilization pipeline."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Run all configured detectors")
    _add_common_arguments(compare)

    run = subparsers.add_parser("run", help="Run one detector")
    _add_common_arguments(run)
    run.add_argument(
        "--detector",
        choices=("yolo11s", "rtdetrv2_s"),
        required=True,
    )

    sweep = subparsers.add_parser(
        "sweep-tracker",
        help="Run one detector once and compare multiple ByteTrack activation thresholds",
    )
    _add_common_arguments(sweep)
    sweep.add_argument(
        "--detector",
        choices=("yolo11s", "rtdetrv2_s"),
        default="yolo11s",
    )
    sweep.add_argument(
        "--activation-thresholds",
        type=float,
        nargs="+",
        help=(
            "ByteTrack activation thresholds. Defaults to "
            "experiment.tracker_activation_thresholds from the config."
        ),
    )

    inspect = subparsers.add_parser("inspect", help="Inspect input video metadata")
    inspect.add_argument("--input", type=Path, required=True)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sample.yaml")
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-frames", type=int)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        info = inspect_video(args.input.expanduser().resolve())
        duration = info.total_frames / info.fps
        print(
            f"size={info.width}x{info.height} fps={info.fps:.6f} "
            f"frames={info.total_frames} duration_sec={duration:.3f}"
        )
        return

    detector_name = getattr(args, "detector", None)
    config = load_config(args.config).with_overrides(
        input_video=args.input,
        output_root=args.output_root,
        max_frames=args.max_frames,
        detector_names=[detector_name]
        if args.command in {"run", "sweep-tracker"}
        else None,
    )
    if args.command == "compare":
        run_root, _ = run_comparison(config)
        print(f"\nAll outputs: {run_root}")
        return

    if args.command == "sweep-tracker":
        thresholds = args.activation_thresholds
        if thresholds is None:
            thresholds = config.values.get("experiment", {}).get(
                "tracker_activation_thresholds"
            )
        if thresholds is None:
            raise ValueError(
                "Provide --activation-thresholds or configure "
                "experiment.tracker_activation_thresholds"
            )
        run_root, _ = run_tracker_activation_sweep(
            config,
            detector_name,
            thresholds=thresholds,
        )
        print(f"\nSweep outputs: {run_root}")
        return

    run_root = config.output_root / f"{config.input_video.stem}_{args.detector}"
    result = run_model(config, args.detector, run_root)
    print(f"\nOutputs: {result.output_dir}")
