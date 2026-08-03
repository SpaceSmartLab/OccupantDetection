from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .detectors import create_detector
from .evaluation import RunMetrics, compare_model_runs
from .occupancy import (
    NormalizedPolygonROI,
    OccupancyCounter,
    TemporalOccupancyStabilizer,
)
from .output import AnnotatedVideoWriter, FrameCsvSink, annotate_frame, write_csv_row
from .output.video_writer import NullVideoWriter
from .schema import track_ids
from .trackers import SharedByteTrack


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    total_frames: int
    width: int
    height: int


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    output_dir: Path
    video_path: Path | None
    frame_csv_path: Path
    summary_csv_path: Path
    summary: dict[str, Any]


def run_comparison(config: PipelineConfig) -> tuple[Path, list[ModelRunResult]]:
    run_root = _make_run_root(config)
    results: list[ModelRunResult] = []
    for detector_name in config.detector_names:
        print(f"\n[{detector_name}] pipeline start", flush=True)
        result = run_model(config, detector_name, run_root / detector_name)
        results.append(result)

    comparison_paths = compare_model_runs(
        run_summaries={result.model_name: result.summary for result in results},
        frame_csv_paths={
            result.model_name: result.frame_csv_path for result in results
        },
        output_dir=run_root / "comparison",
    )
    manifest = {
        "input_video": str(config.input_video),
        "models": {
            result.model_name: {
                "output_dir": str(result.output_dir),
                "video": str(result.video_path) if result.video_path else None,
                "frame_csv": str(result.frame_csv_path),
                "summary_csv": str(result.summary_csv_path),
            }
            for result in results
        },
        "comparison": {key: str(path) for key, path in comparison_paths.items()},
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_root, results


def run_model(
    config: PipelineConfig, detector_name: str, output_dir: Path
) -> ModelRunResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m pip install -e ."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(config.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {config.input_video}")
    info = _read_video_info(cv2, cap)

    common_config = config.section("detectors", "common")
    model_config = config.section("detectors", detector_name)
    tracker_config = config.section("tracker")
    roi_config = config.section("occupancy", "roi")
    temporal_config = config.section("occupancy", "temporal")
    output_config = config.section("output")
    runtime_config = config.section("runtime")

    model_started = time.perf_counter()
    detector = None
    video_writer: AnnotatedVideoWriter | NullVideoWriter | None = None
    frame_sink: FrameCsvSink | None = None
    try:
        detector = create_detector(
            detector_name, common=common_config, model=model_config
        )
        model_load_sec = time.perf_counter() - model_started
        warmup_iterations = int(runtime_config.get("warmup_iterations", 0))
        if warmup_iterations < 0:
            raise ValueError("runtime.warmup_iterations cannot be negative")
        if warmup_iterations:
            success, warmup_frame = cap.read()
            if not success or warmup_frame is None:
                raise RuntimeError("Could not read a frame for detector warm-up")
            for _ in range(warmup_iterations):
                detector.detect(warmup_frame)
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise RuntimeError("Could not rewind the input video after warm-up")

        tracker = SharedByteTrack(
            frame_rate=info.fps,
            track_activation_threshold=tracker_config[
                "track_activation_threshold"
            ],
            lost_track_buffer=tracker_config["lost_track_buffer"],
            minimum_matching_threshold=tracker_config[
                "minimum_matching_threshold"
            ],
        )
        roi = NormalizedPolygonROI(
            roi_config["polygon"], anchor=roi_config.get("anchor", "bottom_center")
        )
        stabilizer = TemporalOccupancyStabilizer(
            fps=info.fps,
            enter_confirm_sec=temporal_config["enter_confirm_sec"],
            exit_confirm_sec=temporal_config["exit_confirm_sec"],
            state_retention_sec=temporal_config["state_retention_sec"],
        )
        counter = OccupancyCounter(
            roi=roi,
            stabilizer=stabilizer,
            width=info.width,
            height=info.height,
        )

        output_video = output_dir / "annotated.mp4"
        if bool(output_config.get("save_video", True)):
            video_writer = AnnotatedVideoWriter(
                path=output_video,
                codec=str(output_config.get("video_codec", "mp4v")),
                fps=info.fps,
                width=info.width,
                height=info.height,
            )
        else:
            video_writer = NullVideoWriter()
            output_video = None

        frame_csv = output_dir / "frames.csv"
        frame_sink = FrameCsvSink(frame_csv)
        metrics = RunMetrics()
        max_frames_value = runtime_config.get("max_frames")
        max_frames = int(max_frames_value) if max_frames_value is not None else None
        progress_every = int(runtime_config.get("progress_every_frames", 500))
        frame_index = 0
        run_started = time.perf_counter()

        while max_frames is None or frame_index < max_frames:
            success, frame = cap.read()
            if not success or frame is None:
                break

            processing_started = time.perf_counter()
            inference_started = time.perf_counter()
            detections = detector.detect(frame)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            tracks = tracker.update(detections)
            occupancy = counter.update(frame_index, tracks)
            processing_ms = (time.perf_counter() - processing_started) * 1000.0

            timestamp_sec = frame_index / info.fps
            timestamp_hms = format_hms(timestamp_sec)
            frame_sink.write(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": f"{timestamp_sec:.6f}",
                    "timestamp_hms": timestamp_hms,
                    "detection_count": len(detections),
                    "tracked_count": len(tracks),
                    "track_ids": _ids_text(track_ids(tracks)),
                    "raw_occupancy_count": occupancy.raw_count,
                    "raw_occupancy_track_ids": _ids_text(
                        occupancy.raw_track_ids
                    ),
                    "occupancy_count": occupancy.stable_count,
                    "occupancy_track_ids": _ids_text(
                        occupancy.stable_track_ids
                    ),
                    "occupied": int(occupancy.stable_count > 0),
                    "inference_ms": f"{inference_ms:.6f}",
                    "processing_ms": f"{processing_ms:.6f}",
                }
            )

            annotated = annotate_frame(
                frame,
                model_name=detector_name,
                timestamp_hms=timestamp_hms,
                tracks=tracks,
                occupancy=occupancy,
                roi=roi,
                draw_confidence=bool(
                    output_config.get("draw_confidence", True)
                ),
            )
            video_writer.write(annotated)
            metrics.add(
                inference_ms=inference_ms,
                processing_ms=processing_ms,
                detection_count=len(detections),
                tracks=tracks,
                occupancy=occupancy,
            )
            frame_index += 1

            if progress_every > 0 and frame_index % progress_every == 0:
                _print_progress(detector_name, frame_index, info.total_frames, run_started)

        wall_time_sec = time.perf_counter() - run_started
        if frame_index == 0:
            raise RuntimeError("No frames were processed from the input video")

        summary = metrics.summary(
            model_name=detector_name,
            checkpoint=str(model_config["checkpoint"]),
            input_video=config.input_video,
            input_fps=info.fps,
            input_width=info.width,
            input_height=info.height,
            total_input_frames=info.total_frames,
            frames_processed=frame_index,
            warmup_iterations=warmup_iterations,
            model_load_sec=model_load_sec,
            wall_time_sec=wall_time_sec,
            tracker_settings=tracker.settings,
        )
        summary_csv = output_dir / "summary.csv"
        write_csv_row(summary_csv, summary)
        print(
            f"[{detector_name}] complete: {frame_index:,} frames, "
            f"{summary['end_to_end_fps']:.2f} FPS",
            flush=True,
        )
        return ModelRunResult(
            model_name=detector_name,
            output_dir=output_dir,
            video_path=output_video,
            frame_csv_path=frame_csv,
            summary_csv_path=summary_csv,
            summary=summary,
        )
    finally:
        cap.release()
        if frame_sink is not None:
            frame_sink.close()
        if video_writer is not None:
            video_writer.close()
        if detector is not None:
            detector.close()


def inspect_video(path: Path) -> VideoInfo:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to inspect a video.") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {path}")
        return _read_video_info(cv2, cap)
    finally:
        cap.release()


def _read_video_info(cv2: Any, cap: Any) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video metadata: fps={fps}, width={width}, height={height}"
        )
    return VideoInfo(
        fps=fps, total_frames=total_frames, width=width, height=height
    )


def _make_run_root(config: PipelineConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = config.output_root / f"{config.input_video.stem}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _ids_text(ids: tuple[int, ...]) -> str:
    return ";".join(str(track_id) for track_id in ids)


def format_hms(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _print_progress(
    model_name: str, frame_index: int, total_frames: int, started: float
) -> None:
    elapsed = time.perf_counter() - started
    fps = frame_index / elapsed if elapsed > 0 else 0.0
    if total_frames > 0:
        percent = frame_index / total_frames * 100
        progress = f"{frame_index:,}/{total_frames:,} ({percent:.2f}%)"
    else:
        progress = f"{frame_index:,}"
    print(
        f"[{model_name}] {progress} | end-to-end {fps:.2f} FPS",
        flush=True,
    )
