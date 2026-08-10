from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .background_gate import EmptyBackgroundGate
from .config import PipelineConfig
from .detectors import create_detector
from .evaluation import RunMetrics, compare_model_runs, compare_tracker_sweep
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


@dataclass
class _TrackerSweepVariant:
    name: str
    activation_threshold: float
    output_dir: Path
    tracker: SharedByteTrack
    counter: OccupancyCounter
    video_writer: AnnotatedVideoWriter | NullVideoWriter
    video_path: Path | None
    frame_sink: FrameCsvSink
    metrics: RunMetrics = field(default_factory=RunMetrics)
    previous_count: int | None = None
    count_transition_count: int = 0
    occupied_episode_count: int = 0


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
    (output_dir / "effective_config.json").write_text(
        json.dumps(config.values, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cap = cv2.VideoCapture(str(config.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {config.input_video}")
    info = _read_video_info(cv2, cap)

    common_config = config.section("detectors", "common")
    model_config = config.section("detectors", detector_name)
    tracker_config = config.section("tracker")
    roi_config = config.section("occupancy", "roi")
    temporal_config = config.section("occupancy", "temporal")
    background_config = dict(
        config.values["occupancy"].get("background_gate", {})
    )
    output_config = config.section("output")
    runtime_config = config.section("runtime")

    model_started = time.perf_counter()
    detector = None
    background_gate: EmptyBackgroundGate | None = None
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

        if bool(background_config.get("enabled", False)):
            initialization_frames = int(
                background_config.get("initialization_frames", 50)
            )
            empty_frames = _read_initial_background_frames(
                cv2, cap, frame_count=initialization_frames
            )
            background_gate = EmptyBackgroundGate.from_frames(
                empty_frames,
                pixel_difference_threshold=int(
                    background_config.get("pixel_difference_threshold", 25)
                ),
                minimum_changed_pixel_ratio=float(
                    background_config.get("minimum_changed_pixel_ratio", 0.2)
                ),
                approved_track_iou_threshold=float(
                    background_config.get("approved_track_iou_threshold", 0.3)
                ),
                blur_kernel_size=int(background_config.get("blur_kernel_size", 5)),
            )
            cv2.imwrite(
                str(output_dir / "empty_background_reference.png"),
                background_gate.reference_gray,
            )

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
        max_consecutive_decode_errors = int(
            runtime_config.get("max_consecutive_decode_errors", 3)
        )
        max_total_decode_errors = int(
            runtime_config.get("max_total_decode_errors", 100)
        )
        if max_consecutive_decode_errors <= 0 or max_total_decode_errors <= 0:
            raise ValueError("Decode error limits must be greater than zero")

        frames_processed = 0
        next_source_frame_index = 0
        consecutive_decode_errors = 0
        decode_error_indices: list[int] = []
        raw_detection_total = 0
        background_rejected_total = 0
        background_retained_total = 0
        approved_tracks: dict[int, tuple[Any, int]] = {}
        run_started = time.perf_counter()

        while max_frames is None or frames_processed < max_frames:
            if (
                info.total_frames > 0
                and next_source_frame_index >= info.total_frames
            ):
                break
            success, frame = cap.read()
            if not success or frame is None:
                failed_index = next_source_frame_index
                decode_error_indices.append(failed_index)
                consecutive_decode_errors += 1
                next_source_frame_index += 1
                print(
                    f"[{detector_name}] decode error at source frame "
                    f"{failed_index:,}; skipping and continuing",
                    flush=True,
                )
                if (
                    consecutive_decode_errors > max_consecutive_decode_errors
                    or len(decode_error_indices) > max_total_decode_errors
                ):
                    raise RuntimeError(
                        "Input decoding could not recover within the configured "
                        "error limits"
                    )
                continue

            frame_index = _capture_source_frame_index(
                cv2,
                cap,
                fps=info.fps,
                fallback=next_source_frame_index,
            )
            if frame_index > next_source_frame_index:
                decode_error_indices.extend(
                    range(next_source_frame_index, frame_index)
                )
            next_source_frame_index = frame_index + 1
            consecutive_decode_errors = 0

            processing_started = time.perf_counter()
            inference_started = time.perf_counter()
            raw_detections = detector.detect(frame)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0

            raw_detection_total += len(raw_detections)
            background_rejected_count = 0
            background_retained_count = 0
            detections = raw_detections
            if background_gate is not None:
                approved_tracks = {
                    track_id: value
                    for track_id, value in approved_tracks.items()
                    if frame_index - value[1]
                    <= int(tracker_config["lost_track_buffer"])
                }
                gate_result = background_gate.filter(
                    frame,
                    raw_detections,
                    approved_boxes=(
                        value[0].xyxy for value in approved_tracks.values()
                    ),
                )
                detections = gate_result.detections
                background_rejected_count = gate_result.rejected_count
                background_retained_count = gate_result.retained_by_track_count
                background_rejected_total += background_rejected_count
                background_retained_total += background_retained_count

            tracks = tracker.update(detections)
            for track in tracks:
                approved_tracks[track.track_id] = (track, frame_index)
            occupancy = counter.update(frame_index, tracks)
            processing_ms = (time.perf_counter() - processing_started) * 1000.0

            timestamp_sec = frame_index / info.fps
            timestamp_hms = format_hms(timestamp_sec)
            frame_sink.write(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": f"{timestamp_sec:.6f}",
                    "timestamp_hms": timestamp_hms,
                    "raw_detection_count": len(raw_detections),
                    "background_rejected_count": background_rejected_count,
                    "background_retained_count": background_retained_count,
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
            frames_processed += 1

            if progress_every > 0 and frames_processed % progress_every == 0:
                _print_progress(
                    detector_name,
                    next_source_frame_index,
                    info.total_frames,
                    run_started,
                )

        wall_time_sec = time.perf_counter() - run_started
        if frames_processed == 0:
            raise RuntimeError("No frames were processed from the input video")

        summary = metrics.summary(
            model_name=detector_name,
            checkpoint=str(model_config["checkpoint"]),
            input_video=config.input_video,
            input_fps=info.fps,
            input_width=info.width,
            input_height=info.height,
            total_input_frames=info.total_frames,
            frames_processed=frames_processed,
            warmup_iterations=warmup_iterations,
            model_load_sec=model_load_sec,
            wall_time_sec=wall_time_sec,
            tracker_settings=tracker.settings,
        )
        background_settings: dict[str, Any] = {
            "enabled": background_gate is not None
        }
        if background_gate is not None:
            background_settings.update(background_gate.settings)
            background_settings["initialization_frames"] = int(
                background_config.get("initialization_frames", 50)
            )
        requested_scope_complete = (
            max_frames is not None and frames_processed >= max_frames
        ) or (
            max_frames is None
            and (
                info.total_frames <= 0
                or next_source_frame_index >= info.total_frames
            )
        )
        summary.update(
            {
                "confidence_threshold": float(
                    common_config["confidence_threshold"]
                ),
                "roi_polygon": json.dumps(roi_config["polygon"]),
                "enter_confirm_sec": float(
                    temporal_config["enter_confirm_sec"]
                ),
                "background_gate_settings": ";".join(
                    f"{key}={value}" for key, value in background_settings.items()
                ),
                "raw_person_detections": raw_detection_total,
                "background_rejected_detections": background_rejected_total,
                "background_retained_detections": background_retained_total,
                "decode_error_count": len(set(decode_error_indices)),
                "decode_error_frame_indices": ";".join(
                    str(index) for index in sorted(set(decode_error_indices))
                ),
                "source_frames_covered": min(
                    next_source_frame_index, info.total_frames
                )
                if info.total_frames > 0
                else next_source_frame_index,
                "completion_status": (
                    "complete" if requested_scope_complete else "partial"
                ),
            }
        )
        summary_csv = output_dir / "summary.csv"
        write_csv_row(summary_csv, summary)
        print(
            f"[{detector_name}] {summary['completion_status']}: "
            f"{frames_processed:,} decoded frames, "
            f"{summary['decode_error_count']} skipped, "
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


def run_tracker_activation_sweep(
    config: PipelineConfig,
    detector_name: str,
    *,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[Path, list[ModelRunResult]]:
    """Run detector inference once and fan the detections out to tracker variants."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m pip install -e ."
        ) from exc

    activation_thresholds = _normalize_activation_thresholds(thresholds)
    common_config = config.section("detectors", "common")
    detector_min_confidence = float(common_config["confidence_threshold"])
    if detector_min_confidence > 0.1:
        raise ValueError(
            "Tracker sweeps must set detectors.common.confidence_threshold to "
            "0.10 or lower so ByteTrack receives its low-score association band"
        )
    background_config = dict(
        config.values["occupancy"].get("background_gate", {})
    )
    if bool(background_config.get("enabled", False)):
        raise ValueError(
            "Tracker activation sweeps require occupancy.background_gate.enabled=false"
        )

    run_root = _make_tracker_sweep_root(config)
    cap = cv2.VideoCapture(str(config.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {config.input_video}")
    info = _read_video_info(cv2, cap)

    model_config = config.section("detectors", detector_name)
    tracker_config = config.section("tracker")
    roi_config = config.section("occupancy", "roi")
    temporal_config = config.section("occupancy", "temporal")
    output_config = config.section("output")
    runtime_config = config.section("runtime")

    detector = None
    variants: list[_TrackerSweepVariant] = []
    try:
        model_started = time.perf_counter()
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

        for threshold in activation_thresholds:
            name = _activation_variant_name(threshold)
            output_dir = run_root / name
            output_dir.mkdir(parents=True, exist_ok=False)
            effective_config = deepcopy(config.values)
            effective_config["tracker"]["track_activation_threshold"] = threshold
            effective_config.setdefault("experiment", {})["active_variant"] = name
            effective_config["experiment"][
                "effective_new_track_min_confidence"
            ] = round(threshold + 0.1, 6)
            (output_dir / "effective_config.json").write_text(
                json.dumps(effective_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            tracker = SharedByteTrack(
                frame_rate=info.fps,
                track_activation_threshold=threshold,
                lost_track_buffer=tracker_config["lost_track_buffer"],
                minimum_matching_threshold=tracker_config[
                    "minimum_matching_threshold"
                ],
            )
            roi = NormalizedPolygonROI(
                roi_config["polygon"],
                anchor=roi_config.get("anchor", "bottom_center"),
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

            video_path: Path | None = output_dir / "annotated.mp4"
            if bool(output_config.get("save_video", True)):
                video_writer: AnnotatedVideoWriter | NullVideoWriter = (
                    AnnotatedVideoWriter(
                        path=video_path,
                        codec=str(output_config.get("video_codec", "mp4v")),
                        fps=info.fps,
                        width=info.width,
                        height=info.height,
                    )
                )
            else:
                video_writer = NullVideoWriter()
                video_path = None

            variants.append(
                _TrackerSweepVariant(
                    name=name,
                    activation_threshold=threshold,
                    output_dir=output_dir,
                    tracker=tracker,
                    counter=counter,
                    video_writer=video_writer,
                    video_path=video_path,
                    frame_sink=FrameCsvSink(output_dir / "frames.csv"),
                )
            )

        max_frames_value = runtime_config.get("max_frames")
        max_frames = int(max_frames_value) if max_frames_value is not None else None
        progress_every = int(runtime_config.get("progress_every_frames", 500))
        max_consecutive_decode_errors = int(
            runtime_config.get("max_consecutive_decode_errors", 3)
        )
        max_total_decode_errors = int(
            runtime_config.get("max_total_decode_errors", 100)
        )
        if max_consecutive_decode_errors <= 0 or max_total_decode_errors <= 0:
            raise ValueError("Decode error limits must be greater than zero")

        frames_processed = 0
        next_source_frame_index = 0
        consecutive_decode_errors = 0
        decode_error_indices: list[int] = []
        raw_detection_total = 0
        run_started = time.perf_counter()

        while max_frames is None or frames_processed < max_frames:
            if info.total_frames > 0 and next_source_frame_index >= info.total_frames:
                break
            success, frame = cap.read()
            if not success or frame is None:
                failed_index = next_source_frame_index
                decode_error_indices.append(failed_index)
                consecutive_decode_errors += 1
                next_source_frame_index += 1
                print(
                    f"[{detector_name} sweep] decode error at source frame "
                    f"{failed_index:,}; skipping and continuing",
                    flush=True,
                )
                if (
                    consecutive_decode_errors > max_consecutive_decode_errors
                    or len(decode_error_indices) > max_total_decode_errors
                ):
                    raise RuntimeError(
                        "Input decoding could not recover within the configured "
                        "error limits"
                    )
                continue

            frame_index = _capture_source_frame_index(
                cv2,
                cap,
                fps=info.fps,
                fallback=next_source_frame_index,
            )
            if frame_index > next_source_frame_index:
                decode_error_indices.extend(
                    range(next_source_frame_index, frame_index)
                )
            next_source_frame_index = frame_index + 1
            consecutive_decode_errors = 0

            inference_started = time.perf_counter()
            detections = detector.detect(frame)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            raw_detection_total += len(detections)
            timestamp_sec = frame_index / info.fps
            timestamp_hms = format_hms(timestamp_sec)

            for variant in variants:
                variant_started = time.perf_counter()
                tracks = variant.tracker.update(detections)
                occupancy = variant.counter.update(frame_index, tracks)
                variant_ms = (time.perf_counter() - variant_started) * 1000.0
                processing_ms = inference_ms + variant_ms

                if (
                    variant.previous_count is not None
                    and occupancy.stable_count != variant.previous_count
                ):
                    variant.count_transition_count += 1
                if (
                    occupancy.stable_count > 0
                    and (variant.previous_count is None or variant.previous_count == 0)
                ):
                    variant.occupied_episode_count += 1
                variant.previous_count = occupancy.stable_count

                variant.frame_sink.write(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.6f}",
                        "timestamp_hms": timestamp_hms,
                        "raw_detection_count": len(detections),
                        "background_rejected_count": 0,
                        "background_retained_count": 0,
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
                    model_name=f"{detector_name} act={variant.activation_threshold:.2f}",
                    timestamp_hms=timestamp_hms,
                    tracks=tracks,
                    occupancy=occupancy,
                    roi=variant.counter.roi,
                    draw_confidence=bool(
                        output_config.get("draw_confidence", True)
                    ),
                )
                variant.video_writer.write(annotated)
                variant.metrics.add(
                    inference_ms=inference_ms,
                    processing_ms=processing_ms,
                    detection_count=len(detections),
                    tracks=tracks,
                    occupancy=occupancy,
                )

            frames_processed += 1
            if progress_every > 0 and frames_processed % progress_every == 0:
                _print_progress(
                    f"{detector_name} sweep",
                    next_source_frame_index,
                    info.total_frames,
                    run_started,
                )

        wall_time_sec = time.perf_counter() - run_started
        if frames_processed == 0:
            raise RuntimeError("No frames were processed from the input video")

        for variant in variants:
            variant.frame_sink.close()
            variant.video_writer.close()

        requested_scope_complete = (
            max_frames is not None and frames_processed >= max_frames
        ) or (
            max_frames is None
            and (
                info.total_frames <= 0
                or next_source_frame_index >= info.total_frames
            )
        )
        results: list[ModelRunResult] = []
        for variant in variants:
            summary = variant.metrics.summary(
                model_name=detector_name,
                checkpoint=str(model_config["checkpoint"]),
                input_video=config.input_video,
                input_fps=info.fps,
                input_width=info.width,
                input_height=info.height,
                total_input_frames=info.total_frames,
                frames_processed=frames_processed,
                warmup_iterations=warmup_iterations,
                model_load_sec=model_load_sec,
                wall_time_sec=wall_time_sec,
                tracker_settings=variant.tracker.settings,
            )
            summary.update(
                {
                    "variant": variant.name,
                    "detector_min_confidence": detector_min_confidence,
                    "confidence_threshold": detector_min_confidence,
                    "track_activation_threshold": variant.activation_threshold,
                    "effective_new_track_min_confidence": round(
                        variant.activation_threshold + 0.1, 6
                    ),
                    "roi_polygon": json.dumps(roi_config["polygon"]),
                    "enter_confirm_sec": float(
                        temporal_config["enter_confirm_sec"]
                    ),
                    "exit_confirm_sec": float(
                        temporal_config["exit_confirm_sec"]
                    ),
                    "state_retention_sec": float(
                        temporal_config["state_retention_sec"]
                    ),
                    "background_gate_settings": "enabled=False",
                    "raw_person_detections": raw_detection_total,
                    "background_rejected_detections": 0,
                    "background_retained_detections": 0,
                    "count_transition_count": variant.count_transition_count,
                    "occupied_episode_count": variant.occupied_episode_count,
                    "decode_error_count": len(set(decode_error_indices)),
                    "decode_error_frame_indices": ";".join(
                        str(index)
                        for index in sorted(set(decode_error_indices))
                    ),
                    "source_frames_covered": min(
                        next_source_frame_index, info.total_frames
                    )
                    if info.total_frames > 0
                    else next_source_frame_index,
                    "completion_status": (
                        "complete" if requested_scope_complete else "partial"
                    ),
                    "shared_detector_sweep": True,
                }
            )
            summary_csv = variant.output_dir / "summary.csv"
            write_csv_row(summary_csv, summary)
            results.append(
                ModelRunResult(
                    model_name=variant.name,
                    output_dir=variant.output_dir,
                    video_path=variant.video_path,
                    frame_csv_path=variant.output_dir / "frames.csv",
                    summary_csv_path=summary_csv,
                    summary=summary,
                )
            )

        comparison_paths = compare_tracker_sweep(
            run_summaries={result.model_name: result.summary for result in results},
            frame_csv_paths={
                result.model_name: result.frame_csv_path for result in results
            },
            output_dir=run_root / "comparison",
        )
        manifest = {
            "experiment": "tracker_activation_sweep",
            "input_video": str(config.input_video),
            "detector": detector_name,
            "detector_min_confidence": detector_min_confidence,
            "background_gate_enabled": False,
            "variants": {
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
        print(
            f"[{detector_name} sweep] complete: {frames_processed:,} decoded "
            f"frames, {len(set(decode_error_indices))} skipped",
            flush=True,
        )
        return run_root, results
    finally:
        cap.release()
        for variant in variants:
            variant.frame_sink.close()
            variant.video_writer.close()
        if detector is not None:
            detector.close()


def _normalize_activation_thresholds(
    thresholds: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in thresholds)
    if len(normalized) < 2:
        raise ValueError("At least two tracker activation thresholds are required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Tracker activation thresholds must be unique")
    if any(not 0.1 < value < 0.9 for value in normalized):
        raise ValueError(
            "Tracker activation thresholds must be greater than 0.1 and less than 0.9"
        )
    names = [_activation_variant_name(value) for value in normalized]
    if len(set(names)) != len(names):
        raise ValueError("Tracker activation thresholds must differ at 0.01 precision")
    return normalized


def _activation_variant_name(threshold: float) -> str:
    return f"activation_{float(threshold):.2f}".replace(".", "_")


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


def _read_initial_background_frames(
    cv2: Any, cap: Any, *, frame_count: int
) -> list[Any]:
    if frame_count <= 0:
        raise ValueError("background_gate.initialization_frames must be positive")
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise RuntimeError("Could not seek to the start for background initialization")
    frames: list[Any] = []
    for index in range(frame_count):
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError(
                "Could not read empty-scene background frame " f"{index}"
            )
        frames.append(frame)
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise RuntimeError("Could not rewind after background initialization")
    return frames


def _capture_source_frame_index(
    cv2: Any, cap: Any, *, fps: float, fallback: int
) -> int:
    """Use decoded presentation time when available, otherwise expected order."""
    timestamp_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_msec < 0:
        return fallback
    candidate = int(round(timestamp_msec * fps / 1000.0))
    return candidate if candidate >= fallback else fallback


def _make_run_root(config: PipelineConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = config.output_root / f"{config.input_video.stem}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _make_tracker_sweep_root(config: PipelineConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = config.output_root / f"tracker_sweep_{timestamp}"
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
