# `sample.py` review and redesign notes

The legacy script is a useful first detector-only baseline, but it cannot answer
the research question fairly because it couples one YOLO model, plotting,
counting, and file output in a single loop.

## What was retained

- Sequential frame processing with source FPS and resolution preserved
- Person-only COCO class filtering
- Annotated MP4 output
- Periodic progress reporting

## What was changed

- `YOLOv8n` was replaced by interchangeable `YOLO11sDetector` and
  `RTDETRv2SDetector` adapters.
- Every detector returns the same `Nx6` array:
  `[x1, y1, x2, y2, score, class_id]`.
- Raw detections are no longer treated as occupants. One shared ByteTrack assigns
  persistent IDs first.
- Occupancy uses a configurable normalized polygon and a track anchor point.
- Entry/exit debounce prevents one-frame detection flicker from changing the
  occupancy state.
- Per-frame rows, per-model summary rows, annotated videos, and cross-model
  agreement outputs are written by dedicated output/evaluation modules.
- Paths and thresholds moved out of code and into YAML/CLI settings.

## Research interpretation

Without labeled ground truth, differences between the two pipelines are
*agreement and runtime* differences, not accuracy differences. A later evaluation
stage should add manually labeled thermal frames/tracks before reporting
precision, recall, mAP, HOTA, IDF1, or MOTA.

