# Example Commands

The `football_video_analyser` CLI exposes several utilities. Use the module form when running via `uv` or an activated virtualenv:

```bash
uv run python -m football_video_analyser <subcommand> [options]
```

## Video Library Metadata

Scan match libraries, filter out short clips, and optionally upsert metadata into PostgreSQL:

```bash
uv run python -m football_video_analyser video-index \
  --root "/mnt/e/Recordings/Old Wilsonians/U9 Forbes" \
  --root "/mnt/e/Recordings/Old Wilsonians/U10 Bradbury" \
  --output data/video-index.json \
  --min-duration-minutes 40 \
  --save-to-db
```

## Detection + Tracking

Run YOLOv8 + ByteTrack on a match, writing detections to CSV/Parquet:

```bash
uv run python -m football_video_analyser detect \
  "/mnt/e/Recordings/Old Wilsonians/U9 Forbes/20250518 - Orpington FC.mp4" \
  --device cuda:0 \
  --weights yolov8x.pt \
  --output data/detections-20250518.csv
```

Convert detections to Parquet (faster for slicing):

```bash
uv run python -m football_video_analyser detections-convert \
  data/detections-20250518.csv \
  --output data/detections-20250518.parquet
```

Render an overlay clip with bounding boxes:

```bash
uv run python -m football_video_analyser detections-overlay \
  "/mnt/e/Recordings/Old Wilsonians/U9 Forbes/20250518 - Orpington FC.mp4" \
  data/detections-20250518.parquet \
  --duration-seconds 120 \
  --output data/overlay-preview.mp4
```

## YOLO Export for Annotation Tools

Generate images and YOLO label files for CVAT/Labelbox:

```bash
uv run python -m football_video_analyser detections-export-yolo \
  "/mnt/e/Recordings/Old Wilsonians/U9 Forbes/20250518 - Orpington FC.mp4" \
  data/detections-20250518.parquet \
  --output exports/20250518_yolo \
  --class person=0 \
  --class "sports ball"=1
```

This writes `exports/20250518_yolo.zip`, containing `images/`, `labels/`, `obj.names`, and `obj.data` files ready for CVAT/Labelbox import. Add `--no-zip` to keep the raw directory, or use `--start-frame`/`--end-frame` and additional `--class` flags to tailor the export.

## Useful Extras

- Run tests locally:
  ```bash
  uv run pytest
  ```
- Convert detections to Parquet and render overlays with progress bars by default.
- When targeting CPU instead of GPU, change `--device` to `cpu` on the `detect` command.
