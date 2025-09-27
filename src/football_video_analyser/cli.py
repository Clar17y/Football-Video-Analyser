"""Command-line interface for local tooling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import cv2

from .annotation import DEFAULT_CLASS_MAP, export_detections_to_yolo, parse_class_mappings
from .data_ingest import build_video_index, metadata_to_dict
from .db import get_session_factory, upsert_match_videos
from .visualisation import (
    convert_detections_to_parquet,
    load_detections,
    render_overlay_segment,
)
from .models.tracking import detections_to_rows, iter_tracked_objects

LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-video-analyser",
        description="Utilities for the Football Video Analyser project",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "video-index",
        help="Scan configured roots and emit video metadata",
    )
    index_parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Additional root directory to search (can be repeated)",
    )
    index_parser.add_argument(
        "--suffix",
        dest="suffixes",
        action="append",
        help="Filter file suffixes (default: common video extensions)",
    )
    index_parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (.json or .csv) for structured metadata",
    )
    index_parser.add_argument(
        "--min-duration-minutes",
        type=float,
        default=40.0,
        help="Minimum duration (minutes) required for inclusion (default: 40)",
    )
    index_parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Persist results into the configured PostgreSQL database",
    )

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run baseline YOLO detection + tracking on a video file",
    )
    detect_parser.add_argument("video", type=Path, help="Video file to process")
    detect_parser.add_argument(
        "--weights",
        default="yolov8x.pt",
        help="Ultralytics YOLO weights to load (default: yolov8x.pt)",
    )
    detect_parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device string (e.g. cuda:0 or cpu)",
    )
    detect_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    detect_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)",
    )
    detect_parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "sports ball"],
        help="Detection class names to retain",
    )
    detect_parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to write detections",
    )

    convert_parser = subparsers.add_parser(
        "detections-convert",
        help="Convert detection CSV outputs into Parquet",
    )
    convert_parser.add_argument("detections", type=Path, help="Input CSV or Parquet")
    convert_parser.add_argument(
        "--output",
        type=Path,
        help="Destination Parquet file (defaults to replacing extension)",
    )
    convert_parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Rows per chunk when converting large CSV files",
    )

    overlay_parser = subparsers.add_parser(
        "detections-overlay",
        help="Render a short clip with detection overlays",
    )
    overlay_parser.add_argument("video", type=Path, help="Video file to read")
    overlay_parser.add_argument("detections", type=Path, help="Detections CSV/Parquet")
    overlay_parser.add_argument(
        "--output",
        type=Path,
        help="Optional MP4 to write with overlays",
    )
    overlay_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to render",
    )
    overlay_parser.add_argument(
        "--duration-frames",
        type=int,
        help="Number of frames to render",
    )
    overlay_parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Approximate duration in seconds (converted using video FPS)",
    )
    overlay_parser.add_argument(
        "--display",
        action="store_true",
        help="Show a real-time preview window while rendering",
    )

    export_parser = subparsers.add_parser(
        "detections-export-yolo",
        help="Convert detections into a YOLO-style dataset for annotation",
    )
    export_parser.add_argument("video", type=Path, help="Source video")
    export_parser.add_argument("detections", type=Path, help="Detections CSV/Parquet")
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for YOLO images+labels",
    )
    export_parser.add_argument(
        "--start-frame",
        type=int,
        help="Optional starting frame",
    )
    export_parser.add_argument(
        "--end-frame",
        type=int,
        help="Optional ending frame (exclusive)",
    )
    export_parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Export every Nth frame (default 1)",
    )
    export_parser.add_argument(
        "--class",
        dest="class_map",
        action="append",
        default=[],
        help="Mapping label=id, e.g. --class person=0 --class 'sports ball'=1",
    )
    export_parser.add_argument(
        "--no-zip",
        dest="zip_output",
        action="store_false",
        help="Do not create a zip archive (default: create .zip)",
    )
    export_parser.set_defaults(zip_output=True)

    return parser


def run_video_index(
    roots: Sequence[Path] | None,
    suffixes: Sequence[str] | None,
    output: Path | None,
    min_duration_minutes: float,
    save_to_db: bool,
) -> int:
    metadata = build_video_index(
        roots=roots,
        suffixes=suffixes,
        min_duration_seconds=min_duration_minutes * 60,
    )
    if not metadata:
        LOGGER.warning("No video files discovered that meet the criteria")
        return 1

    for entry in metadata:
        duration = f"{entry.duration_seconds:.1f}s" if entry.duration_seconds else "?"
        bitrate = f"{entry.bitrate_kbps:.1f} kbps" if entry.bitrate_kbps else "?"
        LOGGER.info(
            "%s | %dx%d @ %.2f fps | %s | %s",
            entry.path,
            entry.width,
            entry.height,
            entry.fps,
            duration,
            bitrate,
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        serialised = [metadata_to_dict(item) for item in metadata]
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(serialised, indent=2))
        elif output.suffix.lower() == ".csv":
            if not serialised:
                output.write_text("")
            else:
                with output.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(serialised[0].keys()))
                    writer.writeheader()
                    writer.writerows(serialised)
        else:
            raise ValueError("Unsupported output format; use .json or .csv")

        LOGGER.info("Wrote metadata for %d videos to %s", len(serialised), output)

    if save_to_db:
        session_factory = get_session_factory()
        with session_factory() as session:
            with session.begin():
                affected = upsert_match_videos(session, metadata)
        LOGGER.info("Upserted %d videos into match_video", affected)

    return 0


def run_detection(
    video: Path,
    *,
    weights: str,
    device: str | None,
    conf: float,
    iou: float,
    classes: Sequence[str],
    output: Path | None,
) -> int:
    if not video.exists():
        LOGGER.error("Video not found: %s", video)
        return 1

    detections = list(
        iter_tracked_objects(
            video_path=video,
            weights=weights,
            device=device,
            conf=conf,
            iou=iou,
            class_names=tuple(classes),
        )
    )

    if not detections:
        LOGGER.warning("No detections found in %s", video)
        return 1

    LOGGER.info(
        "Processed %s | %d detections across %d frames",
        video,
        len(detections),
        detections[-1].frame_index + 1,
    )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = detections_to_rows(detections)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote detections to %s", output)
    else:
        for det in detections[:5]:
            LOGGER.info(
                "Frame %d | %s (track=%s) conf=%.2f bbox=%s",
                det.frame_index,
                det.label,
                det.track_id,
                det.confidence,
                det.bbox_xyxy,
            )

    return 0


def run_detections_convert(
    detections: Path,
    *,
    output: Path | None,
    chunk_rows: int,
) -> int:
    result = convert_detections_to_parquet(
        detections,
        output_path=output,
        chunk_rows=chunk_rows,
        show_progress=True,
    )
    LOGGER.info("Wrote Parquet detections to %s", result)
    return 0


def run_detections_overlay(
    video: Path,
    detections: Path,
    *,
    output: Path | None,
    start_frame: int,
    duration_frames: int | None,
    duration_seconds: float | None,
    display: bool,
) -> int:
    if duration_frames is None and duration_seconds is not None:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        duration_frames = max(1, int(duration_seconds * fps))

    end_frame = None
    if duration_frames is not None:
        end_frame = start_frame + duration_frames

    df = load_detections(detections)

    if df.empty:
        LOGGER.warning("No detections available in the supplied file")
        return 1

    min_frame = int(df["frame_index"].min())
    max_frame = int(df["frame_index"].max())

    if start_frame < min_frame:
        LOGGER.info("Adjusting start frame to first detection: %d", min_frame)
        start_frame = min_frame

    def _video_fps(path: Path) -> float:
        capture = cv2.VideoCapture(str(path))
        fps_val = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        return fps_val

    video_fps = _video_fps(video)

    if duration_frames is not None:
        end_frame = start_frame + duration_frames
    elif duration_seconds is not None:
        duration_frames = max(1, int(duration_seconds * video_fps))
        end_frame = start_frame + duration_frames
    else:
        end_frame = None

    if end_frame is not None and end_frame > max_frame + 1:
        end_frame = max_frame + 1

    df = load_detections(
        detections,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if df.empty:
        LOGGER.warning("No detections within the requested window")
        return 1

    processed = render_overlay_segment(
        video,
        df,
        start_frame=start_frame,
        num_frames=duration_frames,
        output_path=output,
        display=display,
        show_progress=True,
    )

    LOGGER.info(
        "Rendered %d frames starting at frame %d (%.2fs)%s",
        processed,
        start_frame,
        start_frame / video_fps,
        f" to {output}" if output else "",
    )
    return 0


def run_detections_export_yolo(
    video: Path,
    detections: Path,
    *,
    output: Path,
    start_frame: int | None,
    end_frame: int | None,
    frame_step: int,
    class_map_args: list[str],
    zip_output: bool,
) -> int:
    try:
        custom_map = parse_class_mappings(class_map_args)
    except ValueError as exc:
        LOGGER.error(str(exc))
        return 1

    merged_map = {**DEFAULT_CLASS_MAP, **custom_map}
    result = export_detections_to_yolo(
        video_path=video,
        detections_path=detections,
        output_dir=output,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=max(1, frame_step),
        class_map=merged_map,
        show_progress=True,
        zip_output=zip_output,
    )
    LOGGER.info("YOLO export written to %s", result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.command == "video-index":
        roots = args.roots if args.roots else None
        suffixes = args.suffixes if args.suffixes else None
        return run_video_index(
            roots,
            suffixes,
            args.output,
            args.min_duration_minutes,
            args.save_to_db,
        )
    if args.command == "detect":
        return run_detection(
            args.video,
            weights=args.weights,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            output=args.output,
        )
    if args.command == "detections-convert":
        return run_detections_convert(
            args.detections,
            output=args.output,
            chunk_rows=args.chunk_rows,
        )
    if args.command == "detections-overlay":
        return run_detections_overlay(
            args.video,
            args.detections,
            output=args.output,
            start_frame=args.start_frame,
            duration_frames=args.duration_frames,
            duration_seconds=args.duration_seconds,
            display=args.display,
        )
    if args.command == "detections-export-yolo":
        return run_detections_export_yolo(
            args.video,
            args.detections,
            output=args.output,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_step=args.frame_step,
            class_map_args=args.class_map,
            zip_output=args.zip_output,
        )

    parser.print_help()
    return 0


__all__ = [
    "main",
    "create_parser",
    "run_video_index",
    "run_detection",
    "run_detections_convert",
    "run_detections_overlay",
    "run_detections_export_yolo",
]
