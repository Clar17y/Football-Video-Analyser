"""Command-line interface for local tooling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .data_ingest import build_video_index, metadata_to_dict
from .db import get_session_factory, upsert_match_videos
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

    parser.print_help()
    return 0


__all__ = ["main", "create_parser", "run_video_index", "run_detection"]
