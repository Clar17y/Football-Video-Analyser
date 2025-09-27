"""Helpers for exporting detections into annotation-friendly formats."""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
from tqdm import tqdm

from football_video_analyser.visualisation.overlay import load_detections

LOGGER = logging.getLogger(__name__)

DEFAULT_CLASS_MAP = {
    "person": 0,
    "sports ball": 1,
}


def export_detections_to_yolo(
    *,
    video_path: Path,
    detections_path: Path,
    output_dir: Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_step: int = 1,
    class_map: dict[str, int] | None = None,
    image_format: str = "jpg",
    show_progress: bool = True,
    zip_output: bool = False,
) -> Path:
    """Export detection results to a YOLO-style dataset (images + labels).

    Parameters
    ----------
    video_path:
        Source video file for extracting frames.
    detections_path:
        CSV or Parquet file produced by the detector command.
    output_dir:
        Directory where `images/` and `labels/` subfolders will be created.
    start_frame, end_frame:
        Optional frame bounds. Default uses the min/max frames present in detections.
    frame_step:
        Extract every Nth frame (default 1). Use 1 when supplying detections for annotation.
    class_map:
        Mapping from detection labels (e.g. "person") to YOLO class IDs. Defaults to
        `{"person": 0, "sports ball": 1}`.
    image_format:
        Image extension for exported frames (default jpg).
    show_progress:
        Whether to show tqdm progress bars.

    Returns
    -------
    Path to the output directory.
    """

    video_path = video_path.expanduser()
    detections_path = detections_path.expanduser()
    output_dir = output_dir.expanduser()

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    class_map = {**DEFAULT_CLASS_MAP, **(class_map or {})}
    reverse_map = {v: k for k, v in class_map.items()}
    if len(reverse_map) != len(class_map):
        raise ValueError("Class IDs must be unique")

    df = load_detections(detections_path)
    if df.empty:
        raise ValueError("Detections file contains no data")

    min_frame = int(df["frame_index"].min())
    max_frame = int(df["frame_index"].max())

    if start_frame is None:
        start_frame = min_frame
    if end_frame is None:
        end_frame = max_frame + 1

    start_frame = max(start_frame, min_frame)
    end_frame = min(end_frame, max_frame + 1)

    if start_frame >= end_frame:
        raise ValueError("No frames remain after applying start/end bounds")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = (output_dir / "images")
    labels_dir = (output_dir / "labels")
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

    image_rel_paths: list[str] = []

    # Seek sequentially to the starting frame
    current_frame = 0
    bar_disable = not show_progress
    with tqdm(total=end_frame - start_frame, desc="Exporting", unit="frame", disable=bar_disable) as progress:
        while current_frame < start_frame:
            ok, _ = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(
                    f"Video ended before reaching requested start frame {start_frame}"
                )
            current_frame += 1

        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame = capture.read()
            if not ok:
                LOGGER.warning("Video ended before reaching frame %d", end_frame)
                break

            if (frame_idx - start_frame) % frame_step != 0:
                frame_idx += 1
                continue

            detections = df[df["frame_index"] == frame_idx]

            image_name = f"frame_{frame_idx:06d}.{image_format}"
            image_path = images_dir / image_name
            cv2.imwrite(str(image_path), frame)
            image_rel_paths.append(f"images/{image_name}")

            label_lines: list[str] = []
            for _, det in detections.iterrows():
                label = str(det["label"])
                if label not in class_map:
                    LOGGER.debug("Skipping unknown label %s at frame %d", label, frame_idx)
                    continue
                class_id = class_map[label]
                x1, y1, x2, y2 = [float(det[col]) for col in ("x1", "y1", "x2", "y2")]
                # Clamp coordinates within image bounds
                x1 = max(0.0, min(x1, width))
                y1 = max(0.0, min(y1, height))
                x2 = max(0.0, min(x2, width))
                y2 = max(0.0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w_norm = (x2 - x1) / width
                h_norm = (y2 - y1) / height
                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            label_name = image_name.replace(f".{image_format}", ".txt")
            contents = "\n".join(label_lines)

            label_path = labels_dir / label_name
            with label_path.open("w", encoding="utf-8") as handle:
                handle.write(contents)

            inline_label_path = images_dir / label_name
            with inline_label_path.open("w", encoding="utf-8") as handle:
                handle.write(contents)

            frame_idx += 1
            progress.update(1)

    capture.release()

    # Write metadata files expected by CVAT/YOLO
    names_path = output_dir / "obj.names"
    with names_path.open("w", encoding="utf-8") as handle:
        for label, class_id in sorted(class_map.items(), key=lambda kv: kv[1]):
            handle.write(f"{label}\n")

    train_txt = output_dir / "train.txt"
    with train_txt.open("w", encoding="utf-8") as handle:
        for rel_path in image_rel_paths:
            handle.write(f"{rel_path}\n")

    data_path = output_dir / "obj.data"
    with data_path.open("w", encoding="utf-8") as handle:
        handle.write(f"classes={len(class_map)}\n")
        handle.write("names=obj.names\n")
        handle.write("train=train.txt\n")

    if zip_output:
        files = [p for p in output_dir.rglob("*") if p.is_file()]
        disable = (not show_progress) or (not sys.stderr.isatty())
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in tqdm(files, desc="Zipping", unit="file", disable=disable):
                zf.write(file_path, file_path.relative_to(output_dir))
        return zip_path

    return output_dir


def parse_class_mappings(items: Iterable[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid class mapping '{raw}'. Use label=id")
        label, value = raw.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label:
            raise ValueError(f"Empty label in mapping '{raw}'")
        try:
            class_id = int(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid class id '{value}' in mapping '{raw}'") from exc
        mapping[label] = class_id
    return mapping
