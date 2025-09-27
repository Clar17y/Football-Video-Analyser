"""Utility functions for working with detection outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional

import sys

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from football_video_analyser.models.tracking import TrackedObject, detections_to_rows

_DETECTION_COLUMNS = (
    "frame_index",
    "label",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id",
)

_COLUMN_DTYPES = {
    "frame_index": "int64",
    "label": "string",
    "confidence": "float64",
    "x1": "float64",
    "y1": "float64",
    "x2": "float64",
    "y2": "float64",
    "track_id": "Int64",
}


def load_detections(
    detections_path: Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Load detections from CSV or Parquet, optionally filtering by frame range."""

    detections_path = detections_path.expanduser()
    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    suffix = detections_path.suffix.lower()
    filters = None
    if start_frame is not None or end_frame is not None:
        filters = []
        if start_frame is not None:
            filters.append(("frame_index", ">=", int(start_frame)))
        if end_frame is not None:
            filters.append(("frame_index", "<", int(end_frame)))

    if suffix == ".parquet":
        df = pd.read_parquet(detections_path, filters=filters)
    elif suffix == ".csv":
        df = pd.read_csv(detections_path)
        if start_frame is not None:
            df = df[df["frame_index"] >= int(start_frame)]
        if end_frame is not None:
            df = df[df["frame_index"] < int(end_frame)]
    else:
        raise ValueError(f"Unsupported detections format: {detections_path}")

    for col in _DETECTION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(_DETECTION_COLUMNS)].copy()

    for col, dtype in _COLUMN_DTYPES.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception:
            if dtype == "Int64":
                df[col] = df[col].astype("float64").astype("Int64")
            else:
                df[col] = df[col].astype(dtype, copy=False)
    df.sort_values("frame_index", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def convert_detections_to_parquet(
    detections_path: Path,
    *,
    output_path: Path | None = None,
    compression: str = "zstd",
    chunk_rows: int = 500_000,
    show_progress: bool = True,
) -> Path:
    """Convert a detection CSV/Parquet into a compressed Parquet file."""

    detections_path = detections_path.expanduser()
    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    suffix = detections_path.suffix.lower()
    if suffix == ".parquet":
        return detections_path

    if suffix != ".csv":
        raise ValueError("Only CSV or Parquet detections are supported")

    output_path = (
        output_path
        if output_path is not None
        else detections_path.with_suffix(".parquet")
    )

    reader = pd.read_csv(detections_path, chunksize=chunk_rows)
    total_rows = None
    if show_progress:
        with detections_path.open("r", encoding="utf-8", errors="ignore") as handle:
            total_rows = sum(1 for _ in handle) - 1
            if total_rows < 0:
                total_rows = 0
    disable_progress = (not show_progress) or (not sys.stderr.isatty())
    progress = tqdm(total=total_rows, desc="Converting", unit="rows", disable=disable_progress)
    writer = None
    try:
        for chunk in reader:
            chunk = chunk.copy()
            for col in _DETECTION_COLUMNS:
                if col not in chunk.columns:
                    chunk[col] = pd.NA
            chunk = chunk[list(_DETECTION_COLUMNS)]

            for col, dtype in _COLUMN_DTYPES.items():
                if dtype == "Int64":
                    chunk[col] = chunk[col].astype("float64").astype("Int64")
                else:
                    chunk[col] = chunk[col].astype(dtype)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression=compression,
                )
            writer.write_table(table)
            if progress is not None:
                progress.update(len(chunk))
    finally:
        if writer is not None:
            writer.close()
        if progress is not None:
            progress.close()

    return output_path


def render_overlay_segment(
    video_path: Path,
    detections: pd.DataFrame | Iterable[TrackedObject],
    *,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    output_path: Path | None = None,
    display: bool = False,
    wait_key: int = 1,
    label_colors: Mapping[str, tuple[int, int, int]] | None = None,
    show_progress: bool = True,
) -> int:
    """Overlay detections onto video frames and optionally write/display them.

    Returns the number of frames processed.
    """

    video_path = video_path.expanduser()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if isinstance(detections, pd.DataFrame):
        df = detections
    else:
        df = pd.DataFrame(detections_to_rows(detections))

    if df.empty:
        raise ValueError("No detections provided")

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    records = df.to_dict(orient="records")
    for record in records:
        grouped[int(record["frame_index"])].append(record)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    if start_frame:
        current = 0
        while current < start_frame:
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Unable to seek to frame {start_frame}; video ended at {current}"
                )
            current += 1

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path is not None:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    colors = dict(label_colors or {})
    default_colors = {
        "person": (0, 0, 255),           # Red in BGR
        "sports ball": (255, 0, 0),      # Blue in BGR
    }

    frame_index = start_frame
    processed = 0

    disable_progress = (not show_progress) or (not sys.stderr.isatty())
    total_frames = None
    if num_frames is not None:
        total_frames = num_frames
    else:
        total_video_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_video_frames and total_video_frames > 0:
            total_frames = int(total_video_frames) - start_frame

    progress = tqdm(total=total_frames, desc="Rendering", unit="frame", disable=disable_progress)

    det_max_x = float(df["x2"].max()) if not df["x2"].isna().all() else 0.0
    det_max_y = float(df["y2"].max()) if not df["y2"].isna().all() else 0.0

    if det_max_x > width * 1.02:
        scale_x = width / det_max_x
    else:
        scale_x = 1.0

    if det_max_y > height * 1.02:
        scale_y = height / det_max_y
    else:
        scale_y = 1.0

    while True:
        if num_frames is not None and processed >= num_frames:
            break

        ok, frame = capture.read()
        if not ok:
            break

        for det in grouped.get(frame_index, []):
            label = str(det["label"])
            color = colors.get(label) or default_colors.get(label) or (255, 0, 0)
            x1 = int(float(det["x1"]) * scale_x)
            y1 = int(float(det["y1"]) * scale_y)
            x2 = int(float(det["x2"]) * scale_x)
            y2 = int(float(det["y2"]) * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}"
            track_id = det.get("track_id")
            if track_id is not None and not pd.isna(track_id):
                text += f"#{int(track_id)}"
            text += f" {float(det['confidence']):.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if writer is not None:
            writer.write(frame)

        if display:
            cv2.imshow("detections", frame)
            if cv2.waitKey(wait_key) & 0xFF == ord("q"):
                break

        processed += 1
        frame_index += 1
        if progress is not None:
            progress.update(1)

    capture.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    if progress is not None:
        progress.close()

    return processed


__all__ = [
    "load_detections",
    "convert_detections_to_parquet",
    "render_overlay_segment",
]
