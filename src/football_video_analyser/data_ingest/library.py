"""Utilities for discovering match footage and inspecting video metadata."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2  # type: ignore[import]

LOGGER = logging.getLogger(__name__)

DEFAULT_VIDEO_ROOTS: tuple[Path, ...] = (
    Path("/mnt/e/Recordings/Old Wilsonians/U9 Forbes"),
    Path("/mnt/e/Recordings/Old Wilsonians/U10 Bradbury"),
)

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".mkv")


@dataclass(slots=True)
class VideoMetadata:
    """Basic metadata extracted from a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float | None
    filesize_bytes: int
    bitrate_kbps: float | None
    recorded_at: datetime | None

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


def iter_video_files(
    roots: Sequence[Path] | None = None,
    suffixes: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield all discovered video files under the configured roots."""

    search_roots: Sequence[Path] = roots or DEFAULT_VIDEO_ROOTS
    exts = tuple({*(suffix.lower() for suffix in (suffixes or VIDEO_EXTENSIONS))})

    for root in search_roots:
        if not root.exists():
            LOGGER.debug("Video root missing: %s", root)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                yield path


def read_video_metadata(video_path: Path) -> VideoMetadata:
    """Collect width/height/fps/frame count/duration information for a file."""

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    duration = frame_count / fps if fps > 0 else None
    stats = video_path.stat()
    filesize = stats.st_size
    bitrate = (filesize * 8) / duration / 1000 if duration else None
    recorded_at = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)

    return VideoMetadata(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        filesize_bytes=filesize,
        bitrate_kbps=bitrate,
        recorded_at=recorded_at,
    )


def metadata_to_dict(metadata: VideoMetadata) -> dict[str, object]:
    """Convert metadata to a JSON/CSV friendly dictionary."""

    data = asdict(metadata)
    data["path"] = str(metadata.path)
    if metadata.recorded_at is not None:
        data["recorded_at"] = metadata.recorded_at.isoformat()
    return data


def build_video_index(
    roots: Sequence[Path] | None = None,
    suffixes: Iterable[str] | None = None,
    min_duration_seconds: float | None = None,
) -> list[VideoMetadata]:
    """Produce an in-memory index of known videos and their metadata."""

    index: list[VideoMetadata] = []
    for video_path in iter_video_files(roots=roots, suffixes=suffixes):
        try:
            metadata = read_video_metadata(video_path)
            if (
                min_duration_seconds is not None
                and metadata.duration_seconds is not None
                and metadata.duration_seconds < min_duration_seconds
            ):
                LOGGER.debug(
                    "Skipping %s: duration %.1fs < %.1fs",
                    video_path,
                    metadata.duration_seconds,
                    min_duration_seconds,
                )
                continue
            index.append(metadata)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read %s: %s", video_path, exc)
    return index


__all__ = [
    "DEFAULT_VIDEO_ROOTS",
    "VIDEO_EXTENSIONS",
    "VideoMetadata",
    "metadata_to_dict",
    "iter_video_files",
    "read_video_metadata",
    "build_video_index",
]
