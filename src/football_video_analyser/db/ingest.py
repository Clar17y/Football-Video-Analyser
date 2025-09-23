"""Helpers for persisting video metadata into the database."""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data_ingest.library import VideoMetadata
from .models import MatchVideo


def upsert_match_videos(session: Session, items: Iterable[VideoMetadata]) -> int:
    """Insert or update match video rows based on the file path key.

    Returns the number of records inserted or updated.
    """

    count = 0
    for item in items:
        existing = session.execute(
            select(MatchVideo).where(MatchVideo.path == str(item.path))
        ).scalar_one_or_none()

        if existing is None:
            session.add(
                MatchVideo(
                    path=str(item.path),
                    width=item.width,
                    height=item.height,
                    fps=item.fps,
                    frame_count=item.frame_count,
                    duration_seconds=item.duration_seconds,
                    filesize_bytes=item.filesize_bytes,
                    bitrate_kbps=item.bitrate_kbps,
                    recorded_at=item.recorded_at,
                )
            )
        else:
            existing.width = item.width
            existing.height = item.height
            existing.fps = item.fps
            existing.frame_count = item.frame_count
            existing.duration_seconds = item.duration_seconds
            existing.filesize_bytes = item.filesize_bytes
            existing.bitrate_kbps = item.bitrate_kbps
            existing.recorded_at = item.recorded_at
        count += 1

    return count


__all__ = ["upsert_match_videos"]
