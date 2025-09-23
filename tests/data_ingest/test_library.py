from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from football_video_analyser.data_ingest import library


def test_iter_video_files_respects_custom_roots(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    video_a = tmp_path / "match_a.mp4"
    video_a.touch()
    video_b = tmp_path / "nested" / "match_b.mov"
    video_b.touch()
    (tmp_path / "ignore.txt").touch()

    discovered = list(
        library.iter_video_files(
            roots=[tmp_path],
            suffixes=(".mp4", ".mov"),
        )
    )

    assert discovered == sorted([video_a, video_b])


def test_read_video_metadata_produces_expected_values(sample_video: Path) -> None:
    metadata = library.read_video_metadata(sample_video)

    assert metadata.path == sample_video
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.fps == pytest.approx(30, rel=1e-2)
    assert metadata.frame_count == 60
    assert metadata.duration_seconds == pytest.approx(2.0, rel=1e-2)
    assert metadata.filesize_bytes > 0
    assert metadata.bitrate_kbps == pytest.approx(
        (metadata.filesize_bytes * 8) / (metadata.duration_seconds * 1000), rel=1e-2
    )
    assert isinstance(metadata.recorded_at, datetime)


def test_metadata_to_dict_serialises_fields(sample_video: Path) -> None:
    metadata = library.read_video_metadata(sample_video)
    payload = library.metadata_to_dict(metadata)

    assert payload["path"] == str(sample_video)
    assert payload["width"] == metadata.width
    assert payload["recorded_at"] == metadata.recorded_at.isoformat()
