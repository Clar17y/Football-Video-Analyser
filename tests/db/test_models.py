from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import sqlalchemy as sa

from football_video_analyser.db import config as db_config


def test_match_video_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = tmp_path / "test.db"
    monkeypatch.setenv("FOOTBALL_VIDEO_DB_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("FOOTBALL_VIDEO_DB_SCHEMA", "")

    db_config.get_db_url.cache_clear()
    db_config.get_db_schema.cache_clear()

    models_module = importlib.reload(
        importlib.import_module("football_video_analyser.db.models")
    )
    session_module = importlib.reload(
        importlib.import_module("football_video_analyser.db.session")
    )

    engine = session_module.get_engine()
    models_module.Base.metadata.create_all(engine)
    SessionLocal = session_module.get_session_factory(engine)

    with SessionLocal.begin() as session:
        video = models_module.MatchVideo(
            path="test.mp4",
            width=1920,
            height=1080,
            fps=60.0,
            frame_count=3600,
            duration_seconds=60.0,
            filesize_bytes=123456,
            bitrate_kbps=2000.0,
        )
        video.events.append(
            models_module.EventAnnotation(
                event_type="shot",
                timestamp_seconds=12.5,
                duration_seconds=1.2,
                team="Home",
                player_label="Player 7",
                confidence=0.9,
                extra_data={"foot": "left"},
            )
        )
        session.add(video)

    with SessionLocal() as session:
        result = session.execute(sa.select(models_module.MatchVideo)).scalar_one()
        assert result.path == "test.mp4"
        assert result.width == 1920
        assert result.duration_seconds == 60.0
        assert len(result.events) == 1
        assert result.events[0].event_type == "shot"
