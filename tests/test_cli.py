from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from football_video_analyser import cli
from football_video_analyser.db import config as db_config
from football_video_analyser.db import ingest as db_ingest
from football_video_analyser.db import models as db_models
from football_video_analyser.db import session as db_session
from football_video_analyser.models.tracking import TrackedObject


def test_run_video_index_writes_json(tmp_path: Path, sample_video: Path) -> None:
    output = tmp_path / "index.json"
    rc = cli.run_video_index(
        [sample_video.parent],
        suffixes=[".mp4"],
        output=output,
        min_duration_minutes=0.0,
        save_to_db=False,
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload and payload[0]["path"] == str(sample_video)


def test_run_video_index_reports_missing_videos(tmp_path: Path) -> None:
    rc = cli.run_video_index(
        [tmp_path], suffixes=[".mp4"], output=None, min_duration_minutes=0.1, save_to_db=False
    )
    assert rc == 1


def test_run_video_index_filters_out_short_clips(tmp_path: Path, sample_video: Path) -> None:
    output = tmp_path / "index.json"
    rc = cli.run_video_index(
        [sample_video.parent],
        suffixes=[".mp4"],
        output=output,
        min_duration_minutes=120.0,  # 2 hours, should filter out sample clip
        save_to_db=False,
    )

    assert rc == 1
    assert not output.exists()


def test_run_video_index_saves_to_database(
    tmp_path: Path, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("FOOTBALL_VIDEO_DB_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("FOOTBALL_VIDEO_DB_SCHEMA", "")

    db_config.get_db_url.cache_clear()
    db_config.get_db_schema.cache_clear()

    importlib.reload(importlib.import_module("football_video_analyser.db"))
    models_module = importlib.reload(
        importlib.import_module("football_video_analyser.db.models")
    )
    session_module = importlib.reload(
        importlib.import_module("football_video_analyser.db.session")
    )
    importlib.reload(importlib.import_module("football_video_analyser.db.ingest"))
    cli_module = importlib.reload(importlib.import_module("football_video_analyser.cli"))

    engine = session_module.get_engine()
    models_module.Base.metadata.create_all(engine)

    rc = cli_module.run_video_index(
        [sample_video.parent],
        suffixes=[".mp4"],
        output=None,
        min_duration_minutes=0.0,
        save_to_db=True,
    )

    assert rc == 0

    SessionLocal = session_module.get_session_factory(engine)
    with SessionLocal() as session:
        result = session.execute(models_module.MatchVideo.__table__.select()).fetchall()
        assert len(result) == 1
        assert result[0].path == str(sample_video)


def test_run_detection_outputs_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / "game.mp4"
    video_path.touch()
    output_csv = tmp_path / "detections.csv"

    fake_detections = [
        TrackedObject(
            frame_index=0,
            label="person",
            confidence=0.9,
            bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
            track_id=1,
        ),
        TrackedObject(
            frame_index=1,
            label="sports ball",
            confidence=0.8,
            bbox_xyxy=(5.0, 5.0, 8.0, 8.0),
            track_id=None,
        ),
    ]

    monkeypatch.setattr(
        cli,
        "iter_tracked_objects",
        lambda **kwargs: iter(fake_detections),
    )

    rc = cli.run_detection(
        video_path,
        weights="dummy.pt",
        device=None,
        conf=0.1,
        iou=0.2,
        classes=["person", "sports ball"],
        output=output_csv,
    )

    assert rc == 0
    assert output_csv.exists()
    contents = output_csv.read_text().strip().splitlines()
    assert contents[0].startswith("frame_index")
    assert "person" in contents[1]
