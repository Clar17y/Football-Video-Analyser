from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
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

def _make_dummy_video(path: Path, frames: int = 4, size: tuple[int, int] = (64, 64)) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 25) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_run_detections_convert(tmp_path: Path) -> None:
    csv_path = tmp_path / "detections.csv"
    df = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "label": ["person", "sports ball"],
            "confidence": [0.9, 0.8],
            "x1": [0, 5],
            "y1": [0, 5],
            "x2": [10, 15],
            "y2": [10, 15],
            "track_id": [1, None],
        }
    )
    df.to_csv(csv_path, index=False)

    rc = cli.run_detections_convert(csv_path, output=None, chunk_rows=1)

    assert rc == 0
    parquet_path = csv_path.with_suffix(".parquet")
    assert parquet_path.exists()
    loaded = pd.read_parquet(parquet_path)
    assert len(loaded) == 2


def test_run_detections_overlay(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    detections_path = tmp_path / "detections.parquet"
    output_path = tmp_path / "overlay.mp4"

    _make_dummy_video(video_path, frames=3)

    df = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "label": ["person", "sports ball"],
            "confidence": [0.9, 0.7],
            "x1": [5, 10],
            "y1": [5, 10],
            "x2": [20, 25],
            "y2": [20, 25],
            "track_id": [1, None],
        }
    )
    df.to_parquet(detections_path)

    rc = cli.run_detections_overlay(
        video_path,
        detections_path,
        output=output_path,
        start_frame=0,
        duration_frames=2,
        duration_seconds=None,
        display=False,
    )

    assert rc == 0
    assert output_path.exists()


def test_run_detections_export_yolo(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    detections_path = tmp_path / "detections.parquet"
    output_dir = tmp_path / "export"

    _make_dummy_video(video_path, frames=3)

    df = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "label": ["person", "sports ball"],
            "confidence": [0.9, 0.7],
            "x1": [5, 10],
            "y1": [5, 10],
            "x2": [20, 25],
            "y2": [25, 30],
            "track_id": [1, None],
        }
    )
    df.to_parquet(detections_path)

    rc = cli.run_detections_export_yolo(
        video_path,
        detections_path,
        output=output_dir,
        start_frame=None,
        end_frame=None,
        frame_step=1,
        class_map_args=["person=0", "sports ball=1"],
        zip_output=False,
    )

    assert rc == 0
    assert (output_dir / "images").exists()
    assert (output_dir / "labels").exists()
    assert (output_dir / "obj.names").exists()
    assert (output_dir / "obj.data").exists()
    assert (output_dir / "train.txt").exists()
