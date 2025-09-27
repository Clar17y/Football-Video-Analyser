from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

from football_video_analyser.annotation import export_detections_to_yolo


def _make_dummy_video(path: Path, frames: int = 4, size: tuple[int, int] = (64, 64)) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(frames):
        color = (i * 40) % 255
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_export_detections_to_yolo(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    detections_path = tmp_path / "detections.parquet"
    _make_dummy_video(video_path)

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

    output_dir = tmp_path / "export"
    export_detections_to_yolo(
        video_path=video_path,
        detections_path=detections_path,
        output_dir=output_dir,
        zip_output=False,
    )

    image_files = sorted((output_dir / "images").glob("*.jpg"))
    label_files = sorted((output_dir / "labels").glob("*.txt"))

    assert len(image_files) == 2
    assert len(label_files) == 2

    first_label = label_files[0].read_text().strip().splitlines()
    assert first_label
    assert first_label[0].startswith("0 ")

    assert (output_dir / "obj.names").exists()
    assert (output_dir / "obj.data").exists()
    assert (output_dir / "train.txt").exists()


def test_export_detections_to_yolo_zip(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    detections_path = tmp_path / "detections.parquet"
    _make_dummy_video(video_path)

    df = pd.DataFrame(
        {
            "frame_index": [0],
            "label": ["person"],
            "confidence": [0.9],
            "x1": [5],
            "y1": [5],
            "x2": [20],
            "y2": [20],
            "track_id": [1],
        }
    )
    df.to_parquet(detections_path)

    output_dir = tmp_path / "export_zip"
    zip_path = export_detections_to_yolo(
        video_path=video_path,
        detections_path=detections_path,
        output_dir=output_dir,
        zip_output=True,
    )

    assert zip_path.suffix == ".zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "obj.data" in names
        assert "obj.names" in names
        assert any(name.startswith("images/") for name in names)
