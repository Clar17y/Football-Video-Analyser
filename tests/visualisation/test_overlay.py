from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_video_analyser.visualisation import (
    convert_detections_to_parquet,
    load_detections,
    render_overlay_segment,
)


def _create_dummy_video(path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)) -> None:
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_convert_and_load_detections(tmp_path: Path) -> None:
    csv_path = tmp_path / "detections.csv"
    df = pd.DataFrame(
        {
            "frame_index": [0, 1, 2],
            "label": ["person", "sports ball", "person"],
            "confidence": [0.9, 0.8, 0.95],
            "x1": [0, 5, 10],
            "y1": [0, 5, 10],
            "x2": [20, 15, 30],
            "y2": [20, 15, 30],
            "track_id": [1, None, 2],
        }
    )
    df.to_csv(csv_path, index=False)

    parquet_path = convert_detections_to_parquet(csv_path, chunk_rows=1)
    assert parquet_path.exists()

    loaded = load_detections(parquet_path, start_frame=1, end_frame=3)
    assert len(loaded) == 2
    assert loaded.iloc[0]["frame_index"] == 1


def test_render_overlay_segment(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "overlay.mp4"
    _create_dummy_video(video_path, frames=3)

    df = pd.DataFrame(
        {
            "frame_index": [0, 1, 1],
            "label": ["person", "person", "sports ball"],
            "confidence": [0.9, 0.85, 0.7],
            "x1": [5, 10, 15],
            "y1": [5, 10, 15],
            "x2": [25, 30, 35],
            "y2": [25, 30, 35],
            "track_id": [1, 2, None],
        }
    )

    processed = render_overlay_segment(
        video_path,
        df,
        start_frame=0,
        num_frames=2,
        output_path=output_path,
        display=False,
    )

    assert processed == 2
    assert output_path.exists()
