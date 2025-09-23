from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    """Create a tiny synthetic video clip for testing."""

    video_path = tmp_path / "sample.mp4"
    width, height, fps, frames = 320, 240, 30, 60

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    try:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frames):
            writer.write(frame)
    finally:
        writer.release()

    return video_path
