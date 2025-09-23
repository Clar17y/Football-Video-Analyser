from __future__ import annotations

from pathlib import Path

import numpy as np

from football_video_analyser.models import tracking


class _FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _FakeBoxes:
    def __init__(self):
        self.cls = _FakeTensor([0, 1])
        self.conf = _FakeTensor([0.9, 0.8])
        self.id = _FakeTensor([12, -1])
        self.xyxy = _FakeTensor([[0, 0, 10, 10], [5, 5, 12, 12]])


class _FakeResult:
    def __init__(self, names):
        self.names = names
        self.boxes = _FakeBoxes()


class _FakeModel:
    def __init__(self):
        self.model = type("M", (), {"names": {0: "person", 1: "sports ball", 2: "car"}})()

    def track(self, **kwargs):
        yield _FakeResult(self.model.names)


def test_iter_tracked_objects_filters_classes(tmp_path: Path) -> None:
    fake_model = _FakeModel()
    detections = list(
        tracking.iter_tracked_objects(
            video_path=tmp_path,
            model=fake_model,
            class_names=("person", "sports ball"),
        )
    )

    assert len(detections) == 2
    player, ball = detections
    assert player.label == "person"
    assert player.track_id == 12
    assert ball.label == "sports ball"
    assert ball.track_id is None
