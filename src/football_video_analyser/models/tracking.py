"""Baseline detection and tracking utilities using Ultralytics YOLO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Sequence

try:  # pragma: no cover - progress bar is optional
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

import numpy as np

try:  # pragma: no cover - cv2 may not be available in all environments
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional import for type hints
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - ultralytics may be missing in test env
    YOLO = None  # type: ignore


@dataclass(slots=True)
class TrackedObject:
    frame_index: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | None


def _tensor_to_numpy(tensor: object) -> np.ndarray:
    """Convert a torch-like tensor to a NumPy array without importing torch explicitly."""

    if tensor is None:
        return np.empty((0,))

    # Support torch tensors as well as numpy arrays.
    if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        return tensor.cpu().numpy()
    if hasattr(tensor, "detach") and hasattr(tensor.detach(), "cpu"):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, (list, tuple)):
        return np.asarray(tensor)

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def iter_tracked_objects(
    video_path: Path,
    *,
    weights: str = "yolov8x.pt",
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    class_names: Sequence[str] = ("person", "sports ball"),
    model: object | None = None,
    tracker_config: str = "bytetrack.yaml",
) -> Generator[TrackedObject, None, None]:
    """Yield tracked objects for each frame of the provided video.

    Parameters
    ----------
    video_path:
        Path to an `.mp4` file that will be processed.
    weights:
        YOLO weights file or model name to load via Ultralytics.
    device:
        Device string understood by PyTorch (e.g. `cuda:0`, `cpu`).
    conf/iou:
        Confidence and NMS thresholds passed to Ultralytics.
    class_names:
        Set of detection class names to retain. Defaults to people + sports ball.
    model:
        Optional injected YOLO model (used for testing/mocking).
    tracker_config:
        Tracker configuration to use (defaults to ByteTrack).
    """

    if model is None:
        if YOLO is None:  # pragma: no cover - environment missing dependency
            raise RuntimeError(
                "Ultralytics is not installed. Install with `pip install ultralytics`."
            )
        model = YOLO(weights)

    # Ultralytics keeps class names on the underlying model object.
    raw_names = getattr(getattr(model, "model", model), "names", {})
    allowed_class_ids = {
        class_id
        for class_id, name in raw_names.items()
        if name in class_names
    }

    # When tracking, we need to add the `classes` argument as ids.
    classes = sorted(allowed_class_ids) if allowed_class_ids else None

    results = model.track(  # type: ignore[attr-defined]
        source=str(video_path),
        stream=True,
        tracker=tracker_config,
        device=device,
        conf=conf,
        iou=iou,
        classes=classes,
        verbose=False,
    )

    total_frames = None
    if tqdm is not None and cv2 is not None:
        try:
            capture = cv2.VideoCapture(str(video_path))
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            capture.release()
            if frame_count and frame_count > 0:
                total_frames = int(frame_count)
        except Exception:  # pragma: no cover - progress bar is optional
            total_frames = None

    pbar = tqdm(total=total_frames, desc="Tracking", unit="frame") if tqdm is not None else None

    for frame_index, result in enumerate(results):
        if pbar is not None:
            pbar.update(1)
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        cls = _tensor_to_numpy(getattr(boxes, "cls", None)).astype(np.int32)
        confs = _tensor_to_numpy(getattr(boxes, "conf", None)).astype(np.float32)
        ids_raw = getattr(boxes, "id", None)
        ids = (
            _tensor_to_numpy(ids_raw).astype(np.int32)
            if ids_raw is not None
            else np.array([], dtype=np.int32)
        )
        xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", None)).astype(np.float32)

        for idx in range(xyxy.shape[0]):
            class_id = int(cls[idx])
            label = raw_names.get(class_id, str(class_id))
            if class_names and label not in class_names:
                continue

            track_id = None
            if ids.size > idx and ids[idx] >= 0:
                track_id = int(ids[idx])

            bbox = tuple(float(v) for v in xyxy[idx])  # type: ignore[assignment]
            yield TrackedObject(
                frame_index=frame_index,
                label=label,
                confidence=float(confs[idx]),
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                track_id=track_id,
            )

    if pbar is not None:
        pbar.close()


def detections_to_rows(detections: Iterable[TrackedObject]) -> list[dict[str, object]]:
    """Convert tracked objects into serialisable dictionaries."""

    rows: list[dict[str, object]] = []
    for det in detections:
        rows.append(
            {
                "frame_index": det.frame_index,
                "label": det.label,
                "confidence": det.confidence,
                "x1": det.bbox_xyxy[0],
                "y1": det.bbox_xyxy[1],
                "x2": det.bbox_xyxy[2],
                "y2": det.bbox_xyxy[3],
                "track_id": det.track_id,
            }
        )
    return rows


__all__ = ["TrackedObject", "iter_tracked_objects", "detections_to_rows"]
