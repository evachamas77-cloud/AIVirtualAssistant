import os
from pathlib import Path
from threading import Lock

import numpy as np

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "yolo11n.pt"
_model = None
_model_lock = Lock()


def _load_model():
    from ultralytics import YOLO

    return YOLO(os.getenv("YOLO_MODEL_PATH", str(_DEFAULT_MODEL_PATH)), task="detect")


def detect_objects(image: np.ndarray) -> list[dict]:
    """Detect objects in an OpenCV BGR image, using original-image pixel coordinates."""
    global _model
    threshold = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))
    if not 0 <= threshold <= 1:
        raise ValueError("YOLO_CONFIDENCE_THRESHOLD must be between 0 and 1")

    # The shared Ultralytics predictor has mutable state.
    with _model_lock:
        if _model is None:
            _model = _load_model()
        result = _model.predict(image, conf=threshold, device="cpu", verbose=False)[0]
        if result.boxes is None:
            return []
        return [
            {
                "name": result.names[int(class_id)],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
            for (x1, y1, x2, y2), confidence, class_id in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            )
        ]
