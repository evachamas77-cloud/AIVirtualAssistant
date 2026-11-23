from types import SimpleNamespace
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import main, object_detection, ocr


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def fake_engines(monkeypatch):
    detector = Mock()
    detector.predict.return_value = [
        SimpleNamespace(names={0: "person"}, boxes=boxes([], [], []))
    ]
    engine = Mock()
    engine.GetIterator.return_value = None
    load_detector = Mock(return_value=detector)
    load_ocr = Mock(return_value=(engine, "word", "line"))
    monkeypatch.setattr(object_detection, "_model", None)
    monkeypatch.setattr(object_detection, "_load_model", load_detector)
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.setattr(ocr, "_load_engine", load_ocr)
    monkeypatch.delenv("YOLO_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("OCR_CONFIDENCE_THRESHOLD", raising=False)
    return SimpleNamespace(
        detector=detector, engine=engine, load_detector=load_detector, load_ocr=load_ocr
    )


def boxes(coordinates, scores, classes):
    return SimpleNamespace(
        xyxy=np.array(coordinates), conf=np.array(scores), cls=np.array(classes)
    )


def upload(client, extension=".png", mime="image/png", sharp=False):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    if sharp:
        image[:, ::2] = 255
    success, encoded = cv2.imencode(extension, image)
    assert success
    return client.post("/analyze-frame", files={"file": ("frame" + extension, encoded.tobytes(), mime)})


def test_health(client, fake_engines):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    fake_engines.load_detector.assert_not_called()
    fake_engines.load_ocr.assert_not_called()


@pytest.mark.parametrize(("extension", "mime"), [(".png", "image/png"), (".jpg", "image/jpeg")])
def test_upload_preserves_phase_one(client, extension, mime):
    response = upload(client, extension, mime, sharp=True)
    assert response.status_code == 200
    data = response.json()
    assert (data["width"], data["height"]) == (48, 32)
    assert data["sharpness_score"] == 260100.0
    assert data["quality"] == "sharp"
    assert data["processing_time_ms"] >= 0


@pytest.mark.parametrize(
    ("payload", "mime", "status", "detail"),
    [
        (b"not an image", "image/gif", 400, "Only JPEG and PNG images are supported"),
        (b"not an image", "image/png", 400, "Invalid image"),
        (b"", "image/jpeg", 400, "Empty image"),
        (b"x" * (main.MAX_IMAGE_SIZE + 1), "image/png", 413, "Image too large"),
    ],
    ids=["unsupported-media", "corrupt-image", "empty-image", "oversized-image"],
)
def test_bad_upload(client, fake_engines, payload, mime, status, detail):
    response = client.post("/analyze-frame", files={"file": ("frame", payload, mime)})
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    fake_engines.load_detector.assert_not_called()
    fake_engines.load_ocr.assert_not_called()


def test_detection_results_and_model_reuse(client, fake_engines, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIDENCE_THRESHOLD", "0.4")
    fake_engines.detector.predict.return_value = [
        SimpleNamespace(names={0: "person"}, boxes=boxes([[1, 2, 30, 31]], [0.9], [0]))
    ]
    expected = [{"name": "person", "confidence": 0.9, "bbox": [1, 2, 30, 31]}]
    for _ in range(2):
        data = upload(client).json()
        assert data["objects"] == expected
        assert data["analysis_errors"] == {}
    fake_engines.load_detector.assert_called_once_with()
    assert fake_engines.detector.predict.call_args.kwargs["conf"] == 0.4
    assert fake_engines.detector.predict.call_args.args[0].shape == (32, 48, 3)
    fake_engines.load_ocr.assert_called_once_with()
    assert fake_engines.engine.Clear.call_count == 2


def test_ocr_results_and_confidence_filter(client, fake_engines, monkeypatch):
    monkeypatch.setenv("OCR_CONFIDENCE_THRESHOLD", "0.5")
    iterator = Mock()
    iterator.IsAtBeginningOf.side_effect = [True, True, False]
    iterator.BoundingBox.side_effect = [(1, 2, 10, 8), (1, 12, 10, 18), (11, 12, 20, 18)]
    iterator.GetUTF8Text.side_effect = [" EXIT ", "Door", "noise"]
    iterator.Confidence.side_effect = [91.5, 80, 10]
    iterator.Next.side_effect = [True, True, False]
    fake_engines.engine.GetIterator.return_value = iterator

    data = upload(client).json()
    assert data["ocr_text"] == "EXIT\nDoor"
    assert data["ocr_results"] == [
        {"text": "EXIT", "confidence": 0.915, "bbox": [1, 2, 10, 8]},
        {"text": "Door", "confidence": 0.8, "bbox": [1, 12, 10, 18]},
    ]
    assert data["analysis_errors"] == {}
    pixels, width, height, channels, stride = fake_engines.engine.SetImageBytes.call_args.args
    assert (len(pixels), width, height, channels, stride) == (32 * 48, 48, 32, 1, 48)


def test_no_objects_or_text(client, fake_engines):
    iterator = Mock()
    iterator.BoundingBox.return_value = None
    iterator.Next.return_value = False
    fake_engines.engine.GetIterator.return_value = iterator

    data = upload(client).json()
    assert data["objects"] == []
    assert data["ocr_text"] == ""
    assert data["ocr_results"] == []
    assert data["analysis_errors"] == {}
    assert data["quality"] == "blurry"


@pytest.mark.parametrize("failed_stage", ["object_detection", "ocr"])
def test_unavailable_model_is_reported(client, fake_engines, failed_stage):
    if failed_stage == "object_detection":
        fake_engines.load_detector.side_effect = OSError("offline weights location")
    else:
        fake_engines.load_ocr.side_effect = FileNotFoundError("missing traineddata location")
        fake_engines.detector.predict.return_value = [
            SimpleNamespace(names={0: "person"}, boxes=boxes([[1, 2, 30, 31]], [0.9], [0]))
        ]

    response = upload(client)
    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 48
    assert set(data["analysis_errors"]) == {failed_stage}
    assert "location" not in response.text
    if failed_stage == "object_detection":
        assert data["objects"] == []
        fake_engines.engine.Recognize.assert_called_once_with()
    else:
        assert data["ocr_text"] == ""
        assert data["ocr_results"] == []
        assert data["objects"][0]["name"] == "person"
