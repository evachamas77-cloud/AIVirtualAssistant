from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from threading import Event
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import main
from backend.frame_store import utc_timestamp

OBJECTS = [{"name": "Traffic Light", "confidence": 0.9, "bbox": [1, 2, 25, 28]}]
WORDS = [{"text": "Caf\u00e9", "confidence": 0.95, "bbox": [2, 3, 12, 14]}]


@pytest.fixture
def engines(monkeypatch):
    detector = Mock(return_value=OBJECTS)
    reader = Mock(return_value=("Caf\u00e9 exit", WORDS))
    monkeypatch.setattr(main, "detect_objects", detector)
    monkeypatch.setattr(main, "recognize_text", reader)
    monkeypatch.setenv("FRAME_MIN_INTERVAL_SECONDS", "0")
    return detector, reader


@pytest.fixture
def client(engines):
    with TestClient(main.app) as test_client:
        yield test_client


def make_png(level=255, compression=1):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, ::2] = level
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    assert success
    return encoded.tobytes()


def upload(client, level=255, *, compression=1):
    contents = make_png(level, compression)
    return client.post("/frames", files={"file": ("../../untrusted.png", contents, "image/png")})


def set_timestamp(directory, frame_id, timestamp):
    with closing(sqlite3.connect(directory / "frames.sqlite3")) as db, db:
        db.execute("UPDATE frames SET timestamp = ? WHERE id = ?", (timestamp, frame_id))


def test_frame_storage_and_deletion(client, isolate_frame_storage):
    response = upload(client)
    assert response.status_code == 201
    data = response.json()
    assert data["saved"] is True and data["reason"] is None
    frame = data["frame"]
    assert frame["timestamp"].endswith("Z")
    assert datetime.fromisoformat(frame["timestamp"]).tzinfo is UTC
    assert frame["sharpness_score"] == 260100.0
    assert frame["ocr_text"] == "Caf\u00e9 exit"
    assert frame["objects"] == OBJECTS and frame["ocr_results"] == WORDS
    assert Path(frame["image_path"]).parts[0] == "images"
    stored = isolate_frame_storage / frame["image_path"]
    assert stored.read_bytes() == make_png()
    assert (isolate_frame_storage / "frames.sqlite3").is_file()

    url = f'/frames/{frame["id"]}'
    assert client.get(url).json() == frame
    assert client.get("/frames").json()["items"] == [frame]
    assert client.delete(url).status_code == 204
    assert not stored.exists()
    assert client.get(url).status_code == 404
    assert client.delete(url).status_code == 404
    assert client.get("/frames").json()["items"] == []


def test_reencoded_pixels_are_duplicate(client, isolate_frame_storage):
    assert make_png(compression=1) != make_png(compression=9)
    assert upload(client).status_code == 201
    response = upload(client, compression=9)
    assert response.status_code == 200
    assert response.json()["reason"] == "duplicate"
    assert response.json()["frame"] is None
    assert len(list((isolate_frame_storage / "images").iterdir())) == 1


def test_minimum_interval_under_concurrent_uploads(engines, monkeypatch):
    monkeypatch.setenv("FRAME_MIN_INTERVAL_SECONDS", "3600")
    with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda level: upload(client, level), (240, 255)))
        assert sorted(result.status_code for result in results) == [200, 201]
        assert next(result for result in results if result.status_code == 200).json()["reason"] == "min_interval"
        assert len(client.get("/frames").json()["items"]) == 1


def test_configurable_sharpness_filter(engines, monkeypatch, isolate_frame_storage):
    monkeypatch.setenv("FRAME_MIN_SHARPNESS", "300000")
    with TestClient(main.app) as client:
        response = upload(client)
        assert response.status_code == 200
        assert response.json()["saved"] is False
        assert response.json()["reason"] == "too_blurry"
        assert response.json()["analysis"]["sharpness_score"] == 260100.0
        assert list((isolate_frame_storage / "images").iterdir()) == []


def test_analysis_failure_is_not_archived(client, engines):
    detector, _ = engines
    detector.side_effect = RuntimeError("model failure")
    response = upload(client)
    assert response.status_code == 200
    assert response.json()["reason"] == "analysis_failed"
    assert "object_detection" in response.json()["analysis"]["analysis_errors"]
    assert client.get("/frames").json()["items"] == []


@pytest.mark.parametrize(
    ("contents", "mime", "status"),
    [
        (b"", "image/png", 400),
        (b"not an image", "image/png", 400),
        (b"\x89PNG\r\n\x1a\ninvalid", "image/png", 400),
        (b"not an image", "image/gif", 400),
        (b"0" * (main.MAX_IMAGE_SIZE + 1), "image/jpeg", 413),
    ],
)
def test_invalid_frames_do_not_run_inference(client, engines, contents, mime, status):
    response = client.post("/frames", files={"file": ("test", contents, mime)})
    assert response.status_code == status
    for engine in engines:
        engine.assert_not_called()
    assert client.get("/frames").json()["items"] == []


def test_pagination_time_filters_and_keyword_search(client, engines, isolate_frame_storage):
    first = upload(client).json()["frame"]
    detector, reader = engines
    detector.return_value = [{"name": "bus", "confidence": 0.7, "bbox": [0, 0, 10, 10]}]
    reader.return_value = ("Stop 50%", [])
    second = upload(client, 220).json()["frame"]
    set_timestamp(isolate_frame_storage, first["id"], "2026-03-01T10:00:00.000000Z")
    set_timestamp(isolate_frame_storage, second["id"], "2026-03-01T12:00:00.000000Z")

    page = client.get("/frames", params={"limit": 1}).json()
    assert page["items"][0]["id"] == second["id"] and page["next_offset"] == 1
    page = client.get("/frames", params={"limit": 1, "offset": 1}).json()
    assert page["items"][0]["id"] == first["id"] and page["next_offset"] is None
    window = {"start": "2026-03-01T12:00:00+02:00", "end": "2026-03-01T10:00:00Z"}
    filtered = client.get("/frames", params=window).json()["items"]
    assert [item["id"] for item in filtered] == [first["id"]]

    for keyword, expected in [("CAF\u00c9", first), ("TRAFFIC", first), ("50%", second), ("BUS", second)]:
        matches = client.get("/frames/search", params={"q": keyword}).json()["items"]
        assert [item["id"] for item in matches] == [expected["id"]]
    for keyword in ("' OR 1=1 --", "confidence", "airplane", "_"):
        assert client.get("/frames/search", params={"q": keyword}).json()["items"] == []
    assert client.get("/frames/search", params={**window, "q": "bus"}).json()["items"] == []


@pytest.mark.parametrize(
    "url",
    [
        "/frames?limit=101",
        "/frames?offset=-1",
        "/frames?start=2026-03-01T10:00:00",
        "/frames?start=2026-03-02T10:00:00Z&end=2026-03-01T10:00:00Z",
        "/frames/search?q=%20",
        "/frames/search",
        "/frames/100000000000000000000",
    ],
)
def test_invalid_query_parameters(client, url):
    assert client.get(url).status_code == 422


def test_startup_retention_and_disabled_retention(engines, monkeypatch, isolate_frame_storage):
    monkeypatch.setenv("FRAME_RETENTION_DAYS", "0")
    with TestClient(main.app) as client:
        old = upload(client).json()["frame"]
        recent = upload(client, 220).json()["frame"]
    old_image = isolate_frame_storage / old["image_path"]
    set_timestamp(isolate_frame_storage, old["id"], utc_timestamp(datetime.now(UTC) - timedelta(days=2)))
    with TestClient(main.app) as client:
        assert client.get(f'/frames/{old["id"]}').status_code == 200
        assert old_image.is_file()

    monkeypatch.setenv("FRAME_RETENTION_DAYS", "1")
    with TestClient(main.app) as client:
        assert client.get(f'/frames/{old["id"]}').status_code == 404
        assert not old_image.exists()
        assert client.get(f'/frames/{recent["id"]}').status_code == 200
        assert (isolate_frame_storage / recent["image_path"]).is_file()


def test_periodic_retention_runs_without_requests(engines, monkeypatch, isolate_frame_storage):
    monkeypatch.setenv("FRAME_RETENTION_DAYS", "1")
    monkeypatch.setenv("FRAME_CLEANUP_INTERVAL_SECONDS", "1")
    deleted = Event()
    real_delete = main.FrameStore.delete

    def observe_delete(store, frame_id):
        result = real_delete(store, frame_id)
        if result:
            deleted.set()
        return result

    monkeypatch.setattr(main.FrameStore, "delete", observe_delete)
    with TestClient(main.app) as client:
        frame = upload(client).json()["frame"]
        set_timestamp(
            isolate_frame_storage, frame["id"], utc_timestamp(datetime.now(UTC) - timedelta(days=2))
        )
        assert deleted.wait(timeout=5)
        assert not (isolate_frame_storage / frame["image_path"]).exists()
        assert client.get("/frames").json()["items"] == []


def test_failed_insert_removes_new_image(client, isolate_frame_storage):
    with closing(sqlite3.connect(isolate_frame_storage / "frames.sqlite3")) as db, db:
        db.executescript(
            """
            CREATE TRIGGER reject_insert BEFORE INSERT ON frames
            BEGIN
                SELECT RAISE(ABORT, 'test rejection');
            END;
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="test rejection"):
        upload(client)
    assert client.get("/frames").json()["items"] == []
    assert list((isolate_frame_storage / "images").iterdir()) == []


def test_failed_image_delete_preserves_metadata(client, monkeypatch, isolate_frame_storage):
    frame = upload(client).json()["frame"]
    url = f'/frames/{frame["id"]}'
    image = isolate_frame_storage / frame["image_path"]
    real_unlink = Path.unlink

    def refuse_image(path, *args, **kwargs):
        if path == image:
            raise PermissionError("test image is locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_image)
    with pytest.raises(PermissionError, match="test image is locked"):
        client.delete(url)
    assert image.is_file()
    assert client.get(url).json() == frame
