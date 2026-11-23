import pytest


@pytest.fixture(autouse=True)
def isolate_frame_storage(monkeypatch, tmp_path):
    for name in (
        "FRAME_MIN_SHARPNESS",
        "FRAME_MIN_INTERVAL_SECONDS",
        "FRAME_RETENTION_DAYS",
        "FRAME_CLEANUP_INTERVAL_SECONDS",
        "LIVE_SESSION_CODE",
        "LIVE_SAMPLE_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    storage = tmp_path / "frames"
    monkeypatch.setenv("FRAME_STORAGE_DIR", str(storage))
    monkeypatch.setenv("SEMANTIC_ENABLED", "0")
    return storage
