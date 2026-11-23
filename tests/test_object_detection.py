from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend import object_detection


@pytest.mark.parametrize("override", [None, "custom.pt"])
def test_model_path_is_independent_of_working_directory_by_default(monkeypatch, tmp_path, override):
    monkeypatch.chdir(tmp_path)
    constructor = Mock()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=constructor))
    if override is None:
        monkeypatch.delenv("YOLO_MODEL_PATH", raising=False)
    else:
        monkeypatch.setenv("YOLO_MODEL_PATH", override)

    object_detection._load_model()

    default_path = Path(__file__).resolve().parents[1] / "yolo11n.pt"
    constructor.assert_called_once_with(override or str(default_path), task="detect")
