import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

_DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "models" / "tessdata"
_engine = None
_engine_lock = Lock()


def _load_engine():
    from tesserocr import OEM, PSM, RIL, PyTessBaseAPI

    data_path = Path(os.getenv("OCR_TESSDATA_DIR", str(_DEFAULT_DATA_PATH))).resolve()
    languages = os.getenv("OCR_LANGUAGES", "eng")
    for language in languages.split("+"):
        if not (data_path / f"{language}.traineddata").is_file():
            raise FileNotFoundError(f"Missing OCR language data for {language!r} in {data_path}")
    api = PyTessBaseAPI(
        path=str(data_path), lang=languages, psm=PSM.SPARSE_TEXT, oem=OEM.LSTM_ONLY
    )
    return api, RIL.WORD, RIL.TEXTLINE


def _collect_words(iterator, word_level, line_level, threshold: float) -> tuple[str, list[dict]]:
    lines = []
    words = []
    line = []
    while iterator is not None:
        if iterator.IsAtBeginningOf(line_level) and line:
            lines.append(" ".join(line))
            line = []

        bbox = iterator.BoundingBox(word_level)
        if bbox is not None:
            text = (iterator.GetUTF8Text(word_level) or "").strip()
            confidence = iterator.Confidence(word_level) / 100
            if text and confidence >= threshold:
                line.append(text)
                words.append({"text": text, "confidence": confidence, "bbox": list(bbox)})

        if not iterator.Next(word_level):
            break
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines), words


def recognize_text(image: np.ndarray) -> tuple[str, list[dict]]:
    """Return combined text and word details from an OpenCV BGR image."""
    threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0"))
    if not 0 <= threshold <= 1:
        raise ValueError("OCR_CONFIDENCE_THRESHOLD must be between 0 and 1")

    global _engine
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    pixels = gray.tobytes()
    with _engine_lock:
        if _engine is None:
            _engine = _load_engine()
        api, word_level, line_level = _engine
        try:
            api.SetImageBytes(pixels, width, height, 1, width)
            api.Recognize()
            return _collect_words(api.GetIterator(), word_level, line_level, threshold)
        finally:
            api.Clear()
