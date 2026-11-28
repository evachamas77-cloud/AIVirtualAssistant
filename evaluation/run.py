"""Opt-in, offline evaluation with the existing public bus and sign fixtures."""

import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2

from backend.ai_routes import Question
from backend.frame_store import FrameSettings, FrameStore
from backend.object_detection import detect_objects
from backend.ocr import recognize_text
from backend.semantic import SemanticIndex, SiglipEncoder
from backend.vision import VisionService

ROOT = Path(__file__).resolve().parents[1]
RUNS = 5


def stats(values):
    return {"n": len(values), "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2), "max": round(max(values), 2)}


def time_sync(operation):
    sample = operation()
    times = []
    for _ in range(RUNS):
        began = perf_counter()
        operation()
        times.append((perf_counter() - began) * 1000)
    return sample, stats(times)


async def time_async(operation):
    sample = await operation()
    times = []
    for _ in range(RUNS):
        began = perf_counter()
        await operation()
        times.append((perf_counter() - began) * 1000)
    return sample, stats(times)


def char_errors(actual, expected):
    row = list(range(len(expected) + 1))
    for index, actual_letter in enumerate(actual, 1):
        next_row = [index]
        for j, expected_letter in enumerate(expected, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (actual_letter != expected_letter)))
        row = next_row
    return row[-1]


async def archive_checks(store, encoder, label_ids):
    index = SemanticIndex(store, encoder)
    index.initialize()
    try:
        while index.index_next():
            pass
        questions = [("A city bus on a street", "bus"), ("Un autobus dans la rue", "bus"),
                     ("A sign that says HELLO WORLD", "sign"), ("Une pancarte avec le texte HELLO WORLD", "sign")]
        results = {}
        for phrase, expected in questions:
            hits = await index.search(phrase, limit=2)
            results[phrase] = {"expected": expected, "ranked": [label_ids[hit["frame_id"]] for hit in hits]}
        hits_at_1 = sum(row["ranked"][:1] == [row["expected"]] for row in results.values())
        _, search_ms = await time_async(lambda: index.search(questions[0][0], limit=2))
        service = VisionService(store, index, None)
        question = Question(question='What was written on the sign "HELLO WORLD" earlier?', consent=True)
        answer, local_ask_ms = await time_async(lambda: service.ask(question, "evaluation"))
        assert answer["model"] is None and answer["estimated_cost_usd"] == 0
        assert answer["evidence"] and label_ids[answer["evidence"][0]["frame_id"]] == "sign"
        return results, hits_at_1, search_ms, local_ask_ms
    finally:
        await index.stop()


def main():
    weights = ROOT / "yolo11n.pt"
    models = ROOT / "models"
    paths = {key: models / f"phase2-smoke-{key}.jpg" for key in ("bus", "sign")}
    os.environ.update(OPENAI_API_KEY="", YOLO_MODEL_PATH=str(weights), YOLO_CONFIDENCE_THRESHOLD="0.25",
                      OCR_TESSDATA_DIR=str(models / "tessdata"), OCR_LANGUAGES="eng",
                      OCR_CONFIDENCE_THRESHOLD="0", SEMANTIC_ENABLED="1")
    encoder = SiglipEncoder(models / "siglip2")
    if not all(path.is_file() for path in [weights, models / "tessdata" / "eng.traineddata", *paths.values()]) or not encoder.available():
        raise SystemExit("Missing local fixture images or installed YOLO, Tesseract, or SigLIP assets; no downloads attempted.")
    bus = cv2.imread(str(paths["bus"]))
    sign = cv2.imread(str(paths["sign"]))
    objects, detection_ms = time_sync(lambda: detect_objects(bus))
    (ocr_text, words), ocr_ms = time_sync(lambda: recognize_text(sign))
    expected = "HELLO WORLD"
    with TemporaryDirectory(prefix="visual-assistant-evaluation-") as scratch:
        store = FrameStore(FrameSettings(Path(scratch), 0, 0, 0, 3600))
        store.initialize()
        label_ids = {}
        for kind, pixels, detections, text, results in (("bus", bus, objects, "", []), ("sign", sign, [], ocr_text, words)):
            saved, _ = store.save(paths[kind].read_bytes(), ".jpg", kind, {
                "width": pixels.shape[1], "height": pixels.shape[0], "quality": "sharp", "sharpness_score": 100,
                "analysis_errors": {}, "objects": detections, "ocr_text": text, "ocr_results": results,
            }, captured_at=datetime.now(UTC))
            label_ids[saved["id"]] = kind
        rankings, hits, search_ms, local_ask_ms = asyncio.run(archive_checks(store, encoder, label_ids))
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(), "fixture_count": len(paths),
        "semantic_queries": rankings, "precision_at_1": hits / len(rankings), "recall_at_1": hits / len(rankings),
        "bus_labels": [item["name"] for item in objects], "sign_text": ocr_text,
        "sign_exact_match": ocr_text == expected, "sign_character_errors": char_errors(ocr_text, expected),
        "warm_ms": {"yolo": detection_ms, "ocr": ocr_ms, "semantic": search_ms, "local_history_ask": local_ask_ms},
    }, indent=2))


if __name__ == "__main__":
    main()
