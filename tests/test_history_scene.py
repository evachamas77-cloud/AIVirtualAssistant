import asyncio
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fastapi import Response

from backend import ai_routes, history, scene, vision
from backend.ai_routes import Question, date_window
from backend.frame_store import FrameSettings, FrameStore


@pytest.fixture
def store(tmp_path):
    store = FrameStore(FrameSettings(tmp_path, 0, 0, 0, 3600))
    store.initialize()
    return store


def record(store, hour, *, label="laptop", text="", day=28):
    row, reason = store.save(b"fixture", ".jpg", f"{day}:{hour}:{label}:{text}", {
        "width": 100, "height": 100, "quality": "sharp", "sharpness_score": 100,
        "objects": [{"name": label, "confidence": .95, "bbox": [30, 10, 70, 90]}] if label else [],
        "ocr_results": [], "ocr_text": text, "analysis_errors": {},
    }, captured_at=datetime(2026, 11, day, hour, 15, tzinfo=UTC))
    assert reason is None
    return row


def test_rebuild_legacy_object_index_is_repeatable_and_cascades(store):
    original = record(store, 8)
    with sqlite3.connect(store.db_path) as db:
        db.execute("DROP TABLE frame_objects")
        db.execute("PRAGMA user_version=0")
    store.initialize()
    store.initialize()
    assert store.observations(label="laptop") == [original]
    assert store.get(original["id"])["timestamp"] == original["timestamp"]
    assert store.delete(original["id"])
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM frame_objects").fetchone()[0] == 0


def test_beirut_date_and_dst_boundaries():
    assert date_window("2026-11-28", "2026-11-28") == ("2026-11-27T22:00:00.000000Z", "2026-11-28T21:59:59.999999Z")
    assert history.beirut_time("2026-07-20T08:00:00Z").utcoffset() == timedelta(hours=3)
    assert history.beirut_time("2026-11-28T08:00:00Z").utcoffset() == timedelta(hours=2)
    assert history.midnight(date(2026, 3, 30)) - history.midnight(date(2026, 3, 29)) == timedelta(hours=23)
    assert history.midnight(date(2026, 10, 25)) - history.midnight(date(2026, 10, 24)) == timedelta(hours=25)


def test_relative_time_crosses_beirut_midnight():
    now = datetime(2026, 11, 27, 22, 15, tzinfo=UTC)  # 00:15 in Beirut
    assert history.relative_window("today", now)[0] == "2026-11-27T22:00:00.000000Z"
    assert history.relative_window("yesterday", now)[1] == "2026-11-27T21:59:59.999999Z"
    assert history.relative_window("ce matin", now)[1] == "2026-11-28T09:59:59.999999Z"
    assert history.relative_window("two hours ago", now) == ("2026-11-27T19:45:00.000000Z", "2026-11-27T20:45:00.000000Z")


@pytest.mark.parametrize(("phrase", "language", "selection", "part"), [
    ("When did you first see a laptop?", "en", "early", "earliest recorded sighting"),
    ("When did you last see my laptop?", "en", "late", "can't confirm the item's identity"),
    ("When did you last see my phone?", "en", None, "couldn't find a recorded sighting"),
    ("When did you see the laptop at home?", "en", None, "archive does not record named locations"),
    ("Quand as-tu vu mon ordinateur portable pour la premi\u00e8re fois ?", "fr", "early", "10 h 15"),
    ('What was written on the sign "ROOM 204" earlier?', "en", "sign", "ROOM 204"),
    ('What was written on the sign "ROOM 900" earlier?', "en", None, "couldn't find matching text"),
    ("Can I cross?", "en", None, "can't estimate distance"),
])
def test_local_answers_do_not_contact_provider(store, monkeypatch, phrase, language, selection, part):
    records = {"early": record(store, 8), "late": record(store, 13), "sign": record(store, 15, label=None, text="ROOM 204")}
    denied = Mock(side_effect=AssertionError("Local answers must not use credentials or retrieve cloud images"))
    monkeypatch.setattr(vision, "_api_key", denied)
    service = vision.VisionService(store, None, None)
    result = asyncio.run(service.ask(Question(question=phrase, language=language, consent=True), "fixture"))
    assert part in result["answer"]
    assert result["estimated_cost_usd"] == 0
    assert [item["frame_id"] for item in result["evidence"]] == ([records[selection]["id"]] if selection else [])
    if selection:
        assert "+02:00" in result["evidence"][0]["timestamp_beirut"]
    assert "university" not in result["answer"].lower()
    denied.assert_not_called()


def test_hybrid_search_limits_semantic_rerank_to_object_matches(store):
    backpack = record(store, 11, label="backpack")
    record(store, 12, label="laptop")

    async def ranked(phrase, start, end, limit, *, candidate_ids):
        assert "red backpack" in phrase
        assert candidate_ids == [backpack["id"]]
        return [{"frame_id": backpack["id"], "similarity": .4}]

    service = vision.VisionService(store, SimpleNamespace(search=ranked, available=lambda: True), None)
    phrase = "Have you seen a red backpack?"
    query = history.interpret(phrase)

    async def run():
        plan = await service._history_plan(query, phrase, None, None)
        return await service._history_refs(query, phrase, 3, plan)

    assert asyncio.run(run())[0]["match"] == "object_and_semantic"


def test_disconnect_releases_an_inflight_ask(monkeypatch):
    monkeypatch.setattr(ai_routes, "authorize", lambda *args, **kwargs: "fixture")

    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def fake_ask(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def receive():
            await started.wait()
            return {"type": "http.disconnect"}

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(vision=SimpleNamespace(ask=fake_ask))), receive=receive)
        result = await ai_routes.ask(request, Response(), Question(question="What is in front of me?", consent=True))
        assert result.status_code == 499
        assert stopped.is_set()

    asyncio.run(run())


def test_scene_confirmations_priority_cooldown_and_reentry():
    tracker = scene.SceneTracker(scene.SceneSettings(.65, 2, 10))
    chair = {"name": "chair", "confidence": .8, "bbox": [40, 2, 60, 90]}
    bench = {"name": "bench", "confidence": .82, "bbox": [2, 2, 14, 90]}

    def apply(at, *objects):
        tracker.observe({"width": 100, "objects": objects}, at)

    apply(0, chair, bench)
    assert tracker.latest(0) is None
    apply(1, chair, bench)
    assert tracker.latest(1) == {"label": "chair", "region": "center", "sequence": 1}
    apply(2, chair)
    assert tracker.latest(2)["sequence"] == 1
    apply(3)
    apply(4)
    apply(5, chair)
    apply(6, chair)
    assert tracker.latest(6)["sequence"] == 1
    apply(12, chair)
    assert tracker.latest(12)["sequence"] == 2
    assert tracker.latest(19) is None


CASES = json.loads((Path(__file__).parent.parent / "evaluation" / "questions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[item["id"] for item in CASES])
def test_evaluation_question_ground_truth(case):
    query = history.interpret(case["question"])
    assert query.history == (case["scope"] == "history")
    if "mode" in case:
        assert query.mode == case["mode"]
    if "label" in case:
        assert query.label == case["label"]
    if case.get("safety"):
        answer = scene.safety_limit(case["question"], case["language"])
        assert answer and "meter" not in answer
