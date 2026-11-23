import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import BytesIO
import json
import logging
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from backend import main, openai_check, semantic, vision
from backend.frame_store import utc_timestamp

CODE = "assistant-pairing-test-code"


class LocalEncoder:
    model_id = "fixture-model"

    def available(self):
        return True

    def encode_image(self, path):
        blue, _, red = cv2.imread(str(path))[0, 0]
        return [float(red), float(blue)]

    def encode_text(self, phrase):
        return [1, 0] if any(word in phrase.casefold() for word in ("couch", "canap", "red")) else [0, 1]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("LIVE_SESSION_CODE", CODE)
    monkeypatch.setenv("FRAME_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("FRAME_MIN_SHARPNESS", "0")
    monkeypatch.setenv("FRAME_RETENTION_DAYS", "0")
    monkeypatch.setattr(semantic, "SiglipEncoder", LocalEncoder)
    monkeypatch.setattr(main, "detect_objects", lambda frame: [])
    monkeypatch.setattr(main, "recognize_text", lambda frame: ("local OCR", []))
    for name in (
        "AI_VISION_MODEL", "AI_BUDGET_USD", "AI_MAX_IMAGES", "AI_MAX_OUTPUT_TOKENS",
        "AI_TIMEOUT_SECONDS", "AI_INPUT_USD_PER_MILLION", "AI_OUTPUT_USD_PER_MILLION",
        "AI_ESTIMATED_IMAGE_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vision, "_api_key", Mock(side_effect=AssertionError("Never read a real credential in tests")))
    monkeypatch.setattr(vision, "_provider", Mock(side_effect=AssertionError("Never send a real request in tests")))


def login(client):
    assert client.post("/live/session", json={"code": CODE}).status_code == 200
    return client


@pytest.fixture
def client(configured):
    with TestClient(main.app, base_url="https://testserver") as browser:
        yield login(browser)


def upload(client, red, blue):
    frame = np.full((36, 48, 3), (blue, 0, red), dtype=np.uint8)
    ok, contents = cv2.imencode(".png", frame)
    assert ok
    saved = client.post("/frames", files={"file": ("fixture.png", contents.tobytes(), "image/png")})
    assert saved.status_code == 201, saved.text
    return saved.json()["frame"]


def answer_request(client, **changes):
    return client.post("/ai/ask", json={
        "question": "Describe the scene.", "language": "en", "scope": "current", "consent": True,
        **changes,
    })


def wait_index(client, count):
    end = monotonic() + 6
    while monotonic() < end:
        status = client.get("/ai/index").json()
        if status["indexed"] == count:
            return
        sleep(0.04)
    pytest.fail(f"Indexer did not reach {count} frames: {status}")


def set_time(frame, timestamp):
    with main.app.state.frame_store._connect() as db, db:
        db.execute("UPDATE frames SET timestamp = ? WHERE id = ?", (utc_timestamp(timestamp), frame["id"]))


def test_backfill_new_images_ranking_dates_and_expiration(configured, monkeypatch):
    with TestClient(main.app, base_url="https://testserver") as previous:
        login(previous)
        red = upload(previous, 255, 0)
        blue = upload(previous, 0, 255)
        now = datetime.now(UTC)
        set_time(red, now - timedelta(days=2))
        assert previous.get("/ai/index").json()["pending"] == 2
    monkeypatch.setenv("SEMANTIC_ENABLED", "1")
    with TestClient(main.app, base_url="https://testserver") as browser:
        login(browser)
        assert browser.post("/ai/index/backfill").status_code == 202
        wait_index(browser, 2)
        new = upload(browser, 127, 127)
        wait_index(browser, 3)
        ranked = browser.get("/ai/semantic-search", params={"q": "retrouve le canap\u00e9"}).json()["items"]
        assert [item["frame_id"] for item in ranked] == [red["id"], new["id"], blue["id"]]
        assert [item["similarity"] for item in ranked] == pytest.approx([1.0, 0.707107, 0.0])
        assert ranked[0]["image_url"] == f'/live/archive/{red["id"]}/image'
        filtered = browser.get("/ai/semantic-search", params={
            "q": "couch", "start": now.date().isoformat(), "end": now.date().isoformat(),
        }).json()["items"]
        assert [item["frame_id"] for item in filtered] == [new["id"], blue["id"]]
        assert browser.get("/frames/search", params={"q": "local OCR"}).json()["items"]
        assert browser.delete(f'/frames/{blue["id"]}').status_code == 204

    monkeypatch.setenv("FRAME_RETENTION_DAYS", "1")
    with TestClient(main.app, base_url="https://testserver") as restarted:
        login(restarted)
        wait_index(restarted, 1)
        store = main.app.state.frame_store
        with store._connect() as db:
            rows = db.execute("SELECT frame_id FROM frame_embeddings").fetchall()
        assert [row["frame_id"] for row in rows] == [new["id"]]
        assert restarted.get(f'/frames/{red["id"]}').status_code == 404


def test_authorization_and_consent(configured, client):
    client.cookies.clear()
    assert answer_request(client).status_code == 401
    assert client.get("/ai/semantic-search?q=chair").status_code == 401
    assert client.post("/ai/index/backfill").status_code == 401
    assert client.get("/ai/evidence/live-test").status_code == 401
    login(client)
    assert answer_request(client, consent=False).status_code == 403
    assert answer_request(client, consent="true").status_code == 422
    assert client.post("/ai/ask", json={"question": "What?", "consent": True},
                       headers={"Origin": "https://another.example"}).status_code == 403
    vision._api_key.assert_not_called()
    vision._provider.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"question": "   "}, {"question": "x" * 801}, {"scope": "frame"},
    {"language": "de"}, {"frame_id": 9}, {"scope": "history", "start": "not a date"},
    {"scope": "history", "start": "2026-11-22", "end": "2026-11-21"},
    {"scope": "history", "live_only": True}, {"scope": "current", "live_only": "true"},
])
def test_question_validation(client, changes):
    assert answer_request(client, **changes).status_code == 422
    vision._api_key.assert_not_called()


def test_missing_key(client, monkeypatch):
    monkeypatch.setattr(vision, "_api_key", lambda: "")
    result = answer_request(client)
    assert result.status_code == 503
    assert "OPENAI_API_KEY" in result.json()["detail"]
    vision._provider.assert_not_called()


def fake_provider(monkeypatch, handler):
    from openai import AsyncOpenAI

    monkeypatch.setattr(vision, "_api_key", lambda: "local-test-key-not-real")

    def provider(key, timeout):
        assert key == "local-test-key-not-real"
        return AsyncOpenAI(
            api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=timeout,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False),
        )

    monkeypatch.setattr(vision, "_provider", provider)


def openai_reply(api="chat_completions"):
    if api == "chat_completions":
        return httpx.Response(200, json={
            "id": "chatcmpl_fixture", "object": "chat.completion", "model": "gpt-4o-mini",
            "created": 0,
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A visible scene."},
            }],
            "usage": {"prompt_tokens": 3010, "completion_tokens": 8, "total_tokens": 3018},
        })
    return httpx.Response(200, json={
        "id": "resp_fixture", "object": "response", "model": "gpt-4.1-mini",
        "status": "completed", "created_at": 0,
        "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                    "status": "completed", "content": [
                        {"type": "output_text", "text": "A visible scene.", "annotations": []},
                    ]}],
        "usage": {"input_tokens": 3010, "output_tokens": 8, "total_tokens": 3018,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    })


def read_image(content):
    url = content["image_url"]
    if isinstance(url, dict):
        url = url["url"]
    header, data = url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    jpeg = base64.b64decode(data, validate=True)
    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_genuine_sdk_payload_live_evidence_and_archive_fallback(client, monkeypatch):
    from av import VideoFrame

    archived = upload(client, 0, 255)
    live = VideoFrame.from_ndarray(np.full((60, 80, 3), (0, 0, 255), dtype=np.uint8), format="bgr24")
    capture = (live, datetime.now(UTC))
    monkeypatch.setattr(main.app.state.live, "latest_capture", lambda: capture)
    requests = []

    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return openai_reply()

    fake_provider(monkeypatch, handler)
    result = answer_request(client, language="fr", scope="current", live_only=True)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["source"] == "live" and data["answer"] == "A visible scene."
    assert data["estimated_cost_usd"] == pytest.approx(0.0004563)
    payload = requests[-1]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["store"] is False
    assert payload["max_completion_tokens"] == 300
    assert not {"input", "instructions", "max_output_tokens", "max_tokens"} & payload.keys()
    system, user = payload["messages"]
    assert system["role"] == "system" and "Answer in French" in system["content"]
    assert user["role"] == "user"
    assert [part["type"] for part in user["content"]] == ["text", "text", "image_url"]
    assert user["content"][-1]["image_url"]["detail"] == "low"
    assert read_image(user["content"][-1])[0, 0, 2] > 240
    evidence = data["evidence"][0]
    assert evidence["timestamp"] == utc_timestamp(capture[1])
    assert client.get(evidence["image_url"]).content.startswith(b"\xff\xd8")
    original = client.cookies.copy() if hasattr(client.cookies, "copy") else dict(client.cookies)
    client.cookies.clear()
    login(client)
    assert client.get(evidence["image_url"]).status_code == 404
    client.cookies.update(original)

    monkeypatch.setattr(main.app.state.live, "latest_capture", lambda: None)
    no_live = answer_request(client, scope="current", live_only=True)
    assert no_live.status_code == 404 and "No recent live camera frame" in no_live.json()["detail"]
    assert len(requests) == 1
    fallback = answer_request(client).json()
    assert fallback["source"] == "archive_fallback"
    assert fallback["evidence"][0]["frame_id"] == archived["id"]
    assert read_image(requests[-1]["messages"][1]["content"][-1])[0, 0, 0] > 240
    with main.app.state.frame_store._connect() as db:
        charges = db.execute("SELECT input_tokens, output_tokens, status FROM vision_usage").fetchall()
    assert [tuple(row) for row in charges] == [(3010, 8, "completed")] * 2


def test_alternate_model_preserves_responses_format(client, monkeypatch):
    monkeypatch.setenv("AI_VISION_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("AI_INPUT_USD_PER_MILLION", "0.40")
    monkeypatch.setenv("AI_OUTPUT_USD_PER_MILLION", "1.60")
    monkeypatch.setenv("AI_ESTIMATED_IMAGE_TOKENS", "3200")
    upload(client, 0, 255)

    def handler(request):
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        assert payload["store"] is False and payload["max_output_tokens"] == 300
        assert not {"messages", "max_completion_tokens"} & payload.keys()
        assert "Answer in English" in payload["instructions"]
        image = payload["input"][0]["content"][-1]
        assert image["type"] == "input_image" and image["detail"] == "low"
        assert read_image(image)[0, 0, 0] > 240
        return openai_reply("responses")

    fake_provider(monkeypatch, handler)
    response = answer_request(client)
    assert response.status_code == 200
    assert response.json()["estimated_cost_usd"] == pytest.approx(0.0012168)


def test_history_auto_selection_and_exact_frame(configured, monkeypatch):
    monkeypatch.setenv("SEMANTIC_ENABLED", "1")
    monkeypatch.setenv("AI_MAX_IMAGES", "1")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return openai_reply()

    fake_provider(monkeypatch, handler)
    with TestClient(main.app, base_url="https://testserver") as browser:
        login(browser)
        red = upload(browser, 255, 0)
        blue = upload(browser, 0, 255)
        wait_index(browser, 2)
        history = answer_request(
            browser, question="Quand ai-je vu un canap\u00e9 pour la derni\u00e8re fois ?", language="fr", scope="auto"
        ).json()
        assert history["source"] == "history"
        assert [item["frame_id"] for item in history["evidence"]] == [red["id"]]
        assert sum(item["type"] == "image_url" for item in requests[-1]["messages"][1]["content"]) == 1
        exact = answer_request(browser, scope="frame", frame_id=blue["id"]).json()
        assert exact["evidence"][0]["frame_id"] == blue["id"]


@pytest.mark.parametrize(("kind", "expected", "detail"), [
    ("rate", 429, "rate limit"), ("timeout", 504, "timed out"), ("network", 502, "connect"),
    ("credentials", 502, "credentials"), ("provider", 502, "could not complete"),
])
def test_provider_errors(client, monkeypatch, kind, expected, detail):
    upload(client, 0, 255)
    calls = []

    def handler(request):
        calls.append(1)
        if kind == "timeout":
            raise httpx.ReadTimeout("fixture timeout", request=request)
        if kind == "network":
            raise httpx.ConnectError("fixture network error", request=request)
        status = {"rate": 429, "credentials": 401, "provider": 500}[kind]
        return httpx.Response(status, json={"error": {"message": "private provider body"}})

    fake_provider(monkeypatch, handler)
    result = answer_request(client)
    assert result.status_code == expected and detail in result.json()["detail"]
    assert "private provider body" not in result.text
    assert len(calls) == 1
    with main.app.state.frame_store._connect() as db:
        charge = db.execute("SELECT status, estimated_usd FROM vision_usage").fetchone()
    if kind in ("rate", "credentials"):
        assert tuple(charge) == ("rejected", 0)
    else:
        assert charge["status"] == "uncertain" and charge["estimated_usd"] > 0


@pytest.mark.parametrize("recognized", [True, False])
def test_provider_400_is_sanitized_and_not_retried(client, monkeypatch, caplog, recognized):
    upload(client, 0, 255)
    question = "Private fixture prompt, never log this"
    fake_key = "local-test-key-not-real"
    private_field = "private_fixture_field"
    calls = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        image = json.loads(request.content)["messages"][1]["content"][-1]["image_url"]["url"]
        calls.append(image)
        return httpx.Response(400, json={"error": {
            "type": "invalid_request_error" if recognized else private_field,
            "code": "invalid_image_url" if recognized else private_field,
            "param": "messages[1].content[2].image_url.url" if recognized else private_field,
            "message": f"Echoed request data: {question} {fake_key} {image}",
        }})

    fake_provider(monkeypatch, handler)
    result = answer_request(client, question=question)
    assert result.status_code == 502
    assert "HTTP 400" in result.json()["detail"]
    assert len(calls) == 1
    assert "HTTP 400 api=chat_completions" in caplog.text
    if recognized:
        diagnostic = "The endpoint rejected the image data URL."
        assert f"type=invalid_request_error code=invalid_image_url param=image_url message={diagnostic}" in caplog.text
        assert diagnostic in result.text
    else:
        assert "type=- code=- param=- message=No safely identifiable provider detail." in caplog.text
    for private in (question, fake_key, private_field, calls[0], calls[0].split(",", 1)[1]):
        assert private not in caplog.text
        assert private not in result.text
    with main.app.state.frame_store._connect() as db:
        charge = db.execute("SELECT status, estimated_usd FROM vision_usage").fetchone()
    assert tuple(charge) == ("rejected", 0)


def test_provider_html_400_does_not_log_response_body(client, monkeypatch, caplog):
    upload(client, 0, 255)
    private = "private-fixture-response-data"

    def handler(request):
        return httpx.Response(
            400, text=f"<!DOCTYPE html><html>{private}</html>",
            headers={"Content-Type": "text/html; charset=utf-8", "X-Request-ID": private},
        )

    fake_provider(monkeypatch, handler)
    result = answer_request(client)
    assert result.status_code == 502
    assert "HTML error page" in result.json()["detail"]
    assert "Response format: html" in result.json()["detail"]
    assert "body=html request_id=-" in caplog.text
    assert private not in result.text and private not in caplog.text
    assert "<html>" not in caplog.text


def test_service_tier_400_labels_field_and_logs_valid_request_id(client, monkeypatch, caplog):
    upload(client, 0, 255)
    request_id = "req_" + "0123456789abcdef" * 2
    private = "private-fixture-tier"

    def handler(request):
        return httpx.Response(400, json={"error": {
            "type": "invalid_request_error",
            "param": "service_tier",
            "message": f"Invalid `service_tier` argument: {private} is not allowed for this project.",
        }}, headers={"X-Request-ID": request_id})

    fake_provider(monkeypatch, handler)
    result = answer_request(client)
    assert result.status_code == 502
    assert "field: service_tier" in result.json()["detail"]
    assert "does not allow the requested or resolved service tier" in result.json()["detail"]
    assert f"Request ID: {request_id}" in result.json()["detail"]
    assert f"body=api_json request_id={request_id}" in caplog.text
    assert "type=invalid_request_error code=- param=service_tier" in caplog.text
    assert private not in result.text and private not in caplog.text


@pytest.mark.parametrize("http_status", [200, 404])
def test_openai_check_only_retrieves_model_metadata(configured, monkeypatch, capsys, http_status):
    calls = []
    omitted = "private mock provider error"

    def handler(request):
        calls.append((request.method, request.url.path, request.content))
        if http_status == 404:
            return httpx.Response(404, json={"error": {"code": "model_not_found", "message": omitted}})
        return httpx.Response(200, json={
            "id": "gpt-4o-mini", "object": "model", "created": 0, "owned_by": "openai",
        })

    fake_provider(monkeypatch, handler)
    monkeypatch.setattr(openai_check, "_api_key", vision._api_key)
    monkeypatch.setattr(openai_check, "_provider", vision._provider)
    assert asyncio.run(openai_check.check_model("gpt-4o-mini")) == (0 if http_status == 200 else 1)
    assert calls == [("GET", "/v1/models/gpt-4o-mini", b"")]
    output = capsys.readouterr().out
    assert ("lookup succeeded" if http_status == 200 else "HTTP 404") in output
    assert omitted not in output and "local-test-key-not-real" not in output


@pytest.mark.parametrize("http_status", [200, 403])
def test_openai_check_lists_model_metadata(configured, monkeypatch, capsys, http_status):
    calls = []
    private = "sk-do-not-print-from-response"

    def handler(request):
        calls.append((request.method, str(request.url), request.content))
        if http_status == 403:
            return httpx.Response(403, json={"error": {"message": private}})
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "fixture-owner"}
            for name in ("gpt-4.1-mini", "gpt-4.1-mini", private, "private\ntext")
        ]})

    fake_provider(monkeypatch, handler)
    monkeypatch.setattr(openai_check, "_api_key", vision._api_key)
    monkeypatch.setattr(openai_check, "_provider", vision._provider)
    assert asyncio.run(openai_check.check_model(None)) == (0 if http_status == 200 else 1)
    assert calls == [("GET", "https://api.openai.com/v1/models", b"")]
    output = capsys.readouterr().out
    if http_status == 200:
        assert output.splitlines().count("gpt-4.1-mini") == 1
        assert "not all models support images" in output
    else:
        assert "OpenAI model list: HTTP 403" in output
    for omitted in (private, "private\ntext", "fixture-owner", "local-test-key-not-real"):
        assert omitted not in output


def test_openai_check_rejects_key_like_model_without_request(monkeypatch, capsys):
    never = Mock(side_effect=AssertionError("No credential lookup or request expected"))
    monkeypatch.setattr(openai_check, "_api_key", never)
    monkeypatch.setattr(openai_check, "_provider", never)
    assert asyncio.run(openai_check.check_model("sk-fake-key-for-test")) == 1
    assert "sk-fake-key-for-test" not in capsys.readouterr().out
    never.assert_not_called()


@pytest.mark.parametrize("compare", [False, True])
def test_openai_check_compares_empty_400_without_printing_private_values(configured, monkeypatch, capsys, compare):
    calls = []
    private_project = "private-project-fixture"
    private_body = "private response body"

    def handler(request):
        assert request.method == "GET" and request.content == b""
        assert str(request.url) == "https://api.openai.com/v1/models/gpt-4o-mini"
        if request.headers.get("openai-project") == private_project:
            calls.append("sdk")
            return httpx.Response(400)
        assert "openai-organization" not in request.headers
        if request.headers["authorization"] == "Bearer local-test-key-not-real":
            calls.append("configured key")
            return httpx.Response(200, text=private_body)
        assert request.headers["authorization"] == "Bearer deliberately-invalid-diagnostic-token"
        calls.append("dummy key")
        return httpx.Response(401, text=private_body)

    class MockClient(httpx.AsyncClient):
        def __init__(self, **options):
            options.setdefault("transport", httpx.MockTransport(handler))
            super().__init__(**options)

    fake_provider(monkeypatch, handler)
    monkeypatch.setattr(openai_check, "_api_key", vision._api_key)
    monkeypatch.setattr(openai_check, "_provider", vision._provider)
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    monkeypatch.setenv("OPENAI_PROJECT_ID", private_project)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)

    assert asyncio.run(openai_check.check_model("gpt-4o-mini", compare=compare)) == 1
    output = capsys.readouterr().out
    assert "HTTP 400 type=- code=- param=- body=empty" in output
    if compare:
        assert calls == ["sdk", "configured key", "dummy key"]
        assert "OPENAI_ORG_ID=unset, OPENAI_PROJECT_ID=set" in output
        assert "Minimal HTTP lookup (configured key): HTTP 200 body=present" in output
        assert "Minimal HTTP lookup (dummy key): HTTP 401 body=present" in output
    else:
        assert calls == ["sdk"]
        assert "Minimal HTTP lookup" not in output
    for private in (private_project, private_body, "local-test-key-not-real"):
        assert private not in output


@pytest.mark.parametrize("key", ["fake key with spaces", '"quoted"', "control\ncharacter", "inval\u00edd"])
def test_openai_check_rejects_malformed_credential_without_request(monkeypatch, capsys, key):
    never = Mock(side_effect=AssertionError("No request expected"))
    monkeypatch.setattr(openai_check, "_api_key", lambda: key)
    monkeypatch.setattr(openai_check, "_provider", never)
    assert asyncio.run(openai_check.check_model("gpt-4o-mini", compare=True)) == 1
    output = capsys.readouterr().out
    assert "Re-enter just the key" in output and key not in output
    never.assert_not_called()


def test_official_sdk_configuration(monkeypatch):
    import openai

    constructor = Mock()
    monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
    sdk_log = logging.getLogger("openai._base_client")
    monkeypatch.setattr(sdk_log, "level", logging.DEBUG)
    vision._provider("fixture-key", 6)
    assert sdk_log.level == logging.WARNING
    constructor.assert_called_once_with(
        api_key="fixture-key", timeout=6, max_retries=0, base_url="https://api.openai.com/v1",
    )


def test_local_budget_blocks_before_provider_call(client, monkeypatch):
    upload(client, 0, 255)
    monkeypatch.setenv("AI_BUDGET_USD", "0")
    handler = Mock(side_effect=AssertionError("Budget stopped request"))
    fake_provider(monkeypatch, handler)
    result = answer_request(client)
    assert result.status_code == 429 and "budget exhausted" in result.json()["detail"]
    handler.assert_not_called()


def test_duplicate_request_in_flight(client, monkeypatch):
    upload(client, 0, 255)
    waiting, release = Event(), Event()

    async def handler(request):
        waiting.set()
        await asyncio.to_thread(release.wait, 5)
        return openai_reply()

    fake_provider(monkeypatch, handler)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(answer_request, client)
        try:
            assert waiting.wait(4)
            assert answer_request(client).status_code == 409
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200


def test_siglip_text_prep_ignores_serialized_padding(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "<eos>": 1, "<unk>": 3, "bus": 10}, unk_token="<unk>"
    ))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.enable_padding(length=64, pad_id=0, pad_token="<pad>")
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    captured = []

    class TextGraph:
        def get_inputs(self):
            return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask")]

        def get_outputs(self):
            return [SimpleNamespace(name="text_embeds")]

        def run(self, names, feed):
            captured.append(feed)
            return [np.array([[0.5, 0.5]])]

    encoder = semantic.SiglipEncoder(tmp_path)
    monkeypatch.setattr(encoder, "_session", lambda name: TextGraph())
    assert encoder.encode_text("BUS").tolist() == [0.5, 0.5]
    assert captured[0]["input_ids"].tolist() == [[10, 1] + [0] * 62]
    assert captured[0]["attention_mask"].tolist() == [[1, 1] + [0] * 62]
