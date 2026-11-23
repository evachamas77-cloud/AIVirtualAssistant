import asyncio
import json
from threading import Event
from time import monotonic

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from backend import main
from backend.live import COOKIE, LiveService, Peer

CODE = "local-webrtc-test-code"


def test_live_pages_and_disabled_runtime():
    with TestClient(main.app, base_url="https://testserver") as client:
        dashboard = client.get("/").text
        assert "Live monitor" in dashboard and "/ui/dashboard.js" in dashboard
        assert 'id="assistant"' in dashboard and "/ui/manifest.webmanifest" not in dashboard
        phone = client.get("/phone")
        for label in ("Start Streaming", "Start Assistant", "Stop Assistant", "Talk Again", "Repeat Answer"):
            assert label in phone.text
        assert "microphone=(self)" in phone.headers["Permissions-Policy"]
        worker = client.get("/sw.js")
        assert worker.status_code == 200
        assert worker.headers["content-type"].startswith("text/javascript")
        manifest = json.loads(client.get("/ui/manifest.webmanifest").text)
        assert manifest["start_url"] == manifest["scope"] == manifest["id"] == "/phone"
        assert manifest["display"] == "standalone" and manifest["short_name"] == "Visual Assistant"
        for icon in manifest["icons"]:
            png = client.get(icon["src"]).content
            assert png.startswith(b"\x89PNG")
            assert (int.from_bytes(png[16:20]), int.from_bytes(png[20:24])) == tuple(map(int, icon["sizes"].split("x")))
        for name in ("phone", "voice-session", "phone-camera", "mobile-strings", "mobile-question"):
            path = f"/ui/{name}.js"
            assert path in worker.text and client.get(path).status_code == 200
        assert client.get("/live/status").status_code == 503
        assert client.get("/frames").status_code == 200


def test_live_pairing_and_lan_archive_protection(monkeypatch):
    pytest.importorskip("aiortc")
    monkeypatch.setenv("LIVE_SESSION_CODE", CODE)
    with TestClient(main.app, base_url="https://testserver") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/frames").status_code == 401
        assert client.get("/live/status").status_code == 401
        assert client.post("/live/session", json={"code": "incorrect"}).status_code == 401
        assert client.post(
            "/live/session", json={"code": CODE}, headers={"Origin": "https://other.example"}
        ).status_code == 403
        response = client.post("/live/session", json={"code": CODE})
        assert response.status_code == 200
        assert "httponly" in response.headers["set-cookie"].lower()
        assert "secure" in response.headers["set-cookie"].lower()
        assert "samesite=strict" in response.headers["set-cookie"].lower()
        assert CODE not in response.headers["set-cookie"]
        assert client.get("/frames").headers["cache-control"] == "no-store"
        assert client.get("/live/status").json() == {"camera": None, "viewers": 0}
        assert client.get("http://testserver/frames").status_code == 403
        assert client.post("/live/offer", json={
            "role": "camera", "type": "offer", "sdp": "m=audio 9 UDP/TLS/RTP/SAVPF 111",
        }).status_code == 400
        assert client.post("/live/offer", json={
            "role": "viewer", "type": "offer", "sdp": "m=video 9 UDP/TLS/RTP/SAVPF 96",
        }).status_code == 409
        main.app.state.live.sessions[client.cookies[COOKIE]] = monotonic() - 1
        assert client.get("/live/status").status_code == 401


def test_sampler_respects_interval_and_drops_while_busy(monkeypatch):
    class EndOfVideo(Exception):
        pass

    clock = 0
    played = []

    class Track:
        frames = iter([(0, True), (0.1, True), (0.9, True), (1, True), (2, False), (3, True)])

        async def recv(self):
            nonlocal clock
            await asyncio.sleep(0)
            frame = next(self.frames, None)
            if frame is None:
                service.peers.pop("camera")
                raise EndOfVideo()
            clock, finish_previous = frame
            if finish_previous:
                release.set()
                await asyncio.sleep(0)
            return clock

    async def run():
        nonlocal service, release
        service = LiveService(None)
        service.interval = 1
        service.media_error = EndOfVideo
        release = asyncio.Event()
        peer = Peer("camera", "", "camera", "", None, 0, track=Track())
        service.peers[peer.id] = peer

        async def process(_, frame):
            played.append(frame)
            release.clear()
            await release.wait()

        monkeypatch.setattr(service, "_process", process)
        await service._sample(peer)
        release.set()
        await service.work
        assert peer.sampled == 3
        assert played == [0, 1, 3]

    service = release = None
    monkeypatch.setattr("backend.live.monotonic", lambda: clock)
    asyncio.run(run())


def test_webrtc_loopback_relays_samples_archives_and_disconnects(monkeypatch, isolate_frame_storage):
    rtc = pytest.importorskip("aiortc")
    av = pytest.importorskip("av")
    monkeypatch.setenv("LIVE_SESSION_CODE", CODE)
    monkeypatch.setenv("LIVE_SAMPLE_SECONDS", "0.2")
    in_model, release = Event(), Event()

    def detect(image):
        in_model.set()
        release.wait(timeout=12)
        return [{"name": "book", "confidence": 0.9, "bbox": [2, 2, 20, 20]}]

    monkeypatch.setattr(main, "detect_objects", detect)
    monkeypatch.setattr(main, "recognize_text", lambda image: ("LOOPBACK", []))

    class SyntheticCamera(rtc.VideoStreamTrack):
        async def recv(self):
            pts, time_base = await self.next_timestamp()
            image = np.zeros((96, 128, 3), dtype=np.uint8)
            image[:, ::4] = 255
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = pts, time_base
            return frame

    async def negotiate(client, pc, role):
        await pc.setLocalDescription(await pc.createOffer())
        reply = await client.post("/live/offer", json={
            "role": role, "type": "offer", "sdp": pc.localDescription.sdp,
        })
        assert reply.status_code == 200, reply.text
        answer = reply.json()
        await pc.setRemoteDescription(rtc.RTCSessionDescription(answer["sdp"], answer["type"]))
        return answer["peer_id"]

    async def run():
        camera = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        viewer = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        camera.addTransceiver(SyntheticCamera(), direction="sendonly")
        viewer.addTransceiver("video", direction="recvonly")
        received = asyncio.get_running_loop().create_future()

        @viewer.on("track")
        def on_video(track):
            received.set_result(track)

        clients = {"base_url": "https://testserver", "transport": httpx.ASGITransport(app=main.app)}
        async with main.app.router.lifespan_context(main.app):
            async with httpx.AsyncClient(**clients) as sender, httpx.AsyncClient(**clients) as dashboard:
                try:
                    for client in (sender, dashboard):
                        assert (await client.post("/live/session", json={"code": CODE})).status_code == 200
                    camera_id = await negotiate(sender, camera, "camera")
                    await negotiate(dashboard, viewer, "viewer")
                    track = await asyncio.wait_for(received, 3)
                    for _ in range(9):
                        frame = await asyncio.wait_for(track.recv(), 3)
                        assert frame.width == 128 and frame.height == 96

                    assert in_model.is_set()
                    live = (await dashboard.get("/live/status")).json()
                    assert live["camera"]["sampled"] == 1 and live["viewers"] == 1
                    assert live["camera"]["state"] == "connected"
                    release.set()

                    async with asyncio.timeout(5):
                        while True:
                            page = (await dashboard.get("/frames")).json()
                            if page["items"]:
                                break
                            await asyncio.sleep(0.05)
                    stored = page["items"][0]
                    assert stored["ocr_text"] == "LOOPBACK"
                    assert stored["objects"][0]["name"] == "book"
                    assert (isolate_frame_storage / stored["image_path"]).is_file()
                    image = await dashboard.get(f'/live/archive/{stored["id"]}/image')
                    assert image.status_code == 200 and image.content.startswith(b"\xff\xd8")
                    assert image.headers["cache-control"] == "no-store"
                    assert (await dashboard.delete(f"/live/peers/{camera_id}")).status_code == 404
                    assert (await sender.delete(f"/live/peers/{camera_id}")).status_code == 204
                    assert (await dashboard.get("/live/status")).json() == {"camera": None, "viewers": 0}
                finally:
                    release.set()
                    await camera.close()
                    await viewer.close()

    asyncio.run(asyncio.wait_for(run(), 25))
