import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
import logging
import math
import os
from pathlib import Path
import secrets
from time import monotonic
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .scene import SceneSettings, SceneTracker

router = APIRouter()
logger = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parents[1] / "web"
COOKIE = "avi_live_session"
SESSION_SECONDS = 8 * 3600
PEER_TIMEOUT = 25
MAX_VIEWERS = 3


class Login(BaseModel):
    code: str = Field(min_length=1, max_length=256)


class Offer(BaseModel):
    role: Literal["camera", "viewer"]
    sdp: str = Field(min_length=1, max_length=65536)
    type: Literal["offer"]
    label: str = Field(default="Phone camera", max_length=48)


@dataclass
class Peer:
    id: str
    owner: str
    role: str
    label: str
    pc: Any
    activity: float
    source: Any = None
    track: Any = None
    sampler: asyncio.Task | None = None
    sampled: int = 0
    stored: int = 0
    last_frame_id: int | None = None
    last_result: str | None = None
    capture: tuple[Any, datetime] | None = None
    analysis: tuple[datetime, dict] | None = None
    changes: SceneTracker | None = None


class LiveService:
    def __init__(self, process_frame):
        self.code = os.getenv("LIVE_SESSION_CODE", "")
        self.sessions = {}
        self.peers: dict[str, Peer] = {}
        self.camera: Peer | None = None
        self.work: asyncio.Task | None = None
        self.maintenance: asyncio.Task | None = None
        self.process_frame = process_frame
        self.available = False
        self.closing = False
        if not self.code:
            return
        if not 16 <= len(self.code) <= 256:
            raise ValueError("LIVE_SESSION_CODE must contain 16 to 256 characters")
        self.interval = float(os.getenv("LIVE_SAMPLE_SECONDS", "2"))
        self.scene_settings = SceneSettings.from_env()
        if not math.isfinite(self.interval) or self.interval < 0.2:
            raise ValueError("LIVE_SAMPLE_SECONDS must be finite and at least 0.2")
        try:
            from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription
            from aiortc.contrib.media import MediaRelay
            from aiortc.mediastreams import MediaStreamError
        except (ImportError, OSError):
            logger.exception("Live WebRTC runtime unavailable")
            return
        self.new_pc = partial(RTCPeerConnection, RTCConfiguration(iceServers=[]))
        self.description = RTCSessionDescription
        self.media_error = MediaStreamError
        self.relay = MediaRelay()
        self.available = True
        self.maintenance = asyncio.create_task(self._reap_idle())

    def authenticated(self, owner):
        return self.sessions.get(owner, 0) > monotonic()

    def login(self, code, previous):
        if not self.code or not self.available:
            raise HTTPException(503, detail="Live streaming is not configured on this server")
        if not secrets.compare_digest(self.code.encode("utf-8"), code.encode("utf-8")):
            raise HTTPException(401, detail="Incorrect pairing code")
        now = monotonic()
        self.sessions = {key: expiry for key, expiry in self.sessions.items() if expiry > now}
        if previous in self.sessions:
            self.sessions[previous] = now + SESSION_SECONDS
            return previous
        if len(self.sessions) >= 64:
            raise HTTPException(429, detail="Too many live sessions")
        owner = secrets.token_urlsafe(32)
        self.sessions[owner] = now + SESSION_SECONDS
        return owner

    async def create_peer(self, owner: str, offer: Offer):
        media = [line for line in offer.sdp.splitlines() if line.startswith("m=")]
        if len(media) != 1 or not media[0].startswith("m=video "):
            raise HTTPException(400, detail="Offer must contain one video track and no audio")
        if self.closing:
            raise HTTPException(503, detail="Live server is shutting down")
        if offer.role == "camera" and self.camera is not None:
            raise HTTPException(409, detail="A camera is already connected")
        if offer.role == "viewer":
            if self.camera is None or self.camera.source is None:
                raise HTTPException(409, detail="No camera is streaming")
            if sum(peer.role == "viewer" for peer in self.peers.values()) >= MAX_VIEWERS:
                raise HTTPException(429, detail="Viewer limit reached")

        pc = self.new_pc()
        peer = Peer(secrets.token_urlsafe(18), owner, offer.role, offer.label.strip(), pc, monotonic())
        self.peers[peer.id] = peer
        source_camera = self.camera
        if peer.role == "camera":
            self.camera = peer

        @pc.on("track")
        def on_track(track):
            peer.source = track

            @track.on("ended")
            async def on_ended():
                await self.close_peer(peer.id)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                await self.close_peer(peer.id)

        try:
            async with asyncio.timeout(15):
                await pc.setRemoteDescription(self.description(sdp=offer.sdp, type=offer.type))
                if peer.role == "camera":
                    if peer.source is None:
                        raise HTTPException(400, detail="Camera offer must send video")
                    peer.track = self.relay.subscribe(peer.source, buffered=False)
                    peer.sampler = asyncio.create_task(self._sample(peer))
                else:
                    if peer.source is not None:
                        raise HTTPException(400, detail="Viewer offer must only receive video")
                    if self.camera is not source_camera:
                        raise HTTPException(409, detail="Camera disconnected")
                    peer.track = self.relay.subscribe(source_camera.source, buffered=False)
                    pc.addTrack(peer.track)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                if peer.id not in self.peers:
                    raise HTTPException(409, detail="Peer disconnected during negotiation")
        except HTTPException:
            await self.close_peer(peer.id)
            raise
        except asyncio.CancelledError:
            await self.close_peer(peer.id)
            raise
        except Exception:
            logger.exception("WebRTC negotiation failed")
            await self.close_peer(peer.id)
            raise HTTPException(400, detail="Could not negotiate the WebRTC connection") from None
        return {"peer_id": peer.id, "sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def _sample(self, peer: Peer):
        next_sample = 0.0
        try:
            while peer.id in self.peers:
                frame = await peer.track.recv()
                now = peer.activity = monotonic()
                captured_at = datetime.now(UTC)
                peer.capture = (frame, captured_at)
                if now < next_sample or (self.work is not None and not self.work.done()):
                    continue
                next_sample = now + self.interval
                peer.sampled += 1
                # The relay is unbuffered and only this task is allowed to dispatch analysis.
                self.work = asyncio.create_task(self._process(peer, frame, captured_at))
        except self.media_error:
            pass
        finally:
            if peer.id in self.peers:
                # Detach rather than await close, which also joins this sampler.
                asyncio.create_task(self.close_peer(peer.id))

    async def _process(self, peer, frame, captured_at):
        try:
            result, reason, analysis = await run_in_threadpool(self.process_frame, frame, captured_at)
            peer.last_result = reason or "saved"
            if result is not None:
                peer.stored += 1
                peer.last_frame_id = result["id"]
            if self.camera is peer:
                peer.analysis = (captured_at, analysis)
                if peer.changes is None:
                    peer.changes = SceneTracker(self.scene_settings)
                if (datetime.now(UTC) - captured_at).total_seconds() <= 5:
                    peer.changes.observe(analysis, monotonic())
        except Exception:
            peer.last_result = "processing_error"
            logger.exception("Live sample processing failed")

    async def close_peer(self, peer_id: str):
        peer = self.peers.pop(peer_id, None)
        if peer is None:
            return
        viewers = []
        if peer is self.camera:
            self.camera = None
            viewers = [p.id for p in self.peers.values() if p.role == "viewer"]
        if peer.sampler is not None:
            peer.sampler.cancel()
            await asyncio.gather(peer.sampler, return_exceptions=True)
        if peer.track is not None:
            peer.track.stop()
        await asyncio.gather(peer.pc.close(), *(self.close_peer(pid) for pid in viewers))

    async def _reap_idle(self):
        while True:
            await asyncio.sleep(3)
            now = monotonic()
            for peer in list(self.peers.values()):
                stale = now - peer.activity > PEER_TIMEOUT
                if not self.authenticated(peer.owner) or (
                    stale and (peer.role == "camera" or peer.pc.connectionState != "connected")
                ):
                    await self.close_peer(peer.id)

    async def shutdown(self):
        self.closing = True
        if self.maintenance is not None:
            self.maintenance.cancel()
            await asyncio.gather(self.maintenance, return_exceptions=True)
        await asyncio.gather(*(self.close_peer(pid) for pid in list(self.peers)))
        if self.work is not None:
            await self.work

    def status(self):
        camera = self.camera
        return {
            "camera": None if camera is None else {
                "id": camera.id, "label": camera.label or "Phone camera",
                "state": camera.pc.connectionState, "sampled": camera.sampled,
                "stored": camera.stored, "last_frame_id": camera.last_frame_id,
                "last_result": camera.last_result,
            },
            "viewers": sum(peer.role == "viewer" for peer in self.peers.values()),
        }

    def latest_capture(self):
        camera = self.camera
        if camera is not None and monotonic() - camera.activity < 5:
            return camera.capture
        return None

    def recent_analysis(self, captured_at):
        analysis = self.camera.analysis if self.camera else None
        return analysis[1] if analysis and 0 <= (captured_at - analysis[0]).total_seconds() <= 2 else None


def local_client(request: Request) -> bool:
    return request.client is not None and request.client.host in ("127.0.0.1", "::1")


def require_session(request: Request):
    service = request.app.state.live
    token = request.cookies.get(COOKIE)
    if request.url.scheme != "https" and not local_client(request):
        raise HTTPException(403, detail="Trusted HTTPS is required for live sessions")
    if not service.available:
        raise HTTPException(503, detail="Live streaming is not configured on this server")
    if not service.authenticated(token):
        raise HTTPException(401, detail="Pair this device first")
    return service, token


async def protect_archive(request: Request, call_next):
    path = request.url.path
    service = getattr(request.app.state, "live", None)
    if service and service.code and not local_client(request) and (
        path == "/frames" or path.startswith("/frames/") or path == "/analyze-frame"
    ):
        try:
            require_session(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
    return await call_next(request)


def _page(name):
    return FileResponse(
        WEB / name,
        headers={
            "Cache-Control": "no-cache",
            "Permissions-Policy": "camera=(self), microphone=(self), on-device-speech-recognition=(self)",
            "Content-Security-Policy": (
                "default-src 'self'; connect-src 'self'; media-src 'self' blob:; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            ),
        },
    )


@router.get("/", include_in_schema=False)
async def dashboard():
    return _page("dashboard.html")


@router.get("/phone", include_in_schema=False)
async def phone():
    return _page("phone.html")


@router.get("/sw.js", include_in_schema=False)
async def service_worker():
    return FileResponse(WEB / "sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@router.post("/live/session")
async def login(request: Request, response: Response, login: Login):
    if request.url.scheme != "https" and not local_client(request):
        raise HTTPException(403, detail="Trusted HTTPS is required for live sessions")
    if request.headers.get("origin", str(request.base_url).rstrip("/")) != str(request.base_url).rstrip("/"):
        raise HTTPException(403, detail="Cross-origin pairing is not allowed")
    owner = request.app.state.live.login(login.code, request.cookies.get(COOKIE))
    response.headers["Cache-Control"] = "no-store"
    response.set_cookie(
        COOKIE, owner, max_age=SESSION_SECONDS, httponly=True,
        secure=request.url.scheme == "https", samesite="strict",
    )
    return {"paired": True}


@router.get("/live/status")
async def status(request: Request, response: Response):
    service, _ = require_session(request)
    response.headers["Cache-Control"] = "no-store"
    return service.status()


@router.get("/live/changes")
async def changes(request: Request, response: Response):
    service, owner = require_session(request)
    response.headers["Cache-Control"] = "no-store"
    camera = service.camera
    if camera is None or camera.owner != owner or not camera.changes or service.latest_capture() is None:
        return {"change": None}
    event = camera.changes.latest(monotonic())
    return {"change": {**event, "id": f'{camera.id}:{event["sequence"]}'} if event else None}


@router.post("/live/offer")
async def offer(request: Request, offer: Offer):
    service, owner = require_session(request)
    return await service.create_peer(owner, offer)


@router.delete("/live/peers/{peer_id}", status_code=204)
async def close_peer(request: Request, peer_id: str):
    service, owner = require_session(request)
    peer = service.peers.get(peer_id)
    if peer is not None and peer.owner != owner:
        raise HTTPException(404, detail="Peer not found")
    await service.close_peer(peer_id)


@router.get("/live/archive/{frame_id}/image")
async def archive_image(request: Request, frame_id: int):
    require_session(request)
    if not 0 < frame_id < 2**63:
        raise HTTPException(404, detail="Frame not found")
    store = request.app.state.frame_store
    frame = await run_in_threadpool(store.get, frame_id)
    if frame is None:
        raise HTTPException(404, detail="Frame not found")
    image = store.directory / frame["image_path"]
    if not await run_in_threadpool(image.is_file):
        raise HTTPException(404, detail="Image not found")
    return FileResponse(image, headers={"Cache-Control": "no-store"})
