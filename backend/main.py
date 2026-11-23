
import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
import logging
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Path, Query, Request, Response, UploadFile
from pydantic import AwareDatetime
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles
import cv2
import numpy as np

from .ai_routes import router as ai_router
from .frame_store import FrameSettings, FrameStore, utc_timestamp
from .live import LiveService, WEB, protect_archive, router as live_router
from .object_detection import detect_objects
from .ocr import recognize_text
from .semantic import SemanticIndex
from .vision import VisionService

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg": (b"\xff\xd8\xff", ".jpg"), "image/png": (b"\x89PNG\r\n\x1a\n", ".png")}
FrameId = Annotated[int, Path(ge=1, le=2**63 - 1)]


def _cleanup_frames(store: FrameStore):
    try:
        store.cleanup()
    except Exception:
        logger.exception("Frame retention cleanup failed")


async def _schedule_cleanup(store: FrameStore, stop: asyncio.Event):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=store.settings.cleanup_interval_seconds)
        except TimeoutError:
            await run_in_threadpool(_cleanup_frames, store)


@asynccontextmanager
async def lifespan(application: FastAPI):
    store = FrameStore(FrameSettings.from_env())
    await run_in_threadpool(store.initialize)
    application.state.frame_store = store
    live = application.state.live = LiveService(lambda frame: _archive_video_frame(store, frame))
    semantic = application.state.semantic = SemanticIndex(store)
    vision = application.state.vision = VisionService(store, semantic, live)
    await run_in_threadpool(semantic.initialize)
    await run_in_threadpool(vision.initialize)
    stop = asyncio.Event()
    cleanup = None
    if store.settings.retention_days:
        await run_in_threadpool(_cleanup_frames, store)
        cleanup = asyncio.create_task(_schedule_cleanup(store, stop))
    semantic.start()
    try:
        yield
    finally:
        stop.set()
        await live.shutdown()
        await semantic.stop()
        if cleanup is not None:
            await cleanup


app = FastAPI(title="AI Visual Assistant", lifespan=lifespan)
app.middleware("http")(protect_archive)
app.include_router(live_router)
app.include_router(ai_router)
app.mount("/ui", StaticFiles(directory=WEB, check_dir=False), name="ui")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    return await run_in_threadpool(_analyze_image, await _read_upload(file))


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported",
        )

    contents = await file.read(MAX_IMAGE_SIZE + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image")

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large")

    return contents


def _analyze_image(contents: bytes):
    started = perf_counter()
    return _analyze_pixels(_decode_image(contents), started)


def _decode_image(contents: bytes) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image


def _analyze_pixels(image: np.ndarray, started: float):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    errors = {}
    try:
        objects = detect_objects(image)
    except Exception:
        logger.exception("Object detection failed")
        objects = []
        errors["object_detection"] = "Object detection unavailable; check server logs."

    try:
        ocr_text, ocr_results = recognize_text(image)
    except Exception:
        logger.exception("OCR failed")
        ocr_text, ocr_results = "", []
        errors["ocr"] = "OCR unavailable; check server logs."

    return {
        "width": width,
        "height": height,
        "sharpness_score": round(sharpness, 2),
        "quality": "blurry" if sharpness < 100 else "sharp",
        "objects": objects,
        "ocr_text": ocr_text,
        "ocr_results": ocr_results,
        "processing_time_ms": round((perf_counter() - started) * 1000, 2),
        "analysis_errors": errors,
    }


def _analyze_and_save(store: FrameStore, contents: bytes, content_type: str):
    signature, extension = IMAGE_TYPES[content_type]
    if not contents.startswith(signature):
        raise HTTPException(status_code=400, detail="Image data does not match the upload type")

    started = perf_counter()
    image = _decode_image(contents)
    analysis = _analyze_pixels(image, started)
    frame, reason = store.save(contents, extension, _pixel_hash(image), analysis)
    return {"saved": frame is not None, "reason": reason, "frame": frame, "analysis": analysis}


def _pixel_hash(image: np.ndarray) -> str:
    digest = sha256(f"{image.shape}:".encode("ascii"))
    digest.update(memoryview(image))
    return digest.hexdigest()


def _archive_video_frame(store: FrameStore, frame):
    started = perf_counter()
    width, height = frame.width, frame.height
    if max(width, height) > 1280:
        scale = 1280 / max(width, height)
        frame = frame.reformat(width=max(1, round(width * scale)), height=max(1, round(height * scale)))
    image = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
    # Analyze the JPEG that will be stored so box coordinates and exact-pixel dedup match uploads.
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise RuntimeError("Could not encode live frame")
    contents = encoded.tobytes()
    image = _decode_image(contents)
    analysis = _analyze_pixels(image, started)
    return store.save(contents, ".jpg", _pixel_hash(image), analysis)


@app.post("/frames", status_code=201, responses={200: {"description": "Analyzed but not saved"}})
async def create_frame(request: Request, response: Response, file: UploadFile = File(...)):
    contents = await _read_upload(file)
    result = await run_in_threadpool(
        _analyze_and_save, request.app.state.frame_store, contents, file.content_type
    )
    if not result["saved"]:
        response.status_code = 200
    return result


def _frame_filters(
    start: AwareDatetime | None = None,
    end: AwareDatetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
):
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=422, detail="start must be at or before end")
    return {
        "start": utc_timestamp(start) if start is not None else None,
        "end": utc_timestamp(end) if end is not None else None,
        "limit": limit,
        "offset": offset,
    }


@app.get("/frames")
def list_frames(request: Request, filters: Annotated[dict, Depends(_frame_filters)]):
    return request.app.state.frame_store.list_frames(**filters)


@app.get("/frames/search")
def search_frames(
    request: Request,
    filters: Annotated[dict, Depends(_frame_filters)],
    q: Annotated[str, Query(min_length=1, max_length=200)],
):
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be blank")
    return request.app.state.frame_store.list_frames(**filters, keyword=q.strip())


@app.get("/frames/{frame_id}")
def get_frame(request: Request, frame_id: FrameId):
    frame = request.app.state.frame_store.get(frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    return frame


@app.delete("/frames/{frame_id}", status_code=204)
def delete_frame(request: Request, frame_id: FrameId):
    if not request.app.state.frame_store.delete(frame_id):
        raise HTTPException(status_code=404, detail="Frame not found")
