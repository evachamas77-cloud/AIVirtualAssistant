from datetime import UTC, date, datetime, time
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, StrictBool, StringConstraints, model_validator
from starlette.concurrency import run_in_threadpool

from .frame_store import utc_timestamp
from .live import require_session
from .semantic import SemanticBusy, SemanticUnavailable

router = APIRouter(prefix="/ai")
NO_CACHE = {"Cache-Control": "no-store"}


def date_window(start, end):
    def parse(value, for_end):
        if value is None:
            return None
        try:
            if len(value) == 10:
                day = date.fromisoformat(value)
                stamp = datetime.combine(day, time.max if for_end else time.min, tzinfo=UTC)
            else:
                stamp = datetime.fromisoformat(value)
                if stamp.utcoffset() is None:
                    raise ValueError
            return utc_timestamp(stamp)
        except (ValueError, OverflowError):
            raise HTTPException(422, detail="Use YYYY-MM-DD or an ISO timestamp with a time zone.") from None

    first, last = parse(start, False), parse(end, True)
    if first and last and first > last:
        raise HTTPException(422, detail="Start must not be after end.")
    return first, last


class Question(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)]
    language: Literal["en", "fr"] = "en"
    scope: Literal["auto", "current", "history", "frame"] = "auto"
    frame_id: int | None = Field(default=None, ge=1, le=2**63 - 1)
    start: str | None = Field(default=None, max_length=40)
    end: str | None = Field(default=None, max_length=40)
    consent: StrictBool = False
    live_only: StrictBool = False

    @model_validator(mode="after")
    def valid_source(self):
        if (self.scope == "frame") != (self.frame_id is not None):
            raise ValueError("frame_id is required only for scope 'frame'")
        if self.scope in ("current", "frame") and (self.start or self.end):
            raise ValueError("Date filters require history or auto scope")
        if self.live_only and self.scope != "current":
            raise ValueError("live_only requires an explicit current scope")
        return self


def authorize(request, response=None, *, post=False):
    _, owner = require_session(request)
    if post and request.headers.get("origin", str(request.base_url).rstrip("/")) != str(request.base_url).rstrip("/"):
        raise HTTPException(403, detail="Cross-origin requests are not allowed.")
    if response is not None:
        response.headers.update(NO_CACHE)
    return owner


def search_error(exc):
    return HTTPException(409 if isinstance(exc, SemanticBusy) else 503, detail=str(exc))


@router.get("/index")
async def index_status(request: Request, response: Response):
    authorize(request, response)
    return await run_in_threadpool(request.app.state.semantic.progress)


@router.post("/index/backfill", status_code=202)
async def backfill(request: Request, response: Response):
    authorize(request, response, post=True)
    index = request.app.state.semantic
    if not index.available():
        raise HTTPException(503, detail="Local semantic search is not set up. See README.")
    await run_in_threadpool(index.retry_unreadable)
    index.wake.set()
    return await run_in_threadpool(index.progress)


@router.get("/semantic-search")
async def semantic_search(
    request: Request, response: Response,
    q: Annotated[str, Query(min_length=1, max_length=800)],
    start: Annotated[str | None, Query(max_length=40)] = None,
    end: Annotated[str | None, Query(max_length=40)] = None,
    limit: Annotated[int, Query(ge=1, le=24)] = 12,
):
    authorize(request, response)
    if not q.strip():
        raise HTTPException(422, detail="Enter a search phrase.")
    first, last = date_window(start, end)
    index = request.app.state.semantic
    try:
        items = await index.search(q.strip(), first, last, limit)
    except (SemanticUnavailable, SemanticBusy) as exc:
        raise search_error(exc) from None
    return {"items": items, "index": await run_in_threadpool(index.progress)}


@router.post("/ask")
async def ask(request: Request, response: Response, question: Question):
    owner = authorize(request, response, post=True)
    start, end = date_window(question.start, question.end)
    try:
        return await request.app.state.vision.ask(question, owner, start, end)
    except (SemanticUnavailable, SemanticBusy) as exc:
        raise search_error(exc) from None


@router.get("/evidence/{frame_id}")
async def evidence(request: Request, frame_id: str):
    owner = authorize(request)
    image = request.app.state.vision.snapshot(owner, frame_id)
    if image is None:
        raise HTTPException(404, detail="Live evidence has expired or is unavailable.")
    return Response(content=image, media_type="image/jpeg", headers=NO_CACHE)
