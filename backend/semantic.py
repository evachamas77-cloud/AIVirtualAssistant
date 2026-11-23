import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from functools import partial
import heapq
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
MODEL_REPO = "onnx-community/siglip2-base-patch16-224-ONNX"
MODEL_REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
MODEL_ID = f"siglip2-base-224-int8:{MODEL_REVISION}"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "siglip2"
FILES = {
    "onnx/vision_model_quantized.onnx": "5f2b401c1a4fc095702a5d45348e17ad46c4f87064085365b43c6e8eaa5c0070",
    "onnx/text_model_quantized.onnx": "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400",
    "tokenizer.json": None,
}


def model_directory():
    return Path(os.getenv("SEMANTIC_MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()


class SemanticUnavailable(Exception):
    pass


class SemanticBusy(Exception):
    pass


def normalized(embedding):
    vector = np.asarray(embedding, dtype="<f4").reshape(-1)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Encoder returned an invalid embedding")
    return vector / magnitude


class SiglipEncoder:
    model_id = MODEL_ID

    def __init__(self, directory=None):
        self.directory = directory or model_directory()
        self.vision = self.text = self.tokenizer = None

    def available(self):
        return all((self.directory / name).is_file() for name in FILES) and all(
            importlib.util.find_spec(name) is not None for name in ("onnxruntime", "tokenizers", "PIL")
        )

    def _session(self, filename):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return ort.InferenceSession(
            str(self.directory / filename), sess_options=options, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def _embedding(session, kind, feed):
        outputs = {item.name for item in session.get_outputs()}
        name = next((name for name in (f"{kind}_embeds", "pooler_output") if name in outputs), None)
        if name is None:
            raise RuntimeError(f"Unexpected SigLIP 2 {kind} outputs: {outputs}")
        return session.run([name], feed)[0][0]

    def encode_image(self, image_path):
        from PIL import Image, ImageOps

        with Image.open(image_path) as image:
            resized = ImageOps.exif_transpose(image).convert("RGB").resize(
                (224, 224), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
        if self.vision is None:
            self.vision = self._session("onnx/vision_model_quantized.onnx")
        feed = {"pixel_values": np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])}
        return self._embedding(self.vision, "image", feed)

    def encode_text(self, query):
        if self.text is None:
            self.text = self._session("onnx/text_model_quantized.onnx")
        if self.tokenizer is None:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
        # This export uses the lowercased Gemma tokenizer, EOS, then right-padding to 64.
        tokens = self.tokenizer.encode(query.lower(), add_special_tokens=False).ids[:63] + [1]
        padding = [0] * (64 - len(tokens))
        feed = {"input_ids": np.array([tokens + padding], dtype=np.int64)}
        if any(value.name == "attention_mask" for value in self.text.get_inputs()):
            feed["attention_mask"] = np.array([[1] * len(tokens) + padding], dtype=np.int64)
        return self._embedding(self.text, "text", feed)


def archive_reference(frame_id, timestamp):
    return {
        "frame_id": frame_id, "timestamp": timestamp, "source": "archive",
        "image_url": f"/live/archive/{frame_id}/image",
    }


class SemanticIndex:
    def __init__(self, store, encoder=None):
        self.store = store
        self.encoder = encoder or SiglipEncoder()
        self.enabled = os.getenv("SEMANTIC_ENABLED", "1") != "0"
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic")
        self.query_lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.stopping = False
        self.worker = None
        self.error = None

    def initialize(self):
        with self.store._connect() as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS frame_embeddings (
                    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    embedding BLOB,
                    PRIMARY KEY (frame_id, model)
                )
                """
            )

    def available(self):
        return self.enabled and self.encoder.available()

    async def _run(self, fn, *args, **kwargs):
        future = asyncio.get_running_loop().run_in_executor(self.pool, partial(fn, *args, **kwargs))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await asyncio.gather(future, return_exceptions=True)
            raise

    def start(self):
        if self.enabled:
            self.worker = asyncio.create_task(self._backfill())

    async def stop(self):
        self.stopping = True
        self.wake.set()
        if self.worker:
            await self.worker
        self.pool.shutdown(wait=False, cancel_futures=True)

    async def _backfill(self):
        while not self.stopping:
            delay = 3
            try:
                if self.available():
                    indexed = await self._run(self.index_next)
                    delay = 0.2 if indexed else 2
                    self.error = None
            except Exception:
                if self.error is None:
                    logger.exception("Local image indexing failed")
                self.error = "Embedding model failed to load or run; check the server log."
                delay = 30
            if self.stopping:
                break
            try:
                await asyncio.wait_for(self.wake.wait(), delay)
            except TimeoutError:
                pass
            self.wake.clear()

    def index_next(self):
        with self.store._connect() as db:
            row = db.execute(
                """
                SELECT f.id, f.image_path FROM frames f
                LEFT JOIN frame_embeddings e ON e.frame_id = f.id AND e.model = ?
                WHERE e.frame_id IS NULL ORDER BY f.id DESC LIMIT 1
                """,
                (self.encoder.model_id,),
            ).fetchone()
        if row is None:
            return False
        try:
            vector = normalized(self.encoder.encode_image(self.store.directory / row["image_path"]))
            blob = vector.tobytes()
        except OSError:
            # Record unreadable files so one bad image cannot starve the backfill.
            blob = None
        with self.store._connect() as db, db:
            db.execute(
                """INSERT OR IGNORE INTO frame_embeddings (frame_id, model, embedding)
                   SELECT id, ?, ? FROM frames WHERE id = ?""",
                (self.encoder.model_id, blob, row["id"]),
            )
        return True

    def retry_unreadable(self):
        with self.store._connect() as db, db:
            db.execute(
                "DELETE FROM frame_embeddings WHERE model = ? AND embedding IS NULL",
                (self.encoder.model_id,),
            )

    def progress(self):
        with self.store._connect() as db:
            counts = db.execute(
                """
                SELECT count(e.embedding) AS indexed, count(f.id) - count(e.frame_id) AS pending,
                       count(e.frame_id) - count(e.embedding) AS unreadable
                FROM frames f
                LEFT JOIN frame_embeddings e ON e.frame_id = f.id AND e.model = ?
                """,
                (self.encoder.model_id,),
            ).fetchone()
        return {"available": self.available(), "error": self.error, **dict(counts)}

    async def search(self, query, start=None, end=None, limit=12):
        if not self.available():
            raise SemanticUnavailable("Local semantic search is not set up. See the model setup in README.")
        if self.query_lock.locked():
            raise SemanticBusy("A local search is already running. Try again shortly.")
        async with self.query_lock:
            try:
                return await self._run(self.rank, query, start, end, limit)
            except Exception:
                logger.exception("Local semantic search failed")
                raise SemanticUnavailable("Local embedding model failed; check the server log.") from None

    def rank(self, query, start, end, limit):
        conditions = ["e.model = ?", "e.embedding IS NOT NULL"]
        params = [self.encoder.model_id]
        for timestamp, condition in ((start, "f.timestamp >= ?"), (end, "f.timestamp <= ?")):
            if timestamp is not None:
                conditions.append(condition)
                params.append(timestamp)
        with self.store._connect() as db:
            sql = (
                "SELECT f.id, f.timestamp, f.image_path, e.embedding FROM frame_embeddings e "
                "JOIN frames f ON f.id = e.frame_id WHERE " + " AND ".join(conditions)
            )
            rows = db.execute(sql, params)
            batch = rows.fetchmany(128)
            if not batch:
                return []
            vector = normalized(self.encoder.encode_text(query))

            def scored():
                nonlocal batch
                while batch:
                    for row in batch:
                        image_vector = np.frombuffer(row["embedding"], dtype="<f4")
                        if row["embedding"] and image_vector.shape == vector.shape and (
                            self.store.directory / row["image_path"]
                        ).is_file():
                            score = float(np.clip(image_vector @ vector, -1, 1))
                            if np.isfinite(score):
                                yield score, row["timestamp"], row["id"]
                    batch = rows.fetchmany(128)

            ranked = heapq.nlargest(limit, scored())
        return [
            {**archive_reference(frame_id, timestamp), "similarity": round(score, 6)}
            for score, timestamp, frame_id in ranked
        ]
