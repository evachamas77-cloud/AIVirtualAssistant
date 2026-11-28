from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

logger = logging.getLogger(__name__)
DEFAULT_STORAGE_DIR = Path(__file__).resolve().parents[1] / "data" / "frames"


def utc_timestamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("Frame timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class FrameSettings:
    directory: Path
    min_sharpness: float
    min_interval_seconds: float
    retention_days: int
    cleanup_interval_seconds: int

    @classmethod
    def from_env(cls):
        def number(name, default, convert, minimum=0):
            value = convert(os.getenv(name, str(default)))
            if not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be finite and at least {minimum}")
            return value

        return cls(
            directory=Path(os.getenv("FRAME_STORAGE_DIR", str(DEFAULT_STORAGE_DIR))).resolve(),
            min_sharpness=number("FRAME_MIN_SHARPNESS", 100, float),
            min_interval_seconds=number("FRAME_MIN_INTERVAL_SECONDS", 5, float),
            retention_days=number("FRAME_RETENTION_DAYS", 7, int),
            cleanup_interval_seconds=number("FRAME_CLEANUP_INTERVAL_SECONDS", 3600, int, 1),
        )


class FrameStore:
    def __init__(self, settings: FrameSettings):
        self.settings = settings
        self.directory = settings.directory
        self.db_path = self.directory / "frames.sqlite3"

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        connection.create_function("casefold", 1, str.casefold, deterministic=True)
        return closing(connection)

    def initialize(self):
        (self.directory / "images").mkdir(parents=True, exist_ok=True)
        with self._connect() as db, db:
            db.execute("PRAGMA journal_mode = WAL").fetchone()
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL UNIQUE,
                    pixel_hash TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sharpness_score REAL NOT NULL,
                    quality TEXT NOT NULL,
                    ocr_text TEXT NOT NULL,
                    objects_json TEXT NOT NULL,
                    ocr_results_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS frames_timestamp ON frames (timestamp, id);
                """
            )
            db.execute("BEGIN IMMEDIATE")
            if db.execute("PRAGMA user_version").fetchone()[0] < 1:
                db.execute("""CREATE TABLE IF NOT EXISTS frame_objects (
                    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
                    label TEXT NOT NULL, confidence REAL NOT NULL, PRIMARY KEY(frame_id, label))""")
                db.execute("CREATE INDEX IF NOT EXISTS frame_objects_label ON frame_objects(label, frame_id)")
                db.execute("""INSERT OR IGNORE INTO frame_objects SELECT f.id, lower(trim(o.value ->> 'name')),
                    max(CAST(o.value ->> 'confidence' AS REAL)) FROM frames f,
                    json_each(CASE WHEN json_valid(f.objects_json) THEN f.objects_json ELSE '[]' END) o
                    WHERE o.type = 'object' AND length(trim(o.value ->> 'name')) > 0
                      AND json_type(o.value, '$.confidence') IN ('real', 'integer')
                    GROUP BY f.id, lower(trim(o.value ->> 'name'))""")
                db.execute("PRAGMA user_version = 1")

    @staticmethod
    def _metadata(row: sqlite3.Row) -> dict:
        data = dict(row)
        data.pop("pixel_hash")
        data["objects"] = json.loads(data.pop("objects_json"))
        data["ocr_results"] = json.loads(data.pop("ocr_results_json"))
        return data

    def save(
        self, contents: bytes, extension: str, pixel_hash: str, analysis: dict, *, captured_at: datetime | None = None
    ) -> tuple[dict | None, str | None]:
        if analysis["analysis_errors"]:
            return None, "analysis_failed"
        if analysis["sharpness_score"] < self.settings.min_sharpness:
            return None, "too_blurry"

        image = None
        try:
            with self._connect() as db, db:
                # Serialize selection with the insert so concurrent uploads obey both rules.
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT 1 FROM frames WHERE pixel_hash = ?", (pixel_hash,)).fetchone():
                    return None, "duplicate"

                now = captured_at if captured_at is not None else datetime.now(UTC)
                last = db.execute(
                    "SELECT timestamp FROM frames ORDER BY timestamp DESC, id DESC LIMIT 1"
                ).fetchone()
                if last and (
                    now - datetime.fromisoformat(last["timestamp"])
                ).total_seconds() < self.settings.min_interval_seconds:
                    return None, "min_interval"

                image_path = f"images/{uuid4().hex}{extension}"
                image = self.directory / image_path
                image.write_bytes(contents)
                cursor = db.execute(
                    """
                    INSERT INTO frames (
                        timestamp, image_path, pixel_hash, width, height,
                        sharpness_score, quality, ocr_text, objects_json, ocr_results_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        utc_timestamp(now), image_path, pixel_hash,
                        analysis["width"], analysis["height"], analysis["sharpness_score"],
                        analysis["quality"], analysis["ocr_text"],
                        json.dumps(analysis["objects"]), json.dumps(analysis["ocr_results"]),
                    ),
                )
                objects = {}
                for obj in analysis["objects"]:
                    label = obj["name"].strip().lower()
                    if label:
                        objects[label] = max(objects.get(label, 0), obj["confidence"])
                db.executemany("INSERT INTO frame_objects(frame_id, label, confidence) VALUES (?, ?, ?)",
                               [(cursor.lastrowid, label, score) for label, score in objects.items()])
                row = db.execute("SELECT * FROM frames WHERE id = ?", (cursor.lastrowid,)).fetchone()
                frame = self._metadata(row)
        except Exception:
            if image is not None:
                try:
                    image.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove image after a failed frame save: %s", image)
            raise
        return frame, None

    def observations(self, *, start=None, end=None, label=None, min_confidence=0.45,
                     has_text=False, text=None, earliest=False, limit=12):
        filters, args = [], []
        join = ""
        if label is not None:
            join = "JOIN frame_objects o ON o.frame_id = f.id"
            filters.append("o.label = ? AND o.confidence >= ?")
            args.extend((label, min_confidence))
        if start is not None:
            filters.append("f.timestamp >= ?")
            args.append(start)
        if end is not None:
            filters.append("f.timestamp <= ?")
            args.append(end)
        if has_text:
            filters.append("length(trim(f.ocr_text)) > 0")
        if text:
            filters.append("instr(casefold(f.ocr_text), ?) > 0")
            args.append(text.casefold())
        where = " WHERE " + " AND ".join(filters) if filters else ""
        direction = "ASC" if earliest else "DESC"
        with self._connect() as db:
            rows = db.execute(f"SELECT f.* FROM frames f {join}{where} ORDER BY f.timestamp {direction}, "
                              f"f.id {direction} LIMIT ?", (*args, limit)).fetchall()
            return [self._metadata(row) for row in rows]

    def list_frames(
        self, *, start: str | None, end: str | None,
        limit: int, offset: int, keyword: str | None = None,
    ) -> dict:
        conditions, values = [], []
        if start is not None:
            conditions.append("timestamp >= ?")
            values.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            values.append(end)
        if keyword is not None:
            conditions.append(
                """
                (instr(casefold(ocr_text), ?) > 0 OR EXISTS (
                    SELECT 1 FROM json_each(frames.objects_json) AS object
                    WHERE instr(casefold(json_extract(object.value, '$.name')), ?) > 0
                ))
                """
            )
            values.extend([keyword.casefold()] * 2)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM frames" + where + " ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                (*values, limit + 1, offset),
            ).fetchall()
        return {
            "items": [self._metadata(row) for row in rows[:limit]],
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    def get(self, frame_id: int) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
        return self._metadata(row) if row is not None else None

    def delete(self, frame_id: int) -> bool:
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT image_path FROM frames WHERE id = ?", (frame_id,)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM frames WHERE id = ?", (frame_id,))
            # A filesystem error leaves metadata intact so deletion can be retried.
            (self.directory / row["image_path"]).unlink(missing_ok=True)
        return True

    def cleanup(self):
        if not self.settings.retention_days:
            return
        cutoff = utc_timestamp(datetime.now(UTC) - timedelta(days=self.settings.retention_days))
        with self._connect() as db:
            expired = db.execute("SELECT id FROM frames WHERE timestamp < ?", (cutoff,)).fetchall()
        for row in expired:
            try:
                self.delete(row["id"])
            except OSError:
                logger.exception("Could not remove expired frame %s", row["id"])
