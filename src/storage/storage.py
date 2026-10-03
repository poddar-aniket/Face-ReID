"""SQLite storage for face embeddings (contract: Section 4.0 of the plan)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id   TEXT PRIMARY KEY,
    image_path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS faces (
    face_id    TEXT PRIMARY KEY,
    image_id   TEXT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,
    det_score  REAL
);
CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
"""


class FaceStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._conn.close()

    @staticmethod
    def _to_blob(embedding) -> bytes:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Expected shape ({EMBEDDING_DIM},), got {vec.shape}")
        if not np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3):
            raise ValueError("Embedding must be L2-normalized")
        return vec.tobytes()

    def add_image(self, image_id: str, image_path: str, faces: list[dict]) -> None:
        """Insert (or replace) one image and all its face records atomically."""
        rows = [
            (
                f["face_id"],
                image_id,
                self._to_blob(f["embedding"]),
                *(int(round(v)) for v in f["bbox"]),
                f.get("detection_confidence"),
            )
            for f in faces
        ]
        with self._conn:  # single transaction
            self._conn.execute(
                "DELETE FROM images WHERE image_id = ? OR image_path = ?",
                (image_id, image_path),
            )
            self._conn.execute(
                "INSERT INTO images (image_id, image_path) VALUES (?, ?)",
                (image_id, image_path),
            )
            self._conn.executemany(
                "INSERT INTO faces (face_id, image_id, embedding, x1, y1, x2, y2, det_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def has_image_path(self, image_path: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM images WHERE image_path = ?", (image_path,))
        return cur.fetchone() is not None

    def load_all(self) -> tuple[list[dict], np.ndarray]:
        """Return (metadata per face, matrix of shape (N, 512)) in matching order."""
        rows = self._conn.execute(
            "SELECT f.face_id, f.image_id, i.image_path, f.x1, f.y1, f.x2, f.y2, f.embedding "
            "FROM faces f JOIN images i ON i.image_id = f.image_id "
            "ORDER BY f.face_id"
        ).fetchall()
        if not rows:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        meta = [
            {"face_id": r[0], "image_id": r[1], "image_path": r[2], "bbox": [r[3], r[4], r[5], r[6]]}
            for r in rows
        ]
        matrix = np.stack([np.frombuffer(r[7], dtype=np.float32) for r in rows])
        return meta, matrix

    def count_images(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def count_faces(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM faces")
            self._conn.execute("DELETE FROM images")