"""Similarity search + the query() function the UI calls."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ROOT, get_db_path, load_config
from src.storage.storage import FaceStore


def _resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def search_embedding(embedding, threshold=None, top_k=20, db_path=None) -> list[dict]:
    """Cosine search of one embedding against the index.

    Returns one entry per image (best-matching face), sorted by score desc.
    """
    if threshold is None:
        threshold = load_config()["similarity_threshold"]
    db_path = Path(db_path) if db_path else get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"No index at {db_path}. Run: python -m src.search.build_index")

    q = np.asarray(embedding, dtype=np.float32)
    q = q / np.linalg.norm(q)

    with FaceStore(db_path) as store:
        meta, matrix = store.load_all()
    if not meta:
        raise RuntimeError("Index is empty. Run: python -m src.search.build_index")

    scores = matrix @ q  # unit vectors -> dot product == cosine similarity

    best: dict[str, dict] = {}
    for i in np.flatnonzero(scores >= threshold):
        m, s = meta[i], float(scores[i])
        cur = best.get(m["image_id"])
        if cur is None or s > cur["score"]:
            best[m["image_id"]] = {
                "image_id": m["image_id"],
                "image_path": _resolve(m["image_path"]),
                "score": s,
                "bbox": m["bbox"],
            }
    results = sorted(best.values(), key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def query(image_path, face_index=0, threshold=None, top_k=20, db_path=None) -> list[dict]:
    """Upload photo -> detect + embed -> search.

    face_index follows the left-to-right order of detect_faces.
    Returns [] if the query image has no face.
    """
    from src.embedding.embed import embed_faces  # lazy: keeps tests free of insightface

    faces = embed_faces(str(image_path))
    if not faces:
        return []
    if not 0 <= face_index < len(faces):
        raise IndexError(f"face_index {face_index} out of range (found {len(faces)} faces)")
    return search_embedding(faces[face_index]["embedding"], threshold, top_k, db_path)