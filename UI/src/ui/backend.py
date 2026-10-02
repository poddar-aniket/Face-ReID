"""Adapter between the UI and M1/M2's code.

The UI only talks to the three functions below. If the real modules are
importable they are used; otherwise the mock is used so the UI always runs.

Agreed contract (confirm with M2):
    detect_faces(image_path) -> list[dict(bbox=(x1, y1, x2, y2))]
    query(image_path, face_index=0, threshold=0.4, top_k=20) -> list[dict]
        each result: {"image_id": str, "score": float,
                      "bbox": (x1, y1, x2, y2) | None}
    load_image(image_id) -> PIL.Image
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEMO_DIR = ROOT / "data" / "sample_demo_images"

USING_MOCK = True
try:
    # Replace these imports with the real module paths once M1/M2 confirm them.
    from src.search.search import query as _query          # M2
    from src.detection.detect import detect_faces as _detect  # M1
    USING_MOCK = False
except Exception:
    from src.ui import mock_backend as _mock
    _query, _detect = _mock.query, _mock.detect_faces


def detect_faces(image_path: str) -> list[dict]:
    return _detect(image_path)


def query(image_path: str, face_index: int = 0,
          threshold: float = 0.4, top_k: int = 20) -> list[dict]:
    raw = _query(image_path, face_index=face_index,
                 threshold=threshold, top_k=top_k)
    # Tolerate the plan's original (image_id, score) tuples.
    out = []
    for r in raw:
        if isinstance(r, dict):
            out.append({"bbox": None, **r})
        else:
            out.append({"image_id": r[0], "score": float(r[1]), "bbox": None})
    return out


def load_image(image_id: str):
    from PIL import Image
    if USING_MOCK:
        return _mock.load_image(image_id)
    p = Path(image_id)
    return Image.open(p if p.is_absolute() else DEMO_DIR / image_id).convert("RGB")
