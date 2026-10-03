import numpy as np
import pytest

from src.search.search import search_embedding
from src.storage.storage import FaceStore


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _face(face_id, image_id, emb, bbox=(0, 0, 10, 10)):
    return {"face_id": face_id, "image_id": image_id, "embedding": emb,
            "bbox": list(bbox), "detection_confidence": 0.9}


@pytest.fixture
def index(tmp_path):
    rng = np.random.default_rng(0)
    a = _unit(rng.normal(size=512))
    b = _unit(rng.normal(size=512))
    near_a = _unit(a + 0.01 * rng.normal(size=512))  # cosine to a ~0.97
    db = tmp_path / "test.db"
    with FaceStore(db) as s:
        s.add_image("img1", "data/img1.jpg", [_face("img1_face0", "img1", a, (1, 2, 3, 4))])
        s.add_image("img2", "data/img2.jpg", [
            _face("img2_face0", "img2", b),
            _face("img2_face1", "img2", near_a, (5, 6, 7, 8)),
        ])
        s.add_image("img3", "data/img3.jpg", [])  # image with no faces
    return db, a, rng


def test_dedup_keeps_best_face_and_sorts(index):
    db, a, _ = index
    res = search_embedding(a, threshold=0.4, db_path=db)
    assert [r["image_id"] for r in res] == ["img1", "img2"]
    assert res[0]["score"] > res[1]["score"]
    assert res[1]["bbox"] == [5, 6, 7, 8]  # bbox of the matching face, not face 0


def test_threshold_filters(index):
    db, a, _ = index
    res = search_embedding(a, threshold=0.99, db_path=db)
    assert [r["image_id"] for r in res] == ["img1"]


def test_top_k(index):
    db, a, _ = index
    assert len(search_embedding(a, threshold=0.4, top_k=1, db_path=db)) == 1


def test_unrelated_query_returns_nothing(index):
    db, _, rng = index
    assert search_embedding(_unit(rng.normal(size=512)), threshold=0.4, db_path=db) == []


def test_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        search_embedding(np.ones(512, dtype=np.float32), db_path=tmp_path / "nope.db")