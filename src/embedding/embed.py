"""Face embedding module (M1).

Real implementation: InsightFace ArcFace (buffalo_l), 512-d embeddings.
Signature and record schema are unchanged from the stub, so M2 and M4 code
keeps working.
"""

import os

import numpy as np

from src.detection.detect import detect_faces

EMBEDDING_DIM = 512
EMBEDDING_DTYPE = np.float32


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return vec scaled to unit L2 norm, as float32."""
    vec = np.asarray(vec, dtype=EMBEDDING_DTYPE)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return (vec / norm).astype(EMBEDDING_DTYPE)


def embed_faces(image_path: str) -> list[dict]:
    """Return one record per detected face in the image.

    Returns an empty list if no faces are detected.

    Each record:
        face_id: str                 unique per face, e.g. "img014_face2"
        image_id: str                parent image filename without extension
        embedding: np.ndarray        shape (512,), float32, L2-normalized
        bbox: list[int]              [x1, y1, x2, y2] in original image pixels
        detection_confidence: float  0 to 1
    """
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    records = []

    for i, det in enumerate(detect_faces(image_path), start=1):
        embedding = _l2_normalize(det["face"].embedding)
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Unexpected embedding shape {embedding.shape}")

        records.append(
            {
                "face_id": f"{image_id}_face{i}",
                "image_id": image_id,
                "embedding": embedding,
                "bbox": det["bbox"],
                "detection_confidence": det["det_score"],
            }
        )

    return records


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_demo_images/test.jpg"
    recs = embed_faces(path)
    print(f"{len(recs)} record(s) for {path}")
    for rec in recs:
        print(
            rec["face_id"],
            rec["image_id"],
            rec["embedding"].shape,
            rec["embedding"].dtype,
            round(float(np.linalg.norm(rec["embedding"])), 6),
            rec["bbox"],
            round(rec["detection_confidence"], 3),
        )