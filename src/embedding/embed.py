"""Face embedding module (M1).

Current state: STUB. Returns the agreed record schema with deterministic
dummy data so M2 (storage/search) and M4 (calibration) can build against it.
The real InsightFace (RetinaFace + ArcFace) implementation will keep the
same function signature and record shape.
"""

import hashlib
import os

import numpy as np

EMBEDDING_DIM = 512
EMBEDDING_DTYPE = np.float32


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return vec scaled to unit L2 norm, as float32."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(EMBEDDING_DTYPE)
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
    # TODO: STUB: replace with InsightFace detection + ArcFace embedding
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    records = []

    for i in range(1, 3):  # stub: always 2 fake faces
        digest = hashlib.md5(f"{image_path}_{i}".encode()).hexdigest()
        seed = int(digest, 16) % (2**32)
        raw = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM)
        embedding = _l2_normalize(raw.astype(EMBEDDING_DTYPE))

        records.append(
            {
                "face_id": f"{image_id}_face{i}",
                "image_id": image_id,
                "embedding": embedding,
                "bbox": [10 * i, 10 * i, 110 * i, 130 * i],
                "detection_confidence": 0.95,
            }
        )

    return records


if __name__ == "__main__":
    for rec in embed_faces("data/sample_demo_images/test.jpg"):
        print(
            rec["face_id"],
            rec["image_id"],
            rec["embedding"].shape,
            rec["embedding"].dtype,
            float(np.linalg.norm(rec["embedding"])),
            rec["bbox"],
        )