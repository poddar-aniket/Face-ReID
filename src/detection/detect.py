"""Face detection module (M1).

Uses InsightFace's buffalo_l pack (RetinaFace detector + ArcFace recognizer).
The model is loaded once and reused. GPU (CUDA) is used when onnxruntime
exposes it, otherwise it falls back to CPU automatically.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
MIN_DET_SCORE = 0.5
PAD_RATIO = 0.5  # border added on each side when a tight crop yields no detection

_app = None


def get_face_app() -> FaceAnalysis:
    """Load the InsightFace model once and return the shared instance."""
    global _app
    if _app is None:
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        _app = FaceAnalysis(name=MODEL_NAME, providers=providers)
        _app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
        print(f"[detect] model={MODEL_NAME} providers={providers} ctx_id={ctx_id}")
    return _app


def load_image(image_path: str) -> np.ndarray:
    """Read an image as a BGR array. Safe for Windows paths with non-ASCII chars."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return img


def detect_faces(image_path: str, min_det_score: float = MIN_DET_SCORE) -> list[dict]:
    """Detect all faces in an image.

    Returns a list (left to right by bbox x1) of dicts:
        bbox: [x1, y1, x2, y2] ints, clipped to image bounds
        det_score: float detection confidence
        crop: np.ndarray BGR crop of the face region
        face: the raw InsightFace Face object (used by embed.py)
    Returns an empty list if no face passes min_det_score.
    """
    img = load_image(image_path)
    height, width = img.shape[:2]
    app = get_face_app()
    faces = app.get(img)

    # Tight crops (face fills the whole frame) are often missed by RetinaFace.
    # If nothing was found, retry once on a padded copy and map coordinates back.
    offset = 0
    if not faces:
        offset = int(PAD_RATIO * max(height, width))
        padded = cv2.copyMakeBorder(
            img, offset, offset, offset, offset,
            cv2.BORDER_CONSTANT, value=(127, 127, 127),
        )
        faces = app.get(padded)
        for face in faces:
            face.bbox = face.bbox - offset
            face.kps = face.kps - offset

    results = []
    for face in faces:
        det_score = float(face.det_score)
        if det_score < min_det_score:
            continue

        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "det_score": det_score,
                "crop": img[y1:y2, x1:x2].copy(),
                "face": face,
            }
        )

    results.sort(key=lambda r: r["bbox"][0])
    return results


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_demo_images/test.jpg"
    found = detect_faces(path)
    print(f"{len(found)} face(s) found in {path}")
    for i, r in enumerate(found, start=1):
        print(i, r["bbox"], round(r["det_score"], 3), r["crop"].shape)