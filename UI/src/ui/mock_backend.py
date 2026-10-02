"""Stand-in for M1/M2 so the UI can be built and demoed before integration."""
import hashlib
import random
from PIL import Image, ImageDraw

_FACE_COUNT = 24


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16))


def detect_faces(image_path: str) -> list[dict]:
    img = Image.open(image_path)
    w, h = img.size
    n = _rng(image_path + str(w * h)).choice([1, 1, 2, 3])
    boxes = []
    for i in range(n):
        bw = w // (n + 2)
        x = int((i + 1) * w / (n + 1) - bw / 2)
        boxes.append({"bbox": (x, h // 4, x + bw, h // 4 + int(bw * 1.25))})
    return boxes


def query(image_path, face_index=0, threshold=0.4, top_k=20):
    r = _rng(image_path + str(face_index))
    results = []
    for i in range(_FACE_COUNT):
        score = round(r.uniform(0.05, 0.85), 3)
        if score >= threshold:
            results.append({"image_id": f"mock_{i:02d}.jpg", "score": score,
                            "bbox": (60, 50, 180, 200)})
    results.sort(key=lambda d: -d["score"])
    return results[:top_k]


def load_image(image_id: str) -> Image.Image:
    r = _rng(image_id)
    img = Image.new("RGB", (320, 260), tuple(r.randint(150, 235) for _ in range(3)))
    d = ImageDraw.Draw(img)
    d.ellipse((90, 40, 230, 200), fill=(250, 235, 220), outline=(90, 90, 90), width=3)
    d.text((10, 238), image_id, fill=(40, 40, 40))
    return img
