"""Index every image in a folder: python -m src.search.build_index [--rebuild]"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ROOT, get_db_path
from src.storage.storage import FaceStore

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _stored_path(path: Path) -> str:
    """Repo-relative posix path so the index works across teammates' machines."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_index(image_dir=None, db_path=None, rebuild=False) -> None:
    from src.embedding.embed import embed_faces

    image_dir = Path(image_dir) if image_dir else ROOT / "data" / "sample_demo_images"
    db_path = Path(db_path) if db_path else get_db_path()
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {image_dir}")
        return

    seen: dict[str, str] = {}
    n_new = n_skipped = n_faces = 0
    with FaceStore(db_path) as store:
        if rebuild:
            store.clear()
        for path in images:
            rel = _stored_path(path)
            if not rebuild and store.has_image_path(rel):
                n_skipped += 1
                continue
            try:
                faces = embed_faces(str(path))
            except (FileNotFoundError, ValueError) as e:
                print(f"[warn] {path.name}: {e}")
                continue
            image_id = faces[0]["image_id"] if faces else path.stem
            if image_id in seen and seen[image_id] != rel:  # image_id has no extension
                print(f"[warn] {path.name}: image_id '{image_id}' collides with {seen[image_id]}, skipped")
                continue
            seen[image_id] = rel
            store.add_image(image_id, rel, faces)
            n_new += 1
            n_faces += len(faces)
            print(f"{path.name}: {len(faces)} face(s)")
        print(f"\nIndexed {n_new} new image(s), {n_faces} face(s); skipped {n_skipped} already indexed.")
        print(f"Index total: {store.count_images()} images, {store.count_faces()} faces -> {db_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="folder of images (default: data/sample_demo_images)")
    ap.add_argument("--db", default=None)
    ap.add_argument("--rebuild", action="store_true", help="wipe and re-index everything")
    args = ap.parse_args()
    build_index(args.images, args.db, args.rebuild)