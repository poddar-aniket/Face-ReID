from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"similarity_threshold": 0.40, "db_path": "data/face_index.db"}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    for name in ("config.yaml", "config.sample.yaml"):
        path = ROOT / "config" / name
        if not path.exists():
            continue
        try:
            import yaml
        except ImportError:  # PyYAML missing -> fall back to defaults
            break
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
        break
    return cfg


def get_db_path() -> Path:
    p = Path(load_config()["db_path"])
    return p if p.is_absolute() else ROOT / p