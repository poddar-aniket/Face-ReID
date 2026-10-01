"""Calibrate the face-match threshold using LFW verification pairs.

Run from the repository root with:
    python -m src.calibration.calibrate_threshold [--limit N] [--use-cache]
"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMBEDDING_DIM = 512

OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration"
CACHE_PATH = OUTPUT_DIR / "lfw_embeddings.npz"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
ROC_PATH = OUTPUT_DIR / "roc_curve.png"
ACCURACY_PATH = OUTPUT_DIR / "accuracy_vs_threshold.png"


def _positive_limit(value: str) -> int:
    limit = int(value)
    if limit < 1:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return limit


def _largest_face(records: list[dict]) -> np.ndarray:
    record = max(records, key=lambda item: float(item["detection_confidence"]))
    embedding = np.asarray(record["embedding"], dtype=np.float32)
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected an embedding with shape ({EMBEDDING_DIM},), "
            f"received {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Face embedding contains non-finite values")
    return embedding


def _save_lfw_image(image_rgb: np.ndarray, image_path: Path) -> None:
    image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError(f"Could not encode LFW image as PNG: {image_path}")
    encoded.tofile(image_path)


def _compute_embeddings(limit: int | None) -> tuple[np.ndarray, np.ndarray, int, bool]:
    from src.embedding.embed import embed_faces

    lfw = fetch_lfw_pairs(
        subset="10_folds",
        color=True,
        resize=1.0,
        funneled=True,
    )
    pair_count = len(lfw.target)
    processed_count = min(pair_count, limit) if limit is not None else pair_count
    pairs = lfw.pairs[:processed_count]
    labels = np.asarray(lfw.target[:processed_count], dtype=np.uint8)
    embeddings = np.full(
        (processed_count, 2, EMBEDDING_DIM),
        np.nan,
        dtype=np.float32,
    )
    n_skipped = 0

    with TemporaryDirectory(prefix="lfw_calibration_") as temp_dir:
        image_paths = (
            Path(temp_dir) / "lfw_pair_image_0.png",
            Path(temp_dir) / "lfw_pair_image_1.png",
        )
        for pair_index, pair in enumerate(pairs, start=1):
            pair_embeddings = []
            for image, image_path in zip(pair, image_paths):
                _save_lfw_image(image, image_path)
                records = embed_faces(str(image_path))
                pair_embeddings.append(_largest_face(records) if records else None)

            if any(embedding is None for embedding in pair_embeddings):
                n_skipped += 1
            else:
                embeddings[pair_index - 1] = np.stack(pair_embeddings)

            if pair_index % 100 == 0 or pair_index == processed_count:
                print(
                    f"Processed {pair_index}/{processed_count} pairs "
                    f"({n_skipped} skipped)",
                    flush=True,
                )

    return embeddings, labels, n_skipped, limit is None or processed_count == pair_count


def _write_cache(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cache_is_complete: bool,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        embeddings=embeddings,
        labels=labels,
        cache_is_complete=np.asarray(cache_is_complete),
    )
    print(f"Saved embedding cache to {CACHE_PATH}")


def _read_cache(limit: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"No embedding cache found at {CACHE_PATH}; run once without --use-cache"
        )

    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=np.uint8)
        cache_is_complete = bool(cache["cache_is_complete"].item())

    if embeddings.ndim != 3 or embeddings.shape[1:] != (2, EMBEDDING_DIM):
        raise ValueError(f"Invalid embedding cache shape: {embeddings.shape}")
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("Embedding cache labels do not match cached pairs")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Embedding cache labels must be 0 or 1")
    if limit is None and not cache_is_complete:
        raise ValueError("The cache contains a limited run; rerun without --limit")
    if limit is not None and len(labels) < limit:
        raise ValueError(
            f"Cache contains {len(labels)} pairs, fewer than requested limit {limit}"
        )

    if limit is not None:
        embeddings = embeddings[:limit]
        labels = labels[:limit]
    skipped = int((~np.isfinite(embeddings).all(axis=(1, 2))).sum())
    print(f"Loaded {len(labels)} cached pairs ({skipped} skipped) from {CACHE_PATH}")
    return embeddings, labels, skipped


def _rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    predicted_same = scores >= threshold
    actual_same = labels == 1
    positives = int(actual_same.sum())
    negatives = len(labels) - positives
    true_positive = int(np.count_nonzero(predicted_same & actual_same))
    false_positive = int(np.count_nonzero(predicted_same & ~actual_same))
    tpr = true_positive / positives if positives else 0.0
    fpr = false_positive / negatives if negatives else 0.0
    return tpr, fpr


def _plot_results(
    thresholds: np.ndarray,
    accuracies: np.ndarray,
    best_threshold: float,
    labels: np.ndarray,
    scores: np.ndarray,
    roc_data: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.plot(thresholds, accuracies, label="Accuracy")
    axis.axvline(
        best_threshold,
        color="tab:red",
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}",
    )
    axis.set_xlabel("Cosine similarity threshold")
    axis.set_ylabel("Accuracy")
    axis.set_title("LFW accuracy by threshold")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(ACCURACY_PATH, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    if roc_data is not None:
        false_positive_rate, true_positive_rate = roc_data
        axis.plot(false_positive_rate, true_positive_rate, label="ROC curve")
        best_tpr, best_fpr = _rates(labels, scores, best_threshold)
        axis.scatter(
            [best_fpr],
            [best_tpr],
            color="tab:red",
            label=f"Best accuracy threshold = {best_threshold:.2f}",
            zorder=3,
        )
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "ROC requires both positive and negative pairs",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("LFW receiver operating characteristic")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(ROC_PATH, dpi=150)
    plt.close(figure)


def calibrate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_pairs_total: int,
    n_skipped: int,
) -> dict:
    valid_pairs = np.isfinite(embeddings).all(axis=(1, 2))
    valid_embeddings = embeddings[valid_pairs]
    valid_labels = labels[valid_pairs]
    if len(valid_labels) == 0:
        raise ValueError("No valid LFW pairs remain after skipping undetected faces")

    scores = np.einsum(
        "ij,ij->i",
        valid_embeddings[:, 0, :],
        valid_embeddings[:, 1, :],
    )
    thresholds = np.round(np.arange(-1.0, 1.0001, 0.01), 2)
    accuracies = np.asarray(
        [np.mean((scores >= threshold) == (valid_labels == 1)) for threshold in thresholds]
    )
    best_index = int(np.argmax(accuracies))
    best_threshold = float(thresholds[best_index])
    has_both_classes = np.unique(valid_labels).size == 2

    auc = None
    youden_threshold = None
    tpr_at_fpr_001 = None
    roc_data = None
    if has_both_classes:
        fpr, tpr, roc_thresholds = roc_curve(valid_labels, scores)
        auc = float(roc_auc_score(valid_labels, scores))
        finite_thresholds = np.isfinite(roc_thresholds)
        youden_index = int(np.argmax((tpr - fpr)[finite_thresholds]))
        youden_threshold = round(
            float(roc_thresholds[finite_thresholds][youden_index]),
            6,
        )
        tpr_at_fpr_001 = float(np.interp(0.01, fpr, tpr))
        roc_data = (fpr, tpr)

    _plot_results(
        thresholds,
        accuracies,
        best_threshold,
        valid_labels,
        scores,
        roc_data,
    )
    return {
        "n_pairs_total": int(n_pairs_total),
        "n_skipped": int(n_skipped),
        "n_pairs_used": int(len(valid_labels)),
        "best_threshold": best_threshold,
        "best_accuracy": float(accuracies[best_index]),
        "youden_threshold": youden_threshold,
        "auc": auc,
        "tpr_at_fpr_0.01": tpr_at_fpr_001,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the cosine-similarity threshold using LFW pairs."
    )
    parser.add_argument(
        "--limit",
        type=_positive_limit,
        default=None,
        help="Process only the first N LFW pairs",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=f"Load embeddings from {CACHE_PATH} instead of running the model",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.use_cache:
        embeddings, labels, n_skipped = _read_cache(args.limit)
    else:
        embeddings, labels, n_skipped, cache_is_complete = _compute_embeddings(args.limit)
        _write_cache(embeddings, labels, cache_is_complete)

    metrics = calibrate(
        embeddings,
        labels,
        n_pairs_total=len(labels),
        n_skipped=n_skipped,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(
        "Calibration complete: "
        f"{metrics['n_pairs_used']}/{metrics['n_pairs_total']} pairs used, "
        f"{metrics['n_skipped']} skipped; "
        f"best threshold={metrics['best_threshold']:.2f}, "
        f"accuracy={metrics['best_accuracy']:.4f}, "
        f"Youden threshold={metrics['youden_threshold']}, "
        f"AUC={metrics['auc']}"
    )
    print(f"Metrics: {METRICS_PATH}")
    print(f"Plots: {ROC_PATH}, {ACCURACY_PATH}")


if __name__ == "__main__":
    main()
