"""Load a trained model by name. Models live in models/ (gitignored)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
METRICS = ROOT / "results" / "metrics.json"
MODEL_NAMES = ("tfidf_lr", "minilm_lr")


def model_path(name: str, split: str = "split", models_dir: Path = MODELS_DIR) -> Path:
    if name == "tfidf_lr":
        return models_dir / f"tfidf_lr_{split}.joblib"
    if name == "minilm_lr":
        return models_dir / f"minilm_lr_{split}"
    raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")


def load_model(name: str, split: str = "split", models_dir: Path = MODELS_DIR, path: Path | None = None):
    path = path or model_path(name, split, models_dir)
    if not path.exists():
        hint = "scripts/train_baseline.py" if name == "tfidf_lr" else "scripts/train_embedding.py"
        raise FileNotFoundError(f"no trained model at {path}; run python {hint} --split {split} first")
    if name == "tfidf_lr":
        from .baseline import TfidfModel

        return TfidfModel.load(path)
    from .embedding import EmbeddingModel

    return EmbeddingModel.load(path)


def low_fpr_threshold(name: str, split: str = "split", metrics_path: Path = METRICS) -> float | None:
    """The validation-chosen threshold recorded by scripts/evaluate.py, if any."""
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text())
    entry = data.get(f"{name}_{split}")
    return float(entry["low_fpr_threshold"]) if entry else None
