"""Sentence-embedding model: frozen MiniLM encoder + logistic regression head.

all-MiniLM-L6-v2 (22M parameters, Apache-2.0) maps an email to a 384-d vector.
Only the small linear head is trained, so training is a few seconds once the
embeddings are computed, and the whole thing runs on a laptop CPU.

The encoder reads at most ``max_seq_length`` word pieces (256 by default), so
it only sees the subject and roughly the first 150-200 words of the body.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# pinned Hugging Face revision of the encoder used for the reported results
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "models" / "hf-cache"


def load_encoder(name: str = ENCODER_NAME, revision: str = ENCODER_REVISION, max_seq_length: int = 256,
                 cache_folder: Path = DEFAULT_CACHE, threads: int | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads or min(8, os.cpu_count() or 1))
    enc = SentenceTransformer(name, revision=revision, device="cpu", cache_folder=str(cache_folder))
    enc.max_seq_length = max_seq_length
    return enc


def embed(encoder, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    return np.asarray(
        encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=show_progress),
        dtype=np.float32,
    )


class EmbeddingModel:
    name = "minilm_lr"

    def __init__(self, head: LogisticRegression, config: dict, encoder=None):
        self.head = head
        self.config = config
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = load_encoder(self.config["encoder"], self.config["revision"], self.config["max_seq_length"])
        return self._encoder

    @staticmethod
    def train_head(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
        head = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        head.fit(X, y)
        return head

    def predict_proba_embeddings(self, X: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(X)[:, 1]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.predict_proba_embeddings(embed(self.encoder, texts, batch_size=self.config.get("batch_size", 64)))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.head, directory / "head.joblib")
        (directory / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path, encoder=None) -> "EmbeddingModel":
        config = json.loads((directory / "config.json").read_text())
        return cls(joblib.load(directory / "head.joblib"), config, encoder)
