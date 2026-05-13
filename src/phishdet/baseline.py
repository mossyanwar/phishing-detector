"""TF-IDF (word + character n-grams) with logistic regression.

Word n-grams catch phrases ("verify your account"); character n-grams catch
sub-word patterns and survive some spelling tricks ("acc0unt", "paypa1").
Logistic regression keeps the model inspectable: every prediction can be
broken down into per-feature contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# keep "url:host" tokens whole, otherwise ordinary 2+ character words
WORD_TOKEN_PATTERN = r"(?u)url:[\w.\-]+|\b\w\w+\b"


def build_pipeline(
    C: float = 10.0,
    word_max_features: int = 100_000,
    char_max_features: int = 150_000,
    min_df: int = 2,
) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    token_pattern=WORD_TOKEN_PATTERN,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=word_max_features,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=char_max_features,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=3000, solver="liblinear", class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class Contribution:
    feature: str
    value: float  # tf-idf weight in this email
    weight: float  # model coefficient
    contribution: float  # value * weight, in log-odds


class TfidfModel:
    """Thin wrapper around the fitted pipeline with explanation helpers."""

    name = "tfidf_lr"

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._names: np.ndarray | None = None

    @classmethod
    def train(cls, texts: list[str], labels: np.ndarray, **kwargs) -> "TfidfModel":
        pipe = build_pipeline(**kwargs)
        pipe.fit(texts, labels)
        return cls(pipe)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]

    @property
    def feature_names(self) -> np.ndarray:
        if self._names is None:
            union: FeatureUnion = self.pipeline.named_steps["features"]
            names = []
            for prefix, vec in union.transformer_list:
                names.extend(f"{prefix}:{n}" for n in vec.get_feature_names_out())
            self._names = np.array(names, dtype=object)
        return self._names

    @property
    def coef(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].coef_.ravel()

    @property
    def intercept(self) -> float:
        return float(self.pipeline.named_steps["clf"].intercept_[0])

    def top_features(self, k: int = 25) -> dict[str, list[tuple[str, float]]]:
        coef, names = self.coef, self.feature_names
        order = np.argsort(coef)
        return {
            "phishing": [(names[i], float(coef[i])) for i in order[::-1][:k]],
            "legitimate": [(names[i], float(coef[i])) for i in order[:k]],
        }

    def explain(self, text: str, k: int = 10) -> list[Contribution]:
        """Features in this email with the largest absolute contribution to the log-odds."""
        x = self.pipeline.named_steps["features"].transform([text]).tocsr()
        idx, vals = x.indices, x.data
        contrib = vals * self.coef[idx]
        order = np.argsort(-np.abs(contrib))[:k]
        names = self.feature_names
        return [
            Contribution(str(names[idx[i]]), float(vals[i]), float(self.coef[idx[i]]), float(contrib[i]))
            for i in order
        ]

    def contribution_totals(self, text: str) -> dict[str, float]:
        """Sum of contributions per feature family (word / char) for one email."""
        x = self.pipeline.named_steps["features"].transform([text]).tocsr()
        contrib = x.data * self.coef[x.indices]
        fams = np.array([n.split(":", 1)[0] for n in self.feature_names[x.indices]], dtype=object)
        return {fam: float(contrib[fams == fam].sum()) for fam in ("word", "char")}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "TfidfModel":
        return cls(joblib.load(path))
