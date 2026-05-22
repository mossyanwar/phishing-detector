import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from phishdet.embedding import EmbeddingModel, embed


class FakeEncoder:
    """Stands in for SentenceTransformer so the tests need no download."""

    def __init__(self):
        self.vec = HashingVectorizer(n_features=64, alternate_sign=False, norm=None)
        self.calls = 0

    def encode(self, texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        self.calls += 1
        X = self.vec.transform(texts).toarray()
        if normalize_embeddings:
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X


def test_embed_returns_float32_unit_vectors():
    X = embed(FakeEncoder(), ["verify your account", "lunch at noon"])
    assert X.dtype == np.float32 and X.shape == (2, 64)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, rtol=1e-5)


def test_train_predict_save_load(toy_texts, tmp_path):
    texts, labels = toy_texts
    enc = FakeEncoder()
    X = embed(enc, texts)
    model = EmbeddingModel(EmbeddingModel.train_head(X, labels, C=10.0), {"encoder": "fake", "revision": "x", "max_seq_length": 32}, enc)
    p = model.predict_proba(texts)
    assert p.shape == (len(texts),)
    assert p[labels == 1].mean() > p[labels == 0].mean()

    model.save(tmp_path / "m")
    again = EmbeddingModel.load(tmp_path / "m", encoder=enc)
    np.testing.assert_allclose(again.predict_proba(texts), p)
    assert again.config["encoder"] == "fake"


def test_encoder_is_not_loaded_until_needed(tmp_path, toy_texts):
    texts, labels = toy_texts
    X = embed(FakeEncoder(), texts)
    model = EmbeddingModel(EmbeddingModel.train_head(X, labels), {"encoder": "fake", "revision": "x", "max_seq_length": 32})
    # scoring precomputed embeddings must not try to load (download) the real encoder
    assert model.predict_proba_embeddings(X).shape == (len(texts),)
    assert model._encoder is None


@pytest.mark.parametrize("C", [0.1, 10.0])
def test_head_is_balanced_logistic_regression(toy_texts, C):
    texts, labels = toy_texts
    head = EmbeddingModel.train_head(embed(FakeEncoder(), texts), labels, C=C)
    assert head.C == C and head.class_weight == "balanced"
