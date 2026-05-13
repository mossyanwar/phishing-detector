import numpy as np
import pytest

from phishdet.baseline import TfidfModel, build_pipeline


def test_pipeline_has_word_and_char_features(toy_model):
    names = toy_model.feature_names
    assert any(n.startswith("word:") for n in names)
    assert any(n.startswith("char:") for n in names)


def test_url_host_tokens_are_kept_whole(toy_model):
    assert "word:url:secure-bank.example" in set(toy_model.feature_names)


def test_probabilities_in_range_and_separate_toy_classes(toy_model, toy_texts):
    texts, labels = toy_texts
    p = toy_model.predict_proba(texts)
    assert p.shape == (len(texts),)
    assert np.all((p >= 0) & (p <= 1))
    assert p[labels == 1].min() > p[labels == 0].max()


def test_explanations_add_up_to_the_log_odds(toy_model):
    text = "Subject: verify\n\nPlease verify your account at url:secure-bank.example"
    contribs = toy_model.explain(text, k=10**6)
    log_odds = sum(c.contribution for c in contribs) + toy_model.intercept
    expected = toy_model.pipeline.decision_function([text])[0]
    assert log_odds == pytest.approx(expected, abs=1e-4)
    totals = toy_model.contribution_totals(text)
    assert totals["word"] + totals["char"] + toy_model.intercept == pytest.approx(expected, abs=1e-4)


def test_explain_orders_by_absolute_contribution(toy_model):
    contribs = toy_model.explain("Subject: hi\n\nverify your account, thanks for the notes", k=5)
    mags = [abs(c.contribution) for c in contribs]
    assert mags == sorted(mags, reverse=True)
    assert len(contribs) == 5


def test_top_features_point_the_right_way(toy_model):
    top = toy_model.top_features(5)
    assert all(w > 0 for _, w in top["phishing"])
    assert all(w < 0 for _, w in top["legitimate"])


def test_save_and_load_round_trip(toy_model, toy_texts, tmp_path):
    texts, _ = toy_texts
    path = tmp_path / "m.joblib"
    toy_model.save(path)
    again = TfidfModel.load(path)
    np.testing.assert_allclose(again.predict_proba(texts), toy_model.predict_proba(texts), rtol=1e-6)


def test_build_pipeline_uses_balanced_logistic_regression():
    pipe = build_pipeline(C=2.0)
    clf = pipe.named_steps["clf"]
    assert clf.C == 2.0 and clf.class_weight == "balanced"
