import numpy as np
import pytest

from phishdet.dataset import Record
from phishdet.evaluation import evaluate, ham_group, load_predictions, phish_era, save_predictions


def _rec(i, label, source="enron", subset="inbox", year=2001):
    return Record(id=f"r{i}", source=source, subset=subset, label=label, year=year, text="", raw_text="",
                  parsed_text="", has_html=False, origin=str(i))


@pytest.fixture()
def parts(tmp_path):
    rng = np.random.default_rng(0)
    recs, scores = {}, {}
    for part, n in (("val", 400), ("test", 400)):
        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        s = np.clip(np.where(labels == 1, 0.75, 0.25) + rng.normal(0, 0.15, n), 0, 1)
        recs[part] = [
            _rec(f"{part}{i}", int(lab), *(("nazario", "phishing-2023", 2023) if lab else ("enron", "inbox", 2001)))
            for i, lab in enumerate(labels)
        ]
        scores[part] = s
    recs["spam"] = [_rec(f"s{i}", -1, "spamassassin", "spam", 2002) for i in range(50)]
    scores["spam"] = rng.random(50)
    path = tmp_path / "preds.csv"
    save_predictions(path, recs, scores)
    return load_predictions(path), scores


def test_predictions_round_trip(parts):
    loaded, scores = parts
    assert set(loaded) == {"val", "test", "spam"}
    np.testing.assert_allclose(loaded["test"].s, scores["test"], atol=1e-6)
    assert loaded["test"].years[0] in (2001, 2023)


def test_evaluate_uses_the_validation_threshold_on_test(parts):
    loaded, _ = parts
    r = evaluate(loaded, target_fpr=0.01)
    assert r["val_at_low_fpr"]["fpr"] <= 0.01
    assert r["test"]["at_low_fpr"]["threshold"] == r["low_fpr_threshold"]
    assert r["test"]["n"] == 400 and r["test"]["n_phishing"] == 200
    assert set(r["test_by_group"]) == {"legit/enron", "phish/2020-2025"}
    assert r["spam_side_set"]["n"] == 50
    band = r["uncertain_band_0.1_0.9"]
    assert 0 < band["share_of_test"] < 1 and band["errors_at_0.5_in_band"] <= band["errors_at_0.5_total"]


def test_group_helpers():
    assert ham_group("enron", "inbox") == "enron"
    assert ham_group("spamassassin", "hard_ham") == "hard_ham"
    assert phish_era(2006) == "2004-2007" and phish_era(2018) == "2015-2019" and phish_era(2024) == "2020-2025"
    assert phish_era(None) == "unknown"
