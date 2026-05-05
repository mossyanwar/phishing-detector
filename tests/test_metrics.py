import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from phishdet.metrics import (
    brier_score,
    confusion,
    expected_calibration_error,
    full_report,
    precision_at_prevalence,
    reliability_curve,
    threshold_for_fpr,
    threshold_metrics,
)


def test_confusion_counts():
    c = confusion(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1]))
    assert c == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}


def test_threshold_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = threshold_metrics(y, s, 0.5)
    pred = (s >= 0.5).astype(int)
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))


def test_threshold_metrics_no_positive_predictions():
    m = threshold_metrics(np.array([0, 1]), np.array([0.1, 0.2]), 0.9)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_threshold_for_fpr_respects_target():
    rng = np.random.default_rng(1)
    y = np.array([0] * 1000 + [1] * 200)
    s = np.concatenate([rng.random(1000) * 0.8, 0.5 + rng.random(200) * 0.5])
    for target in (0.0, 0.01, 0.05):
        t = threshold_for_fpr(y, s, target)
        fpr = threshold_metrics(y, s, t)["fpr"]
        assert fpr <= target
    # with target 1%, exactly 10 negatives are allowed above the threshold
    t = threshold_for_fpr(y, s, 0.01)
    assert threshold_metrics(y, s, t)["fp"] == 10


def test_threshold_for_fpr_handles_ties():
    y = np.array([0, 0, 0, 0, 1])
    s = np.array([0.9, 0.9, 0.1, 0.1, 0.95])
    t = threshold_for_fpr(y, s, 0.25)
    # allowing one false positive is impossible with a tie at 0.9, so both are excluded
    assert threshold_metrics(y, s, t)["fp"] == 0


def test_perfectly_calibrated_has_zero_ece():
    probs = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert expected_calibration_error(y, probs) == pytest.approx(0.0)


def test_overconfident_has_positive_ece_and_reliability_bins():
    probs = np.array([0.95] * 10)
    y = np.array([1] * 5 + [0] * 5)
    assert expected_calibration_error(y, probs) == pytest.approx(0.45)
    mean_p, frac, counts = reliability_curve(y, probs)
    assert counts.tolist() == [10] and frac.tolist() == [0.5]


def test_probability_one_goes_in_last_bin():
    mean_p, frac, counts = reliability_curve(np.array([1]), np.array([1.0]), n_bins=10)
    assert counts.tolist() == [1]


def test_brier():
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert brier_score(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_precision_at_prevalence():
    # 90% recall, 1% FPR, 1% phishing: 0.009 / (0.009 + 0.0099)
    assert precision_at_prevalence(0.9, 0.01, 0.01) == pytest.approx(0.009 / (0.009 + 0.0099))
    assert precision_at_prevalence(0.9, 0.0, 0.01) == 1.0
    assert precision_at_prevalence(0.0, 0.0, 0.01) == 0.0


def test_full_report_keys():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    r = full_report(y, s, low_fpr_threshold=0.7)
    assert r["roc_auc"] == pytest.approx(0.75)
    assert r["at_low_fpr"]["fp"] == 0 and r["at_low_fpr"]["tp"] == 1
    assert set(r["at_low_fpr"]["precision_at_prevalence"]) == {"0.01", "0.1"}
