"""Evaluation metrics, written out explicitly so they can be unit-tested."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1, FPR and the confusion matrix at one threshold (score >= t is phishing)."""
    pred = (np.asarray(scores) >= threshold).astype(int)
    c = confusion(y_true, pred)
    precision = _safe_div(c["tp"], c["tp"] + c["fp"])
    recall = _safe_div(c["tp"], c["tp"] + c["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(c["fp"], c["fp"] + c["tn"])
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, **c}


def threshold_for_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on (y_true, scores) is <= target_fpr.

    Meant to be fitted on a validation set and then applied, unchanged, to the
    test set.
    """
    y_true = np.asarray(y_true)
    neg = np.sort(np.asarray(scores)[y_true == 0])[::-1]
    if len(neg) == 0:
        raise ValueError("need at least one negative example")
    allowed = int(np.floor(target_fpr * len(neg)))
    if allowed >= len(neg):
        return float(neg[-1])
    # predictions are score >= t, so t must be strictly above the (allowed+1)-th highest negative
    return float(np.nextafter(neg[allowed], np.inf))


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def reliability_curve(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    """Equal-width bins. Returns (mean predicted prob, observed positive rate, count) per non-empty bin."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    mean_p, frac_pos, counts = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            mean_p.append(float(probs[m].mean()))
            frac_pos.append(float(y_true[m].mean()))
            counts.append(int(m.sum()))
    return np.array(mean_p), np.array(frac_pos), np.array(counts)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    mean_p, frac_pos, counts = reliability_curve(y_true, probs, n_bins)
    return float(np.sum(counts * np.abs(mean_p - frac_pos)) / counts.sum())


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(probs, dtype=float) - y_true) ** 2))


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Precision the same detector would have if only ``prevalence`` of mail were phishing.

    The test set is ~40% phishing; a real inbox is not. With TPR and FPR fixed,
    precision = TPR*p / (TPR*p + FPR*(1-p)).
    """
    num = recall * prevalence
    den = num + fpr * (1 - prevalence)
    return _safe_div(num, den)


def full_report(
    y_true: np.ndarray, scores: np.ndarray, low_fpr_threshold: float, prevalences=(0.01, 0.1)
) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    at_half = threshold_metrics(y_true, scores, 0.5)
    at_low = threshold_metrics(y_true, scores, low_fpr_threshold)
    out = {
        "n": int(len(y_true)),
        "n_phishing": int(y_true.sum()),
        **ranking_metrics(y_true, scores),
        "brier": brier_score(y_true, scores),
        "ece": expected_calibration_error(y_true, scores),
        "at_0.5": at_half,
        "at_low_fpr": at_low,
    }
    out["at_low_fpr"]["precision_at_prevalence"] = {
        str(p): precision_at_prevalence(at_low["recall"], at_low["fpr"], p) for p in prevalences
    }
    return out


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; used for the small LLM sample."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (float(lo), float(hi))
