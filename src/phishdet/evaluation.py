"""Turn a predictions CSV into the numbers reported in REPORT.md."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import full_report, threshold_for_fpr, threshold_metrics


@dataclass
class Part:
    ids: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    years: list[int | None] = field(default_factory=list)

    @property
    def y(self) -> np.ndarray:
        return np.array(self.labels, dtype=int)

    @property
    def s(self) -> np.ndarray:
        return np.array(self.scores, dtype=float)


def save_predictions(path: Path, parts: dict[str, list], scores: dict[str, np.ndarray]) -> None:
    """Write one row per email: id, part (train/val/test/spam), label, score and source info."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "part", "label", "score", "source", "subset", "year"])
        for part, recs in parts.items():
            for r, s in zip(recs, scores[part]):
                w.writerow([r.id, part, r.label, f"{s:.6f}", r.source, r.subset, r.year if r.year else ""])


def load_predictions(path: Path) -> dict[str, Part]:
    parts: dict[str, Part] = defaultdict(Part)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            p = parts[row["part"]]
            p.ids.append(row["id"])
            p.labels.append(int(row["label"]))
            p.scores.append(float(row["score"]))
            p.sources.append(row["source"])
            p.subsets.append(row["subset"])
            p.years.append(int(row["year"]) if row["year"] else None)
    return dict(parts)


def ham_group(source: str, subset: str) -> str:
    return "enron" if source == "enron" else subset


def phish_era(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year < 2010:
        return "2004-2007"
    if year < 2020:
        return "2015-2019"
    return "2020-2025"


def evaluate(parts: dict[str, Part], target_fpr: float = 0.01) -> dict:
    """Threshold chosen on validation for ``target_fpr``, then applied to test."""
    val, test = parts["val"], parts["test"]
    t_low = threshold_for_fpr(val.y, val.s, target_fpr)
    report = {
        "target_fpr": target_fpr,
        "low_fpr_threshold": t_low,
        "val_at_low_fpr": threshold_metrics(val.y, val.s, t_low),
        "test": full_report(test.y, test.s, t_low),
    }

    # false-positive rate per legitimate source, recall per phishing era
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (lab, src, sub, yr) in enumerate(zip(test.labels, test.sources, test.subsets, test.years)):
        key = f"legit/{ham_group(src, sub)}" if lab == 0 else f"phish/{phish_era(yr)}"
        groups[key].append(i)
    by_group = {}
    y, s = test.y, test.s
    for key, idx in sorted(groups.items()):
        idx = np.array(idx)
        flagged_low = float(np.mean(s[idx] >= t_low))
        flagged_half = float(np.mean(s[idx] >= 0.5))
        name = "fpr" if key.startswith("legit") else "recall"
        by_group[key] = {"n": int(len(idx)), f"{name}_at_low_fpr": flagged_low, f"{name}_at_0.5": flagged_half}
    report["test_by_group"] = by_group

    # recall per year (useful for the time split)
    years = defaultdict(list)
    for i, (lab, yr) in enumerate(zip(test.labels, test.years)):
        if lab == 1 and yr is not None:
            years[yr].append(i)
    report["test_recall_by_year"] = {
        str(yr): {"n": len(idx), "recall_at_low_fpr": float(np.mean(s[np.array(idx)] >= t_low))}
        for yr, idx in sorted(years.items())
    }

    # how much traffic a cascade would pass on to a slower, costlier second
    # opinion (e.g. an LLM) if it only escalated the uncertain middle band
    err = (s >= 0.5).astype(int) != y
    band = (s >= 0.1) & (s < 0.9)
    report["uncertain_band_0.1_0.9"] = {
        "share_of_test": float(band.mean()),
        "errors_at_0.5_in_band": int(err[band].sum()),
        "errors_at_0.5_total": int(err.sum()),
        "error_rate_outside_band": float(err[~band].mean()) if (~band).any() else 0.0,
    }

    if "spam" in parts:
        sp = parts["spam"].s
        report["spam_side_set"] = {
            "n": int(len(sp)),
            "flagged_at_low_fpr": float(np.mean(sp >= t_low)),
            "flagged_at_0.5": float(np.mean(sp >= 0.5)),
            "median_score": float(np.median(sp)),
        }
    return report
