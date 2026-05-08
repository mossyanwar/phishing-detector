"""How much of the baseline's score comes from corpus artefacts?

Trains the same TF-IDF + LR pipeline (fixed C) on three views of the same
emails, with the same stratified group split:

  raw      the full raw message, headers and MIME encoding included
  parsed   subject + decoded body, before any cleaning
  cleaned  the model input used everywhere else in this repo

and writes test metrics plus the top features of each to results/leakage.json
and results/leakage.md. If "raw" scores near-perfectly on features like mail
server names or years, that is a measure of how easy it is to fool yourself
with this data.

Usage:
    python scripts/leakage_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phishdet.baseline import TfidfModel
from phishdet.dataset import by_split, load_jsonl
from phishdet.metrics import ranking_metrics, threshold_for_fpr, threshold_metrics

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {"raw": "raw_text", "parsed": "parsed_text", "cleaned": "text"}


def main() -> int:
    parts = by_split(load_jsonl(ROOT / "data" / "processed" / "emails.jsonl"), "split")
    out = {}
    for view, attr in VIEWS.items():
        get = lambda recs: [getattr(r, attr) for r in recs]  # noqa: E731
        ytr = np.array([r.label for r in parts["train"]])
        yva = np.array([r.label for r in parts["val"]])
        yte = np.array([r.label for r in parts["test"]])
        model = TfidfModel.train(get(parts["train"]), ytr, C=3.0)
        sva = model.predict_proba(get(parts["val"]))
        ste = model.predict_proba(get(parts["test"]))
        t = threshold_for_fpr(yva, sva, 0.01)
        at = threshold_metrics(yte, ste, t)
        top = model.top_features(20)
        out[view] = {
            **ranking_metrics(yte, ste),
            "recall_at_1pct_val_fpr": at["recall"],
            "test_fpr_at_threshold": at["fpr"],
            "top_phishing_features": top["phishing"],
            "top_legit_features": top["legitimate"],
        }
        print(f"{view:8s} ROC-AUC {out[view]['roc_auc']:.4f}  recall@thr {at['recall']:.4f}")

    (ROOT / "results" / "leakage.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = [
        "# Leakage check",
        "",
        "Same TF-IDF + LR pipeline (C=3), same stratified group split, three views of the text.",
        "Threshold set for 1% FPR on validation.",
        "",
        "| view | ROC-AUC | PR-AUC | recall @thr | test FPR @thr |",
        "|---|---|---|---|---|",
    ]
    for view, r in out.items():
        lines.append(
            f"| {view} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['recall_at_1pct_val_fpr']:.4f} | {r['test_fpr_at_threshold']:.2%} |"
        )
    for view, r in out.items():
        lines += ["", f"## Top features, {view}", "", "| towards phishing | weight | towards legitimate | weight |", "|---|---|---|---|"]
        for (fp, wp), (fl, wl) in zip(r["top_phishing_features"], r["top_legit_features"]):
            show = lambda f: "`" + f.replace("|", "\\|").replace("`", "'").replace("\n", "\\n") + "`"  # noqa: E731
            lines.append(f"| {show(fp)} | {wp:.2f} | {show(fl)} | {wl:.2f} |")
    (ROOT / "results" / "leakage.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
