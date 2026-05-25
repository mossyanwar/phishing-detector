"""Train the TF-IDF + logistic regression baseline.

C is picked on the validation set: the smallest (most regularised) C whose
validation average precision is within --tolerance of the best one. The
differences between C values are tiny here, and a more regularised model is
the safer bet when the test data drifts. Scores for validation, test and the
spam side set are written to results/predictions/tfidf_lr_<split>.csv.

Usage:
    python scripts/train_baseline.py --split split   # stratified group split
    python scripts/train_baseline.py --split time    # train <2020 phishing, test >=2020
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from phishdet.baseline import TfidfModel
from phishdet.dataset import by_split, load_jsonl
from phishdet.evaluation import save_predictions

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["split", "time"], default="split")
    ap.add_argument("--C", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--tolerance", type=float, default=0.001)
    args = ap.parse_args(argv)

    kind = "split" if args.split == "split" else "time_split"
    parts = by_split(load_jsonl(ROOT / "data" / "processed" / "emails.jsonl"), kind)
    Xtr = [r.text for r in parts["train"]]
    ytr = np.array([r.label for r in parts["train"]])
    Xva = [r.text for r in parts["val"]]
    yva = np.array([r.label for r in parts["val"]])
    print(f"train {len(Xtr)} ({ytr.sum()} phishing), val {len(Xva)} ({yva.sum()} phishing)")

    search = []
    for C in sorted(args.C):
        t0 = time.perf_counter()
        model = TfidfModel.train(Xtr, ytr, C=C)
        ap_val = average_precision_score(yva, model.predict_proba(Xva))
        search.append({"C": C, "val_average_precision": ap_val, "fit_seconds": time.perf_counter() - t0})
        print(f"C={C:<6} val AP={ap_val:.5f}  ({search[-1]['fit_seconds']:.1f}s)")
    top = max(s["val_average_precision"] for s in search)
    best_C = min(s["C"] for s in search if s["val_average_precision"] >= top - args.tolerance)
    print(f"chosen C={best_C} (best val AP {top:.5f})")
    model = TfidfModel.train(Xtr, ytr, C=best_C)

    name = f"tfidf_lr_{args.split}"
    model.save(ROOT / "models" / f"{name}.joblib")
    scores = {p: model.predict_proba([r.text for r in recs]) for p, recs in parts.items() if recs}
    save_predictions(ROOT / "results" / "predictions" / f"{name}.csv", {p: parts[p] for p in scores}, scores)
    meta = {
        "model": "tfidf_lr",
        "split": args.split,
        "best_C": best_C,
        "search": search,
        "n_features": int(len(model.feature_names)),
        "n_train": len(Xtr),
    }
    (ROOT / "results" / "predictions" / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
