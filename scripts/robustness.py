"""Score test phishing after simple evasion tricks and compare with the original.

Every phishing email in the stratified test set is perturbed (see
src/phishdet/robustness.py), passed through the normal cleaning step and
scored. Recall is measured at each model's validation-chosen low-FPR threshold.
Two extra rows switch off Unicode normalisation to show what that defence is worth.

Usage:
    python scripts/robustness.py [--models tfidf_lr minilm_lr]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from phishdet import plots
from phishdet.clean import clean_email
from phishdet.dataset import by_split, load_jsonl
from phishdet.registry import load_model, low_fpr_threshold
from phishdet.robustness import homoglyphs, make_perturbations, split_parsed, zero_width

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"tfidf_lr": "TF-IDF + LR", "minilm_lr": "MiniLM + LR"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["tfidf_lr", "minilm_lr"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    parts = by_split(load_jsonl(ROOT / "data" / "processed" / "emails.jsonl"), "split")
    phish = [r for r in parts["test"] if r.label == 1]
    pad_rng = random.Random(args.seed)
    legit_train = [r for r in parts["train"] if r.label == 0]
    padding = [split_parsed(r.parsed_text)[1][:600] for r in pad_rng.sample(legit_train, 200)]
    perturbations = make_perturbations(padding)

    variants: dict[str, list[str]] = {}
    for name, fn in perturbations.items():
        rng = random.Random(args.seed)
        texts = []
        for r in phish:
            s, b = fn(*split_parsed(r.parsed_text), rng)
            texts.append(clean_email(s, b))
        variants[name] = texts
    # the same two attacks with the Unicode defence switched off
    for name, fn in (("homoglyphs, no unicode normalisation", homoglyphs), ("zero-width, no unicode normalisation", zero_width)):
        rng = random.Random(args.seed)
        texts = []
        for r in phish:
            s, b = split_parsed(r.parsed_text)
            texts.append(clean_email(fn(s, rng), fn(b, rng), unicode_normalise=False))
        variants[name] = texts

    rows = []
    for model_name in args.models:
        try:
            model = load_model(model_name, "split")
        except FileNotFoundError as e:
            print(f"skip {model_name}: {e}")
            continue
        t = low_fpr_threshold(model_name, "split")
        if t is None:
            print(f"skip {model_name}: no threshold in results/metrics.json, run scripts/evaluate.py")
            continue
        base = model.predict_proba(variants["none"])
        for name, texts in variants.items():
            s = base if name == "none" else model.predict_proba(texts)
            rows.append(
                {
                    "model": LABELS.get(model_name, model_name),
                    "perturbation": name,
                    "n": len(texts),
                    "threshold": t,
                    "recall": float(np.mean(s >= t)),
                    "mean_score": float(np.mean(s)),
                    "mean_score_change": float(np.mean(s - base)),
                }
            )
            print(f"{model_name:10s} {name:40s} recall {rows[-1]['recall']:.3f}  mean score {rows[-1]['mean_score']:.3f}")

    (ROOT / "results" / "robustness.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Robustness",
        "",
        f"All {len(phish)} phishing emails in the stratified test set, perturbed before cleaning.",
        "Recall at each model's 1%-validation-FPR threshold.",
        "",
        "| model | perturbation | recall | mean score | mean change |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['perturbation']} | {r['recall']:.3f} | {r['mean_score']:.3f} | {r['mean_score_change']:+.3f} |")
    (ROOT / "results" / "robustness.md").write_text("\n".join(lines) + "\n")
    plots.robustness(rows, ROOT / "results" / "figures" / "robustness.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
