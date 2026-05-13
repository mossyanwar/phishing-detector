"""Command-line scoring of .eml files.

    phishdet score suspicious.eml
    phishdet score a.eml b.eml --json
    cat message.eml | phishdet score -
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .clean import clean_email
from .parsing import parse_message
from .registry import MODEL_NAMES, load_model, low_fpr_threshold


def _read(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def score_bytes(raw: bytes, model, threshold: float, top: int = 8) -> dict:
    parsed = parse_message(raw)
    text = clean_email(parsed.subject, parsed.body)
    prob = float(model.predict_proba([text])[0])
    out = {
        "subject": parsed.subject,
        "probability_phishing": prob,
        "threshold": threshold,
        "verdict": "phishing" if prob >= threshold else "legitimate",
        "chars_scored": len(text),
    }
    if hasattr(model, "explain"):
        out["top_features"] = [
            {"feature": c.feature, "contribution": c.contribution, "tfidf": c.value, "weight": c.weight}
            for c in model.explain(text, k=top)
        ]
        out["intercept"] = model.intercept
        out["totals"] = model.contribution_totals(text)
    return out


def _print_human(path: str, r: dict, threshold_source: str) -> None:
    print(f"file:      {path}")
    print(f"subject:   {r['subject'][:100]}")
    print(f"P(phish):  {r['probability_phishing']:.3f}  ->  {r['verdict'].upper()}  (threshold {r['threshold']:.3f}, {threshold_source})")
    feats = r.get("top_features")
    if feats:
        print("why (largest contributions to the log-odds; + pushes towards phishing):")
        for f in feats:
            print(f"  {f['contribution']:+.3f}  {f['feature']}")
        t = r["totals"]
        print(
            f"  all word n-grams {t['word']:+.3f}, all char n-grams {t['char']:+.3f}, intercept {r['intercept']:+.3f}"
            f" (log-odds {t['word'] + t['char'] + r['intercept']:+.3f})"
        )
    else:
        print("why:       per-feature explanations are only available for the tfidf_lr model")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="phishdet", description="Score emails for phishing.")
    sub = ap.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("score", help="score one or more .eml files ('-' reads stdin)")
    sc.add_argument("files", nargs="+")
    sc.add_argument("--model", choices=MODEL_NAMES, default="tfidf_lr")
    sc.add_argument("--split", default="split", help="which trained model: 'split' (default) or 'time'")
    sc.add_argument("--model-path", type=Path, help="load the model from this path instead of models/")
    sc.add_argument("--threshold", type=float, help="decision threshold (default: the 1%%-FPR one from results/metrics.json, else 0.5)")
    sc.add_argument("--top", type=int, default=8, help="how many contributing features to show")
    sc.add_argument("--json", action="store_true", help="print JSON lines instead of text")
    args = ap.parse_args(argv)

    try:
        model = load_model(args.model, args.split, path=args.model_path)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.threshold is not None:
        threshold, source = args.threshold, "set on the command line"
    else:
        t = low_fpr_threshold(args.model, args.split)
        threshold, source = (t, "chosen for 1% FPR on validation") if t is not None else (0.5, "default")

    status = 0
    for path in args.files:
        try:
            raw = _read(path)
        except OSError as e:
            print(f"error: cannot read {path}: {e}", file=sys.stderr)
            status = 2
            continue
        r = score_bytes(raw, model, threshold, args.top)
        if args.json:
            print(json.dumps({"file": path, **r}))
        else:
            _print_human(path, r, source)
    return status


if __name__ == "__main__":
    sys.exit(main())
