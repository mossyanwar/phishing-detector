"""List the stratified-test errors of each model at its low-FPR threshold.

Writes:
  results/error_analysis.json         ids, scores, sources and (for TF-IDF) the
                                      top features of every error; no email text
  data/processed/error_analysis.md    the same with a 500-character excerpt of
                                      each email, for reading; gitignored because
                                      it contains corpus text

Usage:
    python scripts/error_analysis.py [--models tfidf_lr minilm_lr]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from phishdet.dataset import by_split, load_jsonl
from phishdet.evaluation import ham_group, phish_era
from phishdet.registry import load_model, low_fpr_threshold

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["tfidf_lr", "minilm_lr"])
    args = ap.parse_args(argv)

    test = by_split(load_jsonl(ROOT / "data" / "processed" / "emails.jsonl"), "split")["test"]
    texts = [r.text for r in test]
    y = np.array([r.label for r in test])
    summary, md = {}, ["# Error analysis (local only, contains corpus text)", ""]
    for name in args.models:
        try:
            model = load_model(name, "split")
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        t = low_fpr_threshold(name, "split")
        s = model.predict_proba(texts)
        fp = [i for i in np.argsort(-s) if y[i] == 0 and s[i] >= t]
        fn = [i for i in np.argsort(s) if y[i] == 1 and s[i] < t]
        entries = {"false_positives": [], "false_negatives": []}
        for kind, idx in (("false_positives", fp), ("false_negatives", fn)):
            md += [f"## {name}: {kind} ({len(idx)})", ""]
            for i in idx:
                r = test[i]
                e = {"id": r.id, "score": float(s[i]), "source": r.source, "subset": r.subset, "year": r.year}
                if hasattr(model, "explain"):
                    e["top_features"] = [(c.feature, round(c.contribution, 3)) for c in model.explain(r.text, k=6)]
                entries[kind].append(e)
                excerpt = r.text[:500].replace("\n", " ")
                md += [f"- `{r.id}` {r.source}/{r.subset} {r.year} score {s[i]:.3f}", f"  {e.get('top_features', '')}", f"  > {excerpt}", ""]
        summary[name] = {
            "threshold": t,
            "n_false_positives": len(fp),
            "n_false_negatives": len(fn),
            "false_positives_by_source": dict(Counter(ham_group(test[i].source, test[i].subset) for i in fp)),
            "false_negatives_by_era": dict(Counter(phish_era(test[i].year) for i in fn)),
            **entries,
        }
        print(f"{name}: {len(fp)} FP {summary[name]['false_positives_by_source']}, {len(fn)} FN {summary[name]['false_negatives_by_era']}")
    (ROOT / "results" / "error_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "data" / "processed" / "error_analysis.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
