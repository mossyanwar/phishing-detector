"""Write the largest positive and negative coefficients of the trained TF-IDF model.

Usage:
    python scripts/top_features.py [--split split] [--k 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phishdet.registry import load_model

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="split")
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args(argv)

    model = load_model("tfidf_lr", args.split)
    top = model.top_features(args.k)
    words = {
        side: [(f, w) for f, w in model.top_features(5000)[side] if f.startswith("word:")][: args.k]
        for side in ("phishing", "legitimate")
    }
    out = {"all": top, "word_only": words, "n_features": int(len(model.feature_names))}
    (ROOT / "results" / f"top_features_{args.split}.json").write_text(json.dumps(out, indent=2) + "\n")

    def show(f: str) -> str:
        return "`" + f.replace("|", "\\|").replace("`", "'") + "`"

    lines = [f"# Top features, TF-IDF + LR ({args.split})", "", f"{out['n_features']:,} features in total.", ""]
    for title, d in (("All features", top), ("Word n-grams only", words)):
        lines += [f"## {title}", "", "| towards phishing | weight | towards legitimate | weight |", "|---|---|---|---|"]
        for (fp, wp), (fl, wl) in zip(d["phishing"], d["legitimate"]):
            lines.append(f"| {show(fp)} | {wp:.2f} | {show(fl)} | {wl:.2f} |")
        lines.append("")
    (ROOT / "results" / f"top_features_{args.split}.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
