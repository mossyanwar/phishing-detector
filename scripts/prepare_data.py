"""Build data/processed/emails.jsonl from the raw corpora.

Steps: parse -> clean -> drop empty / non-English -> exact dedupe ->
near-duplicate grouping and capping -> stratified group split and time split.
Counts for every step are written to results/data_stats.json.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from phishdet.corpora import LEGIT, PHISHING, SPAM, iter_enron, iter_nazario, iter_spamassassin
from phishdet.dataset import (
    _fingerprint,
    build_records,
    cap_groups,
    dedupe_exact,
    filter_english,
    near_duplicate_groups,
    save_jsonl,
    stratified_group_split,
    time_split,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def count_by(records, key) -> dict:
    c = Counter(key(r) for r in records)
    return dict(sorted(c.items(), key=lambda kv: str(kv[0])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enron-n", type=int, default=4000, help="messages to sample from Enron inboxes")
    ap.add_argument("--near-dup-threshold", type=float, default=0.8)
    ap.add_argument("--max-per-group", type=int, default=3)
    ap.add_argument("--time-cutoff", type=int, default=2020)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    stats: dict = {"params": vars(args)}

    print("reading corpora ...")
    raw_iters = [
        iter_nazario(RAW / "nazario"),
        iter_spamassassin(
            RAW / "spamassassin" / "extracted",
            {"easy_ham": LEGIT, "easy_ham_2": LEGIT, "hard_ham": LEGIT, "spam": SPAM, "spam_2": SPAM},
        ),
    ]
    enron_tar = RAW / "enron" / "enron_mail_20150507.tar.gz"
    if enron_tar.exists():
        raw_iters.append(iter_enron(enron_tar, n=args.enron_n, seed=args.seed))
    else:
        print("warning: Enron tarball missing, continuing without it", file=sys.stderr)

    records, dropped = [], Counter()
    raw_counts: Counter = Counter()
    for it in raw_iters:
        for em in it:
            raw_counts[f"{em.source}/{em.subset}"] += 1
            recs, d = build_records([em])
            records.extend(recs)
            dropped.update(d)
    stats["raw_counts"] = dict(sorted(raw_counts.items()))
    print(f"parsed {len(records)} records")

    records, d = filter_english(records)
    dropped.update(d)

    main_recs = [r for r in records if r.label in (LEGIT, PHISHING)]
    spam_recs = [r for r in records if r.label == SPAM]

    main_recs, d = dedupe_exact(main_recs)
    dropped.update(d)
    spam_recs, d = dedupe_exact(spam_recs)
    dropped.update(d)
    main_fps = {_fingerprint(r.text) for r in main_recs}
    before = len(spam_recs)
    spam_recs = [r for r in spam_recs if _fingerprint(r.text) not in main_fps]
    dropped["spamassassin:spam_duplicates_main"] += before - len(spam_recs)

    print(f"grouping near-duplicates among {len(main_recs)} records ...")
    groups = near_duplicate_groups([r.text for r in main_recs], threshold=args.near_dup_threshold, chunk=500)
    sizes = Counter(Counter(groups.tolist()).values())
    stats["near_duplicate_group_sizes"] = {str(k): v for k, v in sorted(sizes.items())}
    keep, d = cap_groups(main_recs, groups, args.max_per_group, seed=args.seed)
    dropped.update(d)
    main_recs = [main_recs[i] for i in keep]
    groups = groups[keep]

    labels = np.array([r.label for r in main_recs])
    split = stratified_group_split(labels, groups, seed=args.seed)
    tsplit = time_split(labels, [r.year for r in main_recs], groups, split, cutoff=args.time_cutoff)
    for r, g, s, t in zip(main_recs, groups, split, tsplit):
        r.meta = {"group": int(g), "split": str(s), "time_split": str(t)}
    for r in spam_recs:
        r.meta = {"group": -1, "split": "spam", "time_split": "spam"}

    save_jsonl(main_recs + spam_recs, OUT / "emails.jsonl")

    stats["dropped"] = dict(sorted(dropped.items()))
    stats["final"] = {
        "total": len(main_recs),
        "by_label": count_by(main_recs, lambda r: r.label),
        "by_source_subset": count_by(main_recs, lambda r: f"{r.source}/{r.subset}"),
        "by_split_label": count_by(main_recs, lambda r: f"{r.meta['split']}/{r.label}"),
        "by_time_split_label": count_by(main_recs, lambda r: f"{r.meta['time_split']}/{r.label}"),
        "phishing_by_year": count_by([r for r in main_recs if r.label == 1], lambda r: r.year),
        "legit_by_year": count_by([r for r in main_recs if r.label == 0], lambda r: r.year),
        "spam_side_set": len(spam_recs),
        "html_share_by_label": {
            str(lab): round(float(np.mean([r.has_html for r in main_recs if r.label == lab])), 3) for lab in (0, 1)
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats["final"], indent=2))
    print(json.dumps(stats["dropped"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
