"""Time how long scoring takes, end to end, on this machine.

For a fixed sample of test emails, times (a) parse + clean of the raw message
and (b) model inference, one email at a time (the CLI / mail-filter case) and
as one batch. Writes results/latency.json.

Usage:
    python scripts/latency.py [--n 300] [--models tfidf_lr minilm_lr]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from phishdet.clean import clean_email
from phishdet.dataset import by_split, load_jsonl
from phishdet.parsing import parse_message
from phishdet.registry import load_model

ROOT = Path(__file__).resolve().parents[1]


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def summary_ms(seconds: list[float]) -> dict:
    a = np.array(seconds) * 1000
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", nargs="+", default=["tfidf_lr", "minilm_lr"])
    args = ap.parse_args(argv)

    test = by_split(load_jsonl(ROOT / "data" / "processed" / "emails.jsonl"), "split")["test"]
    sample = random.Random(0).sample(test, args.n)

    prep = []
    texts = []
    for r in sample:
        raw = r.raw_text.encode("utf-8", errors="replace")
        t0 = time.perf_counter()
        p = parse_message(raw)
        texts.append(clean_email(p.subject, p.body))
        prep.append(time.perf_counter() - t0)

    out = {
        "machine": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "python": platform.python_version()},
        "n_emails": args.n,
        "parse_and_clean": summary_ms(prep),
        "models": {},
    }
    for name in args.models:
        try:
            model = load_model(name, "split")
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        model.predict_proba(texts[:5])  # warm-up (loads lazily initialised parts)
        single = []
        for t in texts:
            t0 = time.perf_counter()
            model.predict_proba([t])
            single.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model.predict_proba(texts)
        batch = time.perf_counter() - t0
        out["models"][name] = {
            "single_email": summary_ms(single),
            "batch_total_s": batch,
            "batch_emails_per_s": args.n / batch,
        }
        print(f"{name}: single median {out['models'][name]['single_email']['median_ms']:.1f} ms, "
              f"batch {args.n / batch:.0f} emails/s")
    print(f"parse+clean median {out['parse_and_clean']['median_ms']:.2f} ms")
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
