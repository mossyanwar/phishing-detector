"""Evaluate an LLM classifier on a fixed sample of the stratified test set.

The sample (data/llm_sample_ids.json, committed) is 100 phishing + 100
legitimate test emails, so the LLM and the trained models are compared on
exactly the same messages. The email text sent to the API is the cleaned model
input (addresses, phone numbers and the recipient's organisation already
replaced), cut to --max-email-chars.

Calling a paid API needs --yes. Without it the script prints the estimated
cost and stops. The fake provider runs offline and is for checking the
pipeline only; its output goes to results/llm_runs/, never to results/llm/.

Usage:
    python scripts/run_llm_eval.py --make-sample                  # (re)create the sample file
    python scripts/run_llm_eval.py --compare-baselines            # small models on the same sample
    python scripts/run_llm_eval.py --provider fake --mode zero-shot    # offline dry run
    python scripts/run_llm_eval.py --provider anthropic --model claude-haiku-4-5 --mode zero-shot --yes
    python scripts/run_llm_eval.py --provider openai --model gpt-4o-mini --mode few-shot --yes

API keys are read from ANTHROPIC_API_KEY / OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from phishdet.dataset import by_split, load_jsonl
from phishdet.evaluation import load_predictions
from phishdet.llm import LLMClassifier, make_provider, summarise
from phishdet.llm.pricing import price_for
from phishdet.llm.prompts import SYSTEM, user_message
from phishdet.metrics import threshold_metrics, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "llm_sample_ids.json"
EST_OUTPUT_TOKENS = 60


def make_sample(n_per_class: int, seed: int) -> None:
    test = by_split(load_jsonl(ROOT / "data" / "processed" / "emails.jsonl"), "split")["test"]
    rng = random.Random(seed)
    ids = []
    for label in (1, 0):
        pool = sorted(r.id for r in test if r.label == label)
        ids += sorted(rng.sample(pool, n_per_class))
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    SAMPLE.write_text(json.dumps({"seed": seed, "n_per_class": n_per_class, "sha256": digest, "ids": ids}, indent=1) + "\n")
    print(f"wrote {len(ids)} ids to {SAMPLE.relative_to(ROOT)}")


def load_sample(n: int | None):
    spec = json.loads(SAMPLE.read_text())
    wanted = spec["ids"]
    by_id = {r.id: r for r in load_jsonl(ROOT / "data" / "processed" / "emails.jsonl")}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} sample ids not found; rebuild data with scripts/prepare_data.py "
                         "or recreate the sample with --make-sample")
    recs = [by_id[i] for i in wanted]
    if n and n < len(recs):
        # keep the class balance when running on a subset
        rng = random.Random(0)
        ph = [r for r in recs if r.label == 1]
        lg = [r for r in recs if r.label == 0]
        recs = rng.sample(ph, n // 2) + rng.sample(lg, n - n // 2)
    return recs


def compare_baselines() -> None:
    recs = load_sample(None)
    ids = {r.id for r in recs}
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    out = {}
    for name in ("tfidf_lr", "minilm_lr"):
        path = ROOT / "results" / "predictions" / f"{name}_split.csv"
        if not path.exists():
            continue
        test = load_predictions(path)["test"]
        idx = [i for i, rid in enumerate(test.ids) if rid in ids]
        y, s = test.y[idx], test.s[idx]
        t_low = metrics[f"{name}_split"]["low_fpr_threshold"]
        entry = {}
        for tname, t in (("at_0.5", 0.5), ("at_low_fpr", t_low)):
            m = threshold_metrics(y, s, t)
            entry[tname] = {
                **{k: m[k] for k in ("threshold", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn")},
                "recall_95ci": wilson_interval(m["tp"], m["tp"] + m["fn"]),
                "fpr_95ci": wilson_interval(m["fp"], m["fp"] + m["tn"]),
            }
        out[name] = {"n": len(idx), **entry}
        print(f"{name}: n={len(idx)} recall@0.5 {entry['at_0.5']['recall']:.3f} fpr@0.5 {entry['at_0.5']['fpr']:.3f}")
    (ROOT / "results" / "llm_sample_baselines.json").write_text(json.dumps(out, indent=2) + "\n")


def estimate(recs, mode: str, max_chars: int, price) -> dict:
    in_tok = sum((len(SYSTEM) + len(user_message(r.text, mode, max_chars))) // 4 for r in recs)
    out_tok = EST_OUTPUT_TOKENS * len(recs)
    cost = None if price is None else (in_tok * price[0] + out_tok * price[1]) / 1e6
    return {"input_tokens_est": in_tok, "output_tokens_est": out_tok, "cost_usd_est": cost}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--compare-baselines", action="store_true")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", choices=["openai", "anthropic", "fake"])
    ap.add_argument("--model")
    ap.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    ap.add_argument("--n", type=int, help="run on a class-balanced subset of the sample")
    ap.add_argument("--max-email-chars", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--price-in", type=float, help="USD per million input tokens (overrides the built-in table)")
    ap.add_argument("--price-out", type=float, help="USD per million output tokens")
    ap.add_argument("--yes", action="store_true", help="really call the paid API")
    args = ap.parse_args(argv)

    if args.make_sample:
        make_sample(args.n_per_class, args.seed)
        return 0
    if args.compare_baselines:
        compare_baselines()
        return 0
    if not args.provider:
        ap.error("--provider is required")

    recs = load_sample(args.n)
    y = np.array([r.label for r in recs])
    model_name = args.model or {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5", "fake": "fake-rules"}[args.provider]
    price = (args.price_in, args.price_out) if args.price_in is not None and args.price_out is not None else price_for(args.provider, model_name)
    est = estimate(recs, args.mode, args.max_email_chars, price)
    print(f"{len(recs)} emails, mode {args.mode}, ~{est['input_tokens_est']:,} input tokens, "
          f"estimated cost {'unknown (no price)' if est['cost_usd_est'] is None else f'${est['cost_usd_est']:.3f}'}")
    if args.provider != "fake" and not args.yes:
        print("not calling the API: add --yes to run it")
        return 0

    provider = make_provider(args.provider, model_name)
    clf = LLMClassifier(provider, mode=args.mode, max_email_chars=args.max_email_chars, price=price)
    t0 = time.perf_counter()

    def progress(i, r):
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(recs)}")

    results = clf.classify_many([r.text for r in recs], workers=args.workers, on_result=progress)
    wall = time.perf_counter() - t0
    summary = {
        "provider": args.provider,
        "model": model_name,
        "mode": args.mode,
        "sample_file": str(SAMPLE.relative_to(ROOT)),
        "max_email_chars": args.max_email_chars,
        "price_per_mtok": price,
        "wall_clock_s": wall,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        **summarise(results, y),
    }
    tp, fn = summary["confusion"]["tp"], summary["confusion"]["fn"]
    fp, tn = summary["confusion"]["fp"], summary["confusion"]["tn"]
    summary["recall_95ci"] = wilson_interval(tp, tp + fn)
    summary["fpr_95ci"] = wilson_interval(fp, fp + tn)

    tag = f"{args.provider}__{model_name}__{args.mode}"
    runs = ROOT / "results" / "llm_runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f"{tag}.jsonl").open("w") as f:  # per-email output stays local (gitignored)
        for rec, res in zip(recs, results):
            f.write(json.dumps({"id": rec.id, "label": rec.label, **res.to_dict()}) + "\n")
    dest = runs if args.provider == "fake" else ROOT / "results" / "llm"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("precision", "recall", "f1", "fpr", "n_invalid", "cost_usd", "latency_s")}, indent=2))
    print(f"wrote {(dest / f'{tag}.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
