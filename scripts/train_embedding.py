"""Train the MiniLM-embedding + logistic regression model.

Embeddings for every email are computed once and cached in
data/processed/minilm_embeddings.npz. The head's C is picked on validation with
the same rule as the baseline (smallest C within --tolerance of the best
average precision). Predictions go to results/predictions/minilm_lr_<split>.csv.

Usage:
    python scripts/train_embedding.py            # both splits
    python scripts/train_embedding.py --split time
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from phishdet.dataset import by_split, load_jsonl
from phishdet.embedding import ENCODER_NAME, ENCODER_REVISION, EmbeddingModel, embed, load_encoder
from phishdet.evaluation import save_predictions

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed" / "minilm_embeddings.npz"


def get_embeddings(records, max_seq_length: int, batch_size: int) -> dict[str, np.ndarray]:
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=False)
        cached = dict(zip(data["ids"].tolist(), data["X"]))
        if all(r.id in cached for r in records) and int(data["max_seq_length"]) == max_seq_length:
            print(f"using cached embeddings from {CACHE.relative_to(ROOT)}")
            return cached
    enc = load_encoder(max_seq_length=max_seq_length)
    t0 = time.perf_counter()
    X = embed(enc, [r.text for r in records], batch_size=batch_size, show_progress=True)
    print(f"embedded {len(records)} emails in {time.perf_counter() - t0:.0f}s")
    ids = np.array([r.id for r in records])
    np.savez(CACHE, ids=ids, X=X, max_seq_length=max_seq_length)
    return dict(zip(ids.tolist(), X))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["split", "time", "both"], default="both")
    ap.add_argument("--C", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--tolerance", type=float, default=0.001)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    records = load_jsonl(ROOT / "data" / "processed" / "emails.jsonl")
    emb = get_embeddings(records, args.max_seq_length, args.batch_size)

    for split in (["split", "time"] if args.split == "both" else [args.split]):
        parts = by_split(records, "split" if split == "split" else "time_split")
        X = {p: np.stack([emb[r.id] for r in recs]) for p, recs in parts.items() if recs}
        y = {p: np.array([r.label for r in recs]) for p, recs in parts.items() if recs}
        search = []
        for C in sorted(args.C):
            head = EmbeddingModel.train_head(X["train"], y["train"], C)
            ap_val = average_precision_score(y["val"], head.predict_proba(X["val"])[:, 1])
            search.append({"C": C, "val_average_precision": ap_val})
            print(f"[{split}] C={C:<6} val AP={ap_val:.5f}")
        top = max(s["val_average_precision"] for s in search)
        best_C = min(s["C"] for s in search if s["val_average_precision"] >= top - args.tolerance)
        print(f"[{split}] chosen C={best_C}")
        config = {
            "encoder": ENCODER_NAME,
            "revision": ENCODER_REVISION,
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "C": best_C,
        }
        model = EmbeddingModel(EmbeddingModel.train_head(X["train"], y["train"], best_C), config)
        name = f"minilm_lr_{split}"
        model.save(ROOT / "models" / name)
        scores = {p: model.predict_proba_embeddings(Xp) for p, Xp in X.items()}
        save_predictions(ROOT / "results" / "predictions" / f"{name}.csv", {p: parts[p] for p in scores}, scores)
        meta = {"model": "minilm_lr", "split": split, **config, "search": search, "n_train": int(len(y["train"]))}
        (ROOT / "results" / "predictions" / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
