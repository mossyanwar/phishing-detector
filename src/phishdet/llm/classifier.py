"""LLM-based classifier with output validation and cost / latency tracking."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ..metrics import threshold_metrics
from .pricing import cost_usd, price_for
from .prompts import SYSTEM, user_message
from .providers import Provider, ProviderError

LABELS = {"phishing": 1, "legitimate": 0, "legit": 0}


@dataclass
class LLMResult:
    label: int | None  # 1 phishing, 0 legitimate, None if the output was unusable
    score: float | None  # P(phishing) derived from label + confidence
    confidence: float | None
    reason: str
    valid: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> tuple[dict | None, str | None]:
    """Extract and validate {"label", "confidence", "reason"} from a model reply.

    Returns (parsed, None) or (None, error). Tolerates code fences and text
    around the JSON object; rejects unknown labels and non-numeric confidence.
    """
    if not text or not text.strip():
        return None, "empty response"
    m = _FENCED.search(text)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    if start < 0:
        return None, "no JSON object in response"
    try:
        obj, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return None, "JSON is not an object"
    label = str(obj.get("label", "")).strip().lower()
    if label not in LABELS:
        return None, f"unknown label {obj.get('label')!r}"
    conf = obj.get("confidence")
    if conf is not None:
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            return None, f"confidence is not a number: {conf!r}"
        if conf > 1 and conf <= 100:  # some models answer in percent
            conf = conf / 100
        conf = min(1.0, max(0.0, conf))
    reason = str(obj.get("reason", ""))[:300]
    return {"label": LABELS[label], "confidence": conf, "reason": reason}, None


class LLMClassifier:
    def __init__(self, provider: Provider, mode: str = "zero-shot", max_email_chars: int = 3000,
                 max_tokens: int = 200, price: tuple[float, float] | None = None):
        if mode not in ("zero-shot", "few-shot"):
            raise ValueError(f"unknown mode {mode!r}")
        self.provider = provider
        self.mode = mode
        self.max_email_chars = max_email_chars
        self.max_tokens = max_tokens
        self.price = price if price is not None else price_for(provider.name, provider.model)

    def messages(self, email_text: str) -> tuple[str, str]:
        return SYSTEM, user_message(email_text, self.mode, self.max_email_chars)

    def classify(self, email_text: str) -> LLMResult:
        system, user = self.messages(email_text)
        try:
            c = self.provider.complete(system, user, max_tokens=self.max_tokens)
        except ProviderError as e:
            return LLMResult(None, None, None, "", False, str(e), 0, 0, 0.0, None, self.provider.model)
        parsed, err = parse_response(c.text)
        cost = cost_usd(c.input_tokens, c.output_tokens, self.price)
        if parsed is None:
            return LLMResult(None, None, None, c.text[:300], False, err, c.input_tokens, c.output_tokens,
                             c.latency_s, cost, c.model)
        conf = parsed["confidence"]
        if conf is None:
            score = float(parsed["label"])
        else:
            score = conf if parsed["label"] == 1 else 1.0 - conf
        return LLMResult(parsed["label"], score, conf, parsed["reason"], True, None, c.input_tokens,
                         c.output_tokens, c.latency_s, cost, c.model)

    def classify_many(self, texts: list[str], workers: int = 1, on_result=None) -> list[LLMResult]:
        if workers <= 1:
            out = []
            for i, t in enumerate(texts):
                out.append(self.classify(t))
                if on_result:
                    on_result(i, out[-1])
            return out
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self.classify, texts))
        if on_result:
            for i, r in enumerate(results):
                on_result(i, r)
        return results


def summarise(results: list[LLMResult], y_true: np.ndarray) -> dict:
    """Accuracy, cost and latency for a run.

    Unusable outputs are counted, and for the headline metrics they are treated
    as "legitimate" (the email is let through), which is what a mail filter
    would do if the classifier failed.
    """
    y_true = np.asarray(y_true)
    valid = np.array([r.valid for r in results])
    pred = np.array([r.label if r.valid else 0 for r in results])
    m = threshold_metrics(y_true, pred, 0.5)
    out = {
        "n": len(results),
        "n_invalid": int((~valid).sum()),
        "invalid_rate": float((~valid).mean()) if len(results) else 0.0,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "fpr": m["fpr"],
        "confusion": {k: m[k] for k in ("tp", "fp", "tn", "fn")},
    }
    scores = np.array([r.score if r.valid and r.score is not None else np.nan for r in results])
    ok = ~np.isnan(scores)
    if ok.sum() and len(set(y_true[ok])) == 2:
        from sklearn.metrics import roc_auc_score

        out["roc_auc_valid_only"] = float(roc_auc_score(y_true[ok], scores[ok]))
    lat = np.array([r.latency_s for r in results if r.valid or r.input_tokens])
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    out["tokens"] = {
        "input_total": int(sum(r.input_tokens for r in results)),
        "output_total": int(sum(r.output_tokens for r in results)),
    }
    out["cost_usd"] = {
        "total": float(sum(costs)) if costs else None,
        "per_1000_emails": float(sum(costs) / len(results) * 1000) if costs and results else None,
        "priced": bool(costs),
    }
    out["latency_s"] = (
        {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "mean": float(lat.mean())}
        if len(lat) else None
    )
    return out
