"""Build the modelling dataset: parse, clean, filter, deduplicate, split.

Splitting is done by near-duplicate group. Phishing campaigns send hundreds of
almost identical messages; if copies of one campaign land in both train and test,
the test score mostly measures memorisation. Grouping near-duplicates and keeping
each group on one side of the split avoids that.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from .clean import clean_email
from .corpora import RawEmail
from .parsing import parse_message

MIN_BODY_CHARS = 30

# Very common function words per language. A crude but transparent language
# check: a message counts as English unless another language's function words
# outnumber the English ones, or most of its letters are non-Latin.
_STOPWORDS = {
    "en": """the a an and or but if of to in on at by for with from as is are was were be been
        have has had do does did will would can could should may this that these those it its
        you your we our they their he she his her not all any more there here when which who what""",
    # no single letters and nothing that is also a common English word or URL part (com, do, no ...)
    "es": "el la los las de del que en un una por para con su sus es se al lo como pero usted nuestro",
    "pt": "os de da dos das que em um uma por para seu sua ao na você nosso não",
    "fr": "le la les des du de et en un une pour par avec sur est sont vous votre nous cette qui pas",
    "de": "der die das und ist nicht ein eine mit für auf den dem des sie ihr ihre wir zu von bitte",
    "it": "il lo la gli di del della che un una per con sono non al nel tuo vostro",
    "nl": "de het een en van te dat met voor niet zijn uw wij ons deze aan",
}
_STOP = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
_WORD_RE = re.compile(r"[^\W\d_]+")
_PLACEHOLDERS = {"url", "email", "orgname", "recipient", "name", "year", "subject"}


def language_scores(text: str) -> dict[str, int]:
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _PLACEHOLDERS]
    return {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}


def looks_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > 0.3:
        return False
    scores = language_scores(text)
    en = scores.pop("en")
    other = max(scores.values())
    # require a few foreign function words so short English notes are not dropped
    return not (other >= 3 and other > en)


@dataclass
class Record:
    id: str
    source: str
    subset: str
    label: int
    year: int | None
    text: str
    raw_text: str  # original headers + body (truncated), only for the leakage check
    has_html: bool
    origin: str
    meta: dict = field(default_factory=dict)


def _raw_view(raw: bytes, limit: int = 6000) -> str:
    return raw.decode("utf-8", errors="replace")[:limit]


def build_records(emails: Iterable[RawEmail], max_chars: int = 4000) -> tuple[list[Record], Counter]:
    """Parse and clean every message. Returns records and a counter of drop reasons."""
    out: list[Record] = []
    dropped: Counter = Counter()
    for em in emails:
        try:
            parsed = parse_message(em.raw, em.from_line)
        except Exception:
            dropped[f"{em.source}:unparseable"] += 1
            continue
        text = clean_email(parsed.subject, parsed.body, max_chars=max_chars)
        body_only = text.split("\n\n", 1)[1] if text.startswith("Subject:") and "\n\n" in text else text
        if len(body_only.strip()) < MIN_BODY_CHARS:
            dropped[f"{em.source}:empty_body"] += 1
            continue
        rid = hashlib.sha256(f"{em.origin}\n{text}".encode()).hexdigest()[:16]
        out.append(
            Record(
                id=rid,
                source=em.source,
                subset=em.subset,
                label=em.label,
                year=parsed.date.year if parsed.date else None,
                text=text,
                raw_text=_raw_view(em.raw),
                has_html=parsed.has_html,
                origin=em.origin,
            )
        )
    return out, dropped


def filter_english(records: list[Record]) -> tuple[list[Record], Counter]:
    kept, dropped = [], Counter()
    for r in records:
        if looks_english(r.text):
            kept.append(r)
        else:
            dropped[f"{r.source}:not_english"] += 1
    return kept, dropped


def _fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()


def dedupe_exact(records: list[Record]) -> tuple[list[Record], Counter]:
    """Drop exact duplicates (after cleaning, ignoring case and punctuation).

    If the same text appears with both labels, every copy is dropped, since we
    cannot tell which label is right.
    """
    by_fp: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_fp[_fingerprint(r.text)].append(r)
    kept, dropped = [], Counter()
    for group in by_fp.values():
        labels = {r.label for r in group}
        if len(labels) > 1 and labels & {0} and labels & {1}:
            dropped["label_conflict_exact"] += len(group)
            continue
        kept.append(group[0])
        if len(group) > 1:
            dropped[f"{group[0].source}:exact_duplicate"] += len(group) - 1
    return kept, dropped


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(texts: list[str], threshold: float = 0.8, chunk: int = 1000) -> np.ndarray:
    """Group texts whose word-3-gram shingle sets have cosine similarity >= threshold.

    Returns an integer group id per text. Uses binary hashed shingles and a
    chunked sparse product, so memory stays small for tens of thousands of emails.
    """
    vec = HashingVectorizer(ngram_range=(3, 3), n_features=2**20, binary=True, norm="l2", alternate_sign=False)
    X = vec.transform(texts).tocsr().astype(np.float32)
    n = X.shape[0]
    uf = _UnionFind(n)
    XT = X.T.tocsc()
    for start in range(0, n, chunk):
        block = (X[start : start + chunk] @ XT).tocoo()
        mask = (block.data >= threshold) & (block.row + start < block.col)
        for i, j in zip(block.row[mask] + start, block.col[mask]):
            uf.union(int(i), int(j))
    roots = np.array([uf.find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def cap_groups(records: list[Record], groups: np.ndarray, max_per_group: int, seed: int = 0) -> tuple[list[int], Counter]:
    """Keep at most ``max_per_group`` members of each near-duplicate group.

    Returns the indices to keep. Groups that mix labels are dropped entirely.
    """
    rng = np.random.default_rng(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[int(g)].append(i)
    keep: list[int] = []
    dropped: Counter = Counter()
    for idx in members.values():
        labels = {records[i].label for i in idx}
        if 0 in labels and 1 in labels:
            dropped["label_conflict_near_duplicate"] += len(idx)
            continue
        if len(idx) > max_per_group:
            chosen = sorted(rng.choice(idx, size=max_per_group, replace=False).tolist())
            dropped[f"{records[idx[0]].source}:near_duplicate_over_cap"] += len(idx) - max_per_group
            keep.extend(chosen)
        else:
            keep.extend(idx)
    return sorted(keep), dropped


def stratified_group_split(labels: np.ndarray, groups: np.ndarray, seed: int = 42) -> np.ndarray:
    """Assign each row to train / val / test (about 70 / 10 / 20) keeping groups together."""
    split = np.array(["train"] * len(labels), dtype=object)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(np.zeros(len(labels)), labels, groups))
    split[test_idx] = "test"
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    tr, va = next(inner.split(np.zeros(len(rest_idx)), labels[rest_idx], groups[rest_idx]))
    split[rest_idx[va]] = "val"
    return split


def time_split(
    labels: np.ndarray, years: list[int | None], groups: np.ndarray, base_split: np.ndarray, cutoff: int
) -> np.ndarray:
    """Phishing before ``cutoff`` trains, phishing from ``cutoff`` on is tested.

    The legitimate mail only covers 1999-2002, so it cannot be split by time; it
    keeps its train/val/test assignment from the stratified split. Phishing
    whose near-duplicate group already appears in the training period is
    removed from the test period, so a campaign seen in training cannot be
    "detected" again later.
    """
    split = np.array(["unused"] * len(labels), dtype=object)
    for i, (y, lab) in enumerate(zip(years, labels)):
        if lab == 0:
            split[i] = base_split[i]
        elif lab == 1 and y is not None:
            split[i] = "train" if y < cutoff else "test"
    train_groups = {int(groups[i]) for i in range(len(labels)) if split[i] in ("train", "val")}
    for i in range(len(labels)):
        if labels[i] == 1 and split[i] == "test" and int(groups[i]) in train_groups:
            split[i] = "unused"
    # carve a validation set out of the training-period phishing
    rng = np.random.default_rng(0)
    phish_train = [i for i in range(len(labels)) if labels[i] == 1 and split[i] == "train"]
    for i in rng.choice(phish_train, size=len(phish_train) // 8, replace=False):
        split[i] = "val"
    return split


def save_jsonl(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as f:
        return [Record(**json.loads(line)) for line in f]
