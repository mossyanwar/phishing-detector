"""Simple evasion tricks applied to phishing text before it goes through cleaning.

Each perturbation takes the *uncleaned* subject and body, the way an attacker
would write them, so the cleaning step's defences (Unicode normalisation,
URL-to-host tokens) are part of what is being tested.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

_LATIN_TO_CYRILLIC = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445", "y": "\u0443", "i": "\u0456"}
_LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "5"}
_KEYWORDS = re.compile(
    r"\b(account|password|verify|verification|login|log in|sign in|bank|paypal|ebay|update|security|"
    r"confirm|suspend\w*|urgent|click|credit|card|mailbox|webmail|invoice|payment)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'()\[\]{}]+", re.IGNORECASE)


def homoglyphs(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """Swap a share of a, e, o, p, c, x, y, i for Cyrillic look-alikes."""
    return "".join(_LATIN_TO_CYRILLIC[c] if c in _LATIN_TO_CYRILLIC and rng.random() < rate else c for c in text)


def zero_width(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """Insert a zero-width space in the middle of a share of words."""
    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        if len(w) >= 4 and rng.random() < rate:
            k = len(w) // 2
            return w[:k] + "\u200b" + w[k:]
        return w
    return re.sub(r"\w+", repl, text)


def leetspeak(text: str, rng: random.Random) -> str:
    """Rewrite typical phishing keywords: account -> @cc0unt."""
    return _KEYWORDS.sub(lambda m: "".join(_LEET.get(c.lower(), c) for c in m.group(0)), text)


def shorten_urls(text: str, rng: random.Random) -> str:
    """Replace every URL with a random bit.ly link, hiding the real host."""
    def repl(_: re.Match[str]) -> str:
        return "https://bit.ly/" + "".join(rng.choice("abcdefghijkmnpqrstuvwxyz0123456789") for _ in range(7))
    return _URL.sub(repl, text)


def reply_prefix(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else "RE: " + subject


Perturb = Callable[[str, str, random.Random], tuple[str, str]]


def make_perturbations(padding_texts: list[str]) -> dict[str, Perturb]:
    """Return name -> f(subject, body, rng) -> (subject, body)."""

    def pad(subject: str, body: str, rng: random.Random) -> tuple[str, str]:
        # "good word" attack: append a chunk of ordinary legitimate mail
        return subject, body + "\n\n" + rng.choice(padding_texts)

    def pad_front(subject: str, body: str, rng: random.Random) -> tuple[str, str]:
        return subject, rng.choice(padding_texts) + "\n\n" + body

    return {
        "none": lambda s, b, r: (s, b),
        "homoglyphs (30% of letters)": lambda s, b, r: (homoglyphs(s, r), homoglyphs(b, r)),
        "zero-width spaces in words": lambda s, b, r: (zero_width(s, r), zero_width(b, r)),
        "leetspeak keywords": lambda s, b, r: (leetspeak(s, r), leetspeak(b, r)),
        "URLs swapped for bit.ly": lambda s, b, r: (s, shorten_urls(b, r)),
        "'RE:' added to subject": lambda s, b, r: (reply_prefix(s), b),
        "legit text appended": pad,
        "legit text prepended": pad_front,
    }


def split_parsed(parsed_text: str) -> tuple[str, str]:
    """Inverse of the "Subject: ...\\n\\nbody" format stored in Record.parsed_text."""
    if parsed_text.startswith("Subject: "):
        head, _, body = parsed_text.partition("\n\n")
        return head[len("Subject: "):], body
    return "", parsed_text
