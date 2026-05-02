"""Text normalisation that removes corpus artefacts before modelling.

The phishing and legitimate emails come from different corpora collected in
different years by different people, so a lot of surface detail tells you the
*source* of a message rather than whether it is phishing. A model that learns
"mentions Enron -> legitimate" or "has a mailing-list footer -> legitimate" scores
well on this data and is useless in a real inbox. Each step below exists to take
one of those shortcuts away; data/README.md explains them with examples.

The same function is used at training time and by the CLI at scoring time.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

# --- unicode -----------------------------------------------------------------

# Characters that render as nothing and are used to split words past filters.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎͏"), None)

# Cyrillic and Greek letters that look like Latin ones. Only applied inside
# words that also contain Latin letters, so real Russian or Greek text is left alone.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w",
    "г": "r", "п": "n", "к": "k", "м": "m", "т": "t", "ь": "b",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "κ": "k", "τ": "t", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLES)
_WORD = re.compile(r"\w+")
_LATIN = re.compile(r"[A-Za-z]")
_NON_LATIN_LOOKALIKE = re.compile("[" + "".join(_CONFUSABLES) + "]")


def _fix_mixed_script(match: re.Match[str]) -> str:
    word = match.group(0)
    if _LATIN.search(word) and _NON_LATIN_LOOKALIKE.search(word):
        return word.translate(_CONFUSABLE_TABLE)
    return word


def normalise_unicode(text: str) -> str:
    """NFKC, drop zero-width characters, and undo homoglyphs in mixed-script words."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    return _WORD.sub(_fix_mixed_script, text)


# --- structure: replies, signatures, list footers ----------------------------

_SEPARATOR = re.compile(r"^\s*[_\-=*~]{10,}\s*$")
_SIG_DELIM = re.compile(r"^--\s*$")
_FOOTER_MARKERS = re.compile(
    r"mailing list|listinfo|mailman|sponsored by|yahoo! groups|list maintainer|"
    r"to unsubscribe|unsubscribe from this list",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"^\s*(>|\|>)")
_ATTRIBUTION = re.compile(r"^.{0,120}\b(wrote|writes|said)\s*:\s*$", re.IGNORECASE)
_ORIGINAL_MSG = re.compile(r"^\s*-*\s*(original message|forwarded message|forwarded by)\b.*$", re.IGNORECASE)
_EMBEDDED_HEADER = re.compile(r"^\s*(from|to|cc|bcc|sent|date|subject|reply-to)\s*:", re.IGNORECASE)
# Lotus Notes style addresses common in Enron mail, e.g. "Kay Mann/Corp/Enron@ENRON"
_NOTES_ADDRESS = re.compile(r"\b[A-Z][\w.\-' ]{0,40}/[A-Za-z]{2,12}/[A-Za-z]{2,12}(?:@[A-Za-z]{2,12})?")
_MAX_TAIL_LINES = 15


def _cut_tail(lines: list[str]) -> list[str]:
    """Drop a trailing signature or mailing-list footer if it is short."""
    for i, line in enumerate(lines):
        tail = lines[i + 1 :]
        if len(tail) > _MAX_TAIL_LINES or i == 0:
            continue
        if _SIG_DELIM.match(line):
            return lines[:i]
        if _SEPARATOR.match(line) and _FOOTER_MARKERS.search("\n".join(tail)):
            return lines[:i]
    return lines


def strip_structure(body: str) -> str:
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    lines = _cut_tail(lines)
    kept = []
    for line in lines:
        if _QUOTED.match(line) or _ATTRIBUTION.match(line) or _ORIGINAL_MSG.match(line):
            continue
        if _EMBEDDED_HEADER.match(line):
            continue
        if _FOOTER_MARKERS.search(line) and re.search(r"https?://|@", line):
            # single footer lines such as "http://xent.com/mailman/listinfo/fork"
            continue
        kept.append(line)
    return "\n".join(kept)


# --- tokens that identify the corpus or the recipient --------------------------

_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_IP_HOST = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_YEAR = re.compile(r"\b(?:19[89]\d|20[0-3]\d)\b")
_NUMERIC_DATE = re.compile(r"\b\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")

# The recipients' own organisations. In deployment you would put your own
# domain here, so "your ORGNAME mailbox is full" is still learnable, but the
# model cannot learn "mentions Enron" or "mentions monkey.org".
_ORG_TOKENS = re.compile(
    r"spamassassin\.taint\.org|taint\.org|jmason\.org|monkey\.org|\bdomain\.com|enron\.com|\benron\b",
    re.IGNORECASE,
)
# The owner of the phishing corpus, who is the recipient of most of it.
_RECIPIENT_TOKENS = re.compile(r"(?<!san )\b(?:jose|nazario)\b", re.IGNORECASE)


def _url_repl(match: re.Match[str]) -> str:
    url = match.group(0).rstrip(".,;:!?")
    if url.lower().startswith("www."):
        url = "http://" + url
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    if _IP_HOST.match(host):
        host = "ip-address"
    return f" url:{host or 'unknown'} "


def replace_tokens(text: str) -> str:
    text = _URL.sub(_url_repl, text)
    text = _EMAIL.sub(" EMAIL ", text)
    text = _NOTES_ADDRESS.sub(" NAME ", text)
    text = _ORG_TOKENS.sub("ORGNAME", text)
    text = _RECIPIENT_TOKENS.sub("RECIPIENT", text)
    text = _NUMERIC_DATE.sub("DATE", text)
    text = _YEAR.sub("YEAR", text)
    return text


# --- whitespace ----------------------------------------------------------------

def tidy_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\f\v\xa0]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- public API ------------------------------------------------------------------

_LIST_TAG = re.compile(r"^((?:re|fw|fwd)\s*:\s*)*\[[^\]\s]{1,30}\]\s*", re.IGNORECASE)


def clean_subject(subject: str, unicode_normalise: bool = True) -> str:
    if unicode_normalise:
        subject = normalise_unicode(subject)
    # single-word mailing list tags like "[ILUG]" or "Re: [SAtalk]"
    m = _LIST_TAG.match(subject)
    if m:
        subject = (m.group(1) or "") + subject[m.end():]
    return tidy_whitespace(replace_tokens(subject)).replace("\n", " ")


def clean_body(body: str, unicode_normalise: bool = True) -> str:
    if unicode_normalise:
        body = normalise_unicode(body)
    body = strip_structure(body)
    body = replace_tokens(body)
    return tidy_whitespace(body)


def clean_email(subject: str, body: str, max_chars: int = 4000, unicode_normalise: bool = True) -> str:
    """Return the model input text: cleaned subject and body, truncated."""
    s = clean_subject(subject, unicode_normalise)
    b = clean_body(body, unicode_normalise)
    text = f"Subject: {s}\n\n{b}" if s else b
    return text[:max_chars]
