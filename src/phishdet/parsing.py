"""Turn raw RFC 822 messages into a subject, a plain-text body and a date.

Only the Subject header and the body are passed on to the models. Every other
header (Received, Message-ID, X-*, List-*, Date ...) is dropped because those
headers identify which corpus a message came from far more reliably than its
content does (see data/README.md, "Leakage").
"""

from __future__ import annotations

import email
import email.policy
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser


@dataclass
class ParsedEmail:
    subject: str
    body: str
    has_html: bool
    date: datetime | None


_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4",
    "h5", "h6", "blockquote", "hr", "title", "center", "section", "article",
}
_SKIP_TAGS = {"script", "style", "head", "noscript", "xml"}


class _HTMLToText(HTMLParser):
    """Small HTML-to-text converter.

    Link targets are kept next to the anchor text, e.g. ``Sign in http://x.example/login``,
    because the mismatch between what a link says and where it goes is one of
    the more useful phishing signals. They are written bare, the way URLs appear
    in plain-text mail, so the converted text does not carry a marker that says
    "this came from HTML" (almost all the phishing is HTML, almost none of the
    legitimate mail is).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._href_stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            self._href_stack.append(href if href and href.lower().startswith(("http", "www")) else None)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "a" and self._href_stack:
            href = self._href_stack.pop()
            if href:
                self.parts.append(f" {href} ")
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: fall back to stripping tags
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


def decode_bytes(data: bytes, charset: str | None = None) -> str:
    """Decode with the declared charset, then UTF-8, then Windows-1252, then Latin-1.

    Decoding with errors="replace" under a wrong declared charset fills the
    text with U+FFFD. That happened far more often in the phishing corpus than
    in the legitimate one, so replacement characters became a source tell.
    Trying strict decodes first avoids most of them.
    """
    for cs in (charset, "utf-8", "cp1252"):
        if not cs:
            continue
        try:
            return data.decode(cs)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("latin-1")


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    return decode_bytes(payload, part.get_content_charset())


_RAW_SUBJECT = re.compile(rb"^Subject:[ \t]*(.*(?:\r?\n[ \t].*)*)", re.MULTILINE | re.IGNORECASE)


def _subject_from_raw(raw: bytes) -> str | None:
    """Read the Subject straight from the raw header block, decoding encoded
    words and 8-bit bytes with the same fallbacks as the body."""
    head = re.split(rb"\r?\n\r?\n", raw, maxsplit=1)[0]
    m = _RAW_SUBJECT.search(head)
    if not m:
        return None
    value = re.sub(rb"\r?\n[ \t]+", b" ", m.group(1)).decode("latin-1")
    try:
        chunks = decode_header(value)
    except Exception:
        return None
    out = []
    for chunk, cs in chunks:
        if isinstance(chunk, str):
            chunk = chunk.encode("latin-1", errors="replace")
        out.append(decode_bytes(chunk, cs))
    return "".join(out)


def _header_str(msg: Message, name: str) -> str:
    value = msg.get(name)
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _parse_date(msg: Message, envelope: str | None = None) -> datetime | None:
    # The envelope date (mbox "From " line) is when the message was delivered,
    # which is what a time-based split needs. The Date header is set by the
    # sender and is often wrong or forged in phishing, so it is only a fallback.
    candidates = [envelope] if envelope else []
    candidates.append(_header_str(msg, "Date"))
    for value in candidates:
        if not value:
            continue
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            continue
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if 1990 <= dt.year <= 2030:
            return dt
    return None


def parse_message(raw: bytes | str, from_line: str | None = None) -> ParsedEmail:
    """Parse one raw message.

    ``from_line`` is the mbox "From " separator (without the leading "From ").
    When given, its delivery date is preferred over the Date header.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="replace")
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        subject = _header_str(msg, "Subject")
    except Exception:
        msg = email.message_from_bytes(raw, policy=email.policy.compat32)
        subject = _header_str(msg, "Subject")
    raw_subject = _subject_from_raw(raw)
    if raw_subject is not None:
        subject = raw_subject

    plain: list[str] = []
    html: list[str] = []
    try:
        parts = list(msg.walk()) if msg.is_multipart() else [msg]
    except Exception:
        parts = [msg]
    for part in parts:
        if part.is_multipart():
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_decode_part(part))
        elif ctype == "text/html":
            html.append(_decode_part(part))

    plain_text = "\n".join(p for p in plain if p.strip())
    has_html = bool(html)
    if plain_text.strip():
        body = plain_text
    elif html:
        body = html_to_text("\n".join(html))
    else:
        body = ""

    envelope_date = None
    if from_line:
        # "MAILER-DAEMON Fri Jun 10 12:00:00 2005" -> drop the address
        bits = from_line.split(None, 1)
        envelope_date = bits[1] if len(bits) == 2 else from_line

    return ParsedEmail(
        subject=re.sub(r"\s+", " ", subject).strip(),
        body=body,
        has_html=has_html,
        date=_parse_date(msg, envelope_date),
    )
