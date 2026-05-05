"""Readers for the three raw corpora. Each yields RawEmail records."""

from __future__ import annotations

import mailbox
import random
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PHISHING = 1
LEGIT = 0
SPAM = -1  # SpamAssassin spam: kept aside for a side analysis, never trained on


@dataclass
class RawEmail:
    source: str  # nazario | spamassassin | enron
    subset: str  # e.g. phishing-2023, easy_ham, inbox
    label: int
    raw: bytes
    origin: str  # file and key, so any record can be traced back to the raw data
    from_line: str | None = None


def iter_nazario(directory: Path) -> Iterator[RawEmail]:
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or path.suffix in {".part", ".sha256"}:
            continue
        box = mailbox.mbox(str(path), create=False)
        try:
            for key in box.iterkeys():
                raw = box.get_bytes(key)
                from_line = box.get_message(key).get_from()
                yield RawEmail("nazario", path.name.removesuffix(".mbox"), PHISHING, raw, f"{path.name}#{key}", from_line)
        finally:
            box.close()


def iter_spamassassin(extracted: Path, subsets: dict[str, int]) -> Iterator[RawEmail]:
    """``subsets`` maps a folder name (easy_ham, spam_2, ...) to its label."""
    for subset, label in subsets.items():
        folder = extracted / subset
        for path in sorted(folder.iterdir()):
            if path.name == "cmds" or not path.is_file():
                continue
            yield RawEmail("spamassassin", subset, label, path.read_bytes(), f"{subset}/{path.name}")


def iter_enron(tarball: Path, n: int, seed: int = 0, folders: tuple[str, ...] = ("inbox",)) -> Iterator[RawEmail]:
    """Reservoir-sample ``n`` messages from the given mailbox folders of the CMU tarball.

    The tarball is streamed once; only sampled messages are kept in memory. The
    member order inside the tarball is fixed, so the sample is reproducible for a
    given seed.
    """
    rng = random.Random(seed)
    reservoir: list[tuple[str, bytes]] = []
    seen = 0
    with tarfile.open(tarball, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            # maildir/<user>/<folder>/.../<n>.
            if len(parts) < 4 or parts[0] != "maildir" or parts[2] not in folders:
                continue
            if len(reservoir) < n:
                fh = tar.extractfile(member)
                reservoir.append((member.name, fh.read() if fh else b""))
            else:
                j = rng.randint(0, seen)
                if j < n:
                    fh = tar.extractfile(member)
                    reservoir[j] = (member.name, fh.read() if fh else b"")
            seen += 1
    for name, raw in sorted(reservoir):
        yield RawEmail("enron", name.split("/")[2], LEGIT, raw, name)
