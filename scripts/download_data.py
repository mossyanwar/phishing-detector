"""Download the raw email corpora into data/raw/.

Sources (see data/README.md for licences and notes):
  * Nazario phishing corpus  - https://monkey.org/~jose/phishing/
  * SpamAssassin public corpus (ham and spam) - https://spamassassin.apache.org/old/publiccorpus/
  * Enron email corpus (CMU release) - https://www.cs.cmu.edu/~enron/

Files that already exist are skipped. After downloading, SHA-256 checksums are
written to data/raw/checksums.sha256 so a run can be compared with the one
recorded in data/checksums.sha256.

Usage:
    python scripts/download_data.py            # everything (~560 MB, Enron is ~420 MB)
    python scripts/download_data.py --no-enron # skip the Enron tarball
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

NAZARIO_BASE = "https://monkey.org/~jose/phishing/"
# phishing0-3 cover 2004-2007, the yearly files cover 2015-2025.
# private-phishing4.mbox is left out on purpose (see data/README.md).
NAZARIO_FILES = [
    "phishing0.mbox",
    "phishing1.mbox",
    "phishing2.mbox",
    "phishing3.mbox",
] + [f"phishing-{year}" for year in range(2015, 2026)]

SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
]

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

USER_AGENT = "phishdet-research-download/0.1 (+portfolio project)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {dest.relative_to(ROOT)} (exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.rename(dest)


def safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tar archive, refusing members that escape dest or are links."""
    marker = dest / f".extracted-{archive.name}"
    if marker.exists():
        print(f"skip  extract {archive.name}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(root)) or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe member in {archive.name}: {member.name}")
        tar.extractall(dest, filter="data")
    marker.touch()
    print(f"done  extract {archive.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-enron", action="store_true", help="skip the ~420 MB Enron tarball")
    args = parser.parse_args(argv)

    for name in NAZARIO_FILES:
        fetch(NAZARIO_BASE + name, RAW / "nazario" / name)

    for name in SPAMASSASSIN_FILES:
        archive = RAW / "spamassassin" / name
        fetch(SPAMASSASSIN_BASE + name, archive)
        safe_extract_tar(archive, RAW / "spamassassin" / "extracted")

    if not args.no_enron:
        # The Enron tarball is read in place by prepare_data.py (it is large and
        # we only sample from it), so it is not extracted here.
        fetch(ENRON_URL, RAW / "enron" / Path(ENRON_URL).name)

    lines = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file() and not path.name.startswith(".") and "extracted" not in path.parts:
            lines.append(f"{sha256(path)}  {path.relative_to(RAW).as_posix()}")
    (RAW / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} checksums to data/raw/checksums.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
