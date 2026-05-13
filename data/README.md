# Data

No email data is committed to this repository. `scripts/download_data.py`
fetches the raw corpora into `data/raw/` and `scripts/prepare_data.py` builds
`data/processed/emails.jsonl`. Both directories are gitignored.

```
python scripts/download_data.py      # ~610 MB, most of it the Enron tarball
python scripts/prepare_data.py       # ~1 minute, ~0.6 GB RAM
```

`data/checksums.sha256` lists the SHA-256 of every raw file used for the
results in this repo (downloaded 2026-05-14). The Nazario yearly files are
still updated by their maintainer, so a later download may differ.

## Sources

| Corpus | Used as | Files | Raw messages | Terms |
|---|---|---|---|---|
| Nazario phishing corpus, https://monkey.org/~jose/phishing/ | phishing (label 1) | `phishing0.mbox` to `phishing3.mbox` (2004-2007), `phishing-2015` to `phishing-2025` | 8,038 | CC BY 4.0 (stated in the corpus `README.txt` and `LICENSE.txt`). Attribution: Jose Nazario. |
| SpamAssassin public corpus, https://spamassassin.apache.org/old/publiccorpus/ | legitimate (label 0): `easy_ham`, `easy_ham_2`, `hard_ham`; side set: `spam`, `spam_2` | `20030228_*.tar.bz2` | 4,150 ham, 1,897 spam | No formal licence. Published by the SpamAssassin project for testing spam filters; copyright stays with the original senders. The maintainers ask that the mails are never sent into a live mail system. |
| Enron email corpus, May 2015 release, https://www.cs.cmu.edu/~enron/ | legitimate (label 0) | `enron_mail_20150507.tar.gz`, 4,000 messages sampled from users' `inbox` folders (seed 42) | 4,000 sampled | No formal licence. Made public by the US Federal Energy Regulatory Commission, cleaned by the CALO project and distributed by CMU for research, with a request to respect the privacy of the people involved. |

Not used:

* `private-phishing4.mbox` in the Nazario directory. The name suggests it was not
  meant to be public in the same way as the other files, so it is left out.
* `20051114.mbox`, which overlaps in time with `phishing1.mbox`.
* Kaggle/Hugging Face "phishing email" CSVs (for example
  `zefang-liu/phishing-email-dataset`, LGPL-3.0). They are convenient, but they
  are re-packagings of the same public corpora with the headers and provenance
  stripped, so the leakage problems below cannot be inspected or fixed.

Because none of this data is redistributed here, the licences above only
govern your own download. The report quotes a handful of short, trimmed
excerpts with addresses and names removed.

## Pipeline

`scripts/prepare_data.py` runs these steps and writes every count to
`results/data_stats.json`.

1. **Parse** (`src/phishdet/parsing.py`). Decode MIME and charsets, keep the
   Subject and the body. Plain text is preferred; HTML-only messages are
   converted to text with link targets written inline. Attachments are ignored.
   The delivery date comes from the mbox envelope line where there is one (the
   Date header in phishing is often forged), otherwise the Date header.
2. **Clean** (`src/phishdet/clean.py`), described under Leakage below.
3. **Drop** messages with fewer than 30 characters of body text after cleaning
   (394, mostly image-only or attachment-only phishing) and non-English messages
   (330 including spam; a simple function-word comparison against six other
   languages plus a non-Latin script check).
4. **Exact dedupe** on the cleaned text, ignoring case and punctuation (1,742 removed).
5. **Near-duplicate groups.** Word 3-gram shingles, cosine similarity >= 0.8,
   union-find into groups. 10,672 messages had no near-duplicate; the largest
   group (one PayPal phishing campaign) had 69 members. Each group is capped at 3
   members (1,156 removed, 1,049 of them phishing) so a single campaign cannot
   dominate training or test, and groups with mixed labels would be dropped
   (there were none).
6. **Split.**
   * *Stratified group split*: about 70 / 10 / 20 train / validation / test,
     stratified by label, with every near-duplicate group kept on one side.
   * *Time split*: phishing delivered before 2020 is used for training and
     validation, phishing from 2020 onwards for testing. Test phishing whose
     near-duplicate group already appears before 2020 is removed (15 messages).
     The legitimate mail only covers 2001-2002, so it keeps its assignment from
     the stratified split. This tests whether a model trained on older phishing
     still catches newer phishing; it is not a full temporal evaluation.

Final dataset:

| | legitimate | phishing |
|---|---|---|
| total | 7,575 (SpamAssassin 4,022, Enron 3,553) | 5,192 |
| stratified train / val / test | 5,302 / 758 / 1,515 | 3,634 / 519 / 1,039 |
| time split train / val / test | 5,302 / 758 / 1,515 | 3,235 / 462 / 1,480 |

1,696 SpamAssassin spam messages are kept aside (never trained on) to see what
the phishing models make of ordinary spam.

## Leakage

The two classes come from different corpora, collected by different people,
20 years apart. Many surface details therefore identify the *corpus*, and a
classifier will happily learn those instead of anything about phishing. The
leakage check in `scripts/leakage_check.py` trains the same baseline on the raw
messages (headers included) and on the cleaned text; the report shows the top
features of each. What the cleaning does, and why:

| Artefact | Why it leaks | What we do |
|---|---|---|
| All headers except Subject (Received, Message-ID, X-Mailer, List-*, Date, X-Folder ...) | Mail servers, list software and dates are corpus-specific: `spamassassin.taint.org`, `X-FileName: ... .nsf` in Enron, 2002 vs 2023 dates. | Dropped entirely. The model sees subject and body only. |
| Recipient identity: `jose`, `monkey.org` (Nazario's own inbox), `domain.com` / `username@domain.com` (Nazario's anonymisation), `enron`, `taint.org`, `jmason.org` | Phishing often names the target's organisation ("your monkey.org mailbox is full"). Learning the literal names is pure source identification. | Replaced with `ORGNAME` / `RECIPIENT`. At deployment you would map your own domain to `ORGNAME`, so the pattern stays learnable. |
| Email addresses | Obfuscated addresses differ per corpus. | Replaced with `EMAIL`. |
| URLs | Full URLs are long, unique and noisy. | Replaced with `url:<host>`, keeping the host because the host is a real signal (`url:ip-address` for raw IPs). |
| Years and numeric dates | 2001-2002 for legitimate mail vs 2004-2025 for phishing. | Replaced with `YEAR` / `DATE`. |
| Mailing-list footers, `[ILUG]`-style subject tags, `-- ` signatures | Most SpamAssassin ham is mailing-list traffic. | Short trailing footers and signatures are cut; single-word list tags are removed from subjects. |
| Quoted replies (`>` lines, "On ... wrote:"), forwarded headers, Lotus Notes addresses (`Name/HOU/ECT@ECT`) | Ham is full of reply chains and Enron's Notes formatting; phishing almost never is. | Removed. |
| HTML | 95.5% of phishing has an HTML part vs 2.3% of legitimate mail. | Bodies are converted to text the same way for both classes, and link targets are written bare, not in a format that marks the text as converted HTML. The share itself is not given to the models. |
| Language | All legitimate mail is English; about 3% of the phishing is not. | Non-English messages are dropped from both classes. |
| Broken charset decoding | Decoding with the declared charset and `errors="replace"` left U+FFFD replacement characters in 605 phishing messages but only 47 legitimate ones, because phishing more often declares the wrong charset. | Strict decoding with UTF-8, Windows-1252 and Latin-1 fallbacks for bodies and subjects; any remaining U+FFFD is removed. |
| Tags added by the receiving mail system | Some of Nazario's mail passed through filters that rewrote the subject (`[*** POSIBLE SPAM***]`, `****SPAM(5.0)****`, `[EXTERNAL EMAIL - USE CAUTION]`) or appended "scanned by ..." footers. That is another filter's verdict, not a property of the email. | Removed from subjects and bodies. |
| Line wrapping, whitespace | Plain-text ham is wrapped at ~72 columns. | Whitespace is collapsed; the TF-IDF features cannot see line breaks. |
| Unicode tricks (zero-width spaces, homoglyphs) | Not a leak, but obfuscation. | NFKC normalisation, zero-width characters removed, Cyrillic/Greek look-alikes mapped to Latin inside mixed-script words. |

What cleaning cannot remove, and what the results should be read with:

* **Topic.** The legitimate mail is Linux and tech mailing lists, tech news
  feeds and one energy-trading company in 2001. A legitimate email about, say,
  a parcel delivery or a password reset looks nothing like the training ham and
  is more likely to be flagged. The test set shares this bias, so test scores
  are optimistic about real inboxes.
* **Era.** Even with years removed, vocabulary drifts (no "Office 365",
  "DocuSign" or "WeTransfer" in 2001 ham). Some of what the model learns is
  "2020s email vs 2001 email".
* **Formatting.** HTML-converted text is still choppier than hand-written plain
  text, which character n-grams can partly pick up.
* **Labels.** Nazario's labels are hand-assigned; the Enron inbox sample can
  contain the odd spam or newsletter, labelled legitimate.
