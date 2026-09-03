# phishing-detector

A phishing email classifier trained on public corpora, a check of how much
of its score comes from differences between those corpora, and a harness for
comparing it with an LLM.

* Data: Nazario phishing corpus (2004-2025) against SpamAssassin and Enron
  legitimate mail, with the leakage between those corpora found and cleaned
  out. See [`data/README.md`](data/README.md).
* Models trained here: TF-IDF (word + character n-grams) with logistic
  regression, and frozen MiniLM sentence embeddings with logistic regression.
  Both run on a laptop CPU.
* LLM classifier: zero-shot and few-shot prompts with JSON output, behind a
  provider interface for the OpenAI and Anthropic HTTP APIs, with cost and
  latency tracking. It is tested offline with a fake provider and mocked HTTP.
  **It has not been run against a paid API**, so there are no LLM results yet.
* A CLI that scores a `.eml` file and shows which features drove the score.

The full write-up is in [`REPORT.md`](REPORT.md).

## Results

Stratified test set, split so that near-duplicate emails (campaigns) never
appear on both sides: 2,553 emails, 1,038 phishing. The threshold is set for a
1% false-positive rate on the validation set and then applied to test.

| model | ROC-AUC | PR-AUC | precision | recall | F1 | test FPR | median latency per email |
|---|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.9978 | 0.9972 | 0.981 | 0.978 | 0.979 | 1.3% | 1.6 ms |
| MiniLM embeddings + logistic regression | 0.9921 | 0.9899 | 0.975 | 0.908 | 0.940 | 1.6% | 22.9 ms |
| LLM zero-shot / few-shot (Anthropic, OpenAI) | not run yet | | | | | | |

What the numbers do and do not say:

* **Leakage is the main trap.** The same TF-IDF model trained on raw messages
  scores ROC-AUC 1.0000 by learning years, `enron`, `monkey.org` and mail-server
  names. The cleaned results above are what is left after removing those.
* **Drift.** Trained only on phishing from before 2020, TF-IDF recall on
  2020-2025 phishing falls to 92.3% (MiniLM: 78.5%).
* **Prevalence.** The test set is 41% phishing. At 1% phishing, the TF-IDF
  model's precision at this threshold would be about 43%.
* **Weak spots.** Legitimate bills and password resets get flagged; phishing
  padded with innocent text gets through (recall 0.855 when ordinary text is
  prepended).
* The corpora are old and narrow (legitimate mail is from 2001-2002), so these
  scores are an upper bound for a real inbox.

To fill the LLM rows (they run on a fixed sample of 200 test emails; the script
prints a cost estimate and refuses to call the API without `--yes`):

```
python scripts/run_llm_eval.py --provider anthropic --model claude-opus-5-5 --mode zero-shot --yes
python scripts/run_llm_eval.py --provider anthropic --model claude-opus-5-5 --mode few-shot --yes
python scripts/run_llm_eval.py --provider openai --model gpt-4o-mini --mode zero-shot --yes
python scripts/run_llm_eval.py --provider openai --model gpt-4o-mini --mode few-shot --yes
```

API keys are read from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. An offline dry
run with the fake provider: `python scripts/run_llm_eval.py --provider fake`.

## Reproducing

Python 3.12, CPU only. About 610 MB of downloads, 22 minutes for the full
pipeline on a laptop, peak RAM 2.2 GB.

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-embed.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .

python scripts/download_data.py   # raw corpora into data/raw/ (gitignored)
./scripts/reproduce.sh            # prepare data, train, evaluate, write results/ and figures
pytest
```

`scripts/reproduce.sh` lists the individual steps; each script has a `--help`.

## Scoring an email

```
$ phishdet score examples/fake_phish.eml --top 5
file:      examples/fake_phish.eml
subject:   Your mailbox will be suspended
P(phish):  1.000  ->  PHISHING  (threshold 0.439, chosen for 1% FPR on validation)
why (largest contributions to the log-odds; + pushes towards phishing):
  +0.359  word:account
  +0.332  word:your
  +0.271  word:your account
  +0.142  word:mailbox
  +0.136  word:verify
  all word n-grams +2.818, all char n-grams +6.283, intercept -1.167 (log-odds +7.934)
```

`--json` prints one JSON object per file, `--model minilm_lr` uses the
embedding model, and `-` reads a message from stdin. The two files in
`examples/` are made up for trying the CLI. The models must be trained first
(they are not committed).

## Layout

```
data/README.md            sources, licences, cleaning and leakage notes
src/phishdet/
  parsing.py, clean.py    MIME parsing and the leakage-removing text cleaning
  corpora.py, dataset.py  corpus readers, dedupe, near-duplicate groups, splits
  baseline.py             TF-IDF + logistic regression, per-feature explanations
  embedding.py            MiniLM + logistic regression
  metrics.py              threshold for a target FPR, calibration, intervals
  evaluation.py, plots.py report numbers and figures
  robustness.py           evasion perturbations
  llm/                    providers, prompts, classifier, price table
  cli.py                  phishdet score
scripts/                  download, prepare, train, evaluate, leakage check,
                          robustness, error analysis, latency, LLM eval
results/                  metrics, figures and per-email scores from the runs
tests/                    pytest suite (offline, no API calls)
```

## Data and licences

The Nazario corpus is CC BY 4.0 (Jose Nazario). The SpamAssassin corpus and
the Enron corpus have no formal licence and are published for research and
filter testing. None of the email data is redistributed here; see
`data/README.md`.
