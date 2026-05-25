# Robustness

All 1038 phishing emails in the stratified test set, perturbed before cleaning.
Recall at each model's 1%-validation-FPR threshold.

| model | perturbation | recall | mean score | mean change |
|---|---|---|---|---|
| TF-IDF + LR | none | 0.978 | 0.917 | +0.000 |
| TF-IDF + LR | homoglyphs (30% of letters) | 0.978 | 0.917 | +0.000 |
| TF-IDF + LR | zero-width spaces in words | 0.978 | 0.917 | +0.000 |
| TF-IDF + LR | leetspeak keywords | 0.967 | 0.852 | -0.065 |
| TF-IDF + LR | URLs swapped for bit.ly | 0.975 | 0.914 | -0.003 |
| TF-IDF + LR | 'RE:' added to subject | 0.963 | 0.895 | -0.022 |
| TF-IDF + LR | legit text appended | 0.870 | 0.767 | -0.150 |
| TF-IDF + LR | legit text prepended | 0.855 | 0.755 | -0.162 |
| TF-IDF + LR | homoglyphs, no unicode normalisation | 0.957 | 0.858 | -0.059 |
| TF-IDF + LR | zero-width, no unicode normalisation | 0.968 | 0.882 | -0.035 |
| MiniLM + LR | none | 0.908 | 0.927 | +0.000 |
| MiniLM + LR | homoglyphs (30% of letters) | 0.911 | 0.930 | +0.003 |
| MiniLM + LR | zero-width spaces in words | 0.908 | 0.927 | +0.000 |
| MiniLM + LR | leetspeak keywords | 0.777 | 0.847 | -0.080 |
| MiniLM + LR | URLs swapped for bit.ly | 0.908 | 0.927 | -0.000 |
| MiniLM + LR | 'RE:' added to subject | 0.909 | 0.926 | -0.001 |
| MiniLM + LR | legit text appended | 0.716 | 0.802 | -0.125 |
| MiniLM + LR | legit text prepended | 0.623 | 0.736 | -0.191 |
| MiniLM + LR | homoglyphs, no unicode normalisation | 0.798 | 0.864 | -0.063 |
| MiniLM + LR | zero-width, no unicode normalisation | 0.901 | 0.926 | -0.001 |
