# Robustness

All 1038 phishing emails in the stratified test set, perturbed before cleaning.
Recall at each model's 1%-validation-FPR threshold.

| model | perturbation | recall | mean score | mean change |
|---|---|---|---|---|
| TF-IDF + LR | none | 0.978 | 0.918 | +0.000 |
| TF-IDF + LR | homoglyphs (30% of letters) | 0.978 | 0.918 | +0.000 |
| TF-IDF + LR | zero-width spaces in words | 0.978 | 0.918 | +0.000 |
| TF-IDF + LR | leetspeak keywords | 0.968 | 0.854 | -0.064 |
| TF-IDF + LR | URLs swapped for bit.ly | 0.975 | 0.915 | -0.003 |
| TF-IDF + LR | 'RE:' added to subject | 0.963 | 0.896 | -0.022 |
| TF-IDF + LR | legit text appended | 0.871 | 0.768 | -0.150 |
| TF-IDF + LR | legit text prepended | 0.857 | 0.755 | -0.162 |
| TF-IDF + LR | homoglyphs, no unicode normalisation | 0.956 | 0.858 | -0.060 |
| TF-IDF + LR | zero-width, no unicode normalisation | 0.969 | 0.884 | -0.033 |
