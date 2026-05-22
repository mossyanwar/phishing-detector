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
| MiniLM + LR | none | 0.904 | 0.911 | +0.000 |
| MiniLM + LR | homoglyphs (30% of letters) | 0.907 | 0.913 | +0.003 |
| MiniLM + LR | zero-width spaces in words | 0.904 | 0.911 | +0.000 |
| MiniLM + LR | leetspeak keywords | 0.787 | 0.830 | -0.080 |
| MiniLM + LR | URLs swapped for bit.ly | 0.900 | 0.912 | +0.002 |
| MiniLM + LR | 'RE:' added to subject | 0.905 | 0.910 | -0.001 |
| MiniLM + LR | legit text appended | 0.718 | 0.796 | -0.114 |
| MiniLM + LR | legit text prepended | 0.647 | 0.736 | -0.174 |
| MiniLM + LR | homoglyphs, no unicode normalisation | 0.757 | 0.826 | -0.085 |
| MiniLM + LR | zero-width, no unicode normalisation | 0.898 | 0.910 | -0.001 |
