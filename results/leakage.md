# Leakage check

Same TF-IDF + LR pipeline (C=3), same stratified group split, three views of the text.
Threshold set for 1% FPR on validation.

| view | ROC-AUC | PR-AUC | recall @thr | test FPR @thr |
|---|---|---|---|---|
| raw | 1.0000 | 1.0000 | 1.0000 | 0.73% |
| parsed | 0.9995 | 0.9993 | 0.9884 | 1.19% |
| cleaned | 0.9984 | 0.9980 | 0.9817 | 1.32% |

## Top features, raw

| towards phishing | weight | towards legitimate | weight |
|---|---|---|---|
| `word:example com` | 2.29 | `word:2002` | -2.18 |
| `word:example` | 1.94 | `word:enron` | -1.83 |
| `word:status` | 1.67 | `word:enron com` | -1.37 |
| `word:login` | 1.67 | `word:2001` | -1.06 |
| `word:monkey org` | 1.63 | `char:2002` | -1.05 |
| `word:monkey` | 1.63 | `word:pst` | -1.03 |
| `word:jose` | 1.50 | `word:cn` | -1.01 |
| `word:jose monkey` | 1.36 | `char:2002 ` | -0.99 |
| `word:mail example` | 1.35 | `char: 2002` | -0.98 |
| `word:2006` | 1.30 | `word:taint` | -0.98 |
| `word:domain` | 1.22 | `word:taint org` | -0.98 |
| `word:2005` | 1.22 | `char:002 ` | -0.96 |
| `word:login example` | 1.20 | `word:spamassassin taint` | -0.95 |
| `word:domain com` | 1.15 | `word:localhost` | -0.95 |
| `word:user` | 1.14 | `word:spamassassin` | -0.94 |
| `word:3d` | 1.10 | `char:enron` | -0.91 |
| `word:hostedemail` | 1.09 | `char:nron` | -0.90 |
| `word:hostedemail com` | 1.09 | `word:aug 2002` | -0.89 |
| `word:mail1` | 1.08 | `word:sep 2002` | -0.88 |
| `word:user example` | 1.06 | `word:non privileged` | -0.87 |

## Top features, parsed

| towards phishing | weight | towards legitimate | weight |
|---|---|---|---|
| `word:account` | 3.46 | `word:enron` | -3.13 |
| `word:your` | 3.35 | `word:2002` | -2.95 |
| `word:https` | 3.20 | `word:2001` | -2.66 |
| `word:ebay` | 2.68 | `word:the` | -2.05 |
| `word:jose` | 2.63 | `word:713` | -1.83 |
| `word:your account` | 2.57 | `char: > ` | -1.71 |
| `word:monkey` | 2.52 | `char:nron` | -1.66 |
| `word:monkey org` | 2.41 | `char:enron` | -1.66 |
| `word:bank` | 2.38 | `char: i ` | -1.63 |
| `word:jose monkey` | 2.05 | `char:enro` | -1.56 |
| `word:update` | 1.95 | `word:at` | -1.56 |
| `word:dear` | 1.91 | `char:nro` | -1.54 |
| `word:he` | 1.81 | `char:enr` | -1.54 |
| `word:she` | 1.77 | `word:enron com` | -1.53 |
| `char:bank` | 1.72 | `word:list` | -1.53 |
| `word:customer` | 1.67 | `word:me` | -1.51 |
| `word:2025` | 1.63 | `char:2002` | -1.50 |
| `char:ban` | 1.60 | `char:002` | -1.46 |
| `word:email` | 1.58 | `word:01` | -1.43 |
| `word:payment` | 1.58 | `char:ron` | -1.42 |

## Top features, cleaned

| towards phishing | weight | towards legitimate | weight |
|---|---|---|---|
| `word:your` | 4.22 | `word:subject re` | -3.32 |
| `word:account` | 3.89 | `word:the` | -2.75 |
| `word:ebay` | 2.88 | `word:re` | -2.60 |
| `word:your account` | 2.69 | `char: i ` | -2.23 |
| `word:bank` | 2.53 | `word:url` | -1.97 |
| `word:email` | 2.43 | `word:at` | -1.93 |
| `word:she` | 2.29 | `char:re: ` | -1.89 |
| `word:dear` | 2.25 | `word:phone` | -1.89 |
| `word:he` | 2.25 | `char: re: ` | -1.88 |
| `word:update` | 2.06 | `char: re:` | -1.83 |
| `char: ur` | 1.95 | `char:re:` | -1.83 |
| `word:notification` | 1.89 | `word:me` | -1.78 |
| `char:your ` | 1.89 | `char:the` | -1.65 |
| `char:your` | 1.89 | `char: th` | -1.63 |
| `char: your` | 1.84 | `char: the` | -1.58 |
| `word:view` | 1.84 | `word:url:newsisfree.com` | -1.56 |
| `word:her` | 1.83 | `word:url url:newsisfree.com` | -1.48 |
| `char: url:` | 1.79 | `char:e: ` | -1.45 |
| `char:rl:` | 1.79 | `word:is` | -1.44 |
| `char:url:` | 1.79 | `char:the ` | -1.40 |
