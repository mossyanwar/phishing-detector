# Top features, TF-IDF + LR (split)

250,000 features in total.

## All features

| towards phishing | weight | towards legitimate | weight |
|---|---|---|---|
| `word:your` | 3.37 | `word:subject re` | -2.49 |
| `word:account` | 3.12 | `word:re` | -1.99 |
| `word:your account` | 2.08 | `word:the` | -1.96 |
| `word:ebay` | 2.04 | `char: i ` | -1.62 |
| `word:bank` | 2.00 | `word:url` | -1.45 |
| `word:email` | 1.95 | `word:phone` | -1.43 |
| `word:dear` | 1.74 | `char: re: ` | -1.38 |
| `word:she` | 1.66 | `word:me` | -1.38 |
| `word:he` | 1.61 | `char:re: ` | -1.38 |
| `char:your` | 1.55 | `word:at` | -1.35 |
| `char:your ` | 1.55 | `char: re:` | -1.35 |
| `word:update` | 1.55 | `char:re:` | -1.34 |
| `char: your` | 1.52 | `word:url:newsisfree.com` | -1.17 |
| `char:ur ` | 1.45 | `char:the` | -1.17 |
| `char:our ` | 1.45 | `char: the` | -1.14 |
| `char: ur` | 1.42 | `char: th` | -1.14 |
| `word:notification` | 1.41 | `word:url url:newsisfree.com` | -1.10 |
| `char:ccou` | 1.39 | `word:is` | -1.05 |
| `char:accou` | 1.38 | `char:e: ` | -1.05 |
| `char:ccoun` | 1.38 | `word:my` | -1.00 |
| `char:bank` | 1.35 | `word:that` | -1.00 |
| `word:view` | 1.33 | `char:the ` | -1.00 |
| `word:security` | 1.33 | `char: the ` | -0.99 |
| `word:her` | 1.32 | `word:fw` | -0.98 |
| `char: url:` | 1.32 | `word:gas` | -0.98 |
| `char:rl:` | 1.32 | `word:list` | -0.97 |
| `char:url:` | 1.32 | `char:'s ` | -0.94 |
| `char: url` | 1.32 | `word:subject fw` | -0.93 |
| `char:url` | 1.31 | `char:...` | -0.92 |
| `word:year` | 1.30 | `char:he ` | -0.92 |

## Word n-grams only

| towards phishing | weight | towards legitimate | weight |
|---|---|---|---|
| `word:your` | 3.37 | `word:subject re` | -2.49 |
| `word:account` | 3.12 | `word:re` | -1.99 |
| `word:your account` | 2.08 | `word:the` | -1.96 |
| `word:ebay` | 2.04 | `word:url` | -1.45 |
| `word:bank` | 2.00 | `word:phone` | -1.43 |
| `word:email` | 1.95 | `word:me` | -1.38 |
| `word:dear` | 1.74 | `word:at` | -1.35 |
| `word:she` | 1.66 | `word:url:newsisfree.com` | -1.17 |
| `word:he` | 1.61 | `word:url url:newsisfree.com` | -1.10 |
| `word:update` | 1.55 | `word:is` | -1.05 |
| `word:notification` | 1.41 | `word:my` | -1.00 |
| `word:view` | 1.33 | `word:that` | -1.00 |
| `word:security` | 1.33 | `word:fw` | -0.98 |
| `word:her` | 1.32 | `word:gas` | -0.98 |
| `word:year` | 1.30 | `word:list` | -0.97 |
| `word:payment` | 1.29 | `word:subject fw` | -0.93 |
| `word:customer` | 1.27 | `word:for` | -0.92 |
| `word:kindly` | 1.19 | `word:houston` | -0.86 |
| `word:your email` | 1.14 | `word:there` | -0.85 |
| `word:this email` | 1.14 | `word:it` | -0.84 |
| `word:service` | 1.10 | `word:call` | -0.82 |
| `word:update your` | 1.10 | `word:energy` | -0.79 |
| `word:annie` | 1.09 | `word:in the` | -0.79 |
| `word:subject your` | 1.09 | `word:know` | -0.75 |
| `word:paypal` | 1.07 | `word:let me` | -0.75 |
| `word:message` | 1.05 | `word:me know` | -0.73 |
| `word:click` | 1.04 | `word:power` | -0.72 |
| `word:important` | 1.03 | `word:thanks` | -0.71 |
| `word:confirm` | 1.03 | `word:doc` | -0.70 |
| `word:fifth third` | 1.00 | `word:which` | -0.68 |
