# Experiment 3 canary: confab_trap  (abstention)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 1.000 | 0.260 | 0.413 | 26 | 0 | 74 | 0 | 0.360 | 1.000 |
| K=5 | 1.000 | 0.340 | 0.507 | 34 | 0 | 66 | 0 | 0.360 | 1.000 |
| K=10 | 1.000 | 0.360 | 0.529 | 36 | 0 | 64 | 0 | 0.360 | 1.000 |
| K=inf | 1.000 | 0.360 | 0.529 | 36 | 0 | 64 | 0 | 0.360 | 1.000 |

fabricated_share_of_first_fails: 0.306
