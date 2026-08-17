# Experiment 3 canary: confab_trap  (abstention)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.485 | 0.092 | 0.155 | 16 | 17 | 158 | 9 | 0.370 | 4 |
| K=5 | 0.673 | 0.201 | 0.310 | 35 | 17 | 139 | 9 | 0.370 | 4 |
| K=10 | 0.734 | 0.270 | 0.395 | 47 | 17 | 127 | 9 | 0.370 | 4 |
| K=inf | 0.770 | 0.328 | 0.460 | 57 | 17 | 117 | 9 | 0.370 | 4 |

fabricated_share_of_first_fails: 0.662
