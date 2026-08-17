# Experiment 3 canary: sparse_recall  (unrehearsed retention)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.556 | 0.059 | 0.107 | 5 | 4 | 79 | 12 | 0.120 | 1.000 |
| K=5 | 0.600 | 0.071 | 0.128 | 6 | 4 | 78 | 12 | 0.120 | 1.000 |
| K=10 | 0.667 | 0.095 | 0.167 | 8 | 4 | 76 | 12 | 0.120 | 1.000 |
| K=inf | 0.667 | 0.095 | 0.167 | 8 | 4 | 76 | 12 | 0.120 | 1.000 |

rehearsal_rate: 0.002
