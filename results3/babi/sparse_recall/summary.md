# Experiment 3 canary: sparse_recall  (unrehearsed retention)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 1.000 | 0.160 | 0.276 | 16 | 0 | 84 | 0 | 0.220 | 1.000 |
| K=5 | 1.000 | 0.210 | 0.347 | 21 | 0 | 79 | 0 | 0.220 | 1.000 |
| K=10 | 1.000 | 0.220 | 0.361 | 22 | 0 | 78 | 0 | 0.220 | 1.000 |
| K=inf | 1.000 | 0.220 | 0.361 | 22 | 0 | 78 | 0 | 0.220 | 1.000 |

rehearsal_rate: 0.0
