# Experiment 3 canary: sparse_recall  (unrehearsed retention)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.321 | 0.055 | 0.094 | 9 | 19 | 154 | 18 | 0.195 | 3.000 |
| K=5 | 0.457 | 0.098 | 0.162 | 16 | 19 | 147 | 18 | 0.195 | 3.000 |
| K=10 | 0.486 | 0.110 | 0.180 | 18 | 19 | 145 | 18 | 0.195 | 3.000 |
| K=inf | 0.513 | 0.123 | 0.198 | 20 | 19 | 143 | 18 | 0.195 | 3.000 |

rehearsal_rate: 0.0047
