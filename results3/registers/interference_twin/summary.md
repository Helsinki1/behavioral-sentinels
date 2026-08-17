# Experiment 3 canary: interference_twin  (interference)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.421 | 0.107 | 0.170 | 8 | 11 | 67 | 14 | 0.470 | 6.500 |
| K=5 | 0.607 | 0.227 | 0.330 | 17 | 11 | 58 | 14 | 0.470 | 6.500 |
| K=10 | 0.718 | 0.373 | 0.491 | 28 | 11 | 47 | 14 | 0.470 | 6.500 |
| K=inf | 0.766 | 0.480 | 0.590 | 36 | 11 | 39 | 14 | 0.470 | 6.500 |
