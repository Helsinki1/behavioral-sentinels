# Experiment 3 canary: staircase  (headroom)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 1.000 | 0.290 | 0.450 | 29 | 0 | 71 | 0 | 0.460 | 2.000 |
| K=5 | 1.000 | 0.390 | 0.561 | 39 | 0 | 61 | 0 | 0.460 | 2.000 |
| K=10 | 1.000 | 0.450 | 0.621 | 45 | 0 | 55 | 0 | 0.460 | 2.000 |
| K=inf | 1.000 | 0.460 | 0.630 | 46 | 0 | 54 | 0 | 0.460 | 2.000 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 1.000 | 0.390 | 0.561 | 39 | 0 | 61 | 0 | 0.460 | 2.000 |
| theta=0.67 | 1.000 | 0.380 | 0.551 | 38 | 0 | 62 | 0 | 0.450 | 2 |
| theta=0.34 | - | 0.000 | 0.000 | 0 | 0 | 100 | 0 | 0.000 | - |
| theta=0.0 | - | 0.000 | 0.000 | 0 | 0 | 100 | 0 | 0.000 | - |
