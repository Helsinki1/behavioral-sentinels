# Experiment 3 canary: staircase  (headroom)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.750 | 0.158 | 0.261 | 15 | 5 | 80 | 0 | 0.360 | 4 |
| K=5 | 0.821 | 0.242 | 0.374 | 23 | 5 | 72 | 0 | 0.360 | 4 |
| K=10 | 0.853 | 0.305 | 0.450 | 29 | 5 | 66 | 0 | 0.360 | 4 |
| K=inf | 0.861 | 0.326 | 0.473 | 31 | 5 | 64 | 0 | 0.360 | 4 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.821 | 0.242 | 0.374 | 23 | 5 | 72 | 0 | 0.360 | 4 |
| theta=0.67 | 0.885 | 0.242 | 0.380 | 23 | 3 | 72 | 2 | 0.340 | 2 |
| theta=0.34 | 1.000 | 0.095 | 0.173 | 9 | 0 | 86 | 5 | 0.100 | 1.500 |
| theta=0.0 | 1.000 | 0.011 | 0.021 | 1 | 0 | 94 | 5 | 0.020 | 5.500 |
