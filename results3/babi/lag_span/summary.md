# Experiment 3 canary: lag_span  (memory distance)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 1.000 | 0.220 | 0.361 | 22 | 0 | 78 | 0 | 1.000 | 4.500 |
| K=5 | 1.000 | 0.570 | 0.726 | 57 | 0 | 43 | 0 | 1.000 | 4.500 |
| K=10 | 1.000 | 0.960 | 0.980 | 96 | 0 | 4 | 0 | 1.000 | 4.500 |
| K=inf | 1.000 | 1.000 | 1.000 | 100 | 0 | 0 | 0 | 1.000 | 4.500 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 1.000 | 0.570 | 0.726 | 57 | 0 | 43 | 0 | 1.000 | 4.500 |
| theta=0.67 | 1.000 | 0.570 | 0.726 | 57 | 0 | 43 | 0 | 1.000 | 4.500 |
| theta=0.34 | 1.000 | 0.640 | 0.780 | 64 | 0 | 36 | 0 | 0.920 | 4.000 |
| theta=0.0 | 1.000 | 0.620 | 0.765 | 62 | 0 | 38 | 0 | 0.830 | 4 |
