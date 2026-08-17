# Experiment 3 canary: multi_counter  (memory breadth)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 1.000 | 0.090 | 0.165 | 9 | 0 | 91 | 0 | 0.230 | 4 |
| K=5 | 1.000 | 0.170 | 0.291 | 17 | 0 | 83 | 0 | 0.230 | 4 |
| K=10 | 1.000 | 0.220 | 0.361 | 22 | 0 | 78 | 0 | 0.230 | 4 |
| K=inf | 1.000 | 0.230 | 0.374 | 23 | 0 | 77 | 0 | 0.230 | 4 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 1.000 | 0.170 | 0.291 | 17 | 0 | 83 | 0 | 0.230 | 4 |
| theta=0.67 | 1.000 | 0.170 | 0.291 | 17 | 0 | 83 | 0 | 0.230 | 4 |
| theta=0.34 | 1.000 | 0.080 | 0.148 | 8 | 0 | 92 | 0 | 0.120 | 4.500 |
| theta=0.0 | 1.000 | 0.010 | 0.020 | 1 | 0 | 99 | 0 | 0.020 | 7.500 |
