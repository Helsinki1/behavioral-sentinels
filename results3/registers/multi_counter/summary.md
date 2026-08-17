# Experiment 3 canary: multi_counter  (memory breadth)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.571 | 0.145 | 0.231 | 12 | 9 | 71 | 8 | 0.410 | 4.500 |
| K=5 | 0.667 | 0.217 | 0.327 | 18 | 9 | 65 | 8 | 0.410 | 4.500 |
| K=10 | 0.769 | 0.361 | 0.492 | 30 | 9 | 53 | 8 | 0.410 | 4.500 |
| K=inf | 0.780 | 0.386 | 0.516 | 32 | 9 | 51 | 8 | 0.410 | 4.500 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.667 | 0.217 | 0.327 | 18 | 9 | 65 | 8 | 0.410 | 4.500 |
| theta=0.67 | 0.667 | 0.217 | 0.327 | 18 | 9 | 65 | 8 | 0.410 | 4.500 |
| theta=0.34 | 0.375 | 0.036 | 0.066 | 3 | 5 | 80 | 12 | 0.140 | 8 |
| theta=0.0 | 1.000 | 0.012 | 0.024 | 1 | 0 | 82 | 17 | 0.060 | 8.500 |
