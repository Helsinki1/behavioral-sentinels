# Experiment 3 canary: staircase  (headroom)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.531 | 0.201 | 0.292 | 34 | 30 | 135 | 1 | 0.595 | 4 |
| K=5 | 0.634 | 0.308 | 0.414 | 52 | 30 | 117 | 1 | 0.595 | 4 |
| K=10 | 0.720 | 0.456 | 0.558 | 77 | 30 | 92 | 1 | 0.595 | 4 |
| K=inf | 0.748 | 0.527 | 0.618 | 89 | 30 | 80 | 1 | 0.595 | 4 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.634 | 0.308 | 0.414 | 52 | 30 | 117 | 1 | 0.595 | 4 |
| theta=0.67 | 0.641 | 0.296 | 0.405 | 50 | 28 | 119 | 3 | 0.575 | 4 |
| theta=0.34 | 0.683 | 0.166 | 0.267 | 28 | 13 | 141 | 18 | 0.260 | 3 |
| theta=0.0 | 0.600 | 0.035 | 0.067 | 6 | 4 | 163 | 27 | 0.060 | 1.500 |
