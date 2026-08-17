# Experiment 3 canary: lag_span  (memory distance)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.302 | 0.098 | 0.148 | 16 | 37 | 147 | 0 | 0.810 | 8 |
| K=5 | 0.513 | 0.239 | 0.326 | 39 | 37 | 124 | 0 | 0.810 | 8 |
| K=10 | 0.675 | 0.472 | 0.556 | 77 | 37 | 86 | 0 | 0.810 | 8 |
| K=inf | 0.772 | 0.767 | 0.769 | 125 | 37 | 38 | 0 | 0.810 | 8 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.513 | 0.239 | 0.326 | 39 | 37 | 124 | 0 | 0.810 | 8 |
| theta=0.67 | 0.513 | 0.239 | 0.326 | 39 | 37 | 124 | 0 | 0.810 | 8 |
| theta=0.34 | 0.543 | 0.270 | 0.361 | 44 | 37 | 119 | 0 | 0.775 | 8.000 |
| theta=0.0 | 0.479 | 0.209 | 0.291 | 34 | 37 | 129 | 0 | 0.645 | 8.000 |
