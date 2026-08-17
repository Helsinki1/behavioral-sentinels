# Experiment 3 canary: lag_span  (memory distance)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.409 | 0.103 | 0.165 | 9 | 13 | 78 | 0 | 0.890 | 10.000 |
| K=5 | 0.618 | 0.241 | 0.347 | 21 | 13 | 66 | 0 | 0.890 | 10.000 |
| K=10 | 0.772 | 0.506 | 0.611 | 44 | 13 | 43 | 0 | 0.890 | 10.000 |
| K=inf | 0.854 | 0.874 | 0.864 | 76 | 13 | 11 | 0 | 0.890 | 10.000 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.618 | 0.241 | 0.347 | 21 | 13 | 66 | 0 | 0.890 | 10.000 |
| theta=0.67 | 0.618 | 0.241 | 0.347 | 21 | 13 | 66 | 0 | 0.890 | 10.000 |
| theta=0.34 | 0.683 | 0.322 | 0.438 | 28 | 13 | 59 | 0 | 0.800 | 7 |
| theta=0.0 | 0.655 | 0.218 | 0.328 | 19 | 10 | 68 | 3 | 0.500 | 6.500 |
