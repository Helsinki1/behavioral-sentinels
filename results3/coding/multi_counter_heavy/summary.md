# Experiment 3 canary: multi_counter_heavy  (memory breadth (heavy))

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.323 | 0.065 | 0.107 | 10 | 21 | 145 | 24 | 0.225 | 3.000 |
| K=5 | 0.447 | 0.110 | 0.176 | 17 | 21 | 138 | 24 | 0.225 | 3.000 |
| K=10 | 0.500 | 0.136 | 0.213 | 21 | 21 | 134 | 24 | 0.225 | 3.000 |
| K=inf | 0.533 | 0.155 | 0.240 | 24 | 21 | 131 | 24 | 0.225 | 3.000 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.447 | 0.110 | 0.176 | 17 | 21 | 138 | 24 | 0.225 | 3.000 |
| theta=0.67 | 0.500 | 0.026 | 0.049 | 4 | 4 | 151 | 41 | 0.050 | 3.500 |
| theta=0.34 | 0.600 | 0.019 | 0.037 | 3 | 2 | 152 | 43 | 0.030 | 3.500 |
| theta=0.0 | 1.000 | 0.006 | 0.013 | 1 | 0 | 154 | 45 | 0.005 | 2 |
