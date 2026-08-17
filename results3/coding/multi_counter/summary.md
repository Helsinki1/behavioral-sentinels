# Experiment 3 canary: multi_counter  (memory breadth)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.600 | 0.071 | 0.128 | 12 | 8 | 156 | 24 | 0.175 | 3 |
| K=5 | 0.704 | 0.113 | 0.195 | 19 | 8 | 149 | 24 | 0.175 | 3 |
| K=10 | 0.750 | 0.143 | 0.240 | 24 | 8 | 144 | 24 | 0.175 | 3 |
| K=inf | 0.771 | 0.161 | 0.266 | 27 | 8 | 141 | 24 | 0.175 | 3 |

score-threshold sweep at K=5 (fire when score <= theta; strict rule above is theta<1):
| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| theta=0.99 | 0.704 | 0.113 | 0.195 | 19 | 8 | 149 | 24 | 0.175 | 3 |
| theta=0.67 | 0.704 | 0.113 | 0.195 | 19 | 8 | 149 | 24 | 0.175 | 3 |
| theta=0.34 | 0.875 | 0.042 | 0.080 | 7 | 1 | 161 | 31 | 0.045 | 1.500 |
| theta=0.0 | 0.667 | 0.012 | 0.023 | 2 | 1 | 166 | 31 | 0.015 | 2.500 |
