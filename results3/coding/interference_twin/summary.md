# Experiment 3 canary: interference_twin  (interference)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.526 | 0.115 | 0.189 | 20 | 18 | 154 | 8 | 0.360 | 4.000 |
| K=5 | 0.660 | 0.201 | 0.308 | 35 | 18 | 139 | 8 | 0.360 | 4.000 |
| K=10 | 0.719 | 0.264 | 0.387 | 46 | 18 | 128 | 8 | 0.360 | 4.000 |
| K=inf | 0.750 | 0.310 | 0.439 | 54 | 18 | 120 | 8 | 0.360 | 4.000 |
