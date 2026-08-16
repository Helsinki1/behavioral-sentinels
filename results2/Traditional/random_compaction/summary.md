# Traditional signal: random compaction (uniform random firing turn)

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.319 | 0.140 | 0.195 | 21.550 | 46.000 | 132.450 | 0.000 | 1.000 | 6 |
| K=5 | 0.464 | 0.258 | 0.332 | 39.740 | 46.000 | 114.260 | 0.000 | 1.000 | 6 |
| K=10 | 0.573 | 0.401 | 0.471 | 61.680 | 46.000 | 92.320 | 0.000 | 1.000 | 6 |
| K=inf | 0.631 | 0.512 | 0.565 | 78.820 | 46.000 | 75.180 | 0.000 | 1.000 | 6 |

