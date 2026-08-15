# Traditional signal: random compaction (uniform random firing turn)

## model: gpt-4o-mini (proprietary), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.925 | 0.125 | 0.220 | 24.700 | 2.000 | 173.300 | 0.000 | 1.000 | 4.000 |
| K=5 | 0.957 | 0.227 | 0.367 | 44.920 | 2.000 | 153.080 | 0.000 | 1.000 | 4.000 |
| K=10 | 0.969 | 0.317 | 0.478 | 62.850 | 2.000 | 135.150 | 0.000 | 1.000 | 4.000 |
| K=inf | 0.972 | 0.346 | 0.510 | 68.430 | 2.000 | 129.570 | 0.000 | 1.000 | 4.000 |

# Traditional signal: random compaction (uniform random firing turn)

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.260 | 0.120 | 0.164 | 17.910 | 51.000 | 131.090 | 0.000 | 1.000 | 8 |
| K=5 | 0.401 | 0.229 | 0.291 | 34.120 | 51.000 | 114.880 | 0.000 | 1.000 | 8 |
| K=10 | 0.531 | 0.388 | 0.448 | 57.730 | 51.000 | 91.270 | 0.000 | 1.000 | 8 |
| K=inf | 0.637 | 0.601 | 0.619 | 89.550 | 51.000 | 59.450 | 0.000 | 1.000 | 8 |

