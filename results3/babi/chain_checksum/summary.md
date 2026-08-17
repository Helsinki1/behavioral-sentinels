# Experiment 3 canary: chain_checksum  (reasoning composition)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 1.000 | 0.410 | 0.582 | 41 | 0 | 59 | 0 | 0.480 | 0.000 |
| K=5 | 1.000 | 0.440 | 0.611 | 44 | 0 | 56 | 0 | 0.480 | 0.000 |
| K=10 | 1.000 | 0.480 | 0.649 | 48 | 0 | 52 | 0 | 0.480 | 0.000 |
| K=inf | 1.000 | 0.480 | 0.649 | 48 | 0 | 52 | 0 | 0.480 | 0.000 |
