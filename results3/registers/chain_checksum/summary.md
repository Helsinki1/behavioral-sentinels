# Experiment 3 canary: chain_checksum  (reasoning composition)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.714 | 0.112 | 0.194 | 10 | 4 | 79 | 7 | 0.380 | 4.500 |
| K=5 | 0.846 | 0.247 | 0.383 | 22 | 4 | 67 | 7 | 0.380 | 4.500 |
| K=10 | 0.882 | 0.337 | 0.488 | 30 | 4 | 59 | 7 | 0.380 | 4.500 |
| K=inf | 0.895 | 0.382 | 0.535 | 34 | 4 | 55 | 7 | 0.380 | 4.500 |
