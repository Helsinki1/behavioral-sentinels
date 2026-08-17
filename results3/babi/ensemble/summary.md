# Experiment 3 canary: ensemble  (ensemble (lag+checksum+confab))

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.969 | 0.313 | 0.473 | 31 | 1 | 68 | 0 | 0.940 | 5 |
| K=5 | 0.983 | 0.596 | 0.742 | 59 | 1 | 40 | 0 | 0.940 | 5 |
| K=10 | 0.989 | 0.889 | 0.936 | 88 | 1 | 11 | 0 | 0.940 | 5 |
| K=inf | 0.989 | 0.939 | 0.964 | 93 | 1 | 6 | 0 | 0.940 | 5 |
