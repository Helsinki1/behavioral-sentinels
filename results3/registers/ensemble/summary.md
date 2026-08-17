# Experiment 3 canary: ensemble  (ensemble (lag+checksum+confab))

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.957 | 0.222 | 0.361 | 22 | 1 | 77 | 0 | 0.920 | 6 |
| K=5 | 0.975 | 0.394 | 0.561 | 39 | 1 | 60 | 0 | 0.920 | 6 |
| K=10 | 0.986 | 0.697 | 0.817 | 69 | 1 | 30 | 0 | 0.920 | 6 |
| K=inf | 0.989 | 0.919 | 0.953 | 91 | 1 | 8 | 0 | 0.920 | 6 |
