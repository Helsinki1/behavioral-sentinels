# Experiment 3 canary: ensemble  (ensemble (lag+checksum+confab))

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.731 | 0.204 | 0.319 | 38 | 14 | 148 | 0 | 0.780 | 6.000 |
| K=5 | 0.823 | 0.349 | 0.491 | 65 | 14 | 121 | 0 | 0.780 | 6.000 |
| K=10 | 0.892 | 0.624 | 0.734 | 116 | 14 | 70 | 0 | 0.780 | 6.000 |
| K=inf | 0.910 | 0.763 | 0.830 | 142 | 14 | 44 | 0 | 0.780 | 6.000 |
