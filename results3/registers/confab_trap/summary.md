# Experiment 3 canary: confab_trap  (abstention)

## model: gpt-oss-20b, n=100 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.667 | 0.087 | 0.154 | 8 | 4 | 84 | 4 | 0.230 | 4 |
| K=5 | 0.778 | 0.152 | 0.255 | 14 | 4 | 78 | 4 | 0.230 | 4 |
| K=10 | 0.818 | 0.196 | 0.316 | 18 | 4 | 74 | 4 | 0.230 | 4 |
| K=inf | 0.826 | 0.206 | 0.330 | 19 | 4 | 73 | 4 | 0.230 | 4 |

fabricated_share_of_first_fails: 0.783
