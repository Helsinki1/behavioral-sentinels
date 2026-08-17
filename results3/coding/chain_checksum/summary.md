# Experiment 3 canary: chain_checksum  (reasoning composition)

## model: gpt-oss-20b, n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.442 | 0.115 | 0.183 | 19 | 24 | 146 | 11 | 0.500 | 5.500 |
| K=5 | 0.613 | 0.230 | 0.335 | 38 | 24 | 127 | 11 | 0.500 | 5.500 |
| K=10 | 0.727 | 0.388 | 0.506 | 64 | 24 | 101 | 11 | 0.500 | 5.500 |
| K=inf | 0.760 | 0.461 | 0.574 | 76 | 24 | 89 | 11 | 0.500 | 5.500 |
