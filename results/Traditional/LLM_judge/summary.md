# Traditional signal: LLM judge (gpt-4o-mini, last-8-turn window)

## model: gpt-4o-mini (proprietary), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.955 | 0.212 | 0.347 | 42 | 2 | 156 | 0 | 0.985 | 6 |
| K=5 | 0.979 | 0.470 | 0.635 | 93 | 2 | 105 | 0 | 0.985 | 6 |
| K=10 | 0.988 | 0.859 | 0.919 | 170 | 2 | 28 | 0 | 0.985 | 6 |
| K=inf | 0.990 | 0.985 | 0.987 | 195 | 2 | 3 | 0 | 0.985 | 6 |

# Traditional signal: LLM judge (gpt-4o-mini, last-8-turn window)

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.203 | 0.087 | 0.122 | 13 | 51 | 136 | 0 | 1.000 | 14 |
| K=5 | 0.338 | 0.174 | 0.230 | 26 | 51 | 123 | 0 | 1.000 | 14 |
| K=10 | 0.490 | 0.329 | 0.394 | 49 | 51 | 100 | 0 | 1.000 | 14 |
| K=inf | 0.745 | 1.000 | 0.854 | 149 | 51 | 0 | 0 | 1.000 | 14 |

