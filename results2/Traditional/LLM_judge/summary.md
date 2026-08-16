# Traditional signal: LLM judge (gpt-oss-20b, last-8-turn window, n judged = 200)

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.516 | 0.318 | 0.394 | 49 | 46 | 105 | 0 | 0.975 | 6 |
| K=5 | 0.610 | 0.468 | 0.529 | 72 | 46 | 82 | 0 | 0.975 | 6 |
| K=10 | 0.703 | 0.708 | 0.706 | 109 | 46 | 45 | 0 | 0.975 | 6 |
| K=inf | 0.764 | 0.968 | 0.854 | 149 | 46 | 5 | 0 | 0.975 | 6 |

