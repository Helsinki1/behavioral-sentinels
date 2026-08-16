# Experiment 2 canary: conditional_rule

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.293 | 0.077 | 0.122 | 12 | 29 | 144 | 15 | 0.465 | 7.000 |
| K=5 | 0.482 | 0.173 | 0.255 | 27 | 29 | 129 | 15 | 0.465 | 7.000 |
| K=10 | 0.633 | 0.321 | 0.425 | 50 | 29 | 106 | 15 | 0.465 | 7.000 |
| K=inf | 0.688 | 0.410 | 0.514 | 64 | 29 | 92 | 15 | 0.465 | 7.000 |

first-hallucination error mix: {'wrong_sig': 47, 'fabricated_symbol': 39, 'wrong_def_sig': 34, 'missing_def': 29, 'no_code_block': 14, 'missing_sig': 5, 'fabricated_sig': 1}
