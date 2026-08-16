# Experiment 2 canary: rotating_prefix

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | - | 0.000 | 0.000 | 0 | 0 | 159 | 41 | 0.000 | - |
| K=5 | - | 0.000 | 0.000 | 0 | 0 | 159 | 41 | 0.000 | - |
| K=10 | - | 0.000 | 0.000 | 0 | 0 | 159 | 41 | 0.000 | - |
| K=inf | - | 0.000 | 0.000 | 0 | 0 | 159 | 41 | 0.000 | - |

first-hallucination error mix: {'missing_def': 45, 'no_code_block': 34, 'wrong_sig': 33, 'fabricated_symbol': 30, 'wrong_def_sig': 29, 'missing_sig': 3, 'missing_wired_call': 1}
