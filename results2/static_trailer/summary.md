# Experiment 2 canary: static_trailer

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | - | 0.000 | 0.000 | 0 | 0 | 153 | 47 | 0.000 | - |
| K=5 | - | 0.000 | 0.000 | 0 | 0 | 153 | 47 | 0.000 | - |
| K=10 | - | 0.000 | 0.000 | 0 | 0 | 153 | 47 | 0.000 | - |
| K=inf | - | 0.000 | 0.000 | 0 | 0 | 153 | 47 | 0.000 | - |

first-hallucination error mix: {'missing_def': 40, 'fabricated_symbol': 39, 'wrong_sig': 30, 'wrong_def_sig': 28, 'no_code_block': 25, 'missing_sig': 2, 'fabricated_sig': 1}
