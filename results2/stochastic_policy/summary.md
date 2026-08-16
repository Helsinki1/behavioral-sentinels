# Experiment 2 canary: stochastic_policy

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.800 | 0.025 | 0.048 | 4 | 1 | 158 | 37 | 0.035 | 0.000 |
| K=5 | 0.800 | 0.025 | 0.048 | 4 | 1 | 158 | 37 | 0.035 | 0.000 |
| K=10 | 0.833 | 0.031 | 0.059 | 5 | 1 | 157 | 37 | 0.035 | 0.000 |
| K=inf | 0.857 | 0.037 | 0.071 | 6 | 1 | 156 | 37 | 0.035 | 0.000 |

first-hallucination error mix: {'missing_def': 50, 'wrong_sig': 49, 'fabricated_symbol': 36, 'wrong_def_sig': 28, 'no_code_block': 18, 'missing_sig': 6, 'missing_wired_call': 2, 'fabricated_sig': 1}
