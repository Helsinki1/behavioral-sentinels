# Experiment 2 canary: lagged_echo

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.520 | 0.083 | 0.143 | 13 | 12 | 144 | 31 | 0.270 | 6.000 |
| K=5 | 0.613 | 0.121 | 0.202 | 19 | 12 | 138 | 31 | 0.270 | 6.000 |
| K=10 | 0.714 | 0.191 | 0.301 | 30 | 12 | 127 | 31 | 0.270 | 6.000 |
| K=inf | 0.778 | 0.268 | 0.398 | 42 | 12 | 115 | 31 | 0.270 | 6.000 |

first-hallucination error mix: {'missing_def': 50, 'wrong_sig': 35, 'no_code_block': 33, 'fabricated_symbol': 28, 'wrong_def_sig': 16, 'missing_sig': 1, 'missing_wired_call': 1}
