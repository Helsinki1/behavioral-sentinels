# Experiment 2 canary: escalating_ledger

## model: gpt-oss-20b (open), n=200 tasks

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| K=2 | 0.476 | 0.185 | 0.267 | 30 | 33 | 132 | 5 | 0.550 | 4 |
| K=5 | 0.612 | 0.321 | 0.421 | 52 | 33 | 110 | 5 | 0.550 | 4 |
| K=10 | 0.670 | 0.414 | 0.511 | 67 | 33 | 95 | 5 | 0.550 | 4 |
| K=inf | 0.700 | 0.475 | 0.566 | 77 | 33 | 85 | 5 | 0.550 | 4 |

first-hallucination error mix: {'wrong_sig': 46, 'missing_def': 40, 'wrong_def_sig': 37, 'fabricated_symbol': 24, 'no_code_block': 20, 'missing_sig': 11, 'fabricated_sig': 3, 'missing_wired_call': 1}
