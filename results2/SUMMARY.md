# Experiment 2 — Dynamic Canaries on a Coding Task

200 (gpt-oss-20b) synthetic incremental-coding tasks (maintain a Python module across 12-30 turns of add/rename/delete/re-signature edits). Primary prediction window K=5 turns. TP = signal fired at/before the first hallucination and within K turns of it; FP = fired on a clean trajectory; FN = hallucination not predicted (no firing, fired late, or window exceeded); TN = clean and silent. Context-length/turn-number rows use the best-F1 threshold from their sweep. Random compaction is an analytic expectation over a uniform firing turn.

Canaries are DYNAMIC (the required output changes over the trajectory) except `static_trailer`, which is the experiment-1-style fixed-string control.


## model: gpt-oss-20b (open), n=200 paired tasks, K=5

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| rotating_prefix | - | 0.000 | 0.000 | 0 | 0 | 159 | 41 | 0.000 | - |
| stochastic_policy | 0.800 | 0.025 | 0.048 | 4 | 1 | 158 | 37 | 0.035 | 0.000 |
| lagged_echo | 0.613 | 0.121 | 0.202 | 19 | 12 | 138 | 31 | 0.270 | 6.000 |
| conditional_rule | 0.482 | 0.173 | 0.255 | 27 | 29 | 129 | 15 | 0.465 | 7.000 |
| escalating_ledger | 0.612 | 0.321 | 0.421 | 52 | 33 | 110 | 5 | 0.550 | 4 |
| static_trailer | - | 0.000 | 0.000 | 0 | 0 | 153 | 47 | 0.000 | - |
| Traditional/context_length (th=1000) | 0.471 | 0.266 | 0.340 | 41 | 46 | 113 | 0 | 0.730 | 7.000 |
| Traditional/turn_number (th=10) | 0.500 | 0.299 | 0.374 | 46 | 46 | 108 | 0 | 0.655 | 5 |
| Traditional/LLM_judge | 0.610 | 0.468 | 0.529 | 72 | 46 | 82 | 0 | 0.975 | 6 |
| Traditional/random_compaction | 0.464 | 0.258 | 0.332 | 39.740 | 46.000 | 114.260 | 0.000 | 1.000 | 6 |

Hallucination base rate (canary runs vary slightly): 0.795

First-hallucination error mix (baseline runs): {'wrong_sig': 39, 'fabricated_symbol': 36, 'wrong_def_sig': 29, 'no_code_block': 28, 'missing_def': 23, 'missing_sig': 5, 'fabricated_sig': 1, 'missing_wired_call': 1}
