# Experiment 3 -- task set: coding

Primary prediction window K=5. Signal = first turn with an applicable canary score < 1. Same TP/FP/FN/TN rule as experiments 1-2 (experiments/metrics.py). Each canary isolates one axis (second column of the table).


## model: gpt-oss-20b, n=200 paired tasks, K=5

| signal | axis | precision | recall | F1 | fire rate | median lead |
|---|---|---|---|---|---|---|
| lag_span | memory distance | 0.513 | 0.239 | 0.326 | 0.810 | 8 |
| multi_counter | memory breadth | 0.704 | 0.113 | 0.195 | 0.175 | 3 |
| chain_checksum | reasoning composition | 0.613 | 0.230 | 0.335 | 0.500 | 5.500 |
| interference_twin | interference | 0.660 | 0.201 | 0.308 | 0.360 | 4.000 |
| confab_trap | abstention | 0.673 | 0.201 | 0.310 | 0.370 | 4 |
| sparse_recall | unrehearsed retention | 0.457 | 0.098 | 0.162 | 0.195 | 3.000 |
| staircase | headroom | 0.634 | 0.308 | 0.414 | 0.595 | 4 |
| static_trailer | null control | - | 0.000 | 0.000 | 0.000 | - |
| multi_counter_heavy | memory breadth (heavy) | 0.447 | 0.110 | 0.176 | 0.225 | 3.000 |
| ensemble | ensemble (lag+checksum+confab) | 0.823 | 0.349 | 0.491 | 0.780 | 6.000 |
| ensemble_2of3 | ensemble, 2-of-3 vote | 0.808 | 0.204 | 0.326 | 0.310 | 4 |
| Traditional/context_length (th=1000) |  | 0.511 | 0.312 | 0.387 | 0.730 | 6.000 |
| Traditional/turn_number (th=5) |  | 0.545 | 0.357 | 0.431 | 0.845 | 7 |
| Traditional/random_compaction |  | 0.463 | 0.258 | 0.331 | 1.000 | 6 |

### excluding turn-1 hallucinations (K=5)

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| lag_span | 0.513 | 0.287 | 0.368 | 39 | 37 | 97 | 0 | 0.936 | 8 |
| multi_counter | 0.704 | 0.128 | 0.217 | 19 | 8 | 129 | 24 | 0.194 | 3 |
| chain_checksum | 0.613 | 0.266 | 0.371 | 38 | 24 | 105 | 11 | 0.562 | 5.500 |
| interference_twin | 0.660 | 0.241 | 0.353 | 35 | 18 | 110 | 8 | 0.421 | 4.000 |
| confab_trap | 0.673 | 0.222 | 0.333 | 35 | 17 | 123 | 9 | 0.402 | 4 |
| sparse_recall | 0.457 | 0.107 | 0.173 | 16 | 19 | 134 | 18 | 0.209 | 3.000 |
| staircase | 0.630 | 0.345 | 0.445 | 51 | 30 | 97 | 1 | 0.659 | 4.000 |
| static_trailer | - | 0.000 | 0.000 | 0 | 0 | 151 | 42 | 0.000 | - |
| multi_counter_heavy | 0.447 | 0.122 | 0.192 | 17 | 21 | 122 | 24 | 0.245 | 3.000 |
| ensemble | 0.811 | 0.411 | 0.545 | 60 | 14 | 86 | 0 | 0.944 | 6 |
| ensemble_2of3 | 0.808 | 0.260 | 0.394 | 38 | 9 | 108 | 5 | 0.388 | 4 |
| Traditional/context_length | 0.511 | 0.338 | 0.407 | 48 | 46 | 94 | 0 | 0.777 | 6.000 |
| Traditional/turn_number | 0.545 | 0.387 | 0.453 | 55 | 46 | 87 | 0 | 0.899 | 7 |

### observer effect: hallucination rate by condition

| condition | hallucination rate |
|---|---|
| baseline (no canary) | 0.77 |
| lag_span | 0.815 (+0.045) |
| multi_counter | 0.84 (+0.070) |
| chain_checksum | 0.825 (+0.055) |
| interference_twin | 0.87 (+0.100) |
| confab_trap | 0.87 (+0.100) |
| sparse_recall | 0.815 (+0.045) |
| staircase | 0.845 (+0.075) |
| ensemble | 0.93 (+0.160) |
| static_trailer | 0.79 (+0.020) |
| multi_counter_heavy | 0.775 (+0.005) |

First-hallucination error mix (baseline runs): {'missing_def': 48, 'wrong_sig': 35, 'no_code_block': 28, 'wrong_def_sig': 27, 'fabricated_symbol': 27, 'missing_sig': 3, 'fabricated_sig': 1, 'missing_wired_call': 1}
