# Experiment 3 -- task set: babi

Primary prediction window K=5. Signal = first turn with an applicable canary score < 1. Same TP/FP/FN/TN rule as experiments 1-2 (experiments/metrics.py). Each canary isolates one axis (second column of the table).


## model: gpt-oss-20b, n=100 paired tasks, K=5

| signal | axis | precision | recall | F1 | fire rate | median lead |
|---|---|---|---|---|---|---|
| lag_span | memory distance | 1.000 | 0.570 | 0.726 | 1.000 | 4.500 |
| multi_counter | memory breadth | 1.000 | 0.170 | 0.291 | 0.230 | 4 |
| chain_checksum | reasoning composition | 1.000 | 0.440 | 0.611 | 0.480 | 0.000 |
| interference_twin | interference | 1.000 | 0.130 | 0.230 | 0.150 | 0 |
| confab_trap | abstention | 1.000 | 0.340 | 0.507 | 0.360 | 1.000 |
| sparse_recall | unrehearsed retention | 1.000 | 0.210 | 0.347 | 0.220 | 1.000 |
| staircase | headroom | 1.000 | 0.390 | 0.561 | 0.460 | 2.000 |
| static_trailer | null control | - | 0.000 | 0.000 | 0.000 | - |
| ensemble | ensemble (lag+checksum+confab) | 0.983 | 0.596 | 0.742 | 0.940 | 5 |
| ensemble_2of3 | ensemble, 2-of-3 vote | 0.964 | 0.273 | 0.425 | 0.330 | 2.000 |
| Traditional/context_length (th=1000) |  | 1.000 | 0.010 | 0.020 | 0.010 | 2 |
| Traditional/turn_number (th=5) |  | 1.000 | 0.640 | 0.780 | 0.980 | 3.000 |
| Traditional/random_compaction |  | 1.000 | 0.301 | 0.463 | 1.000 | 4 |

### excluding turn-1 hallucinations (K=5)

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| lag_span | 1.000 | 0.570 | 0.726 | 57 | 0 | 43 | 0 | 1.000 | 4.500 |
| multi_counter | 1.000 | 0.170 | 0.291 | 17 | 0 | 83 | 0 | 0.230 | 4 |
| chain_checksum | 1.000 | 0.440 | 0.611 | 44 | 0 | 56 | 0 | 0.480 | 0.000 |
| interference_twin | 1.000 | 0.131 | 0.232 | 13 | 0 | 86 | 0 | 0.151 | 0 |
| confab_trap | 1.000 | 0.340 | 0.507 | 34 | 0 | 66 | 0 | 0.360 | 1.000 |
| sparse_recall | 1.000 | 0.210 | 0.347 | 21 | 0 | 79 | 0 | 0.220 | 1.000 |
| staircase | 1.000 | 0.390 | 0.561 | 39 | 0 | 61 | 0 | 0.460 | 2.000 |
| static_trailer | - | 0.000 | 0.000 | 0 | 0 | 100 | 0 | 0.000 | - |
| ensemble | 0.983 | 0.602 | 0.747 | 59 | 1 | 39 | 0 | 0.950 | 5 |
| ensemble_2of3 | 0.964 | 0.276 | 0.429 | 27 | 1 | 71 | 0 | 0.333 | 2.000 |
| Traditional/context_length | 1.000 | 0.010 | 0.020 | 1 | 0 | 99 | 0 | 0.010 | 2 |
| Traditional/turn_number | 1.000 | 0.640 | 0.780 | 64 | 0 | 36 | 0 | 0.980 | 3.000 |

### observer effect: hallucination rate by condition

| condition | hallucination rate |
|---|---|
| baseline (no canary) | 1.0 |
| lag_span | 1.0 (+0.000) |
| multi_counter | 1.0 (+0.000) |
| chain_checksum | 1.0 (+0.000) |
| interference_twin | 1.0 (+0.000) |
| confab_trap | 1.0 (+0.000) |
| sparse_recall | 1.0 (+0.000) |
| staircase | 1.0 (+0.000) |
| ensemble | 0.99 (-0.010) |
| static_trailer | 1.0 (+0.000) |

First-hallucination error mix (baseline runs): {'wrong_answer': 100}
