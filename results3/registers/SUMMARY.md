# Experiment 3 -- task set: registers

Primary prediction window K=5. Signal = first turn with an applicable canary score < 1. Same TP/FP/FN/TN rule as experiments 1-2 (experiments/metrics.py). Each canary isolates one axis (second column of the table).


## model: gpt-oss-20b, n=100 paired tasks, K=5

| signal | axis | precision | recall | F1 | fire rate | median lead |
|---|---|---|---|---|---|---|
| lag_span | memory distance | 0.618 | 0.241 | 0.347 | 0.890 | 10.000 |
| multi_counter | memory breadth | 0.667 | 0.217 | 0.327 | 0.410 | 4.500 |
| chain_checksum | reasoning composition | 0.846 | 0.247 | 0.383 | 0.380 | 4.500 |
| interference_twin | interference | 0.607 | 0.227 | 0.330 | 0.470 | 6.500 |
| confab_trap | abstention | 0.778 | 0.152 | 0.255 | 0.230 | 4 |
| sparse_recall | unrehearsed retention | 0.600 | 0.071 | 0.128 | 0.120 | 1.000 |
| staircase | headroom | 0.821 | 0.242 | 0.374 | 0.360 | 4 |
| static_trailer | null control | 1.000 | 0.024 | 0.046 | 0.020 | 0.000 |
| ensemble | ensemble (lag+checksum+confab) | 0.975 | 0.394 | 0.561 | 0.920 | 6 |
| ensemble_2of3 | ensemble, 2-of-3 vote | 1.000 | 0.232 | 0.377 | 0.340 | 4.000 |
| Traditional/context_length (th=1000) |  | 0.450 | 0.403 | 0.425 | 0.770 | 4.000 |
| Traditional/turn_number (th=15) |  | 0.421 | 0.358 | 0.387 | 0.750 | 4.000 |
| Traditional/random_compaction |  | 0.322 | 0.234 | 0.271 | 1.000 | 8.000 |

### excluding turn-1 hallucinations (K=5)

| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |
|---|---|---|---|---|---|---|---|---|---|
| lag_span | 0.618 | 0.241 | 0.347 | 21 | 13 | 66 | 0 | 0.890 | 10.000 |
| multi_counter | 0.667 | 0.217 | 0.327 | 18 | 9 | 65 | 8 | 0.410 | 4.500 |
| chain_checksum | 0.846 | 0.247 | 0.383 | 22 | 4 | 67 | 7 | 0.380 | 4.500 |
| interference_twin | 0.607 | 0.227 | 0.330 | 17 | 11 | 58 | 14 | 0.470 | 6.500 |
| confab_trap | 0.778 | 0.152 | 0.255 | 14 | 4 | 78 | 4 | 0.230 | 4 |
| sparse_recall | 0.600 | 0.071 | 0.128 | 6 | 4 | 78 | 12 | 0.120 | 1.000 |
| staircase | 0.821 | 0.242 | 0.374 | 23 | 5 | 72 | 0 | 0.360 | 4 |
| static_trailer | 1.000 | 0.024 | 0.046 | 2 | 0 | 82 | 16 | 0.020 | 0.000 |
| ensemble | 0.975 | 0.398 | 0.565 | 39 | 1 | 59 | 0 | 0.929 | 6 |
| ensemble_2of3 | 1.000 | 0.235 | 0.380 | 23 | 0 | 75 | 1 | 0.343 | 4.000 |
| Traditional/context_length | 0.450 | 0.403 | 0.425 | 27 | 33 | 40 | 0 | 0.770 | 4.000 |
| Traditional/turn_number | 0.421 | 0.358 | 0.387 | 24 | 33 | 43 | 0 | 0.750 | 4.000 |

### observer effect: hallucination rate by condition

| condition | hallucination rate |
|---|---|
| baseline (no canary) | 0.67 |
| lag_span | 0.87 (+0.200) |
| multi_counter | 0.83 (+0.160) |
| chain_checksum | 0.89 (+0.220) |
| interference_twin | 0.75 (+0.080) |
| confab_trap | 0.92 (+0.250) |
| sparse_recall | 0.84 (+0.170) |
| staircase | 0.95 (+0.280) |
| ensemble | 0.99 (+0.320) |
| static_trailer | 0.84 (+0.170) |

First-hallucination error mix (baseline runs): {'wrong_value': 66, 'missing_value': 3}
