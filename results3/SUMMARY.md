# Experiment 3 -- Results Summary (axis-isolating canaries x 3 task sets)

One row per (task set, model, signal) at the primary window K=5. Full per-set tables (score sweeps, turn-1-excluded, observer effect) live in results3/<set>/SUMMARY.md.


## coding / gpt-oss-20b (n=200)

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

## registers / gpt-oss-20b (n=100)

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

## babi / gpt-oss-20b (n=100)

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
