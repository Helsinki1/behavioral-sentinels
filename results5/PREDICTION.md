# Experiment 5 — same-trajectory prediction metrics (K=5)

Model `gpt-oss-20b`. Signal quality separated from intervention value: each table below re-scores every signal — including the turn-number / context-length / random baselines — on ONE set of trajectories (the exp-3 cross-condition pitfall handled by construction). Scoring unit = the pre-first-reset segment; S = first fire, H = first hallucination in the segment; TP within K=5 turns (experiments/metrics.py rule). Thresholded baselines are tuned to best F1 on the evaluation set itself — maximally generous to them.

Censoring: where a set's own signal triggers the reset, that signal's segment ends at its first fire (lead ≈ 0 by construction, prevented outcomes unobservable). Clean reads: the zero-carry monitor on `A_no_reset`, the carried probe on `C_prime_routed`.

## Trajectory set `A_no_reset` — never resets: full-horizon trajectories, the cleanest read

90 segments (30 coding, 30 registers, 30 babi), median length 21.0 turns, 0% truncated by a reset.

### Pooled

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| zero-carry monitor | 0.783 | 0.261 | 0.391 | 0.622 | 0.000 | 90 |
| turn_number (th=5) | 0.596 | 0.449 | 0.512 | 1.000 | 5.000 | 90 |
| context_length (th=1000) | 0.462 | 0.261 | 0.333 | 0.989 | 2.500 | 90 |
| random (expected) | 0.458 | 0.257 | 0.329 | 1.000 | 5 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | zero-carry monitor | 0.643 | 0.450 | 0.529 | 0.700 | 4 | 30 |
| coding | turn_number (th=5) | 0.444 | 0.400 | 0.421 | 1.000 | 4 | 30 |
| coding | context_length (th=800) | 0.474 | 0.450 | 0.462 | 1.000 | 4 | 30 |
| coding | random (expected) | 0.309 | 0.223 | 0.259 | 1.000 | 4 | 30 |
| registers | zero-carry monitor | 1.000 | 0.105 | 0.191 | 0.300 | 0.000 | 30 |
| registers | turn_number (th=15) | 0.389 | 0.368 | 0.378 | 1.000 | 4 | 30 |
| registers | context_length (th=1000) | 0.476 | 0.526 | 0.500 | 1.000 | 3.500 | 30 |
| registers | random (expected) | 0.297 | 0.245 | 0.268 | 1.000 | 8 | 30 |
| babi | zero-carry monitor | 1.000 | 0.233 | 0.378 | 0.867 | 0 | 30 |
| babi | turn_number (th=5) | 1.000 | 0.633 | 0.775 | 1.000 | 3 | 30 |
| babi | context_length (th=800) | 1.000 | 0.033 | 0.065 | 1.000 | 1 | 30 |
| babi | random (expected) | 1.000 | 0.287 | 0.446 | 1.000 | 4 | 30 |

## Trajectory set `D_routed` — the probe itself triggers resets: its lead is right-censored at the fire; the same-set baselines share the segments

90 segments (30 coding, 30 registers, 30 babi), median length 9.0 turns, 88% truncated by a reset. Self-triggered signal: `routed probe`.

### Pooled

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| routed probe ✂ | 0.109 | 0.162 | 0.130 | 0.911 | 0.000 | 90 |
| zero-carry monitor | 0.538 | 0.189 | 0.280 | 0.311 | 1.000 | 90 |
| turn_number (th=5) | 0.360 | 0.486 | 0.414 | 0.767 | 4.000 | 90 |
| context_length (th=800) | 0.386 | 0.460 | 0.420 | 0.689 | 5 | 90 |
| random (expected) | 0.230 | 0.428 | 0.299 | 1.000 | 4 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | routed probe | 0.167 | 0.214 | 0.188 | 0.933 | 0 | 30 |
| coding | zero-carry monitor | 0.455 | 0.357 | 0.400 | 0.433 | 1.500 | 30 |
| coding | turn_number (th=5) | 0.364 | 0.571 | 0.444 | 0.933 | 4 | 30 |
| coding | context_length (th=800) | 0.391 | 0.643 | 0.486 | 0.933 | 3.500 | 30 |
| coding | random (expected) | 0.285 | 0.457 | 0.351 | 1.000 | 3 | 30 |
| registers | routed probe | 0.000 | 0.000 | 0.000 | 0.800 | - | 30 |
| registers | zero-carry monitor | 1.000 | 0.062 | 0.118 | 0.367 | 0 | 30 |
| registers | turn_number (th=10) | 0.389 | 0.438 | 0.412 | 0.867 | 3 | 30 |
| registers | context_length (th=1000) | 0.389 | 0.438 | 0.412 | 0.867 | 2.500 | 30 |
| registers | random (expected) | 0.257 | 0.303 | 0.278 | 1.000 | 6 | 30 |
| babi | routed probe | 0.115 | 0.429 | 0.182 | 1.000 | 0 | 30 |
| babi | zero-carry monitor | 1.000 | 0.143 | 0.250 | 0.133 | 0 | 30 |
| babi | turn_number (th=5) | 0.556 | 0.714 | 0.625 | 0.367 | 1 | 30 |
| babi | context_length (th=800) | 1.000 | 0.143 | 0.250 | 0.167 | 0 | 30 |
| babi | random (expected) | 0.167 | 0.657 | 0.266 | 1.000 | 3 | 30 |

## Trajectory set `C_prime_routed` — the CLOCK ends segments (first reset at turn 6), so the probe's fires are not self-censored -- but segments are short

90 segments (30 coding, 30 registers, 30 babi), median length 5.0 turns, 100% truncated by a reset.

### Pooled

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| routed probe | 0.087 | 0.133 | 0.105 | 0.256 | 1.500 | 90 |
| zero-carry monitor | 0.286 | 0.133 | 0.182 | 0.078 | 0.500 | 90 |
| turn_number (th=3) | 0.128 | 0.733 | 0.218 | 1.000 | 1 | 90 |
| context_length (th=800) | 0.077 | 0.133 | 0.098 | 0.333 | 0.500 | 90 |
| random (expected) | 0.118 | 0.667 | 0.200 | 1.000 | 1.000 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | routed probe | 0.000 | 0.000 | 0.000 | 0.067 | - | 30 |
| coding | zero-carry monitor | 0.286 | 0.333 | 0.308 | 0.233 | 0.500 | 30 |
| coding | turn_number (th=3) | 0.111 | 0.500 | 0.182 | 1.000 | 2 | 30 |
| coding | context_length (th=800) | 0.077 | 0.333 | 0.125 | 1.000 | 0.500 | 30 |
| coding | random (expected) | 0.130 | 0.600 | 0.214 | 1.000 | 1.000 | 30 |
| registers | routed probe | 1.000 | 0.125 | 0.222 | 0.033 | 2 | 30 |
| registers | zero-carry monitor | - | 0.000 | 0.000 | 0.000 | - | 30 |
| registers | turn_number (th=3) | 0.241 | 0.875 | 0.378 | 1.000 | 1 | 30 |
| registers | context_length (th=800) | - | 0.000 | 0.000 | 0.000 | - | 30 |
| registers | random (expected) | 0.209 | 0.725 | 0.324 | 1.000 | 1 | 30 |
| babi | routed probe | 0.050 | 1.000 | 0.095 | 0.667 | 1 | 30 |
| babi | zero-carry monitor | - | 0.000 | 0.000 | 0.000 | - | 30 |
| babi | turn_number (th=3) | 0.033 | 1.000 | 0.065 | 1.000 | 0 | 30 |
| babi | context_length (th=800) | - | 0.000 | 0.000 | 0.000 | - | 30 |
| babi | random (expected) | 0.020 | 0.600 | 0.039 | 1.000 | 1 | 30 |

## Trajectory set `D_labeled` — deterministically routed probe triggers resets (same censoring as D_routed, no router noise)

90 segments (30 coding, 30 registers, 30 babi), median length 9.0 turns, 91% truncated by a reset. Self-triggered signal: `labeled probe`.

### Pooled

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| labeled probe ✂ | 0.060 | 0.148 | 0.085 | 0.933 | 0.000 | 90 |
| zero-carry monitor | 0.200 | 0.074 | 0.108 | 0.211 | 4 | 90 |
| turn_number (th=8) | 0.275 | 0.407 | 0.328 | 0.589 | 3.000 | 90 |
| context_length (th=1000) | 0.294 | 0.370 | 0.328 | 0.556 | 3.000 | 90 |
| random (expected) | 0.143 | 0.389 | 0.209 | 1.000 | 4.000 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | labeled probe | 0.130 | 0.300 | 0.182 | 1.000 | 0 | 30 |
| coding | zero-carry monitor | 0.200 | 0.200 | 0.200 | 0.367 | 4 | 30 |
| coding | turn_number (th=8) | 0.261 | 0.600 | 0.364 | 0.833 | 1.500 | 30 |
| coding | context_length (th=1000) | 0.280 | 0.700 | 0.400 | 0.933 | 3 | 30 |
| coding | random (expected) | 0.216 | 0.552 | 0.311 | 1.000 | 4 | 30 |
| registers | labeled probe | 0.000 | 0.000 | 0.000 | 0.800 | - | 30 |
| registers | zero-carry monitor | - | 0.000 | 0.000 | 0.267 | - | 30 |
| registers | turn_number (th=3) | 0.333 | 0.438 | 0.378 | 1.000 | 6 | 30 |
| registers | context_length (th=800) | 0.278 | 0.312 | 0.294 | 0.967 | 4 | 30 |
| registers | random (expected) | 0.221 | 0.249 | 0.234 | 1.000 | 5 | 30 |
| babi | labeled probe | 0.033 | 1.000 | 0.065 | 1.000 | 0 | 30 |
| babi | zero-carry monitor | - | 0.000 | 0.000 | 0.000 | - | 30 |
| babi | turn_number (th=3) | 0.048 | 1.000 | 0.091 | 0.700 | 1 | 30 |
| babi | context_length (th=800) | - | 0.000 | 0.000 | 0.000 | - | 30 |
| babi | random (expected) | 0.033 | 1.000 | 0.065 | 1.000 | 1.500 | 30 |

## Trajectory set `C_judge` — the judge triggers resets: its lead is right-censored at the fire

90 segments (30 coding, 30 registers, 30 babi), median length 9.0 turns, 98% truncated by a reset. Self-triggered signal: `LLM judge`.

### Pooled

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| LLM judge ✂ | 0.147 | 0.458 | 0.222 | 0.978 | 0 | 90 |
| zero-carry monitor | 0.545 | 0.250 | 0.343 | 0.144 | 0.000 | 90 |
| turn_number (th=5) | 0.254 | 0.667 | 0.368 | 0.756 | 3 | 90 |
| context_length (th=800) | 0.097 | 0.125 | 0.109 | 0.378 | 0 | 90 |
| random (expected) | 0.197 | 0.676 | 0.306 | 1.000 | 4.000 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | LLM judge | 0.069 | 1.000 | 0.129 | 0.967 | 0.000 | 30 |
| coding | zero-carry monitor | 0.000 | 0.000 | 0.000 | 0.167 | - | 30 |
| coding | turn_number (th=3) | 0.048 | 0.500 | 0.087 | 0.700 | 0 | 30 |
| coding | context_length (th=800) | 0.000 | 0.000 | 0.000 | 0.300 | - | 30 |
| coding | random (expected) | 0.067 | 1.000 | 0.125 | 1.000 | 1 | 30 |
| registers | LLM judge | 0.074 | 0.500 | 0.129 | 0.967 | 0.000 | 30 |
| registers | zero-carry monitor | - | 0.000 | 0.000 | 0.000 | - | 30 |
| registers | turn_number (th=8) | 0.111 | 0.750 | 0.194 | 0.900 | 2 | 30 |
| registers | context_length (th=800) | 0.136 | 0.750 | 0.231 | 0.733 | 0 | 30 |
| registers | random (expected) | 0.088 | 0.627 | 0.154 | 1.000 | 4 | 30 |
| babi | LLM judge | 0.368 | 0.389 | 0.378 | 1.000 | 0 | 30 |
| babi | zero-carry monitor | 1.000 | 0.333 | 0.500 | 0.267 | 0.000 | 30 |
| babi | turn_number (th=5) | 0.560 | 0.778 | 0.651 | 0.967 | 2.000 | 30 |
| babi | context_length (th=800) | - | 0.000 | 0.000 | 0.100 | - | 30 |
| babi | random (expected) | 0.494 | 0.652 | 0.562 | 1.000 | 4.000 | 30 |

✂ = self-triggered on this set: the segment ends at this signal's first fire, so its lead time is right-censored.
