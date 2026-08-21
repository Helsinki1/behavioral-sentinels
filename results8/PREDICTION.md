# Experiment 8 — signal quality of every observation method (K=5)

Model `gpt-oss-20b`. S = first fire, H = first hallucination in the pre-first-reset segment; TP within K=5 turns (experiments/metrics.py rule). Quiz checkpoints occur every 3 turns, so quiz S has 3-turn granularity. Thresholded baselines are tuned to best F1 on the evaluation set itself (maximally generous to them).

## Quiz fail-threshold ablation (shadow pass, full-horizon A_no_reset trajectories)

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| frozen-state quiz (shadow, fail>=1) | 0.559 | 0.275 | 0.369 | 0.889 | 5 | 90 |
| frozen-state quiz (shadow, fail>=2) | 0.600 | 0.087 | 0.152 | 0.456 | 0.500 | 90 |
| frozen-state quiz (shadow, fail>=3) | 0.000 | 0.000 | 0.000 | 0.167 | - | 90 |

## Trajectory set `A_no_reset` — never resets -- the same-trajectory table: every signal (quiz via the shadow pass) on identical full-horizon trajectories. THE Fig-2 source.

90 segments, median 21.0 turns, 0% truncated by a reset.

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| zero-carry trace monitor | 0.783 | 0.261 | 0.391 | 0.622 | 0.000 | 90 |
| frozen-state quiz (shadow) | 0.600 | 0.087 | 0.152 | 0.456 | 0.500 | 90 |
| turn_number (th=5) | 0.596 | 0.449 | 0.512 | 1.000 | 5.000 | 90 |
| context_length (th=1000) | 0.462 | 0.261 | 0.333 | 0.989 | 2.500 | 90 |
| random (expected) | 0.458 | 0.257 | 0.329 | 1.000 | 5 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | zero-carry trace monitor | 0.643 | 0.450 | 0.529 | 0.700 | 4 | 30 |
| coding | frozen-state quiz (shadow) | 0.400 | 0.100 | 0.160 | 0.500 | 0.500 | 30 |
| coding | turn_number (th=5) | 0.444 | 0.400 | 0.421 | 1.000 | 4 | 30 |
| coding | context_length (th=800) | 0.474 | 0.450 | 0.462 | 1.000 | 4 | 30 |
| coding | random (expected) | 0.309 | 0.223 | 0.259 | 1.000 | 4 | 30 |
| registers | zero-carry trace monitor | 1.000 | 0.105 | 0.191 | 0.300 | 0.000 | 30 |
| registers | frozen-state quiz (shadow) | 0.500 | 0.053 | 0.095 | 0.167 | 1 | 30 |
| registers | turn_number (th=15) | 0.389 | 0.368 | 0.378 | 1.000 | 4 | 30 |
| registers | context_length (th=1000) | 0.476 | 0.526 | 0.500 | 1.000 | 3.500 | 30 |
| registers | random (expected) | 0.297 | 0.245 | 0.268 | 1.000 | 8 | 30 |
| babi | zero-carry trace monitor | 1.000 | 0.233 | 0.378 | 0.867 | 0 | 30 |
| babi | frozen-state quiz (shadow) | 1.000 | 0.100 | 0.182 | 0.700 | 0 | 30 |
| babi | turn_number (th=5) | 1.000 | 0.633 | 0.775 | 1.000 | 3 | 30 |
| babi | context_length (th=800) | 1.000 | 0.033 | 0.065 | 1.000 | 1 | 30 |
| babi | random (expected) | 1.000 | 0.287 | 0.446 | 1.000 | 4 | 30 |

## Trajectory set `QUIZ` — the quiz triggers this arm's resets: its live lead is right-censored at the fire

90 segments, median 18.0 turns, 41% truncated by a reset. Self-triggered signal: `frozen-state quiz (live)` ✂.

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| frozen-state quiz (live) ✂ | 0.200 | 0.015 | 0.027 | 0.411 | 0 | 90 |
| zero-carry trace monitor | 0.821 | 0.333 | 0.474 | 0.556 | 0.000 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | frozen-state quiz (live) | 0.250 | 0.050 | 0.083 | 0.400 | 0 | 30 |
| coding | zero-carry trace monitor | 0.615 | 0.400 | 0.485 | 0.633 | 2 | 30 |
| registers | frozen-state quiz (live) | 0.000 | 0.000 | 0.000 | 0.067 | - | 30 |
| registers | zero-carry trace monitor | 1.000 | 0.263 | 0.417 | 0.433 | 0 | 30 |
| babi | frozen-state quiz (live) | - | 0.000 | 0.000 | 0.767 | - | 30 |
| babi | zero-carry trace monitor | 1.000 | 0.333 | 0.500 | 0.600 | 0.000 | 30 |

## Trajectory set `ACT_carry_clock` — the CLOCK ends segments (first reset at turn 6), so the carried probe's fires are not self-censored -- the clean active-signal read, on observer-shifted trajectories by necessity

90 segments, median 5.0 turns, 100% truncated by a reset.

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe (clock-truncated read) | 0.032 | 0.077 | 0.045 | 0.344 | 2 | 90 |
| zero-carry trace monitor | 0.000 | 0.000 | 0.000 | 0.078 | - | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | carried probe (clock-truncated read) | 0.000 | 0.000 | 0.000 | 0.067 | - | 30 |
| coding | zero-carry trace monitor | 0.000 | 0.000 | 0.000 | 0.233 | - | 30 |
| registers | carried probe (clock-truncated read) | 1.000 | 0.125 | 0.222 | 0.033 | 2 | 30 |
| registers | zero-carry trace monitor | - | 0.000 | 0.000 | 0.000 | - | 30 |
| babi | carried probe (clock-truncated read) | 0.000 | - | 0.000 | 0.933 | - | 30 |
| babi | zero-carry trace monitor | - | - | 0.000 | 0.000 | - | 30 |

## Trajectory set `ACT_probe` — the probe triggers resets: its lead is right-censored

90 segments, median 9.0 turns, 92% truncated by a reset. Self-triggered signal: `carried probe` ✂.

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe ✂ | 0.098 | 0.182 | 0.128 | 0.944 | 0.000 | 90 |
| zero-carry trace monitor | 0.364 | 0.121 | 0.182 | 0.200 | 0 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | carried probe | 0.191 | 0.308 | 0.235 | 1.000 | 0.000 | 30 |
| coding | zero-carry trace monitor | 0.300 | 0.231 | 0.261 | 0.433 | 2.000 | 30 |
| registers | carried probe | 0.000 | 0.000 | 0.000 | 0.833 | - | 30 |
| registers | zero-carry trace monitor | 1.000 | 0.056 | 0.105 | 0.167 | 0 | 30 |
| babi | carried probe | 0.067 | 1.000 | 0.125 | 1.000 | 0.000 | 30 |
| babi | zero-carry trace monitor | - | 0.000 | 0.000 | 0.000 | - | 30 |

## Trajectory set `C_judge` — the judge triggers resets: its lead is right-censored

90 segments, median 8.0 turns, 98% truncated by a reset. Self-triggered signal: `LLM judge` ✂.

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| LLM judge ✂ | 0.208 | 0.593 | 0.308 | 0.978 | 0.000 | 90 |
| zero-carry trace monitor | 0.583 | 0.259 | 0.359 | 0.156 | 0 | 90 |

### Per domain

| domain | signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|---|
| coding | LLM judge | 0.138 | 0.800 | 0.235 | 1.000 | 0.000 | 30 |
| coding | zero-carry trace monitor | 0.286 | 0.400 | 0.333 | 0.233 | 1.000 | 30 |
| registers | LLM judge | 0.179 | 1.000 | 0.303 | 0.933 | 0 | 30 |
| registers | zero-carry trace monitor | 1.000 | 0.200 | 0.333 | 0.033 | 0 | 30 |
| babi | LLM judge | 0.350 | 0.412 | 0.378 | 1.000 | 0 | 30 |
| babi | zero-carry trace monitor | 1.000 | 0.235 | 0.381 | 0.200 | 0.000 | 30 |

✂ = the signal that triggers this set's resets: its lead time is right-censored at the first fire.
