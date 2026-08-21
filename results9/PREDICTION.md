# Experiment 9 — signal quality per model (K=5)

Quiz checkpoints every 3 turns (3-turn S granularity); thresholded baselines tuned to best F1 on the evaluation set itself. ✂ = the signal triggers that set's resets (lead right-censored).

## gpt-oss-120b

Quiz fail-threshold ablation (shadow pass):

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| quiz fail>=1 | 0.875 | 0.212 | 0.342 | 0.706 | 0 | 34 |
| quiz fail>=2 | 1.000 | 0.030 | 0.059 | 0.088 | 0 | 34 |
| quiz fail>=3 | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `A_no_reset` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| zero-carry trace monitor | 1.000 | 1.000 | 1.000 | 0.971 | 0 | 34 |
| frozen-state quiz (shadow) | 1.000 | 0.030 | 0.059 | 0.088 | 0 | 34 |
| turn_number (th=8) | 0.955 | 0.636 | 0.764 | 1.000 | 2.500 | 34 |
| context_length (th=400) | 0.955 | 0.636 | 0.764 | 1.000 | 3 | 34 |
| random (expected) | 0.920 | 0.349 | 0.506 | 1.000 | 4.000 | 34 |

### set `QUIZ` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| frozen-state quiz (live) ✂ | 1.000 | 0.061 | 0.114 | 0.147 | 0.000 | 34 |
| zero-carry trace monitor | 1.000 | 0.970 | 0.985 | 0.971 | 0.000 | 34 |

### set `ACT_carry_clock` — 34 segments, median 5.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe (clock-truncated read) | 0.000 | - | 0.000 | 1.000 | - | 34 |
| zero-carry trace monitor | - | - | 0.000 | 0.000 | - | 34 |

### set `ACT_probe` — 34 segments, median 2.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe ✂ | 0.000 | - | 0.000 | 1.000 | - | 34 |
| zero-carry trace monitor | - | - | 0.000 | 0.000 | - | 34 |

## deepseek-v4-flash

Quiz fail-threshold ablation (shadow pass):

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| quiz fail>=1 | 0.600 | 0.100 | 0.171 | 0.353 | 3 | 34 |
| quiz fail>=2 | - | 0.000 | 0.000 | 0.000 | - | 34 |
| quiz fail>=3 | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `A_no_reset` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.000 | - | 34 |
| frozen-state quiz (shadow) | - | 0.000 | 0.000 | 0.000 | - | 34 |
| turn_number (th=3) | 0.818 | 0.600 | 0.692 | 1.000 | 4.500 | 34 |
| context_length (th=400) | 0.636 | 0.233 | 0.342 | 1.000 | 5 | 34 |
| random (expected) | 0.715 | 0.334 | 0.456 | 1.000 | 4 | 34 |

### set `QUIZ` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| frozen-state quiz (live) ✂ | - | 0.000 | 0.000 | 0.000 | - | 34 |
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.029 | - | 34 |

### set `ACT_carry_clock` — 34 segments, median 5.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe (clock-truncated read) | 0.147 | 1.000 | 0.256 | 1.000 | 3 | 34 |
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `ACT_probe` — 34 segments, median 2.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe ✂ | 0.000 | - | 0.000 | 1.000 | - | 34 |
| zero-carry trace monitor | - | - | 0.000 | 0.000 | - | 34 |

## qwen3p7-plus

Quiz fail-threshold ablation (shadow pass):

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| quiz fail>=1 | 0.333 | 0.037 | 0.067 | 0.147 | 0 | 34 |
| quiz fail>=2 | 1.000 | 0.037 | 0.071 | 0.029 | 0 | 34 |
| quiz fail>=3 | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `A_no_reset` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| zero-carry trace monitor | 1.000 | 0.037 | 0.071 | 0.029 | 0 | 34 |
| frozen-state quiz (shadow) | 1.000 | 0.037 | 0.071 | 0.029 | 0 | 34 |
| turn_number (th=5) | 0.682 | 0.556 | 0.612 | 1.000 | 5.000 | 34 |
| context_length (th=400) | 0.588 | 0.370 | 0.455 | 1.000 | 5.000 | 34 |
| random (expected) | 0.572 | 0.347 | 0.432 | 1.000 | 5 | 34 |

### set `QUIZ` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| frozen-state quiz (live) ✂ | - | 0.000 | 0.000 | 0.000 | - | 34 |
| zero-carry trace monitor | 1.000 | 0.037 | 0.071 | 0.029 | 0 | 34 |

### set `ACT_carry_clock` — 34 segments, median 5.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe (clock-truncated read) | 0.088 | 1.000 | 0.162 | 1.000 | 3 | 34 |
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `ACT_probe` — 34 segments, median 2.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe ✂ | 0.000 | - | 0.000 | 1.000 | - | 34 |
| zero-carry trace monitor | - | - | 0.000 | 0.000 | - | 34 |

## gpt-4o-mini

Quiz fail-threshold ablation (shadow pass):

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| quiz fail>=1 | 1.000 | 0.059 | 0.111 | 0.765 | 2.500 | 34 |
| quiz fail>=2 | - | 0.000 | 0.000 | 0.029 | - | 34 |
| quiz fail>=3 | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `A_no_reset` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.147 | - | 34 |
| frozen-state quiz (shadow) | - | 0.000 | 0.000 | 0.029 | - | 34 |
| turn_number (th=3) | 1.000 | 0.794 | 0.885 | 1.000 | 3.000 | 34 |
| context_length (th=400) | 1.000 | 0.206 | 0.342 | 1.000 | 1.500 | 34 |
| random (expected) | 1.000 | 0.325 | 0.490 | 1.000 | 3 | 34 |

### set `QUIZ` — 34 segments, median 17.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| frozen-state quiz (live) ✂ | - | 0.000 | 0.000 | 0.000 | - | 34 |
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.118 | - | 34 |

### set `ACT_carry_clock` — 34 segments, median 5.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe (clock-truncated read) | 0.324 | 1.000 | 0.489 | 1.000 | 3 | 34 |
| zero-carry trace monitor | - | 0.000 | 0.000 | 0.000 | - | 34 |

### set `ACT_probe` — 34 segments, median 2.0 turns

| signal | precision | recall | F1 | fire rate | median lead | n |
|---|---|---|---|---|---|---|
| carried probe ✂ | 0.000 | - | 0.000 | 1.000 | - | 34 |
| zero-carry trace monitor | - | - | 0.000 | 0.000 | - | 34 |

