# Experiment 8 — active vs passive observation, deployed

Model gpt-oss-20b — 90 tasks paired across every arm (30 coding, 30 registers, 30 babi), full horizon, no early stop, reset operator = R1 reground for every resetting arm. New arms (QUIZ, ACT_probe, ACT_carry_clock) from runs8; every other arm read verbatim from runs6.

Categories: **active** = the observation writes into the agent's trajectory (carried probe); **passive-behavioural** = frozen-state quiz on a discarded fork (extra tokens, zero contamination); **passive-observational** = reads the existing trace only.

## Arms

| arm | category | policy | accuracy | success@0.9 | resets/task | prompt tok | quiz tok |
|---|---|---|---|---|---|---|---|
| A_no_reset | bound | (runs6) | 0.823 | 0.478 | 0.00 | 22,105 | 0 |
| B_random | baseline | (runs6) | 0.824 | 0.478 | 0.86 | 18,858 | 0 |
| C_clock | baseline | (runs6) | 0.847 | 0.589 | 3.28 | 12,580 | 0 |
| C_ctx | baseline | (runs6) | 0.843 | 0.556 | 1.52 | 14,633 | 0 |
| G_dense | baseline | (runs6) | 0.839 | 0.556 | 5.74 | 12,195 | 0 |
| ACT_probe | active | probe | 0.792 | 0.389 | 4.33 | 19,184 | 0 |
| ACT_carry_clock | active | scheduled | 0.811 | 0.478 | 3.28 | 18,024 | 0 |
| QUIZ | passive-behavioural | quiz | 0.813 | 0.422 | 0.60 | 19,964 | 9,294 |
| Z_reground | passive-observational | (runs6) | 0.841 | 0.522 | 0.86 | 18,330 | 0 |
| C_judge | passive-observational | (runs6) | 0.850 | 0.611 | 4.69 | 29,513 | 0 |
| F_oracle | bound | (runs6) | 0.822 | 0.478 | 0.77 | 19,152 | 0 |

## Per-domain accuracy

| arm | coding | registers | babi |
|---|---|---|---|
| A_no_reset | 0.894 | 0.945 | 0.629 |
| B_random | 0.887 | 0.939 | 0.646 |
| C_clock | 0.938 | 0.975 | 0.630 |
| C_ctx | 0.913 | 0.971 | 0.645 |
| G_dense | 0.929 | 0.970 | 0.617 |
| ACT_probe | 0.857 | 0.934 | 0.585 |
| ACT_carry_clock | 0.887 | 0.956 | 0.591 |
| QUIZ | 0.875 | 0.937 | 0.627 |
| Z_reground | 0.914 | 0.954 | 0.655 |
| C_judge | 0.943 | 0.975 | 0.632 |
| F_oracle | 0.871 | 0.948 | 0.647 |

## Paired contrasts, pooled (bootstrap 95% CI on the per-task delta)

| contrast | delta accuracy | 95% CI | significant | better/worse/tied |
|---|---|---|---|---|
| QUIZ - C_clock: frozen-state QUIZ vs turn-count clock | -0.0347 | [-0.057, -0.013] | **yes** | 18/48/24 |
| QUIZ - C_ctx: quiz vs context-growth trigger | -0.0303 | [-0.054, -0.007] | **yes** | 28/45/17 |
| QUIZ - C_judge: quiz vs LLM judge (passive-observational) | -0.0371 | [-0.059, -0.016] | **yes** | 22/42/26 |
| QUIZ - B_random: quiz vs random resets | -0.0114 | [-0.033, +0.010] | no | 25/39/26 |
| QUIZ - Z_reground: quiz vs zero-carry trace monitor | -0.0281 | [-0.050, -0.007] | **yes** | 29/41/20 |
| QUIZ - G_dense: quiz vs densest schedule | -0.0258 | [-0.047, -0.005] | **yes** | 21/41/28 |
| QUIZ - A_no_reset: quiz vs never resetting | -0.0100 | [-0.032, +0.012] | no | 27/34/29 |
| QUIZ - F_oracle: quiz vs perfect-timing oracle | -0.0094 | [-0.032, +0.013] | no | 29/36/25 |
| ACT_probe - QUIZ: ACTIVE probe vs passive-behavioural quiz | -0.0209 | [-0.046, +0.004] | no | 31/39/20 |
| ACT_probe - C_clock: active probe vs clock | -0.0556 | [-0.079, -0.032] | **yes** | 11/51/28 |
| ACT_probe - Z_reground: active probe vs zero-carry trace monitor | -0.0490 | [-0.074, -0.025] | **yes** | 24/45/21 |
| ACT_probe - A_no_reset: active probe vs never resetting | -0.0309 | [-0.057, -0.005] | **yes** | 28/45/17 |
| ACT_carry_clock - C_clock: OBSERVER-EFFECT COST: carrying the probe at an identical schedule | -0.0363 | [-0.057, -0.016] | **yes** | 21/36/33 |
| ACT_probe - ACT_carry_clock: timing value of the active signal | -0.0192 | [-0.040, +0.002] | no | 23/43/24 |
| Z_reground - C_clock: zero-carry trace monitor vs clock (anchor) | -0.0066 | [-0.027, +0.014] | no | 31/37/22 |
| Z_reground - A_no_reset: zero-carry vs never resetting (anchor) | +0.0181 | [-0.004, +0.041] | no | 33/29/28 |
| C_clock - A_no_reset: clock vs never resetting (anchor) | +0.0247 | [+0.002, +0.047] | **yes** | 44/21/25 |
| G_dense - A_no_reset: densest schedule vs never resetting (anchor) | +0.0158 | [-0.007, +0.039] | no | 41/28/21 |
| F_oracle - A_no_reset: oracle vs never resetting (anchor) | -0.0006 | [-0.029, +0.026] | no | 38/33/19 |

## Key contrasts per domain

| domain | contrast | delta | 95% CI | sig |
|---|---|---|---|---|
| coding | QUIZ - C_clock | -0.0629 | [-0.106, -0.021] | **yes** |
| registers | QUIZ - C_clock | -0.0380 | [-0.056, -0.020] | **yes** |
| babi | QUIZ - C_clock | -0.0031 | [-0.046, +0.038] | no |
| coding | QUIZ - C_judge | -0.0687 | [-0.114, -0.026] | **yes** |
| registers | QUIZ - C_judge | -0.0376 | [-0.058, -0.018] | **yes** |
| babi | QUIZ - C_judge | -0.0051 | [-0.044, +0.034] | no |
| coding | QUIZ - Z_reground | -0.0394 | [-0.082, +0.001] | no |
| registers | QUIZ - Z_reground | -0.0168 | [-0.036, +0.002] | no |
| babi | QUIZ - Z_reground | -0.0280 | [-0.075, +0.017] | no |
| coding | QUIZ - A_no_reset | -0.0196 | [-0.068, +0.027] | no |
| registers | QUIZ - A_no_reset | -0.0079 | [-0.031, +0.016] | no |
| babi | QUIZ - A_no_reset | -0.0024 | [-0.041, +0.036] | no |
| coding | QUIZ - F_oracle | +0.0038 | [-0.044, +0.053] | no |
| registers | QUIZ - F_oracle | -0.0111 | [-0.034, +0.015] | no |
| babi | QUIZ - F_oracle | -0.0208 | [-0.059, +0.016] | no |
| coding | ACT_probe - QUIZ | -0.0181 | [-0.075, +0.035] | no |
| registers | ACT_probe - QUIZ | -0.0032 | [-0.029, +0.021] | no |
| babi | ACT_probe - QUIZ | -0.0413 | [-0.080, +0.000] | no |
| coding | ACT_probe - C_clock | -0.0810 | [-0.125, -0.041] | **yes** |
| registers | ACT_probe - C_clock | -0.0412 | [-0.068, -0.016] | **yes** |
| babi | ACT_probe - C_clock | -0.0444 | [-0.093, +0.004] | no |
| coding | ACT_carry_clock - C_clock | -0.0509 | [-0.094, -0.010] | **yes** |
| registers | ACT_carry_clock - C_clock | -0.0190 | [-0.038, -0.000] | **yes** |
| babi | ACT_carry_clock - C_clock | -0.0391 | [-0.081, -0.001] | **yes** |

## Cost of observation

The observer-effect accuracy delta is the `ACT_carry_clock - C_clock` contrast above (same trigger, same operator; the only difference is that the agent carries the probe). Monitoring tokens: `quiz tok` column for QUIZ (fork tokens, never in agent context); the judge's calls are folded into C_judge's totals; trace monitors cost 0 by construction.
