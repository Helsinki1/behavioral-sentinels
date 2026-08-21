# Experiment 9 — active vs passive observation on sharded GSM8K, four models

Sessions of 3 sharded math problems (lost_in_conversation `math`, arXiv:2505.06120), one verbatim shard per turn, WAIT/ANSWER protocol; hallucination = premature ANSWER, missing WAIT, or missing/wrong final ANSWER vs the GSM8K key. R1 reground resets; same arms/policies as exp 8.

## Cross-model headline

| model | A_no_reset | C_clock | ACT_probe | ACT_carry_clock | QUIZ | Z_trace | F_oracle | observer Δ (sig) | quiz prec (shadow≥2) | trace-monitor prec |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 0.883 | 0.941 | 0.977 | 0.962 | 0.893 | 0.919 | 0.921 | +0.021 (y) | 1.000 | 1.000 |
| deepseek-v4-flash | 0.906 | 0.901 | 0.915 | 0.909 | 0.913 | 0.908 | 0.912 | +0.009 (n) | - | - |
| qwen3p7-plus | 0.933 | 0.924 | 0.898 | 0.908 | 0.930 | 0.936 | 0.937 | -0.016 (y) | 1.000 | 1.000 |
| gpt-4o-mini | 0.845 | 0.859 | 0.862 | 0.854 | 0.848 | 0.854 | 0.841 | -0.005 (n) | - | - |

## gpt-oss-120b — 34 sessions paired

| arm | category | accuracy | success@0.9 | resets/task | prompt tok | quiz tok |
|---|---|---|---|---|---|---|
| A_no_reset | bound | 0.883 | 0.324 | 0.00 | 8,787 | 0 |
| C_clock | baseline | 0.941 | 0.647 | 2.41 | 6,994 | 0 |
| ACT_probe | active | 0.977 | 0.941 | 5.94 | 10,809 | 0 |
| ACT_carry_clock | active | 0.962 | 0.882 | 2.41 | 11,441 | 0 |
| QUIZ | passive-behavioural | 0.893 | 0.500 | 0.15 | 8,503 | 3,811 |
| Z_trace | passive-observational | 0.919 | 0.588 | 1.21 | 7,570 | 0 |
| F_oracle | bound | 0.921 | 0.676 | 0.97 | 8,624 | 0 |

| contrast | delta | 95% CI | sig | better/worse/tied |
|---|---|---|---|---|
| QUIZ - C_clock: frozen-state QUIZ vs clock | -0.0477 | [-0.072, -0.024] | **yes** | 5/21/8 |
| QUIZ - Z_trace: quiz vs zero-carry trace monitor | -0.0259 | [-0.050, -0.002] | **yes** | 8/16/10 |
| QUIZ - A_no_reset: quiz vs never resetting | +0.0102 | [-0.012, +0.031] | no | 14/8/12 |
| QUIZ - F_oracle: quiz vs oracle | -0.0274 | [-0.051, -0.005] | **yes** | 7/16/11 |
| ACT_probe - QUIZ: ACTIVE probe vs passive quiz | +0.0831 | [+0.063, +0.105] | **yes** | 29/1/4 |
| ACT_probe - C_clock: active probe vs clock | +0.0354 | [+0.015, +0.058] | **yes** | 16/3/15 |
| ACT_probe - A_no_reset: active probe vs never resetting | +0.0934 | [+0.073, +0.114] | **yes** | 29/0/5 |
| ACT_carry_clock - C_clock: OBSERVER-EFFECT COST | +0.0210 | [+0.003, +0.039] | **yes** | 13/6/15 |
| ACT_probe - ACT_carry_clock: timing value of the active signal | +0.0145 | [-0.001, +0.031] | no | 10/4/20 |
| Z_trace - C_clock: trace monitor vs clock | -0.0219 | [-0.042, -0.001] | **yes** | 6/16/12 |
| Z_trace - A_no_reset: trace monitor vs never resetting | +0.0361 | [+0.017, +0.055] | **yes** | 18/4/12 |
| C_clock - A_no_reset: clock vs never resetting | +0.0579 | [+0.034, +0.082] | **yes** | 21/4/9 |
| F_oracle - A_no_reset: oracle vs never resetting | +0.0376 | [+0.013, +0.063] | **yes** | 21/9/4 |

## deepseek-v4-flash — 34 sessions paired

| arm | category | accuracy | success@0.9 | resets/task | prompt tok | quiz tok |
|---|---|---|---|---|---|---|
| A_no_reset | bound | 0.906 | 0.471 | 0.00 | 7,063 | 0 |
| C_clock | baseline | 0.901 | 0.441 | 2.41 | 5,982 | 0 |
| ACT_probe | active | 0.915 | 0.647 | 5.88 | 9,990 | 0 |
| ACT_carry_clock | active | 0.909 | 0.500 | 2.41 | 10,352 | 0 |
| QUIZ | passive-behavioural | 0.913 | 0.500 | 0.00 | 7,063 | 2,915 |
| Z_trace | passive-observational | 0.908 | 0.500 | 0.00 | 7,063 | 0 |
| F_oracle | bound | 0.912 | 0.471 | 0.88 | 7,081 | 0 |

| contrast | delta | 95% CI | sig | better/worse/tied |
|---|---|---|---|---|
| QUIZ - C_clock: frozen-state QUIZ vs clock | +0.0124 | [-0.003, +0.028] | no | 10/4/20 |
| QUIZ - Z_trace: quiz vs zero-carry trace monitor | +0.0051 | [-0.004, +0.016] | no | 3/1/30 |
| QUIZ - A_no_reset: quiz vs never resetting | +0.0070 | [-0.004, +0.020] | no | 4/2/28 |
| QUIZ - F_oracle: quiz vs oracle | +0.0009 | [-0.011, +0.013] | no | 5/5/24 |
| ACT_probe - QUIZ: ACTIVE probe vs passive quiz | +0.0015 | [-0.015, +0.017] | no | 10/9/15 |
| ACT_probe - C_clock: active probe vs clock | +0.0139 | [-0.000, +0.028] | no | 12/5/17 |
| ACT_probe - A_no_reset: active probe vs never resetting | +0.0085 | [-0.008, +0.026] | no | 10/8/16 |
| ACT_carry_clock - C_clock: OBSERVER-EFFECT COST | +0.0085 | [-0.002, +0.019] | no | 7/2/25 |
| ACT_probe - ACT_carry_clock: timing value of the active signal | +0.0054 | [-0.006, +0.017] | no | 8/5/21 |
| Z_trace - C_clock: trace monitor vs clock | +0.0073 | [-0.005, +0.020] | no | 8/4/22 |
| Z_trace - A_no_reset: trace monitor vs never resetting | +0.0020 | [-0.006, +0.010] | no | 3/2/29 |
| C_clock - A_no_reset: clock vs never resetting | -0.0053 | [-0.017, +0.007] | no | 4/8/22 |
| F_oracle - A_no_reset: oracle vs never resetting | +0.0062 | [-0.005, +0.019] | no | 5/3/26 |

## qwen3p7-plus — 34 sessions paired

| arm | category | accuracy | success@0.9 | resets/task | prompt tok | quiz tok |
|---|---|---|---|---|---|---|
| A_no_reset | bound | 0.933 | 0.735 | 0.00 | 8,262 | 0 |
| C_clock | baseline | 0.924 | 0.647 | 2.41 | 6,635 | 0 |
| ACT_probe | active | 0.898 | 0.412 | 6.00 | 10,920 | 0 |
| ACT_carry_clock | active | 0.908 | 0.500 | 2.41 | 11,393 | 0 |
| QUIZ | passive-behavioural | 0.930 | 0.676 | 0.00 | 8,262 | 3,299 |
| Z_trace | passive-observational | 0.936 | 0.706 | 0.03 | 8,209 | 0 |
| F_oracle | bound | 0.937 | 0.765 | 0.79 | 7,933 | 0 |

| contrast | delta | 95% CI | sig | better/worse/tied |
|---|---|---|---|---|
| QUIZ - C_clock: frozen-state QUIZ vs clock | +0.0062 | [-0.006, +0.018] | no | 9/5/20 |
| QUIZ - Z_trace: quiz vs zero-carry trace monitor | -0.0058 | [-0.018, +0.005] | no | 3/5/26 |
| QUIZ - A_no_reset: quiz vs never resetting | -0.0023 | [-0.011, +0.006] | no | 3/4/27 |
| QUIZ - F_oracle: quiz vs oracle | -0.0068 | [-0.020, +0.006] | no | 5/8/21 |
| ACT_probe - QUIZ: ACTIVE probe vs passive quiz | -0.0326 | [-0.048, -0.018] | **yes** | 1/16/17 |
| ACT_probe - C_clock: active probe vs clock | -0.0264 | [-0.040, -0.013] | **yes** | 1/14/19 |
| ACT_probe - A_no_reset: active probe vs never resetting | -0.0350 | [-0.049, -0.022] | **yes** | 0/16/18 |
| ACT_carry_clock - C_clock: OBSERVER-EFFECT COST | -0.0161 | [-0.028, -0.005] | **yes** | 1/9/24 |
| ACT_probe - ACT_carry_clock: timing value of the active signal | -0.0103 | [-0.025, +0.005] | no | 4/10/20 |
| Z_trace - C_clock: trace monitor vs clock | +0.0120 | [-0.001, +0.026] | no | 10/4/20 |
| Z_trace - A_no_reset: trace monitor vs never resetting | +0.0035 | [-0.005, +0.012] | no | 4/2/28 |
| C_clock - A_no_reset: clock vs never resetting | -0.0086 | [-0.021, +0.005] | no | 5/10/19 |
| F_oracle - A_no_reset: oracle vs never resetting | +0.0045 | [-0.007, +0.017] | no | 6/4/24 |

## gpt-4o-mini — 34 sessions paired

| arm | category | accuracy | success@0.9 | resets/task | prompt tok | quiz tok |
|---|---|---|---|---|---|---|
| A_no_reset | bound | 0.845 | 0.118 | 0.00 | 7,247 | 0 |
| C_clock | baseline | 0.859 | 0.118 | 2.41 | 6,032 | 0 |
| ACT_probe | active | 0.862 | 0.118 | 6.00 | 9,754 | 0 |
| ACT_carry_clock | active | 0.854 | 0.118 | 2.41 | 10,183 | 0 |
| QUIZ | passive-behavioural | 0.848 | 0.118 | 0.00 | 7,246 | 2,967 |
| Z_trace | passive-observational | 0.854 | 0.118 | 0.15 | 7,173 | 0 |
| F_oracle | bound | 0.841 | 0.118 | 1.00 | 7,365 | 0 |

| contrast | delta | 95% CI | sig | better/worse/tied |
|---|---|---|---|---|
| QUIZ - C_clock: frozen-state QUIZ vs clock | -0.0111 | [-0.029, +0.005] | no | 5/8/21 |
| QUIZ - Z_trace: quiz vs zero-carry trace monitor | -0.0064 | [-0.018, +0.004] | no | 2/5/27 |
| QUIZ - A_no_reset: quiz vs never resetting | +0.0029 | [-0.006, +0.014] | no | 2/2/30 |
| QUIZ - F_oracle: quiz vs oracle | +0.0070 | [-0.005, +0.019] | no | 8/5/21 |
| ACT_probe - QUIZ: ACTIVE probe vs passive quiz | +0.0147 | [-0.004, +0.035] | no | 8/5/21 |
| ACT_probe - C_clock: active probe vs clock | +0.0036 | [-0.005, +0.013] | no | 5/3/26 |
| ACT_probe - A_no_reset: active probe vs never resetting | +0.0176 | [-0.000, +0.037] | no | 8/4/22 |
| ACT_carry_clock - C_clock: OBSERVER-EFFECT COST | -0.0047 | [-0.015, +0.005] | no | 3/6/25 |
| ACT_probe - ACT_carry_clock: timing value of the active signal | +0.0083 | [-0.002, +0.018] | no | 7/2/25 |
| Z_trace - C_clock: trace monitor vs clock | -0.0047 | [-0.017, +0.007] | no | 4/6/24 |
| Z_trace - A_no_reset: trace monitor vs never resetting | +0.0092 | [+0.002, +0.019] | **yes** | 4/0/30 |
| C_clock - A_no_reset: clock vs never resetting | +0.0139 | [-0.002, +0.031] | no | 8/4/22 |
| F_oracle - A_no_reset: oracle vs never resetting | -0.0041 | [-0.016, +0.008] | no | 5/7/22 |
