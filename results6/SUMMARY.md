# Experiment 6 — Sentinel-triggered re-grounding, deployed

Model gpt-oss-20b — 90 tasks paired across every arm (30 coding, 30 registers, 30 babi), full horizon, no early stop.
Reset operator: deterministic re-grounding from the external store (R1)
or verbatim user-log replay (R2). No LLM call at reset time; no probe
carried in any arm. A_no_reset imported verbatim from runs5.

## Arms

| arm | policy | operator | accuracy | success@0.9 | resets/task | prompt tok |
|---|---|---|---|---|---|---|
| A_no_reset | none | — | 0.823 | 0.478 | 0.00 | 22,105 |
| B_random | random | reground | 0.824 | 0.478 | 0.86 | 18,858 |
| C_clock | scheduled | reground | 0.847 | 0.589 | 3.28 | 12,580 |
| C_ctx | ctx_growth | reground | 0.843 | 0.556 | 1.52 | 14,633 |
| C_judge | judge | reground | 0.850 | 0.611 | 4.69 | 29,513 |
| Z_reground | zerocarry | reground | 0.841 | 0.522 | 0.86 | 18,330 |
| F_oracle | oracle | reground | 0.822 | 0.478 | 0.77 | 19,152 |
| G_dense | scheduled | reground | 0.839 | 0.556 | 5.74 | 12,195 |
| Z_replay | zerocarry | replay | 0.799 | 0.433 | 1.12 | 20,535 |
| C_clock_replay | scheduled | replay | 0.724 | 0.278 | 3.28 | 18,916 |
| F_oracle_replay | oracle | replay | 0.774 | 0.389 | 0.77 | 21,351 |

## Per-domain accuracy

| arm | coding | registers | babi |
|---|---|---|---|
| A_no_reset | 0.894 | 0.945 | 0.629 |
| B_random | 0.887 | 0.939 | 0.646 |
| C_clock | 0.938 | 0.975 | 0.630 |
| C_ctx | 0.913 | 0.971 | 0.645 |
| C_judge | 0.943 | 0.975 | 0.632 |
| Z_reground | 0.914 | 0.954 | 0.655 |
| F_oracle | 0.871 | 0.948 | 0.647 |
| G_dense | 0.929 | 0.970 | 0.617 |
| Z_replay | 0.810 | 0.940 | 0.647 |
| C_clock_replay | 0.581 | 0.949 | 0.641 |
| F_oracle_replay | 0.766 | 0.929 | 0.629 |

## Paired contrasts, pooled (bootstrap 95% CI on the per-task delta)

| contrast | delta accuracy | 95% CI | significant | better/worse/tied |
|---|---|---|---|---|
| F_oracle - A_no_reset: GATE: perfect timing vs never resetting | -0.0006 | [-0.029, +0.026] | no | 38/33/19 |
| Z_reground - C_clock: zero-carry reground vs turn-count clock | -0.0066 | [-0.027, +0.014] | no | 31/37/22 |
| Z_reground - C_ctx: zero-carry reground vs context-growth trigger | -0.0022 | [-0.022, +0.018] | no | 34/37/19 |
| Z_reground - C_judge: zero-carry reground vs LLM judge | -0.0090 | [-0.030, +0.012] | no | 32/37/21 |
| Z_reground - B_random: zero-carry reground vs random, budget-matched | +0.0167 | [-0.006, +0.041] | no | 37/35/18 |
| Z_reground - A_no_reset: zero-carry reground vs never resetting | +0.0181 | [-0.004, +0.041] | no | 33/29/28 |
| Z_reground - F_oracle: zero-carry reground vs perfect timing | +0.0187 | [-0.003, +0.043] | no | 38/26/26 |
| G_dense - A_no_reset: densest schedule vs never resetting | +0.0158 | [-0.007, +0.039] | no | 41/28/21 |
| G_dense - F_oracle: densest schedule vs perfect timing | +0.0165 | [-0.005, +0.041] | no | 39/26/25 |
| Z_reground - G_dense: sentinel placement vs densest schedule | +0.0023 | [-0.017, +0.023] | no | 29/39/22 |
| C_clock - A_no_reset: clock reground vs never resetting | +0.0247 | [+0.002, +0.047] | **yes** | 44/21/25 |
| F_oracle - C_clock: perfect timing vs clock | -0.0253 | [-0.050, -0.002] | **yes** | 24/41/25 |
| Z_replay - Z_reground: replay (R2) vs reground (R1), zero-carry | -0.0419 | [-0.071, -0.014] | **yes** | 29/42/19 |
| C_clock_replay - C_clock: replay vs reground, clock | -0.1239 | [-0.166, -0.083] | **yes** | 18/60/12 |
| F_oracle_replay - F_oracle: replay vs reground, oracle | -0.0477 | [-0.080, -0.015] | **yes** | 25/45/20 |
| Z_replay - A_no_reset: zero-carry REPLAY vs never resetting | -0.0238 | [-0.049, +0.001] | no | 26/43/21 |
| Z_replay - C_clock_replay: zero-carry vs clock, both replay | +0.0754 | [+0.041, +0.111] | **yes** | 47/29/14 |

## Operator effect: same trigger, exp-6 operator minus exp-5 compaction

(paired on task across experiments; positive = re-grounding beats
compaction at that trigger)

| contrast | delta | 95% CI | sig |
|---|---|---|---|
| **pooled**: Z_reground(6) - Z_routed(5): zero-carry trigger: reground vs compaction | +0.0240 | [-0.001, +0.050] | no |
| coding: zero-carry trigger: reground vs compaction | +0.0241 | [-0.017, +0.065] | no |
| registers: zero-carry trigger: reground vs compaction | +0.0218 | [-0.008, +0.062] | no |
| babi: zero-carry trigger: reground vs compaction | +0.0260 | [-0.024, +0.075] | no |
| **pooled**: C_clock(6) - C_clock(5): clock trigger: reground vs compaction | +0.0511 | [+0.029, +0.074] | **yes** |
| coding: clock trigger: reground vs compaction | +0.1038 | [+0.057, +0.151] | **yes** |
| registers: clock trigger: reground vs compaction | +0.0195 | [+0.003, +0.037] | **yes** |
| babi: clock trigger: reground vs compaction | +0.0299 | [-0.006, +0.067] | no |
| **pooled**: C_ctx(6) - C_ctx(5): ctx trigger: reground vs compaction | +0.0283 | [+0.003, +0.053] | **yes** |
| coding: ctx trigger: reground vs compaction | +0.0800 | [+0.036, +0.126] | **yes** |
| registers: ctx trigger: reground vs compaction | +0.0067 | [-0.013, +0.027] | no |
| babi: ctx trigger: reground vs compaction | -0.0018 | [-0.053, +0.051] | no |
| **pooled**: C_judge(6) - C_judge(5): judge trigger: reground vs compaction | +0.0509 | [+0.020, +0.082] | **yes** |
| coding: judge trigger: reground vs compaction | +0.1333 | [+0.068, +0.204] | **yes** |
| registers: judge trigger: reground vs compaction | +0.0108 | [-0.005, +0.028] | no |
| babi: judge trigger: reground vs compaction | +0.0087 | [-0.039, +0.056] | no |
| **pooled**: F_oracle(6) - F_oracle(5): oracle trigger: reground vs compaction | +0.0052 | [-0.018, +0.030] | no |
| coding: oracle trigger: reground vs compaction | +0.0151 | [-0.031, +0.066] | no |
| registers: oracle trigger: reground vs compaction | -0.0081 | [-0.029, +0.011] | no |
| babi: oracle trigger: reground vs compaction | +0.0087 | [-0.038, +0.055] | no |

## Key contrasts per domain

| domain | contrast | delta | 95% CI | sig |
|---|---|---|---|---|
| coding | F_oracle - A_no_reset | -0.0235 | [-0.089, +0.036] | no |
| registers | F_oracle - A_no_reset | +0.0032 | [-0.028, +0.029] | no |
| babi | F_oracle - A_no_reset | +0.0183 | [-0.029, +0.064] | no |
| coding | Z_reground - C_clock | -0.0234 | [-0.051, +0.003] | no |
| registers | Z_reground - C_clock | -0.0212 | [-0.041, -0.002] | **yes** |
| babi | Z_reground - C_clock | +0.0249 | [-0.028, +0.074] | no |
| coding | Z_reground - C_ctx | +0.0006 | [-0.026, +0.028] | no |
| registers | Z_reground - C_ctx | -0.0165 | [-0.039, +0.005] | no |
| babi | Z_reground - C_ctx | +0.0092 | [-0.041, +0.058] | no |
| coding | Z_reground - C_judge | -0.0293 | [-0.063, +0.002] | no |
| registers | Z_reground - C_judge | -0.0207 | [-0.043, -0.000] | **yes** |
| babi | Z_reground - C_judge | +0.0229 | [-0.027, +0.072] | no |
| coding | Z_reground - A_no_reset | +0.0198 | [-0.020, +0.060] | no |
| registers | Z_reground - A_no_reset | +0.0089 | [-0.012, +0.031] | no |
| babi | Z_reground - A_no_reset | +0.0256 | [-0.023, +0.076] | no |
| coding | G_dense - A_no_reset | +0.0350 | [-0.008, +0.077] | no |
| registers | G_dense - A_no_reset | +0.0245 | [+0.004, +0.046] | **yes** |
| babi | G_dense - A_no_reset | -0.0120 | [-0.058, +0.037] | no |
| coding | Z_reground - G_dense | -0.0152 | [-0.046, +0.018] | no |
| registers | Z_reground - G_dense | -0.0156 | [-0.033, +0.003] | no |
| babi | Z_reground - G_dense | +0.0376 | [-0.006, +0.081] | no |
| coding | C_clock - A_no_reset | +0.0432 | [+0.004, +0.083] | **yes** |
| registers | C_clock - A_no_reset | +0.0301 | [+0.007, +0.053] | **yes** |
| babi | C_clock - A_no_reset | +0.0007 | [-0.046, +0.050] | no |
| coding | Z_replay - Z_reground | -0.1042 | [-0.164, -0.045] | **yes** |
| registers | Z_replay - Z_reground | -0.0142 | [-0.033, +0.004] | no |
| babi | Z_replay - Z_reground | -0.0072 | [-0.062, +0.045] | no |
