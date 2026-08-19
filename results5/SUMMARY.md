# Experiment 5 — Task-conditioned sentinel routing, deployed

Model gpt-oss-20b — 90 tasks paired across every arm (30 coding, 30 registers, 30 babi), full horizon, no early stop.
Primary outcome: share of turns with zero errors. Compaction = the agent's own
validated structured state snapshot (no ground truth injected), identical in
every resetting arm.

## Router

- **coding**: 27/30 tasks routed to the intended genre; probes assigned: chain_checksum x3, staircase x27
- **registers**: 30/30 tasks routed to the intended genre; probes assigned: chain_checksum x30
- **babi**: 24/30 tasks routed to the intended genre; probes assigned: chain_checksum x2, lag_span x24, staircase x4

## Arms

| arm | policy | probe | accuracy | success@0.9 | resets/task | prompt tok |
|---|---|---|---|---|---|---|
| A_no_reset | none | none | 0.823 | 0.478 | 0.00 | 22,105 |
| B_random | random | none | 0.806 | 0.400 | 4.28 | 21,094 |
| C_clock | scheduled | none | 0.796 | 0.389 | 3.27 | 18,955 |
| C_ctx | ctx_growth | none | 0.815 | 0.444 | 1.51 | 18,117 |
| C_judge | judge | none | 0.799 | 0.478 | 4.57 | 37,123 |
| C_prime_routed | scheduled | routed | 0.778 | 0.289 | 3.27 | 23,327 |
| D_routed | probe | routed | 0.791 | 0.400 | 4.30 | 26,364 |
| D_blanket | probe | blanket | 0.774 | 0.333 | 4.87 | 24,558 |
| D_rotated | probe | rotated | 0.780 | 0.367 | 4.94 | 26,234 |
| Z_routed | zerocarry | none | 0.817 | 0.478 | 1.09 | 20,508 |
| F_oracle | oracle | none | 0.817 | 0.478 | 0.76 | 21,165 |

## Per-domain accuracy

| arm | coding | registers | babi |
|---|---|---|---|
| A_no_reset | 0.894 | 0.945 | 0.629 |
| B_random | 0.843 | 0.950 | 0.624 |
| C_clock | 0.834 | 0.956 | 0.600 |
| C_ctx | 0.834 | 0.964 | 0.647 |
| C_judge | 0.810 | 0.964 | 0.623 |
| C_prime_routed | 0.779 | 0.917 | 0.639 |
| D_routed | 0.813 | 0.929 | 0.630 |
| D_blanket | 0.844 | 0.901 | 0.576 |
| D_rotated | 0.842 | 0.916 | 0.583 |
| Z_routed | 0.890 | 0.932 | 0.629 |
| F_oracle | 0.856 | 0.956 | 0.639 |

## Paired contrasts, pooled (bootstrap 95% CI on the per-task delta)

| contrast | delta accuracy | 95% CI | significant | better/worse/tied |
|---|---|---|---|---|
| D_routed - C_clock: routed sentinel vs turn-count clock | -0.0055 | [-0.035, +0.024] | no | 35/36/19 |
| D_routed - C_ctx: routed sentinel vs context-growth trigger | -0.0240 | [-0.051, +0.003] | no | 29/41/20 |
| D_routed - C_judge: routed sentinel vs LLM judge | -0.0081 | [-0.042, +0.026] | no | 30/41/19 |
| D_routed - B_random: routed sentinel vs random, budget-matched | -0.0148 | [-0.044, +0.013] | no | 36/36/18 |
| Z_routed - C_clock: ZERO-CARRY routed vs turn-count clock | +0.0205 | [-0.008, +0.050] | no | 43/30/17 |
| Z_routed - C_ctx: zero-carry routed vs context-growth trigger | +0.0021 | [-0.026, +0.031] | no | 27/40/23 |
| Z_routed - C_judge: zero-carry routed vs LLM judge | +0.0179 | [-0.011, +0.048] | no | 34/36/20 |
| Z_routed - A_no_reset: zero-carry routed vs never resetting | -0.0059 | [-0.031, +0.019] | no | 31/35/24 |
| D_routed - D_blanket: ROUTING vs blanket probe (exp-1..4 approach) | +0.0172 | [-0.011, +0.046] | no | 45/34/11 |
| D_routed - D_rotated: ROUTING vs mis-assigned probes (anti-routing) | +0.0107 | [-0.019, +0.040] | no | 39/34/17 |
| C_prime_routed - C_clock: carrying cost of the routed probe | -0.0182 | [-0.049, +0.010] | no | 34/37/19 |
| D_routed - C_prime_routed: timing value of the routed probe | +0.0127 | [-0.021, +0.047] | no | 41/32/17 |
| C_clock - A_no_reset: does clock resetting help at all | -0.0264 | [-0.053, -0.001] | **yes** | 26/42/22 |
| F_oracle - C_clock: PERFECT predictor vs clock (headroom) | +0.0206 | [-0.004, +0.044] | no | 42/26/22 |
| F_oracle - A_no_reset: perfect predictor vs no reset | -0.0058 | [-0.032, +0.019] | no | 36/32/22 |

## Key contrasts per domain

| domain | contrast | delta | 95% CI | sig |
|---|---|---|---|---|
| coding | D_routed - C_clock | -0.0211 | [-0.087, +0.046] | no |
| registers | D_routed - C_clock | -0.0262 | [-0.060, +0.004] | no |
| babi | D_routed - C_clock | +0.0307 | [-0.013, +0.075] | no |
| coding | Z_routed - C_clock | +0.0563 | [+0.000, +0.116] | **yes** |
| registers | Z_routed - C_clock | -0.0234 | [-0.066, +0.008] | no |
| babi | Z_routed - C_clock | +0.0288 | [-0.017, +0.074] | no |
| coding | Z_routed - C_judge | +0.0799 | [+0.012, +0.152] | **yes** |
| registers | Z_routed - C_judge | -0.0317 | [-0.056, -0.011] | **yes** |
| babi | Z_routed - C_judge | +0.0056 | [-0.037, +0.047] | no |
| coding | D_routed - D_blanket | -0.0311 | [-0.092, +0.029] | no |
| registers | D_routed - D_blanket | +0.0280 | [-0.009, +0.064] | no |
| babi | D_routed - D_blanket | +0.0548 | [+0.013, +0.100] | **yes** |
| coding | D_routed - D_rotated | -0.0296 | [-0.097, +0.039] | no |
| registers | D_routed - D_rotated | +0.0136 | [-0.030, +0.056] | no |
| babi | D_routed - D_rotated | +0.0479 | [+0.014, +0.086] | **yes** |
| coding | C_prime_routed - C_clock | -0.0551 | [-0.131, +0.010] | no |
| registers | C_prime_routed - C_clock | -0.0384 | [-0.066, -0.012] | **yes** |
| babi | C_prime_routed - C_clock | +0.0390 | [-0.002, +0.080] | no |
| coding | F_oracle - C_clock | +0.0221 | [-0.036, +0.079] | no |
| registers | F_oracle - C_clock | +0.0007 | [-0.021, +0.021] | no |
| babi | F_oracle - C_clock | +0.0389 | [+0.001, +0.076] | **yes** |
