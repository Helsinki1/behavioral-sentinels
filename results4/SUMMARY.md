# Experiment 4 — Sentinel-triggered resets vs a clock

Model gpt-oss-20b - 40 tasks, paired across every arm - full horizon, no early stop.
Primary outcome: share of turns with zero errors. Compaction = the agent's own
state self-summary (no ground truth injected), identical in every resetting arm.

| arm | policy | carries probe | accuracy | success@0.9 | resets/task | prompt tok |
|---|---|---|---|---|---|---|
| A_no_reset | none | no | 0.854 | 0.475 | 0.00 | 30,434 |
| B_random | random | no | 0.811 | 0.375 | 4.10 | 19,988 |
| C_scheduled | scheduled | no | 0.854 | 0.575 | 3.10 | 18,267 |
| C_prime_carried | scheduled | yes | 0.811 | 0.375 | 3.08 | 21,208 |
| D_sentinel | sentinel | yes | 0.792 | 0.400 | 4.15 | 23,809 |
| F_oracle | oracle | no | 0.912 | 0.700 | 0.78 | 27,065 |

## Paired contrasts (bootstrap 95% CI on the per-task delta)

| contrast | delta accuracy | 95% CI | significant | better/worse/tied |
|---|---|---|---|---|
| D_sentinel - C_prime_carried: timing value (both carry probe) | -0.0191 | [-0.081, +0.047] | no | 15/19/6 |
| C_prime_carried - C_scheduled: observer-effect cost | -0.0426 | [-0.092, +0.008] | no | 10/22/8 |
| D_sentinel - C_scheduled: sentinel vs plain clock | -0.0617 | [-0.128, +0.007] | no | 11/21/8 |
| D_sentinel - B_random: sentinel vs random, budget-matched | -0.0193 | [-0.076, +0.038] | no | 13/17/10 |
| C_scheduled - A_no_reset: does resetting help at all | -0.0003 | [-0.056, +0.055] | no | 18/16/6 |
| F_oracle - C_scheduled: PERFECT predictor vs clock (headroom) | +0.0587 | [+0.012, +0.108] | **yes** | 20/11/9 |
| F_oracle - A_no_reset: perfect predictor vs no reset | +0.0583 | [+0.014, +0.106] | **yes** | 24/9/7 |
