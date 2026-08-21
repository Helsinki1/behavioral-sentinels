# Experiment 7, Study B0 — does a perfect carried sentinel pay?

Model gpt-oss-20b, 100 coding tasks paired across all four cells,
experiment-4 regime (self-summary compaction — the operator under which a
timing prize exists at all). Primary outcome: share of turns with zero errors.

## The 2x2

| | no useful reset | oracle-timed reset |
|---|---|---|
| **no sentinel** | A = 0.849 | B = 0.908 |
| **carries probe** | C = 0.823 | D = 0.851 |

## Contrasts (paired, bootstrap 95% CI)

| contrast | meaning | delta | 95% CI | significant |
|---|---|---|---|---|
| C - A | carrying cost (no reset in either) | -0.0257 | [-0.058, +0.007] | no |
| B - A | maximum timing value (no probe in either) | +0.0593 | [+0.029, +0.092] | **yes** |
| D - C | timing value WHILE carrying the probe | +0.0275 | [-0.007, +0.062] | no |
| D - A | can a PERFECT CARRIED sentinel pay for itself? | +0.0019 | [-0.037, +0.040] | no |
| D - B | cost of carrying, given oracle timing | -0.0575 | [-0.090, -0.026] | **yes** |
| (D-C) - (B-A) | interaction; 0 = additive | -0.0318 | [-0.075, +0.011] | no |

## Additivity check

- additive prediction for D (A + carrying cost + timing prize): **0.8825**
- observed D: **0.8507**
- error: **-0.0318**

