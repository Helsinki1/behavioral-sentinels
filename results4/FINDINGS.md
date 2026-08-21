# Experiment 4 — Findings

**Headline: H3, with a twist that keeps the project alive.** The sentinel does
not earn its keep — `D_sentinel` scores *below* the plain clock (0.792 vs
0.854 accuracy) despite spending more resets. But the oracle arm shows the
headroom is real and statistically solid: a **perfect** predictor beats the
clock by **+0.059 accuracy (95% CI [+0.012, +0.108])** while using **four times
fewer resets**. The information is worth having; `escalating_ledger` just
isn't good enough to extract it.

Model `gpt-oss-20b`, 40 coding tasks paired across all six arms, full horizon
(no early stop), compaction = the agent's own state self-summary with no
ground truth injected.

## The arms

| arm | policy | carries probe | accuracy | success@0.9 | resets/task | prompt tokens |
|---|---|---|---|---|---|---|
| A_no_reset | none | no | 0.854 | 0.475 | 0.00 | 30,434 |
| B_random | random | no | 0.811 | 0.375 | 4.10 | 19,988 |
| C_scheduled | scheduled | no | **0.854** | 0.575 | 3.10 | 18,267 |
| C_prime_carried | scheduled | **yes** | 0.811 | 0.375 | 3.08 | 21,208 |
| D_sentinel | sentinel | **yes** | 0.792 | 0.400 | 4.15 | 23,809 |
| F_oracle | oracle | no | **0.912** | **0.700** | **0.78** | 27,065 |

## The contrasts that matter

| contrast | Δ accuracy | 95% CI | significant |
|---|---|---|---|
| **F_oracle − C_scheduled** — headroom for a perfect signal | **+0.059** | [+0.012, +0.108] | **yes** |
| **F_oracle − A_no_reset** — perfect signal vs never resetting | **+0.058** | [+0.014, +0.106] | **yes** |
| D_sentinel − C_prime_carried — *value of the timing information* | −0.019 | [−0.081, +0.047] | no |
| C_prime_carried − C_scheduled — *cost of carrying the probe* | −0.043 | [−0.092, +0.008] | no |
| D_sentinel − C_scheduled — sentinel vs plain clock | −0.062 | [−0.128, +0.007] | no |
| D_sentinel − B_random — sentinel vs random, budget-matched | −0.019 | [−0.076, +0.038] | no |
| C_scheduled − A_no_reset — does resetting help at all | −0.000 | [−0.056, +0.055] | no |

### 1. The sentinel's timing information is worth nothing here

This is what the `C_prime_carried` arm was added to measure, and it settles the
ambiguity the original plan would have left. Decomposing the sentinel's −0.062
deficit against the clock:

- **−0.043** is the *observer-effect cost* of carrying the probe at all
  (`C' − C`) — two arms that reset on the identical schedule, differing only in
  whether the agent is also maintaining a `LEDGER:` line.
- **−0.019** is the *timing information itself* (`D − C'`) — the same probe,
  the same load, differing only in whether its failures drive the resets.

So roughly **two-thirds of the sentinel's deficit is the price of carrying it,
and one-third is that its timing is no better than a clock's.** Neither
component is individually significant at n=40, but both point the same way and
they replicate the observer effect that experiment 3 found independently.

### 2. Resetting on a schedule is free accuracy but a large cost saving

`C_scheduled − A_no_reset` is **−0.0003** — 3.1 resets per task buy exactly
nothing in accuracy. But scheduled compaction cuts prompt tokens **40%**
(30,434 → 18,267) and lifts the share of trajectories that never err at all
from 9/40 to 15/40. Compaction is worth doing for cost; it is not worth doing
for accuracy unless it is *well timed*.

### 3. Well-timed resets are what actually work

The oracle result is the one that should shape what comes next. It resets
**0.78 times per task** — a quarter of the clock's budget — and is the only arm
that beats everything else on both accuracy (0.912) and binary success (0.700
vs 0.575). Combined with (2), the mechanism is clear:

> Resets are not intrinsically good. A reset placed just before the state
> corrupts preserves a correct state; a reset placed anywhere else mostly
> churns context. Frequency is not the lever — **placement is.**

That is a genuinely encouraging result for the research direction, and it is
the first quantitative evidence in this project that early warning has any
cash value at all. It also explains why the clock does so well in the
prediction benchmarks and so little here: firing often enough to be "recalled"
is cheap, but landing on the right turn is not.

## What this means for the project

- **H3 is the outcome**: this sentinel loses to a schedule. `escalating_ledger`
  should not be deployed as a compaction trigger.
- **But the direction is not dead**: there is a significant +0.059 accuracy
  gap available to a sufficiently good predictor, at a *quarter* of the
  intervention budget. The task is now to close the gap between the sentinel
  (F1 ≈ 0.42 as a predictor) and the oracle, not to test more probes for
  prediction skill.
- **The probe may need to get cheaper rather than smarter — but this is not
  yet established.** Two-thirds of the loss is carrying cost, and a heavier
  probe plausibly loses more (consistent with experiment 3's ensembles). The
  tempting next step is to subtract: prize +0.059 minus carrying cost −0.043
  leaves ≈ +0.016, "so even a perfect carried probe barely pays". **That
  subtraction assumes additivity and is not measured here.** The two effects
  could interact: carrying a probe might change *where* the failure lands, or
  a well-timed reset might partly refund the probe's cost.

  Experiment 7 completes the 2×2 to settle it — `P_carry_noreset` (carry, no
  reset) and `P_carry_oracle` (carry, oracle-timed reset) alongside the
  existing `A_no_reset` and `F_oracle`. `D − C` gives the timing value *while
  carrying the probe*, and `D − A` answers directly whether a perfect carried
  sentinel can pay for itself. Until then the additive estimate should be read
  as a hypothesis.

  The zero-carry direction — a monitor read off output the agent already
  produces — is motivated regardless, since it has no carrying cost to trade
  against at all.

## Limitations

1. **n = 40, and no sentinel contrast is significant.** All four sentinel-
   related CIs cross zero. The direction is consistent across four independent
   contrasts, but this experiment can rule out a large sentinel benefit, not a
   small one. n≈200 with 3 seeds would be needed to resolve ±0.02.
2. **One model, one task family, one probe.** `gpt-oss-20b` on the exp-2
   coding tasks with `escalating_ledger`. Experiment 2 showed probe difficulty
   does not transfer across models, so the sentinel arm in particular may look
   different on `gpt-4o-mini`.
3. **One schedule, not a swept frontier.** `C_scheduled` is a single point
   (first reset turn 6, every 6 thereafter), not a tuned curve. A better-tuned
   clock would only widen its lead over the sentinel, but the honest tradeoff
   plot needs the schedule swept over several budgets.
4. **The oracle is optimistic by construction.** It resets immediately before
   the turn that failed *in the no-reset arm*; once a trajectory diverges, that
   turn is no longer guaranteed to be the failure point. It is an upper bound
   on placement value, which is exactly what it is used as here — not an
   achievable target.
5. **The compaction operator is one choice among several.** Self-summary was
   chosen because reconstructing from the user's instruction history would
   perfectly restore the true state and make every reset policy look identical
   and excellent. A harness with an authoritative external state (real files on
   disk) would behave differently, and probably better.
