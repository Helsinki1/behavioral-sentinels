# Experiment 11 — Matched vs mismatched probes (PRE-REGISTERED)

*This file was written and committed **before** the mismatched arms were run.
The predictions below are therefore predictions, not descriptions.*

## The claim under test

Experiment 9 observed that carrying a probe **helped** on `gpt-oss-120b` +
sharded math (+0.021) and **hurt** on `gpt-oss-20b` + synthetic tasks (−0.036),
and explained the difference by *probe–task capability match*: `lag_span` forces
re-attention to messages 1/3/6 turns back, which is exactly the capacity sharded
math destroys, so the chore acted as an attention refresh rather than
interference.

That explanation was constructed **after** seeing the result. Experiment 11
turns it into a prediction and tries to falsify it.

> **H1 — the matching hypothesis.** The sign and size of a carried probe's
> effect on task accuracy depend on whether the probe exercises the capacity
> the task is losing. A *matched* probe costs less (or helps); a *mismatched*
> probe costs more.

## Design — 2 regimes × 3 carried conditions, everything else held fixed

The reset policy, operator, schedule, pool and model are **identical in every
cell**. The only thing that varies is which chore (if any) the agent carries.
That isolates Δ_carry, exactly as experiment 10's `C − A` did, but now crossed
with regime.

| | **coding** (evolving-state degradation) | **babi** (retrieval-distance degradation) |
|---|---|---|
| no probe | `C_clock` *(reuse runs6)* | `C_clock` *(reuse runs6)* |
| `staircase` (evolving state) | **matched** — `ACT_carry_clock` *(reuse runs8)* | **mismatched** — `MM_carry_clock` *(new)* |
| `lag_span` (retrieval distance) | **mismatched** — `MM_carry_clock` *(new)* | **matched** — `ACT_carry_clock` *(reuse runs8)* |

- **Probes.** `staircase` demands maintenance of an evolving irrelevant ledger;
  `lag_span` demands recall of tickets 1/3/6 messages back. The match
  assignment is exp 5's `INTENDED_GENRE` mapping (`coding →
  ARTIFACT_ACCUMULATION → staircase`, `babi → RETRIEVAL_DISTANCE → lag_span`),
  fixed in that experiment long before this one.
- **Everything held fixed:** `gpt-oss-20b`, exp-5/6 pool (30 tasks/domain),
  R1 re-grounding operator, clock reset every 6 turns, 6-reset cap, full
  horizon, no early stop, per-turn accuracy.
- Only the two mismatched cells are new runs (60 trajectories); the matched and
  no-probe cells are read verbatim from `runs8` and `runs6`.

## Why the reset policy is a *clock* and not the probe

If the probe also triggered the resets, Δ would confound "carrying this probe"
with "resetting at the moments this probe fires". Holding the schedule fixed at
every-6-turns makes the contrast purely about carrying. This is the
`ACT_carry_clock − C_clock` design from exp 8, generalised to a second probe.

## Pre-registered predictions

Let Δ_carry(probe, domain) = accuracy(probe arm) − accuracy(`C_clock`), paired
by task.

- **P1 (primary).** Within each domain, the matched probe costs less than the
  mismatched one:
  Δ(staircase, coding) > Δ(lag_span, coding) **and**
  Δ(lag_span, babi) > Δ(staircase, babi).
- **P2 (interaction).** The domain × probe interaction is positive:
  [Δ(staircase, coding) − Δ(lag_span, coding)] −
  [Δ(staircase, babi) − Δ(lag_span, babi)] > 0.
  This is the single number H1 predicts; it is sign-free of any per-domain
  baseline difference.
- **P3 (weak, directional).** At least one matched cell is ≥ 0 — i.e. a matched
  probe is not merely cheaper but can be free or beneficial, as exp 9 found on
  120b.

**What would falsify H1:** P2 ≈ 0 or negative — the probes cost the same
regardless of which task they are attached to, meaning the exp-9 explanation
was post-hoc storytelling and the observer effect is a property of the probe
alone.

## Screening (experiment-3 style inclusion criterion, applied first)

Only domains that actually degrade are eligible; a probe cannot predict or
mitigate a failure that does not happen. From `runs5/A_no_reset` (no-reset arm,
so the screen is uncontaminated by any intervention):

| domain | no-reset accuracy | eligible |
|---|---|---|
| babi | 0.629 | **yes** — ample failure |
| coding | 0.894 | **yes** — degrades |
| registers | 0.945 | **no** — near-saturated, little room to move |

`registers` is excluded on this criterion, which is also why the design is
2 regimes rather than 3.

## Analysis plan (fixed in advance)

Paired bootstrap CIs on per-task deltas, as in experiments 4/10. Primary
statistic is P2, the interaction. Per-domain Δ values are reported alongside
but are secondary. No other contrast is treated as confirmatory.
