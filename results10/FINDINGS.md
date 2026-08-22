# Experiment 10 — Findings

> **Update (Study B1): the Study B0 headline below is operator-specific.** The
> sub-additive penalty it reports is caused by compaction, and vanishes under
> deterministic re-grounding (interaction −0.032 → +0.001). Under re-grounding
> the same carried probe with the same oracle timing yields **+0.039,
> significant** — a carried sentinel *does* pay when recovery is lossless. See
> [`STUDY_B1.md`](STUDY_B1.md). Read Study B0 as "carried probes cannot pay
> **when recovery is lossy self-summarisation**", not as a general claim.

## Study B0 — the observer cost consumes the entire timing prize

**Headline: granting a carried sentinel *perfect* knowledge of when to
intervene leaves it exactly break-even against doing nothing.** The timing
prize is real and significant (+0.059); the net benefit of capturing all of it
with a carried probe is **+0.002, CI [−0.037, +0.040]**.

`gpt-oss-20b`, **100** coding tasks paired across all four cells, experiment-4
regime (self-summary compaction — the operator under which a timing prize
exists at all; under experiment 6's re-grounding the oracle is worthless, so
the interaction is untestable there).

|  | no useful reset | oracle-timed reset |
|---|---|---|
| **no sentinel** | A = 0.849 | B = **0.908** |
| **carries probe** | C = 0.823 | D = 0.851 |

| contrast | meaning | Δ | 95% CI | sig |
|---|---|---|---|---|
| `B − A` | maximum timing value | **+0.0593** | [+0.029, +0.092] | **yes** |
| `C − A` | carrying cost | −0.0257 | [−0.058, +0.007] | no |
| `D − C` | timing value *while carrying the probe* | +0.0275 | [−0.007, +0.062] | no |
| **`D − A`** | **can a perfect carried sentinel pay for itself?** | **+0.0019** | **[−0.037, +0.040]** | no |
| `D − B` | cost of carrying, given oracle timing | **−0.0575** | [−0.090, −0.026] | **yes** |
| `(D−C) − (B−A)` | interaction (0 = additive) | −0.0318 | [−0.075, +0.011] | no |

### Why this had to be run rather than derived

Experiment 4 estimated this by subtraction: prize +0.059 minus carrying cost
−0.043 ⇒ "+0.016, barely pays". **That assumed the two effects are additive,
which was never measured.** They are not:

- additive prediction for `D`: **0.8825**
- observed `D`: **0.8507**
- **additivity error: −0.032**

The interaction is **sub-additive and stable** — −0.032 at n=100, −0.035 at
n=40. Timing is worth **+0.059 without a probe but only +0.027 while carrying
one**. A carried probe does not merely levy a fixed toll; it **degrades the
value of good timing itself**. The mechanism is visible in the harness: at a
reset the agent's self-summary must now also reproduce the probe's ledger
state, so the compaction that was supposed to rescue it has more to get wrong.
Consistently, `D − B` (−0.058, significant) shows carrying costs *more* when
you are also resetting than when you are not (`C − A`, −0.026).

### What can and cannot be claimed

**Can:** the entire measured timing prize is consumed. The point estimate of
`D − A` is +0.002 and the CI excludes any benefit above +0.040 — smaller than
the +0.059 prize a perfect detector was supposed to deliver. Even granting a
carried sentinel oracle-quality knowledge, it captures none of it.

**Cannot:** that carried probes are harmful. `D − A` is a precise null, not a
negative. The n=40 pass estimated −0.020; at n=100 it is +0.002. The honest
statement is *break-even*, not *worse than nothing*.

**Scope:** one probe (`escalating_ledger`), one model, one task family, one
operator regime. A lighter probe would have a smaller carrying cost and might
clear zero — which is exactly the zero-carry direction, where the cost is
structurally absent rather than merely small.

## Study A — the break-even surface (re-analysis, no new API calls)

Descriptive decision analysis, not a causal claim. Scoring every policy from
experiments 5–6 under `U = accuracy − R·restarts − T·prompt_ktokens` and taking
the upper envelope:

- **lossless operator**: the clock wins only while restarts are nearly free
  (R < 0.003); the **zero-carry sentinel owns the large middle region**;
  never-reset takes over once restarts are expensive.
- **lossy operator**: never-reset dominates almost everywhere — the sentinel
  beats the clock, but both lose to not restarting.
- Under a lossless operator the zero-carry sentinel overtakes the clock once a
  restart costs more than **0.0027 accuracy-equivalents**.

See [`STUDY_A.md`](STUDY_A.md) and `figures/fig1_decision_map.png`.

## Where this leaves the project

Three quantities jointly decide whether adaptive monitoring is worth it —
**monitor interference, monitor quality, and intervention fidelity/cost** — and
two are now measured:

| | measured | value |
|---|---|---|
| interference (carried) | ✅ | consumes the whole prize; net +0.002 with perfect timing |
| interference (zero-carry) | ✅ | structurally zero |
| intervention fidelity | partially | two endpoints (exps 4/5 vs 6); the sweep is Study B |
| monitor quality required | ✗ | Study C frontier |

The remaining work is Study B (fidelity as **nested concrete state classes** —
repo state → user constraints → completed subtasks → prior decisions → tool
state — so that φ\* names an engineering decision rather than an arbitrary
percentage) and Study C (the recall × false-alarm frontier, with the carried
sentinel, zero-carry monitor, LLM judge and clock all located on it).
