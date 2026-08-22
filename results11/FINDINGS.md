# Experiment 11 — Findings: matched vs mismatched probes

**Verdict: H1 is directionally supported in both domains but is NOT confirmed
at this sample size.** The matched probe was cheaper than the mismatched one in
2/2 domains and the pre-registered interaction is positive (**P2 = +0.027**),
but every interval crosses zero. **P3 is falsified outright:** on this model
every probe — matched or not — cost significant accuracy. Matching makes a probe
*cheaper*, not *free*.

Pre-registered in [`README_EXPERIMENT11.md`](../README_EXPERIMENT11.md) and
committed before these runs. `gpt-oss-20b`, R1 re-grounding, clock reset every
6 turns, full horizon. Policy, operator, schedule, pool and model identical in
every cell — only the carried chore varies. `registers` excluded by the
degradation screen (0.945 no-reset accuracy, near-saturated).

| domain | no probe | matched | Δ matched | mismatched | Δ mismatched | matched − mismatched |
|---|---|---|---|---|---|---|
| **coding** (n=30) | 0.938 | `staircase` 0.887 | **−0.051** ✱ | `lag_span` 0.881 | **−0.056** ✱ | +0.005 [−0.028, +0.037] |
| **babi** (n=30) | 0.630 | `lag_span` 0.591 | **−0.039** ✱ | `staircase` 0.569 | **−0.061** ✱ | +0.022 [−0.012, +0.055] |

✱ = 95% bootstrap CI excludes zero.

**P2 (pre-registered primary) = +0.0272, CI [−0.021, +0.073] — not significant.**

## Reading it honestly

**P1 — matched cheaper than mismatched.** True in both domains (+0.005 coding,
+0.022 babi), and the effect is larger in `babi`, the domain with far more room
to move (0.630 vs 0.938 baseline). Two out of two in the predicted direction is
weak evidence, not none — but neither per-domain difference is individually
significant.

**P2 — the interaction.** +0.027, right sign, interval spans zero. At n=30 per
domain the CI half-width is ~0.047, so this design can detect an interaction of
roughly 0.05 and the observed effect is about half that. **This is an
underpowered test, not a null result** — it neither confirms nor refutes H1.
Resolving +0.027 would need roughly 3× the tasks per domain (n≈90).

**P3 — falsified.** The prediction was that at least one matched cell would be
≥ 0, as experiment 9 found on `gpt-oss-120b` (+0.021). Nothing close: matched
probes cost −0.039 and −0.051, both significant. On this model a matched probe
is still a real tax.

## What this does to the experiment-9 story

Exp 9's explanation was that a matched probe stops being interference and
becomes an *attention refresh*. These data are consistent with a weaker version
of that:

> Matching reduces the carrying cost. Whether it reduces it **past zero**
> appears to depend on the model, not just the match.

Exp 9 saw a sign flip on `gpt-oss-120b`; here on `gpt-oss-20b` the cost shrinks
but stays firmly negative. That is a coherent joint reading — capability
determines whether a matched chore is affordable enough to become net helpful —
but it is now an explicit hypothesis for the cross-model run, not something
these data establish.

## Limitations

- **Power is the binding constraint.** n=30/domain is what the exp-5/6 pool
  provides; the pre-registered primary statistic needs ~3× that.
- **Two probes, two domains, one model.** The matched/mismatched assignment is
  exp 5's `INTENDED_GENRE` mapping, fixed independently and long before this
  experiment, so it is not tuned to this result — but it is one mapping.
- **`babi` and `coding` differ in baseline accuracy by 0.31**, so per-domain Δ
  values are not directly comparable. That is exactly why P2 (a
  difference-of-differences) was pre-registered as primary rather than the
  per-domain deltas.
- An analysis bug was caught before reporting: P2 was defined by *probe
  identity*, but the matched probe differs between domains, so a
  matched-minus-mismatched implementation flips its sign in `babi`. The first
  computation returned −0.016; the corrected, pre-registered form returns
  +0.027. Both are recorded here.
