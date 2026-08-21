# Experiment 10 — When is a sentinel worth its cost?

*(Numbered 10 because it ran after experiment 9. Study B0 reuses the
experiment-4 harness and regime, so its arms live in `experiments4/config4.py`
and its trajectories in `runs4/`; analysis and results are here and in
`results10/`.)*

## The question, and why the previous six experiments couldn't answer it

Experiments 1–6 ran a horse race: which trigger reaches the highest accuracy.
That framing silently **prices a restart at zero**, which is exactly the
assumption that decides the answer. In a deployed agent a restart costs
re-onboarding latency, a destroyed prompt cache, re-run tool calls, and
sometimes human attention — and the zero-carry sentinel's entire remaining
value proposition is that it buys most of the clock's benefit at **a quarter of
the restarts** (exp 6: 73% of the lift at 26% of the restarts).

So the question is not *which policy wins*. It is **which policy wins at what
price**, and the deliverable is not a winner but a **decision rule with
empirical constants**.

## Study A — the break-even surface (DONE, no new API calls)

Score every policy by a linear utility over its measured operating point:

```
U = accuracy − R · restarts_per_task − T · prompt_ktokens_per_task
```

`R` = cost of one restart, `T` = cost of 1k prompt tokens, both in
accuracy-equivalent units (`R = 0.01` ⇒ one restart costs one accuracy point).
Each policy is then a plane over the (R, T) cost plane and the winner is the
upper envelope. Results: [`results10/STUDY_A.md`](results10/STUDY_A.md),
map in `results10/figures/fig1_decision_map.png`.

**Findings (re-analysis of exps 5–6, 90 tasks):**

| operator | who wins, and where |
|---|---|
| **lossless** (restart restores true state) | clock only while restarts are nearly free (R < 0.003); context-growth in a narrow band; **the zero-carry sentinel owns the large middle region**; never-reset once restarts are very expensive |
| **lossy** (restart keeps the agent's own summary) | **never-reset dominates almost everywhere** — the sentinel beats the clock, but both lose to not restarting at all |

**The headline constant: under a lossless operator the zero-carry sentinel
overtakes the clock as soon as one restart costs more than 0.0027
accuracy-equivalents — 0.27 accuracy points.** That is a very low bar. Any
deployment where a restart burns a prompt cache or a few seconds of latency is
already past it.

Study A is cheap and already done. Studies B–D are what turn two operator
points and one detector into an actual surface.

---

## Study B — the operator-fidelity axis (primary new run)

Experiments 4–6 established the project's central law: *the lossier the reset
operator, the more timing matters; the cheaper the operator, the more frequency
wins.* But that law rests on **two points** — exp 6's re-grounding and exp 5's
compaction — and Study A shows the sentinel's winning region flips entirely
between them. The crossover is the thing to measure.

**Manipulation.** Make fidelity continuous. At a reset, restore a fraction φ of
state entries from the external store; carry the remaining (1−φ) over from the
agent's **own** believed value. φ = 1 reproduces exp 6 exactly; φ = 0
reproduces exp 5's self-summary compaction. Sweep

```
φ ∈ {1.0, 0.75, 0.5, 0.25, 0.0}
```

This is a clean interpolation between two already-validated operators, so both
endpoints are checkable against existing runs — a built-in correctness test.

**Arms per φ:** `never_reset` (φ-independent, run once), `clock`,
`zero_carry_sentinel`, `oracle`. 5 φ × 3 + 1 = **16 arms**.

**Primary outcome.** The φ at which `zero_carry_sentinel − clock` crosses zero,
and the φ at which `never_reset` stops being dominated.

**Pre-registered hypotheses.**
- **H1** The sentinel−clock delta increases monotonically as φ falls (timing
  matters more as restarts get lossier).
- **H2** There is a φ* below which never-reset dominates every policy. Exp 5
  says φ* > 0; exp 6 says φ* < 1.
- **H3** The oracle−clock delta also rises as φ falls — it was *negative* at
  φ=1 (exp 6) and *positive* at φ=0 (exp 4).

H3 is the sharpest test, because it predicts a **sign flip** in a quantity
already measured at both ends.

## Study C — how good would a detector have to be? (primary new run)

Rather than hunting for better probes, **synthesise detectors of known quality**
by degrading the oracle, and find the quality threshold at which a sentinel
beats the clock. For each trajectory, with the true first-failure turn H:

- with probability `r` (recall) fire at turn `H − L`, L drawn from the measured
  lead distribution;
- independently with probability `f` (false-alarm rate) fire at a uniformly
  random turn.

Sweep `r ∈ {0.2, 0.4, 0.6, 0.8, 1.0}` × `f ∈ {0, 0.25, 0.5}`, at the two φ
values that Study B identifies as most informative.

**Deliverable:** an iso-utility contour in (recall, false-alarm) space marking
where a detector starts beating the clock — and the **real** zero-carry
monitor's measured (recall, false-alarm) plotted on it. That answers, in one
figure, *how far off are we and in which direction* — precision or recall.

This is the highest-information run in the design: it converts "our probes lose"
into "a probe needs recall ≥ X at false-alarm ≤ Y, and ours is at (x, y)."

## Study D — scenario axes (secondary)

The cost model handles *price*; these handle *situation*.

1. **Horizon.** Extend a subset to 60 turns. Monitoring should matter more as
   horizon grows; the clock's fixed period does not adapt.
2. **Base failure rate.** Stratify the existing pool into difficulty terciles.
   A sentinel should be worthless when failure is near-certain (exp 3's bAbI at
   1.00) and most valuable in the middle band.
3. **Restart cost, grounded.** Study A leaves R abstract. Measure the real
   components in the existing runs — prompt-cache tokens invalidated per
   restart, added latency — so the decision map can be annotated with where
   realistic deployments actually sit.

## Cost and scale

Study A: **done, $0.** Studies B+C: 16 + ~30 arms × 30 coding tasks × ~20 turns
≈ 28k calls, comparable to experiment 2 — roughly **$8–12 and 3–4 hours** on
Fireworks. Study D is re-analysis plus one 60-turn subset.

Use the **coding** domain only for B/C: exp 5 showed effects are largest there,
registers saturate at 0.95+, and bAbI saturates at a 1.00 base rate.

## What the paper gets out of this

A decision rule with numbers in it, replacing six experiments of horse race:

> Run a zero-carry sentinel when your restart operator restores less than φ* of
> state, **or** when a restart costs more than R* accuracy-equivalents. Below
> both thresholds, restart on a clock and skip monitoring. If your operator is
> lossy and restarts are expensive, do not restart at all. A carried probe never
> pays: its −0.043 observer cost exceeds the entire +0.059 timing prize.

with φ*, R* measured, and the detector-quality requirement stated as a contour
that any future probe can be plotted against.
