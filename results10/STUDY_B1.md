# Experiment 10, Study B1 — the carried-probe penalty is compaction-specific

**Pre-registered prediction (from Study B0): if the sub-additive penalty is
caused by the agent's self-summary having to reproduce the probe's ledger as
well as the task state, then it must shrink toward zero under deterministic
re-grounding, where the harness supplies both and the agent summarises nothing.**

**Confirmed.** The interaction goes from −0.032 to **+0.001**.

`gpt-oss-20b`, 100 coding tasks, paired. Cells A and C never reset, so they are
operator-independent and are shared between the two 2×2s — only B and D were
re-run. The re-grounding operator replays the user's own instructions through a
deterministic reducer (verified against the generator's truth on all 2,060
turns of the pool) and re-seeds the probe's ledger, so no LLM call happens at
reset and the agent never summarises anything.

## The same 2×2 under both operators

| operator | A | B | C | D |
|---|---|---|---|---|
| compaction (agent self-summary) | 0.849 | 0.908 | 0.823 | 0.851 |
| re-grounding (deterministic) | 0.849 | 0.912 | 0.823 | **0.888** |

| contrast | compaction | re-grounding |
|---|---|---|
| `C − A` carrying cost, no resets *(shared cells)* | −0.026 | −0.026 |
| `B − A` timing value, **no probe** | **+0.059** ✱ | **+0.063** ✱ |
| `D − C` timing value **while carrying a probe** | +0.028 | **+0.065** ✱ |
| **`(D−C) − (B−A)` interaction** | **−0.032** | **+0.001** |
| `D − B` cost of carrying, given oracle timing | **−0.058** ✱ | −0.024 |
| **`D − A` does a perfect CARRIED sentinel pay?** | **+0.002** | **+0.039** ✱ |

✱ = 95% bootstrap CI excludes zero.

## What this establishes

**1. The mechanism was correctly identified.** Under compaction, timing is
worth +0.059 to an unencumbered agent but only +0.028 to one carrying a probe —
the probe corrodes the repair operation. Under re-grounding the two are
+0.063 and +0.065: **identical**. The interaction is +0.001, i.e. perfectly
additive. Removing the agent-generated summary removes the entire penalty,
which is exactly what the mechanism predicted and nothing else obviously does.

**2. Study B0's headline was operator-specific, and is now bounded.** "A
perfect carried sentinel is only break-even" is true *under compaction*
(+0.002). Under re-grounding the same probe with the same timing yields
**+0.039, significant**. Carried probes are not fundamentally unable to pay —
they are unable to pay *when recovery is lossy self-summarisation*.

**3. The carrying cost itself is not the problem.** It is −0.026 in both
regimes (the same cells), and it is not significant. What killed the carried
sentinel under compaction was never the toll — it was the interaction.

## The rule this yields

> A carried probe must be paid for twice: once in its own overhead, and again
> in whatever it does to your recovery mechanism. The second charge is the
> larger one, and it is levied only when recovery requires the agent to
> restate its own state. With a deterministic external store, the second
> charge disappears and a well-timed carried probe pays for itself.

## Limitations

- One probe (`escalating_ledger`), one model, one task family. Experiment 9
  shows the *sign* of the carrying cost varies with model × task × probe-match,
  so `C − A` = −0.026 is not a constant of nature.
- The oracle is an upper bound on timing, not an achievable detector; `D − A`
  = +0.039 is the ceiling a real carried monitor would have to approach.
- Re-grounding here reconstructs state from the user's own instruction history,
  which is exact for this task family. A deployment whose external store is
  partial sits somewhere between the two rows — which is what the
  operator-fidelity sweep (state classes, not percentages) is for.
