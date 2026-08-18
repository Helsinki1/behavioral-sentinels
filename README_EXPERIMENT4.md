# Experiment 4 — Do sentinel-triggered resets improve end-task success?

Experiments 1–3 were **prediction** benchmarks: does a signal fire before a
hallucination. Experiment 4 is a **deployment** benchmark: does *acting* on the
signal produce a better outcome than acting on a clock, at a matched
intervention budget. It is the step-8 experiment the project README always
specified, and experiment 3's observer effect made it mandatory rather than
optional — the probe now has a measured cost, so it has to earn it.

## Three things that differ from every earlier experiment

1. **No early stop.** Trajectories run the full horizon, so errors can be
   recovered from. "First hallucination" is no longer the end of the story.
2. **The outcome is task accuracy**, not a prediction score: the share of turns
   with zero errors.
3. **Arms are budget-matched on resets**, so the comparison is about *when* you
   intervene, not *how often*.

## The arms

| arm | resets on | carries probe | why it exists |
|---|---|---|---|
| `A_no_reset` | never | no | floor |
| `B_random` | random turns | no | budget-matched control |
| `C_scheduled` | a clock | no | the thing to beat |
| **`C_prime_carried`** | **a clock** | **yes** | **isolates the observer-effect cost** |
| `D_sentinel` | `escalating_ledger` failures | yes | the hypothesis |
| `F_oracle` | just before the true first failure | no | upper bound / go-no-go gate |

`C_prime_carried` is the arm a naive design omits. Without it, `D − C` sums two
different effects — "the timing information was useless" and "carrying the
probe hurt" — and a negative result is uninterpretable. With it:

* `D − C′` = value of the timing information (probe load held constant)
* `C′ − C` = cost of carrying the probe (reset schedule held constant)

`F_oracle` is run as a **gate**: it upper-bounds what any signal could buy. If
a perfect predictor cannot beat the clock, no sentinel can, and the rest of the
experiment is wasted money.

## The compaction operator

This is where the experiment lives or dies, so it is fixed and identical in
every resetting arm. On a reset the agent is asked to write out the module
state *as it currently believes it to be*; that self-summary replaces the
conversation, and the task continues.

**No ground truth is injected.** Resetting *early* preserves a still-correct
state and drops accumulated noise; resetting *late* canonicalises whatever
error the agent has already made. That asymmetry is the mechanism that would
make early warning valuable, so it gives the sentinel a fair chance to win.

The alternative — replaying the user's own instruction history — would
perfectly reconstruct the true state and make every reset policy look identical
and excellent. That is oracle leakage dressed up as compaction.

## Result

See [`results4/FINDINGS.md`](results4/FINDINGS.md). In one line: **the sentinel
loses (H3), but the oracle wins significantly at a quarter of the reset budget
— placement, not frequency, is the lever.**

## Running it

```bash
python -m experiments4.run_all4 --gate     # A, C, F only — the go/no-go gate
python -m experiments4.run_all4            # all six arms (F needs A; B needs D)
python -m experiments4.metrics4
python -m experiments4.figures4
```

Resumable. `--limit N`, `--arms`, `--model` subset the run.

```
experiments4/
  config4.py     arms, budgets, the primary outcome definition
  harness4.py    reset-capable runner + the compaction operator
  policies4.py   none / scheduled / sentinel / random / oracle
  metrics4.py    paired contrasts with bootstrap CIs
  figures4.py    tradeoff plot + contrast forest plot
```
