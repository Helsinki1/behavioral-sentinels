# Experiment 6 — Sentinel-triggered re-grounding: when to start a fresh session

## Why the operator, not the signal, is the variable now

Experiments 4–5 tested resets whose operator was **compaction**: the agent
writes a snapshot of state that lives only in context, and the snapshot —
errors included — becomes the new session. Exp 5's verdict was that this
operator, not the signal, had become the bottleneck: a free signal matched
the oracle, yet even the *oracle* no longer beat never resetting, because
every reset risked canonicalising the agent's own errors.

But the project's original motivating scenario — *when should you start a new
Claude Code session instead of continuing a long one?* — has a property the
benchmark lacked: an **external ground truth**. A fresh Claude Code session
re-reads the repo; the disk, not the conversation, is the state store.
Experiment 6 rebuilds the benchmark around that property and re-asks every
exp-5 question under it.

## The two reset operators (both deterministic — no LLM call at reset)

**R1 `reground`** — the conversation is replaced by the original task
briefing with the **current true state** substituted for the initial state:
the current module source (+ deletion tombstones), the current register
values, or the current story so far. State is materialised by a harness
reducer (`store6.py`) that applies each user-issued instruction exactly as a
file system applies edits. Fidelity is proven, not assumed: an offline
verifier checks the store equals the generator's own truth fields on **every
turn of every task in the pool** (8,535 turns, zero mismatches), and every
reset's full resume message is recorded in the trajectory as an audit trail.
Nothing enters the store that a repo + its history would not hold.

**R2 `replay`** — the conversation is replaced by the verbatim log of every
prior *user* message, assistant turns dropped. Zero harness intelligence:
the conservative bracket for anyone who reads R1 as oracle-feeding. R2 also
isolates a mechanism question: if R2 helps, the harm in long sessions is the
agent's own accumulated outputs (self-conditioning), not the user content.

## Arms

Protocol identical to exp 5 (full horizon, no early stop, per-turn accuracy,
6-reset cap, 2-turn behavioural grace, same 90-task pool, difficulty-
stratified). **No arm carries a probe** — exp 5's `D_labeled` closed that
design. `A_no_reset` is imported verbatim from `runs5` (it never resets, so
it is operator-independent), which makes the exp-5↔exp-6 operator contrasts
exactly paired.

| arm | trigger | operator |
|---|---|---|
| `A_no_reset` | never | — (imported from runs5) |
| `B_random` | random turns, budget-matched to `Z_reground` | R1 |
| `C_clock` | every 6 turns | R1 |
| `C_ctx` | prompt grew ≥600 tokens since last reset | R1 |
| `C_judge` | LLM judge (window 8) says degraded | R1 |
| `Z_reground` | zero-carry self-consistency monitor fires | R1 |
| `F_oracle` | once, just before `A_no_reset`'s first-failure turn | R1 |
| `G_dense` | every 3 turns — the densest schedule the cap allows | R1 |
| `Z_replay` / `C_clock_replay` / `F_oracle_replay` | as above | R2 |

## The questions, in evaluation order

1. **Gate:** `F_oracle − A_no_reset`. If perfect timing with a loss-free
   operator still doesn't beat never resetting, context rot in this pool is
   not recoverable by restarting, and the rest is moot.
2. **Goal line:** `Z_reground` vs clock / ctx / judge / random.
3. **Is timing even a question when resets are cheap?** `G_dense` vs
   `F_oracle` vs `Z_reground` — if the densest schedule matches the oracle,
   the honest advice is "restart freely" and the sentinel's remaining value
   is cost (see the Pareto figure).
4. **The operator effect:** exp-6 arm minus exp-5 arm at the same trigger,
   paired on task — the direct price of compaction vs re-grounding.
5. **The bracket:** R2 vs R1 on matched arms — how much of the gain needs a
   materialised store vs a mere transcript replay.

## Run

```bash
python -m experiments6.store6              # offline: prove store == truth
python -m experiments6.selftest6           # offline, mock LLM, all arms
python -m experiments6.run_all6 --gate     # A (import), C_clock, F_oracle
python -m experiments6.run_all6            # all arms
python -m experiments6.metrics6            # results6/SUMMARY.md + metrics.json
python -m experiments6.figures6            # results6/figures/
```

## Limitations stated up front

The harness-as-file-system is the *definition* of the regime under test, not
a loophole — but only R2 fully silences the oracle-feeding objection, so both
operators run. These tasks underrepresent the cost of losing in-flight
reasoning at a restart (real sessions carry tacit plans). Restarts fix
context rot, not skill: the difficulty ramp bounds every arm, and
`G_dense` vs `A_no_reset` decomposes the two. One model (`gpt-oss-20b`);
the gpt-4o-mini arm inherits exp 2–5's rolling-RPD limitation.

Results: [`results6/SUMMARY.md`](results6/SUMMARY.md) ·
[`results6/FINDINGS.md`](results6/FINDINGS.md)
