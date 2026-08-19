# Experiment 5 — Task-conditioned sentinel routing, deployed

## Why routing, and why now

Four experiments established a consistent ledger for behavioural sentinels:

1. A probe only fires if its answer cannot be copied from the model's own
   previous reply (exp 2).
2. **Which axis of strain degrades first depends on the task domain** (exp 3):
   `staircase` (headroom) was the best single probe on coding (F1 0.414),
   `chain_checksum` (update composition) on registers (0.383), and `lag_span`
   (memory distance) on babi (0.726) — the same three probes, ranked
   differently by domain, and each ranking matches what the domain actually
   demands.
3. Carrying a probe costs what it measures (exp 3's observer effect, priced
   at −0.043 accuracy in exp 4), and placement beats frequency (exp 4's
   oracle: +0.059 over the clock at a quarter of the resets).

Experiments 1–4 always deployed **one blanket probe on every task**. The
routing hypothesis says that was the bottleneck: signal–failure coupling.
A probe predicts a task's failure only if it stresses the **same mechanism
the task will fail by** — so classify the task first, then attach the one
matched probe.

## The two routed systems under test

**Carried routing (`D_routed`).** A router — one `gpt-oss-20b` call per task,
reading only the task briefing — classifies the task's dominant state demand
into one of five genres, each mapped to the exp-3 probe that stresses that
mechanism:

| genre | failure it names | probe |
|---|---|---|
| ARTIFACT_ACCUMULATION | losing accumulated artifact state under a growing rule set | `staircase` |
| UPDATE_APPLICATION | mis-applying precise per-turn updates | `chain_checksum` |
| RETRIEVAL_DISTANCE | failing to recall facts given many turns ago | `lag_span` |
| INTERFERENCE_TRACKING | confusing many similar moving entities | `interference_twin` |
| EVIDENCE_ABSTENTION | asserting what was never given | `confab_trap` |

The router is part of the measured system: its label is used verbatim, its
mistakes are not corrected, and its agreement with the intended
domain→genre mapping is reported separately. The taxonomy is open-ended
(spatial reasoning, plan-state tracking, … would slot in as new genre→probe
rows); these five cover the pool used here.

**Zero-carry routing (`Z_routed`).** Exp 4 showed two-thirds of the blanket
sentinel's deficit was the *price of carrying it*. So the second routed
system carries nothing: deterministic monitors read degradation off output
the agent already produces —

- *coding*: reply missing its code block or a queried `SIG` line; a `SIG`
  that contradicts the agent's **own** earlier report for a function no user
  message has touched since; a `SIG` for a function the agent itself
  reported `DELETED`;
- *registers*: a missing `VALUE` line; a re-queried register whose reported
  value changed while nothing touched it;
- *babi*: a missing `ANSWER` line; an answer word that never occurred in the
  active story (fabrication, detectable with no answer key).

These are self-consistency signals: they cannot fire before the first slip,
but they can place a reset before the slip becomes a cascade — at exactly
zero carrying cost.

## Arms

Deployment protocol as in exp 4: full horizon, no early stop, outcome =
share of turns with zero errors, every arm capped at 6 resets with a 2-turn
post-reset grace for behavioural triggers, all arms paired over the same
90 tasks (30 coding / 30 registers / 30 babi, difficulty-stratified from
`data3`, which carries every probe's payloads on every domain).

| arm | trigger | carries |
|---|---|---|
| `A_no_reset` | never | — |
| `B_random` | random turns, budget-matched to `D_routed` | — |
| `C_clock` | every 6 turns | — |
| `C_ctx` | prompt grew ≥600 tokens since last reset | — |
| `C_judge` | LLM judge (window 8) says degraded | — |
| `C_prime_routed` | clock | routed probe |
| `D_routed` | routed probe fails | routed probe |
| `D_labeled` | pre-labeled probe fails (deterministic `INTENDED_GENRE` lookup, no router call) | labeled probe |
| `D_blanket` | blanket probe fails (`staircase` everywhere — the exp-1..4 design) | blanket probe |
| `D_rotated` | deliberately mis-assigned probe fails (coding→`chain_checksum`, registers→`lag_span`, babi→`staircase`) | rotated probe |
| `Z_routed` | routed zero-carry monitor fires | **nothing** |
| `F_oracle` | one reset just before `A_no_reset`'s first-failure turn | — |

The routing claim is controlled twice: `D_blanket` (does routing beat the
old one-probe-everywhere design?) and `D_rotated` (same three probes, same
load, only the *assignment* scrambled — if routing is real,
`D_routed > D_rotated`). `C_prime_routed` isolates the carrying cost of the
routed probe from the value of its timing, as in exp 4. `D_labeled` prices
the router itself: it is `D_routed` with the LLM router replaced by the
ground-truth genre label, so `D_labeled − D_routed` is the accuracy the
router's noise costs.

## Compaction

The exp-4 operator, upgraded: a dedicated compaction system prompt produces
a structured snapshot (progress, durable rules, accumulated rule facts, open
items, uncertainties, plus a domain state section — module / registers /
active story), which is **structurally validated** (headers, turn numbers,
parseable state block) with one retry, then replaces the conversation. No
ground truth is ever injected; a failed snapshot keeps the original context.

## What would count as success

The project's goal line: a sentinel-triggered system that beats turn count,
context length, LLM judge and random resets on end-task accuracy. Secondary
questions, answerable regardless: does routing beat blanket and rotated
probes (the routing claim itself), does the zero-carry variant finally
escape the observer effect that killed exp 4's sentinel, and — via
`D_labeled` — how much of the routed sentinel's remaining deficit is the
router's own noise?

## Run

```bash
python -m experiments5.selftest5          # offline, mock LLM, all arms
python -m experiments5.run_all5 --gate    # A, C_clock, D_routed
python -m experiments5.run_all5           # all twelve arms
python -m experiments5.metrics5           # results5/SUMMARY.md + metrics.json
python -m experiments5.prediction5        # results5/PREDICTION.md (no API calls)
python -m experiments5.figures5           # results5/figures/
```

## The prediction layer

`prediction5.py` asks the exp-1..3 question — precision/recall of each
signal at predicting the first failure within K=5 turns — but re-scored on
**one trajectory set at a time**, with turn-number / context-length / random
thresholds re-tuned on that same set, so signals are never compared across
observer-effect-shifted distributions (the exp-3 pitfall, handled by
construction). Scoring unit is the pre-first-reset segment of each
trajectory; where a set's own signal triggers the reset its lead time is
right-censored, and the report says so per table. No new API calls.

Results: [`results5/SUMMARY.md`](results5/SUMMARY.md) ·
[`results5/FINDINGS.md`](results5/FINDINGS.md) ·
[`results5/PREDICTION.md`](results5/PREDICTION.md)
