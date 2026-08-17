# Experiment 3 — Findings

**Headline: no single axis of strain wins everywhere, but a three-probe
ensemble does.** The combined `lag_span + chain_checksum + confab_trap`
instruction is the best canary on **all three task sets** (F1@5 =
0.491 / 0.561 / 0.742 on coding / registers / babi) and beats every cheap
traditional baseline on two of the three — at 0.82–0.98 precision, far above
any traditional signal on the same tasks. The second headline is a cost the
earlier experiments never measured: **carrying a canary accelerates the very
hallucinations it predicts** (observer effect up to +0.32 on registers).

All numbers: `gpt-oss-20b`, paired task sets (200 coding / 100 registers /
100 babi), prediction window K=5, full tables in [`SUMMARY.md`](SUMMARY.md)
and `results3/<set>/SUMMARY.md`. The LLM-judge baseline is **not yet run**
(deferred to protect the OpenAI rolling-RPD budget; see Run order below).

## 1. The axis ranking, by domain

F1@5 per single-axis canary (best per set in bold):

| axis (canary) | coding | registers | babi |
|---|---|---|---|
| headroom (`staircase`) | **0.414** | 0.374 | 0.561 |
| reasoning composition (`chain_checksum`) | 0.335 | **0.383** | 0.611 |
| memory distance (`lag_span`) | 0.326 | 0.347 | **0.726** |
| abstention (`confab_trap`) | 0.310 | 0.255 | 0.507 |
| interference (`interference_twin`) | 0.308 | 0.330 | 0.230 |
| memory breadth (`multi_counter`) | 0.195 | 0.327 | 0.291 |
| unrehearsed retention (`sparse_recall`) | 0.162 | 0.128 | 0.347 |
| *null control (`static_trailer`)* | 0.000 | 0.046 | 0.000 |

Three stable facts sit under the domain-to-domain variation:

- **The top three axes are the same everywhere** — headroom, reasoning
  composition, and memory distance — and they are exactly the probes that
  demand *fresh computation over accumulated context every single turn*.
  `staircase` replicates experiment 2's `escalating_ledger` almost exactly on
  coding (0.414 vs 0.421) with a different field set, so that result is not an
  artifact of its particular fields.
- **Applicability caps recall.** `confab_trap` and `sparse_recall` can only
  fire on ~30% / ~4 turns of a trajectory by design; their recall ceilings are
  structural, not cognitive. Per-firing they are precise (confab precision
  0.67–1.00). A deployed probe should be checkable **every** turn.
- **The null control stays null** (third experiment in a row), and
  `sparse_recall` partially redeems experiment 1's `remember_fact`: the same
  planted-fact idea moves from F1 ≈ 0 to 0.13–0.35 once every-turn rehearsal
  is banned — supporting the copyability story: `remember_fact` failed because
  repetition kept the answer in the previous reply, not because static facts
  are unprobeable.

## 2. The ensemble result

| signal | coding | registers | babi |
|---|---|---|---|
| **ensemble (any of 3 fails)** | **0.491** (P 0.82) | **0.561** (P 0.98) | 0.742 (P 0.98) |
| ensemble 2-of-3 vote | 0.326 (P 0.81) | 0.377 (P 1.00) | 0.425 (P 0.96) |
| best traditional | turn_number 0.431 (P 0.55) | context_length 0.425 (P 0.45) | **turn_number 0.780** (P 1.00) |
| random compaction | 0.331 | 0.271 | 0.463 |

- On **coding** and **registers** the ensemble beats every traditional cheap
  signal on F1 while roughly **doubling their precision** — the rebuttal the
  README asks for, now replicated across two domains. Median lead where it
  fires early: 6 turns.
- On **babi** the base rate is 1.00 — every trajectory eventually fails the
  published benchmark — so precision is 1.0 for anything that fires and the
  contest is recall-only. There, "compact at turn 5" (F1 0.780) edges the
  ensemble (0.742). **When failure is certain and onset is concentrated, a
  schedule beats a signal**; canaries earn their keep where failure is *not*
  certain (coding: 0.77 base rate, registers: 0.67). Note context_length
  collapses entirely on babi (F1 0.020): short prompts never cross a useful
  threshold, while the canaries transfer unchanged.
- The 2-of-3 vote trades roughly half the recall for near-perfect precision
  (1.00 on registers) — the right variant when the triggered intervention is
  expensive. Any-fail is the right compaction trigger; 2-of-3 the right
  escalation trigger.
- Against the judge: experiment 2's judge scored F1 0.529 on the same coding
  tasks at ~1 extra LLM call per turn and 0.975 fire rate. The ensemble
  reaches 0.491 on coding for ~3 extra output lines. The direct exp-3
  comparison still needs the judge run (below).

## 3. The observer effect — canaries are not free

Hallucination rate by condition vs the paired baseline:

| | coding | registers | babi |
|---|---|---|---|
| baseline | 0.77 | 0.67 | 1.00 |
| median single canary | ~0.84 | ~0.87 | 1.00 |
| ensemble (3 probes) | 0.93 (+0.16) | 0.99 (+0.32) | 0.99 (—) |
| static_trailer (null) | 0.79 (+0.02) | 0.84 (+0.17) | 1.00 (—) |

Carrying probes measurably degrades the task itself, roughly in proportion to
probe load, and worst on the domain with the tightest per-turn arithmetic
(registers). Two consequences:

1. **Prior experiments' comparisons remain internally fair** (every signal was
   scored on its own condition's trajectories), but a deployed canary must
   price this in: the ensemble buys early warning at the cost of making
   failure ~somewhat more likely and earlier. Whether that trade is positive
   depends on the intervention — which is exactly the README's step-8
   deployment question (does canary-triggered compaction beat no canary
   at end-task success?). The observer effect makes that experiment the
   necessary next one, not an optional extension.
2. The near-null control (`static_trailer`) shows part of the effect is
   instruction *presence*, not difficulty (+0.17 on registers from a fixed
   string). The `multi_counter` light-vs-heavy pair on coding showed no
   monotone dose effect (+0.07 vs +0.005), so within-canary load matters less
   than probe count.

## 4. Deployment reading

For a compaction trigger on a small open model today:

- **Use the any-fail ensemble of `lag_span` + `chain_checksum` +
  `confab_trap`** (or add `staircase` where turn budgets are long): F1 beats
  every zero-extra-call baseline on the two non-saturated domains, precision
  0.82–0.98, median lead ~6 turns — enough runway to compact before broken
  output starts.
- Keep the probes **checkable every turn**; sparse probes are precise but
  recall-starved.
- If the environment is one where failure is near-certain regardless
  (babi-like saturation), skip the canary and compact on a schedule.

## Limitations

1. **Single model.** Everything above is `gpt-oss-20b`. The `gpt-4o-mini` arm
   (100 tasks/set) and the smaller `llama-v3p1-8b` rung are configured but not
   run; experiment 2's partial data says the fire rates will shift by an order
   of magnitude, so the axis *ranking* — not just the levels — needs the
   cross-model check.
2. **No LLM judge yet** (`--skip-judge`; one command, uses OpenAI RPD):
   `python -m experiments3.run_all3 --models gpt-oss-20b --sets coding,registers,babi --conditions baseline`
   after the exp-2 gpt-4o-mini arm is finished, or with
   `SENTINEL_JUDGE_MODEL=gpt-oss-20b` for a budget-free (but weaker) judge.
3. **babi saturates** at a 1.00 base rate even with the story-difficulty ramp;
   a canary experiment wants base rates in the 0.5–0.9 band. Dropping bAbI
   task 3 (or ending sessions at task 2) would deflate it.
4. **lag_span turn-1 semantics were patched mid-run** (models echo the current
   ticket in the vacuous all-NONE turn-1 slot; such slots are now excluded as
   not-applicable) and all stored trajectories were re-scored post-hoc from
   the saved text — exact, since canary scores never influence trajectory
   control flow. Fire rates before the fix are not comparable to the tables.
5. **Observer-effect rates are measured under early stopping** (trajectories
   end at the first hallucination), so they compare *whether* a trajectory
   ever hallucinated, at matched task sets — not full-horizon error counts.
