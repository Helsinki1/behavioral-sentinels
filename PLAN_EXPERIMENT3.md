# Experiment 3 Plan — Which strain predicts hallucination? (→ `results3/`)

> **Status: implemented in [`experiments3/`](experiments3/)** — see
> [`README_EXPERIMENT3.md`](README_EXPERIMENT3.md) for what was actually built.
> One extension beyond this plan: the experiment runs on **three task sets**
> (coding = exp-2 tasks augmented, registers = exp-1 tasks augmented, and the
> published **bAbI** benchmark wrapped into long sessions), so every table and
> figure is produced once per domain and replication is part of the result.
> The `ensemble` condition is a real combined-instruction run (not a post-hoc
> merge), scored any-fail and 2-of-3. Offline `selftest3` passes on all three
> sets; the full `gpt-oss-20b` arm is launched via
> `python -m experiments3.run_all3 --models gpt-oss-20b --skip-judge`
> (judge afterwards, to respect the OpenAI rolling-RPD budget).

## Where experiments 1–2 leave us

Experiment 1: a task-irrelevant chore *can* fire before a task hallucination, but
only one of six static probes carried signal (`variable_check`, F1 0.587).
Experiment 2: "the answer must change every turn" was the wrong explanation —
`rotating_prefix` changes every turn and never fired once in 400 trajectories.
The real gradient is **copyable → must be recomputed**: a canary is a sentinel
only if its correct answer cannot be lifted from the model's own previous reply
(`escalating_ledger` F1 0.421 > context_length 0.340, turn_number 0.374,
random 0.332 — but < LLM judge 0.529).

Experiment 3 takes non-copyability as a *prerequisite*, not a variable. The new
question is: **which cognitive resource, when strained, degrades first — and
does the axis of strain that fires earliest match the axis the task hallucination
lives on?** Every canary below isolates one axis, so the results table reads as
a diagnosis of *why* small models fall apart, not just *when*.

## The seven canaries (one axis each)

All non-copyable by construction. All ~1 line of output appended to a reply the
agent was producing anyway. All with a **difficulty knob** (exp-2's partial
gpt-4o-mini arm showed fire rates differ 27× between models, so difficulty must
be calibratable per model).

| id | canary | axis strained | knob |
|---|---|---|---|
| `lag_span` | `PREV<k>: <ticket>` — echo the ticket from **k turns ago**, for k ∈ {1,3,6} simultaneously (three tokens on one line) | memory **distance** | the set of lags |
| `multi_counter` | maintain 2–4 independent **sparse** counters (deletions, renames, vowel-named additions…) that only some turns increment — no mechanical +1 path | memory **breadth** (items held) | number of counters |
| `chain_checksum` | `CHECK: <n>` where n = (previous n + first-function-name length) mod 97 — each turn *composes* with the last, so errors compound and can't be locally patched | **reasoning depth** (sequential composition) | modulus / arity of the update rule |
| `interference_twin` | maintain a tiny **shadow symbol table** of 3 fake function names that look like module names (same style vocabulary) and are renamed by canary-only instructions; report one on request | **interference** between similar items | name similarity to real module symbols |
| `confab_trap` | occasional query `TAG for ENG-xxxx?` where some tickets were never assigned a tag — correct answer `TAG: NONE`. Fires when the model **invents** a tag | **abstention / fabrication** | trap frequency, distance to the assignment turn |
| `sparse_recall` | plant one fact at turn 1 (exp-1's audit code) but probe it only at ~4 unpredictable turns, never every turn — removes the rehearsal/copy path that made `remember_fact` useless | **retention without rehearsal** | probe spacing |
| `staircase` | `escalating_ledger`, but the escalation schedule is a **per-model calibrated staircase** (a 20-task pilot picks the escalation rate so pilot fire rate lands in 30–70%) | **headroom** (dose-response) | escalation period |

Why these seven are the informative set:

- `confab_trap` is the highest-value bet: it measures the *same behavior* as the
  dominant task failure (24% wrong_sig + 22% fabricated_symbol + the
  deleted-function trap = the model inventing instead of saying "doesn't
  exist") on an irrelevant channel. If fabrication-on-probe precedes
  fabrication-on-task, that is a mechanism claim, not just a correlation.
- `interference_twin` mirrors the single most common trigger observed in exp 2
  (rename a function, then call it by its old name). Prediction: it fires
  earliest specifically before `wrong_sig`/`fabricated_symbol` first-errors.
- `lag_span` gives a **graded** signal for free: failing k=6 while passing k=1
  is an earlier, softer alarm than a binary miss — effective memory span
  shrinking is the early warning.
- `sparse_recall` is the redemption test for experiment 1: if it fires where
  `remember_fact` never did, the exp-1 nulls are explained by rehearsal
  (the fact was restated every turn, so it stayed copyable), closing the loop
  on the copyability story.
- `chain_checksum` vs `multi_counter` vs `lag_span` cleanly separates
  *reasoning* strain from *breadth* strain from *distance* strain — the
  "what should we strain next?" question is answered by whichever column wins.
- `staircase` turns the canary into a continuous gauge (highest rung passed per
  turn), scored with a threshold sweep exactly like context_length — the first
  apples-to-apples canary-vs-traditional comparison.

Plus two non-canary rows:

- `ensemble` — first failure across {`confab_trap`, `chain_checksum`,
  `lag_span`} (three cheap orthogonal probes ≈ one line each). This is the shot
  at beating the LLM judge (F1 0.529) at ~1% of its cost. Scored two ways:
  fire-on-first-failure and fire-on-2-of-3 (precision variant).
- `static_trailer` — kept again as the null control.

## Controls the reviewers will ask for

1. **Observer effect (canary load):** run `multi_counter` at light (2 counters)
   and heavy (4) settings and compare *hallucination base rate* against
   baseline on the paired task set. If heavy canaries accelerate task failure,
   lead-time claims are confounded — this must be measured, not assumed.
2. **Turn-1 hallucinations** (~20% of trajectories): report headline tables
   both including (comparable to exp 1–2) and excluding them (secondary table),
   since no signal can lead a turn-1 failure.
3. **Soft scoring:** canaries with graded output (`lag_span` rungs,
   `multi_counter` fraction-of-counters-correct, `staircase` rung) also get a
   swept-threshold row, so binary canaries are not handicapped against
   best-F1-swept traditional signals.

## Models

- `gpt-oss-20b` (Fireworks) — full 200-task paired arm, as before.
- `gpt-4o-mini` — 100-task arm (fits the rolling RPD budget per exp-2
  FINDINGS), and **finish the exp-2 arm first** so the calibration hypothesis
  has data.
- **Add one smaller open model** (e.g. `llama-3.1-8b` or `qwen2.5-7b` on
  Fireworks): the project thesis is that smaller models surface hallucination
  sooner; a 3-point size ladder (8B → 20B → 4o-mini) turns "canary difficulty
  must be calibrated per model" from a hypothesis into a measured curve, and the
  staircase canary gives the calibration mechanism.

## Implementation (`experiments3/`, mirroring `experiments2/`)

1. `tasks3.py` — extend the exp-2 generator (same coding task family, same
   symbol-table simulator) with per-turn canary payloads: ticket history for
   lags, counter ground truths, shadow-symbol instructions + truth, tag
   assignments/traps, sparse probe schedule, staircase ledger expectations.
   Same seed discipline; write `data3/tasks3.json`.
2. `canaries3.py` — instructions + checkers. Checkers return a **score** (float
   0–1) not just pass/fail; binary canaries return {0,1}. `check_hallucination`
   imported unchanged from `canaries2`.
3. `selftest3.py` — perfect-agent synthesis for every condition, zero
   errors/failures asserted, plus fault injection per canary kind. Non-negotiable
   given it caught 3 bugs in exp 2 before any spend.
4. `runner3.py` — same loop; two changes: (a) record per-turn canary *score*,
   (b) `--pilot` mode: 20 tasks, fire-rate report, writes the chosen difficulty
   knob per model to `data3/calibration.json`, which the full run reads.
5. `metrics3.py` — imports `classify`/`summarize`/`summarize_random` from
   `experiments/metrics.py` (same scoring, third experiment in a row). Adds:
   score-threshold sweep for graded canaries, ensemble rows, turn-1-excluded
   secondary tables, observer-effect table (hallucination rate by condition),
   and a **conditional matrix**: P(first error kind | which canary fired first)
   — the axis-matching result.
6. `judge3.py` / `figures3.py` — as exp 2; judge on gpt-4o-mini this time
   (restoring the exp-1 judge and settling FINDINGS limitation #2). Figures:
   the standard 4, plus fig6 axis-vs-error-kind heatmap and fig7 fire-rate vs
   model-size curve.
7. Outputs land in `results3/` (`SUMMARY.md`, `FINDINGS.md`, per-condition
   folders, `figures/`).

### Run order & budget

```bash
python -m experiments2.run_all2 --limit 100 --models gpt-4o-mini  # finish exp-2 arm (day 0)
python -m experiments3.tasks3
python -m experiments3.selftest3
python -m experiments3.run_all3 --pilot                            # 20 tasks × models, calibrate knobs
python -m experiments3.run_all3                                    # full run (resumable)
python -m experiments3.figures3
```

Cost envelope: 9 conditions (7 canaries + light/heavy control + baseline) ×
200 tasks × ~20 turns ≈ 36k calls on Fireworks (fine), 100-task arm ≈ 9k calls
on OpenAI/day — schedule gpt-4o-mini conditions across 2 rolling-window days,
judge included.

## What each outcome would tell us

- `confab_trap` or `interference_twin` wins → hallucination onset is an
  *abstention/interference* failure; next step is intervention: fire → inject a
  state refresh and measure end-task recovery (the "Doomed from the Start"
  endpoint, and the README's step 8 deployment).
- `lag_span`/`multi_counter` win → it's a *memory capacity* failure; compaction
  triggered on span shrinkage is the product story.
- `chain_checksum` wins → *reasoning fatigue*; harder tasks per turn, not longer
  context, is the axis to titrate next.
- `staircase` tracks the others' firing order per model size → canaries are
  calibratable gauges, and one adaptive canary can replace the per-model zoo.
- `ensemble` ≥ judge F1 at <5% fire-rate cost → the headline deployment claim:
  a free behavioral sentinel replaces a per-turn judge call.
