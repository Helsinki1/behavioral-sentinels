# Experiment 3 — Which strain predicts hallucination? (three task sets)

Experiment 2 established that a canary only works as a sentinel if its answer
**cannot be copied from the model's own previous reply**. Experiment 3 takes
that as a prerequisite, not a variable: every probe below is non-copyable by
construction, and each isolates **one cognitive axis**. The results table
therefore reads as a diagnosis of *which* resource, when strained, degrades
first — and whether that axis matches the axis the task hallucination lives on.

Two design changes from experiments 1–2, and nothing else:

1. **Axis-isolating canaries** (7 single-axis probes + ensemble + controls).
2. **Three task sets instead of one**, so every figure and table exists three
   times over — replication across domains is part of the result:

| set | task | source | N | horizon |
|---|---|---|---|---|
| `coding` | maintain a Python module | exp-2 tasks (`data2/tasks2.json`), **augmented** with exp-3 payloads | 200 | 12–30 |
| `registers` | integer register book-keeping | exp-1 tasks (`data/tasks.json`), augmented | 100 | 15–35 |
| `babi` | story state tracking | **published bAbI QA benchmark** (tasks 1–3, items verbatim, Muennighoff/babi mirror) wrapped into 3–5-story sessions with explicit `--- NEW STORY ---` resets; one question per turn, single-word objective answers; story difficulty ramps 1→2→3 across the session, mirroring the exp-1/2 difficulty ramps | 100 | 15–25 |

Reusing the exp-1/exp-2 task data unchanged means zero new generator bugs on
those sets and full comparability with the earlier results; bAbI adds an
external, published benchmark with the same objective per-turn labels.

## The canaries (one axis each)

All are one line of output appended to a reply the agent was producing anyway.
Payloads (tickets, keys, events, traps) are precomputed into
`data3/tasks3_<set>.json`, so every canary answer is machine-checkable.
Checkers return a **score in [0,1]** (a failure is score < 1 on an applicable
turn), so graded canaries also get a score-threshold sweep — the same
best-threshold treatment the traditional signals get.

| canary | axis | what it demands |
|---|---|---|
| `lag_span` | memory **distance** | `ECHO: <a> <b> <c>` — the tickets from 1, 3 and 6 messages back. Graded: failing the long lag first is a soft early alarm. |
| `multi_counter` | memory **breadth** | `COUNTS: red=n blue=n gold=n` — running totals of sparse `EVENT:` lines. No mechanical +1 path. (`multi_counter_heavy`, coding only: 4 colors at higher rate — the observer-effect / load-titration control.) |
| `chain_checksum` | reasoning **composition** | `CHECK: (prev + KEY) mod 97` — each turn composes with the last; errors compound and cannot be locally patched. |
| `interference_twin` | **interference** | three shadow entries whose names look like the task's own symbols, occasionally renamed; report the queried slot. Mirrors exp-2's #1 trigger (rename → stale name). |
| `confab_trap` | **abstention** | `Tag query:` for a past ticket that is sometimes untagged — the correct answer is exactly `TAG: NONE`; a made-up tag is a *fabrication*, the same behavior as inventing a signature for a deleted function. |
| `sparse_recall` | **unrehearsed retention** | an audit code given once, probed at ~4 unpredictable turns, with an explicit *do-not-repeat-it-otherwise* rule — removing the rehearsal/copy path that made exp-1's `remember_fact` useless. Rehearsal violations are recorded. |
| `staircase` | **headroom** | a `LEDGER:` whose rule gains a field every `STAIR_PERIOD` turns (turn → even-ending ticket count → turn-1 ticket → digit-7 count). The escalation period is the per-model difficulty knob. |
| `ensemble` | — | `lag_span` + `chain_checksum` + `confab_trap` in a single run; scored both as any-member-fails and 2-of-3. The shot at beating the LLM judge at ~1% of its cost. |
| `static_trailer` | null control | the exp-2 fixed string, unchanged. |

## What metrics3 adds

- per-set results in `results3/<set>/` plus a cross-set `results3/SUMMARY.md`
- score-threshold sweeps for graded canaries
- an **observer-effect table**: hallucination rate per condition vs baseline —
  does carrying the canary itself accelerate task failure?
- a secondary table **excluding turn-1 hallucinations** (no signal can lead those)
- `confab_trap` fabrication share, `sparse_recall` rehearsal rate
- the standard four figures **per task set** in `results3/figures/<set>/`

## Running it

```bash
python -m experiments3.tasks3       # build/augment all three task sets
python -m experiments3.selftest3    # offline: all checkers, all sets, no API calls
python -m experiments3.run_all3 --models gpt-oss-20b --skip-judge   # trajectories + metrics
python -m experiments3.run_all3 --models gpt-oss-20b                # + LLM judge (uses OpenAI RPD)
python -m experiments3.figures3
```

Everything is resumable; completed trajectories/judge files are skipped.
`--pilot` runs 20 tasks under the canary conditions only and prints fire rates
(target band 0.30–0.70) for tuning the difficulty knobs in `config3.py` per
model — exp 2 showed canary difficulty does not transfer across models.
`--sets`, `--conditions`, `--limit` subset the run. A third, smaller model
(`llama-v3p1-8b` on Fireworks) is configured for the size-ladder question;
opt in with `--models`.

`selftest3` synthesises a perfect agent's reply for every turn × condition ×
set (≈76k checks) and asserts zero false positives, then injects wrong-value
faults into every canary line and every task answer and asserts each is
caught. It runs before any API spend.

## Files

```
experiments3/
  config3.py     task sets, conditions, difficulty knobs, models, K windows
  payloads3.py   domain-independent canary payload generator
  tasks3.py      augments exp-1/exp-2 tasks; builds bAbI sessions
  canaries3.py   instructions, score-based checkers, per-domain hallucination checks
  runner3.py     trajectory loop (early stop at first hallucination, resumable)
  judge3.py      LLM-judge baseline, per task set
  metrics3.py    scoring (imports classify/summarize from experiments/metrics.py)
  figures3.py    the standard 4 figures, once per task set
  selftest3.py   offline validation of every checker on all three sets
  run_all3.py    orchestrator (--sets --models --conditions --limit --pilot)
data3/
  raw/babi_train.jsonl   published bAbI items (Muennighoff/babi mirror)
  tasks3_{coding,registers,babi}.json
```
