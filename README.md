# Behavioral Sentinels

Early-warning signals for hallucination onset in long-horizon LLM agents.

The research design, taxonomy and motivation live in [`README.txt`](README.txt).
This file indexes the five experiments in the repo.

## Experiment 5 — task-conditioned sentinel routing (latest)

Experiments 1–4 deployed one blanket probe on every task. Experiment 5 routes
instead: an LLM router classifies each task's dominant state demand (five-genre
taxonomy) and attaches the one exp-3 probe matched to that failure mechanism —
plus a **zero-carry** routed variant whose monitors read self-consistency off
output the agent already produces. 11 arms × 90 mixed tasks (coding /
registers / babi), deployment protocol (full horizon, end-task accuracy,
budget-capped resets). Design: [`README_EXPERIMENT5.md`](README_EXPERIMENT5.md)
· results: [`results5/SUMMARY.md`](results5/SUMMARY.md) ·
[`results5/FINDINGS.md`](results5/FINDINGS.md) ·
[`results5/PREDICTION.md`](results5/PREDICTION.md). Headline: the zero-carry
routed sentinel **matches the perfect-timing oracle** (0.817 accuracy at ~1
reset per task) and outscores clock, context, judge and random triggers
pooled; routing beats blanket and anti-routed probes exactly where the
assignment changes the probe; and in this regime the remaining bottleneck is
compaction loss, not signal quality — even the oracle no longer beats never
resetting. Two follow-ups sharpen the mechanism: a deterministic pre-labeled
arm (`D_labeled`) shows the LLM router's noise costs nothing (−0.005 vs
`D_routed`, ns), and a same-trajectory prediction layer shows the zero-carry
signal's edge is **precision** (0.78 vs 0.60 turn-count / 0.46 context /
0.46 random on identical trajectories) — the property that matters when
every reset risks compaction loss.

## The first four experiments

|  | **Experiment 1** | **Experiment 2** | **Experiment 3** | **Experiment 4** |
|---|---|---|---|---|
| question | can a task-irrelevant chore fire *before* a task hallucination? | does it matter whether the chore's answer **moves**? | **which axis of strain** degrades first — and does it replicate across domains? | does **acting** on a sentinel beat acting on a clock? |
| kind | prediction | prediction | prediction | **deployment** |
| task | integer register book-keeping | incremental Python coding | **three sets**: coding, registers, published **bAbI** | exp-2 coding tasks, full horizon |
| probes | 6 **static** | 6 **dynamic** + static control | 7 **axis-isolating** + ensemble + controls | 1 (`escalating_ledger`) × 6 reset policies |
| code | [`experiments/`](experiments/) | [`experiments2/`](experiments2/) | [`experiments3/`](experiments3/) | [`experiments4/`](experiments4/) |
| data | `data/tasks.json` | `data2/tasks2.json` | `data3/tasks3_*.json` | reuses `data2/` |
| trajectories | `runs/` | `runs2/` | `runs3/<set>/` | `runs4/` |
| results | [`results/`](results/SUMMARY.md) | [`results2/`](results2/SUMMARY.md) | [`results3/`](results3/SUMMARY.md) | [`results4/`](results4/SUMMARY.md) |
| write-up | — | [`README_EXPERIMENT2.md`](README_EXPERIMENT2.md) · [`FINDINGS`](results2/FINDINGS.md) | [`README_EXPERIMENT3.md`](README_EXPERIMENT3.md) · [`FINDINGS`](results3/FINDINGS.md) | [`README_EXPERIMENT4.md`](README_EXPERIMENT4.md) · [`FINDINGS`](results4/FINDINGS.md) |

Experiments 1–3 all ask the same question — *does a signal light up before a
failure* — and score it with the **same code**: `metrics2.py` and `metrics3.py`
import `classify` / `summarize` / `summarize_random` directly from
`experiments/metrics.py` rather than reimplementing them. Any difference in
those numbers comes from the task domain and the probe design, never from the
measurement.

Experiment 4 changes the question. It measures **end-task accuracy** under
different reset policies at a matched intervention budget, with no early stop.

## State of the findings

**No *carried* probe has ever beaten a clock — but the zero-carry routed
sentinel of experiment 5 finally does** (pooled point estimates over all four
traditional triggers, significant vs clock and judge on coding, at the lowest
intervention budget of any behavioural arm). Across the first four
experiments, three task domains and 22 probe designs, "compact every N turns"
matched or beat every bolted-on sentinel tried. What the project produced
along the way:

1. **A probe only fires if its answer cannot be copied from the model's own
   previous reply** (exp 2). `rotating_prefix` changes every single turn — the
   strongest form of "make it dynamic" — and fired **0 times in 400
   trajectories** while the task underneath broke ~78% of the time, because
   `BUILD 27` → `BUILD 28` is a local copy that survives total state loss.
   This also corrects experiment 1's explanation of its own result: its one
   working probe won by requiring a *comparison*, not by changing.

2. **Probes cost what they measure** (exp 3, priced in exp 4). Carrying a probe
   raises the failure rate — up to +0.32 in exp 3's binary measure, and −0.043
   task accuracy (a ~29% relative increase in errors) in exp 4's per-turn
   measure. Even a *fixed string* costs +0.17 on registers, so part of the toll
   is the presence of an extra standing instruction, not its difficulty.

3. **Placement beats frequency** (exp 4). Resetting on a clock buys **zero**
   accuracy (−0.0003) despite 3.1 resets per task. A perfect-timing oracle
   beats it by **+0.059 (95% CI [+0.012, +0.108])** using **0.78** resets per
   task. Resets are not intrinsically good; landing on the right turn is.

4. **The carrying cost eats most of the prize.** Total headroom for perfect
   timing is +0.059; carrying a probe costs −0.043. A *flawless* signal that
   must be carried nets ≈ **+0.016** — within noise. The viable direction is a
   signal with **zero carrying cost**, read off work the agent is already
   doing, not an extra chore bolted on.

### A measurement pitfall worth knowing about

Scoring a probe on **its own** trajectories while scoring baselines on
**clean** trajectories inflates the probe, because the observer effect raises
the base rate and precision then rises mechanically for *any* signal —
including a clock. Experiment 3's summary tables do this, and its ensemble
headline does not survive correction: re-scoring `turn_number` on each probe's
own trajectories gives **0 wins, 2 ties, 22 losses** across all sets and probes
(the two ties are the ensemble at +0.011 and +0.016). Experiment 2 is much less
affected — its probes induce a far smaller observer effect — and
`escalating_ledger` still beats a fixed turn-5 clock there by +0.047.

**Any future comparison must score every signal on the same trajectories.**

### Definitions (from `README.txt`)

- **Degradation** — losing track of state, violating constraints, abandoning
  parts of tasks, fabricating facts not explicitly given.
- **Hallucination** — a degradation event that actually impacts the task at hand.
- **Sentinel / Canary** — a degradation event on a task-irrelevant chore, nuance
  or probe.

## Models

`gpt-4o-mini` (proprietary, OpenAI) and `gpt-oss-20b` (open, Fireworks).
Experiment 1 ran both arms in full. Experiments 2–5 are `gpt-oss-20b`-complete;
the `gpt-4o-mini` arm is partial in exp 2 and not yet run in exps 3–5, limited
by the OpenAI account's rolling requests-per-day allowance. Experiment 2 found
probe difficulty does **not** transfer across models (a probe firing on 27% of
`gpt-oss-20b` trajectories fired on 1% of `gpt-4o-mini`'s), so the cross-model
check is a real gap, not a formality.

## Quick start

```bash
printf 'OPENAI_API_KEY=...\nFIREWORKS_API_KEY=...\n' > .env   # gitignored

# experiment 1 — static probes, registers
python -m experiments.run_all
python -m experiments.figures

# experiment 2 — dynamic probes, coding
python -m experiments2.selftest2      # validate checkers offline, no API calls
python -m experiments2.run_all2
python -m experiments2.figures2

# experiment 3 — axis-isolating probes, three task sets
python -m experiments3.selftest3      # offline, all three sets
python -m experiments3.run_all3       # see README_EXPERIMENT3.md for flags
python -m experiments3.figures3

# experiment 4 — deployment: do sentinel-triggered resets help?
python -m experiments4.run_all4 --gate   # A, C, F only — the go/no-go gate
python -m experiments4.run_all4          # all six arms
python -m experiments4.metrics4
python -m experiments4.figures4

# experiment 5 — deployment: routed + zero-carry sentinels, mixed pool
python -m experiments5.selftest5         # offline, mock LLM, all arms
python -m experiments5.run_all5          # all twelve arms
python -m experiments5.metrics5
python -m experiments5.prediction5       # same-trajectory precision/recall
python -m experiments5.figures5
```

Every runner is resumable and skips trajectories already on disk.

## What would move this forward

1. **Better compaction, not better signals.** Experiment 5 put a free signal
   at oracle-level timing and found the oracle itself no longer beats never
   resetting — the loss is in the snapshot operator now. Hybrid snapshots
   (schema + verbatim recent tail), external state checkpoints, and
   selective compaction are the candidates.
2. **A second model on experiments 2–5.** The single-model limitation is the
   biggest hole, and exp 2 already showed the effect is model-dependent
   (`gpt-4o-mini` arms are scaffolded everywhere, rolling-RPD-limited).
3. **Experiment 5 at n ≈ 400.** The pooled Z_routed contrasts are +0.002 to
   +0.021 with CIs of ±0.03; the per-domain anchors are significant but the
   pooled ranking is not yet.
4. **An unseen genre.** The router's taxonomy is open-ended, but every genre
   tested so far had an exp-3-screened probe waiting. A spatial or
   plan-tracking task family would test routing where no probe was tuned.
