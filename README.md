# Behavioral Sentinels

Early-warning signals for hallucination onset in long-horizon LLM agents.

The research design, taxonomy and motivation live in [`README.txt`](README.txt).
This file indexes the ten experiments in the repo, newest first.

**A note on numbering and layout.** Experiments are numbered in the order they
were *run*. Experiment 10 reuses the experiment-4 harness (it needs that
regime — see its section), so its arms live in `experiments4/config4.py` and
its trajectories in `runs4/`, while its analysis, results and write-up live in
`experiments10/` and `results10/`. Everywhere else the mapping is the obvious
one: experiment *n* → `experimentsN/`, `resultsN/`, `README_EXPERIMENTN.md`.

## Experiment 10 — the carried-sentinel 2×2, and the break-even surface (latest)

Two studies that price observation rather than ranking triggers.
[`README_EXPERIMENT10.md`](README_EXPERIMENT10.md) · [`FINDINGS`](results10/FINDINGS.md)

**Study B0 — does a perfect carried sentinel pay?** Exps 3–4 estimated this by
*subtracting* the carrying cost from the timing prize, which assumes the two
effects are additive. They are not. Completing the 2×2 (n=100 coding tasks,
exp-4 compaction regime — the operator under which a timing prize exists):

|  | no useful reset | oracle-timed reset |
|---|---|---|
| **no sentinel** | A = 0.849 | B = **0.908** |
| **carries probe** | C = 0.823 | D = 0.851 |

The timing prize is real (`B − A` = **+0.059**, sig), but granting a carried
sentinel *perfect* timing leaves it **exactly break-even**: `D − A` = **+0.002,
CI [−0.037, +0.040]**. The additive model predicted 0.883 against an observed
0.851 — an error of −0.032, stable across n. Timing is worth +0.059 without a
probe but only +0.027 while carrying one, so **a probe degrades the value of
good timing itself** rather than levying a fixed toll. This is a precise
*null*, not a negative: break-even, not harmful. Scope: one probe, one model,
one task family — and exp 9 shows the observer effect's *sign* is
regime-dependent, so this is a statement about this regime, not a constant.

**Study B1 — the penalty is compaction-specific (mechanism confirmed).** B0
attributed the sub-additivity to the agent's self-summary having to reproduce
the probe's ledger, and predicted it would vanish under deterministic
re-grounding. It does. Re-running only cells B and D (A and C never reset, so
they are operator-independent and shared):

| contrast | compaction | re-grounding |
|---|---|---|
| timing value, no probe | +0.059 ✱ | +0.063 ✱ |
| timing value **while carrying** | +0.028 | **+0.065** ✱ |
| **interaction** | **−0.032** | **+0.001** |
| **does a perfect carried sentinel pay?** | +0.002 | **+0.039** ✱ |

Under re-grounding the probe stops corroding the repair operation entirely —
timing is worth the same with or without it — and the carried sentinel becomes
significantly profitable. **Carried probes are not fundamentally unable to pay;
they are unable to pay when recovery is lossy self-summarisation.** The
carrying cost itself (−0.026) was never the problem — the interaction was.
[`STUDY_B1.md`](results10/STUDY_B1.md)

**Study A — the break-even surface.** Scoring every exp-5/6 policy under
`U = accuracy − R·restarts − T·prompt_ktokens` and taking the upper envelope
turns "sentinel vs clock" into a decision boundary: under a lossless operator
the clock wins only while restarts are nearly free (R < 0.003), the zero-carry
sentinel owns the large middle region, and never-reset takes over once restarts
are expensive; under a lossy operator never-reset dominates almost everywhere.
Descriptive decision analysis over existing runs, not a causal claim.
[`STUDY_A.md`](results10/STUDY_A.md)

## Experiment 9 — sharded GSM8K x four models

Exp 8's active-vs-passive design re-run on a **respected external
benchmark** — the `math` split of *LLMs Get Lost in Multi-Turn Conversation*
(arXiv:2505.06120), 3 sharded GSM8K problems concatenated per session — and
**four models** (`gpt-oss-120b`, `deepseek-v4-flash`, `qwen3p7-plus`,
`gpt-4o-mini`) through one harness, with fully decidable per-turn
hallucinations (premature ANSWER / missing WAIT / wrong final ANSWER vs the
GSM8K key). Design: [`README_EXPERIMENT9.md`](README_EXPERIMENT9.md) ·
results: [`results9/SUMMARY.md`](results9/SUMMARY.md) ·
[`results9/PREDICTION.md`](results9/PREDICTION.md) ·
[`results9/FINDINGS.md`](results9/FINDINGS.md). Headline: **the observer
effect changes sign with the model** — carrying the probe costs −0.036 (exp
8, oss-20b) and −0.016 (qwen3.7) but *pays* **+0.021** on gpt-oss-120b
(sig), where the matched chore acts as an attention-refresh *treatment* and
the active arm is the best arm outright (0.977); only one of four models
shows a true lost-in-conversation regime under verbatim shards; the
frozen-state quiz's precise-but-blind profile replicates across benchmark
and models (a replicated negative result).

## Experiment 8 — active vs passive observation

The project's reframe: not "which sentinel is best" but **how should you
observe a long-horizon agent at all**, with the taxonomy split on whether the
observation *writes into the agent's trajectory* — **active** (exp 1–5's
carried probes), **passive-behavioural** (new: a frozen-state quiz asked on a
forked, discarded copy of the conversation — zero contamination, fork tokens
only), and **passive-observational** (exp 5's zero-carry trace monitors, the
LLM judge). Same 90-task pool, reset operator fixed to exp 6's R1 reground;
only three arms are new (`QUIZ`, `ACT_probe`, `ACT_carry_clock`) plus an
offline **shadow pass** that asks the quiz retrospectively on `runs5`
prefixes, so quiz precision/recall is scored on the same full-horizon
trajectories as every passive baseline. Design:
[`README_EXPERIMENT8.md`](README_EXPERIMENT8.md) · results:
[`results8/SUMMARY.md`](results8/SUMMARY.md) ·
[`results8/PREDICTION.md`](results8/PREDICTION.md) ·
[`results8/FINDINGS.md`](results8/FINDINGS.md). Headline: the category
ordering is **monotone on both axes** (trace monitor > quiz > carried probe
in signal precision 0.78/0.60/≤0.10 *and* deployed accuracy
0.841/0.813/0.792 — no Intervention-Paradox inversion); the observer effect
is priced at **−0.036 accuracy (sig, all domains)** at a matched reset
schedule, replicating exp 4's −0.043 under a different operator; the quiz
removes the contamination but its recall is too low to pay for its 9.3K
fork tokens; and exp 6's law extends to observation itself — **when
restarts are cheap and loss-free, the best observer is a clock.** Decision
table in FINDINGS §5.

## Experiment 6 — sentinel-triggered re-grounding

The faithful test of the original question: *when should you start a new
Claude Code session?* Exp 5 showed compaction-style resets lose more than
timing gains; exp 6 changes the reset operator to what a fresh Claude Code
session actually does — **re-read true state from an external store** (a
harness reducer plays the file system; store == generator truth proven on
all 8,535 turns) — plus a verbatim-transcript **replay** bracket, on the same
90-task pool with `A_no_reset` imported from runs5 for exactly-paired
operator contrasts. Design: [`README_EXPERIMENT6.md`](README_EXPERIMENT6.md)
· results: [`results6/SUMMARY.md`](results6/SUMMARY.md) ·
[`results6/FINDINGS.md`](results6/FINDINGS.md). Headline: the regime
**flips** — scheduled re-grounded restarts beat never restarting (+0.025,
sig) at 43% fewer tokens, perfectly-timed single restarts add nothing
(sig below the clock), and the transcript-replay bracket collapses (−0.124,
sig) with the zero-carry sentinel significantly beating the clock there
(+0.075) — yielding one law: **the lossier the restart, the more timing
matters; the cheaper the restart, the more frequency wins.** Decision table
in FINDINGS §6.

## Experiment 5 — task-conditioned sentinel routing

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

# experiment 6 — deployment: re-grounding vs replay restarts
python -m experiments6.store6            # offline: prove store == truth
python -m experiments6.selftest6         # offline, mock LLM, all arms
python -m experiments6.run_all6 --gate   # A (import), C_clock, F_oracle
python -m experiments6.run_all6          # all eleven arms
python -m experiments6.metrics6
python -m experiments6.figures6

# experiment 10 — is a carried sentinel worth it? (2x2) + break-even surface
python -m experiments4.run_all4 --limit 100 \
       --arms A_no_reset,P_carry_noreset,F_oracle,P_carry_oracle
python -m experiments10.factorial10    # the 2x2 with paired bootstrap CIs
python -m experiments10.breakeven10    # decision map (re-analysis, no API calls)
python -m experiments10.figures10
```

Every runner is resumable and skips trajectories already on disk.

## What would move this forward

1. **Agent-corrupted stores.** Exp 6's store derives entirely from user
   instructions, so restarts always re-read *correct* state; a real repo also
   contains the agent's mistaken edits. Letting the agent's own (possibly
   wrong) writes land in the store — and measuring whether scheduled
   re-grounding still wins — closes the last fidelity gap.
2. **An oracle over schedules.** The single-reset oracle retired with exp 6
   (repetition beats placement when restarts are cheap); the true ceiling is
   the best reset *schedule* per task, searchable offline on recorded
   trajectories.
3. **A second model on experiments 2–6.** Still the biggest hole
   (`gpt-4o-mini` arms scaffolded everywhere, rolling-RPD-limited).
4. **Restart overhead priced in.** Exp 6's decision table hinges on fixed
   restart cost (re-onboarding, cache loss, human time); adding an explicit
   per-reset penalty sweep would locate the crossover where the zero-carry
   sentinel overtakes the schedule.
5. **An unseen genre.** Every domain tested had an exp-3-screened probe and
   a hand-built monitor; a spatial or plan-tracking family would test the
   taxonomy and the store abstraction where nothing was tuned.
