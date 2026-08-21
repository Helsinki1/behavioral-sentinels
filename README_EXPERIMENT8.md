# Experiment 8 — Active vs. passive observation: what does watching an agent cost, and what does it buy?

## The reframe

Experiments 1–6 asked *"can a sentinel beat a clock?"* Experiment 8 reorganises
the same machinery around a cleaner question: **how should you observe a
long-horizon agent?** The taxonomy's dividing line is whether the observation
**writes into the agent's trajectory**:

- **Active observation** — the monitor injects work into the agent's own
  context: the carried probes of exp 1–5. The agent answers the chore inside
  the session; the chore stays in context; the observation itself degrades
  the trajectory (exp 3/4's observer effect, −0.043 accuracy).
- **Passive-behavioural observation (new)** — the harness *freezes* the
  agent's context at a checkpoint, forks it, asks a short quiz about current
  state and direction in the forked call, and **discards the exchange** —
  nothing is ever written to agent state or conversation history. Costs
  tokens, contaminates nothing.
- **Passive-observational** — no extra queries at all: monitors read the
  reasoning/action trace the agent already produces (exp 5's zero-carry
  self-consistency monitors; the LLM judge).

Every method is evaluated three ways — signal quality, downstream gain when
its signal routes exp-6 re-grounded restarts, and total cost of observation —
on **one shared task pool**, so the three figures share an x-axis and the
reader sees in one glance whether signal quality *converts* into gain (the
Intervention Paradox question, arXiv:2602.03338) and what each signal cost.

## Comparability contract (nothing is compared across pools)

Everything runs on the exp-5/6 pool: **90 tasks (30 coding / 30 registers /
30 babi), difficulty-stratified from `data3`**, full horizon, no early stop,
per-turn accuracy, 6-reset cap, 2-turn post-reset grace for behavioural
triggers, reset operator fixed to **R1 `reground`** (`store6.py`; store ==
generator truth proven on all 8,535 turns). `A_no_reset` is imported verbatim
from `runs5`, as in exp 6, so all contrasts are exactly paired on task.
Exp 1–4 numbers are cited as motivation only; **no cross-pool number appears
in any exp-8 figure.**

## Methods under test (the shared x-axis)

| category | method | signal | trajectories |
|---|---|---|---|
| active | `ACT_probe` | labeled carried probe fails (exp 5 `D_labeled` design — deterministic genre lookup, no router; router noise already priced at −0.005 ns) | **new runs** (exp 5 ran this trigger only with the compaction operator) |
| passive-behavioural | `QUIZ` | frozen-state quiz score drops below threshold | **new runs** + offline shadow pass |
| passive-observational | `Z_trace` | zero-carry self-consistency monitor fires | **reuse `runs6/Z_reground`** |
| passive-observational | `JUDGE` | LLM judge (window 8) says degraded | **reuse `runs6/C_judge`** |
| baseline | `CLOCK` / `CTX` / `RANDOM` | every 6 turns / ≥600-token growth / budget-matched random | **reuse `runs6`** |
| bounds | `NONE` / `ORACLE` | never / once before `A_no_reset`'s first failure | **reuse `runs5/A_no_reset`, `runs6/F_oracle`** |

One auxiliary arm isolates observer-effect cost under the exp-8 operator:

- `ACT_carry_clock` — carries the labeled probe but resets **on the clock**
  (exp 5's `C_prime` design, re-run with R1). `ACT_carry_clock − CLOCK` is
  the pure price of *being actively observed* at an identical intervention
  schedule; `ACT_probe − ACT_carry_clock` is the value of the active signal's
  timing. Both differences are paired on task.

## The quiz (the one genuinely new component)

At each checkpoint (every 3 turns — `G_dense`'s cadence, so quiz frequency is
never the confound vs. the densest baseline schedule):

1. freeze the conversation; copy it into a fork;
2. append one quiz message with 4 questions, drawn deterministically from the
   task generator's own fields so grading needs **no LLM judge** for 3 of 4:
   - **recall** — value of a register touched ≥3 turns ago / signature of a
     function / where a story entity is (graded against `store6` truth);
   - **artifact** — which functions (registers, stories) have been modified /
     deleted so far (graded against store diff);
   - **update** — apply the most recent instruction to the state as of the
     checkpoint (graded against store);
   - **direction** — "what should the next turn do?" (graded by the exp-5
     judge prompt; reported separately, never mixed into the trigger).
3. discard the fork. Nothing enters agent state; the trajectory file records
   the quiz Q/A in a side-channel field for audit only.

Trigger rule: reset when ≥2 of the 3 deterministic questions are wrong at a
checkpoint. Threshold ablation (≥1, ≥2, 3) is offline — it re-reads the
shadow-pass scores, no new agent runs.

**Shadow pass (recycling instead of re-running):** because `runs5/A_no_reset`
stores full per-turn conversation state, the quiz can be asked *retro-
spectively* on every checkpoint prefix of those 90 trajectories. This yields
the quiz's precision / recall / lead-time on the **identical trajectories**
where every passive signal is already scored (`prediction5.py`'s same-
trajectory protocol) — no new agent trajectories, only quiz calls on frozen
prefixes (~90 × 5 checkpoints). The active probe's signal quality cannot be
scored on those trajectories (the probe isn't present); it is scored on its
own arm's pre-first-reset segments with the same K=5 window, and the figure
marks it as observer-shifted, exactly as exp 5 did.

## The three claims, one figure each (plus taxonomy and decision rule)

1. **Fig 1 — taxonomy.** The active / passive-behavioural /
   passive-observational split, methods placed, with "writes into the
   trajectory?" as the axis. (No data.)
2. **Fig 2 — signal quality.** Precision / recall (and lead time) at
   predicting first failure within K=5 turns, per method, grouped by
   category; clock / ctx / random thresholds re-tuned per trajectory set as
   reference lines. Source: `prediction5` numbers + shadow pass + `ACT_probe`
   segments.
3. **Fig 3 — downstream gain.** End-task accuracy of each method's
   signal-routed re-ground arm, same x-axis order as Fig 2; `NONE`, `CLOCK`,
   `RANDOM`, `ORACLE` as horizontal reference lines; paired deltas with
   permutation tests as in exp 5/6. The Fig 2 → Fig 3 comparison *is* the
   Intervention Paradox test: does the best signal produce the best outcome?
4. **Fig 4 — total cost of observation.** Two panels, same x-axis:
   (a) observer-effect accuracy delta — `ACT_carry_clock − CLOCK` for active;
   **structurally zero** for both passive categories (nothing enters the
   trajectory), shown as such; (b) monitoring tokens per task (probe payload
   tokens in-context for active; quiz fork tokens for `QUIZ`; judge-call
   tokens for `JUDGE`; ~0 for `Z_trace`).
5. **Fig 5 — decision rule.** Accuracy vs. total token spend (agent + resets
   + monitoring), one point per arm, Pareto front drawn; extends the exp-6
   decision table with one row per observation category. Target takeaway of
   the form: *"passive quizzing captures X% of the active probe's signal at
   zero observer cost, and converts it to Y% of the oracle gap at Z fewer
   tokens."*

## What would count as success

The exposition works regardless of direction, but the hypotheses are:

- **H1 (dominance):** `QUIZ` ≥ `ACT_probe` on Fig 2 *and* Fig 3 while paying
  zero observer cost on Fig 4 — active observation is strictly dominated;
  the chore's predictive power never required contaminating the context.
- **H2 (surprise):** `ACT_probe` > `QUIZ` on signal — the *in-context* nature
  of the chore is itself what makes it predictive, i.e. the observer effect
  and the signal are inseparable. Either outcome is a paper.
- **H3 (translation):** rank order of Fig 2 ≠ rank order of Fig 3 — signal
  quality does not convert to gain — would replicate the Intervention Paradox
  under a loss-free operator, where exp 5/6 (zero-carry matching the oracle)
  suggest it *does* convert. Precision, not recall, should predict Fig 3 rank
  (exp 5's finding).

## Cost of the experiment

New agent runs: 3 arms × 90 tasks (`QUIZ`, `ACT_probe`, `ACT_carry_clock`) —
the same order of magnitude as one exp-6 gate run. Shadow pass: ~450 quiz
calls on frozen prefixes, no agent loop. Everything else — 8 comparison arms,
both bounds, all baseline signal scores — is read from `runs5`/`runs6` and
`results5/PREDICTION.md` machinery. Model: `gpt-oss-20b`, as exp 5/6.

## Run

```bash
python -m experiments8.selftest8           # offline, mock LLM, all arms + shadow pass
python -m experiments8.shadow8             # quiz shadow pass over runs5/A_no_reset
python -m experiments8.run_all8 --gate     # QUIZ arm only, 15 tasks
python -m experiments8.run_all8            # QUIZ, ACT_probe, ACT_carry_clock
python -m experiments8.metrics8            # results8/SUMMARY.md (+ imports runs5/6)
python -m experiments8.figures8            # results8/figures/ (Figs 2–5)
```

## Limitations stated up front

Synthetic state-tracking pool (clean ground truth, cheap oracles; one model);
the quiz's deterministic grading leans on the same generator truth that
defines the benchmark — fair *within* the pool, but transfer to tasks without
a harness-computable truth (real repos) is the explicit next step, per the
workshop-vs-main-track bar. The active probe's signal quality is measured on
observer-shifted trajectories by necessity; the figure says so. Quiz
checkpoints pause wall-clock but not the agent's context — latency cost is
reported, not modelled.
