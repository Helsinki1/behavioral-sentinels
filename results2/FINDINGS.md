# Experiment 2 — Findings

**Headline: a canary only works as a sentinel if its answer cannot be copied
from the model's own previous reply.** Changing the answer every turn is *not*
what makes a probe informative — several of the most "dynamic" canaries turned
out to be the least informative ones.

All numbers below: `gpt-oss-20b`, 200 incremental-coding tasks, horizons 12–30
turns, prediction window K=5, paired task set. Full tables in
[`SUMMARY.md`](SUMMARY.md).

## 1. The copyability gradient

| canary | what it demands | fire rate | recall@5 | **F1@5** |
|---|---|---|---|---|
| `static_trailer` | nothing (fixed string) | 0.000 | 0.000 | **0.000** |
| `rotating_prefix` | +1 on your own last line | 0.000 | 0.000 | **0.000** |
| `stochastic_policy` | hold a token that's in your last line | 0.035 | 0.025 | **0.048** |
| `lagged_echo` | shift a buffer off the *user's* prior message | 0.270 | 0.121 | **0.202** |
| `conditional_rule` | recompute a branch each turn | 0.465 | 0.173 | **0.255** |
| `escalating_ledger` | running counts + long-range recall | 0.550 | 0.321 | **0.421** |

The ordering is not "static → dynamic". It is **"copyable → must be
recomputed"**:

- `rotating_prefix` changes *every single turn* — the strongest possible form of
  the "change it every turn" idea — and it **never fired once in 400
  trajectories across both models**. The reason is visible in the transcripts:
  the previous line (`BUILD 27`) is sitting right there in context, so emitting
  `BUILD 28` is a local copy-and-increment that survives even after the model
  has completely lost track of the module.
- `stochastic_policy` changes at unpredictable turns — the "change it at random
  times" idea — and barely fired (0.035). Same reason: between refreshes the
  token is verbatim in the model's own last reply.
- The probes that *do* fire are the ones whose answer is **not** present in the
  previous reply: a one-turn lag off the user's message, a per-turn branch that
  must be recomputed, and an accumulating ledger that must be maintained.

This is a sharper restatement of experiment 1's result. There, `variable_check`
was the only canary that worked, and it was described as "the one whose value
changes each turn". Experiment 2 shows that description was the wrong
explanation: `rotating_prefix` also changes each turn and is worthless.
`variable_check` worked because it required a **comparison against the previous
user message** — a recomputation.

## 2. Does it beat the "context window is running out" rebuttal?

This is the objection named in `README.txt`. Against the traditional signals on
the same 200 tasks:

| signal | precision | recall | **F1@5** | fire rate |
|---|---|---|---|---|
| `escalating_ledger` (best canary) | 0.612 | 0.321 | **0.421** | 0.550 |
| `Traditional/turn_number` (θ=10) | 0.500 | 0.299 | 0.374 | 0.655 |
| `Traditional/context_length` (θ=1000) | 0.471 | 0.266 | 0.340 | 0.730 |
| `Traditional/random_compaction` | 0.464 | 0.258 | 0.332 | 1.000 |
| `Traditional/LLM_judge` | 0.610 | 0.468 | **0.529** | 0.975 |

**Yes for the cheap baselines, no for the judge.** `escalating_ledger` beats
context length, turn number and random compaction on F1, and does so at a
*lower* fire rate (0.55 vs 0.73/0.66/1.00) — it is more selective, not just
more trigger-happy. That is the rebuttal the README asks for: the canary is not
merely a proxy for "the context is getting long".

But the **LLM judge outperforms every canary here (F1 0.529)**, which did *not*
happen in experiment 1 (judge F1 0.230 on this model). This should be reported
plainly rather than buried:

- The judge's fire rate is **0.975** — it says "degraded" on essentially every
  trajectory, so as a *compaction trigger* it is close to always-on. Its value
  is therefore almost entirely in **when** it first fires, not in whether it
  fires. Its precision (0.610) is genuinely better than random compaction's
  (0.464) at a comparable fire rate, so this is a real result, not an artifact
  — but a trigger that fires on 97.5% of trajectories buys little over "compact
  on a schedule" unless the timing is acted on.
- The judge costs an **extra LLM call per turn**. A canary costs ~10 tokens
  appended to a response the agent was already producing.
- Caveat: this judge ran on `gpt-oss-20b`, not `gpt-4o-mini` as in experiment 1
  (see Limitations).

## 3. Lead time

Where `escalating_ledger` fires at or before the hallucination, the median lead
is **4 turns** (`conditional_rule` 7, `lagged_echo` 6) — enough runway to
trigger state compaction before the agent starts emitting broken code.

Note the honest reading of `fig3_onset_curves.png`: the dashed (signal) curve
sits *below* the solid (outcome) curve. That is because the canary fires on
fewer trajectories than hallucinate — **not** because it fires later on the
ones it does catch. Lead time is a per-trajectory quantity and is reported as
such in `fig4_lead_time.png`.

## 4. What coding hallucination actually looks like

From the baseline (no-canary) runs, the first hallucinating turn was:

| kind | share |
|---|---|
| wrong signature reported | 24% |
| **call to a symbol that does not exist** | 22% |
| emitted `def` has wrong params | 18% |
| no code emitted at all | 17% |
| requested edit not emitted | 14% |
| other (rarer kinds) | 5% |

The "nonexistent API call" failure mode from the taxonomy is not a corner case
— it is more than a fifth of first failures. The most common single trigger is
an agent renaming a function and then immediately calling it by its old name.

## Limitations

1. **The `gpt-4o-mini` arm is incomplete.** The OpenAI account's
   requests-per-day allowance (10,000, rolling 24h) was exhausted partway
   through the run. `gpt-oss-20b` is complete (200 tasks × 7 conditions);
   `gpt-4o-mini` completed only 3 of 7 conditions and is therefore excluded
   from the scored tables, which require a paired task set across all
   conditions. What it did produce corroborates the headline:

   | condition | n | hallucination rate | canary fire rate |
   |---|---|---|---|
   | `rotating_prefix` | 200 | 0.775 | **0.000** |
   | `stochastic_policy` | 193 | 0.756 | 0.021 |
   | `lagged_echo` | 42 | 0.976 | 0.024 |

   `rotating_prefix` never fired on either model — 400 trajectories, zero
   firings — so the central negative result is not model-specific.

   To finish the arm once quota resets:
   ```bash
   python -m experiments2.run_all2 --models gpt-4o-mini
   python -m experiments2.figures2
   ```
   The runner is resumable and skips everything already on disk.

2. **The LLM judge ran on `gpt-oss-20b`, not `gpt-4o-mini`** as experiment 1
   did, for the same quota reason. The judge model is recorded in
   `results2/Traditional/LLM_judge/metrics_*.json` (`judge_model`). Since the
   judge is the strongest baseline here, this substitution should be re-run
   with `gpt-4o-mini` before the comparison is treated as settled.

3. **Single task family.** All 200 tasks are the same *kind* of coding task
   (signature/symbol-table maintenance). The failure modes it induces are real
   but narrow; SWE-Bench-style tasks would exercise different ones.

4. **Turn-1 hallucinations** (~20% of trajectories) leave no room for a lead
   signal and count as FN for every signal. They depress recall uniformly
   across signals, so comparisons remain fair, but absolute recall figures
   would be higher if these were excluded.
