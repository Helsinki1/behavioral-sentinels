# Behavioral Sentinels

Early-warning signals for hallucination onset in long-horizon LLM agents.

The research design, taxonomy and motivation live in [`README.txt`](README.txt).
This file indexes the three experiments in the repo.

| | **Experiment 1** | **Experiment 2** | **Experiment 3** |
|---|---|---|---|
| question | can a task-irrelevant chore fire *before* a task hallucination? | does it matter whether the chore's answer **moves**? | **which axis of strain** degrades first — and does it replicate across domains? |
| task | integer register book-keeping | **incremental Python coding** | **three sets**: coding (exp-2 tasks), registers (exp-1 tasks), published **bAbI** stories |
| canaries | 6 **static** probes | 6 **dynamic** probes + 1 static control | 7 **axis-isolating** probes + ensemble + controls |
| code | [`experiments/`](experiments/) | [`experiments2/`](experiments2/) | [`experiments3/`](experiments3/) |
| data | `data/tasks.json` | `data2/tasks2.json` | `data3/tasks3_*.json` |
| trajectories | `runs/` | `runs2/` | `runs3/<set>/` |
| results | [`results/SUMMARY.md`](results/SUMMARY.md) | [`results2/SUMMARY.md`](results2/SUMMARY.md) | `results3/SUMMARY.md` + `results3/<set>/` |
| figures | `results/figures/` | `results2/figures/` | `results3/figures/<set>/` (the same 4 charts, once per set) |
| write-up | — | [`README_EXPERIMENT2.md`](README_EXPERIMENT2.md), [`results2/FINDINGS.md`](results2/FINDINGS.md) |

**Experiment 4** ([`experiments4/`](experiments4/), [`results4/FINDINGS.md`](results4/FINDINGS.md))
moves from prediction to **deployment**: does acting on a sentinel beat acting on a
clock, at a matched reset budget? Answer: no — the sentinel loses — but a *perfect*
predictor beats the clock by +0.059 accuracy (95% CI [+0.012, +0.108]) using four
times fewer resets. Placement, not frequency, is the lever. [`README_EXPERIMENT3.md`](README_EXPERIMENT3.md), [`results3/FINDINGS.md`](results3/FINDINGS.md) |
| headline | a task-irrelevant chore can fire before a task hallucination | a canary is only a sentinel if its answer **can't be copied from the model's own last reply** | no single axis wins everywhere; a **3-probe ensemble** beats every cheap baseline on the non-saturated domains — and canaries **accelerate the failures they predict** (observer effect) |

Both experiments use the **same two models** (`gpt-4o-mini` proprietary,
`gpt-oss-20b` open via Fireworks), the **same N** (200 tasks), the **same
prediction windows** (K = 2, 5, 10, ∞) and — importantly — the **same scoring
code**: `experiments2/metrics2.py` imports `classify`, `summarize` and
`summarize_random` directly from `experiments/metrics.py` rather than
reimplementing them. Any difference in the numbers therefore comes from the
task domain and the canary design, never from the measurement.

### Definitions (from `README.txt`)

- **Degradation** — losing track of state, violating constraints, abandoning
  parts of tasks, fabricating facts not explicitly given.
- **Hallucination** — a degradation event that actually impacts the task at hand.
- **Sentinel / Canary** — a degradation event on a task-irrelevant chore, nuance
  or probe.

### Quick start

```bash
printf 'OPENAI_API_KEY=...\nFIREWORKS_API_KEY=...\n' > .env   # gitignored

python -m experiments.run_all      # experiment 1
python -m experiments.figures

python -m experiments2.selftest2   # validate checkers offline, no API calls
python -m experiments2.run_all2    # experiment 2
python -m experiments2.figures2

python -m experiments3.selftest3   # validate all exp-3 checkers offline (3 task sets)
python -m experiments3.run_all3    # experiment 3 (see README_EXPERIMENT3.md for flags)
python -m experiments3.figures3
```
