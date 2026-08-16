# Behavioral Sentinels

Early-warning signals for hallucination onset in long-horizon LLM agents.

The research design, taxonomy and motivation live in [`README.txt`](README.txt).
This file indexes the two experiments in the repo.

| | **Experiment 1** | **Experiment 2** |
|---|---|---|
| question | can a task-irrelevant chore fire *before* a task hallucination? | does it matter whether the chore's answer **moves**? |
| task | integer register book-keeping | **incremental Python coding** |
| canaries | 6 **static** probes | 6 **dynamic** probes + 1 static control |
| code | [`experiments/`](experiments/) | [`experiments2/`](experiments2/) |
| data | `data/tasks.json` | `data2/tasks2.json` |
| trajectories | `runs/` | `runs2/` |
| results | [`results/SUMMARY.md`](results/SUMMARY.md) | [`results2/SUMMARY.md`](results2/SUMMARY.md) |
| figures | `results/figures/` | `results2/figures/` |
| write-up | — | [`README_EXPERIMENT2.md`](README_EXPERIMENT2.md), [`results2/FINDINGS.md`](results2/FINDINGS.md) |
| headline | a task-irrelevant chore can fire before a task hallucination | a canary is only a sentinel if its answer **can't be copied from the model's own last reply** |

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
```
