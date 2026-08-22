# Experiment 10, Study A — the sentinel break-even surface

Re-analysis of experiments 5 and 6; no new API calls. Each policy is scored by

    U = accuracy - R x resets_per_task - T x prompt_ktokens_per_task

with R the cost of one restart and T the cost of 1k prompt tokens, both in
accuracy-equivalent units (R = 0.01 means a restart costs one accuracy point).
The question is not which policy wins, but at what price each one wins.

## Lossless operator (exp 6 re-grounding: restart restores true external state)

| policy | accuracy | resets/task | prompt ktok |
|---|---|---|---|
| llm_judge | 0.850 | 4.69 | 29.5 |
| clock | 0.847 | 3.28 | 12.6 |
| ctx_growth | 0.843 | 1.52 | 14.6 |
| sentinel_zerocarry | 0.841 | 0.86 | 18.3 |
| dense_clock | 0.839 | 5.74 | 12.2 |
| random | 0.824 | 0.86 | 18.9 |
| never_reset | 0.823 | 0.00 | 22.1 |
| oracle | 0.822 | 0.77 | 19.2 |

**Break-even: the sentinel overtakes the clock once one restart costs more than 0.0027 accuracy-equivalents** (0.27 accuracy points per restart), at zero token cost.

## Lossy operator (exp 5 compaction: restart keeps the agent's own summary)

| policy | accuracy | resets/task | prompt ktok |
|---|---|---|---|
| never_reset | 0.823 | 0.00 | 22.1 |
| sentinel_zerocarry | 0.817 | 1.09 | 20.5 |
| oracle | 0.817 | 0.76 | 21.2 |
| ctx_growth | 0.815 | 1.51 | 18.1 |
| random | 0.806 | 4.28 | 21.1 |
| llm_judge | 0.799 | 4.57 | 37.1 |
| clock | 0.796 | 3.27 | 19.0 |
| sentinel_carried | 0.791 | 4.30 | 26.4 |

**The sentinel dominates the clock outright** — higher accuracy *and* fewer restarts, so it wins at any restart price including zero.

