# Experiment 8 — Findings: what a way of watching costs, and what it buys

Model `gpt-oss-20b`, the exp-5/6 pool (90 tasks, 30/domain, difficulty-
stratified), reset operator fixed to R1 reground everywhere. New arms QUIZ /
ACT_probe / ACT_carry_clock (runs8); every baseline, bound and
passive-observational arm read verbatim from runs6. Full tables:
[`SUMMARY.md`](SUMMARY.md) · [`PREDICTION.md`](PREDICTION.md) · figures in
[`figures/`](figures).

## 1. The taxonomy ordering is monotone — and honest

Signal quality and downstream outcome give the SAME ranking across the three
observation categories:

| method (category) | precision @K=5 | deployed accuracy |
|---|---|---|
| zero-carry trace monitor (passive-observational) | **0.783** | **0.841** |
| frozen-state quiz (passive-behavioural) | 0.600 | 0.813 |
| carried probe (active) | ≤0.098 | 0.792 |

No Intervention-Paradox inversion: better signal → better outcome,
monotonically. The paradox in this regime lives elsewhere — **no behavioural
signal of any category beats cheap frequency** (§3).

## 2. The observer effect, priced under a loss-free operator

`ACT_carry_clock − C_clock` — identical trigger, identical operator, the only
difference is that the agent carries the probe — is **−0.036 accuracy
[−0.057, −0.016], significant pooled and in all three domains**, plus ~5.4K
extra prompt tokens/task. This replicates exp 4's −0.043 (compaction
operator) under re-grounding: the ~4-point toll of active observation is
operator-independent. Active observation also pays twice: the carried
probe's *signal* is the worst of any method (its own trajectory degradation
poisons the probe), so `ACT_probe` lands below never-resetting (−0.031,
sig).

## 3. The quiz removes the contamination, but the signal doesn't pay its way

The frozen-state quiz achieves exactly what it was designed for — zero
contamination (structurally), precision above every tuned baseline (0.600 vs
0.596 turn-count / 0.462 context / 0.458 random), and better than the
carried probe on every axis, at −0.021 vs the probe in deployment (ns,
direction consistent). But:

- recall is poor (0.087 at fail≥2; 0.275 at fail≥1) → only 0.60 resets/task
  → `QUIZ` ≈ `A_no_reset` (−0.010 ns) while paying **9.3K fork tokens/task**;
- it loses significantly to the clock (−0.035), the trace monitor (−0.028),
  and the judge (−0.037) in deployment.

Mechanism: per-question error rates show the agent almost never misses the
**direct recall** question (4–8% wrong — the very capacity the task needs),
while the **counting/enumeration** questions fail often (17–45%) but couple
weakly to task failure. Exp 5's routing lesson generalises: a probe predicts
failure only when it stresses the failing mechanism — and a quiz you can
answer by re-reading context stresses retrieval, not the accumulation/update
machinery that actually breaks.

## 4. The regime law now covers observation, not just timing

Exp 6's law was "the cheaper the restart, the more frequency wins." Exp 8
extends it: **when restarts are cheap and loss-free, the best observer is a
clock.** `C_clock` Pareto-dominates (0.847 at 15.8K total tokens);
`C_judge`'s nominal top accuracy (0.850) costs 45.7K tokens and 4.7
resets/task — an expensive dense schedule, not a better signal. Every
behavioural observation pays one of: contamination (active), fork tokens
(quiz), or missed resets (low recall).

## 5. Decision table (extends results6/FINDINGS §6)

| if you can... | then observe by... | because |
|---|---|---|
| restart cheaply from external ground truth | a clock (or context budget) | frequency beats every signal; monitoring spend is waste |
| restart only lossily (compaction) | zero-carry trace monitors | precision is what matters when resets are risky (exp 5) |
| not modify the agent's context at all | frozen-state quizzes over carried probes | same-or-better signal, zero observer effect |
| afford only one observation mechanism | trace monitors | best precision, zero tokens, zero contamination |
| never inject chores into a production agent | (you lose nothing) | active observation is dominated on every axis measured here |

## Caveats

One model, synthetic pool; quiz questions graded strictly (exact counts);
the quiz deployment arm ran only at fail≥2 (the ablation is offline,
signal-side only); the carried probe's clean read (`ACT_carry_clock`
segments) is clock-truncated to ~5 turns, so its precision is measured on
short segments; the fork-token cost of the quiz scales with context length
and would shrink with a cheaper monitor model or sparser checkpoints —
untested.
