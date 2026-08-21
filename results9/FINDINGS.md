# Experiment 9 — Findings: the observer effect changes sign with the model

Four models × seven arms × 34 sharded-GSM8K sessions (lost_in_conversation
`math`, 3 problems/session, R1 reground). Tables:
[`SUMMARY.md`](SUMMARY.md) · [`PREDICTION.md`](PREDICTION.md).

## 1. The headline: active observation is not uniformly costly — its sign is a property of the model × task regime

`ACT_carry_clock − C_clock` (identical schedule and operator; the only
difference is carrying the lag_span probe):

| model | observer Δ | significant |
|---|---|---|
| gpt-oss-20b, synthetic pool (exp 8) | **−0.036** | yes |
| gpt-oss-120b, sharded math | **+0.021** | yes |
| qwen3.7-plus, sharded math | **−0.016** | yes |
| deepseek-v4-flash / gpt-4o-mini, sharded math | +0.009 / −0.005 | no |

On gpt-oss-120b the carried probe *helps* — and `ACT_probe` is the best arm
outright (0.977, beating the clock +0.035 sig and never-resetting +0.093
sig). Mechanism: lag_span forces the model to re-attend to messages 1/3/6
turns back **every turn**, which is precisely the retrieval-over-distance
capacity that sharded math breaks (losing early constraints → premature
answers). When the chore's mechanism matches the task's failure mode, the
probe stops being interference and becomes an implicit mitigation — an
attention refresh. When it doesn't match, or the model is small (exp 8's
20b) or chat-tuned differently (qwen −0.016), it is a tax. **The sign of the
observer effect is an empirical property of the deployment, not a constant
of active observation.**

Confound noted honestly: on 120b the probe fails nearly every turn (fire
rate 1.0), so `ACT_probe` also resets on a near-dense schedule; the clean
carry-only read is `ACT_carry_clock` (+0.021 at exactly the clock's 2.41
resets/task), which is what the headline claim rests on.

## 2. Only one model is actually "lost" under verbatim shards

The degradation regime itself is model-dependent: gpt-oss-120b shows the
paper's full signature (no-reset 0.883, premature answers mid-session, clock
restarts +0.058 sig, oracle +0.038 sig); qwen3.7-plus and deepseek-v4-flash
barely degrade (0.93 / 0.91, nothing significant — verbatim shards without
the paper's paraphrase simulator are evidently easier); gpt-4o-mini is
uniformly mediocre but flat (0.845, only `Z_trace − A` +0.009 sig). Any
claim about monitoring or resets is conditional on being in a degradation
regime at all — a point the single-model exp 8 could not surface.

## 3. The quiz replicates across models and benchmark: precise, blind, and expensive

Frozen-state quiz precision is again the best-in-class of behavioural
signals (0.875–1.0 at fail≥1 on the two models with degradation) with tiny
recall (0.03–0.21) — so `QUIZ` deploys like `A_no_reset` plus ~3K fork
tokens on every model. Exp 8's mechanism transfers: agents ace direct-recall
questions while the failure mode (premature commitment) never shows up as a
wrong quiz answer. Passive-behavioural probing keeps failing for the same
reason on both benchmarks — that is now a replicated negative result.

## 4. The trace monitor's perfection here is partly definitional

`Z_trace` reads 1.0/1.0 precision-recall on gpt-oss-120b because in this
protocol most hallucinations ARE protocol violations (premature answers),
which the structural monitor detects by construction; it is blind to wrong
final answers. Its deployed gain (+0.036 over no-reset, sig, on 120b) is
real but the signal-quality number should not be compared to exp 8's 0.783
at face value.

## 5. Updated decision rule

Exp 8 said: when restarts are cheap, the best observer is a clock. Exp 9
adds the precondition and one genuinely new row: **first measure whether you
are in a degradation regime at all** (two of four models were not — every
observation method is then pure cost), and **measure the observer effect's
sign before banning active probes**: a chore matched to the failure
mechanism can be worth carrying not as a signal but as a *treatment* —
on gpt-oss-120b it was worth +0.02 to +0.09 accuracy.

## Caveats

34 sessions/model (power is limited for small deltas); verbatim shards
understate the original benchmark's paraphrase difficulty; one task family;
the lag_span probe's near-saturated failure rate makes `ACT_probe`'s trigger
approximate a dense schedule (use `ACT_carry_clock` for the carry effect);
quiz deployment ran only at fail≥2 while fail≥1 has the usable recall —
untested as a live arm.
