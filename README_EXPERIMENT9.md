# Experiment 9 — Active vs passive observation on sharded GSM8K, four models

Experiment 8 established the active/passive-behavioural/passive-observational
taxonomy on the project's synthetic pool with one model. Experiment 9 re-asks
its three questions — signal quality, downstream gain, cost of observation —
under the two external-validity upgrades a publishable version needs:

1. **A respected benchmark.** The `math` split of Microsoft's *LLMs Get Lost
   in Multi-Turn Conversation* (arXiv:2505.06120): GSM8K problems split into
   4–12 constraint shards revealed one per turn. To restore the long horizons
   the reset dynamics need, **3 sharded problems are concatenated per
   session** (34 sessions, horizons 14–21 turns), exactly as the bAbI set
   strings multiple stories through one conversation.
2. **Four models through one harness:** `gpt-oss-120b`, `deepseek-v4-flash`,
   `qwen3p7-plus`, `gpt-4o-mini` (thinking disabled on the reasoning models).

## Protocol (deviations from the source paper stated up front)

Shards are revealed **verbatim** (no LLM user-simulator paraphrase), one per
turn. The assistant must reply `WAIT` until the turn marked
`FINAL SHARD - answer now.`, then `ANSWER: <number>`. Every turn is decidable
against the GSM8K `####` key:

- non-final turn: an `ANSWER` line is a **premature answer** (the paper's
  headline failure mode, here decidable per turn); neither WAIT nor ANSWER is
  malformed. (`WAITWAIT` stutters count as WAIT — decoding artifacts are not
  state loss.)
- final turn: missing or numerically wrong `ANSWER` is a hallucination.

The gate run reproduced the paper's texture: premature answers emerge
mid-session in the no-reset arm and clock restarts largely eliminate them.

## Arms (exp-8 design, R1 reground operator)

| arm | category | trigger | carries |
|---|---|---|---|
| `A_no_reset` / `F_oracle` | bounds | never / once before A's first failure | — |
| `C_clock` | baseline | every 6 turns | — |
| `Z_trace` | passive-observational | structural monitor (premature answer, missing WAIT/final-answer — no answer key) | — |
| `QUIZ` | passive-behavioural | frozen-state quiz fails (fork, discarded; first-shard verbatim / shard count / problems completed, graded vs the store) | — |
| `ACT_carry_clock` | active | clock, probe carried (observer-cost control) | `lag_span` |
| `ACT_probe` | active | carried probe fails | `lag_span` |

`lag_span` is the labeled probe: the sharded failure mode is losing
constraints revealed many turns ago — exp 3's RETRIEVAL_DISTANCE genre. The
reground resume replays the standing rule and the verbatim ticket log (all
user-issued content). A **shadow pass** asks the quiz on reconstructed
prefixes of each model's own `A_no_reset` trajectories, giving uncensored
quiz precision/recall per model with no extra agent runs.

The zero-carry monitor overlaps the checker on *protocol* errors by
construction but is blind to *wrong final answers* — the honest limit of
trace-only observation in this domain, reported as such.

## Run

```bash
python -m experiments9.build9                     # data9/tasks9_shardmath.json
python -m experiments9.selftest9                  # offline, mock LLM
python -m experiments9.run_all9 --model M --gate  # A + C_clock, protocol check
python -m experiments9.run_all9 --model M         # all seven arms
python -m experiments9.shadow9  --model M         # quiz shadow pass
python -m experiments9.metrics9                   # results9/SUMMARY.md + PREDICTION.md
```

## Limitations stated up front

Verbatim shards understate the paraphrase noise of the original benchmark;
one task family (math) from the suite; sessions concatenate independent
problems, so cross-problem interference is the only long-range coupling; the
monitor's protocol-error overlap with the checker inflates its apparent
precision on premature-answer failures (wrong-final-answer recall is the
discriminating statistic).
