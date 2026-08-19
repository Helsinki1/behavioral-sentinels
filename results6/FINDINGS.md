# Experiment 6 — Findings

**Headline: the answer to "when should you start a new Claude Code session?"
depends on one variable — how much a restart loses — and both halves of the
answer are now measured.** When a restart re-grounds from true external state
(the repo analog), restarting **often, on a schedule** is strictly better
than never restarting: +0.025 accuracy (CI [+0.002, +0.047]) at **43% fewer
prompt tokens**, and clever timing adds nothing — the single perfectly-placed
oracle reset is significantly *worse* than the every-6-turns clock (−0.025,
CI [−0.050, −0.002]). When a restart must re-derive state from the transcript
instead (the replay bracket), scheduled restarts collapse (−0.124 vs
re-grounding, CI [−0.166, −0.083]) and the high-precision zero-carry sentinel
becomes the only safe trigger (+0.075 over the clock under replay, CI
[+0.041, +0.111]). Across experiments 4→5→6 this resolves into one law:
**the lossier the reset operator, the more reset timing matters; the cheaper
the operator, the more frequency wins.**

Model `gpt-oss-20b`, same 90-task pool and protocol as exp 5 (full horizon,
6-reset cap, paired arms); `A_no_reset` imported verbatim from runs5, so the
exp-5↔exp-6 operator contrasts are exactly paired. Store fidelity is proven
offline (reducer == generator truth on all 8,535 turns of the full pool) and
every reset's resume message is recorded in the trajectory as an audit trail.
Full tables in [`SUMMARY.md`](SUMMARY.md).

## 1. The regime flipped, significantly: restarts now pay

Exp 5's darkest result was `C_clock − A_no_reset` = **−0.026 (sig)** —
scheduled resets destroyed accuracy. With the operator changed to
re-grounding and everything else identical:

| contrast | exp 5 (compaction) | exp 6 (re-grounding) |
|---|---|---|
| `C_clock − A_no_reset` | −0.026 **(sig neg)** | **+0.025 (sig pos)**, CI [+0.002, +0.047] |

The operator effect at a fixed trigger, paired on task across experiments
(fig3): clock **+0.051 (sig)**, judge **+0.051 (sig)**, ctx **+0.028 (sig)**,
zero-carry +0.024 (CI [−0.001, +0.050]), oracle +0.005 (ns — one reset,
little at stake). On coding, where compaction was most destructive, the
effect reaches **+0.104 (clock)** and **+0.133 (judge)**. Exp 5's diagnosis
is confirmed causally: the accuracy lost in exp 5 was compaction loss, and
removing it converts resets from a tax into a profit.

## 2. When restarts are cheap, frequency beats timing — the sentinel question dissolves

The exp-4/5 framing — "place a scarce, risky reset exactly before failure" —
presupposed expensive resets. With cheap resets it inverts:

- `F_oracle` (one reset, perfectly placed before the first failure) = 0.822 ≈
  never resetting (−0.001), and **significantly below the clock** (−0.025).
  Placement of a single restart is worth ~nothing; the value is in *repeated*
  re-grounding that keeps state fresh throughout.
- `C_judge` 0.850 and `C_clock` 0.847 lead; `C_ctx` 0.843, `G_dense` 0.839,
  `Z_reground` 0.841 cluster just behind. success@0.9 jumps from 0.478
  (no-reset) to 0.589 (clock) / 0.611 (judge).
- The zero-carry sentinel keeps a different, real value: **efficiency**. At
  0.86 resets/task (26% of the clock's 3.28) it captures ~73% of the clock's
  accuracy lift, and beats budget-matched random at the same reset count
  (+0.017, CI [−0.006, +0.041]). If each restart has a fixed overhead
  (re-onboarding time, cache loss), the sentinel buys most of the gain at a
  quarter of the restarts.

## 3. Restarting is now cheaper AND better — the Pareto plane collapsed

Re-grounded restarts pay twice (fig4): `C_clock` uses **12.6k prompt
tokens/task vs 22.1k for never resetting (−43%)** while gaining +0.025
accuracy — short fresh contexts beat one long stale context on both axes
simultaneously. `G_dense` is cheapest (12.2k). The judge tops raw accuracy
(0.850) but at 29.5k tokens (an extra LLM call per turn) it is
Pareto-dominated by the clock, which matches it (−0.003, ns) at 43% of the
cost. **For a deployed agent with a reliable external store, the practical
advice is embarrassingly simple: restart on a schedule; you don't need a
signal at all.**

## 4. The replay bracket: the gain needs a materialized store, not a clean transcript

R2 ("replay") restarts get the verbatim log of every prior user message —
zero harness intelligence — and they are a disaster: `C_clock_replay` 0.724
(**−0.124** vs re-grounding; coding 0.581), `F_oracle_replay` −0.048,
`Z_replay` −0.042 (all sig). The audit trail shows why: right after a replay
reset the model drops functions it should re-derive from the edit log
(`missing_def` immediately post-reset) — re-applying a 15-turn edit history
in one pass is itself a long-horizon state-tracking task, the very thing
that was failing. Two conclusions:

1. **Mechanism:** the harm in long sessions is not (mainly) the agent's own
   accumulated outputs — dropping them (replay) makes things *worse* than
   keeping them. The gain comes from never having to *re-derive* state:
   materialized external state is the whole ballgame.
2. **The exp-5 pattern reappears exactly where predicted:** under a lossy
   operator, the zero-carry sentinel significantly beats the clock
   (`Z_replay − C_clock_replay` = **+0.075, sig**) by firing rarely and only
   on real degradation — precision-timing matters again because every reset
   is now a risk. This is the same inversion exp 5 measured for compaction,
   reproduced under a second lossy operator.

## 5. Per-domain: restarts fix context rot, not skill

Coding and registers carry the whole effect (clock − no-reset: +0.043 sig,
+0.030 sig); babi is flat at ~0.63 in *every* arm (spread 0.617–0.655).
babi's failures are within-story retrieval/attention errors on recent
context — skill, not rot — and no restart schedule can buy skill. This is
the boundary condition on the headline advice: restarting helps exactly
where failure comes from accumulated stale context, and not at all where
the task is hard per-turn.

## 6. The decision rule (the project's deliverable)

When to start a new Claude Code session, as measured:

| your situation | what to do | evidence |
|---|---|---|
| state lives externally (repo, files, notes) and a fresh session re-reads it | **restart liberally, on a schedule** (here: every ~6 turns); no signal needed | clock +0.025 acc, −43% tokens, sig; oracle timing adds nothing |
| each restart has real fixed overhead (re-onboarding, cache, human time) | restart on the **zero-carry self-contradiction signal** | 73% of the clock's lift at 26% of the restarts; beats random at matched budget |
| state lives only in the conversation and a restart must re-derive it (compaction or transcript replay) | **don't restart on a schedule** — restart only on a high-precision degradation signal, or not at all | scheduled: −0.026 (exp 5) / −0.124 (replay), both sig; sentinel beats clock +0.075 sig under replay |
| the task is hard per-turn rather than long-horizon (babi-like) | restarts won't help; invest in the task, not the session | all arms within noise on babi |

Corollary for agent design: the highest-leverage investment is not a better
restart *signal* but a better external *store* — keep state on disk
(commits, state files, notes) so that restarts are re-grounding rather than
re-derivation. The sentinel literature of exps 1–5 survives as the fallback
for when you can't.

## Limitations

1. **The single-reset oracle is no longer a meaningful upper bound** — under
   a cheap operator, repeated resets beat one perfect placement, so exp 4–6's
   "oracle headroom" framing retires with exp 5. An oracle over *schedules*
   is the right ceiling and was not constructed.
2. **n = 90**: the regime-defining contrasts (clock vs no-reset, all
   replay-vs-reground, replay sentinel-vs-clock, operator effects at
   clock/ctx/judge) are significant; the Z-arm efficiency contrasts pooled
   are not (±0.02 CIs).
3. **The harness-as-file-system is the regime's definition**, licensed by
   the 8,535-turn store↔truth proof and bracketed from below by R2 — but a
   real repo also contains the agent's *mistaken* edits; here all state
   derives from user instructions, so the store is never corrupted by the
   agent. A follow-up where the agent's own (possibly wrong) writes land in
   the store would close that last gap.
4. **Restart amnesia is underrepresented**: these tasks carry little
   in-flight reasoning that a restart would lose; real sessions carry tacit
   plans. The fixed-overhead row of the decision table is where that cost
   would bite, favoring the sentinel further.
5. **One model** (`gpt-oss-20b`), one compaction operator as the exp-5
   comparator, and re-grounding tested only at the 6-reset cap.
