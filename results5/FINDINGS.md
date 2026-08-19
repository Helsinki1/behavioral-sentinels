# Experiment 5 — Findings

**Headline: routing fixes the signal, zero-carry fixes the cost, and the
combination reaches the oracle — but the oracle itself has moved.** The
zero-carry routed sentinel (`Z_routed`) posts exactly the same pooled
accuracy as the perfect-timing oracle (0.817 vs 0.817) at a comparable
budget (1.09 vs 0.76 resets/task), and its point estimate beats **all four
traditional triggers** the project set out to beat — turn-count clock
(+0.021), context growth (+0.002), LLM judge (+0.018), and budget-matched
random (+0.011). Meanwhile the deeper result is that in this regime the
*intervention* ran out of headroom: perfectly timed resets no longer beat
never resetting at all (−0.006, ns), and the clock is now **significantly
worse than doing nothing** (−0.026, CI [−0.053, −0.001]).

Model `gpt-oss-20b`, 90 tasks (30 coding / 30 registers / 30 babi,
difficulty-stratified), 11 arms paired on every task, full horizon, validated
structured compaction snapshots, identical 6-reset cap everywhere. Full
tables in [`SUMMARY.md`](SUMMARY.md).

## 1. The router works, and routing is real signal engineering

The router (one `gpt-oss-20b` call reading only the briefing) matched the
intended genre on **81/90 tasks** (coding 27/30 → `staircase`, registers
30/30 → `chain_checksum`, babi 24/30 → `lag_span`); its off-label calls were
defensible readings (e.g. incremental coding as UPDATE_APPLICATION), used
verbatim, never corrected.

Against the two controls the routing claim needs:

| contrast (same trigger rule, same load) | pooled | babi | registers | coding |
|---|---|---|---|---|
| `D_routed − D_blanket` (vs the exp-1..4 one-probe-everywhere design) | +0.017 | **+0.055 (sig)** | +0.028 | −0.031 |
| `D_routed − D_rotated` (same probes, deliberately mis-assigned) | +0.011 | **+0.048 (sig)** | +0.014 | −0.030 |

Both effects are significant exactly where routing changes the probe (babi;
registers directionally), and absent on coding, where 27/30 routed tasks
carry the same probe as the blanket arm anyway — the coding gaps are the
noise floor of two independent runs of a near-identical configuration. So:
**matched probes predict a domain's failures better than mismatched ones,
holding everything else fixed** — the deployment-side confirmation of
experiment 3's screening result.

## 2. Carried probes are much cheaper than exp 4's, but still not free

The routed probe is one output line. Its ledger, versus exp 4's blanket
sentinel:

| component | exp 4 (blanket, coding) | exp 5 (routed, mixed pool) |
|---|---|---|
| carrying cost (`C′ − C`) | −0.043 | −0.018 |
| timing value (`D − C′`) | −0.019 | +0.013 |
| net vs clock (`D − C`) | −0.062 | −0.006 |

Routing turned the timing information from worthless to mildly positive and
halved the carrying cost, moving the carried sentinel from clearly losing to
statistical parity with the clock. But it still cannot beat a probe-free
baseline pooled, because the carrying cost eats the timing value — the same
conclusion exp 4 reached, now at a finer resolution.

## 3. The zero-carry routed sentinel is the design that works

`Z_routed` carries nothing: deterministic monitors over output the agent
already emits (self-contradicted signatures on coding, re-query
inconsistencies on registers, story-ungrounded answers on babi). Results:

- **Accuracy 0.817 — identical to the oracle** (0.817), above every
  traditional trigger and every carried probe, at the lowest behavioural
  budget in the experiment (1.09 resets/task).
- Pooled point estimates beat all four goal baselines; per-domain it beats
  the clock **significantly on coding** (+0.056, CI [+0.000, +0.116]) and
  the judge significantly on coding (+0.080), while never losing
  significantly anywhere except to the judge on registers (−0.032, where
  the judge spends 3.2 resets to Z's 0.37 and an extra LLM call per turn —
  `C_judge` uses 37k prompt tokens per task to Z's 20.5k).
- `success@0.9` 0.478 — tied with no-reset, judge and oracle for best.

The mechanism is exactly what exp 4 prescribed: the signal costs nothing to
carry, so every bit of its (reactive) timing information is profit. It
cannot fire before the first slip — it is a "recent-error / self-consistency"
signal, not a premonition — but placing the reset right after the first slip
and nowhere else turns out to be worth as much as knowing the future.

## 4. The twist: the intervention, not the signal, is now the bottleneck

Experiment 4 (coding only, freeform self-summary compaction) measured
**+0.059** of oracle headroom over the clock. Here, with a structured
validated snapshot and a mixed pool:

- `F_oracle − C_clock` = +0.021 (ns) — the headroom shrank by two thirds;
- `F_oracle − A_no_reset` = **−0.006** — perfect timing no longer beats
  never resetting;
- `C_clock − A_no_reset` = **−0.026 (significant)** — scheduled resets now
  actively cost accuracy; every resetting arm loses to no-reset on coding.

Two non-exclusive readings, both honest: (a) this model+pool sits in a
regime where compaction is lossy enough that *any* reset risks more than it
saves (coding modules with bodies are the worst case: every resetting arm
drops 0.04–0.12 there), and (b) exp 4's headroom was partly a property of
its lossier compaction operator — when late resets canonicalise errors more
destructively, good timing is worth more. Either way the strategic
conclusion flips: **the signal problem is essentially solved (a free signal
matches the oracle); the remaining loss is in the compaction operator
itself.** Resets do still pay in cost — the clock cuts prompt tokens 14%,
`C_ctx` 18% — so the practical frontier is accuracy-per-token, where
`Z_routed` and `C_ctx` are the two Pareto points among non-oracle arms.

## 5. What this means for the project

1. **The goal line is met on point estimates, and honestly caveated.**
   `Z_routed` outscores turn count, context length, LLM judge and random
   resets pooled, significantly beats clock and judge on coding, and does so
   with the fewest interventions and zero carried load. Pooled CIs still
   cross zero at n=90; the claim is "matches the oracle and Pareto-dominates
   the traditionals", not "significantly beats every baseline everywhere".
2. **Sentinel design rules, final form:** (i) non-copyable, (ii) matched to
   the task's dominant failure mechanism (routing — worth +0.01–0.05
   depending on how wrong the default probe was), (iii) zero carrying cost —
   read off work the agent already does. Rule (iii) dominates: it converts
   the entire observer-effect literature of exps 3–4 from a tax into zero.
3. **Next lever: compaction quality, not signal quality.** With signal ≈
   oracle, every remaining point of loss is the operator. Candidates: hybrid
   snapshots (validated schema + verbatim tail of recent turns), domain
   state checkpointing outside the context (files on disk), and
   selective compaction (drop noise, keep the artifact verbatim).
4. **`C_ctx` is the baseline to beat going forward** — context *growth*
   (not absolute length, which collapsed on babi in exp 3) is cheap, robust
   across domains, and within noise of `Z_routed` on accuracy at lower
   token cost. A deployed system should probably run `Z_routed`'s monitors
   *and* a growth cap: the monitors catch state corruption, the cap bounds
   cost.

## Limitations

1. **n = 90 pooled; most pooled contrasts are not individually significant.**
   The significant results are directional anchors (clock < no-reset;
   Z > clock and Z > judge on coding; routed > blanket/rotated on babi;
   oracle > clock on babi). Resolving ±0.01 pooled effects needs n ≈ 400.
2. **One model.** Exp 2 showed probe difficulty does not transfer across
   models; the router, the monitors' fire rates, and the compaction-loss
   regime may all shift on `gpt-4o-mini` (arm scaffolded, RPD-limited).
3. **The zero-carry monitors are reactive**, not predictive: they fire at or
   after the first visible slip. That they still match the oracle says the
   *cascade*, not the first slip, is where recoverable accuracy lives — but
   a task family with catastrophic first errors would reopen the gap.
4. **Router and probes were validated on the same three domains exp 3
   screened.** The genre taxonomy is written to be open-ended, but nothing
   here tests an unseen genre (spatial, planning, tool-use).
5. **The oracle is exp 4's construction**: it resets before the no-reset
   arm's first-failure turn, an upper bound on *single-reset placement*, not
   on all possible intervention schedules.
