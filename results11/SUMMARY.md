# Experiment 11 — matched vs mismatched probes (results)

Pre-registered in README_EXPERIMENT11.md before these runs. gpt-oss-20b,
R1 re-grounding, clock reset every 6 turns, full horizon. Identical policy,
operator, schedule and pool in every cell; only the carried chore varies.
`registers` excluded by the degradation screen.

| domain | no probe | matched probe | Δ matched | mismatched probe | Δ mismatched | matched − mismatched |
|---|---|---|---|---|---|---|
| coding (n=30) | 0.938 | staircase 0.887 | -0.0509 [-0.094,-0.010] **sig** | lag_span 0.881 | -0.0563 [-0.098,-0.017] **sig** | +0.0054 [-0.028,+0.037] |
| babi (n=30) | 0.630 | lag_span 0.591 | -0.0391 [-0.081,-0.001] **sig** | staircase 0.569 | -0.0609 [-0.103,-0.023] **sig** | +0.0218 [-0.012,+0.055] |

## P2 — the pre-registered primary statistic

Domain x probe interaction: [Δ(staircase,coding) − Δ(lag_span,coding)] −
[Δ(staircase,babi) − Δ(lag_span,babi)], i.e. matched-minus-mismatched in
coding minus the same in babi. H1 predicts > 0.

**P2 = +0.0272, 95% CI [-0.021, +0.073] — not significant**

