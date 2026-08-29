# NewInML paper graphs

This folder contains the six readable PNG figures selected for the NewInML
2026 workshop draft. Frozen quiz is excluded from every displayed method
comparison, although its frozen result rows remain untouched in the underlying
analysis artifacts.

- `02-active-probe-observer-effect.png`: powered paired observer effects for
  active recomputation, grouped by benchmark and sorted within each group.
- `03-signal-quality-auprc.png`: AUPRC, precision, and recall for all seven
  powered model-by-task slices across the remaining active, passive, and
  baseline monitoring methods.
- `05-overall-recovery-success.png`: equal-weighted recovery success across six
  deployed methods on Evolving Intent.
- `06-active-recovery-success.png`: active recomputation under deterministic
  regrounding versus lossy compaction.
- `07-passive-recovery-success.png`: the same recovery comparison for Frozen
  recompute, Trace rules, and Trace judge.
- `08-controlled-oracle-timing.png`: the GPT-OSS-20B controlled coding
  factorial, comparing no intervention with oracle-timed compaction and
  re-grounding with and without a carried probe.

Recovery deployment covers GPT-5.6 Luna on 40 Evolving-Intent tasks only; BFCL
recovery bars cannot be reported because Experiment 12 did not deploy recovery
interventions on BFCL. Methods are sorted within their taxonomic groups, and
truncated success axes are visibly marked.

All six figures are built by
`../scripts/posthoc/build_workshop_figures12.py` from frozen derived summaries.
The script is provider-free and cannot rerun agent traces. Reproducible SVG
sources and a hash receipt are under
`../data_results/derived/workshop-figures12/`.
