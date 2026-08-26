# Experiment 12 final ledger and run audit

Overall: **PASS**. Snapshot: `2026-08-26T19:24:17Z`.

Frozen source/config hash: `851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e` (expected hash matched).

## Budget ledger

| Provider | Accounted spend | Operational cap | Hard cap | Active reserved |
|---|---:|---:|---:|---:|
| fireworks | $7.510099 | $24.000000 | $30.000000 | $0.000000 |
| openai | $75.794907 | $400.000000 | $500.000000 | $0.000000 |

All 77,205 reservations are financially settled: 0 active reservations and 0 unaccounted billing ambiguities. There are 15 request-status-unknown rows; each is reconciled at its full reserved upper bound.

The 77,195 append-only attempt rows have unique reservation and event IDs. Ten historical killed-process reservations have no event row; all ten are non-paper runs and are fully charged upper bounds.

## Final paper-run coverage

| Role | Run | Pairs | Outputs | Jobs | Semantic event logs | Calls | Unreferenced calls |
|---|---|---:|---:|---:|---:|---:|---:|
| calibration_bfcl | `e12-calibration-bfcl-core-v1` | 120 | 120+60 | 120+60 | 120+60 | 2090 | 0 |
| calibration_evolving | `e12-calibration-evolving-core-v2` | 160 | 160+80 | 160+80 | 160+80 | 3520 | 0 |
| confirmatory_bfcl | `e12-confirmatory-bfcl-core-v3` | 336 | 336+168 | 336+168 | 336+156 | 5463 | 0 |
| confirmatory_evolving | `e12-confirmatory-evolving-core-v2` | 448 | 448+224 | 448+224 | 448+224 | 9856 | 0 |
| two_pass_source_observation | `e12-deploy-twopass-pass1-evolving-luna-40-v1` | 80 | 80+40 | 80+40 | 80+40 | 1761 | 1 |
| two_pass_yoked_sensitivity | `e12-deploy-twopass-yoked-evolving-luna-40-v1` | 480 | 480 | 480 | 480 | 4080 | 0 |
| online_primary | `e12-deploy-online-evolving-luna-40-v1` | 1120 | 1120 | 1120 | 1120 | 11687 | 6 |

The BFCL confirmatory run has 156 shadow event files for 168 complete shadow outputs because 12 official trajectories have zero observable checkpoints; their complete shadow outputs correctly contain empty record lists.

## Documented recovery boundary

The online run has all 1,120 current outputs, complete jobs, and semantic event logs, plus exactly three hash-bound recovery archives/receipts. Its 6 raw unreferenced call attempts are exactly the malformed max-output judge responses listed by those receipts. The pass-one run has exactly one analogous documented max-output judge attempt. No other final paper run has an unreferenced call attempt.

The online run also contains one referenced HTTP-503 attempt whose billing was ambiguous; the ledger conservatively charges its full $0.002578 reservation before the successful retry.

## Superseded directories

| Run | Pairs | Trajectories | Shadows | Selection note |
|---|---:|---:|---:|---|
| `e12-calibration-evolving-core-v1` | 160 | 160 | 79 | superseded by calibration evolving v2 |
| `e12-confirmatory-bfcl-core-v1` | 336 | 50 | 0 | aborted/superseded by BFCL confirmatory v3 |
| `e12-confirmatory-bfcl-core-v2` | 336 | 3 | 0 | aborted/superseded by BFCL confirmatory v3 |

## Qualification

This audit does not normalize recovery data or certify the analysis staging copy. It certifies the immutable production ledger/log boundary, current paid-run coverage, and the exact raw attempts that the separate staging receipt must account for.

Reproduce with: `python3 experiments12/generated/final_ledger_run_audit12.py`
