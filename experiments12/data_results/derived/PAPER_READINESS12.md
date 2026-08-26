# Experiment 12 paper readiness

This is the concise claim-to-evidence map for the `README.md` **Paper Contents** backbone. The machine-readable companion is `paper-readiness-map12.json`. No provider calls were made to produce either file.

## What is already paper-ready

- Active recomputation shows an observer-effect **trend, not a rule**: 6/7 model-by-benchmark success effects are negative, three paired 95% intervals exclude zero on the negative side, and one slice is a small positive exception. Mean total tokens rise in all seven slices.

- Signal quality is conditional. Active recomputation leads AUPRC in 3/7 slices; a passive method or context baseline leads 4/7. The paper should explain when each class works, not claim a universal hierarchy.

- The seven primary precision-recall figures already cover every powered model-by-benchmark signal slice: four Evolving Intent reasoning slices and three BFCL action slices.

- Provider-backed passive observation costs tokens, latency, and dollars but does not enter target history. Deterministic trace rules, clock, and context-use add no provider calls.

- BFCL supports genuine action-trace and turn-level early-warning claims. Evolving Intent supports final-success prediction, observer effect, cost, and deployment—not independently graded within-horizon failure.

## What the deployment runs can add

- The primary online run can support one sharply bounded claim: on 40 held-out Evolving Intent tasks for Luna, which of seven methods helps or hurts under no action, lossy compaction, public-state regrounding, and bounded feedback.

- The frozen two-pass run is a controlled sensitivity for four methods and three operators. Its active anchor acts at checkpoint 1 on all 40 tasks, so describe it as an aggressive checkpoint-1 yoked control—not a 20% matched-rate policy.

- The online study uses unequal natural scalar firing rates. Report action rate beside success and phrase method differences as deployed-policy differences, not pure signal-quality effects.

## Exposition that must change

- Replace Qwen-27B and GPT-4-mini with the actual powered slate: DeepSeek V4 Flash, GPT-OSS-120B, GPT-5.6 Luna, and GPT-5.6 Terra for Evolving Intent; Luna, Terra, and GPT-OSS for BFCL. Qwen3.7 Plus is exploratory only.

- Do not claim active signals are always worse, passive signals always beat baselines, or observer harm is consistent without exceptions.

- Treat copy, recall, counter, and recomputation as exploratory mechanism arms. Their effects are heterogeneous; no monotonic complexity conclusion is supported.

- The deployed `good_bad_watch_feedback` arm is a deterministic quote-only WATCH reminder. It is not LLM-generated and does not assess good versus bad decisions.

- `operator=none` is a monitored no-action control. Neither deployment manifest contains a truly unmonitored arm or an oracle bound.

- Deployment generalization is limited to one model and one reasoning benchmark. BFCL was not deployed.

## Online semantic-retry decision

The copy-on-write normalization is acceptable for a workshop paper if it is fully disclosed. Three passive trace-judge cells needed semantic recovery after transport-success responses failed strict parsing. Retrying each identical frozen prefix does not enter or advance target history, and every physical attempt's resources can be counted.

The production logs and ledger must remain untouched. In an analysis-only staging copy:

1. For all three documented cells, reclassify every capped/unparseable transport-success response as a logical semantic failure while preserving its physical receipt.
2. Number the final parseable response as the next logical attempt and map its staged request key to the canonical logical sequence.
3. Reference every physical event ID in order. The call record carries final-attempt usage and cost, sums every elapsed time, and the unmodified analyzer sums all attempts' tokens, latency, and dollars.
4. Bind every field-level change, source/staging inventory, ledger snapshot, script hash, analyzer command, and output hash in the normalization receipt specified in the JSON map.
5. Run a cumulative paired sensitivity excluding both affected source tasks from all 28 method-operator cells (56 rows; n=38). Report every material change.

For the strongest audit trail, record the unmodified analyzer's expected failure on raw production artifacts before running the same unmodified analyzer successfully on staging. Hash both stderr/stdout records.

Suggested disclosure: “Three passive judge cells required same-prefix semantic recovery, spanning two of 40 source tasks. Raw logs and the ledger remain unchanged; an analysis-only, hash-bound copy classifies every unparseable response as a logical failed attempt, counts every physical attempt's resources, and is checked with a cumulative paired leave-two-source-task-out sensitivity (n=38).”

The two-pass source pass also has one already documented recovery: a missing trace-judge observation at checkpoint 6. Its validation is primary-ready with one unreferenced-call warning, and the recovery receipt is under `results/recovery/9d8591ea71f67026d743d434/`. The final yoked schedule does not use trace judge—it uses active recomputation, frozen recomputation, clock, and context—so this warning does not select the four-method schedule, but it should remain in the reproducibility notes.

## Exact provider-free post-run commands

Run these only after all worker processes finish and the recovery/normalization receipt is complete.

Verify the frozen deployment manifests:

```bash
sha256sum experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/manifest.json experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/pairs.jsonl
```

Expected digests, in order: `7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7` and `16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6`.

```bash
sha256sum experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/manifest.json experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/pairs.jsonl
```

Expected digests, in order: `8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250` and `e45838cb64c522100fb2f0c3f212a00736ab5e1dfb9c501d22f8710c4b6a006e`.

Build and analyze the copy-on-write online staging root with the finalized provider-free builder (SHA256 `792a53a27127482ae890aceae361fc4858d706f58ac928b0244bfd5861b685ce`):

```bash
python3 -m experiments12.generated.build_adaptive_analysis_staging12 analyze
```

The finalized receipt binds the stock n=40 analysis (`c296291f61b1e0134cac1f68f0d94b2f46286f0710354ed0284c2a454db98b9e`), cumulative n=38 JSON (`0e2bae5f026be2d44e3c8e9e90986057503f9a8d0ed65eb65e25180d0213a756`), and Markdown (`7ec8f718f5a49825a2185b6571638aa91ff0cd1d4f3d58810ca9411b46ddec46`). The analysis receipt SHA256 is `dfc03904181b6dc5f48d2ed691cb0889a3d7a03518b583fac59d747e32e0cf65`.

Audit and analyze the two-pass sensitivity:

```bash
python3 -m experiments12.two_pass_analysis12 validate --run-id e12-deploy-twopass-yoked-evolving-luna-40-v1 --manifest-sha256 8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250 --output experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/validation-two-pass.json --artifacts experiments12/artifacts
```

```bash
python3 -m experiments12.two_pass_analysis12 extract --run-id e12-deploy-twopass-yoked-evolving-luna-40-v1 --manifest-sha256 8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250 --output experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-analysis.json --tables experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-tables --figures experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-figures --artifacts experiments12/artifacts --bootstrap-iterations 2000 --bootstrap-seed 12012
```

Check the online analysis dimensions:

```bash
jq -e '.artifact_type == "online_adaptive_deployment_analysis" and .source_manifest_sha256 == "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7" and (.rows | length) == 1120 and ([.rows[].unit_id] | unique | length) == 40 and ([.rows[].method] | unique | length) == 7 and ([.rows[].operator] | unique | length) == 4 and (.metric_summaries | length) == 224 and (.operator_effects | length) == 168 and ([.metric_summaries[].n_tasks] | unique) == [40]' experiments12/generated/adaptive-analysis-staging-v1/analysis/adaptive-analysis.json
```

Check the finalized cumulative paired n=38 sensitivity:

```bash
jq -e '.artifact_type == "experiment12_online_adaptive_leave_two_source_units_sensitivity" and .source_analysis_sha256 == "c296291f61b1e0134cac1f68f0d94b2f46286f0710354ed0284c2a454db98b9e" and .excluded_rows == 56 and .remaining_rows == 1064 and .remaining_source_tasks_per_treatment == 38 and .treatments == 28 and .balanced_paired_design_after_exclusion == true and (.rows | length) == 1064 and (.metric_summaries | length) == 224 and (.operator_effects | length) == 168 and ([.metric_summaries[].n_tasks] | unique) == [38]' experiments12/generated/adaptive-analysis-staging-v1/analysis/adaptive-analysis-leave-two-units.json
```

The final release inventory independently fails closed unless the n=40 source summaries reproduce exactly; recovery cells `d52046b6eb74a76ecdc3debc`, `89df41e0daa1262a43fa5e55`, and `786d95760ccdb86713c26936` map to their declared trace-judge treatments and exactly two frozen source units; both units are removed from every one of the 28 treatments (56 rows total); and all n=38 summaries reproduce with the frozen 2,000-bootstrap/seed-12012 function. A future recovery on another source unit forces an expanded cumulative exclusion. The audit compares all 21 success operator effects and all 105 key-resource operator effects, plus every absolute summary.

Check the two-pass validation and analysis dimensions:

```bash
jq -e '.artifact_type == "two_pass_deployment_validation" and .primary_ready == true and .expected_cells == 480 and .valid_outputs == 480 and .valid_jobs == 480 and .valid_event_logs == 480 and .canonical_regraded_cells == 480 and .cached_official_cells == 0' experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/validation-two-pass.json
```

```bash
jq -e '.artifact_type == "two_pass_deployment_analysis" and .source_manifest_sha256 == "8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250" and .validation.primary_ready == true and (.rows | length) == 480 and ([.rows[].unit_id] | unique | length) == 40 and ([.rows[].method] | unique | length) == 4 and ([.rows[].operator] | unique | length) == 3 and (.metric_summaries | length) == 168 and (.operator_effects | length) == 112 and (.method_effects | length) == 252 and ([.metric_summaries[].n_tasks] | unique) == [40]' experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-analysis.json
```

Run the complete provider-free test gate:

```bash
python3 -m experiments12.cli12 selftest
```

After the staged online analysis and cumulative sensitivity exist, build the fail-closed paper-material release inventory:

```bash
python3 experiments12/generated/build_paper_material_inventory12.py
```

Final acceptance is not “all files exist.” It is: both unmodified analyzers exit 0; every assertion above returns true; the normalization receipt proves raw-production immutability and complete physical-attempt accounting; the cumulative n=38 sensitivity reports every frozen comparison; and all paper claims use the qualifications in this map.
