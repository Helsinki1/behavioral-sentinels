# Experiment 12 deployment-analysis readiness

Provider-free snapshot: `2026-08-26T17:58:39Z`. Code tree: `851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e`.

## Run gates

| analysis | required | outputs | jobs | events | ready |
|---|---:|---:|---:|---:|---|
| online adaptive | 1120 | 547 | 548 | 556 | no |
| yoked two-pass | 480 | 288 | 288 | 296 | no |

Run the commands below only when each row has exact coverage.

## Exact commands

### online extract

```bash
python3 -m experiments12.adaptive_analysis12 extract --run-id e12-deploy-online-evolving-luna-40-v1 --manifest-sha256 7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7 --output experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/results/adaptive-analysis.json --figures experiments12/artifacts/e12-deploy-online-evolving-luna-40-v1/results/adaptive-figures --artifacts experiments12/artifacts --bootstrap-iterations 2000 --bootstrap-seed 12012
```

### yoked validate

```bash
python3 -m experiments12.two_pass_analysis12 validate --run-id e12-deploy-twopass-yoked-evolving-luna-40-v1 --manifest-sha256 8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250 --output experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/validation-two-pass.json --artifacts experiments12/artifacts
```

### yoked extract

```bash
python3 -m experiments12.two_pass_analysis12 extract --run-id e12-deploy-twopass-yoked-evolving-luna-40-v1 --manifest-sha256 8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250 --output experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-analysis.json --tables experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-tables --figures experiments12/artifacts/e12-deploy-twopass-yoked-evolving-luna-40-v1/results/two-pass-figures --artifacts experiments12/artifacts --bootstrap-iterations 2000 --bootstrap-seed 12012
```

## Claim and figure map

1. **Overall deployed task success by observation method and class** — `direct_once_runs_complete`
   - Online: metric_summaries[metric=success].{observation_class,method,operator,n_tasks,mean,ci_low,ci_high}; rows[].{unit_id,method,operator,success}
   - Yoked: metric_summaries[metric=success].{observation_class,method,operator,n_tasks,mean,ci_low,ci_high}; method_effects[metric=success].{reference_method,comparison_method,operator,effect,ci_low,ci_high}
   - Gap: Built-in online plot has four classes (baseline, active, passive-behavioral, passive-observational), while README asks for three; merge the two passive classes only in a clearly labeled paper plot.

2. **Which method helps under none, lossy compaction, public-state regrounding, or GOOD/BAD/WATCH feedback** — `fields_available_but_provider_free_interaction_plot_needed`
   - Online: metric_summaries[metric=success] keyed by method/operator; operator_effects[metric=success].{method,operator,control_mean,operator_mean,effect,ci_low,ci_high}; rows[].{unit_id,method,operator,success} for paired custom interactions
   - Yoked: operator_effects[metric=success].{method,operator,effect,ci_low,ci_high}; method_effects[metric=success].{reference_method,comparison_method,operator,effect,ci_low,ci_high}
   - Gap: Neither analyzer emits method-by-operator difference-in-differences. Compute task-paired interactions from rows before claiming one method is specifically better in one regime.

3. **GOOD/BAD/WATCH feedback can improve deployed performance** — `direct_online_only_once_complete`
   - Online: metric_summaries[metric=success,operator=good_bad_watch_feedback]; operator_effects[metric=success,operator=good_bad_watch_feedback]; rows[operator=good_bad_watch_feedback].{selected_actions,success,total_tokens,actual_cost_usd}
   - Yoked: not available
   - Gap: The implemented feedback is deterministic, current-prefix, exact-quote-only GOOD/BAD/WATCH—not an LLM-generated breakdown. Narrow the exposition or run a new operator; there is no yoked sensitivity for feedback.

4. **Token, latency, and dollar cost of deployed observation** — `data_direct_but_online_cost_plot_needed`
   - Online: metric_summaries[metric in task_tokens,observer_tokens,total_tokens,latency_ms,actual_cost_usd]; rows[].{task_tokens,observer_tokens,total_tokens,latency_ms,actual_cost_usd}
   - Yoked: metric_summaries[metric in total_tokens,latency_ms,actual_cost_usd]; operator_effects and method_effects for the same metrics
   - Gap: Two-pass resources exclude frozen pass-one passive-observer cost, so use online rows for end-to-end cost; label yoked resource plots as pass-two-only sensitivity.

5. **Natural firing/action frequency and its relationship to success** — `direct_but_must_be_reported_with_success`
   - Online: metric_summaries[metric in threshold_firings,selected_actions]; rows[].{observations,threshold_firings,selected_actions,applied_interventions,success}
   - Yoked: metric_summaries[metric in action_rate,acted_on_task,scheduled_actions,applied_interventions]
   - Gap: Calibration used fixed-count ranking, but online execution uses score >= scalar cutoff. Existing diagnostic shows deployed scalar-implied rates from 17.5% to 100%, so never describe deployment as a uniform 20% firing policy.

6. **Carrying an active recomputation probe degrades task performance** — `luna_deployment_sensitivity_once_complete`
   - Online: metric_summaries[metric=success] for active_recompute versus zero-carry methods (ecological, natural-policy comparison)
   - Yoked: method_effects[metric=success, reference/comparison includes active_recompute] under the same task and checkpoint-1 schedule
   - Gap: Deployment is one model and one benchmark. Cross-model observer-effect language must come from the completed confirmatory runs, not these deployment analyses.

7. **Signal precision, recall, AUPRC, and early-warning lead time** — `already_available_upstream`
   - Online: not available
   - Yoked: not available
   - Gap: Do not infer signal accuracy from deployment success. Use score-confirmatory.json metrics[].{precision,recall,auprc,firing_rate,lead_time} and disclose the scalar/rank diagnostic separately.

8. **Complexity ladder across say-name, increment, and recomputation active chores** — `not_supported`
   - Online: not available
   - Yoked: not available
   - Gap: Both deployment analyses contain only active_recompute. Retaining a complexity-ladder claim requires new provider runs or clearly labeled legacy evidence.

9. **Deployment generalizes across reasoning and action traces, models, and benchmarks** — `not_supported`
   - Online: not available
   - Yoked: not available
   - Gap: Both deployments are Luna on Evolving-Intent GSM8K only. BFCL supplies completed action-trace detection/observer-effect evidence, but no deployment evidence.

10. **Ground-truth outcome provenance** — `direct_once_runs_complete`
   - Online: rows[].success; validated by exact adaptive replay and job/evaluation binding
   - Yoked: rows[].{success,outcome_source}; validation.{canonical_regraded_cells,cached_official_cells,primary_ready}
   - Gap: Explain benchmark-specific final-answer ground truth separately from the earlier event labels used for signal precision/recall.

## Post-run checklist

- [ ] **P0 · provider_free** — Run both exact validation/extraction commands only after exact cell coverage reaches 1120/1120 and 480/480.
- [ ] **P0 · exposition** — Describe online thresholds as scalar-cutoff policies with observed action rates, not fixed 20% rank policies; pair every success plot with firing/action incidence.
- [ ] **P0 · exposition** — Rename GOOD/BAD/WATCH as deterministic quote-only feedback unless a genuinely LLM-generated feedback operator is rerun.
- [ ] **P0 · exposition** — Replace the README's stale Qwen-27B/GPT-4-mini model list with the actual frozen confirmatory models; deployments themselves are Luna-only.
- [ ] **P1 · provider_free** — Add task-paired method contrasts and method-by-operator interactions from rows; create success-effect, action-rate, and online resource figures.
- [ ] **P1 · provider_free** — Create the requested three-cluster paper plot by merging passive subclasses visually while retaining method labels and the four-class machine-readable source.
- [ ] **P1 · exposition** — Present yoked results as a checkpoint-1, one-action sensitivity isolating carry/operator effects; it does not test comparative trigger timing or natural firing quality.
- [ ] **P1 · exposition** — Call the yoked run a sensitivity analysis on the same 40 source tasks—not an independent replication—and fill the README's third passive method slot with trace_rules.
- [ ] **P2 · new_provider_runs_only_if_claim_retained** — Run missing active-complexity variants, multi-model/multi-benchmark deployments, expanded yoked methods/feedback, or LLM-generated feedback only if those broader claims remain central.

## Built-in outputs

- Online: one success SVG plus exact sidecar; JSON contains rows, metric summaries, and paired operator-minus-none effects.
- Yoked: validation JSON; full analysis JSON; four CSV tables; success, action-rate, total-token, latency, and cost SVGs with sidecars.
- Missing but derivable without providers: cross-method online effects, method-by-operator interactions, three-class paper plot, online action/resource plots.
