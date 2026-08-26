# Online cumulative leave-two-source-units sensitivity

This provider-free sensitivity derives two omitted source units from the complete documented recovery-cell set and removes both units from every method/operator treatment. No outcome field is used for selection.

- Source analysis SHA256: `c296291f61b1e0134cac1f68f0d94b2f46286f0710354ed0284c2a454db98b9e`
- Receipt JSON SHA256: `1e9cca87b2c04763bd888262b1e3751819d6bd9b02e79080d76fbfa56c7d8b6e`
- Script SHA256: `8addbccc347880f2aa80932ced47fd3307dabe80b11d32abf9680f25ed92c0e6`
- Documented recovery cells: `786d95760ccdb86713c26936`, `89df41e0daa1262a43fa5e55`, `d52046b6eb74a76ecdc3debc`
- Derived source units: `extracted-gsm8k-test-814::t7/r0`, `extracted-gsm8k-test-989::t7/r0`
- Removed rows: 56 (two from each of 28 treatments)
- Denominator: 40 -> 38 paired source tasks
- Scientific outcome changes: 23
- Action-policy changes: 12
- Resource-sensitivity flags: 18
- Overall assessment: **scientific_outcome_conclusion_changed**

## Frozen rules

- Bootstrap: 2,000 paired source-task resamples, seed 12012, 95% intervals.
- An operator effect is flagged if its point sign changes or its 95% interval changes among negative/excludes-zero, includes-zero, and positive/excludes-zero.
- Absolute success and selected-action-rate means flag at 1/40 (0.025), or for zero/nonzero changes.
- Threshold-firing and resource means flag at a 5% relative shift, or for zero/nonzero changes.

## Material operator-effect changes

| group | method | operator | metric | n=40 effect / inference | n=38 effect / inference | reason |
|---|---|---|---|---|---|---|
| resource | active_recompute | good_bad_watch_feedback | actual_cost_usd | 4.65e-06 / includes_zero | -2.77368e-05 / includes_zero | point_effect_sign_changed |
| scientific_outcome | active_recompute | good_bad_watch_feedback | success | 0 / includes_zero | 0.0263158 / includes_zero | point_effect_sign_changed |
| resource | active_recompute | good_bad_watch_feedback | task_tokens | 141.7 / positive_excludes_zero | 103.079 / includes_zero | confidence_interval_relation_to_zero_changed |
| resource | context_use | lossy_compaction | task_tokens | -163.6 / includes_zero | -181.816 / negative_excludes_zero | confidence_interval_relation_to_zero_changed |
| resource | turn_clock | good_bad_watch_feedback | task_tokens | -6.2 / includes_zero | 3.65789 / includes_zero | point_effect_sign_changed |
| resource | turn_clock | good_bad_watch_feedback | total_tokens | -6.2 / includes_zero | 3.65789 / includes_zero | point_effect_sign_changed |
| resource | frozen_probe:recompute | good_bad_watch_feedback | latency_ms | 1222.03 / includes_zero | 1377.45 / positive_excludes_zero | confidence_interval_relation_to_zero_changed |
| resource | frozen_quiz | good_bad_watch_feedback | observer_tokens | 106.55 / positive_excludes_zero | 100.211 / includes_zero | confidence_interval_relation_to_zero_changed |
| resource | frozen_quiz | lossy_compaction | task_tokens | 13.775 / includes_zero | -0.605263 / includes_zero | point_effect_sign_changed |
| resource | trace_judge | lossy_compaction | observer_tokens | 10.375 / includes_zero | -22.3158 / includes_zero | point_effect_sign_changed |
| action_policy | trace_judge | lossy_compaction | selected_actions | 0 / includes_zero | 0.0263158 / includes_zero | point_effect_sign_changed |
| resource | trace_judge | lossy_compaction | total_tokens | 32.175 / includes_zero | -1.02632 / includes_zero | point_effect_sign_changed |
| resource | trace_judge | public_state_reground | actual_cost_usd | 0.000898975 / includes_zero | -0.000460316 / includes_zero | point_effect_sign_changed |
| resource | trace_judge | public_state_reground | latency_ms | 920.775 / includes_zero | -550.263 / includes_zero | point_effect_sign_changed |
| resource | trace_judge | public_state_reground | observer_tokens | 71.725 / includes_zero | -58.3684 / includes_zero | point_effect_sign_changed |
| resource | trace_judge | public_state_reground | total_tokens | 63.775 / includes_zero | -82.3158 / includes_zero | point_effect_sign_changed |
| resource | trace_rules | good_bad_watch_feedback | task_tokens | 85.175 / includes_zero | 101.158 / positive_excludes_zero | confidence_interval_relation_to_zero_changed |
| resource | trace_rules | good_bad_watch_feedback | total_tokens | 85.175 / includes_zero | 101.158 / positive_excludes_zero | confidence_interval_relation_to_zero_changed |
| resource | trace_rules | lossy_compaction | actual_cost_usd | 9.8025e-05 / includes_zero | 0.000145789 / positive_excludes_zero | confidence_interval_relation_to_zero_changed |
| resource | trace_rules | public_state_reground | actual_cost_usd | -2.265e-05 / includes_zero | 1.02895e-05 / includes_zero | point_effect_sign_changed |

## Material absolute-summary shifts

| group | method | operator | metric | n=40 mean | n=38 mean | delta |
|---|---|---|---|---:|---:|---:|
| scientific_outcome | active_recompute | good_bad_watch_feedback | success | 0.725 | 0.763158 | 0.0381579 |
| scientific_outcome | context_use | good_bad_watch_feedback | success | 0.725 | 0.763158 | 0.0381579 |
| action_policy | context_use | good_bad_watch_feedback | threshold_firings | 0.325 | 0.342105 | 0.0171053 |
| scientific_outcome | context_use | lossy_compaction | success | 0.525 | 0.552632 | 0.0276316 |
| action_policy | context_use | lossy_compaction | threshold_firings | 0.225 | 0.236842 | 0.0118421 |
| scientific_outcome | context_use | none | success | 0.725 | 0.763158 | 0.0381579 |
| action_policy | context_use | none | threshold_firings | 0.4 | 0.421053 | 0.0210526 |
| scientific_outcome | context_use | public_state_reground | success | 0.7 | 0.736842 | 0.0368421 |
| action_policy | context_use | public_state_reground | threshold_firings | 0.225 | 0.236842 | 0.0118421 |
| scientific_outcome | turn_clock | good_bad_watch_feedback | success | 0.775 | 0.815789 | 0.0407895 |
| scientific_outcome | turn_clock | none | success | 0.725 | 0.763158 | 0.0381579 |
| scientific_outcome | turn_clock | public_state_reground | success | 0.75 | 0.789474 | 0.0394737 |
| scientific_outcome | frozen_probe:recompute | good_bad_watch_feedback | success | 0.75 | 0.789474 | 0.0394737 |
| scientific_outcome | frozen_probe:recompute | lossy_compaction | success | 0.8 | 0.842105 | 0.0421053 |
| scientific_outcome | frozen_probe:recompute | none | success | 0.725 | 0.763158 | 0.0381579 |
| scientific_outcome | frozen_probe:recompute | public_state_reground | success | 0.775 | 0.815789 | 0.0407895 |
| scientific_outcome | frozen_quiz | good_bad_watch_feedback | success | 0.75 | 0.789474 | 0.0394737 |
| scientific_outcome | frozen_quiz | none | success | 0.75 | 0.789474 | 0.0394737 |
| scientific_outcome | frozen_quiz | public_state_reground | success | 0.675 | 0.710526 | 0.0355263 |
| action_policy | trace_judge | good_bad_watch_feedback | selected_actions | 0.125 | 0.0789474 | -0.0460526 |
| scientific_outcome | trace_judge | good_bad_watch_feedback | success | 0.675 | 0.710526 | 0.0355263 |
| action_policy | trace_judge | good_bad_watch_feedback | threshold_firings | 0.325 | 0.210526 | -0.114474 |
| scientific_outcome | trace_judge | lossy_compaction | success | 0.7 | 0.736842 | 0.0368421 |
| action_policy | trace_judge | lossy_compaction | threshold_firings | 0.475 | 0.421053 | -0.0539474 |
| action_policy | trace_judge | none | selected_actions | 0.25 | 0.210526 | -0.0394737 |
| scientific_outcome | trace_judge | none | success | 0.725 | 0.763158 | 0.0381579 |
| action_policy | trace_judge | none | threshold_firings | 0.35 | 0.289474 | -0.0605263 |
| action_policy | trace_judge | public_state_reground | selected_actions | 0.225 | 0.184211 | -0.0407895 |
| scientific_outcome | trace_judge | public_state_reground | success | 0.7 | 0.736842 | 0.0368421 |
| action_policy | trace_judge | public_state_reground | threshold_firings | 0.275 | 0.210526 | -0.0644737 |
| scientific_outcome | trace_rules | good_bad_watch_feedback | success | 0.825 | 0.868421 | 0.0434211 |
| scientific_outcome | trace_rules | lossy_compaction | success | 0.775 | 0.815789 | 0.0407895 |
| scientific_outcome | trace_rules | none | success | 0.75 | 0.789474 | 0.0394737 |

The JSON receipt contains every n=40 versus n=38 outcome, action, and resource comparison and the complete recomputed n=38 summaries/effects.
