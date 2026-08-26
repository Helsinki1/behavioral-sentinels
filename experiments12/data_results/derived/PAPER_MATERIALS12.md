# Experiment 12 paper materials

Fail-closed, provider-free inventory for the 8-page workshop paper. Numerical claims below are derived from the listed machine-readable artifacts; this document deliberately corrects unsupported wording in `README.md`.

- Companion JSON SHA256: `24a63d784d97d011d7fa27ee69e413051c5a535a8c395d5dd1280a18e4863aa2`
- Generator SHA256: `22b474f2a61cc43ad073d3b6c2aa53b50d257988e7c5601547f736be775f8ae3`
- Frozen Experiment 12 code tree: `851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e`

## Evidence at a glance

| study | scope | statistical unit / denominator | role |
|---|---|---|---|
| Powered observation: Evolving Intent | 4 models, 2 trajectory arms, 4 zero-carry shadows | 56 tasks/model/arm; 56 signal tasks/model | confirmatory reasoning traces |
| Powered observation: BFCL | 3 models, 2 trajectory arms, 4 zero-carry shadows | 56 tasks/model/arm; 52 primary signal tasks/model | confirmatory action traces |
| Active mechanism | 9 model-benchmark strata, 4 probe variants | n=20 paired tasks/arm/stratum | exploratory only |
| Online deployment | Luna × Evolving Intent, 7 methods × 4 operators | 40 paired source tasks; 1120 cells | primary ecological deployment |
| Yoked deployment | Luna × Evolving Intent, 4 methods × 3 operators | 40 paired source tasks; 480 cells | checkpoint-1 controlled sensitivity |
| Recovery sensitivity | remove two affected source tasks from all 28 online treatments | 40 → 38 tasks | cumulative robustness audit |

## Sharp conclusions supported by the data

- **supported_as_trend:** Active recomputation usually reduced success, but not universally: 6/7 point estimates were negative, 3 paired 95% intervals excluded zero below, and one stratum was positive.
  - Qualification: Seven model-by-benchmark strata are descriptive replications, not independent samples for a pooled universal claim.
- **supported:** Signal quality is conditional: active recomputation has the highest AUPRC in 3/7 powered slices; another passive method or a context/clock baseline leads 4/7.
  - Qualification: Active signals are measured on carried trajectories and zero-carry signals on clean trajectories; this is an ecological comparison, not a same-trajectory counterfactual.
- **supported:** Provider-backed active and passive observation both consume calls, tokens, elapsed compute, and dollars; deterministic trace rules, clocks, and context-use add zero provider calls.
  - Qualification: Passive provider latency is off the target path and must not be described as agent delay.
- **supported_for_one_deployment_slice:** Natural-policy deployment is method-by-operator specific. Descriptive leaders were No state action: Frozen quiz, Trace rules (0.750); Lossy compaction: Frozen recompute (0.800); Public-state reground: Active recompute (0.850); quote-only WATCH reminder: Trace rules (0.825).
  - Qualification: Use the exact paired intervals, action incidence, and interactions—not point-estimate ranks alone. Deployment covers Luna on 40 Evolving Intent tasks only.
- **sensitivity_result:** The paired n=38 cumulative two-source-task omission audit assessment is `scientific_outcome_conclusion_changed`.
  - Qualification: Report every outcome/action/resource flag listed in the sensitivity receipt, including any changed conclusion.
- **supported_as_controlled_sensitivity:** Under the common aggressive checkpoint-1 schedule, the best point estimate depends on the state operator; no method is universally dominant.
  - Qualification: The yoked study reuses the same 40 source tasks and is neither natural timing nor an independent replication.

## Powered active observer effect

Active recomputation usually reduced success, but not universally: 6/7 point estimates were negative, 3 paired 95% intervals excluded zero below, and one stratum was positive.

| benchmark | model | clean success | active success | active − clean | paired 95% CI |
|---|---|---:|---:|---:|---:|
| bfcl_multi_turn | GPT-5.6 Luna | 0.464 | 0.357 | -0.107 | [-0.214, +0.000] |
| bfcl_multi_turn | GPT-5.6 Terra | 0.411 | 0.393 | -0.018 | [-0.107, +0.071] |
| bfcl_multi_turn | GPT-OSS-120B | 0.268 | 0.089 | -0.179 | [-0.286, -0.089] |
| evolving_intent_gsm8k | DeepSeek V4 Flash | 0.821 | 0.571 | -0.250 | [-0.393, -0.107] |
| evolving_intent_gsm8k | GPT-5.6 Luna | 0.750 | 0.786 | +0.036 | [-0.071, +0.143] |
| evolving_intent_gsm8k | GPT-5.6 Terra | 0.786 | 0.732 | -0.054 | [-0.161, +0.054] |
| evolving_intent_gsm8k | GPT-OSS-120B | 0.679 | 0.500 | -0.179 | [-0.339, -0.018] |

## Signal-quality winners (AUPRC)

No universal method wins. Full precision, recall, AUPRC, thresholds, and denominators are in the companion JSON.

| benchmark | model | analyzable tasks | highest AUPRC | method(s) |
|---|---|---:|---:|---|
| evolving_intent_gsm8k | DeepSeek V4 Flash | 56 | 0.429 | Active recompute |
| evolving_intent_gsm8k | GPT-5.6 Luna | 56 | 0.421 | Trace judge |
| evolving_intent_gsm8k | GPT-5.6 Terra | 56 | 0.525 | Context use |
| evolving_intent_gsm8k | GPT-OSS-120B | 56 | 0.491 | Active recompute |
| bfcl_multi_turn | GPT-5.6 Luna | 52 | 0.797 | Trace judge |
| bfcl_multi_turn | GPT-5.6 Terra | 52 | 0.787 | Trace judge |
| bfcl_multi_turn | GPT-OSS-120B | 52 | 0.942 | Active recompute |

## Online deployment: exact success estimates

These are natural-policy results. Firing/action incidence must be shown beside them.

| operator | method | success | paired 95% CI |
|---|---|---:|---:|
| No state action | Active recompute | 0.725 | [0.575, 0.875] |
| Lossy compaction | Active recompute | 0.725 | [0.575, 0.850] |
| Public-state reground | Active recompute | 0.850 | [0.725, 0.950] |
| quote-only WATCH reminder | Active recompute | 0.725 | [0.575, 0.850] |
| No state action | Frozen recompute | 0.725 | [0.575, 0.850] |
| Lossy compaction | Frozen recompute | 0.800 | [0.675, 0.925] |
| Public-state reground | Frozen recompute | 0.775 | [0.625, 0.900] |
| quote-only WATCH reminder | Frozen recompute | 0.750 | [0.625, 0.875] |
| No state action | Frozen quiz | 0.750 | [0.600, 0.875] |
| Lossy compaction | Frozen quiz | 0.450 | [0.300, 0.600] |
| Public-state reground | Frozen quiz | 0.675 | [0.525, 0.825] |
| quote-only WATCH reminder | Frozen quiz | 0.750 | [0.625, 0.875] |
| No state action | Trace judge | 0.725 | [0.575, 0.850] |
| Lossy compaction | Trace judge | 0.700 | [0.550, 0.850] |
| Public-state reground | Trace judge | 0.700 | [0.550, 0.825] |
| quote-only WATCH reminder | Trace judge | 0.675 | [0.525, 0.825] |
| No state action | Trace rules | 0.750 | [0.625, 0.875] |
| Lossy compaction | Trace rules | 0.775 | [0.650, 0.900] |
| Public-state reground | Trace rules | 0.725 | [0.575, 0.850] |
| quote-only WATCH reminder | Trace rules | 0.825 | [0.700, 0.925] |
| No state action | Turn clock | 0.725 | [0.575, 0.850] |
| Lossy compaction | Turn clock | 0.050 | [0.000, 0.125] |
| Public-state reground | Turn clock | 0.750 | [0.600, 0.875] |
| quote-only WATCH reminder | Turn clock | 0.775 | [0.650, 0.900] |
| No state action | Context use | 0.725 | [0.575, 0.850] |
| Lossy compaction | Context use | 0.525 | [0.350, 0.675] |
| Public-state reground | Context use | 0.700 | [0.550, 0.825] |
| quote-only WATCH reminder | Context use | 0.725 | [0.575, 0.850] |

### Active versus other methods within each operator

Positive effects favor the comparison method.

| operator | comparison − active | effect | paired 95% CI |
|---|---|---:|---:|
| No state action | Frozen recompute − active | +0.000 | [-0.150, +0.150] |
| Lossy compaction | Frozen recompute − active | +0.075 | [-0.025, +0.175] |
| Public-state reground | Frozen recompute − active | -0.075 | [-0.175, +0.025] |
| quote-only WATCH reminder | Frozen recompute − active | +0.025 | [-0.075, +0.125] |
| No state action | Frozen quiz − active | +0.025 | [-0.100, +0.150] |
| Lossy compaction | Frozen quiz − active | -0.275 | [-0.425, -0.150] |
| Public-state reground | Frozen quiz − active | -0.175 | [-0.300, -0.075] |
| quote-only WATCH reminder | Frozen quiz − active | +0.025 | [-0.075, +0.125] |
| No state action | Trace judge − active | +0.000 | [-0.125, +0.150] |
| Lossy compaction | Trace judge − active | -0.025 | [-0.175, +0.125] |
| Public-state reground | Trace judge − active | -0.150 | [-0.300, -0.025] |
| quote-only WATCH reminder | Trace judge − active | -0.050 | [-0.200, +0.075] |
| No state action | Trace rules − active | +0.025 | [-0.125, +0.175] |
| Lossy compaction | Trace rules − active | +0.050 | [-0.025, +0.150] |
| Public-state reground | Trace rules − active | -0.125 | [-0.225, -0.025] |
| quote-only WATCH reminder | Trace rules − active | +0.100 | [+0.025, +0.200] |
| No state action | Turn clock − active | +0.000 | [-0.150, +0.150] |
| Lossy compaction | Turn clock − active | -0.675 | [-0.800, -0.525] |
| Public-state reground | Turn clock − active | -0.100 | [-0.225, +0.000] |
| quote-only WATCH reminder | Turn clock − active | +0.050 | [-0.025, +0.150] |
| No state action | Context use − active | +0.000 | [-0.125, +0.125] |
| Lossy compaction | Context use − active | -0.200 | [-0.350, -0.049] |
| Public-state reground | Context use − active | -0.150 | [-0.275, -0.025] |
| quote-only WATCH reminder | Context use − active | +0.000 | [-0.100, +0.100] |

## Robustness and required disclosures

- Paired cumulative n=38 assessment: **scientific_outcome_conclusion_changed**.
- Scientific-outcome change flags: 23; action-policy flags: 12; resource flags: 18.
- Three passive trace-judge cells spanning two of 40 deployment source tasks required same-prefix semantic recovery after output-cap truncation. Cell 89df recovered checkpoint 6 with cap 640; cell 786 used cap 640 at checkpoints 5 and 6. Executor attribution for the 89df and 786 recovery suffixes is unknown. Raw production artifacts were retained; a hash-bound copy counted every physical attempt and normalized one ordinary HTTP 503 ledger status, and a paired n=38 sensitivity omitted both affected source tasks from all 28 treatments.
- The two-pass passive pass had one documented trace-judge recovery. Trace judge was not among the four methods selected for the final active-anchored yoked schedule; the recovery remains part of the reproducibility record.
- Evolving Intent has final-success labels, not independently graded within-horizon failure labels. BFCL supplies the action-trace/turn-level early-warning evidence.
- `good_bad_watch_feedback` is a deterministic quote-only WATCH reminder—not an LLM-generated assessment of good and bad decisions.

## Claims the paper must not make

- ~~Active observation always harms task success.~~ One of seven powered active-recompute effects is positive; four intervals include zero.
- ~~Observer harm is consistent without exceptions across four or five powered models.~~ The result is a 6/7 trend with a positive Luna/Evolving exception, not a rule.
- ~~Active signals are universally less accurate than passive signals or trivial baselines.~~ Active recomputation leads AUPRC in some powered slices.
- ~~Passive signals universally outperform clock and context baselines.~~ Powered signal winners vary by model and benchmark.
- ~~Increasing active chore complexity monotonically improves detection or worsens observer harm.~~ The n=20 mechanism arms are heterogeneous and explicitly exploratory.
- ~~Evolving Intent supplies independently graded within-N-turn degradation labels.~~ It supplies verified final success; BFCL supplies turn-level action-failure evidence.
- ~~The deployed feedback is an LLM-generated GOOD/BAD decision critique.~~ It is a deterministic, bounded, quote-only WATCH reminder.
- ~~Primary online intervention counts are matched across methods.~~ The primary study uses unequal natural scalar firing rates with a one-action cap.
- ~~Deployment results generalize across models or to action traces.~~ Deployment covers one model and one reasoning benchmark; BFCL was not deployed.
- ~~The final deployment has a truly unmonitored arm or an oracle bound.~~ operator=none is monitored no-action; neither final manifest includes an oracle.

## Recommended figure allocation

### Main paper

- **main_figure_1:** Cross-model active observer-effect trend plus explicitly exploratory burden ladder
  - `experiments12/generated/active-probe-ladder-confirmatory-v1.svg` (`958448b050709abacc3cb2db590a7691ff4bd859e7bdbcbc27643cc936f61b73`)
- **main_figure_2:** Provider overhead of active and passive observation
  - `experiments12/generated/observer-overhead-confirmatory-v1.svg` (`78b12d9f0601ec0d22a3582d85932b0c92f8c6cd8c0885e2138833687e545bc6`)
- **main_figure_3_source_panel:** One of all seven powered signal-quality panels; assemble all panels without cherry-picking
  - `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-deepseek-v4-flash-0731.svg` (`3341cc6f03eab6f4cc2840372f89167d794024f6f1311d7334cec573d145b573`)
  - `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-gpt-5.6-luna.svg` (`19e7949be3408823e8b23c8b1cffb92aee4f5537ad3ea79abb2f6b4c1415a04c`)
  - `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-gpt-5.6-terra.svg` (`4db9b312062cd7e8cb5c1853f763d5f0e172c2e0ab0046e160c7c510ea305872`)
  - `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/signal-figures/signal-pr-evolving_intent_gsm8k-gpt-oss-120b.svg` (`e5793ecfc737e0c3e34c7688e931ae011a3c56022a82454d1d67149b33fb8e34`)
  - `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures/signal-pr-bfcl_multi_turn-gpt-5.6-luna.svg` (`53d08f32e0c6516ef8b83127abcbd5417cf72a870371201c65a4fea28743b0bd`)
  - `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures/signal-pr-bfcl_multi_turn-gpt-5.6-terra.svg` (`6907e6e7ddacf9abb0f2993637b69f492650bc77a2ae79c85231f03916f602d3`)
  - `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures/signal-pr-bfcl_multi_turn-gpt-oss-120b.svg` (`47bd4a5ab3be8727c638bfe1138f670e70749a57c413a817777513634b882d05`)
- **main_figure_4a:** Natural-policy success by observation method and state operator
  - `experiments12/generated/deployment-paper-post-analysis-v1/online-performance.svg` (`dabc101efbb6091227936cc5fbf4fd727f56164d1837addce18a58006d99ce3e`)
- **main_figure_4b:** Realized firing/action incidence needed to interpret natural-policy success
  - `experiments12/generated/deployment-paper-post-analysis-v1/online-firing-actions.svg` (`f47840a3f685a72a8b41afc1195a3066ae61d86bf1fd4da34da5254191b5dc65`)
- **main_figure_5_if_space:** Specific method-by-operator success interactions relative to active recomputation
  - `experiments12/generated/deployment-paper-post-analysis-v1/online-success-interactions.svg` (`da78adaddec4970f2f0efe422c44c1b4b772ead6e0aaab57468dd038dcd40d8b`)

For the seven precision-recall source panels, assemble all four Evolving plus all three BFCL panels into one figure; showing a hand-picked subset would be misleading.

### Appendix

- `experiments12/generated/active-probe-mechanism-exploratory-v1.svg` — Exploratory copy/recall/counter/recompute mechanism effects
- `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/observer-figures/observer-effect-active_recompute-actual_cost_usd.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/observer-figures/observer-effect-active_recompute-latency_ms.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/observer-figures/observer-effect-active_recompute-observer_tokens.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/observer-figures/observer-effect-active_recompute-task_tokens.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/observer-figures/observer-effect-active_recompute-total_tokens.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-evolving-core-v2/results/observer-figures/observer-effect-active_recompute.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/observer-figures/observer-effect-active_recompute-actual_cost_usd.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/observer-figures/observer-effect-active_recompute-latency_ms.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/observer-figures/observer-effect-active_recompute-observer_tokens.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/observer-figures/observer-effect-active_recompute-task_tokens.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/observer-figures/observer-effect-active_recompute-total_tokens.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/observer-figures/observer-effect-active_recompute.svg` — Per-benchmark observer-effect/resource detail
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures-complete-case/signal-pr-bfcl_multi_turn-gpt-5.6-luna.svg` — BFCL complete-case signal sensitivity
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures-complete-case/signal-pr-bfcl_multi_turn-gpt-5.6-terra.svg` — BFCL complete-case signal sensitivity
- `experiments12/artifacts/e12-confirmatory-bfcl-core-v3/results/signal-figures-complete-case/signal-pr-bfcl_multi_turn-gpt-oss-120b.svg` — BFCL complete-case signal sensitivity
- `experiments12/generated/deployment-paper-post-analysis-v1/online-resources.svg` — Online end-to-end tokens/cost
- `experiments12/generated/deployment-paper-post-analysis-v1/yoked-controlled-sensitivity.svg` — Aggressive checkpoint-1 active-anchored controlled sensitivity

## Immutable material inventory

Every input, validation, receipt, figure, and sidecar path has an exact SHA256 and byte size in `PAPER_MATERIALS12.json`.
