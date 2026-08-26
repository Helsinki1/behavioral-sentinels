# Experiment 12: clean study protocol

Experiment 12 is the paper experiment for **Active and passive observation
methods for reasoning and action agent traces**. It is a clean restart: it does
not reuse old outcomes, and it leaves Experiments 1--11 intact.

## Questions, in order

1. **Observer effect:** does writing an active probe and its answer into the
   continuing trace change task success, tokens, or latency?
2. **Signal quality:** from the same information cutoff, how accurately and how
   early do active probes, frozen quizzes, an outside trace judge, deterministic
   trace checks, a clock, and context length predict failure?
3. **Deployment value:** after thresholds are fixed on calibration tasks, when
   does each signal help if it triggers lossy compaction, exact public-state
   regrounding, or a short evidence-based `GOOD / BAD / WATCH` note?

These are empirical questions, not assumed rules. Model-specific exceptions are
part of the result. We will not change tasks, thresholds, exclusions, or sample
sizes after seeing a confirmatory effect's sign.

## Core design

Every task/model pair is blocked and arm order is randomized. The clean arm and
each active arm receive the same task turns, tools, decoding settings, and token
limits. Active probe text and answers remain in the target agent's history.
Passive methods read a frozen prefix and are discarded; their text never enters
the target trajectory.

The observer-effect arms are:

| arm | active chore | main burden | exact probe output length |
|---|---|---|---|
| `clean` | none | none | 0 |
| `active_name_copy` | say the supplied name/code | direct copy | fixed |
| `active_name_recall` | say the initial name/code | memory ablation | fixed |
| `active_counter` | increment a carried number | memory + update | fixed |
| `active_recompute` | recompute a new arithmetic check | reasoning | fixed |

Name-copy, name-recall, and counter are pilot/mechanism arms. The powered
cross-model core is `clean` versus `active_recompute`: recompute is the only
reasoning chore with an exact passive, discarded counterpart. All four active
formats use the same delimiter and character count. The pilot therefore tests
copyability, memory load, and reasoning load without multiplying the main
comparison after target outcomes are visible.

Passive observations on clean prefixes are:

- `frozen_probe`: for the stateless name-copy and recompute variants, the exact
  active probe call is made on a copied prefix and discarded. This isolates
  carry from probe content and compute. Stateful recall/counter probes cannot
  honestly have an exact zero-carry history and are not labeled matched.
- `frozen_quiz`: the same target model answers benchmark-state questions on a
  copied conversation; the fork is graded and discarded.
- `trace_judge`: one fixed outside model returns structured risk and evidence;
  benchmark answers and future turns are never shown to it.
- `trace_rules`: deterministic, answer-blind checks for invalid structure,
  repetition, contradictions, execution errors, or broken task protocol.

Clock and context-use scores are non-adaptive baselines. Threshold sweeps are
fit only on calibration tasks. Passive monitors are not trajectory arms: making
them separate target runs would waste money and destroy the zero-carry claim.
Every run embeds the same hashed passive-monitor contract from
`passive_spec12.py`: all six methods, the two matched stateless variants, both
domain quiz generators and versions, checkpoint timing, decoding settings, and
the enabled Sol judge/model/output limit. A shadow is reusable only if its spec
hash and exact method/variant coverage match at every checkpoint.

## Benchmarks and legal/reproducibility gates

### Evolving Intent GSM8K (reasoning/deployment core)

Use [Microsoft Evolving Intent](https://github.com/microsoft/evolving-intent)
at commit `993d6be9597ac03854b46362ccd647eb1bfd267a` under its MIT license. Use the
official 200 GSM8K IDs and compare the one-turn condition with the seven-turn,
two-revision, two-goal-switch condition. Do not add the old artificial `WAIT`
instruction.

The one-turn condition is a clean task baseline only: it has no actionable
checkpoint, so active observation arms are rejected at initialization and at
runtime. Observer-effect and signal cells use the seven-turn condition.

The exact generated GSM8K conversations are not released upstream. Before a
real run, either obtain the authors' file and checksum or generate one frozen
reproduction, record its generator model/settings/seed, and hash every rendered
conversation. Never regenerate data by arm or model. This benchmark has only a
verified final answer, so it supports final success, observer-effect, cost, and
deployment claims—not primary turn-level precision/recall.

### BFCL multi-turn (action core)

Use the [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla)
at pinned commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (Apache-2.0), with native tools
for all five target models. Preserve BFCL's environment, execution results, and
official episode score. Log invalid calls and execution failures per turn; only
call a per-turn label "ground truth" if the official state checker supports it.
A common prompt-emulated JSON-action subset is a sensitivity check, not the
primary ecological result.

### TurnBench-MS (controlled reasoning, permission-gated)

Use [TurnBench-MS](https://github.com/grantzyr/TurnBench-MS) at commit
`b3a9daa914e66f62048b62cff06bcaf4151aadb5`, beginning with the 45-game Classic
subset. The repository currently has no license. Its prompts, data, and code
must stay in a user-supplied external checkout (`TURNBENCH_MS_ROOT`) and must not
be copied here unless the authors grant permission or add a license. The
paper's process extractor/annotations are also absent; any replacement must be
frozen and manually audited. Until both issues are resolved, TurnBench is an
explicit extension—not evidence silently mixed into the core paper.

Each external checkout, generated dataset, task list, prompt, and tool schema is
content-hashed into the run manifest.
For Evolving Intent, the rendered dataset and its generation/build receipt are
both required and separately hashed into that manifest.

## Outcomes and timing

Primary task outcome is the benchmark's official episode success. A checkpoint
is observed **after** target turn `t`, so it can only prevent failures from turn
`t+1` onward. We report same-turn detection separately from genuine early
warning.

For benchmarks with valid per-turn labels, the primary checkpoint label is
"any independently graded failure in `t+1 ... t+H`". BFCL uses its recorded
invalid-call, execution-failure, and state-check-failure indicators; a remaining
official episode failure is placed at the final horizon. Evolving Intent has no
valid intermediate labels, so its failure label remains final-only. We report
precision-recall curves/AUPRC, calibration, lead time, firing rate, tokens,
latency, and dollars. The task—not the turn—is the statistical unit.

Observer effect is six paired `active − clean` contrasts on the exact same
task IDs: success, task-call tokens, observer-call tokens, total tokens, total
call latency, and reconciled recorded cost. Every interval resamples paired
tasks; cached and reasoning subtotals are not counted twice. Retries are
included; when billing is unknowable, the ledger's explicit conservative upper
bound remains in the total rather than being silently discarded.

Fixed-rate signal comparisons rank complete tasks by score and use a seeded
hash of task identity only to break ties. They fire on `floor(rate × n)` tasks,
so binary monitors attain the closest non-exceeding rate with error below
`1/n`, without using outcome labels. Natural thresholds remain secondary.

## Deployment study

Thresholds for the frozen recomputation variant are selected on calibration
tasks and then frozen. Deployment uses new tasks and compares:

- fixed clock and context limit;
- selected active probe;
- matched frozen (zero-carry) probe;
- frozen quiz;
- outside trace judge;
- deterministic trace rules;
- no-intervention and oracle bounds where definable.

Intervention counts are matched or yoked so a method is not rewarded merely for
resetting more often. The two primary recovery operators are:

1. `compact`: clear the trace and seed a short, deterministic lossy excerpt;
2. `reground`: clear the trace and reconstruct instructions plus only the public
   state available from the environment.

The secondary `feedback` operator keeps the trace and appends at most 80 tokens
under `GOOD`, `BAD`, and `WATCH`, each tied to quoted trace evidence. Its cost is
separate and it is never described as observation alone.

The primary deployment experiment is genuinely online: after each current
checkpoint, the chosen method sees only the prefix available then, decides
whether to intervene, and any later observation sees the resulting altered
trace. Natural thresholds are locked on calibration tasks and each policy may
intervene at most once per task. Online preparation rejects any other cap, and
the value is bound into both the threshold lock and run manifest. Frozen
two-pass replay is retained only as one controlled sensitivity analysis; it is
not presented as an adaptive agent run. That sensitivity freezes the
`yoked_anchor` estimand with at most one action per task, `active_recompute` as
the anchor, methods `active_recompute`, `frozen_probe:recompute`, `turn_clock`,
and `context_use`, and operators `none`, `lossy_compaction`, and
`public_state_reground`. The online core uses seven-turn Evolving Intent tasks
because its public state can be reconstructed exactly; BFCL supplies the
separate action-trace observer and signal-quality evidence.

## Fixed five-model slate

- Fireworks: GPT-OSS-120B, DeepSeek-V4-Flash-0731, Qwen3.7 Plus.
- OpenAI: GPT-5.6 Luna and GPT-5.6 Terra.
- GPT-5.6 Sol is an offline trace judge, never task ground truth.

Exact IDs, endpoints, prices, and official source links are frozen in
`model_prices.json`. Availability is checked through the providers' free
`/models` endpoints before a paid call. For mutable OpenAI aliases, every call
records date, returned model identifier, response/request identifiers when
available, service tier, settings, and usage.

## Spend and stage gates

The ledger refuses reservations above **$500 OpenAI** or **$30 Fireworks**.
Operational stops are $400 and $24, leaving $100 and $6 for failures or a
deliberately approved extension. The planned OpenAI envelope is $100 target
trajectories, $250 offline judging, and $50 retry/audit reserve. Prices are
estimated without cache savings.

Runs advance only through these gates:

1. `offline`: schemas, grading, probe invariants, reset contamination tests, and
   mocked budget-failure tests; $0. The mandatory `test_launch_gate12.py` also
   runs the complete provider-free freeze → observe/shadow → calibrate →
   two-pass deploy → validate/analyze/plot path on disjoint source tasks.
2. `smoke`: three tasks/benchmark on one model, then five tasks per cell on all
   models. Validate adapters, parsing, history separation, returned usage, and
   hashes. Stage caps: $5 Fireworks / $25 OpenAI.
3. `baseline_gate`: 20 held-out task IDs per benchmark/model for benchmark
   difficulty, trace length, failure prevalence, and measured token profile.
   Do not choose benchmarks based on an active-effect sign.
4. `calibration`: freeze the exact passive-method set, fit monitor thresholds
   and hash-stable firing-rate matches on declared calibration source IDs, then
   freeze prompts, exclusions, arms, operators, `H`, and sample size.
5. `confirmatory`: bind the threshold-artifact hash and its source calibration-
   manifest hash into the new run manifest, then run it once. Calibration and
   test source IDs are globally disjoint across models, conditions, and
   replicates. Missing/extra methods or cells fail closed; analysis never
   intersects whatever outputs happen to exist.

The final sample size is set after the baseline token/variance estimate but
before active comparisons are inspected. Fireworks is the binding budget: if
traces are near the conservative 20k-input/3k-output profile, six broad
conditions support roughly 55 tasks per model/benchmark at $24. If traces are
longer, reduce secondary arms/benchmarks transparently; never reduce only cells
whose early results are inconvenient.

## Reproducible artifact contract

Each run contains a manifest, a predeclared pair table, an SQLite call/job/spend
ledger, append-only call events, exact transcripts, monitor outputs keyed by
source-trajectory hash, and a completeness report. A retry is a separate call
attempt and its usage/cost is never dropped. Confirmatory analysis requires the
manifest's code/config/task hashes to match and reports every missing/error cell.
The measured baseline profile also freezes each model/benchmark/condition's
exact official success count and derived success rate alongside resource p95s.

The provider-free production preparation command is
`python -m experiments12.prepare_deployment12 --help`. It accepts a completed
observation run plus the frozen calibration extract/thresholds. It requires the
deployment source-allocation registry, measured-baseline resource profile, and
cost/sample-size projection lock; checks global source-task disjointness; and
writes the exact pass-one, deployment-threshold, schedule, pair,
Evolving-provenance, and manifest receipt chain before any pass-two outcome
exists. An outcome-blind realized-allocation receipt is accepted only for
registry-ordered structural replacements. Active signal hashes are recomputed from the carried history through
each probe response; passive signal hashes are recomputed from the immutable
clean task-record prefix.

The required two-pass source observation run is initialized provider-free with
`python -m experiments12.deployment_pass_one12 --help`. It binds the final
deployment allocation, method/operator cost lock, calibration thresholds,
measured profile, deployment estimand, one-action caps, yoked anchor, and final
randomization seed. It declares only `clean` plus the selected active source
arms under `operator=none`; passive/baseline scores then come from the clean
shadow. Production pass-one and online deployment both freeze exactly one
replicate so the source task remains the statistical unit.

The completed two-pass sensitivity is analyzed provider-free with
`python -m experiments12.two_pass_analysis12 extract --help` (or the compatible
`experiments12.deployment_analysis12` entry point). Analysis requires exact
manifest, schedule, output, job, event, attempt, and ledger coverage; regrades
Evolving Intent from the frozen dataset; and writes paired task-bootstrap
summaries, method contrasts, tables, and figures.

The completed primary online run is analyzed provider-free with
`python -m experiments12.adaptive_analysis12 extract --help`. That path requires
exact manifest, pair, output, job, event, threshold, attempt, and ledger
coverage; reports absolute task-level performance and paired
`operator - none` effects with task-bootstrap intervals; and writes deployment
success figures with exact JSON sidecars.

No full paid run should begin until the Evolving Intent dataset is frozen,
TurnBench's permission decision is recorded, BFCL integration passes its state
checks, and the calibration manifest is signed off.
