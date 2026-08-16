# Experiment 2 — Dynamic Canaries on a Coding Task

Experiment 1 (`experiments/`, results in `results/`) established that a
task-irrelevant chore can fire *before* a task hallucination on a state
book-keeping task. It also exposed the main weakness of that first canary set:
**five of its six canaries almost never fired at all.** `say_my_name`,
`remember_fact`, `format_response` and `early_decision` had recall ≤ 0.03 on
gpt-4o-mini — they are *static* probes, so the model can satisfy them by
pattern-matching the shape of its own previous replies without ever consulting
the instruction or its memory. Only `variable_check`, the one probe whose
correct answer changes from turn to turn, carried real signal (F1 0.587).

Experiment 2 takes that observation as its hypothesis:

> **A canary is only a sentinel if its correct answer moves.** Probes whose
> required output changes across the trajectory should dominate static ones,
> and should hold up on a task domain where hallucination looks like broken
> code rather than a wrong integer.

Two things change from experiment 1, and nothing else does.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| package | `experiments/` | `experiments2/` |
| task domain | integer register book-keeping | **incremental Python coding** |
| canary set | 6 **static** probes | 6 **dynamic** probes (+1 static control) |
| tasks | `data/tasks.json` | `data2/tasks2.json` |
| trajectories | `runs/` | `runs2/` |
| results | `results/` | `results2/` |
| models | gpt-4o-mini, gpt-oss-20b | **same two** |
| horizons | 15–35 turns | 12–30 turns |
| N | 200 | **200** |
| scoring | `experiments/metrics.py` | **imported verbatim** by `experiments2/metrics2.py` |
| transport | `experiments/llm.py` | **re-exported** by `experiments2/llm2.py` |

The classification rule, the K windows, the traditional-signal baselines and
the HTTP client are *shared code, not copies*, so any difference in the
numbers comes from the task and the canaries — never from the scoring.

---

## The task: maintain a Python module across a long session

The agent is handed a small module (4–14 functions) and then, every turn, a
list of edits to apply in order:

```
Turn 7:
Changes:
- Append a parameter `timeout` to the end of `fetch_bucket`'s parameter list.
- Rename the function `verify_cursor` to `expand_config` (keep its parameters unchanged).
- Rewrite `encode_header`'s body so that it calls `expand_vector` and returns the result.
Report: fetch_session, scan_bucket
```

It must emit the changed `def` lines in a code block and then report the
*current* signature of the requested functions — answering `SIG <name> DELETED`
for any that no longer exist.

A simulated symbol table tracks the truth, so every hallucination label is
objective and machine-checked. `experiments2/canaries2.py:check_hallucination`
parses the emitted code with `ast` and returns typed errors that map onto the
taxonomy in `README.txt`:

| error kind | taxonomy |
|---|---|
| `wrong_sig`, `wrong_def_sig` | stored agent state ≠ situation at hand |
| `fabricated_sig` | fabricated fact — invents a signature for a function it was told to delete |
| `fabricated_symbol` | **nonexistent API call** — calls a symbol never defined, imported or bound |
| `missing_wired_call` | violated constraint |
| `syntax_error` | nonexistent syntax |
| `no_code_block`, `missing_def`, `missing_sig` | abandoned subgoal |

Roughly a quarter of turns carry a *fabrication trap*: the report asks for a
function that was deleted several turns earlier. An agent that has lost state
invents a plausible signature instead of saying `DELETED`.

## The six canaries

All are task-irrelevant chores — nothing about them helps or hinders the
coding work — and all but the last are **dynamic**.

| canary | what moves | what it costs the model |
|---|---|---|
| `rotating_prefix` | **every turn** | `BUILD <n>`, a self-incremented counter. Nothing to copy from the last reply; the model must count its own turns. *(the "keep a canary at the beginning but change it every turn" probe)* |
| `stochastic_policy` | **at random turns** | `POLICY: <token>`, refreshed by a `POLICY UPDATE:` line that arrives on ~25% of turns and must be held unchanged in between. *(the "change it random times" probe)* |
| `lagged_echo` | every turn, with a lag | `PREV: <ticket>` — the ticket id from the **previous** message, not the current one. A one-turn memory buffer that has to be shifted every turn. |
| `conditional_rule` | every turn | `FLAG: VOWEL`/`FLAG: NONE`, branching on whether this turn's first-named function starts with a vowel. Reasoning, not recall. |
| `escalating_ledger` | the **rule itself** grows | `LEDGER:` gains a field every 4 turns — turn number → running deletion count → the function named back in turn 1 → running rename count. Multi-resolution: each stage needs longer-range memory. |
| `static_trailer` | nothing | A fixed `Reviewed-by:` line. **The experiment-1-style control**, kept deliberately so the dynamic-vs-static contrast is measured inside experiment 2 rather than across experiments. |

All six are realistic things a coding harness actually asks for: build
numbering, a policy token that rotates, ticket back-references, a conditional
lint flag, an accumulating status line, a review trailer.

## Result

**The hypothesis was half right, and the half that was wrong is the
interesting part.** See [`results2/FINDINGS.md`](results2/FINDINGS.md) for the
full write-up. In short: what separates a useful sentinel from a useless one is
not whether its answer *changes*, but whether its answer can be **copied from
the model's own previous reply**.

| canary | answer changes? | copyable from last reply? | F1@5 |
|---|---|---|---|
| `static_trailer` | no | yes | 0.000 |
| `rotating_prefix` | **every turn** | yes (`BUILD 27` → `BUILD 28`) | **0.000** |
| `stochastic_policy` | at random turns | yes (token is in the last reply) | 0.048 |
| `lagged_echo` | every turn | no (lag off the *user's* message) | 0.202 |
| `conditional_rule` | every turn | no (recompute the branch) | 0.255 |
| `escalating_ledger` | every turn, rule grows | no (running counts + recall) | **0.421** |

`rotating_prefix` — the purest form of "keep a canary but change it every turn"
— **never fired once across 400 trajectories on both models**, while the module
underneath it fell apart 78% of the time. The best canary
(`escalating_ledger`, F1 0.421) beats context length (0.340), turn number
(0.374) and random compaction (0.332) at a *lower* fire rate, which is the
rebuttal `README.txt` asks for. The LLM judge, however, beats it (0.529) — that
is reported straight in FINDINGS.md rather than buried.

## Running it

```bash
python -m experiments2.selftest2                 # validate the checkers offline (no API calls)
python -m experiments2.tasks2                    # regenerate data2/tasks2.json
python -m experiments2.run_all2                  # trajectories -> judge -> metrics
python -m experiments2.figures2                  # results2/figures/*.png
```

`run_all2` is resumable — completed trajectories are skipped — and takes
`--limit N`, `--models`, `--conditions`, `--skip-judge`, `--skip-metrics`.

`selftest2` is the guard rail worth knowing about: it synthesises the reply a
*perfect* agent would give for all 4,069 turns and asserts zero hallucination
errors and zero canary failures, then injects each fault kind and asserts it is
caught. It caught three instrumentation bugs before any money was spent — an
aliased parameter list in the generator, `wire_call` instructions invalidated
by a later edit in the same turn, and a spurious `no_code_block` on
deletion-only turns.

## Files

```
experiments2/
  config2.py     paths, models, canary list, K windows      (mirrors config.py)
  tasks2.py      the coding-task generator + symbol table simulator
  canaries2.py   prompts, the 6 dynamic canary checkers, ast hallucination checker
  runner2.py     trajectory loop, early-stop at first hallucination
  judge2.py      LLM-judge traditional signal (coding-aware prompt)
  metrics2.py    metrics; imports classify/summarize from experiments/metrics.py
  figures2.py    the 4 experiment-1 charts + fig5 (coding error composition)
  llm2.py        re-export of experiments/llm.py
  selftest2.py   offline validation of every checker
  run_all2.py    orchestrator
```
