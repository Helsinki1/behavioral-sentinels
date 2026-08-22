"""The reset-capable agent harness for experiment 4.

Compaction operator (identical in every arm that resets)
--------------------------------------------------------
On a reset at turn t we ask the agent to write out the module state *as it
currently believes it to be*, then throw away the conversation and restart from
that self-summary plus the standing rules.

No ground truth is injected. This is the honest operator for state that lives
only in context: resetting EARLY preserves a still-correct state and drops the
accumulated noise, while resetting LATE canonicalises whatever error the agent
has already made. That asymmetry is exactly the mechanism that would make an
early-warning signal valuable, so it gives the sentinel a fair chance to win.

(The alternative -- replaying the user's own instruction history -- would
perfectly reconstruct the true state and make every reset policy look
identical and excellent. That is oracle leakage dressed up as compaction.)
"""
import re

from experiments2.canaries2 import (CANARY_INSTRUCTIONS, SYSTEM_PROMPT,
                                    build_turn_body, check_canary,
                                    check_hallucination, render_module)
from .config4 import TEMPERATURE
from experiments.llm import chat

SENTINEL = "escalating_ledger"

BRIEFING = """You are maintaining a small Python module for me over a long session.

Current module:
```python
{module}
```

Every turn I will give you (a) changes to apply, in order, and (b) functions to report.
Rules:
- Track the state of the module across the entire conversation. Do not use tools, and do not ask me to re-send the module.
- Apply the changes strictly in the order listed.
- Emit exactly one ```python code block containing, for every function you created or changed this turn, its full updated `def` line followed by a one-line body. Use `return None` as the body unless a change tells you otherwise. Do not include functions you did not touch.
- After the code block, for each function I ask you to report, output exactly one line:
  SIG <name>(<parameter names, comma-separated>)
  If that function no longer exists, output exactly: SIG <name> DELETED
- Do not restate the rest of the module. Do not show your working. Be concise.

For example, if I say `Report: parse_config, old_helper` and parse_config currently takes
(path, strict) while old_helper has been deleted, the report part of your reply is exactly:
SIG parse_config(path, strict)
SIG old_helper DELETED"""

DUMP_REQUEST = (
    "Pause the task for one turn. Write out the CURRENT state of the module exactly as you "
    "believe it stands right now: one `def` line per function that currently exists, with its "
    "full parameter list, inside a single ```python code block, each with body `return None`. "
    "Include every function you believe exists and no others. No commentary."
)

RESUME = """We are resuming a session that was compacted to save context. This is the module state carried forward:

```python
{module}
```

{rules}"""


def _ledger_seed(turn):
    """After a reset the agent has lost the history its LEDGER depends on, so
    carry the current field values forward -- compaction legitimately preserves
    a small amount of state. Without this the sentinel would be guaranteed to
    fail on the turn after every reset, which would be an artifact, not a
    signal."""
    return ("For reference, your LEDGER line as of the last turn was: LEDGER: "
            + " ".join(turn["ledger_expect"]) + ". Continue it from there.")


def extract_module(text):
    """Parse the agent's self-summary into {fn: [params]}."""
    out = {}
    for m in re.finditer(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", text):
        params = []
        for p in m.group(2).split(","):
            p = p.strip().strip("`*_ ").split("=")[0].split(":")[0].strip().lstrip("*").strip()
            if p and p != "/":
                params.append(p)
        out[m.group(1)] = params
    return out


def true_module_at(task, turn_index):
    """Replay the user's own instructions to get the TRUE module state entering
    turn `turn_index` (0-based). Verified against the generator's truth fields
    on all 2,060 turns of the pool. This is the external store a re-grounding
    reset reads from -- the analogue of a fresh session re-reading the repo."""
    fns = {k: list(v) for k, v in task["initial_module"].items()}
    deleted = set()
    for t in task["turns"][:turn_index]:
        for op in t["ops"]:
            o = op["op"]
            if o == "add_fn":
                fns[op["fn"]] = list(op["params"])
            elif o == "add_param":
                fns[op["fn"]].append(op["param"])
            elif o == "remove_param":
                fns[op["fn"]].remove(op["param"])
            elif o == "rename_param":
                fns[op["fn"]][fns[op["fn"]].index(op["old"])] = op["new"]
            elif o == "rename_fn":
                fns[op["new"]] = fns.pop(op["old"])
                deleted.discard(op["new"])
            elif o == "delete_fn":
                del fns[op["fn"]]
                deleted.add(op["fn"])
    return fns, sorted(deleted)


def reground(task, next_turn, carries_sentinel):
    """Deterministic re-grounding: rebuild the session from the external store.
    No LLM call, and -- the point of the experiment -- the agent never has to
    reproduce its own state OR the probe's ledger, because the harness supplies
    both. If the compaction penalty is caused by the agent's summary having to
    carry the probe's bookkeeping, it must vanish here."""
    module, deleted = true_module_at(task, next_turn - 1)
    rules = "Every turn I will give you" + BRIEFING.split("Every turn I will give you", 1)[1]
    if carries_sentinel:
        rules += "\n\n" + CANARY_INSTRUCTIONS[SENTINEL]
        rules += "\n" + _ledger_seed(task["turns"][next_turn - 2])
    body = render_module(module)
    if deleted:
        body += "\n\n# functions deleted earlier in this session: " + ", ".join(deleted)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME.format(module=body, rules=rules)},
        {"role": "assistant", "content": "Understood. Ready to continue."},
    ]


async def compact(cfg, messages, task, next_turn, carries_sentinel):
    """Run the compaction operator. Returns (new_messages, dump_text, usage)."""
    ask = messages + [{"role": "user", "content": DUMP_REQUEST}]
    dump, usage = await chat(cfg, ask, temperature=TEMPERATURE)
    module = extract_module(dump)
    if not module:                      # summary unparseable: keep the context
        return None, dump, usage

    rules = BRIEFING.split("Every turn I will give you", 1)[1]
    rules = "Every turn I will give you" + rules
    if carries_sentinel:
        rules += "\n\n" + CANARY_INSTRUCTIONS[SENTINEL]
        rules += "\n" + _ledger_seed(task["turns"][next_turn - 2])
    new_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME.format(module=render_module(module), rules=rules)},
        {"role": "assistant", "content": "Understood. Ready to continue."},
    ]
    return new_messages, dump, usage


async def run_arm(cfg, task, arm_name, arm, decide_reset):
    """Run one task under one arm for its FULL horizon (no early stop).

    decide_reset(turn_idx, state) -> bool, supplied by the policy layer.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    first = BRIEFING.format(module=render_module(task["initial_module"]))
    if arm["carries_sentinel"]:
        first += "\n\n" + CANARY_INSTRUCTIONS[SENTINEL]
    first += "\n\n" + build_turn_body(task, task["turns"][0], SENTINEL
                                      if arm["carries_sentinel"] else "baseline")
    memo = {}
    records, resets = [], []
    prompt_tok = completion_tok = 0
    turns_since_reset = 10**6
    first_hallu = None

    for i, turn in enumerate(task["turns"]):
        t = turn["turn"]
        # ---- reset decision is made BEFORE the turn is answered
        if i > 0:
            state = {"turn": t, "index": i, "records": records,
                     "turns_since_reset": turns_since_reset,
                     "n_resets": len(resets), "first_hallu": first_hallu}
            if decide_reset(state):
                if arm.get("operator", "compaction") == "reground":
                    new_msgs = reground(task, t, arm["carries_sentinel"])
                else:
                    new_msgs, dump, usage = await compact(
                        cfg, messages, task, t, arm["carries_sentinel"])
                    prompt_tok += usage["prompt_tokens"] or 0
                    completion_tok += usage["completion_tokens"] or 0
                if new_msgs is not None:
                    messages = new_msgs
                    turns_since_reset = 0
                    resets.append({"turn": t, "resume": new_msgs[1]["content"]})

        user_msg = (first if i == 0 else
                    build_turn_body(task, turn,
                                    SENTINEL if arm["carries_sentinel"] else "baseline"))
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})
        prompt_tok += usage["prompt_tokens"] or 0
        completion_tok += usage["completion_tokens"] or 0

        errors = check_hallucination(content, turn)
        hallu = len(errors) > 0
        if hallu and first_hallu is None:
            first_hallu = t
        canary_pass = None
        if arm["carries_sentinel"]:
            canary_pass = check_canary(SENTINEL, content, task, turn, memo)

        records.append({
            "turn": t, "assistant": content, "hallucination": hallu,
            "errors": [list(e) for e in errors], "canary_pass": canary_pass,
            "turns_since_reset": turns_since_reset,
            "prompt_tokens": usage["prompt_tokens"],
        })
        turns_since_reset += 1

    clean = sum(1 for r in records if not r["hallucination"])
    return {
        "task_id": task["task_id"], "arm": arm_name, "horizon": task["horizon"],
        "operator": arm.get("operator", "compaction"),
        "difficulty": task["difficulty"], "turns_run": len(records),
        "accuracy": clean / len(records),
        "clean_turns": clean,
        "n_resets": len(resets),
        "reset_turns": [r["turn"] for r in resets],
        "first_hallucination": first_hallu,
        "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
        "records": records, "complete": True,
    }
