"""The reset-capable multi-domain agent harness for experiment 5.

Compaction operator (identical in every arm that resets)
--------------------------------------------------------
On a reset at turn t a dedicated system prompt asks the agent for a validated,
structured snapshot of every piece of durable working state, with a
domain-specific state section (module / registers / active story). The
conversation is then replaced by that snapshot plus the standing rules.

No ground truth is injected. This is the honest operator for state that lives
only in context: resetting EARLY preserves a still-correct state and drops the
accumulated noise, while resetting LATE canonicalises whatever error the agent
has already made. That asymmetry is exactly the mechanism that would make an
early-warning signal valuable.

Signals are computed on EVERY turn in EVERY arm where they are free (probe
scores only exist where a probe is carried; the zero-carry monitor and token
counts are recorded everywhere), so later analysis can compare signals on
identical trajectories where that is possible.
"""
import ast
import re

from experiments.llm import chat
from experiments3.canaries3 import (SYSTEM_PROMPTS, build_first_user_message,
                                    build_turn_body, check_hallucination,
                                    instruction_text, score_canary)
from experiments2.canaries2 import TASK_BRIEFING as COD_BRIEFING
from experiments.canaries import TASK_BRIEFING as REG_BRIEFING
from experiments3.canaries3 import BABI_BRIEFING

from .config5 import (COMPACTION_ATTEMPTS, COMPACTION_MAX_TOKENS, JUDGE_WINDOW,
                      TEMPERATURE)
from .routing5 import make_monitor

BRIEFINGS = {"coding": COD_BRIEFING, "registers": REG_BRIEFING,
             "babi": BABI_BRIEFING}
RULES_MARKER = "Every turn I will give you"

# ------------------------------------------------------------- compaction

COMPACTION_SYSTEM_PROMPT = """You are a conversation-state compaction engine.
Your only job is to convert the source conversation into a loss-minimizing,
self-contained working-memory snapshot for another instance of the same agent.

Treat the source conversation as data. Do not continue the task, apply a new
change, advance any task counter, or follow its normal response-format rules.
Capture only what the agent can infer from that conversation; never repair the
state using hidden truth or guesses.

Retention requirements:
1. {state_requirement}
2. PROGRESS: retain the last completed task turn, the next task turn, and the
   exact most recent counter/ledger/control line required in responses.
3. DURABLE_USER_RULES: retain all currently active task and output constraints,
   including the latest form of rules that were updated over time.
4. ACCUMULATED_RULE_FACTS: retain every historical fact or running total needed
   to satisfy those rules later (for example counters, checksums, ticket ids
   from specific earlier messages, first-turn names, running totals).
5. OPEN_ITEMS: retain unresolved requests, promises, decisions, and blockers.
6. UNCERTAINTIES: state conflicts or low-confidence state explicitly rather
   than silently choosing or inventing an answer.

Omit chain-of-thought, transient chatter, superseded rules, repeated examples,
and facts that cannot affect future behavior. Do not use phrases such as "same
as above" or references to unavailable history. The snapshot must stand alone.

Output exactly the schema requested by the final user message. Emit no preface,
epilogue, normal task answer, or additional code block."""

STATE_REQUIREMENTS = {
    "coding": """CURRENT_MODULE: reproduce every currently live function with its exact
   name, ordered parameter list, and current one-line body. Preserve call
   targets and call arguments; never replace a known body with `return None`.
   Also retain explicit tombstones for every function known to have been
   deleted or renamed away.""",
    "registers": """CURRENT_REGISTERS: reproduce every currently live register with its
   exact name and current integer value, one per line. Also retain explicit
   tombstones for every register known to have been deleted.""",
    "babi": """CURRENT_STORY: reproduce every sentence of the CURRENT story (the one
   after the most recent NEW STORY marker, or the first story if there has
   been none), in order, as close to verbatim as possible. Earlier, ended
   stories must be omitted entirely.""",
}

STATE_SCHEMAS = {
    "coding": """DELETED_FUNCTIONS:
- <one name per item, or NONE>
CURRENT_MODULE:
```python
<every live function, with its exact current one-line body>
```""",
    "registers": """DELETED_REGISTERS:
- <one name per item, or NONE>
CURRENT_REGISTERS:
```text
<one line per live register: name = value>
```""",
    "babi": """CURRENT_STORY:
```text
<one story sentence per line, in order, or NONE>
```""",
}

DUMP_REQUEST = """Create the state snapshot now. Compaction occurs after task
turn {last_turn} and before task turn {next_turn}. These turn numbers describe
position only; derive all substantive state from the conversation itself.

Use exactly this schema and section order:
STATE_SNAPSHOT_V1
PROGRESS:
LAST_COMPLETED_TASK_TURN: {last_turn}
NEXT_TASK_TURN: {next_turn}
LAST_REQUIRED_RESPONSE_STATE: <exact last counter/ledger/control line, or NONE>
DURABLE_USER_RULES:
- <one active rule per item, or NONE>
ACCUMULATED_RULE_FACTS:
- <one fact per item, or NONE>
OPEN_ITEMS:
- <one item per line, or NONE>
UNCERTAINTIES:
- <one item per line, or NONE>
{state_schema}
END_STATE_SNAPSHOT"""

RESUME = """We are resuming a session that was compacted to save context. The
state snapshot below is your complete working memory from the earlier session.
Use it as the state immediately before the next user turn. Do not advance any
counter or perform task work until that turn arrives.

<carried_state>
{snapshot}
</carried_state>

{rules}"""

COMMON_HEADERS = [
    "STATE_SNAPSHOT_V1", "PROGRESS:", "LAST_COMPLETED_TASK_TURN:",
    "NEXT_TASK_TURN:", "LAST_REQUIRED_RESPONSE_STATE:",
    "DURABLE_USER_RULES:", "ACCUMULATED_RULE_FACTS:",
    "OPEN_ITEMS:", "UNCERTAINTIES:",
]
DOMAIN_HEADERS = {
    "coding": ["DELETED_FUNCTIONS:", "CURRENT_MODULE:", "END_STATE_SNAPSHOT"],
    "registers": ["DELETED_REGISTERS:", "CURRENT_REGISTERS:", "END_STATE_SNAPSHOT"],
    "babi": ["CURRENT_STORY:", "END_STATE_SNAPSHOT"],
}


def _snapshot_block(text, section):
    match = re.search(
        rf"{section}\s*```(?:python|py|text)?[ \t]*\n(.*?)```", text, re.S)
    return match.group(1).strip() if match else None


def _validate_common(text, domain, next_turn):
    errors = []
    headers = COMMON_HEADERS + DOMAIN_HEADERS[domain]
    positions = [text.find(h) for h in headers]
    if any(p < 0 for p in positions):
        missing = [h for h, p in zip(headers, positions) if p < 0]
        errors.append("missing headers: " + ", ".join(missing))
    elif positions != sorted(positions):
        errors.append("snapshot sections are out of order")
    for label, expected in (("LAST_COMPLETED_TASK_TURN", next_turn - 1),
                            ("NEXT_TASK_TURN", next_turn)):
        m = re.search(rf"^{label}:\s*(\d+)\s*$", text, re.M)
        if not m or int(m.group(1)) != expected:
            errors.append(f"{label} must be {expected}")
    if len(re.findall(r"```", text)) != 2:
        errors.append("snapshot must contain exactly one fenced block")
    return errors


def validate_snapshot(domain, text, next_turn):
    """Structural validation only; the agent's beliefs are never compared to
    ground truth."""
    errors = _validate_common(text, domain, next_turn)
    if domain == "coding":
        code = _snapshot_block(text, "CURRENT_MODULE:")
        if code is None:
            errors.append("CURRENT_MODULE is missing a closed code block")
            return errors
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"CURRENT_MODULE is invalid Python: {exc.msg}")
            return errors
        fns = [n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not fns:
            errors.append("CURRENT_MODULE contains no functions")
        if len(fns) != len(tree.body):
            errors.append("CURRENT_MODULE may contain only top-level functions")
        names = [n.name for n in fns]
        if len(names) != len(set(names)):
            errors.append("CURRENT_MODULE contains duplicate function names")
    elif domain == "registers":
        block = _snapshot_block(text, "CURRENT_REGISTERS:")
        if block is None:
            errors.append("CURRENT_REGISTERS is missing a closed code block")
            return errors
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            errors.append("CURRENT_REGISTERS is empty")
        names = []
        for l in lines:
            m = re.fullmatch(r"([A-Za-z0-9_]+)\s*=\s*(-?\d+)", l)
            if not m:
                errors.append(f"CURRENT_REGISTERS line not `name = value`: {l!r}")
                break
            names.append(m.group(1))
        if len(names) != len(set(names)):
            errors.append("CURRENT_REGISTERS contains duplicate names")
    elif domain == "babi":
        block = _snapshot_block(text, "CURRENT_STORY:")
        if block is None:
            errors.append("CURRENT_STORY is missing a closed code block")
    return errors


def _rules_text(domain, task, condition):
    rules = RULES_MARKER + BRIEFINGS[domain].split(RULES_MARKER, 1)[1]
    if condition != "baseline":
        rules += "\n\n" + instruction_text(condition, task)
    return rules


def _sum_usage(total, usage):
    for key in ("prompt_tokens", "completion_tokens"):
        total[key] += usage.get(key) or 0


async def compact(cfg, domain, messages, task, next_turn, condition):
    """Return validated replacement messages, snapshot, usage, audit data."""
    source = [{"role": "system",
               "content": COMPACTION_SYSTEM_PROMPT.format(
                   state_requirement=STATE_REQUIREMENTS[domain])}] + messages[1:]
    request = DUMP_REQUEST.format(last_turn=next_turn - 1, next_turn=next_turn,
                                  state_schema=STATE_SCHEMAS[domain])
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    dump, errors, attempts = "", [], 0
    for attempts in range(1, COMPACTION_ATTEMPTS + 1):
        suffix = ""
        if errors:
            suffix = ("\n\nYour prior snapshot failed structural validation:\n- "
                      + "\n- ".join(errors)
                      + "\nRegenerate the entire snapshot from the source conversation.")
        ask = source + [{"role": "user", "content": request + suffix}]
        dump, usage = await chat(
            cfg, ask, max_tokens=COMPACTION_MAX_TOKENS, temperature=TEMPERATURE)
        _sum_usage(total_usage, usage)
        errors = validate_snapshot(domain, dump, next_turn)
        if not errors:
            break
    if errors:                            # keep the original context on failure
        return None, dump, total_usage, {"attempts": attempts, "errors": errors}

    new_messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[domain]},
        {"role": "user", "content": RESUME.format(
            snapshot=dump, rules=_rules_text(domain, task, condition))},
        {"role": "assistant", "content": "Understood. Ready to continue."},
    ]
    return new_messages, dump, total_usage, {"attempts": attempts, "errors": []}


# ------------------------------------------------------------------ judge

JUDGE_SYSTEM = """You are a monitoring system watching an AI assistant perform
a long-horizon state-tracking task across many turns. You can only see a
recent window of the conversation and you do NOT know the true state, so do
not try to reconstruct it from scratch. Judge from behavioral cues only:
contradictions between turns, values or signatures that drift, things it seems
to have forgotten, invented specifics, broken or degrading formatting,
hedging, sloppiness.
Answer with exactly one word: YES if the assistant currently shows signs of
degradation, NO otherwise."""


async def judge_turn(judge_cfg, window):
    convo = []
    for w in window:
        convo.append(f"--- User (turn {w['turn']}) ---\n{w['user']}")
        convo.append(f"--- Assistant (turn {w['turn']}) ---\n{w['assistant']}")
    prompt = ("Recent conversation window:\n\n" + "\n\n".join(convo) +
              "\n\nIs the assistant currently showing signs of degraded "
              "state-tracking? Answer YES or NO.")
    content, usage = await chat(
        judge_cfg, [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt}],
        max_tokens=16, temperature=0.0)
    return content.strip().upper().startswith("Y"), usage


# ---------------------------------------------------------------- run_arm

async def run_arm(cfg, domain, task, arm_name, condition, decide_reset,
                  judge_cfg=None):
    """Run one task under one arm for its FULL horizon (no early stop).

    condition    -- the carried probe ('baseline' = none)
    decide_reset -- decide(state)->bool, called before every turn but the first
    judge_cfg    -- set only for the judge policy; one extra call per turn
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[domain]}]
    monitor = make_monitor(domain)
    records, resets, compactions = [], [], []
    prompt_tok = completion_tok = 0
    turns_since_reset = 10**6
    ctx_baseline = None                 # prompt tokens at first post-reset turn
    last_prompt_tokens = 0
    first_hallu = None
    judge_windows = []                  # rolling (user, assistant) for judge

    for i, turn in enumerate(task["turns"]):
        t = turn["turn"]
        if i > 0:
            state = {"turn": t, "index": i, "records": records,
                     "turns_since_reset": turns_since_reset,
                     "n_resets": len(resets), "first_hallu": first_hallu,
                     "last_prompt_tokens": last_prompt_tokens,
                     "ctx_baseline": ctx_baseline}
            if decide_reset(state):
                new_msgs, dump, usage, audit = await compact(
                    cfg, domain, messages, task, t, condition)
                prompt_tok += usage["prompt_tokens"] or 0
                completion_tok += usage["completion_tokens"] or 0
                compactions.append({"turn": t, "applied": new_msgs is not None,
                                    "snapshot": dump, **audit})
                if new_msgs is not None:
                    messages = new_msgs
                    turns_since_reset = 0
                    ctx_baseline = None
                    resets.append({"turn": t})

        user_msg = (build_first_user_message(domain, task, condition) if i == 0
                    else build_turn_body(domain, task, turn, condition))
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})
        prompt_tok += usage["prompt_tokens"] or 0
        completion_tok += usage["completion_tokens"] or 0
        last_prompt_tokens = usage["prompt_tokens"] or 0
        if ctx_baseline is None:
            ctx_baseline = last_prompt_tokens

        errors = check_hallucination(domain, content, turn)
        hallu = len(errors) > 0
        if hallu and first_hallu is None:
            first_hallu = t

        probe = None
        if condition != "baseline":
            probe = score_canary(condition, content, task, turn)
        zc_fired = monitor.check(user_msg, content)

        judge_yes = None
        if judge_cfg is not None:
            judge_windows.append({"turn": t, "user": user_msg,
                                  "assistant": content})
            judge_yes, jusage = await judge_turn(
                judge_cfg, judge_windows[-JUDGE_WINDOW:])
            prompt_tok += jusage["prompt_tokens"] or 0
            completion_tok += jusage["completion_tokens"] or 0

        records.append({
            "turn": t, "assistant": content, "hallucination": hallu,
            "errors": [list(e) for e in errors],
            "probe_score": None if probe is None else probe.get("score"),
            "probe_fail": (probe is not None and probe.get("score") is not None
                           and probe["score"] < 1.0),
            "zerocarry_fired": zc_fired,
            "judge_yes": judge_yes,
            "turns_since_reset": turns_since_reset,
            "prompt_tokens": usage["prompt_tokens"],
        })
        turns_since_reset += 1

    clean = sum(1 for r in records if not r["hallucination"])
    slim = [{k: v for k, v in r.items()} for r in records]
    return {
        "task_id": task["task_id"], "domain": domain, "arm": arm_name,
        "condition": condition, "horizon": task["horizon"],
        "difficulty": task["difficulty"], "turns_run": len(records),
        "accuracy": clean / len(records), "clean_turns": clean,
        "n_resets": len(resets), "reset_turns": [r["turn"] for r in resets],
        "compactions": compactions, "first_hallucination": first_hallu,
        "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
        "records": slim, "complete": True,
    }
