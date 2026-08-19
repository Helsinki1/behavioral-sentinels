"""The reset-capable harness for experiment 6.

Identical to experiments5.harness5 except for the reset operator: instead of
asking the agent for a compaction snapshot (an LLM call whose errors become
canon), a reset deterministically rebuilds the conversation from the external
store (R1 "reground") or from the verbatim user-message log (R2 "replay").
No LLM call happens at reset time; the only cost of a reset is the re-read
tokens in subsequent prompts.

No probes are carried in any arm (condition is always baseline). The
zero-carry monitor and the judge are recorded exactly as in exp 5 so the
signals stay comparable across experiments.
"""
from experiments.llm import chat
from experiments3.canaries3 import (SYSTEM_PROMPTS, build_first_user_message,
                                    build_turn_body, check_hallucination)
from experiments5.harness5 import judge_turn
from experiments5.routing5 import make_monitor

from .config6 import JUDGE_WINDOW, TEMPERATURE
from .store6 import make_store, resume_reground, resume_replay

ACK = "Understood. Ready to continue."


def reset_messages(operator, domain, task, store, next_turn):
    if operator == "reground":
        resume = resume_reground(domain, task, store, next_turn)
    elif operator == "replay":
        resume = resume_replay(domain, task, next_turn)
    else:
        raise ValueError(operator)
    return [{"role": "system", "content": SYSTEM_PROMPTS[domain]},
            {"role": "user", "content": resume},
            {"role": "assistant", "content": ACK}]


async def run_arm(cfg, domain, task, arm_name, operator, decide_reset,
                  judge_cfg=None):
    """Run one task under one arm for its FULL horizon (no early stop)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[domain]}]
    store = make_store(domain, task)
    monitor = make_monitor(domain)
    records, resets = [], []
    prompt_tok = completion_tok = 0
    turns_since_reset = 10**6
    ctx_baseline = None
    last_prompt_tokens = 0
    first_hallu = None
    judge_windows = []

    for i, turn in enumerate(task["turns"]):
        t = turn["turn"]
        if i > 0:
            state = {"turn": t, "index": i, "records": records,
                     "turns_since_reset": turns_since_reset,
                     "n_resets": len(resets), "first_hallu": first_hallu,
                     "last_prompt_tokens": last_prompt_tokens,
                     "ctx_baseline": ctx_baseline}
            if decide_reset(state):
                messages = reset_messages(operator, domain, task, store, t)
                turns_since_reset = 0
                ctx_baseline = None
                # keep the full resume text: it is the audit trail proving
                # what a reset injected (and nothing else)
                resets.append({"turn": t, "resume": messages[1]["content"]})

        user_msg = (build_first_user_message(domain, task, "baseline") if i == 0
                    else build_turn_body(domain, task, turn, "baseline"))
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})
        store.apply(turn)          # the "file system" applies this turn's edits
        prompt_tok += usage["prompt_tokens"] or 0
        completion_tok += usage["completion_tokens"] or 0
        last_prompt_tokens = usage["prompt_tokens"] or 0
        if ctx_baseline is None:
            ctx_baseline = last_prompt_tokens

        errors = check_hallucination(domain, content, turn)
        hallu = len(errors) > 0
        if hallu and first_hallu is None:
            first_hallu = t

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
            "probe_score": None, "probe_fail": False,
            "zerocarry_fired": zc_fired,
            "judge_yes": judge_yes,
            "turns_since_reset": turns_since_reset,
            "prompt_tokens": usage["prompt_tokens"],
        })
        turns_since_reset += 1

    clean = sum(1 for r in records if not r["hallucination"])
    return {
        "task_id": task["task_id"], "domain": domain, "arm": arm_name,
        "condition": "baseline", "operator": operator,
        "horizon": task["horizon"], "difficulty": task["difficulty"],
        "turns_run": len(records),
        "accuracy": clean / len(records), "clean_turns": clean,
        "n_resets": len(resets), "reset_turns": [r["turn"] for r in resets],
        "resets": resets, "compactions": [],
        "first_hallucination": first_hallu,
        "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
        "records": records, "complete": True,
    }
