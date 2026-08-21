"""The experiment-9 harness: exp 8's design (reground resets, carried probes
for the active arms, frozen-state quiz for the passive-behavioural arm) on
the shardmath domain, model-agnostic.
"""
from experiments.llm import chat
from experiments3.canaries3 import score_canary

from .config9 import QUIZ_EVERY, QUIZ_FIRST, TEMPERATURE
from .domain9 import (SYSTEM_PROMPT, ask_quiz, build_first_user_message,
                      build_turn_body, check_hallucination, make_monitor,
                      make_store, resume_reground)

ACK = "Understood. Ready to continue."


def reset_messages(task, store, next_turn, condition):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": resume_reground(task, store, next_turn, condition)},
            {"role": "assistant", "content": ACK}]


def quiz_due(t, horizon):
    return t >= QUIZ_FIRST and (t - QUIZ_FIRST) % QUIZ_EVERY == 0 and t < horizon


async def run_arm(cfg, task, arm_name, condition, decide_reset, quiz=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    store = make_store(task)
    monitor = make_monitor()
    records, resets, quizzes = [], [], []
    prompt_tok = completion_tok = 0
    quiz_prompt_tok = quiz_completion_tok = 0
    turns_since_reset = 10**6
    ctx_baseline = None
    last_prompt_tokens = 0
    first_hallu = None

    for i, turn in enumerate(task["turns"]):
        t = turn["turn"]
        if i > 0:
            state = {"turn": t, "index": i, "records": records,
                     "turns_since_reset": turns_since_reset,
                     "n_resets": len(resets), "first_hallu": first_hallu,
                     "last_prompt_tokens": last_prompt_tokens,
                     "ctx_baseline": ctx_baseline}
            if decide_reset(state):
                messages = reset_messages(task, store, t, condition)
                turns_since_reset = 0
                ctx_baseline = None
                resets.append({"turn": t, "resume": messages[1]["content"]})

        user_msg = (build_first_user_message(task, condition) if i == 0
                    else build_turn_body(task, turn, condition))
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})
        store.apply(turn)
        prompt_tok += usage["prompt_tokens"] or 0
        completion_tok += usage["completion_tokens"] or 0
        last_prompt_tokens = usage["prompt_tokens"] or 0
        if ctx_baseline is None:
            ctx_baseline = last_prompt_tokens

        errors = check_hallucination(content, turn)
        hallu = len(errors) > 0
        if hallu and first_hallu is None:
            first_hallu = t

        probe = None
        if condition != "baseline":
            probe = score_canary(condition, content, task, turn)
        zc_fired = monitor.check(user_msg, content)

        quiz_fail = None
        if quiz and quiz_due(t, task["horizon"]):
            qrec, qusage = await ask_quiz(cfg, messages, task, store, t)
            quiz_prompt_tok += qusage["prompt_tokens"] or 0
            quiz_completion_tok += qusage["completion_tokens"] or 0
            quizzes.append(qrec)
            quiz_fail = qrec["fail"]

        records.append({
            "turn": t, "assistant": content, "hallucination": hallu,
            "errors": [list(e) for e in errors],
            "probe_score": None if probe is None else probe.get("score"),
            "probe_fail": (probe is not None and probe.get("score") is not None
                           and probe["score"] < 1.0),
            "zerocarry_fired": zc_fired,
            "judge_yes": None,
            "quiz_fail": quiz_fail,
            "quiz_n_wrong": quizzes[-1]["n_wrong"] if quiz_fail is not None else None,
            "turns_since_reset": turns_since_reset,
            "prompt_tokens": usage["prompt_tokens"],
        })
        turns_since_reset += 1

    clean = sum(1 for r in records if not r["hallucination"])
    return {
        "task_id": task["task_id"], "domain": "shardmath", "arm": arm_name,
        "condition": condition, "operator": "reground",
        "horizon": task["horizon"], "difficulty": task["difficulty"],
        "turns_run": len(records),
        "accuracy": clean / len(records), "clean_turns": clean,
        "n_resets": len(resets), "reset_turns": [r["turn"] for r in resets],
        "resets": resets, "compactions": [], "quizzes": quizzes,
        "first_hallucination": first_hallu,
        "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
        "quiz_prompt_tokens": quiz_prompt_tok,
        "quiz_completion_tokens": quiz_completion_tok,
        "records": records, "complete": True,
    }
