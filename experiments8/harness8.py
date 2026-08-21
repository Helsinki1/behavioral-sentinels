"""The experiment-8 harness: exp 6's re-grounding harness extended with
(a) carried probes -- the ACTIVE arms re-instate exp 5's condition machinery
    under the R1 reground operator, and
(b) the frozen-state quiz -- the PASSIVE-BEHAVIOURAL arm's forked spot-check,
    asked every QUIZ_EVERY turns and never written back into the agent's
    messages.

Reground resume for a carried arm: the probe's standing rule is part of the
session's user-issued content, so a faithful fresh session re-reads it -- the
resume appends the rule (instruction_text) plus a verbatim log of every
probe-payload line issued so far (tickets, KEYs, events, shadow renames,
ledger updates). Nothing is included that the prior user messages did not
contain; the agent's probe ANSWERS are never replayed.

Signals are recorded on every turn wherever they are free (probe scores only
exist where a probe is carried; the zero-carry monitor is recorded
everywhere; quiz results only exist where the quiz runs), mirroring exp 5 so
the prediction layer can compare signals on identical trajectories.
"""
from experiments.llm import chat
from experiments3.canaries3 import (SYSTEM_PROMPTS, _payload_lines,
                                    build_first_user_message, build_turn_body,
                                    check_hallucination, instruction_text,
                                    score_canary)
from experiments5.harness5 import judge_turn
from experiments5.routing5 import make_monitor
from experiments6.store6 import make_store, resume_reground

from .config8 import JUDGE_WINDOW, QUIZ_EVERY, QUIZ_FIRST, TEMPERATURE
from .quiz8 import ask_quiz

ACK = "Understood. Ready to continue."


def payload_log(condition, task, next_turn):
    """Verbatim probe-payload lines from every prior turn -- user-issued
    content a fresh session would find in the transcript."""
    entries = []
    for turn in task["turns"]:
        if turn["turn"] >= next_turn:
            break
        lines = _payload_lines(condition, turn)
        if lines:
            entries.append(f"Turn {turn['turn']}: " + " | ".join(lines))
    return entries


def reset_messages(domain, task, store, next_turn, condition):
    resume = resume_reground(domain, task, store, next_turn)
    if condition != "baseline":
        resume += "\n\n" + instruction_text(condition, task)
        log = payload_log(condition, task, next_turn)
        if log:
            resume += ("\n\nStanding-rule payload lines from the earlier "
                       "messages of this session, verbatim:\n"
                       + "\n".join(log))
    return [{"role": "system", "content": SYSTEM_PROMPTS[domain]},
            {"role": "user", "content": resume},
            {"role": "assistant", "content": ACK}]


def quiz_due(t, horizon):
    return t >= QUIZ_FIRST and (t - QUIZ_FIRST) % QUIZ_EVERY == 0 and t < horizon


async def run_arm(cfg, domain, task, arm_name, condition, decide_reset,
                  judge_cfg=None, quiz=False):
    """Run one task under one arm for its FULL horizon (no early stop)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[domain]}]
    store = make_store(domain, task)
    monitor = make_monitor(domain)
    records, resets, quizzes = [], [], []
    prompt_tok = completion_tok = 0
    quiz_prompt_tok = quiz_completion_tok = 0
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
                messages = reset_messages(domain, task, store, t, condition)
                turns_since_reset = 0
                ctx_baseline = None
                resets.append({"turn": t, "resume": messages[1]["content"]})

        user_msg = (build_first_user_message(domain, task, condition) if i == 0
                    else build_turn_body(domain, task, turn, condition))
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})
        store.apply(turn)
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

        quiz_fail = None
        if quiz and quiz_due(t, task["horizon"]):
            qrec, qusage = await ask_quiz(cfg, messages, domain, task, store, t)
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
            "judge_yes": judge_yes,
            "quiz_fail": quiz_fail,
            "quiz_n_wrong": quizzes[-1]["n_wrong"] if quiz_fail is not None else None,
            "turns_since_reset": turns_since_reset,
            "prompt_tokens": usage["prompt_tokens"],
        })
        turns_since_reset += 1

    clean = sum(1 for r in records if not r["hallucination"])
    return {
        "task_id": task["task_id"], "domain": domain, "arm": arm_name,
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
