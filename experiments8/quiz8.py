"""The frozen-state quiz: experiment 8's passive-behavioural observation.

At a checkpoint the harness copies the live conversation, appends ONE user
message containing three questions about the current task state, sends the
fork to the model, grades the reply deterministically against the store6
reducer (store == generator truth, proven on the whole pool), and DISCARDS
the fork. Nothing enters agent state or conversation history; the audit copy
lives only in the trajectory file.

The three questions per domain (chosen deterministically per (task, turn)):

  registers  Q1 current value of one live register
             Q2 which registers have been deleted so far (NONE if none)
             Q3 how many registers are currently live
  coding     Q1 exact parameter list of one live function
             Q2 which functions have been deleted so far (NONE if none)
             Q3 how many functions are currently live
  babi       Q1 the first sentence of the current story, verbatim
             Q2 how many stories there have been so far (including current)
             Q3 how many sentences the current story has so far

Grading is intentionally shaped like the task's own checkers: regex over a
required `A<n>:` line, lenient to markdown decoration, strict on content.
A question whose ground truth is undefined this checkpoint (e.g. no live
register) grades as None and is excluded from the fail count.
"""
import random
import re

from experiments.llm import chat

from .config8 import QUIZ_FAIL_MIN, QUIZ_MAX_TOKENS, TEMPERATURE

QUIZ_MARKER = "SIDE-CHECK (ephemeral)"

QUIZ_PREAMBLE = (
    QUIZ_MARKER + ": this message is a monitoring spot-check, not a task "
    "turn. Do not advance any counter or apply any change. Answer the three "
    "questions below from the current state of the session, each on its own "
    "line, exactly in the form `A1: <answer>`, `A2: <answer>`, `A3: <answer>`. "
    "No other text.")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _answer_line(reply, n):
    m = re.findall(rf"A{n}\s*[:\-]\s*(.+)", reply or "", re.I)
    return m[-1].strip() if m else None


def _first_int(s):
    m = re.search(r"-?\d+", s or "")
    return int(m.group()) if m else None


def _idents(s):
    return [w.lower() for w in re.findall(r"[A-Za-z_]\w*", s or "")]


def _grade_int(expect):
    return lambda ans: ans is not None and _first_int(ans) == expect


def _grade_name_set(expect_set):
    none_words = {"none", "no", "nothing", "nil", "na", "n", "a"}
    def g(ans):
        if ans is None:
            return False
        got = set(_idents(ans)) - none_words
        return got == expect_set
    return g


def _grade_params(fn, expect_params):
    want = [p.lower() for p in expect_params]
    def g(ans):
        if ans is None:
            return False
        got = [w for w in _idents(ans) if w != fn.lower() and w != "def"]
        return got == want
    return g


def _grade_sentence(expect):
    want = _norm(expect)
    return lambda ans: ans is not None and want in _norm(ans)


def quiz_questions(domain, task, store, t):
    """Three {text, grade} questions from the store's state after turn t.
    Deterministic: the picked item depends only on (task_id, t)."""
    rng = random.Random((task["task_id"] << 8) ^ t)
    qs = []
    if domain == "registers":
        live = sorted(store.regs)
        if live:
            k = live[rng.randrange(len(live))]
            qs.append({"text": f"What is the current value of register {k}?",
                       "grade": _grade_int(store.regs[k]), "expect": store.regs[k]})
        else:
            qs.append({"text": "unused", "grade": None, "expect": None})
        qs.append({"text": "Which registers have been deleted so far in this "
                           "session? Answer NONE if none.",
                   "grade": _grade_name_set(set(n.lower() for n in store.deleted)),
                   "expect": sorted(store.deleted)})
        qs.append({"text": "How many registers are currently live?",
                   "grade": _grade_int(len(store.regs)), "expect": len(store.regs)})
    elif domain == "coding":
        live = sorted(store.fns)
        if live:
            fn = live[rng.randrange(len(live))]
            qs.append({"text": f"What is the exact current parameter list of "
                               f"function {fn}, in order?",
                       "grade": _grade_params(fn, store.fns[fn]),
                       "expect": {fn: list(store.fns[fn])}})
        else:
            qs.append({"text": "unused", "grade": None, "expect": None})
        qs.append({"text": "Which functions have been deleted so far in this "
                           "session? Answer NONE if none.",
                   "grade": _grade_name_set(set(n.lower() for n in store.deleted)),
                   "expect": sorted(store.deleted)})
        qs.append({"text": "How many functions are currently live in the module?",
                   "grade": _grade_int(len(store.fns)), "expect": len(store.fns)})
    elif domain == "babi":
        story = store.story
        if story:
            qs.append({"text": "What was the first sentence of the CURRENT "
                               "story, verbatim?",
                       "grade": _grade_sentence(story[0]), "expect": story[0]})
        else:
            qs.append({"text": "unused", "grade": None, "expect": None})
        n_stories = sum(1 for turn in task["turns"][:t]
                        if turn["new_story"] or turn["turn"] == 1)
        qs.append({"text": "How many stories have there been so far in this "
                           "session, counting the current one?",
                   "grade": _grade_int(n_stories), "expect": n_stories})
        qs.append({"text": "How many sentences does the current story have "
                           "so far?",
                   "grade": _grade_int(len(story)), "expect": len(story)})
    else:
        raise ValueError(domain)
    return qs


def build_quiz_message(questions, t):
    lines = [QUIZ_PREAMBLE, f"(spot-check after task turn {t})", ""]
    lines += [f"Q{i}: {q['text']}" for i, q in enumerate(questions, 1)
              if q["grade"] is not None]
    return "\n".join(lines)


def grade_quiz(questions, reply):
    """Per-question verdicts (True/False/None-not-applicable) and checkpoint
    fail flag."""
    correct = []
    for i, q in enumerate(questions, 1):
        if q["grade"] is None:
            correct.append(None)
        else:
            correct.append(bool(q["grade"](_answer_line(reply, i))))
    n_wrong = sum(1 for c in correct if c is False)
    return correct, n_wrong, n_wrong >= QUIZ_FAIL_MIN


async def ask_quiz(cfg, messages, domain, task, store, t):
    """Fork the frozen conversation, ask, grade, discard. Returns the audit
    record and the fork's token usage."""
    questions = quiz_questions(domain, task, store, t)
    fork = messages + [{"role": "user", "content": build_quiz_message(questions, t)}]
    reply, usage = await chat(cfg, fork, max_tokens=QUIZ_MAX_TOKENS,
                              temperature=TEMPERATURE)
    correct, n_wrong, fail = grade_quiz(questions, reply)
    record = {
        "turn": t, "reply": reply, "correct": correct, "n_wrong": n_wrong,
        "fail": fail,
        "questions": [q["text"] for q in questions],
        "expect": [q["expect"] for q in questions],
    }
    return record, usage
