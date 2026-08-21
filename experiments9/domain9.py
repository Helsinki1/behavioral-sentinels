"""The shardmath domain: messages, checkers, store, monitor and quiz for
sharded GSM8K sessions (lost_in_conversation `math`, 3 problems/session).

Protocol (deviations from the source paper stated in config9): one verbatim
shard per user turn; the assistant replies exactly `WAIT` until the turn
marked FINAL SHARD, then exactly `ANSWER: <number>`. Every turn is decidable
against the GSM8K `####` key: a premature ANSWER, a missing WAIT, or a
missing/wrong final ANSWER is a hallucination event.

The zero-carry monitor reads only message structure (no answer key): it
fires on malformed replies, answers before the final shard, and missing
final answers. It therefore overlaps the checker on protocol errors but is
blind to WRONG final answers -- the honest limit of trace-only observation
in this domain, reported as such.
"""
import re

from experiments.llm import chat
from experiments8.quiz8 import (QUIZ_PREAMBLE, _grade_int, _grade_sentence,
                                build_quiz_message, grade_quiz)

from .config9 import QUIZ_MAX_TOKENS, TEMPERATURE

SYSTEM_PROMPT = "You are a careful, concise math assistant."

NEW_PROBLEM_LINE = "--- NEW PROBLEM ---"
FINAL_LINE = "FINAL SHARD - answer now."

BRIEFING = f"""You are solving math word problems with me over a long session.
Each problem is revealed one piece (shard) at a time, one shard per message.

Every turn I will give you one shard. Rules:
- Track all shards of the current problem across the conversation. Do not use tools.
- A line reading {NEW_PROBLEM_LINE} means a new problem begins: discard the previous problem completely.
- Until the final shard arrives, reply with exactly one line: WAIT
- When my message contains the line "{FINAL_LINE}", compute the answer from ALL shards of the current problem and reply with exactly one line: ANSWER: <number>
- Never answer before the final shard. Do not restate shards or show your working."""

_ANSWER_RE = re.compile(r"ANSWER\s*:?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)
# substring, not word-bounded: models stutter "WAITWAIT", which is intent to
# wait (a decoding artifact), not a state-tracking failure
_WAIT_RE = re.compile(r"WAIT", re.I)


def _num(s):
    try:
        return float(str(s).replace(",", "").replace("$", ""))
    except ValueError:
        return None


# ------------------------------------------------------------ message build

def build_turn_body(task, turn, condition):
    lines = [f"Turn {turn['turn']}:"]
    if condition != "baseline":
        lines.append(f"Ticket: {turn['c3']['ticket']}")
    if turn["new_problem"]:
        lines.append(NEW_PROBLEM_LINE)
    lines.append(f"Shard: {turn['shard']}")
    if turn["final"]:
        lines.append(FINAL_LINE)
    return "\n".join(lines)


def build_first_user_message(task, condition):
    from experiments3.canaries3 import instruction_text
    msg = BRIEFING
    if condition != "baseline":
        msg += "\n\n" + instruction_text(condition, task)
    return msg + "\n\n" + build_turn_body(task, task["turns"][0], condition)


# ------------------------------------------------------------ hallucination

def check_hallucination(text, turn):
    ms = _ANSWER_RE.findall(text)
    if turn["final"]:
        if not ms:
            return [("missing_answer", turn["shard"][:40])]
        got, want = _num(ms[-1]), _num(turn["answer"])
        if got is None or want is None or abs(got - want) > 1e-6:
            return [("wrong_answer", f"want {turn['answer']} got {ms[-1]}")]
        return []
    errors = []
    if ms:
        errors.append(("premature_answer", ms[-1]))
    elif not _WAIT_RE.search(text):
        errors.append(("missing_wait", text[:40]))
    return errors


# ------------------------------------------------------------------- store

class ShardStore:
    """The external record: shards revealed so far for the current problem,
    and how many problems have been completed. Pure function of user-issued
    content."""

    def __init__(self, task):
        self.shards = []
        self.completed = 0

    def apply(self, turn):
        if turn["new_problem"]:
            self.shards = []
        self.shards.append(turn["shard"])
        if turn["final"]:
            self.completed += 1


def make_store(task):
    return ShardStore(task)


PREAMBLE = """This is a fresh session resuming an in-progress task. The
previous session ended after task turn {last_turn}; the next message will be
Turn {next_turn}. The task rules and the current state, read back from the
session's external record, follow. Do not perform any task work until Turn
{next_turn} arrives."""


def resume_reground(task, store, next_turn, condition):
    from experiments3.canaries3 import instruction_text
    from experiments8.harness8 import payload_log

    parts = [PREAMBLE.format(last_turn=next_turn - 1, next_turn=next_turn),
             BRIEFING]
    if condition != "baseline":
        parts.append(instruction_text(condition, task))
    parts.append(f"Problems fully answered so far: {store.completed}")
    nxt = task["turns"][next_turn - 1]
    if nxt["new_problem"]:
        state = "(none -- a NEW PROBLEM begins next turn)"
    else:
        state = "\n".join(f"- {s}" for s in store.shards)
    parts.append("Shards of the CURRENT problem so far:\n" + state)
    if condition != "baseline":
        log = payload_log(condition, task, next_turn)
        if log:
            parts.append("Standing-rule payload lines from the earlier "
                         "messages of this session, verbatim:\n"
                         + "\n".join(log))
    return "\n\n".join(parts)


# ----------------------------------------------------------------- monitor

class ShardMonitor:
    """Zero-carry: structural checks only, no answer key."""

    def check(self, user_text, reply):
        # standalone-line match only: the briefing QUOTES the marker, so a
        # substring test would mark turn 1 (and every post-reset resume) final
        is_final = any(l.strip() == FINAL_LINE
                       for l in user_text.splitlines())
        has_answer = bool(_ANSWER_RE.search(reply))
        has_wait = bool(_WAIT_RE.search(reply))
        fired = []
        if is_final:
            if not has_answer:
                fired.append("malformed:missing_final_answer")
        else:
            if has_answer:
                fired.append("premature:answer_before_final")
            elif not has_wait:
                fired.append("malformed:no_wait")
        return fired


def make_monitor():
    return ShardMonitor()


# -------------------------------------------------------------------- quiz

def quiz_questions(task, store, t):
    qs = []
    if store.shards:
        qs.append({"text": "What was the first shard of the CURRENT problem, "
                           "verbatim?",
                   "grade": _grade_sentence(store.shards[0]),
                   "expect": store.shards[0]})
    else:
        qs.append({"text": "unused", "grade": None, "expect": None})
    qs.append({"text": "How many shards have been revealed so far for the "
                       "current problem?",
               "grade": _grade_int(len(store.shards)),
               "expect": len(store.shards)})
    qs.append({"text": "How many problems have you fully answered so far in "
                       "this session?",
               "grade": _grade_int(store.completed),
               "expect": store.completed})
    return qs


async def ask_quiz(cfg, messages, task, store, t):
    questions = quiz_questions(task, store, t)
    fork = messages + [{"role": "user",
                        "content": build_quiz_message(questions, t)}]
    reply, usage = await chat(cfg, fork, max_tokens=QUIZ_MAX_TOKENS,
                              temperature=TEMPERATURE)
    correct, n_wrong, fail = grade_quiz(questions, reply)
    return ({"turn": t, "reply": reply, "correct": correct,
             "n_wrong": n_wrong, "fail": fail,
             "questions": [q["text"] for q in questions],
             "expect": [q["expect"] for q in questions]}, usage)
