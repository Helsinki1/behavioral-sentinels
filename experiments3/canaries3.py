"""Experiment-3 canaries: axis-isolating probes shared across three task sets.

Every canary is NON-COPYABLE by construction (experiment 2's finding is the
entry requirement here): the correct output can never be lifted verbatim from
the model's own previous reply -- it must be recomputed, recalled over a
distance, or abstained on.

Checkers return a SCORE in [0,1] (1 = perfect) or None when the canary is not
applicable this turn (confab_trap without a query, sparse_recall without a
probe).  A canary FAILURE is score < 1.0 on an applicable turn.
"""
import re

from experiments.canaries import TASK_BRIEFING as REG_BRIEFING
from experiments.canaries import check_hallucination as _hallu_registers
from experiments.tasks import render_op
from experiments2.canaries2 import (TASK_BRIEFING as COD_BRIEFING,
                                    check_hallucination as _hallu_coding,
                                    render_module)

from .config3 import CHECKSUM_MOD, ENSEMBLE_MEMBERS, LAGS, STAIR_PERIOD

SYSTEM_PROMPTS = {
    "coding": "You are a careful, concise coding assistant.",
    "registers": "You are a careful, concise assistant.",
    "babi": "You are a careful, concise assistant.",
}

BABI_BRIEFING = """You are following short stories with me over a long session.

Every turn I will give you a few new story sentences and then ask one question.
Rules:
- Track the story state across the entire conversation. Do not use tools.
- A line reading --- NEW STORY --- means the previous story has ended: discard it completely and start fresh.
- Answer using only what the story sentences say.
- For each question, output exactly one line: ANSWER: <one word>
- Do not restate the story. Do not show your working. Be concise."""

STATIC_TRAILER = "Reviewed-by: Dr. Vance KX-4471-TANGO"

NEW_STORY_LINE = "--- NEW STORY ---"


# ------------------------------------------------------------ instructions

def _instr_lag_span(task):
    return ("every message of mine includes a line `Ticket: ENG-####`. Every single "
            "response must include one line `ECHO: <a> <b> <c>` where <a> is the ticket "
            "from my previous message (1 message back), <b> the ticket from 3 messages "
            "back, and <c> the ticket from 6 messages back -- counting backwards from "
            "the message you are replying to, whose own ticket is never echoed. Write "
            "NONE for any slot that reaches back before my first message. "
            "Example: `ECHO: ENG-1234 ENG-5678 NONE`.")


def _instr_multi_counter(task, heavy=False):
    colors = task["c3"]["colors_heavy" if heavy else "colors"]
    fmt = " ".join(f"{c}=<n>" for c in colors)
    return (f"some of my messages include a line `EVENT: <color>`, where <color> is one "
            f"of: {', '.join(colors)}. Every single response must include one line with "
            f"the running total of EVENT lines I have sent for each color so far, "
            f"counting this message's if it has one: `COUNTS: {fmt}` -- always all "
            f"colors, in that order, starting from zero.")


def _instr_chain_checksum(task):
    return ("every message of mine includes a line `KEY: <number>`. Maintain a running "
            "checksum, starting at 0 before my first message: each turn the new checksum "
            f"is (previous checksum + KEY) mod {CHECKSUM_MOD}. Every single response "
            "must include one line `CHECK: <checksum>` with the value after applying "
            "this message's KEY. Do not show the arithmetic.")


def _instr_interference_twin(task):
    s = task["c3"]["shadow_init"]
    return ("separately from the task, I am tracking three shadow entries in slots "
            f"1-3. Initially slot 1 = `{s[0]}`, slot 2 = `{s[1]}`, slot 3 = `{s[2]}`. "
            "Some of my messages include a line `Shadow: slot <k> is renamed to "
            "<name>.` Every message of mine includes a line `Shadow query: <k>`; every "
            "single response must include one line `SHADOW: <name>` with the current "
            "name in the queried slot. The shadow entries are not part of the task "
            "itself -- never mix them into it.")


def _instr_confab_trap(task):
    return ("every message of mine includes a line `Ticket: ENG-####`. Some messages "
            "also include a line `Note: tag for ENG-#### is <word>.` When a message of "
            "mine includes a line `Tag query: ENG-####?`, your response must include "
            "one line `TAG: <word>` with the tag I gave that ticket -- or exactly "
            "`TAG: NONE` if I never gave that ticket a tag. Never guess a tag. If "
            "there is no tag query in my message, do not output a TAG line.")


def _instr_sparse_recall(task):
    return (f"your audit code for this session is `{task['c3']['audit_code']}`. "
            "Whenever a message of mine contains the line `Audit check.`, your "
            "response must include one line `AUDIT: <the code>`. Do not repeat or "
            "mention the code at any other time.")


def _instr_staircase(task):
    return ("every message of mine includes a line `Ticket: ENG-####`. Every single "
            "response must include a line `LEDGER: <n>` where n is the turn number. "
            "I will add further fields to this line as the session goes on.")


def _instr_static_trailer(task):
    return f"end every single response with the exact line `{STATIC_TRAILER}`."


_INSTR = {
    "lag_span": _instr_lag_span,
    "multi_counter": _instr_multi_counter,
    "multi_counter_heavy": lambda task: _instr_multi_counter(task, heavy=True),
    "chain_checksum": _instr_chain_checksum,
    "interference_twin": _instr_interference_twin,
    "confab_trap": _instr_confab_trap,
    "sparse_recall": _instr_sparse_recall,
    "staircase": _instr_staircase,
    "static_trailer": _instr_static_trailer,
}


def instruction_text(condition, task):
    if condition == "ensemble":
        parts = [f"{i}. {_INSTR[m](task)[0].upper()}{_INSTR[m](task)[1:]}"
                 for i, m in enumerate(ENSEMBLE_MEMBERS, 1)]
        return "Three more standing rules, all of which apply to every response:\n" + \
            "\n".join(parts)
    return "One more standing rule: " + _INSTR[condition](task)


STAIR_UPDATES = {
    2: ("LEDGER UPDATE: from now on the LEDGER line must be `LEDGER: <turn number> "
        "<count of tickets I have sent so far whose last digit is even, including "
        "this message's>`."),
    3: ("LEDGER UPDATE: from now on the LEDGER line must be `LEDGER: <turn number> "
        "<even-ending ticket count> <the ticket id from my very first message>`."),
    4: ("LEDGER UPDATE: from now on the LEDGER line must be `LEDGER: <turn number> "
        "<even-ending ticket count> <the first message's ticket id> <count of "
        "tickets I have sent so far that contain the digit 7>`."),
}


# ------------------------------------------------------------ message build

def _payload_lines(condition, turn):
    c3 = turn["c3"]
    lines = []
    needs_ticket = condition in ("lag_span", "confab_trap", "staircase", "ensemble")
    if needs_ticket:
        lines.append(f"Ticket: {c3['ticket']}")
    if condition == "multi_counter" and c3["event"]:
        lines.append(f"EVENT: {c3['event']}")
    if condition == "multi_counter_heavy" and c3["event_heavy"]:
        lines.append(f"EVENT: {c3['event_heavy']}")
    if condition in ("chain_checksum", "ensemble"):
        lines.append(f"KEY: {c3['key']}")
    if condition == "interference_twin":
        if c3["shadow_rename"]:
            r = c3["shadow_rename"]
            lines.append(f"Shadow: slot {r['slot']} is renamed to {r['new']}.")
        lines.append(f"Shadow query: {c3['shadow_query']}")
    if condition in ("confab_trap", "ensemble"):
        if c3["tag_note"]:
            n = c3["tag_note"]
            lines.append(f"Note: tag for {n['ticket']} is {n['word']}.")
        if c3["tag_query"]:
            lines.append(f"Tag query: {c3['tag_query']}?")
    if condition == "sparse_recall" and c3["audit_probe"]:
        lines.append("Audit check.")
    if condition == "staircase" and c3["stair_update"]:
        lines.append(STAIR_UPDATES[c3["stair_update"]])
    return lines


def _task_body(task_set, task, turn):
    if task_set == "coding":
        lines = ["Changes:"]
        lines += [f"- {i}" for i in turn["instructions"]]
        lines.append("Report: " + ", ".join(q["fn"] for q in turn["queries"]))
        return lines
    if task_set == "registers":
        lines = ["Updates:"]
        lines += [f"- {render_op(op)}" for op in turn["ops"]]
        lines.append("Report: " + ", ".join(turn["queries"]))
        return lines
    if task_set == "babi":
        lines = []
        if turn["new_story"] and turn["turn"] > 1:
            lines.append(NEW_STORY_LINE)
        lines += turn["story_lines"]
        lines.append(f"Question: {turn['question']}")
        return lines
    raise ValueError(task_set)


def build_turn_body(task_set, task, turn, condition):
    lines = [f"Turn {turn['turn']}:"]
    if condition != "baseline":
        lines += _payload_lines(condition, turn)
    lines += _task_body(task_set, task, turn)
    return "\n".join(lines)


def build_first_user_message(task_set, task, condition):
    if task_set == "coding":
        msg = COD_BRIEFING.format(module=render_module(task["initial_module"]))
    elif task_set == "registers":
        reg_lines = "\n".join(f"- {k} = {v}" for k, v in task["initial_state"].items())
        msg = REG_BRIEFING.format(registers=reg_lines)
    else:
        msg = BABI_BRIEFING
    if condition != "baseline":
        msg += "\n\n" + instruction_text(condition, task)
    msg += "\n\n" + build_turn_body(task_set, task, task["turns"][0], condition)
    return msg


# ------------------------------------------------------------ hallucination

def check_hallucination(task_set, text, turn):
    """List of (kind, detail) errors; empty = clean turn."""
    if task_set == "coding":
        return _hallu_coding(text, turn)
    if task_set == "registers":
        return [("missing_value" if got is None else "wrong_value",
                 f"{reg}: want {want}" + ("" if got is None else f" got {got}"))
                for reg, want, got in _hallu_registers(text, turn)]
    if task_set == "babi":
        ms = re.findall(r"ANSWER\s*:?\s*[`*_]*([A-Za-z]+)", text, re.I)
        if not ms:
            return [("missing_answer", turn["question"])]
        got = ms[-1].lower()
        want = turn["answer"].lower()
        if got != want:
            return [("wrong_answer", f"want {want} got {got}")]
        return []
    raise ValueError(task_set)


# ------------------------------------------------------------ canary checks

def _last(pattern, text):
    ms = re.findall(pattern, text, re.I)
    return ms[-1] if ms else None


def _norm_token(s):
    return re.sub(r"[^A-Za-z0-9_-]", "", s).upper()


def _score_lag_span(text, task, turn):
    # Only slots whose expected value is a real ticket test the memory-distance
    # axis. Vacuous NONE slots are ignored: on turn 1 every slot is vacuous and
    # models systematically echo the CURRENT ticket there (an instruction-
    # comprehension quirk, observed on 65 percent of turn-1 replies), which is
    # not the degradation this canary measures.  Turn 1 is therefore N/A,
    # turn 2 tests lag-1 only, turn 7+ tests all three lags.
    real = [(i, turn["c3"][f"lag{k}"]) for i, k in enumerate(LAGS)
            if turn["c3"][f"lag{k}"] != "NONE"]
    if not real:
        return {"score": None}
    line = _last(r"ECHO\s*:?\s*(.+)", text)
    if line is None:
        return {"score": 0.0}
    got = re.findall(r"(ENG-\d{4}|NONE)", line, re.I)
    correct = sum(1 for i, want in real
                  if i < len(got) and _norm_token(got[i]) == want.upper())
    return {"score": correct / len(real)}


def _score_multi_counter(text, task, turn, heavy=False):
    colors = task["c3"]["colors_heavy" if heavy else "colors"]
    truth = turn["c3"]["counts_heavy" if heavy else "counts"]
    line = _last(r"COUNTS\s*:?\s*(.+)", text)
    if line is None:
        return {"score": 0.0}
    correct = 0
    for c in colors:
        cm = re.search(rf"{c}\s*[=:]\s*[`*_]*(\d+)", line, re.I)
        if cm and int(cm.group(1)) == truth[c]:
            correct += 1
    return {"score": correct / len(colors)}


def _score_chain_checksum(text, task, turn):
    got = _last(r"CHECK\s*:?\s*[`*_]*(\d+)", text)
    return {"score": 1.0 if got is not None and int(got) == turn["c3"]["check"] else 0.0}


def _score_interference_twin(text, task, turn):
    got = _last(r"SHADOW\s*:?\s*[`*_]*([A-Za-z_]\w*)", text)
    want = turn["c3"]["shadow_truth"]
    return {"score": 1.0 if got is not None and got.lower() == want.lower() else 0.0}


def _score_confab_trap(text, task, turn):
    c3 = turn["c3"]
    if not c3["tag_query"]:
        return {"score": None}
    got = _last(r"TAG\s*:?\s*[`*_]*([A-Za-z]+)", text)
    want = c3["tag_truth"]
    ok = got is not None and got.lower() == want.lower()
    fabricated = (c3["tag_is_trap"] and got is not None and got.upper() != "NONE")
    return {"score": 1.0 if ok else 0.0, "fabricated": fabricated}


def _score_sparse_recall(text, task, turn):
    code = task["c3"]["audit_code"]
    norm = re.sub(r"[^A-Z0-9]", "", text.upper())
    present = re.sub(r"[^A-Z0-9]", "", code.upper()) in norm
    if not turn["c3"]["audit_probe"]:
        # not applicable, but record rehearsal: repeating the code anyway would
        # restore the copy path that made experiment 1's remember_fact useless
        return {"score": None, "rehearsed": present}
    return {"score": 1.0 if present else 0.0}


def _score_staircase(text, task, turn):
    line = _last(r"LEDGER(?!\s+UPDATE)\s*:?\s*(.+)", text)
    if line is None:
        return {"score": 0.0}
    got = [_norm_token(w) for w in re.split(r"[\s,|]+", line) if _norm_token(w)]
    want = [w.upper() for w in turn["c3"]["stair_expect"]]
    correct = sum(1 for i, w in enumerate(want) if i < len(got) and got[i] == w)
    return {"score": correct / len(want)}


def _score_static_trailer(text, task, turn):
    tail = re.sub(r"[^a-z0-9]", "", " ".join(text.splitlines()[-3:]).lower())
    ok = "reviewedbydrvance" in tail and "kx4471tango" in tail
    return {"score": 1.0 if ok else 0.0}


_SCORERS = {
    "lag_span": _score_lag_span,
    "multi_counter": _score_multi_counter,
    "multi_counter_heavy": lambda text, task, turn: _score_multi_counter(
        text, task, turn, heavy=True),
    "chain_checksum": _score_chain_checksum,
    "interference_twin": _score_interference_twin,
    "confab_trap": _score_confab_trap,
    "sparse_recall": _score_sparse_recall,
    "staircase": _score_staircase,
    "static_trailer": _score_static_trailer,
}


def score_canary(condition, text, task, turn):
    """Returns {"score": float|None, ...extras}. score None = not applicable."""
    if condition == "ensemble":
        subs = {m: _SCORERS[m](text, task, turn) for m in ENSEMBLE_MEMBERS}
        applicable = [d["score"] for d in subs.values() if d["score"] is not None]
        return {"score": min(applicable) if applicable else None,
                "subs": {m: d["score"] for m, d in subs.items()},
                "fabricated": subs["confab_trap"].get("fabricated")}
    return _SCORERS[condition](text, task, turn)
